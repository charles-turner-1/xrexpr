"""Tests for the lowering stage: ``to_lower_ir`` and ``emit``.

Two things are tested here:

- **The contract.** Lowering preserves semantics and is idempotent. ``emit`` rebuilds
  each call from the node's *semantic fields* (the reverse of ``to_opnode``), so a
  rewritten field ends up in the replayed call.
- **The fusion policy.** Which builder pairs get fused, and that everything else takes
  the opaque fallback rather than being modelled on a guess.

Because calls are *rebuilt*, ``emit`` produces the **canonical** spelling, which may
differ from what was recorded (``isel(time=0)`` comes back as ``isel({"time": 0})``).
They replay the same. The contract is equivalence, not identical calls. The round trip
is checked by re-parsing an emitted call with ``to_opnode`` (it should reach a fixed
point), and end to end by the replay-equality tests in ``test_accessor.py`` and
``test_properties.py``.

Refusing to fuse is always safe, so the refusal tests matter as much as the fusion
tests: each one pins a deliberate limit on what's fused, not a bug.

``emit`` is a pure function of the plan, so nothing here needs a dataset. The weighted
cases do build a ``DataArray``, but only so fusion can read its ``.dims``, which
computes nothing.

``to_lower_ir``'s second argument is the base dataset's dim names, the one thing
lowering can't read off the calls. :data:`_DIM_NAMES` stands in for a dataset with three
dims, so a grouper naming anything else counts as a *coordinate* grouper.
"""

from dataclasses import replace

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.ir import (
    ALL_DIMS,
    ContextOpen,
    GroupedReduce,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.lower import Call, emit, to_lower_ir
from xrexpr.schema import to_opnode

#: The dim names of the notional dataset these plans are lowered against.
_DIM_NAMES = frozenset({"time", "lat", "lon"})


@pytest.fixture
def plan():
    """A plan covering every ``Op`` variant, built the way the recorder builds one."""
    return [
        to_opnode("chunk", ({"time": 100},), {}),
        to_opnode("__getitem__", (["temperature"],), {}),
        to_opnode("isel", (), {"time": 0, "drop": True}),
        to_opnode("cumsum", ("lat",), {}),
        to_opnode("mean", ("lat",), {}),
        to_opnode("where", ("cond",), {}),
    ]


def test_the_fixture_plan_really_covers_every_variant(plan):
    """The fixture plan contains one node of every ``Op`` variant.

    Notes
    -----
    Keeps the round-trip tests below from passing vacuously on a partial vocabulary.
    """
    assert {type(node) for node in plan} == {
        Rechunk,
        Project,
        Select,
        Scan,
        Reduce,
        Opaque,
    }


def test_a_plan_with_no_builder_chain_passes_through_unchanged(plan):
    """A plan with no builder pair lowers to itself, node for node.

    Notes
    -----
    Every multi-call spelling is a ``ContextOpen`` pair, so a plan without one is
    already what it means and lowering must leave it alone.
    """
    assert to_lower_ir(plan, _DIM_NAMES) == plan


def test_lowering_is_idempotent(plan):
    """Lowering an already-lowered plan returns it unchanged — half the stated contract."""
    once = to_lower_ir(plan, _DIM_NAMES)
    assert to_lower_ir(once, _DIM_NAMES) == once


def test_lowering_does_not_mutate_its_input(plan):
    """Lowering builds a fresh list, so appending to the result leaves the input alone."""
    before = list(plan)
    to_lower_ir(plan, _DIM_NAMES).append(to_opnode("mean", (), {}))
    assert plan == before


def _semantics(node):
    """The part of a node the optimiser cares about: everything except its replay call.

    Two nodes with equal semantics mean the same thing however their calls are spelled,
    which is the equivalence ``emit`` preserves. :class:`~xrexpr.ir.Elementwise` and
    :class:`~xrexpr.ir.Opaque` have no semantic fields, so their verbatim call *is*
    their content.
    """
    match node:
        case Reduce():
            return ("reduce", node.consumes)
        case Scan():
            return ("scan", node.dims)
        case Select():
            return ("select", node.indexer)
        case Rechunk():
            return ("rechunk", node.chunks, node.uniform)
        case Project():
            return ("project", node.variables, node.single)
        case (
            _
        ):  # Drop / Rename / Elementwise / Opaque — none in this fixture's flat kinds
            return ("header", node.name, node.args, node.kwargs)


def test_emit_derives_a_reparseable_canonical_header(plan):
    """Every emitted call re-parses to the same semantics and re-emits identically.

    Notes
    -----
    ``emit`` rebuilds the call from the node's semantic fields, so it may not be the
    recorded spelling. But it must:

    - keep the meaning, checked by re-parsing with ``to_opnode`` and comparing
      semantics;
    - be a canonical *fixed point*: parsing an emitted call and emitting it again gives
      the identical call.

    Together these stand in for "replays what was written" without needing a dataset.
    The actual replay equality is checked in the property suite.
    """
    lowered = to_lower_ir(plan, _DIM_NAMES)
    calls = emit(lowered)

    reparsed = [to_opnode(c.name, c.args, c.kwargs) for c in calls]
    assert [_semantics(r) for r in reparsed] == [_semantics(n) for n in lowered]
    assert emit(reparsed) == calls


def test_emit_canonicalises_a_kwargs_select(plan):
    """A select recorded with kwargs emits as the canonical positional dict plus option kwargs.

    Notes
    -----
    The reverse of ``_select_indexer``: the indexer becomes one positional ``{dim:
    index}`` dict, and only the *option* kwargs (``drop``) stay as kwargs.
    ``isel(time=0, drop=True)`` and this form select the same thing.
    """
    select = emit(to_lower_ir(plan, _DIM_NAMES))[2]
    assert select == Call(
        name="isel", args=({"time": 0},), kwargs=frozendict({"drop": True})
    )


def test_emit_of_an_empty_plan_is_empty():
    """An empty plan lowers and emits to nothing, rather than raising."""
    assert emit(to_lower_ir([], _DIM_NAMES)) == []


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "expected"),
    [
        # Reduce: dim canonicalised, every option kwarg preserved.
        (
            "mean",
            ("lat",),
            {"skipna": False, "keepdims": True},
            Call("mean", ("lat",), frozendict({"skipna": False, "keepdims": True})),
        ),
        ("sum", (), {"dim": "time"}, Call("sum", ("time",))),  # dim= → positional
        ("mean", (["lon", "lat"],), {}, Call("mean", (["lat", "lon"],))),  # sorted
        ("mean", (), {}, Call("mean", ())),  # bare → no dim
        # Scan: diff's ``n`` rides along, positional or keyword.
        ("diff", ("time", 2), {}, Call("diff", ("time", 2))),
        ("diff", ("time",), {"n": 2}, Call("diff", ("time",), frozendict({"n": 2}))),
        ("cumsum", (), {}, Call("cumsum", ())),  # bare scan → no dim
        # Select: indexer → one positional dict; only option kwargs survive.
        (
            "isel",
            (),
            {"time": 0, "lat": slice(0, 5), "drop": True},
            Call(
                "isel", ({"time": 0, "lat": slice(0, 5)},), frozendict({"drop": True})
            ),
        ),
        # Rechunk: uniform, mapping (+ option kwarg), and empty forms.
        ("chunk", ("auto",), {}, Call("chunk", ("auto",))),
        (
            "chunk",
            ({"time": 100},),
            {"token": "x"},
            Call("chunk", ({"time": 100},), frozendict({"token": "x"})),
        ),
        ("chunk", (), {}, Call("chunk", ())),
        # Project: single (bare name), list, and tuple-as-one-name keys.
        ("__getitem__", ("tas",), {}, Call("__getitem__", ("tas",))),
        ("__getitem__", (["a", "b"],), {}, Call("__getitem__", (["a", "b"],))),
        ("__getitem__", (("a", "b"),), {}, Call("__getitem__", (("a", "b"),))),
        # Drop: canonical list form; the redundant ``names=`` goes, ``errors=`` stays.
        ("drop_vars", ("region",), {}, Call("drop_vars", (["region"],))),
        (
            "drop_vars",
            (),
            {"names": ["a", "b"], "errors": "ignore"},
            Call("drop_vars", (["a", "b"],), frozendict({"errors": "ignore"})),
        ),
        # Rename: kwargs → positional mapping (rename has no option kwargs).
        ("rename", (), {"month": "time"}, Call("rename", ({"month": "time"},))),
        # Elementwise: no semantic field, so the header is verbatim.
        ("fillna", (0,), {}, Call("fillna", (0,))),
    ],
)
def test_emit_derives_the_canonical_header(name, args, kwargs, expected):
    """Each flat ``Op`` kind emits the canonical inverse of its ``to_opnode`` parse."""
    node = to_opnode(name, args, kwargs)
    assert emit([node]) == [expected]


def test_emit_reduce_keeps_the_function_of_a_reduce_call():
    """``.reduce(func, dim)`` — dim at position 1 — re-emits with its function intact."""
    node = to_opnode("reduce", (np.mean, "time"), {})
    assert emit([node]) == [Call("reduce", (np.mean, "time"))]


def test_emit_follows_the_semantic_field_not_a_stale_header():
    """A changed ``indexer`` drives the emitted call even when ``args`` still says otherwise.

    Notes
    -----
    An optimiser (a rule, or a Rust backend that never sees ``args``/``kwargs``) changes
    the *semantic field* and leaves the recorded call alone. ``emit`` must read the
    field, not the stale call, or a rewritten plan would replay the call from before the
    rewrite.
    """
    stale = Select(
        name="isel",
        args=({"time": 99},),  # what the call was recorded as
        indexer=frozendict({"time": 0}),  # what the optimiser left it as
    )
    assert emit([stale]) == [Call("isel", ({"time": 0},))]


def test_emit_follows_the_reduce_dim_not_a_stale_header():
    """A reduce whose ``consumes`` was rewritten emits the *new* dim, not the recorded one.

    Notes
    -----
    The ``Select`` case above, for the reduce arm, which passes the recorded
    ``args``/``kwargs`` through :func:`~xrexpr.lower._dim_call`. The test:

    1. builds a reduce as the recorder would, with the call and ``consumes`` agreeing;
    2. rewrites *only* ``consumes``, as a rule or a Rust backend would, leaving the call
       stale;
    3. checks ``emit`` follows ``consumes``.

    ``_dim_call`` removes the recorded dim from the call and rebuilds it from the field,
    so the stale spelling never reaches replay. Only the non-dim parts (a function, an
    option kwarg) pass through, and they have no semantic field to disagree with.
    """
    recorded = to_opnode("mean", ("time",), {})
    assert emit([recorded]) == [Call("mean", ("time",))]  # header and field agree

    # An optimiser rewrites the field alone; the header still spells the old dim.
    rewritten = replace(recorded, consumes=frozenset({"lat"}))
    assert rewritten.args == ("time",)  # header untouched — still "time"
    assert emit([rewritten]) == [Call("mean", ("lat",))]  # emit follows consumes

    # Same when the recorded dim was a ``dim=`` kwarg rather than a positional.
    from_kwarg = replace(
        to_opnode("mean", (), {"dim": "time"}), consumes=frozenset({"lat"})
    )
    assert emit([from_kwarg]) == [Call("mean", ("lat",))]

    # Rewriting to ALL_DIMS drops the stale positional and re-inserts nothing (bare
    # ``mean()``).
    to_all = replace(recorded, consumes=ALL_DIMS)
    assert emit([to_all]) == [Call("mean", ())]

    # The non-dim payload survives the rewrite: ``skipna``/``keepdims`` here, ...
    optioned = replace(
        to_opnode("mean", ("time",), {"skipna": True, "keepdims": True}),
        consumes=frozenset({"lat"}),
    )
    assert emit([optioned]) == [
        Call("mean", ("lat",), {"skipna": True, "keepdims": True})
    ]

    # ... and ``reduce``'s function, whose dim sits at position 1.
    with_func = replace(
        to_opnode("reduce", (np.sum, "time"), {}), consumes=frozenset({"lat"})
    )
    assert emit([with_func]) == [Call("reduce", (np.sum, "lat"))]


def test_emit_follows_the_scan_dim_not_a_stale_header():
    """A scan whose ``dims`` was rewritten emits the *new* dim, not the recorded one.

    Notes
    -----
    The same guarantee as :func:`test_emit_follows_the_reduce_dim_not_a_stale_header`,
    for a scan. A scan keeps its dim rather than removing it, but goes through the same
    :func:`_dim_call`.
    """
    recorded = to_opnode("cumsum", ("time",), {})
    assert emit([recorded]) == [Call("cumsum", ("time",))]  # header and field agree

    rewritten = replace(recorded, dims=frozenset({"lat"}))
    assert rewritten.args == ("time",)  # header untouched — still "time"
    assert emit([rewritten]) == [Call("cumsum", ("lat",))]  # emit follows dims


def test_call_coerces_to_immutable_containers_and_hashes():
    """A ``Call`` coerces its args and kwargs to immutable containers, and hashes."""
    call = Call(name="mean", args=["lat"], kwargs={"skipna": True})
    assert call.args == ("lat",)
    assert call.kwargs == frozendict({"skipna": True})
    assert hash(call)


def test_calls_compare_by_value():
    """Two ``Call``s are equal when their headers are, and unequal when the args differ."""
    assert Call(name="mean", args=("lat",)) == Call(name="mean", args=("lat",))
    assert Call(name="mean", args=("lat",)) != Call(name="mean", args=("lon",))


# --- fusing the groupby family -------------------------------------------------------


def _lower(*calls, dim_names=_DIM_NAMES):
    """Lower the plan the recorder would build from ``(name, args, kwargs)`` triples.

    Parameters
    ----------
    *calls : tuple
        One ``(name, args, kwargs)`` triple per recorded call.
    dim_names : frozenset of Hashable, optional
        The base dataset's dim names. Defaults to :data:`_DIM_NAMES`.

    Returns
    -------
    list of LoweredOp
        The lowered plan.
    """
    return to_lower_ir([to_opnode(*call) for call in calls], dim_names)


@pytest.mark.parametrize(
    ("grouper", "group_dim", "new_dim"),
    [
        # verified against xarray 2026.7.0 -- see GroupedReduce's docstring
        (("groupby", ("time.month",), {}), "time", "month"),
        (("groupby", ("lat",), {}), "lat", "lat"),
        (("groupby_bins", ("lat", 2), {}), "lat", "lat_bins"),
        (("resample", (), {"time": "2D"}), "time", "time"),
    ],
)
def test_grouper_dims_are_read_off_the_opener(grouper, group_dim, new_dim):
    """Each groupby-family spelling fuses with the group and minted dims it really has.

    Notes
    -----
    Verified against xarray 2026.7.0 — see ``GroupedReduce``'s docstring for the
    algebra.
    """
    (node,) = _lower(grouper, ("mean", (), {}))
    assert isinstance(node, GroupedReduce)
    assert (node.group_dim, node.new_dim) == (group_dim, new_dim)
    assert node.reduce == "mean"


def test_bare_closer_consumes_nothing_extra():
    """A grouped bare ``mean()`` consumes nothing beyond the group dim.

    Notes
    -----
    Easy to get backwards: a grouped bare ``mean()`` only reduces along the group dim,
    so unlike a Dataset-level bare reduce it leaves the other dims alone. ``consumes``
    is what it removes *in addition*, which here is nothing.
    """
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", (), {}))
    assert node.consumes == frozenset()


def test_closer_naming_the_group_dim_fuses_with_no_extra():
    """A closer naming exactly the group dim is an aggregation, consuming nothing extra."""
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", ("time",), {}))
    assert node.consumes == frozenset()


def test_closer_naming_more_than_the_group_dim_keeps_the_rest_as_consumes():
    """A closer naming the group dim *and* another records the other one in ``consumes``."""
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", (["time", "lat"],), {}))
    assert node.consumes == frozenset({"lat"})


def test_within_group_map_does_not_fuse():
    """A closer naming only a non-group dim refuses to fuse, and replays as a verbatim pair.

    Notes
    -----
    ``ds.groupby("time.month").mean("lat")`` is a per-group *map*: it keeps ``time`` and
    creates no ``month``, so the aggregation node would describe it wrongly. Refusing
    leaves a verbatim pair: correct, just not optimised.
    """
    lowered = _lower(("groupby", ("time.month",), {}), ("mean", ("lat",), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]
    assert [n.name for n in lowered] == ["groupby", "mean"]


@pytest.mark.parametrize(
    "opener",
    [
        # not a fixed window: an exponential weighting, which ``window`` cannot describe
        ("rolling_exp", (), {"time": 3}),
        # a scan wearing a builder's clothes
        ("cumulative", ("time",), {}),
    ],
)
def test_openers_without_a_fusion_rule_demote_the_whole_pair(opener):
    """``rolling_exp`` and ``cumulative`` pairs become two opaque nodes.

    Notes
    -----
    No fused node describes these, so both halves are opaque and no rule can fire on or
    across the pair.
    """
    lowered = _lower(opener, ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_demoting_the_closer_is_not_optional():
    """A refused pair leaves no ``Reduce`` behind: the closer is demoted with the opener.

    Notes
    -----
    If the closer stayed as the ``Reduce`` ``to_opnode`` provisionally built, it would
    keep a *Dataset-level* reading of a call that was never Dataset-level. A bare
    ``mean()`` after an unfusable opener would carry ``ALL_DIMS``, and a following
    select would be rejected against dims it never removed.
    """
    lowered = _lower(("rolling_exp", (), {"time": 3}), ("mean", (), {}))
    assert not any(isinstance(n, Reduce) for n in lowered)


# --- fusing rolling and coarsen ------------------------------------------------------


@pytest.mark.parametrize(
    ("opener", "window"),
    [
        (("rolling", (), {"time": 3}), {"time": 3}),
        (("rolling", ({"time": 3},), {}), {"time": 3}),
        (("rolling", (), {"time": 3, "center": True}), {"time": 3}),
        (("rolling", (), {"time": 3, "lat": 2}), {"time": 3, "lat": 2}),
        (("coarsen", (), {"time": 2}), {"time": 2}),
        (("coarsen", (), {"time": 2, "boundary": "trim"}), {"time": 2}),
    ],
)
def test_window_spec_is_read_off_the_opener(opener, window):
    """Every rolling/coarsen spelling fuses with the ``{dim: window}`` mapping it names."""
    (node,) = _lower(opener, ("mean", (), {}))
    assert isinstance(node, WindowedReduce)
    assert dict(node.window) == window
    assert node.reduce == "mean"


def test_a_windowed_closer_with_a_parsed_dim_spec_still_fuses():
    """``rolling(time=3).mean("lat")`` fuses: the parsed dim is a misparse with no effect.

    Notes
    -----
    ``DatasetRolling.mean`` is ``(keep_attrs=None, **kwargs)``. It takes no dim
    argument, so this passes ``"lat"`` as *keep_attrs* and reduces nothing
    (``rolling(time=3).mean("lat")`` equals ``.mean()``). ``to_opnode`` can't know that
    and records ``consumes={"lat"}``, but a ``WindowedReduce`` drops that and only reads
    ``window``, so the pair fuses rather than falling back to ``Opaque``.
    ``reduce_args`` is kept verbatim so replay preserves whether ``keep_attrs`` is
    truthy.
    """
    (node,) = _lower(("rolling", (), {"time": 3}), ("mean", ("lat",), {}))
    assert isinstance(node, WindowedReduce)
    assert dict(node.window) == {"time": 3}
    assert node.reduce == "mean"
    assert node.reduce_args == ("lat",)


def test_windowed_option_kwargs_are_not_windows():
    """Option kwargs stay out of ``window`` but are kept verbatim in ``kwargs``.

    Notes
    -----
    ``center``/``boundary`` are options; only real dim keywords are windows. They stay
    in ``kwargs`` because ``boundary`` decides coarsen's output size.
    """
    (node,) = _lower(
        ("coarsen", (), {"time": 2, "boundary": "pad", "side": "right"}),
        ("sum", (), {}),
    )
    assert dict(node.window) == {"time": 2}
    assert node.kwargs["boundary"] == "pad"


def test_a_windowless_opener_does_not_fuse():
    """``rolling()`` with no dim keyword names no window, so there is nothing to model."""
    lowered = _lower(("rolling", (), {}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_non_integer_window_does_not_fuse():
    """A non-integer window is an unrecognised spelling, so the pair replays verbatim."""
    lowered = _lower(("rolling", (), {"time": "3D"}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_windowed_node_emits_both_calls_verbatim():
    """A fused ``WindowedReduce`` emits the two calls it fused, headers untouched."""
    (node,) = _lower(("rolling", (), {"time": 3, "center": True}), ("mean", (), {}))
    assert emit([node]) == [
        Call(name="rolling", kwargs=frozendict({"time": 3, "center": True})),
        Call(name="mean"),
    ]


# --- fusing weighted -----------------------------------------------------------------


def _weights(*dims):
    """Build a weights ``DataArray`` over ``dims``.

    Parameters
    ----------
    *dims : Hashable
        The dims the weights span.

    Returns
    -------
    xarray.DataArray
        A ones array. Only its ``.dims`` is ever read here — metadata, materialising
        nothing.
    """
    return xr.DataArray(np.ones([2] * len(dims)), dims=dims)


def test_weighted_pair_fuses_with_a_bare_closer():
    """A weighted pair with a bare closer fuses, keeping ``ALL_DIMS`` and the weight dims."""
    (node,) = _lower(("weighted", (_weights("lat"),), {}), ("mean", (), {}))
    assert isinstance(node, WeightedReduce)
    assert node.reduce == "mean"
    assert node.consumes is ALL_DIMS
    assert node.weight_dims == frozenset({"lat"})


def test_weighted_closer_keeps_the_dims_it_named():
    """A weighted closer that names dims keeps them, unlike a windowed closer, whose dim has no effect.

    Notes
    -----
    The difference from ``rolling`` comes from the two signatures, not a choice.
    ``DatasetWeighted.mean`` is ``(dim=None, *, skipna, keep_attrs)``, so it really does
    take a dim, and its ``consumes`` is carried into the node. ``DatasetRolling.mean``
    takes none, so ``rolling(time=3).mean("lat")`` fuses too, but *drops* the dim rather
    than consuming it.
    """
    (node,) = _lower(("weighted", (_weights("lat"),), {}), ("mean", (["lat"],), {}))
    assert node.consumes == frozenset({"lat"})


@pytest.mark.parametrize(
    "opener",
    [
        ("weighted", (_weights("lat", "lon"),), {}),
        ("weighted", (), {"weights": _weights("lat", "lon")}),  # the keyword spelling
    ],
)
def test_weight_dims_are_read_off_the_weights(opener):
    """Both weights spellings — positional and ``weights=`` — yield the same weight dims."""
    (node,) = _lower(opener, ("mean", ("lat",), {}))
    assert node.weight_dims == frozenset({"lat", "lon"})


def test_zero_dimensional_weights_fuse_with_no_weight_dims():
    """0-d weights fuse and report no weight dims, which is correct.

    Notes
    -----
    Not a special case: 0-d weights broadcast and align nothing, so the node's dim
    effect really *is* a plain reduce's.
    """
    (node,) = _lower(("weighted", (_weights(),), {}), ("mean", ("lat",), {}))
    assert node.weight_dims == frozenset()


@pytest.mark.parametrize("weights", ["lat", np.ones(3), {"lat": 1}, None])
def test_weights_with_no_dims_tuple_do_not_fuse(weights):
    """Weights with no ``.dims`` tuple refuse to fuse, whatever they are.

    Notes
    -----
    The weight dims are what make this node more than a renamed ``Reduce``, so weights
    they can't be read from refuse. That costs nothing: xarray rejects such weights
    itself (``ValueError: `weights` must be a DataArray``), so the pair couldn't have
    replayed anyway, and refusing lets xarray report it in its own words.
    """
    lowered = _lower(("weighted", (weights,), {}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


@pytest.mark.parametrize("closer", ["sum_of_weights", "sum_of_squares", "quantile"])
def test_weighted_only_closers_do_not_fuse(closer):
    """The weighted-only closers refuse to fuse, because they aren't tabulated reductions.

    Notes
    -----
    They aren't in ``OP_TABLE``, so they record ``Opaque`` and refuse automatically.
    That's the right answer for ``quantile`` in particular: it takes ``q`` first, which
    a reduce's dim parsing would read as a dim.
    """
    lowered = _lower(("weighted", (_weights("lat"),), {}), (closer, (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_weighted_node_emits_both_calls_verbatim():
    """A fused ``WeightedReduce`` emits both calls, weights payload and kwargs untouched."""
    weights = _weights("lat")
    (node,) = _lower(("weighted", (weights,), {}), ("mean", ("lat",), {"skipna": True}))
    assert emit([node]) == [
        Call(name="weighted", args=(weights,)),
        Call(name="mean", args=("lat",), kwargs=frozendict({"skipna": True})),
    ]


def test_lowering_stays_idempotent_with_a_weighted_pair():
    """Idempotence holds with an array payload, which also checks the payload is shared.

    Notes
    -----
    The weights are the one payload plan equality can't compare by value (an elementwise
    ``==`` has no single truth value). So this also checks that the *same* object is
    carried through rather than rebuilt; see ``test_ir.py``.
    """
    once = _lower(("weighted", (_weights("lat"),), {}), ("mean", (), {}))
    assert to_lower_ir(once, _DIM_NAMES) == once


def test_non_string_grouper_does_not_fuse():
    """A non-string grouper refuses: a ``DataArray``/``Grouper`` has no single group dim."""
    lowered = _lower(("groupby", (object(),), {}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


@pytest.mark.parametrize(
    "opener",
    [
        ("groupby", ("lat",), {}),
        ("rolling", (), {"time": 3}),
        ("weighted", (_weights("lat"),), {}),
    ],
)
def test_a_keepdims_closer_refuses_to_fuse(opener):
    """A ``keepdims=True`` closer never fuses: each builder family falls back to a verbatim opaque pair.

    Notes
    -----
    ``keepdims=True`` keeps the reduced dims at size 1, so a fused node (which removes
    them) would describe it wrongly, the same reason a within-group map refuses. The
    closer is a real ``Reduce``, so it's the ``closer.keepdims`` check in each
    ``_fuse_*`` that refuses.
    """
    lowered = _lower(opener, ("mean", (), {"keepdims": True}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


@pytest.mark.parametrize(
    "opener",
    [
        # each of these gives a plausible-looking ``group_dim`` if read off the call,
        # and each is wrong: the dim actually consumed is the one the *coordinate* is
        # defined on, which the call doesn't say. So they refuse.
        ("groupby", ("region",), {}),
        ("groupby", ("region.foo",), {}),
        ("groupby_bins", ("region", 2), {}),
        ("resample", (), {"region": "2D"}),
        # dotted bins: the part before the dot is a dim, so checking only that would
        # accept it, but the new dim is ``month_bins``, not ``time.month_bins`` (checked
        # against xarray 2026.7.0). Refused rather than modelled.
        ("groupby_bins", ("time.month", 2), {}),
    ],
)
def test_a_grouper_whose_head_is_not_a_dim_does_not_fuse(opener):
    """A grouper naming a coordinate rather than a dim refuses, and replays verbatim.

    Notes
    -----
    Each of these gives a plausible ``group_dim`` if read off the call, and each is
    wrong: the dim actually consumed is the one the *coordinate* is defined on, which
    the call doesn't say.
    """
    lowered = _lower(opener, ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]
    assert [n.name for n in lowered] == [opener[0], "mean"]


def test_a_coordinate_grouper_fuses_once_its_name_is_a_dim():
    """The same grouper fuses against a dataset where its name really is a dim.

    Notes
    -----
    The check is on membership, not spelling. This shows the refusals above come from
    the dim check, not from some other rule about how the grouper is written.
    """
    (node,) = _lower(
        ("groupby", ("region",), {}),
        ("mean", (), {}),
        dim_names=_DIM_NAMES | {"region"},
    )
    assert isinstance(node, GroupedReduce)
    assert (node.group_dim, node.new_dim) == ("region", "region")


def test_untabulated_closer_does_not_fuse():
    """An untabulated closer refuses: ``first`` records ``Opaque``, so there is no dim spec."""
    lowered = _lower(("groupby", ("lat",), {}), ("first", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_unclosed_context_demotes_to_a_single_opaque():
    """An opener at the end of a plan demotes on its own — the opener is the whole of it."""
    lowered = _lower(("groupby", ("lat",), {}))
    assert [type(n) for n in lowered] == [Opaque]


def test_no_context_open_ever_survives_lowering():
    """No ``ContextOpen`` survives lowering, whether the pair fused, refused, or was never closed.

    Notes
    -----
    The ``LoweredOp`` type already enforces this at type-check time; this checks it at
    runtime too.
    """
    plans = [
        _lower(("groupby", ("time.month",), {}), ("mean", (), {})),
        _lower(("groupby", ("time.month",), {}), ("mean", ("lat",), {})),
        _lower(("rolling", (), {"time": 3}), ("mean", (), {})),
        _lower(("weighted", (_weights("lat"),), {}), ("mean", (), {})),
        _lower(("groupby", ("lat",), {})),
    ]
    assert not any(isinstance(n, ContextOpen) for plan in plans for n in plan)


def test_ops_after_a_context_are_modelled_again():
    """A reduce after a refused pair is modelled normally, not swallowed by the fallback.

    Notes
    -----
    Only the pair becomes opaque, so the ``mean`` after an unfusable ``first`` is a
    normal ``Reduce``.
    """
    lowered = _lower(
        ("groupby", ("lat",), {}), ("first", (), {}), ("mean", ("time",), {})
    )
    assert [type(n) for n in lowered] == [Opaque, Opaque, Reduce]
    assert lowered[-1].consumes == frozenset({"time"})


def test_fused_node_emits_both_calls_verbatim():
    """A fused ``GroupedReduce`` emits the opener and closer it fused, headers untouched."""
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", (), {"skipna": True}))
    assert emit([node]) == [
        Call(name="groupby", args=("time.month",)),
        Call(name="mean", kwargs=frozendict({"skipna": True})),
    ]


def test_lowering_stays_idempotent_with_fusion():
    """Idempotence holds for a plan that actually fused, not only one that passed through.

    Notes
    -----
    It follows from the output containing no ``ContextOpen``: a second pass has nothing
    to match.
    """
    once = _lower(
        ("groupby", ("time.month",), {}), ("mean", (), {}), ("cumsum", ("lat",), {})
    )
    assert to_lower_ir(once, _DIM_NAMES) == once
