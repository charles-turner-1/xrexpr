"""Tests for the lowering stage: ``to_lower_ir`` and ``emit``.

Two things are pinned here. The *contract* — lowering is semantics-preserving and
idempotent, and ``emit`` reproduces the calls the recorder saw, spelling included — and
the *fusion policy*: which builder pairs v1 claims to understand, and that everything else
takes the mandatory opaque fallback rather than being modelled on a guess.

Refusing to fuse is always safe, so the negative cases matter as much as the positive
ones: each pins a narrowing of what is claimed, not a bug.

``emit`` is a pure function of the plan, so everything here runs without a dataset; the
end-to-end equality lives in ``test_accessor.py`` and ``test_properties.py``. The weighted
cases do build a ``DataArray``, but only as a payload whose ``.dims`` fusion reads —
metadata, materialising nothing.

``to_lower_ir``'s second argument is the base dataset's dim names, the one thing lowering
cannot read off the calls. :data:`_DIMS` stands in for a dataset with those three dims, so a
grouper naming anything else is a *coordinate* grouper as far as fusion is concerned.
"""

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

#: The dims of the notional dataset these plans are lowered against.
_DIMS = frozenset({"time", "lat", "lon"})


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
    Every multi-call spelling is a ``ContextOpen`` pair, so a plan without one is already
    what it means and lowering must leave it alone.
    """
    assert to_lower_ir(plan, _DIMS) == plan


def test_lowering_is_idempotent(plan):
    """Lowering an already-lowered plan returns it unchanged — half the stated contract."""
    once = to_lower_ir(plan, _DIMS)
    assert to_lower_ir(once, _DIMS) == once


def test_lowering_does_not_mutate_its_input(plan):
    """Lowering builds a fresh list, so appending to the result leaves the input alone."""
    before = list(plan)
    to_lower_ir(plan, _DIMS).append(to_opnode("mean", (), {}))
    assert plan == before


def test_emit_reproduces_every_call_verbatim(plan):
    """An unmodified plan emits exactly the calls the recorder saw.

    Notes
    -----
    One node, one call, header untouched — so an unrewritten plan replays what was
    written.
    """
    assert emit(to_lower_ir(plan, _DIMS)) == [
        Call(name=node.name, args=node.args, kwargs=node.kwargs) for node in plan
    ]


def test_emit_keeps_the_recorded_spelling(plan):
    """A select recorded as kwargs emits as kwargs, not canonicalised to a positional dict.

    Notes
    -----
    ``emit`` must not re-spell calls the pipeline never needed to touch — the positional
    dict is the form the merge rule happens to build, not a normal form.
    """
    select = emit(to_lower_ir(plan, _DIMS))[2]
    assert select == Call(name="isel", kwargs=frozendict({"time": 0, "drop": True}))


def test_emit_of_an_empty_plan_is_empty():
    """An empty plan lowers and emits to nothing, rather than raising."""
    assert emit(to_lower_ir([], _DIMS)) == []


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


def _lower(*calls, dims=_DIMS):
    """Lower the plan the recorder would build from ``(name, args, kwargs)`` triples.

    Parameters
    ----------
    *calls : tuple
        One ``(name, args, kwargs)`` triple per recorded call.
    dims : frozenset of Hashable, optional
        The base dataset's dim names. Defaults to :data:`_DIMS`.

    Returns
    -------
    list of LoweredOp
        The lowered plan.
    """
    return to_lower_ir([to_opnode(*call) for call in calls], dims)


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
    Verified against xarray 2026.7.0 — see ``GroupedReduce``'s docstring for the algebra.
    """
    (node,) = _lower(grouper, ("mean", (), {}))
    assert isinstance(node, GroupedReduce)
    assert (node.group_dim, node.new_dim) == (group_dim, new_dim)
    assert node.reduce == "mean"


def test_bare_closer_consumes_nothing_extra():
    """A grouped bare ``mean()`` consumes nothing beyond the group dim.

    Notes
    -----
    The correction that is easy to get backwards: a grouped bare ``mean()`` reduces only
    along the group dim, so unlike a Dataset-level bare reduce it leaves the other dims
    alone. ``consumes`` is what it removes *in addition*, which here is nothing.
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
    mints no ``month``, so the aggregation node would misdescribe it. Refusing leaves a
    verbatim pair, which is correct if unoptimised.
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
    """``rolling_exp`` and ``cumulative`` pairs demote to two opaque nodes.

    Notes
    -----
    No fused node describes these, so they behave permanently as the accessor's barrier
    made them: both halves opaque, and no rule can fire on or across the pair.
    """
    lowered = _lower(opener, ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_demoting_the_closer_is_not_optional():
    """A refused pair leaves no ``Reduce`` behind — the closer is demoted with the opener.

    Notes
    -----
    The failure this guards: leaving the closer as the ``Reduce`` ``to_opnode``
    provisionally built keeps a *Dataset-level* reading of a call that was never
    Dataset-level — a bare ``mean()`` after an unfusable opener would carry ``ALL_DIMS``
    and a following select would be rejected against dims it never removed.
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


def test_a_windowed_closer_with_a_parsed_dim_spec_does_not_fuse():
    """``rolling(time=3).mean("lat")`` refuses to fuse, because the dim spec is a misparse.

    Notes
    -----
    The trap: ``DatasetRolling.mean`` is ``(keep_attrs=None, **kwargs)`` — it takes no dim
    argument, so this passes ``"lat"`` as *keep_attrs* and reduces nothing. ``to_opnode``
    cannot know that and records ``consumes={"lat"}``. Fusing would import a dim effect
    invented by a misparse.
    """
    lowered = _lower(("rolling", (), {"time": 3}), ("mean", ("lat",), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_windowed_option_kwargs_are_not_windows():
    """Option kwargs stay out of ``window`` but survive verbatim in ``kwargs``.

    Notes
    -----
    ``center``/``boundary`` are options; only real dim keywords are windows — but they
    stay in ``kwargs``, because ``boundary`` decides coarsen's output size.
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
    """A weighted closer that named dims fuses and carries them over, unlike a windowed one.

    Notes
    -----
    The asymmetry with ``rolling``, and a fact about the two signatures rather than a
    judgement: ``DatasetWeighted.mean`` is ``(dim=None, *, skipna, keep_attrs)`` — it
    really does take a dim — while ``DatasetRolling.mean`` takes none, which is what makes
    ``rolling(time=3).mean("lat")`` a misparse that must not fuse.
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
    """0-d weights fuse and report the empty set, which is the honest answer.

    Notes
    -----
    Not a special case: a 0-d weights array broadcasts nothing and aligns nothing, so such
    a node's dim effect really *is* a plain reduce's.
    """
    (node,) = _lower(("weighted", (_weights(),), {}), ("mean", ("lat",), {}))
    assert node.weight_dims == frozenset()


@pytest.mark.parametrize("weights", ["lat", np.ones(3), {"lat": 1}, None])
def test_weights_with_no_dims_tuple_do_not_fuse(weights):
    """Weights with no ``.dims`` tuple refuse to fuse, whatever the payload is.

    Notes
    -----
    The weight dims are what make this node more than a relabelled ``Reduce``, so a
    payload they can't be read from refuses. That costs nothing: xarray itself rejects
    such weights (``ValueError: `weights` must be a DataArray``), so the pair could not
    have replayed either way — and refusing lets xarray say so in its own words.
    """
    lowered = _lower(("weighted", (weights,), {}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


@pytest.mark.parametrize("closer", ["sum_of_weights", "sum_of_squares", "quantile"])
def test_weighted_only_closers_do_not_fuse(closer):
    """The weighted-only closers refuse to fuse, because they are not tabulated reduces.

    Notes
    -----
    They aren't in ``OP_TABLE``, so they record ``Opaque`` and refuse for free — which is
    the right answer for ``quantile`` in particular: it takes ``q`` first positionally,
    which a reduce's dim-spec parse would read as a dim.
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
    """Idempotence survives an array payload, which also pins that the payload is shared.

    Notes
    -----
    The weights are the one payload plan equality cannot compare structurally (an
    elementwise ``==`` has no truth value), so this also pins that the *same* payload
    object is carried through rather than rebuilt — see ``test_ir.py``.
    """
    once = _lower(("weighted", (_weights("lat"),), {}), ("mean", (), {}))
    assert to_lower_ir(once, _DIMS) == once


def test_non_string_grouper_does_not_fuse():
    """A non-string grouper refuses: a ``DataArray``/``Grouper`` has no single group dim."""
    lowered = _lower(("groupby", (object(),), {}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


@pytest.mark.parametrize(
    "opener",
    [
        # issue #90: each of these reads a plausible ``group_dim`` off the call, and each
        # reading is wrong -- the dim actually consumed is whichever one the *coordinate*
        # is defined on, which the call does not say. Refusing is the fix.
        ("groupby", ("region",), {}),
        ("groupby", ("region.foo",), {}),
        ("groupby_bins", ("region", 2), {}),
        ("resample", (), {"region": "2D"}),
        # dotted bins: the head is a dim, so a head-only guard would admit it, but the
        # minted name is ``month_bins`` rather than ``time.month_bins`` (verified against
        # xarray 2026.7.0). Refused until it has its own goldens.
        ("groupby_bins", ("time.month", 2), {}),
    ],
)
def test_a_grouper_whose_head_is_not_a_dim_does_not_fuse(opener):
    """A grouper naming a coordinate rather than a dim refuses, and replays verbatim.

    Notes
    -----
    Issue #90: each of these reads a plausible ``group_dim`` off the call, and each
    reading is wrong — the dim actually consumed is whichever one the *coordinate* is
    defined on, which the call does not say.
    """
    lowered = _lower(opener, ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]
    assert [n.name for n in lowered] == [opener[0], "mean"]


def test_a_coordinate_grouper_fuses_once_its_name_is_a_dim():
    """The same grouper fuses against a dataset where its name really is a dim.

    Notes
    -----
    The guard tests membership, not spelling. Pins that the refusals above are the dim
    check firing rather than some new syntactic narrowing.
    """
    (node,) = _lower(
        ("groupby", ("region",), {}), ("mean", (), {}), dims=_DIMS | {"region"}
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
    """No ``ContextOpen`` survives lowering, across fused, refused and unclosed shapes.

    Notes
    -----
    The invariant the ``LoweredOp`` alias enforces at type-check time, asserted at runtime
    too.
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
    """A reduce following a refused pair is modelled normally, not swept into the barrier.

    Notes
    -----
    What retires the trailing barrier: only the pair goes opaque, so the ``mean`` that
    follows an unfusable ``first`` is a real ``Reduce`` again.
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
    """Idempotence holds for a plan that actually fused, not only for one that passed through.

    Notes
    -----
    Not incidental: the output contains no ``ContextOpen`` at all, so a second pass has
    nothing left to match.
    """
    once = _lower(
        ("groupby", ("time.month",), {}), ("mean", (), {}), ("cumsum", ("lat",), {})
    )
    assert to_lower_ir(once, _DIMS) == once
