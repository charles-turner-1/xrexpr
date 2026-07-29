"""Tests for the plan optimiser: the fixpoint loop and its rules.

Golden op-list assertions on :func:`~xrexpr.optimize.optimize`. Nodes are built with
``to_opnode`` — the same normalised metadata the real recorder produces, and a pure
function of the call — so these pin the optimiser without going through the accessor.
The ``schema`` fixture is passed to :func:`~xrexpr.optimize.optimize`, not to node
construction: it is the *base* the optimiser folds forward.
Covers merge-adjacent-selects (PR 7) and select-pushdown past reductions (PR 8): a
select hops left past any reduce with disjoint dims, and the two rules compose via the
fixpoint (bubble-then-merge). Classifying the non-disjoint conflict is PR 9's job.

Projection pushdown (#32) is the variable-level rule, and the one that needs the
``schema`` argument for more than decoration: whether ``[["temperature"]]`` may cross an
op depends on the dims that variable carries at that point, so the fixture dataset holds
a second variable (``elevation``) that is *missing* ``time`` — the case that must not be
reordered, and must not raise either.
"""

import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.indexers import classify
from xrexpr.ir import ALL_DIMS, GroupedReduce, WeightedReduce, WindowedReduce
from xrexpr.optimize import dim_effect, optimize
from xrexpr.schema import SchemaState, to_opnode


def _node(name, *args, **kwargs):
    return to_opnode(name, args, kwargs)


def _ix(**dims):
    """The ``indexer`` a ``Select`` would hold for these raw values (each classified)."""
    return frozendict({d: classify(v) for d, v in dims.items()})


def test_merge_consecutive_isel_kwargs(schema):
    plan = [_node("isel", time=0), _node("isel", lat=1)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].name == "isel"
    assert out[0].indexer == _ix(time=0, lat=1)
    assert out[0].args == ({"time": 0, "lat": 1},)
    assert out[0].consumes == frozenset({"time", "lat"})


def test_merge_consecutive_isel_positional_dict(schema):
    plan = [_node("isel", {"time": 0}), _node("isel", {"lat": 1})]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=0, lat=1)


def test_merge_run_of_three_isel(schema):
    plan = [
        _node("isel", time=0),
        _node("isel", lat=1),
        _node("isel", lon=2),
    ]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=0, lat=1, lon=2)


def test_merge_consecutive_sel(schema):
    plan = [_node("sel", lat=1), _node("sel", lon=2)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].name == "sel"
    assert out[0].indexer == _ix(lat=1, lon=2)


def test_isel_keeps_slice_dim_when_merging(schema):
    # a slice keeps its dim (no consume); scalar drops it -> merged consumes only lat
    plan = [_node("isel", time=slice(0, 2)), _node("isel", lat=1)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=slice(0, 2), lat=1)
    assert out[0].consumes == frozenset({"lat"})


def test_isel_and_sel_not_merged(schema):
    # different indexing semantics -> two separate nodes
    plan = [_node("isel", time=0), _node("sel", lat=1)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "sel"]


def test_option_kwarg_select_is_a_barrier(schema):
    # ``drop=True`` can't be carried by a bare indexer -> the two isels stay split
    plan = [_node("isel", time=0, drop=True), _node("isel", lat=1)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "isel"]
    assert out[0].kwargs == frozendict({"time": 0, "drop": True})  # verbatim, unmerged


def test_non_select_plan_unchanged(schema):
    plan = [_node("mean", "lat"), _node("mean", "lon")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "mean"]


def test_pushdown_isel_past_mean(schema):
    plan = [_node("mean", "lat"), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=0)


def test_pushdown_generalises_to_sum(schema):
    # the headline: sum (not just mean) now reorders too -> fixes the mean-only limit
    plan = [_node("sum", "lat"), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "sum"]


def test_pushdown_generalises_to_any_reduce(schema):
    # std / max / ... are reduces too; all push a disjoint select in front
    for reduce_op in ("std", "max", "median"):
        plan = [_node(reduce_op, "lat"), _node("isel", time=0)]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["isel", reduce_op]


def test_pushdown_sel_past_reduce(schema):
    plan = [_node("mean", "lat"), _node("sel", lon=2)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["sel", "mean"]


def test_select_on_reduced_dim_raises(schema):
    # isel indexes ``lat``, which mean("lat") already removed -> unreplayable
    plan = [_node("mean", "lat"), _node("isel", lat=0)]
    with pytest.raises(InvalidExpressionError, match="lat"):
        optimize(plan, schema)


def test_bare_mean_then_select_raises_empty_dim_bug(schema):
    # bare mean() consumes *every* dim, so a following isel is invalid -- the empty-dim
    # reorder bug, now caught instead of silently swapped
    plan = [_node("mean"), _node("isel", time=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, schema)


def test_bare_mean_needs_no_schema_to_reject_a_following_select():
    # ``ALL_DIMS`` makes the rejection independent of any dim names: whatever dims exist
    # when the reduce runs, it removes all of them, so *every* select dim intersects.
    # Note the empty schema -- under the old eager expansion ``consumes`` would have been
    # ``frozenset()`` here, and the select would have been swapped in front.
    plan = [_node("mean"), _node("isel", whatever=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, SchemaState())


def test_bare_mean_after_a_rename_rejects_a_select_on_the_new_name(schema):
    # The stale-name divergence. ``rename`` is Opaque and ``apply_schema`` models it as
    # dim-preserving, so the fold past it still says {time, lat, lon}. Expanding the bare
    # ``mean()`` against that recorded *stale* names: ``t2`` tested disjoint from them and
    # the select was swapped to the front, so the plan silently returned data where the
    # eager chain raises (``mean()`` leaves no ``t2`` to index). Symbolically there are no
    # names to be stale about.
    plan = [_node("rename", time="t2"), _node("mean"), _node("isel", t2=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, schema)


def test_bare_mean_still_admits_an_empty_select(schema):
    # ``isel()`` names no dim, so it intersects nothing even against ALL_DIMS -- the one
    # select that may still cross a bare reduce.
    plan = [_node("mean"), _node("isel")]
    assert [n.name for n in optimize(plan, schema)] == ["isel", "mean"]


def test_scan_then_select_on_scan_dim_left_untouched(schema):
    # cumsum is a scan, not a reduce: order matters, so leave it -- and never raise
    plan = [_node("cumsum", "time"), _node("isel", time=5)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_scan_then_disjoint_select_left_untouched(schema):
    # even a disjoint select is left behind a scan (pushdown only fires on reduces)
    plan = [_node("cumsum", "time"), _node("isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_pushdown_composes_past_two_reduces(schema):
    # the fixpoint hops the select left one reduce at a time until it reaches the front
    plan = [
        _node("mean", "lat"),
        _node("mean", "lon"),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean", "mean"]


def test_pushdown_then_merge_across_a_reduce(schema):
    # the trailing isel hops past mean and merges with the leading isel
    plan = [
        _node("isel", time=0),
        _node("mean", "lat"),
        _node("isel", lon=2),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=0, lon=2)


def test_pushdown_projection_past_reduce(schema):
    # the headline: only ``temperature`` is reduced, not every variable
    plan = [
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].variables == ("temperature",)


def test_pushdown_projection_past_bare_reduce(schema):
    # ``ALL_DIMS`` resolved against the schema *entering* the reduce: {time, lat, lon},
    # which ``temperature`` spans -- so the projection may lead and the replayed bare
    # ``mean()`` still reduces the same dims.
    plan = [_node("mean"), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]


def test_projection_not_pushed_past_bare_reduce_on_missing_dim(schema):
    # ``elevation`` lacks ``time``, so leading with it would leave the bare ``mean()``
    # reducing a smaller dim set than it does in the plan as written. Left alone, never
    # raised -- the chain is valid, merely immovable.
    plan = [_node("mean"), _node("__getitem__", ["elevation"])]
    assert optimize(plan, schema) == plan


def test_pushdown_single_variable_projection(schema):
    # ``ds["temperature"]`` (a DataArray result) pushes down just the same
    plan = [_node("mean", "time"), _node("__getitem__", "temperature")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].single


def test_pushdown_projection_past_select(schema):
    plan = [
        _node("isel", time=0),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "isel"]


def test_projection_not_pushed_past_reduce_on_missing_dim(schema):
    # ``elevation`` has no ``time``, so ``ds[["elevation"]].mean("time")`` would raise:
    # leave the plan alone -- and, unlike a select on a reduced dim, don't raise either
    plan = [_node("mean", "time"), _node("__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_not_pushed_past_select_on_missing_dim(schema):
    plan = [_node("isel", time=0), _node("__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "__getitem__"]


def test_projection_of_unknown_name_left_alone(schema):
    # not a tracked data variable (a coord, or something we can't see) -> no rewrite
    plan = [_node("mean", "time"), _node("__getitem__", ["lat"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_behind_an_opaque_left_alone(schema):
    # past an unmodelled op the schema's ``data_vars`` is a guess, so stay out
    plan = [
        _node("rename", {"temperature": "t2m"}),
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["rename", "mean", "__getitem__"]


def test_mask_style_getitem_is_not_a_projection(schema):
    # a dict key is xarray's ``isel`` spelling, not a projection -> opaque, unmoved
    plan = [_node("mean", "time"), _node("__getitem__", {"lat": 0})]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_composes_past_two_reduces(schema):
    plan = [
        _node("mean", "lat"),
        _node("mean", "lon"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean", "mean"]


def test_projection_and_select_pushdown_compose(schema):
    # both rules run: the select hops past the reduce, the projection past both
    plan = [
        _node("mean", "lat"),
        _node("isel", time=0),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "isel", "mean"]


def test_scalar_isel_past_rechunk_drops_the_spent_rechunk(schema):
    # the headline (#57): chunk({time: 100}).isel(time=0) -> isel(time=0). The rechunk's
    # only named dim is gone, and chunk({}) would buy nothing but a single-chunk array
    plan = [_node("chunk", {"time": 100}), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel"]
    assert out[0].indexer == _ix(time=0)


def test_scalar_isel_past_rechunk_strips_only_the_dropped_dim(schema):
    # lat survives the select, so the rechunk stays -- minus the dim that no longer exists
    plan = [
        _node("chunk", {"time": 100, "lat": 50}),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"lat": 50})
    assert out[1].args == ({"lat": 50},)  # replayable: no stale ``time`` key


def test_slice_isel_pushes_with_the_spec_intact(schema):
    # a slice keeps its dim, so nothing is stripped; pushing means the rechunk sees
    # less data *and* lands on regular blocks instead of ragged ones
    plan = [
        _node("chunk", {"time": 100}),
        _node("isel", time=slice(0, 2)),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"time": 100})


def test_select_on_unchunked_dim_is_a_plain_swap(schema):
    plan = [_node("chunk", {"lat": 2}), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"lat": 2})


def test_rechunk_kwarg_form_pushes(schema):
    plan = [_node("chunk", time=100), _node("isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"time": 100})


def test_uniform_rechunk_forms_push_and_are_kept(schema):
    # chunk() / chunk(100) / chunk("auto") name no dim: nothing to strip, nothing spent.
    # "auto" simply re-picks block sizes against whatever survives the select
    for args in ((), (100,), ("auto",)):
        plan = [_node("chunk", *args), _node("isel", time=0)]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["isel", "chunk"]
        assert out[1].args == args


def test_explicit_block_tuple_is_a_barrier(schema):
    # blocks must sum to the dim length, so a select must never cross -- not even a
    # scalar one, though that case would merely strip the key
    for indexer in ({"time": 0}, {"time": slice(0, 2)}):
        plan = [
            _node("chunk", {"time": (1, 1, 2)}),
            _node("isel", **indexer),
        ]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["chunk", "isel"]


def test_rechunk_option_kwarg_is_a_barrier(schema):
    # a rebuilt spec couldn't carry the option faithfully -> leave the call alone
    plan = [
        _node("chunk", {"time": 100}, chunked_array_type="dask"),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk", "isel"]


def test_rechunk_never_raises_on_a_reduced_dim(schema):
    # unlike a reduce, a rechunk can't make a select unreplayable
    plan = [_node("chunk", {"time": 100}), _node("sel", time=0)]
    assert [n.name for n in optimize(plan, schema)] == ["sel"]


def test_select_reaches_the_front_past_rechunk_and_reduce(schema):
    # the fixpoint composes both pushdown rules
    plan = [
        _node("chunk", {"time": 100}),
        _node("mean", "lat"),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]


def test_rechunk_pushdown_is_idempotent(schema):
    plan = [
        _node("chunk", {"time": 100, "lat": 50}),
        _node("isel", time=0),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_empty_plan(schema):
    assert optimize([], schema) == []


def test_optimize_is_idempotent(schema):
    plan = [
        _node("isel", time=0),
        _node("isel", lat=1),
        _node("mean", "lon"),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_optimize_is_idempotent_with_a_projection(schema):
    plan = [
        _node("mean", "lat"),
        _node("__getitem__", ["temperature"]),
        _node("isel", time=0),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


# --- same-dim composition (#33) ------------------------------------------------
#
# The regression these guard: a run of selects used to merge by ``dict.update``, so a
# second indexer on an already-indexed dim *replaced* the first instead of composing
# with it. The later indexer addresses positions within the earlier one's result, so
# ``isel(time=slice(100, 1000)).isel(time=slice(10, 20))`` is ``slice(110, 120)``, not
# ``slice(10, 20)``. Cases with no provable composition must end the run (two nodes)
# rather than fold to a wrong one.


@pytest.mark.parametrize(
    "outer, inner, expected",
    [
        (slice(100, 1000), slice(10, 20), slice(110, 120, 1)),
        (slice(100, None), slice(10, 20), slice(110, 120, 1)),
        (slice(None, 50), slice(10, None), slice(10, 50, 1)),
        (slice(None), slice(2, 5), slice(2, 5, 1)),
        (slice(10, 100, 2), slice(1, 4), slice(12, 18, 2)),  # every 2nd, then 3 of them
        (slice(10, 100), slice(0, 5, 3), slice(10, 15, 3)),
        (slice(100, 1000), 5, 105),  # slice then scalar -> scalar (dim drops)
        (slice(10, 20), slice(50, 60), slice(60, 20, 1)),  # past the end -> empty
        ([10, 20, 30], 1, 20),  # concrete positions: just index them
        ([10, 20, 30], slice(1, 3), [20, 30]),
        ([10, 20, 30], [2, 0], [30, 10]),
        ([10, 20, 30], -1, 30),  # negatives are exact against a known sequence
    ],
)
def test_same_dim_selects_compose(schema, outer, inner, expected):
    plan = [_node("isel", time=outer), _node("isel", time=inner)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=expected)


@pytest.mark.parametrize(
    "outer, inner",
    [
        (slice(0, 10), slice(-3, None)),  # negative bound needs the dim length
        (slice(0, 10), slice(None, None, -1)),  # reversal needs the dim length
        (slice(-5, None), slice(0, 2)),
        (slice(0, 10), -1),  # negative scalar into a bounded slice
        (slice(0, 10), 20),  # out of the outer slice's range: xarray should raise
        (0, 0),  # scalar outer already dropped the dim
        ([10, 20, 30], 7),  # out of range against a known sequence
    ],
)
def test_uncomposable_same_dim_selects_are_left_separate(schema, outer, inner):
    plan = [_node("isel", time=outer), _node("isel", time=inner)]
    out = optimize(plan, schema)
    assert [n.indexer for n in out] == [_ix(time=outer), _ix(time=inner)]


def test_same_dim_sel_is_never_composed(schema):
    # label indexers would need coordinate values to compose; positions are all we have
    plan = [
        _node("sel", time=slice(0, 10)),
        _node("sel", time=slice(2, 4)),
    ]
    assert len(optimize(plan, schema)) == 2


def test_uncomposable_dim_abandons_the_whole_merge(schema):
    # lat *could* merge, but time can't -> neither does, or the plan would be neither select
    plan = [
        _node("isel", time=0, lat=slice(0, 3)),
        _node("isel", time=0, lat=1),
    ]
    out = optimize(plan, schema)
    assert len(out) == 2
    assert out[0].indexer == _ix(time=0, lat=slice(0, 3))


def test_composed_run_of_three_on_one_dim(schema):
    plan = [
        _node("isel", time=slice(100, 1000)),
        _node("isel", time=slice(10, 20)),
        _node("isel", time=2),
    ]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=112)
    assert out[0].consumes == frozenset({"time"})


def test_composition_survives_pushdown_past_a_reduce(schema):
    # the trailing isel hops past mean, then composes with the leading one on time
    plan = [
        _node("isel", time=slice(1, 4)),
        _node("mean", "lat"),
        _node("isel", time=1),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=2)


# --- weighted reduces: the node that exists to be left alone (#02-lowering §8.1) ------


def _weighted(
    reduce="mean", consumes=frozenset({"lat"}), weight_dims=frozenset({"lat"})
):
    """A ``WeightedReduce`` as ``to_lower_ir`` would build it, weights payload included."""
    weights = xr.DataArray([1.0, 2.0, 3.0], dims="lat")
    return WeightedReduce(
        name="weighted",
        reduce=reduce,
        weight_dims=weight_dims,
        args=(weights,),
        reduce_args=(sorted(consumes),),
        consumes=consumes,
    )


@pytest.mark.parametrize(
    "trailing",
    [
        # the rewrite the variant exists to block: hopping this left would run the
        # weighted mean over one latitude with the *weights* left un-subset
        lambda: _node("isel", time=0),
        lambda: _node("chunk", {"time": 2}),
    ],
)
def test_no_select_or_rechunk_crosses_a_weighted_reduce(schema, trailing):
    # ``WeightedReduce``'s ``consumes`` is exactly a plain reduce's, so lowered to a
    # ``Reduce`` it would be indistinguishable from one and ``pushdown_selects`` would
    # match it. It is a separate variant precisely so no select can. Subsetting the weights
    # alongside is a data-touching rewrite with its own workstream (§8.1). A *projection*
    # does cross it -- see ``test_a_projection_hops_past_a_weighted_reduce`` -- because it
    # discards variables instead of subsetting them, so the weights need no rewriting.
    plan = [_weighted(), trailing()]
    assert optimize(plan, schema) == plan


def test_a_select_on_a_weighted_reduces_own_dim_is_left_not_raised(schema):
    # ``pushdown_selects`` raises on an intersecting ``(Reduce, Select)`` because the plan
    # can never replay. That leg is deliberately out of reach here: the rule matches
    # ``Reduce`` only, so xarray reports the invalid selection at replay in its own words
    # rather than the optimiser guessing on a node it models less exactly.
    plan = [_weighted(consumes=frozenset({"lat"})), _node("isel", lat=0)]
    assert optimize(plan, schema) == plan


def test_a_bare_weighted_closer_is_left_alone_too(schema):
    # ``ALL_DIMS`` is the shape that most invites a rule to fire, since it is what
    # ``pushdown_selects``'s no-schema-needed reasoning keys on.
    plan = [
        WeightedReduce(name="weighted", reduce="mean", consumes=ALL_DIMS),
        _node("isel", time=0),
    ]
    assert optimize(plan, schema) == plan


def test_ops_after_a_weighted_reduce_are_still_optimised(schema):
    # The payoff for modelling the pair as one node: lowering opaques only the *pair*, so
    # the plan past it is inside the trusted prefix and rewrites normally (§8.1's closing
    # claim). Under the accessor's barrier everything from ``weighted`` onward was opaque
    # forever. Since W2 PR 9 the projection does not stop at the trailing reduce either --
    # ``temperature`` carries both the consumed and the weight dims, so it reaches the front
    # and the weighted mean itself runs over one variable instead of two.
    plan = [
        _weighted(consumes=frozenset({"lat"})),
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "WeightedReduce", "Reduce"]


# --- select pushdown past the fused reduces (W2 PR 6) ---------------------------------


def _grouped(group_dim="time", new_dim="month", consumes=frozenset()):
    """A ``GroupedReduce`` as ``to_lower_ir`` would build it from ``groupby(...).mean()``."""
    return GroupedReduce(
        name="groupby",
        group_dim=group_dim,
        new_dim=new_dim,
        reduce="mean",
        args=(f"{group_dim}.{new_dim}" if group_dim != new_dim else group_dim,),
        consumes=consumes,
    )


def _windowed(name="rolling", window=frozendict({"time": 3})):
    return WindowedReduce(name=name, reduce="mean", window=window, kwargs=dict(window))


@pytest.mark.parametrize(
    ("fused", "select"),
    [
        # the headline: a climatology over one latitude instead of all of them
        (_grouped(), lambda: _node("isel", lat=0)),
        (_grouped(), lambda: _node("isel", lat=[0, 2])),
        (_grouped(), lambda: _node("sel", lon=slice(1, 3))),
        (_grouped(group_dim="lat", new_dim="lat"), lambda: _node("isel", lon=0)),
        (_windowed(), lambda: _node("isel", lat=0)),
        (
            _windowed(name="coarsen", window=frozendict({"time": 2})),
            lambda: _node("isel", lat=0),
        ),
    ],
)
def test_select_hops_past_a_fused_reduce_on_disjoint_dims(schema, fused, select):
    out = optimize([fused, select()], schema)
    assert [type(n).__name__ for n in out] == [
        "Select",
        type(fused).__name__,
    ]


@pytest.mark.parametrize(
    ("fused", "select", "why"),
    [
        (_grouped(), lambda: _node("isel", month=0), "the minted dim"),
        (_grouped(), lambda: _node("isel", time=0), "the consumed group dim"),
        (
            _grouped(consumes=frozenset({"lat"})),
            lambda: _node("isel", lat=0),
            "a dim the closer also reduced",
        ),
        (
            _grouped(group_dim="lat", new_dim="lat"),
            lambda: _node("isel", lat=0),
            "group dim and minted dim coincide",
        ),
        (_windowed(), lambda: _node("isel", time=0), "a windowed dim"),
        (
            _windowed(name="coarsen", window=frozendict({"time": 2})),
            lambda: _node("isel", time=0),
            "a coarsened dim, whose size the node changes",
        ),
        (_grouped(), lambda: _node("isel", lat=0, month=1), "one dim of several"),
    ],
)
def test_intersecting_select_is_left_alone_never_raised(schema, fused, select, why):
    # The scan discipline, not ``pushdown_selects``'. Three of these are *valid* eagerly --
    # selecting the minted dim, or a dim a window kept -- so raising would reject a working
    # chain; the group-dim one is invalid, and leaving it lets xarray report that at replay
    # rather than the optimiser guessing. Either way: leave, never raise.
    plan = [fused, select()]
    assert optimize(plan, schema) == plan, why


def test_select_reaches_the_front_past_a_fused_reduce_and_a_reduce(schema):
    # The fixpoint composing two different pushdown rules: the select hops the grouped
    # node, then the plain reduce, and ends up first.
    plan = [_grouped(), _node("mean", "lat"), _node("isel", lon=0)]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Select", "GroupedReduce", "Reduce"]


def test_a_hopped_select_then_merges_with_the_one_in_front_of_it(schema):
    # ...and then composes with ``merge_adjacent_selects``, which is the whole point of
    # running local rules to a fixpoint.
    plan = [_node("isel", lat=0), _grouped(), _node("isel", lon=1)]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Select", "GroupedReduce"]
    assert out[0].indexer == _ix(lat=0, lon=1)


def test_no_select_hops_past_a_weighted_reduce(schema):
    # The variant exists to block exactly this rule: the dim algebra would permit the hop,
    # but the weights would be left un-subset. ``WeightedReduce`` is absent from the rule's
    # match, so it cannot fire.
    plan = [_weighted(consumes=frozenset({"lat"})), _node("isel", time=0)]
    assert optimize(plan, schema) == plan


# --- projection pushdown past the fused reduces (W2 PR 7) -----------------------------


@pytest.mark.parametrize(
    "fused",
    [
        _grouped(),
        _grouped(group_dim="lat", new_dim="lat"),
        _windowed(),
        _windowed(name="coarsen", window=frozendict({"time": 2})),
    ],
)
def test_projection_hops_past_a_fused_reduce_when_the_dims_survive(schema, fused):
    # ``temperature(time, lat, lon)`` carries every dim these name, so projecting first
    # aggregates one variable instead of two and then discarding one.
    plan = [fused, _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", type(fused).__name__]


@pytest.mark.parametrize(
    ("fused", "why"),
    [
        (_grouped(), "the group dim is not carried by the projected variable"),
        (
            _grouped(group_dim="time", new_dim="time"),
            "same, for a self-minting grouper",
        ),
        (_windowed(), "the window dim is not carried by the projected variable"),
        (
            _grouped(group_dim="lat", new_dim="lat", consumes=frozenset({"time"})),
            "the group dim survives but the extra consumes does not",
        ),
    ],
)
def test_projection_is_left_when_the_fused_node_would_lose_its_dim(schema, fused, why):
    # ``elevation(lat, lon)`` has no ``time``. Projecting first would drop the ``time``
    # coord entirely -- no surviving variable uses it -- and every fused kind then *raises*
    # rather than quietly doing nothing. Left alone, never raised: the chain is valid
    # eagerly, just not reorderable.
    plan = [fused, _node("__getitem__", ["elevation"])]
    assert optimize(plan, schema) == plan, why


def test_projection_needed_set_excludes_the_minted_dim(schema):
    # The asymmetry with the select rule, pinned: a grouped reduce *mints* ``month``, but
    # the projection runs in front of it where no ``month`` exists, so requiring the
    # projected variables to carry it would block every hop. ``DimEffect.blocks`` includes
    # it, because a select comes from the other side; ``requires`` must not.
    plan = [_grouped(new_dim="month"), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "GroupedReduce"]


def test_a_projection_hops_past_a_weighted_reduce(schema):
    # The sharpest instance of §8's contract. As written this chain *raises*: a weighted
    # reduce maps per variable and ``elevation`` has no ``time``, where a plain
    # ``.mean("time")`` would merely waste effort on it. Hopping the projection in front
    # discards ``elevation`` before the reduce sees it, so the value the plan actually asked
    # for is delivered instead of an error about work it was throwing away.
    plan = [
        _weighted(consumes=frozenset({"time"}), weight_dims=frozenset({"lat"})),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "WeightedReduce"]


def test_a_projection_is_left_when_it_would_orphan_a_weight_dim(schema):
    # The guard that makes the hop above sound, and the one case where a weighted reduce's
    # ``needed`` set is strictly larger than a plain reduce's. ``elevation`` has no ``time``,
    # so projecting to it drops the ``time`` *coordinate* -- and the weights carry ``time``,
    # so they would switch from inner-joining against that coord to broadcasting a fresh
    # one. Verified against xarray: that changes values, not just errors. Drop the
    # ``weight_dims`` term from ``requires`` and this plan hops, wrongly.
    plan = [
        _weighted(consumes=frozenset({"lat"}), weight_dims=frozenset({"time"})),
        _node("__getitem__", ["elevation"]),
    ]
    assert optimize(plan, schema) == plan


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        # ``temperature`` spans every dim, so the replayed bare closer reduces the same
        # ones and no coord can go missing -- a bare closer clears the weight dims too
        (["temperature"], ["Project", "WeightedReduce"]),
        # ``elevation`` does not span ``time``, so the bare closer would reduce a different
        # set of dims after the hop
        (["elevation"], ["WeightedReduce", "Project"]),
    ],
)
def test_a_bare_weighted_closer_admits_a_projection_spanning_every_dim(
    schema, variables, expected
):
    # ``ALL_DIMS`` needs no ``weight_dims`` term: a bare closer clears every dim, minted
    # weight dims included, so nothing survives for a coordinate to be missing from. It
    # also could not have one -- a weight dim the dataset lacks can never be a subset of
    # the projected variables' dims, so requiring it would refuse every hop.
    plan = [
        WeightedReduce(name="weighted", reduce="mean", consumes=ALL_DIMS),
        _node("__getitem__", variables),
    ]
    assert [type(n).__name__ for n in optimize(plan, schema)] == expected


def test_projection_and_select_both_cross_a_fused_reduce(schema):
    # The two PR 6 / PR 7 rules composing through the fixpoint: both end up in front of
    # the grouping, which is the plan the workstream was aiming at.
    plan = [_grouped(), _node("isel", lat=0), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "Select", "GroupedReduce"]


# --- the derived dim effect (02-lowering §11.1) ---------------------------------------


@pytest.mark.parametrize(
    ("node", "blocks", "requires"),
    [
        # a reduce removes exactly what it names, so both answers coincide
        (_node("mean", "lat"), frozenset({"lat"}), frozenset({"lat"})),
        (_node("mean"), ALL_DIMS, ALL_DIMS),
        # a select is never crossed by another select, but is by a projection
        (_node("isel", time=0), None, frozenset({"time"})),
        # the asymmetry the two fields exist for: ``new_dim`` blocks a select coming from
        # the right, but is *not* required of the input a projection would leave behind
        (_grouped(), frozenset({"time", "month"}), frozenset({"time"})),
        (
            _grouped(consumes=frozenset({"lat"})),
            frozenset({"time", "month", "lat"}),
            frozenset({"time", "lat"}),
        ),
        # a window resizes what it names and needs the same dims present
        (_windowed(), frozenset({"time"}), frozenset({"time"})),
        # nothing crosses these, in either direction
        (_node("cumsum", "time"), None, None),
        (_node("where", "cond"), None, None),
        (_node("chunk", {"time": 2}), None, None),
        (_node("__getitem__", ["temperature"]), None, None),
        # a select never crosses a weighted reduce (the weights would need subsetting), but
        # a projection does -- and must supply the weight dims as well as the consumed ones,
        # or it can orphan the coord the weights align against
        (_weighted(consumes=frozenset({"time"})), None, frozenset({"time", "lat"})),
        (
            WeightedReduce(name="weighted", reduce="mean", consumes=ALL_DIMS),
            None,
            ALL_DIMS,
        ),
    ],
)
def test_dim_effect_table(node, blocks, requires):
    effect = dim_effect(node)
    assert effect.blocks == blocks
    assert effect.requires == requires


def test_only_a_plain_reduce_calls_a_conflict_invalid():
    # A reduce *removes* what it names, so a select on one of those dims can never replay
    # and the optimiser is entitled to reject the chain. Every other node either mints or
    # keeps the dims it blocks, so an overlapping select there is merely immovable -- the
    # distinction that lets one rule serve both, and the one a new variant must get right.
    invalid = [
        node
        for node in (
            _node("mean", "lat"),
            _grouped(),
            _windowed(),
            _weighted(),
            _node("cumsum", "time"),
        )
        if dim_effect(node).on_conflict == "invalid"
    ]
    assert invalid == [_node("mean", "lat")]


def test_dim_effect_covers_every_lowered_variant():
    # ``assert_never`` makes this a type error too; asserted at runtime as well so the
    # table above cannot silently stop covering the union it claims to.
    from typing import get_args

    from xrexpr.ir import LoweredOp

    covered = {
        type(node)
        for node in (
            _node("mean", "lat"),
            _node("isel", time=0),
            _node("cumsum", "time"),
            _node("__getitem__", ["temperature"]),
            _node("chunk", {"time": 2}),
            _node("where", "cond"),
            _grouped(),
            _windowed(),
            _weighted(),
        )
    }
    assert covered == set(get_args(LoweredOp))
