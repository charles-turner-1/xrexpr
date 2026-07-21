"""Tests for the plan optimiser: the fixpoint loop and its rules.

Golden op-list assertions on :func:`~xrexpr.optimize.optimize`. Nodes are built with
``to_opnode`` (PR 5) against a fixed schema — the same normalised metadata the real
recorder produces — so these pin the optimiser without going through the accessor.
Covers merge-adjacent-selects (PR 7) and select-pushdown past reductions (PR 8): a
select hops left past any reduce with disjoint dims, and the two rules compose via the
fixpoint (bubble-then-merge). Classifying the non-disjoint conflict is PR 9's job.
"""

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, to_opnode


@pytest.fixture
def schema() -> SchemaState:
    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.zeros((4, 3, 5)))},
        coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(5)},
    )
    return SchemaState.from_dataset(ds)


def _node(schema, name, *args, **kwargs):
    return to_opnode(schema, name, args, kwargs)


def test_merge_consecutive_isel_kwargs(schema):
    plan = [_node(schema, "isel", time=0), _node(schema, "isel", lat=1)]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].name == "isel"
    assert out[0].indexer == frozendict({"time": 0, "lat": 1})
    assert out[0].args == ({"time": 0, "lat": 1},)
    assert out[0].consumes == frozenset({"time", "lat"})


def test_merge_consecutive_isel_positional_dict(schema):
    plan = [_node(schema, "isel", {"time": 0}), _node(schema, "isel", {"lat": 1})]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": 0, "lat": 1})


def test_merge_run_of_three_isel(schema):
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "isel", lat=1),
        _node(schema, "isel", lon=2),
    ]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": 0, "lat": 1, "lon": 2})


def test_merge_consecutive_sel(schema):
    plan = [_node(schema, "sel", lat=1), _node(schema, "sel", lon=2)]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].name == "sel"
    assert out[0].indexer == frozendict({"lat": 1, "lon": 2})


def test_isel_keeps_slice_dim_when_merging(schema):
    # a slice keeps its dim (no consume); scalar drops it -> merged consumes only lat
    plan = [_node(schema, "isel", time=slice(0, 2)), _node(schema, "isel", lat=1)]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": slice(0, 2), "lat": 1})
    assert out[0].consumes == frozenset({"lat"})


def test_isel_and_sel_not_merged(schema):
    # different indexing semantics -> two separate nodes
    plan = [_node(schema, "isel", time=0), _node(schema, "sel", lat=1)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "sel"]


def test_option_kwarg_select_is_a_barrier(schema):
    # ``drop=True`` can't be carried by a bare indexer -> the two isels stay split
    plan = [_node(schema, "isel", time=0, drop=True), _node(schema, "isel", lat=1)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "isel"]
    assert out[0].kwargs == frozendict({"time": 0, "drop": True})  # verbatim, unmerged


def test_non_select_plan_unchanged(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "mean", "lon")]
    out = optimize(plan)
    assert [n.name for n in out] == ["mean", "mean"]


def test_pushdown_isel_past_mean(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "isel", time=0)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == frozendict({"time": 0})


def test_pushdown_generalises_to_sum(schema):
    # the headline: sum (not just mean) now reorders too -> fixes the mean-only limit
    plan = [_node(schema, "sum", "lat"), _node(schema, "isel", time=0)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "sum"]


def test_pushdown_generalises_to_any_reduce(schema):
    # std / max / ... are reduces too; all push a disjoint select in front
    for reduce_op in ("std", "max", "median"):
        plan = [_node(schema, reduce_op, "lat"), _node(schema, "isel", time=0)]
        out = optimize(plan)
        assert [n.name for n in out] == ["isel", reduce_op]


def test_pushdown_sel_past_reduce(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "sel", lon=2)]
    out = optimize(plan)
    assert [n.name for n in out] == ["sel", "mean"]


def test_select_on_reduced_dim_raises(schema):
    # isel indexes ``lat``, which mean("lat") already removed -> unreplayable
    plan = [_node(schema, "mean", "lat"), _node(schema, "isel", lat=0)]
    with pytest.raises(InvalidExpressionError, match="lat"):
        optimize(plan)


def test_bare_mean_then_select_raises_empty_dim_bug(schema):
    # bare mean() consumes *every* dim (PR 5), so a following isel is invalid -- the
    # empty-dim reorder bug, now caught instead of silently swapped
    plan = [_node(schema, "mean"), _node(schema, "isel", time=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan)


def test_scan_then_select_on_scan_dim_left_untouched(schema):
    # cumsum is a scan, not a reduce: order matters, so leave it -- and never raise
    plan = [_node(schema, "cumsum", "time"), _node(schema, "isel", time=5)]
    out = optimize(plan)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_scan_then_disjoint_select_left_untouched(schema):
    # even a disjoint select is left behind a scan (pushdown only fires on reduces)
    plan = [_node(schema, "cumsum", "time"), _node(schema, "isel", lat=0)]
    out = optimize(plan)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_pushdown_composes_past_two_reduces(schema):
    # the fixpoint hops the select left one reduce at a time until it reaches the front
    plan = [
        _node(schema, "mean", "lat"),
        _node(schema, "mean", "lon"),
        _node(schema, "isel", time=0),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "mean", "mean"]


def test_pushdown_then_merge_across_a_reduce(schema):
    # the trailing isel hops past mean and merges with the leading isel
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "mean", "lat"),
        _node(schema, "isel", lon=2),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == frozendict({"time": 0, "lon": 2})


def test_empty_plan(schema):
    assert optimize([]) == []


def test_optimize_is_idempotent(schema):
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "isel", lat=1),
        _node(schema, "mean", "lon"),
    ]
    once = optimize(plan)
    assert optimize(once) == once


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
    plan = [_node(schema, "isel", time=outer), _node(schema, "isel", time=inner)]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": expected})


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
    plan = [_node(schema, "isel", time=outer), _node(schema, "isel", time=inner)]
    out = optimize(plan)
    assert [n.indexer for n in out] == [
        frozendict({"time": outer}),
        frozendict({"time": inner}),
    ]


def test_same_dim_sel_is_never_composed(schema):
    # label indexers would need coordinate values to compose; positions are all we have
    plan = [
        _node(schema, "sel", time=slice(0, 10)),
        _node(schema, "sel", time=slice(2, 4)),
    ]
    assert len(optimize(plan)) == 2


def test_uncomposable_dim_abandons_the_whole_merge(schema):
    # lat *could* merge, but time can't -> neither does, or the plan would be neither select
    plan = [
        _node(schema, "isel", time=0, lat=slice(0, 3)),
        _node(schema, "isel", time=0, lat=1),
    ]
    out = optimize(plan)
    assert len(out) == 2
    assert out[0].indexer == frozendict({"time": 0, "lat": slice(0, 3)})


def test_composed_run_of_three_on_one_dim(schema):
    plan = [
        _node(schema, "isel", time=slice(100, 1000)),
        _node(schema, "isel", time=slice(10, 20)),
        _node(schema, "isel", time=2),
    ]
    out = optimize(plan)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": 112})
    assert out[0].consumes == frozenset({"time"})


def test_composition_survives_pushdown_past_a_reduce(schema):
    # the trailing isel hops past mean, then composes with the leading one on time
    plan = [
        _node(schema, "isel", time=slice(1, 4)),
        _node(schema, "mean", "lat"),
        _node(schema, "isel", time=1),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == frozendict({"time": 2})
