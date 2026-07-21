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


def test_scalar_isel_past_rechunk_drops_the_spent_rechunk(schema):
    # the headline (#57): chunk({time: 100}).isel(time=0) -> isel(time=0). The rechunk's
    # only named dim is gone, and chunk({}) would buy nothing but a single-chunk array
    plan = [_node(schema, "chunk", {"time": 100}), _node(schema, "isel", time=0)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel"]
    assert out[0].indexer == frozendict({"time": 0})


def test_scalar_isel_past_rechunk_strips_only_the_dropped_dim(schema):
    # lat survives the select, so the rechunk stays -- minus the dim that no longer exists
    plan = [
        _node(schema, "chunk", {"time": 100, "lat": 50}),
        _node(schema, "isel", time=0),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"lat": 50})
    assert out[1].args == ({"lat": 50},)  # replayable: no stale ``time`` key


def test_slice_isel_pushes_with_the_spec_intact(schema):
    # a slice keeps its dim, so nothing is stripped; pushing means the rechunk sees
    # less data *and* lands on regular blocks instead of ragged ones
    plan = [
        _node(schema, "chunk", {"time": 100}),
        _node(schema, "isel", time=slice(0, 2)),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"time": 100})


def test_select_on_unchunked_dim_is_a_plain_swap(schema):
    plan = [_node(schema, "chunk", {"lat": 2}), _node(schema, "isel", time=0)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"lat": 2})


def test_rechunk_kwarg_form_pushes(schema):
    plan = [_node(schema, "chunk", time=100), _node(schema, "isel", lat=0)]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"time": 100})


def test_uniform_rechunk_forms_push_and_are_kept(schema):
    # chunk() / chunk(100) / chunk("auto") name no dim: nothing to strip, nothing spent.
    # "auto" simply re-picks block sizes against whatever survives the select
    for args in ((), (100,), ("auto",)):
        plan = [_node(schema, "chunk", *args), _node(schema, "isel", time=0)]
        out = optimize(plan)
        assert [n.name for n in out] == ["isel", "chunk"]
        assert out[1].args == args


def test_explicit_block_tuple_is_a_barrier(schema):
    # blocks must sum to the dim length, so a select must never cross -- not even a
    # scalar one, though that case would merely strip the key
    for indexer in ({"time": 0}, {"time": slice(0, 2)}):
        plan = [
            _node(schema, "chunk", {"time": (1, 1, 2)}),
            _node(schema, "isel", **indexer),
        ]
        out = optimize(plan)
        assert [n.name for n in out] == ["chunk", "isel"]


def test_rechunk_option_kwarg_is_a_barrier(schema):
    # a rebuilt spec couldn't carry the option faithfully -> leave the call alone
    plan = [
        _node(schema, "chunk", {"time": 100}, chunked_array_type="dask"),
        _node(schema, "isel", time=0),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["chunk", "isel"]


def test_rechunk_never_raises_on_a_reduced_dim(schema):
    # unlike a reduce, a rechunk can't make a select unreplayable
    plan = [_node(schema, "chunk", {"time": 100}), _node(schema, "sel", time=0)]
    assert [n.name for n in optimize(plan)] == ["sel"]


def test_select_reaches_the_front_past_rechunk_and_reduce(schema):
    # the fixpoint composes both pushdown rules
    plan = [
        _node(schema, "chunk", {"time": 100}),
        _node(schema, "mean", "lat"),
        _node(schema, "isel", time=0),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "mean"]


def test_rechunk_pushdown_is_idempotent(schema):
    plan = [
        _node(schema, "chunk", {"time": 100, "lat": 50}),
        _node(schema, "isel", time=0),
    ]
    once = optimize(plan)
    assert optimize(once) == once


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
