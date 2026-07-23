"""Tests for the plan optimiser: the fixpoint loop and its rules.

Golden op-list assertions on :func:`~xrexpr.optimize.optimize`. Nodes are built with
``to_opnode`` (PR 5) against a fixed schema — the same normalised metadata the real
recorder produces — so these pin the optimiser without going through the accessor.
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
from frozendict import frozendict

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.optimize import optimize
from xrexpr.schema import to_opnode


def _node(schema, name, *args, **kwargs):
    return to_opnode(schema, name, args, kwargs)


def test_merge_consecutive_isel_kwargs(schema):
    plan = [_node(schema, "isel", time=0), _node(schema, "isel", lat=1)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].name == "isel"
    assert out[0].indexer == frozendict({"time": 0, "lat": 1})
    assert out[0].args == ({"time": 0, "lat": 1},)
    assert out[0].consumes == frozenset({"time", "lat"})


def test_merge_consecutive_isel_positional_dict(schema):
    plan = [_node(schema, "isel", {"time": 0}), _node(schema, "isel", {"lat": 1})]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": 0, "lat": 1})


def test_merge_run_of_three_isel(schema):
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "isel", lat=1),
        _node(schema, "isel", lon=2),
    ]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": 0, "lat": 1, "lon": 2})


def test_merge_consecutive_sel(schema):
    plan = [_node(schema, "sel", lat=1), _node(schema, "sel", lon=2)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].name == "sel"
    assert out[0].indexer == frozendict({"lat": 1, "lon": 2})


def test_isel_keeps_slice_dim_when_merging(schema):
    # a slice keeps its dim (no consume); scalar drops it -> merged consumes only lat
    plan = [_node(schema, "isel", time=slice(0, 2)), _node(schema, "isel", lat=1)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == frozendict({"time": slice(0, 2), "lat": 1})
    assert out[0].consumes == frozenset({"lat"})


def test_isel_and_sel_not_merged(schema):
    # different indexing semantics -> two separate nodes
    plan = [_node(schema, "isel", time=0), _node(schema, "sel", lat=1)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "sel"]


def test_option_kwarg_select_is_a_barrier(schema):
    # ``drop=True`` can't be carried by a bare indexer -> the two isels stay split
    plan = [_node(schema, "isel", time=0, drop=True), _node(schema, "isel", lat=1)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "isel"]
    assert out[0].kwargs == frozendict({"time": 0, "drop": True})  # verbatim, unmerged


def test_non_select_plan_unchanged(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "mean", "lon")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "mean"]


def test_pushdown_isel_past_mean(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == frozendict({"time": 0})


def test_pushdown_generalises_to_sum(schema):
    # the headline: sum (not just mean) now reorders too -> fixes the mean-only limit
    plan = [_node(schema, "sum", "lat"), _node(schema, "isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "sum"]


def test_pushdown_generalises_to_any_reduce(schema):
    # std / max / ... are reduces too; all push a disjoint select in front
    for reduce_op in ("std", "max", "median"):
        plan = [_node(schema, reduce_op, "lat"), _node(schema, "isel", time=0)]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["isel", reduce_op]


def test_pushdown_sel_past_reduce(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "sel", lon=2)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["sel", "mean"]


def test_select_on_reduced_dim_raises(schema):
    # isel indexes ``lat``, which mean("lat") already removed -> unreplayable
    plan = [_node(schema, "mean", "lat"), _node(schema, "isel", lat=0)]
    with pytest.raises(InvalidExpressionError, match="lat"):
        optimize(plan, schema)


def test_bare_mean_then_select_raises_empty_dim_bug(schema):
    # bare mean() consumes *every* dim (PR 5), so a following isel is invalid -- the
    # empty-dim reorder bug, now caught instead of silently swapped
    plan = [_node(schema, "mean"), _node(schema, "isel", time=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, schema)


def test_scan_then_select_on_scan_dim_left_untouched(schema):
    # cumsum is a scan, not a reduce: order matters, so leave it -- and never raise
    plan = [_node(schema, "cumsum", "time"), _node(schema, "isel", time=5)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_scan_then_disjoint_select_left_untouched(schema):
    # even a disjoint select is left behind a scan (pushdown only fires on reduces)
    plan = [_node(schema, "cumsum", "time"), _node(schema, "isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_pushdown_composes_past_two_reduces(schema):
    # the fixpoint hops the select left one reduce at a time until it reaches the front
    plan = [
        _node(schema, "mean", "lat"),
        _node(schema, "mean", "lon"),
        _node(schema, "isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean", "mean"]


def test_pushdown_then_merge_across_a_reduce(schema):
    # the trailing isel hops past mean and merges with the leading isel
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "mean", "lat"),
        _node(schema, "isel", lon=2),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == frozendict({"time": 0, "lon": 2})


def test_pushdown_projection_past_reduce(schema):
    # the headline: only ``temperature`` is reduced, not every variable
    plan = [
        _node(schema, "mean", "time"),
        _node(schema, "__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].variables == ("temperature",)


def test_pushdown_single_variable_projection(schema):
    # ``ds["temperature"]`` (a DataArray result) pushes down just the same
    plan = [_node(schema, "mean", "time"), _node(schema, "__getitem__", "temperature")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].single


def test_pushdown_projection_past_select(schema):
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "isel"]


def test_projection_not_pushed_past_reduce_on_missing_dim(schema):
    # ``elevation`` has no ``time``, so ``ds[["elevation"]].mean("time")`` would raise:
    # leave the plan alone -- and, unlike a select on a reduced dim, don't raise either
    plan = [_node(schema, "mean", "time"), _node(schema, "__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_not_pushed_past_select_on_missing_dim(schema):
    plan = [_node(schema, "isel", time=0), _node(schema, "__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "__getitem__"]


def test_projection_of_unknown_name_left_alone(schema):
    # not a tracked data variable (a coord, or something we can't see) -> no rewrite
    plan = [_node(schema, "mean", "time"), _node(schema, "__getitem__", ["lat"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_behind_an_opaque_left_alone(schema):
    # past an unmodelled op the schema's ``data_vars`` is a guess, so stay out
    plan = [
        _node(schema, "rename", {"temperature": "t2m"}),
        _node(schema, "mean", "time"),
        _node(schema, "__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["rename", "mean", "__getitem__"]


def test_mask_style_getitem_is_not_a_projection(schema):
    # a dict key is xarray's ``isel`` spelling, not a projection -> opaque, unmoved
    plan = [_node(schema, "mean", "time"), _node(schema, "__getitem__", {"lat": 0})]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_composes_past_two_reduces(schema):
    plan = [
        _node(schema, "mean", "lat"),
        _node(schema, "mean", "lon"),
        _node(schema, "__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean", "mean"]


def test_projection_and_select_pushdown_compose(schema):
    # both rules run: the select hops past the reduce, the projection past both
    plan = [
        _node(schema, "mean", "lat"),
        _node(schema, "isel", time=0),
        _node(schema, "__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "isel", "mean"]


def test_empty_plan(schema):
    assert optimize([], schema) == []


def test_optimize_is_idempotent(schema):
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "isel", lat=1),
        _node(schema, "mean", "lon"),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_optimize_is_idempotent_with_a_projection(schema):
    plan = [
        _node(schema, "mean", "lat"),
        _node(schema, "__getitem__", ["temperature"]),
        _node(schema, "isel", time=0),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once
