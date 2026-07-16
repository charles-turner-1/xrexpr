"""Tests for the plan optimiser (PR 7): the fixpoint loop + merge-adjacent-selects.

Golden op-list assertions on :func:`~xrexpr.optimize.optimize`. Nodes are built with
``to_opnode`` (PR 5) against a fixed schema — the same normalised metadata the real
recorder produces — so these pin the optimiser without going through the accessor.
The pushdown rules (reordering selects past reductions) land in PR 8/9; here the only
rewrite is folding consecutive same-op selects into one indexer.
"""

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

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


def test_reduction_between_selects_blocks_merge(schema):
    # the mean separates the two isels; nothing to fold (pushdown is PR 8)
    plan = [
        _node(schema, "isel", time=0),
        _node(schema, "mean", "lat"),
        _node(schema, "isel", lon=2),
    ]
    out = optimize(plan)
    assert [n.name for n in out] == ["isel", "mean", "isel"]


def test_non_select_plan_unchanged(schema):
    plan = [_node(schema, "mean", "lat"), _node(schema, "mean", "lon")]
    out = optimize(plan)
    assert [n.name for n in out] == ["mean", "mean"]


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
