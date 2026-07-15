"""Tests for the ``.plan`` accessor (PR 1).

Two flavours:
- *equality* tests — ``ds.plan.<chain>.compute()`` matches the eager ``ds.<chain>``;
- *op-list* tests — assert the exact optimised op list, which pins down the
  optimiser's behaviour (including branches that are deliberately left for later
  PRs to fix).

Together they exercise every branch of ``accessor.py``.
"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

import xrexpr  # noqa: F401 -- registers the ``.plan`` accessor
from xrexpr.accessor import LazyDatasetProxy


@pytest.fixture
def ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"temperature": (("time", "lat", "lon"), rng.random((4, 3, 5)))},
        coords={
            "time": np.arange(4),
            "lat": np.arange(3),
            "lon": np.arange(5),
        },
    )




def test_plan_accessor_returns_proxy(ds):
    assert isinstance(ds.plan, LazyDatasetProxy)
    assert ds.plan._ops == []
    assert ds.plan._base_ds is ds


def test_repr_shows_recorded_ops(ds):
    proxy = ds.plan.mean("lat").isel(time=0)
    text = repr(proxy)
    assert "LazyDatasetProxy" in text
    assert "mean" in text and "isel" in text


def test_getattr_underscore_raises(ds):
    with pytest.raises(AttributeError):
        ds.plan._not_a_real_attr


def test_getattr_property_materialises(ds):
    # ``.sizes`` is not callable -> forces compute and reads off the result
    assert dict(ds.plan.sizes) == dict(ds.sizes)


def test_getitem_records_and_computes(ds):
    assert_equal(ds.plan["temperature"].compute(), ds["temperature"])




def test_readme_pipeline_positional_equal(ds):
    got = ds.plan.mean("lat").mean("lon").isel(time=0).compute()
    assert_equal(got, ds.mean("lat").mean("lon").isel(time=0))


def test_readme_pipeline_kwargs_equal(ds):
    got = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).compute()
    assert_equal(got, ds.mean(dim="lat").mean(dim="lon").isel(time=0))


def test_reduce_tuple_dims_equal(ds):
    got = ds.plan.mean(dim=("lat", "lon")).isel(time=0).compute()
    assert_equal(got, ds.mean(dim=("lat", "lon")).isel(time=0))


def test_sum_pushdown_equal(ds):
    got = ds.plan.sum("lat").isel(time=0).compute()
    assert_equal(got, ds.sum("lat").isel(time=0))


def test_sel_merge_equal(ds):
    got = ds.plan.sel(lat=1).sel(lon=2).compute()
    assert_equal(got, ds.sel(lat=1).sel(lon=2))




def _optimized(proxy: LazyDatasetProxy):
    return proxy._optimize_ops(proxy._ops)


def test_isel_merge_kwargs(ds):
    assert _optimized(ds.plan.isel(time=0).isel(lat=1)) == [
        ("isel", ({"time": 0, "lat": 1},), {}),
    ]


def test_isel_merge_positional_dict(ds):
    assert _optimized(ds.plan.isel({"time": 0}).isel({"lat": 1})) == [
        ("isel", ({"time": 0, "lat": 1},), {}),
    ]


def test_sel_merge_kwargs(ds):
    assert _optimized(ds.plan.sel(lat=1).sel(lon=2)) == [
        ("sel", ({"lat": 1, "lon": 2},), {}),
    ]


def test_sel_merge_positional_dict(ds):
    assert _optimized(ds.plan.sel({"lat": 1}).sel({"lon": 2})) == [
        ("sel", ({"lat": 1, "lon": 2},), {}),
    ]


def test_pushdown_indexer_positional_dict(ds):
    # isel given a positional dict, disjoint from the reduced dim -> swap
    assert _optimized(ds.plan.mean("lat").isel({"time": 0})) == [
        ("isel", ({"time": 0},), {}),
        ("mean", ("lat",), {}),
    ]


def test_no_swap_when_dims_overlap(ds):
    # isel touches the reduced dim -> NOT reorderable, left in place
    assert _optimized(ds.plan.mean("time").isel(time=0)) == [
        ("mean", ("time",), {}),
        ("isel", ({"time": 0},), {}),
    ]


def test_reduction_not_followed_by_isel_unchanged(ds):
    # neither reduction is followed by an isel -> no pushdown attempted
    assert _optimized(ds.plan.mean("lat").mean("lon")) == [
        ("mean", ("lat",), {}),
        ("mean", ("lon",), {}),
    ]


def test_empty_dim_reduction_swaps_known_bug(ds):
    """``mean()`` (no dim) is treated as reducing *nothing*, so the isel is
    wrongly pushed in front. This is the demo optimiser's known bug; PR 9's
    schema-aware validity check fixes it. Pinned here so the fix is visible."""
    assert _optimized(ds.plan.mean().isel(time=0)) == [
        ("isel", ({"time": 0},), {}),
        ("mean", (), {}),
    ]
