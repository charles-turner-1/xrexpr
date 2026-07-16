"""Tests for the ``.plan`` accessor.

Covers recording behaviour and *equality* — ``ds.plan.<chain>.compute()`` matches
the eager ``ds.<chain>`` for the README pipelines, exercising record → optimise →
replay end to end. The golden op-list assertions that pin the optimiser itself live
in ``tests/test_optimize.py`` (it owns ``optimize``); here we only care that the
accessor records ``OpNode``s, threads the schema, and replays to the right result.
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


# --- PR 6: recording now builds OpNodes and threads the schema ------------------


def test_record_builds_opnodes(ds):
    from xrexpr.ir import OpNode

    ops = ds.plan.mean("lat").isel(time=0)._ops
    assert all(isinstance(n, OpNode) for n in ops)
    mean_node, isel_node = ops
    assert mean_node.kind == "reduce" and mean_node.consumes == frozenset({"lat"})
    assert isel_node.kind == "select" and isel_node.consumes == frozenset({"time"})


def test_schema_threads_as_ops_are_recorded(ds):
    # each recorded op evolves the proxy's logical schema, no materialisation
    assert dict(ds.plan._schema.dims) == {"time": 4, "lat": 3, "lon": 5}
    assert dict(ds.plan.mean("lat")._schema.dims) == {"time": 4, "lon": 5}
    assert dict(ds.plan.mean("lat").isel(time=0)._schema.dims) == {"lon": 5}


def test_getitem_records_opaque_node(ds):
    node = ds.plan["temperature"]._ops[0]
    assert node.name == "__getitem__" and node.kind == "opaque"




def test_readme_pipeline_positional_equal(ds):
    got = ds.plan.mean("lat").mean("lon").isel(time=0).compute()
    assert_equal(got, ds.mean("lat").mean("lon").isel(time=0))


def test_readme_pipeline_kwargs_equal(ds):
    got = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).compute()
    assert_equal(got, ds.mean(dim="lat").mean(dim="lon").isel(time=0))


def test_reduce_tuple_dims_equal(ds):
    got = ds.plan.mean(dim=("lat", "lon")).isel(time=0).compute()
    assert_equal(got, ds.mean(dim=("lat", "lon")).isel(time=0))


def test_reduce_then_select_equal(ds):
    got = ds.plan.sum("lat").isel(time=0).compute()
    assert_equal(got, ds.sum("lat").isel(time=0))


def test_isel_merge_equal(ds):
    # two isels fold into one indexer; the replayed result is unchanged
    got = ds.plan.isel(time=0).isel(lat=1).compute()
    assert_equal(got, ds.isel(time=0).isel(lat=1))


def test_sel_merge_equal(ds):
    got = ds.plan.sel(lat=1).sel(lon=2).compute()
    assert_equal(got, ds.sel(lat=1).sel(lon=2))
