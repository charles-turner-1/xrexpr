"""Tests for the ``.plan`` accessor.

Covers recording behaviour and *equality* — ``ds.plan.<chain>.collect()`` matches
the eager ``ds.<chain>`` for the README pipelines, exercising record → optimise →
replay end to end. The golden op-list assertions that pin the optimiser itself live
in ``tests/test_optimize.py`` (it owns ``optimize``); here we only care that the
accessor records ``Op`` nodes, threads the schema, and replays to the right result.
"""

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

import xrexpr  # noqa: F401 -- registers the ``.plan`` accessor
from xrexpr.accessor import Explanation, LazyDatasetProxy
from xrexpr.exceptions import InvalidExpressionError


@pytest.fixture
def ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), rng.random((4, 3, 5))),
            # no ``time`` dim: projecting it can't hop past a ``time`` reduction
            "elevation": (("lat", "lon"), rng.random((3, 5))),
        },
        coords={
            "time": np.arange(4),
            "lat": np.arange(3),
            "lon": np.arange(5),
            # an auxiliary (non-dim) coord: reordering must not silently drop or keep
            # it differently from the eager chain
            "area": (("lat", "lon"), rng.random((3, 5))),
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
    assert_equal(ds.plan["temperature"].collect(), ds["temperature"])


# --- PR 6: recording now builds Op variants and threads the schema --------------


def test_record_builds_op_variants(ds):
    from xrexpr.ir import Op, Reduce, Select

    ops = ds.plan.mean("lat").isel(time=0)._ops
    assert all(isinstance(n, Op) for n in ops)
    mean_node, isel_node = ops
    assert isinstance(mean_node, Reduce) and mean_node.consumes == frozenset({"lat"})
    assert isinstance(isel_node, Select) and isel_node.consumes == frozenset({"time"})


def test_schema_threads_as_ops_are_recorded(ds):
    # each recorded op evolves the proxy's logical schema, no materialisation
    assert dict(ds.plan._schema.dims) == {"time": 4, "lat": 3, "lon": 5}
    assert dict(ds.plan.mean("lat")._schema.dims) == {"time": 4, "lon": 5}
    assert dict(ds.plan.mean("lat").isel(time=0)._schema.dims) == {"lon": 5}


def test_getitem_records_projection_node(ds):
    from xrexpr.ir import Project

    node = ds.plan["temperature"]._ops[0]
    assert node.name == "__getitem__" and isinstance(node, Project)
    assert node.variables == ("temperature",)


def test_readme_pipeline_positional_equal(ds):
    got = ds.plan.mean("lat").mean("lon").isel(time=0).collect()
    assert_equal(got, ds.mean("lat").mean("lon").isel(time=0))


def test_readme_pipeline_kwargs_equal(ds):
    got = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).collect()
    assert_equal(got, ds.mean(dim="lat").mean(dim="lon").isel(time=0))


def test_reduce_tuple_dims_equal(ds):
    got = ds.plan.mean(dim=("lat", "lon")).isel(time=0).collect()
    assert_equal(got, ds.mean(dim=("lat", "lon")).isel(time=0))


def test_reduce_then_select_equal(ds):
    got = ds.plan.sum("lat").isel(time=0).collect()
    assert_equal(got, ds.sum("lat").isel(time=0))


def test_isel_merge_equal(ds):
    # two isels fold into one indexer; the replayed result is unchanged
    got = ds.plan.isel(time=0).isel(lat=1).collect()
    assert_equal(got, ds.isel(time=0).isel(lat=1))


def test_sel_merge_equal(ds):
    got = ds.plan.sel(lat=1).sel(lon=2).collect()
    assert_equal(got, ds.sel(lat=1).sel(lon=2))


def test_select_on_reduced_dim_raises(ds):
    # mean removes lon; isel(lon=0) then references a dim that is gone -> invalid
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean(dim="lon").isel(lon=0).collect()


def test_bare_mean_then_select_raises(ds):
    # mean() reduces every dim; the empty-dim reorder bug is now an error, not a wrong swap
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean().isel(time=0).collect()


def test_cumsum_then_select_computes(ds):
    # cumsum is a scan (order matters): left in place, not reordered and not raised
    got = ds.plan.cumsum("time").isel(time=2).collect()
    assert_equal(got, ds.cumsum("time").isel(time=2))


def test_projection_pushdown_equal(ds):
    # the rewrite reduces only ``temperature``; the result must still match the eager
    # chain -- coords included, which is what ``assert_equal`` checks
    got = ds.plan.mean("time")[["temperature"]].collect()
    assert_equal(got, ds.mean("time")[["temperature"]])


def test_single_variable_projection_pushdown_equal(ds):
    # ``ds["temperature"]`` yields a DataArray; pushing it down must not change that
    got = ds.plan.mean("time")["temperature"].collect()
    assert isinstance(got, xr.DataArray)
    assert_equal(got, ds.mean("time")["temperature"])


def test_projection_pushdown_past_select_equal(ds):
    got = ds.plan.isel(time=0).mean("lat")[["temperature"]].collect()
    assert_equal(got, ds.isel(time=0).mean("lat")[["temperature"]])


def test_blocked_projection_still_replays(ds):
    # ``elevation`` has no ``time``, so the projection can't move -- but the plan is
    # perfectly valid eagerly and must still replay to the same result
    got = ds.plan.mean("time")[["elevation"]].collect()
    assert_equal(got, ds.mean("time")[["elevation"]])


def test_explain_shows_projection_pushdown(ds):
    text = ds.plan.mean("time")[["temperature"]].explain()
    assert text.index("temperature") < text.index("mean")


def test_explain_shows_optimised_plan(ds):
    text = ds.plan.mean("lat").isel(time=0).explain()
    assert text.startswith("plan (2 ops):")
    # the optimisation is visible: the isel has been pushed in front of the mean
    assert text.index("isel") < text.index("mean")


def test_explain_formats_getitem(ds):
    assert "['temperature']" in ds.plan["temperature"].explain()


def test_explain_empty_plan(ds):
    assert ds.plan.explain() == "plan (0 ops)"


def test_explain_raises_on_invalid_plan(ds):
    # explain optimises too, so an invalid plan raises the same error collect() would
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean(dim="lon").isel(lon=0).explain()


def test_explain_repr_is_unescaped_text(ds):
    # the REPL echoes repr(); it must be the formatted multi-line plan, not the
    # escaped one-liner ``'plan (2 ops):\n  ...'`` a plain str would produce.
    text = ds.plan.mean("lat").isel(time=0).explain()
    assert isinstance(text, Explanation)
    assert isinstance(text, str)
    assert repr(text) == str(text)
    assert "\\n" not in repr(text)


def test_compute_is_alias_for_collect(ds):
    # ``.compute()`` is a synonym for ``.collect()`` (xarray muscle memory), not a
    # recorded op -- it returns the materialised result, not another proxy
    chain = ds.plan.mean("lat").isel(time=0)
    assert_equal(chain.compute(), chain.collect())
