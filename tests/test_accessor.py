"""Tests for the ``.plan`` accessor.

Covers recording behaviour and *equality* — ``ds.plan.<chain>.collect()`` matches
the eager ``ds.<chain>`` for the README pipelines, exercising record → optimise →
replay end to end. The golden op-list assertions that pin the optimiser itself live
in ``tests/test_optimize.py`` (it owns ``optimize``); here we only care that the
accessor records ``Op`` nodes, threads the schema, and replays to the right result.
"""

import importlib.util

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_equal

import xrexpr  # noqa: F401 -- registers the ``.plan`` accessor
from xrexpr.accessor import Explanation, LazyDatasetProxy
from xrexpr.exceptions import InvalidExpressionError
from xrexpr.optimize import optimize

#: xrexpr itself never needs dask, but replaying a ``chunk()`` call does -- without a
#: chunk manager xarray raises ``ImportError``. Only the rechunk tests below need it.
requires_dask = pytest.mark.skipif(
    importlib.util.find_spec("dask") is None, reason="dask is not installed"
)


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


def test_getitem_records_opaque_node(ds):
    from xrexpr.ir import Opaque

    node = ds.plan["temperature"]._ops[0]
    assert node.name == "__getitem__" and isinstance(node, Opaque)


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


# --- #57: selections push in front of rechunks ----------------------------------


@pytest.fixture
def chunky_ds() -> xr.Dataset:
    """A dataset big enough along ``time`` for chunk topology to be observable."""
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"temperature": (("time", "lat"), rng.random((1000, 4)))},
        coords={"time": np.arange(1000), "lat": np.arange(4)},
    )


def _replayed(proxy) -> xr.Dataset:
    """The optimised plan replayed but *not* computed, so ``.chunks`` survives."""
    return proxy._replay(optimize(proxy._ops))


@requires_dask
def test_scalar_isel_past_rechunk_matches_eager(chunky_ds):
    ds = chunky_ds
    assert_equal(
        ds.plan.chunk({"time": 100}).isel(time=0).collect(),
        ds.chunk({"time": 100}).isel(time=0).compute(),
    )


@requires_dask
def test_spent_rechunk_leaves_a_single_op(chunky_ds):
    assert "chunk" not in chunky_ds.plan.chunk({"time": 100}).isel(time=0).explain()


@requires_dask
def test_stripped_rechunk_still_replays(chunky_ds):
    # the regression the rewrite exists to avoid: leaving ``time`` in the spec after
    # the select has dropped it raises ``ValueError: chunks keys ... not found``
    ds = chunky_ds
    chain = ds.plan.chunk({"time": 100, "lat": 2}).isel(time=0)
    assert_equal(chain.collect(), ds.chunk({"time": 100, "lat": 2}).isel(time=0))
    assert _replayed(chain).chunks["lat"] == (2, 2)  # lat chunking preserved


@requires_dask
def test_slice_isel_past_rechunk_is_no_coarser(chunky_ds):
    ds = chunky_ds
    indexer = {"time": slice(50, 250)}
    chain = ds.plan.chunk({"time": 100}).isel(**indexer)
    eager = ds.chunk({"time": 100}).isel(**indexer)

    assert_equal(chain.collect(), eager.compute())
    # eager cuts across block boundaries -> ragged (50, 100, 50); pushed rechunks the
    # already-selected data -> regular (100, 100). Same values, fewer blocks.
    assert eager.chunks["time"] == (50, 100, 50)
    assert _replayed(chain).chunks["time"] == (100, 100)


@requires_dask
def test_explicit_block_tuple_chain_matches_eager(chunky_ds):
    # the barrier case: nothing moves, and the plan still replays as written
    ds = chunky_ds
    blocks = (200, 300, 500)
    assert_equal(
        ds.plan.chunk({"time": blocks}).isel(time=slice(50, 250)).collect(),
        ds.chunk({"time": blocks}).isel(time=slice(50, 250)).compute(),
    )


@requires_dask
def test_rechunk_and_reduce_pushdown_compose(chunky_ds):
    ds = chunky_ds
    assert_equal(
        ds.plan.chunk({"time": 100}).mean("lat").isel(time=0).collect(),
        ds.chunk({"time": 100}).mean("lat").isel(time=0).compute(),
    )


def test_compute_is_alias_for_collect(ds):
    # ``.compute()`` is a synonym for ``.collect()`` (xarray muscle memory), not a
    # recorded op -- it returns the materialised result, not another proxy
    chain = ds.plan.mean("lat").isel(time=0)
    assert_equal(chain.compute(), chain.collect())
