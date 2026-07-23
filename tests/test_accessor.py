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
from xrexpr.accessor import _EAGER_ATTRS, Explanation, LazyDatasetProxy
from xrexpr.exceptions import InvalidExpressionError
from xrexpr.optimize import optimize

#: xrexpr itself never needs dask, but replaying a ``chunk()`` call does -- without a
#: chunk manager xarray raises ``ImportError``. Only the rechunk tests below need it.
requires_dask = pytest.mark.skipif(
    importlib.util.find_spec("dask") is None, reason="dask is not installed"
)

#: ``.plot`` delegates to xarray's plotting, which needs matplotlib.
requires_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed"
)


# A richer dataset than the shared ``conftest.ds``: this module shadows it to add an
# auxiliary (non-dimension) ``area`` coord, exercising that projection pushdown preserves
# aux coords when it reorders the plan.
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
    return proxy._replay(optimize(proxy._ops, proxy._base_schema()))


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


# --- terminal operations -------------------------------------------------------------
# ``.plot`` / ``.to_*`` consume the plan into a figure/file/array rather than another
# link in the chain. They are callable (``ds.plot`` is an accessor with ``__call__``),
# so without special handling they would be *recorded* and silently never run. They must
# instead trigger ``collect()`` (firing the rewrite) and delegate to the realised object.


@pytest.mark.parametrize("term", sorted(_EAGER_ATTRS))
def test_every_terminal_is_routed_not_recorded(ds, term):
    # Coverage guard over the whole ``_EAGER_ATTRS`` set: catches a typo'd / nonexistent
    # name in the set, or a terminal that gets silently recorded instead of delegated.
    # Route through the chain whose collected result actually carries the terminal --
    # some are Dataset-only (``to_array``, ``to_stacked_array``), some DataArray-only
    # (``to_series``, ``to_numpy``). Picking by ``hasattr`` keeps this declarative and
    # avoids hardcoded lists that would rot if the set changes.
    ds_chain = ds.plan.mean("lat")  # collects to a Dataset
    da_chain = ds.plan["temperature"].mean("lat")  # collects to a DataArray
    chain = ds_chain if hasattr(ds_chain.collect(), term) else da_chain

    # Access only -- never call, so no args / netcdf backend / matplotlib are needed.
    attr = getattr(chain, term)
    # A recorded terminal would come back as the recording wrapper (a plain function) or
    # a proxy; a routed one is the *materialised* object's own attribute.
    assert not isinstance(attr, LazyDatasetProxy)
    assert type(attr) is type(getattr(chain.collect(), term))


@pytest.mark.xfail(
    strict=True,
    reason="``to_nowhere`` is not a registered terminal, so it is never routed -- this "
    "pins that the routing guard above discriminates, passing only for names in "
    "``_EAGER_ATTRS`` rather than vacuously.",
)
def test_unregistered_terminal_is_not_routed(ds):
    # The negative of ``test_every_terminal_is_routed_not_recorded``: a name absent from
    # ``_EAGER_ATTRS`` must NOT reach the eager-delegate path. It falls through to the
    # record / property branch and (being bogus) raises ``AttributeError`` -- so the
    # routing assertions below never hold. Strict xfail, so if ``to_nowhere`` is ever
    # added to the set (or otherwise made to route) this turns green and fails the suite,
    # forcing a deliberate update.
    term = "to_nowhere"
    assert term not in _EAGER_ATTRS
    chain = ds.plan["temperature"].mean("lat")
    attr = getattr(chain, term)
    assert not isinstance(attr, LazyDatasetProxy)
    assert type(attr) is type(getattr(chain.collect(), term))


def test_terminal_to_dataframe_triggers_collect_and_delegates(ds):
    # regression: ``.to_dataframe()`` used to return a proxy (recorded, never run). It
    # must now materialise -- proving the rewrite ran and delegation reached xarray.
    chain = ds.plan["temperature"].isel(time=0)
    result = chain.to_dataframe()

    assert not isinstance(result, LazyDatasetProxy)
    eager = ds["temperature"].isel(time=0).to_dataframe()
    assert result.equals(eager)


def test_terminal_to_dict_matches_eager(ds):
    chain = ds.plan.mean("lat")
    assert chain.to_dict() == ds.mean("lat").to_dict()


@requires_matplotlib
def test_plot_triggers_collect_and_delegates(ds):
    import matplotlib

    matplotlib.use("Agg")

    chain = ds.plan["temperature"].isel(time=0)

    # ``.plot`` is xarray's plot accessor off the *materialised* DataArray, not a
    # recording wrapper -- so calling it draws instead of returning a proxy.
    assert not isinstance(chain.plot, LazyDatasetProxy)
    artist = chain.plot()
    assert not isinstance(artist, LazyDatasetProxy)
    assert type(artist) is type(ds["temperature"].isel(time=0).plot())

    # delegating the whole ``plot`` attribute (not just calling it) also reaches the
    # accessor's own methods, e.g. ``plot.line`` on a 1-D selection.
    line = ds.plan["temperature"].isel(time=0, lat=0).plot.line()
    assert not isinstance(line, LazyDatasetProxy)
