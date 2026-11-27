"""Tests for the ``.plan`` accessor.

These check recording, and *equality*: ``ds.plan.<chain>.collect()`` matches the eager
``ds.<chain>``, running record, optimise and replay end to end. Assertions about the
exact optimised node list belong in ``tests/test_optimize.py``. Here we only check that
the accessor records the right nodes and that the result is right.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from frozendict import frozendict
from xarray.testing import assert_equal

import xrexpr  # noqa: F401 -- registers the ``.plan`` accessor
from xrexpr.accessor import _EAGER_ATTRS, Explanation, LazyProxy
from xrexpr.exceptions import InvalidExpressionError
from xrexpr.ir import (
    ALL_DIMS,
    ContextOpen,
    Drop,
    Elementwise,
    GroupedReduce,
    Opaque,
    Project,
    Reduce,
    Rename,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.lower import emit
from xrexpr.optimize import AUTOCHUNK_EMPTY_DIMS_SAFE, _schemas, optimize
from xrexpr.schema import SchemaState

#: xrexpr itself never needs dask, but replaying a ``chunk()`` call does -- without a
#: chunk manager xarray raises ``ImportError``. Only the rechunk tests below need it.
requires_dask = pytest.mark.skipif(
    importlib.util.find_spec("dask") is None, reason="dask is not installed"
)

#: ``.plot`` delegates to xarray's plotting, which needs matplotlib.
requires_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None, reason="matplotlib is not installed"
)

#: ``rolling_exp`` is the one opener xarray refuses to run without numbagg (an explicit
#: ``ImportError``, not a slow fallback), so only its *replay* needs this. The lowering
#: half of those tests passes either way. Nothing else in the suite needs numbagg
#: **installed**. The ``test-nonumbagg`` environment exists because numbagg changes
#: reduction *results*, not because it gates features.
requires_numbagg = pytest.mark.skipif(
    importlib.util.find_spec("numbagg") is None, reason="numbagg is not installed"
)


# This overrides the shared ``conftest.ds`` to add an auxiliary (non-dimension) ``area``
# coord, so tests can check that projection pushdown keeps aux coords when it reorders
# the plan.
@pytest.fixture
def ds() -> xr.Dataset:
    """A richer dataset than the shared ``conftest.ds``, with an auxiliary ``area`` coord.

    Returns
    -------
    xarray.Dataset
        ``temperature(time, lat, lon)`` and ``elevation(lat, lon)`` as in the shared
        fixture, plus a non-dimension ``area(lat, lon)`` coord, so tests can check that
        pushdown preserves auxiliary coords when it reorders the plan.
    """
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
    """The ``.plan`` accessor returns a ``LazyProxy`` wrapping the base dataset with an empty op list."""
    assert isinstance(ds.plan, LazyProxy)
    assert ds.plan._ops == []
    assert ds.plan._base is ds


def test_repr_shows_recorded_ops(ds):
    """``repr()`` of a proxy names the class and lists the recorded op names."""
    proxy = ds.plan.mean("lat").isel(time=0)
    text = repr(proxy)
    assert "LazyProxy" in text
    assert "mean" in text and "isel" in text


def test_getattr_underscore_raises(ds):
    """An underscore-prefixed attribute access raises, never recording a bogus op."""
    with pytest.raises(AttributeError):
        ds.plan._not_a_real_attr


def test_getattr_property_materialises(ds):
    """A non-callable attribute like ``.sizes`` forces compute and reads the result off the materialised dataset."""
    assert dict(ds.plan.sizes) == dict(ds.sizes)


def test_getitem_records_and_computes(ds):
    """``ds.plan["temperature"]`` records a projection and collects to the same DataArray as eager."""
    assert_equal(ds.plan["temperature"].collect(), ds["temperature"])


# --- recording builds Op variants ----------------------------------------------


def test_record_builds_op_variants(ds):
    """Recording a chain builds the right ``Op`` subclass for each call: ``Reduce`` for ``mean``, ``Select`` for ``isel``."""
    from xrexpr.ir import Op, Reduce, Select

    ops = ds.plan.mean("lat").isel(time=0)._ops
    assert all(isinstance(n, Op) for n in ops)
    mean_node, isel_node = ops
    assert isinstance(mean_node, Reduce) and mean_node.consumes == frozenset({"lat"})
    assert isinstance(isel_node, Select) and isel_node.consumes == frozenset({"time"})


def test_recording_holds_no_schema_and_the_optimiser_folds_it(ds):
    """A recorded proxy carries no schema of its own; only the optimiser folds the schema forward.

    Notes
    -----
    Recording only appends: the proxy holds the base dataset and the plan, nothing else.
    The optimiser does the one schema fold, starting from the front of the plan, which
    is where it's exact.
    """
    chain = ds.plan.mean("lat").isel(time=0)
    assert not hasattr(chain, "_schema")

    base = chain._base_schema()
    entering = _schemas(chain._ops, base)
    assert [dict(s.sizes) for s in entering] == [
        {"time": 4, "lat": 3, "lon": 5},  # what mean("lat") sees
        {"time": 4, "lon": 5},  # what isel(time=0) sees
    ]


def test_bare_reduce_records_all_dims_symbolically(ds):
    """A bare ``mean()`` records ``ALL_DIMS`` symbolically, with no dim set frozen in at record time.

    Notes
    -----
    The expansion happens against the optimiser's fold, so it can't bake in names an
    unmodelled op has since changed.
    """
    node = ds.plan.mean()._ops[0]
    assert node.consumes is ALL_DIMS
    assert node.args == () and node.kwargs == frozendict()  # replays as bare mean()


#: Every way to spell "all dims" with xarray's ``...``, for a reduce and a scan, as
#: ``(label, chain)`` pairs applied to both the plan and the eager dataset.
ELLIPSIS_ALL_DIMS_CHAINS = [
    ("mean(...)", lambda d: d.mean(...)),
    ("mean(dim=...)", lambda d: d.mean(dim=...)),
    ("mean(..., keepdims=True)", lambda d: d.mean(..., keepdims=True)),
    ("reduce(np.mean, ...)", lambda d: d.reduce(np.mean, ...)),
    ("cumsum(...)", lambda d: d.cumsum(...)),
    ("cumsum(dim=...)", lambda d: d.cumsum(dim=...)),
    ("isel(time=0).mean(...)", lambda d: d.isel(time=0).mean(...)),
    ("cumsum(...).isel(time=0)", lambda d: d.cumsum(...).isel(time=0)),
    ("project then mean(...)", lambda d: d[["elevation"]].mean(dim=...)),
]


@pytest.mark.parametrize(
    "chain",
    [c for _, c in ELLIPSIS_ALL_DIMS_CHAINS],
    ids=[i for i, _ in ELLIPSIS_ALL_DIMS_CHAINS],
)
def test_ellipsis_dim_reduces_over_every_dim_and_matches_eager(ds, chain):
    """A bare ``...`` dim spec means every dim, as in xarray, and replays equal.

    Notes
    -----
    Read as a name, ``...`` would be a dim *named* ``Ellipsis``, and the plan would
    replay a reduce over nothing: ``ds.plan.mean(...)`` would come back with every dim
    intact.
    """
    assert_equal(chain(ds.plan).collect(), chain(ds))


@pytest.mark.parametrize("spelling", ["positional", "kwarg"])
@pytest.mark.parametrize("method", ["mean", "cumsum"])
def test_ellipsis_dim_records_all_dims_symbolically(ds, method, spelling):
    """``...`` records the same ``ALL_DIMS`` sentinel as a bare call, and replays bare."""
    proxy = getattr(ds.plan, method)
    node = (proxy(...) if spelling == "positional" else proxy(dim=...))._ops[0]
    dims = node.consumes if method == "mean" else node.dims
    assert dims is ALL_DIMS
    assert emit([node])[0].args == () and emit([node])[0].kwargs == frozendict()


def test_ellipsis_reduce_then_select_on_any_dim_raises(ds):
    """``mean(...)`` then a select raises, exactly as a bare ``mean()`` does: no dim survives."""
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean(...).isel(time=0).collect()


#: An ``...`` *inside* a dim collection. xarray exempts it from the Dataset existence
#: check, where it then matches no dim, so these reduce only the named dims (or none).
ELLIPSIS_IN_LIST_CHAINS = [
    ("mean([lat, ...])", lambda d: d.mean(dim=["lat", ...])),
    ("mean([...])", lambda d: d.mean(dim=[...])),
    ("mean((...,))", lambda d: d.mean(dim=(...,))),
    ("reduce(np.mean, [lat, ...])", lambda d: d.reduce(np.mean, ["lat", ...])),
    ("cumsum([lat, ...])", lambda d: d.cumsum(dim=["lat", ...])),
    ("mean([lat, ...]).isel(time=0)", lambda d: d.mean(dim=["lat", ...]).isel(time=0)),
    ("isel(lon=0).mean([lat, ...])", lambda d: d.isel(lon=0).mean(dim=["lat", ...])),
    ("mean([...]).isel(lat=0)", lambda d: d.mean(dim=[...]).isel(lat=0)),
]


@pytest.mark.parametrize(
    "chain",
    [c for _, c in ELLIPSIS_IN_LIST_CHAINS],
    ids=[i for i, _ in ELLIPSIS_IN_LIST_CHAINS],
)
def test_ellipsis_inside_a_dim_list_matches_eager_dataset(ds, chain):
    """An ``...`` inside a dim list is a literal element, inert on a Dataset as in xarray."""
    assert_equal(chain(ds.plan).collect(), chain(ds))


def test_ellipsis_inside_a_dim_list_raises_on_a_dataarray_as_eager_does(ds):
    """A DataArray rejects ``...`` inside a dim list eagerly, and the plan must not launder that.

    Notes
    -----
    This is why :func:`~xrexpr.schema._dim_spec` keeps the element rather than stripping
    it: a stripped header would replay as ``mean(dim=["lat"])`` and succeed.
    """
    da = ds["temperature"]
    with pytest.raises(ValueError, match="Ellipsis"):
        da.mean(dim=["lat", ...])
    with pytest.raises(ValueError, match="Ellipsis"):
        da.plan.mean(dim=["lat", ...]).collect()


def test_getitem_records_projection_node(ds):
    """``ds.plan["temperature"]`` records a ``Project`` node naming the variable."""
    from xrexpr.ir import Project

    node = ds.plan["temperature"]._ops[0]
    assert node.name == "__getitem__" and isinstance(node, Project)
    assert node.variables == ("temperature",)


def test_readme_pipeline_positional_equal(ds):
    """The README's positional-args pipeline (``mean`` chained twice, then ``isel``) matches the eager chain."""
    got = ds.plan.mean("lat").mean("lon").isel(time=0).collect()
    assert_equal(got, ds.mean("lat").mean("lon").isel(time=0))


def test_readme_pipeline_kwargs_equal(ds):
    """The same README pipeline, written with ``dim=`` keyword args, also matches the eager chain."""
    got = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).collect()
    assert_equal(got, ds.mean(dim="lat").mean(dim="lon").isel(time=0))


def test_reduce_tuple_dims_equal(ds):
    """A ``mean`` reducing a tuple of dims matches the eager chain."""
    got = ds.plan.mean(dim=("lat", "lon")).isel(time=0).collect()
    assert_equal(got, ds.mean(dim=("lat", "lon")).isel(time=0))


def test_reduce_then_select_equal(ds):
    """A ``sum`` reduce followed by an ``isel`` matches the eager chain."""
    got = ds.plan.sum("lat").isel(time=0).collect()
    assert_equal(got, ds.sum("lat").isel(time=0))


def test_reduce_method_then_select_matches_eager(ds):
    """``.reduce(func, dim)`` records an honest ``Reduce``, so a disjoint select hops in front of it.

    Notes
    -----
    The dim spec is read from the *second* positional, so the node carries a true
    ``consumes`` and ``.reduce`` chains optimise like any other reduction.
    """
    chain = ds.plan.reduce(np.mean, "lat").isel(time=0)
    assert [type(n) for n in chain._optimized()] == [Select, Reduce]
    assert_equal(chain.collect(), ds.reduce(np.mean, "lat").isel(time=0))


def test_reduce_method_on_a_consumed_dim_still_raises(ds):
    """A select on a dim ``.reduce`` consumed is rejected, exactly as for a named ``mean``."""
    with pytest.raises(InvalidExpressionError):
        ds.plan.reduce(np.mean, "lon").isel(lon=0).collect()


def test_bare_reduce_method_then_select_raises(ds):
    """A bare ``.reduce(func)`` consumes every dim, so a following select is invalid — as for ``mean()``."""
    with pytest.raises(InvalidExpressionError):
        ds.plan.reduce(np.mean).isel(time=0).collect()


def test_keepdims_reduce_then_select_on_kept_dim_matches_eager(ds):
    """A ``keepdims=True`` reduce keeps its dim at size 1, so a select on that dim replays, not raises.

    Notes
    -----
    ``keepdims=True`` keeps ``time`` at size 1. Treated as a plain reduce of ``time``,
    ``pushdown_selects`` would read the select as indexing a reduced-away dim and raise
    on a chain that works eagerly. The ``immovable`` dim effect leaves the select in
    place instead, so it replays equal.
    """
    got = ds.plan.mean("time", keepdims=True).isel(time=0).collect()
    assert_equal(got, ds.mean("time", keepdims=True).isel(time=0))


def test_keepdims_reduce_then_disjoint_select_matches_eager(ds):
    """A disjoint select behind a ``keepdims=True`` reduce reorders under the hood yet replays equal."""
    got = ds.plan.mean("time", keepdims=True).isel(lat=0).collect()
    assert_equal(got, ds.mean("time", keepdims=True).isel(lat=0))


def test_isel_merge_equal(ds):
    """Two folded ``isel`` calls replay to the same result as the two eager calls."""
    got = ds.plan.isel(time=0).isel(lat=1).collect()
    assert_equal(got, ds.isel(time=0).isel(lat=1))


def test_sel_merge_equal(ds):
    """Two folded ``sel`` calls replay to the same result as the two eager calls, as ``isel`` does."""
    got = ds.plan.sel(lat=1).sel(lon=2).collect()
    assert_equal(got, ds.sel(lat=1).sel(lon=2))


def test_select_on_reduced_dim_raises(ds):
    """Selecting a dim a preceding ``mean`` already removed raises ``InvalidExpressionError`` at collect time."""
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean(dim="lon").isel(lon=0).collect()


def test_bare_mean_then_select_raises(ds):
    """A bare ``mean()`` followed by a select raises, end to end, rather than being swapped."""
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean().isel(time=0).collect()


def test_cumsum_then_select_computes(ds):
    """A ``cumsum`` followed by a select on its own dim computes to the eager result, left in place since a scan's order matters."""
    got = ds.plan.cumsum("time").isel(time=2).collect()
    assert_equal(got, ds.cumsum("time").isel(time=2))


def test_cumsum_then_disjoint_select_matches_eager(ds):
    """A ``cumsum`` then a select on a disjoint dim reorders, and still collects to the eager result.

    Notes
    -----
    The optimiser hops ``isel(lat=0)`` in front of ``cumsum("time")``; the two commute
    because the scan runs independently at each ``lat``, and ``assert_equal`` is what
    proves the reorder changed no value.
    """
    got = ds.plan.cumsum("time").isel(lat=0).collect()
    assert_equal(got, ds.cumsum("time").isel(lat=0))


def test_diff_then_disjoint_select_matches_eager(ds):
    """A ``diff`` then a disjoint select reorders and collects to the eager result, the shrunk dim included."""
    got = ds.plan.diff("time").isel(lat=0).collect()
    assert_equal(got, ds.diff("time").isel(lat=0))


def test_projection_past_scan_matches_eager(ds):
    """A projection hopped in front of a scan collects to the same result as eager, coords included."""
    got = ds.plan.cumsum("time")[["temperature"]].collect()
    assert_equal(got, ds.cumsum("time")[["temperature"]])


def test_projection_pushdown_equal(ds):
    """A projection hopped in front of a reduce collects to the same result as eager, coords included.

    Notes
    -----
    The rewrite reduces only ``temperature``; the result must still match the eager
    chain -- coords included, which is what ``assert_equal`` checks.
    """
    got = ds.plan.mean("time")[["temperature"]].collect()
    assert_equal(got, ds.mean("time")[["temperature"]])


def test_single_variable_projection_pushdown_equal(ds):
    """A single-variable projection still yields a DataArray after pushdown, matching the eager result.

    Notes
    -----
    ``ds["temperature"]`` yields a DataArray; pushing it down must not change that.
    """
    got = ds.plan.mean("time")["temperature"].collect()
    assert isinstance(got, xr.DataArray)
    assert_equal(got, ds.mean("time")["temperature"])


def test_projection_pushdown_past_select_equal(ds):
    """A projection hopped past a select collects to the same result as the eager chain."""
    got = ds.plan.isel(time=0).mean("lat")[["temperature"]].collect()
    assert_equal(got, ds.isel(time=0).mean("lat")[["temperature"]])


def test_fillna_in_a_chain_records_elementwise_and_replays_equal(ds):
    """A chain with ``fillna(0)`` records an ``Elementwise`` and collects to the eager result.

    Notes
    -----
    Records the variant and exercises record -> optimise -> replay end to end, with the
    select free to hop the elementwise op.
    """
    chain = ds.plan.mean("lat").fillna(0.0).isel(time=0)
    assert any(isinstance(n, Elementwise) for n in chain._ops)
    assert_equal(chain.collect(), ds.mean("lat").fillna(0.0).isel(time=0))


def test_fillna_with_dataarray_records_opaque_and_replays_equal(ds):
    """``fillna(<DataArray>)`` demotes to ``Opaque`` (unsafe arg) and replays verbatim to eager."""
    other = xr.zeros_like(ds["temperature"])
    chain = ds.plan.fillna(other)
    assert any(isinstance(n, Opaque) and n.name == "fillna" for n in chain._ops)
    assert_equal(chain.collect(), ds.fillna(other))


def test_drop_vars_records_a_drop_node(ds):
    """``drop_vars`` records a ``Drop`` node, not an ``Opaque`` barrier, with the names in order."""
    chain = ds.plan.drop_vars(["elevation", "area"])
    (node,) = chain._ops
    assert isinstance(node, Drop)
    assert node.variables == ("elevation", "area")
    assert not any(isinstance(n, Opaque) for n in chain._ops)


def test_drop_data_var_in_a_chain_replays_equal(ds):
    """A chain dropping a data variable collects to the eager result, auxiliary coords included."""
    got = ds.plan.drop_vars("elevation").mean("time")[["temperature"]].collect()
    assert_equal(got, ds.drop_vars("elevation").mean("time")[["temperature"]])


def test_drop_auxiliary_coord_replays_equal(ds):
    """Dropping a non-dimension coordinate (``area``) collects to the eager result."""
    got = ds.plan.drop_vars("area").mean("lat").collect()
    assert_equal(got, ds.drop_vars("area").mean("lat"))


def test_drop_dimension_coord_keeps_the_dim_and_replays_equal(ds):
    """Dropping a *dimension* coordinate keeps the dim (a data var still spans it) and replays to eager.

    Notes
    -----
    ``drop_vars("time")`` removes the ``time`` coordinate but leaves ``time`` as a
    dimension without a coordinate, because ``temperature`` still spans it. That's the
    schema case that isn't a plain ``_aggregated`` drop, and the property generator
    doesn't produce it (a dim without a coordinate breaks a later generated ``sel``), so
    it's tested here.
    """
    got = ds.plan.drop_vars("time").isel(time=0).collect()
    assert_equal(got, ds.drop_vars("time").isel(time=0))


def test_a_redundant_drop_is_eliminated_and_replays_equal(ds):
    """A ``drop_vars`` of a data var the projection excludes is removed entirely, and the plan still equals eager.

    Notes
    -----
    ``drop_vars("elevation")[["temperature"]]`` collapses to ``[["temperature"]]`` --
    ``elevation`` is never computed -- via
    :func:`~xrexpr.optimize.push_projection_past_drop`. The value must still match the
    eager chain.
    """
    chain = ds.plan.drop_vars("elevation")[["temperature"]]
    optimised = optimize(list(chain._ops), chain._base_schema())
    assert [type(n).__name__ for n in optimised] == ["Project"]
    assert_equal(chain.collect(), ds.drop_vars("elevation")[["temperature"]])


def test_an_absent_drop_is_eliminated_when_the_projection_excludes_it(ds):
    """An absent dropped name is ignored when it cannot affect the requested projection."""
    chain = ds.plan.drop_vars("nope")[["temperature"]]
    optimised = optimize(list(chain._ops), chain._base_schema())
    assert [type(n).__name__ for n in optimised] == ["Project"]
    with pytest.raises(ValueError):
        ds.drop_vars("nope")[["temperature"]]
    assert_equal(chain.collect(), ds[["temperature"]])


def test_a_surviving_coord_drop_hops_the_projection_and_replays_equal(ds):
    """A ``drop_vars`` of a retained coordinate hops behind the projection, trimmed, and still equals eager."""
    chain = ds.plan.drop_vars(["elevation", "area"])[["temperature"]]
    optimised = optimize(list(chain._ops), chain._base_schema())
    assert [type(n).__name__ for n in optimised] == ["Project", "Drop"]
    assert optimised[1].variables == ("area",)
    assert_equal(chain.collect(), ds.drop_vars(["elevation", "area"])[["temperature"]])


def test_select_commutes_with_a_drop_and_replays_equal(ds):
    """A select hopped in front of a ``drop_vars`` (pinned structurally in ``test_optimize``) still equals eager."""
    got = ds.plan.drop_vars("elevation").isel(time=0).collect()
    assert_equal(got, ds.drop_vars("elevation").isel(time=0))


def test_rename_records_a_rename_node(ds):
    """``rename`` records a ``Rename`` node, not an ``Opaque`` barrier, with the mapping gathered."""
    chain = ds.plan.rename({"time": "t"})
    (node,) = chain._ops
    assert isinstance(node, Rename)
    assert dict(node.mapping) == {"time": "t"}
    assert not any(isinstance(n, Opaque) for n in chain._ops)


def test_rename_kwargs_form_records_a_rename_node(ds):
    """The keyword spelling ``rename(time="t")`` records the same ``Rename`` mapping as the dict form."""
    (node,) = ds.plan.rename(time="t")._ops
    assert isinstance(node, Rename) and dict(node.mapping) == {"time": "t"}


def test_rename_dimension_coord_replays_equal(ds):
    """Renaming a dimension coordinate (``time`` -> ``t``) collects to the eager result, coords included."""
    got = ds.plan.rename({"time": "t"}).mean("lat")[["temperature"]].collect()
    assert_equal(got, ds.rename({"time": "t"}).mean("lat")[["temperature"]])


def test_rename_data_var_replays_equal(ds):
    """Renaming a data variable collects to the eager result."""
    got = ds.plan.rename({"temperature": "temp"}).mean("time").collect()
    assert_equal(got, ds.rename({"temperature": "temp"}).mean("time"))


def test_rename_auxiliary_coord_replays_equal(ds):
    """Renaming a non-dimension coordinate (``area`` -> ``cell``) collects to the eager result."""
    got = ds.plan.rename({"area": "cell"}).mean("time").collect()
    assert_equal(got, ds.rename({"area": "cell"}).mean("time"))


def test_select_on_renamed_dim_after_rename_replays_equal(ds):
    """A ``sel`` on the renamed dim is left behind the rename (pinned in ``test_optimize``) and still equals eager."""
    got = ds.plan.rename({"time": "t"}).isel(t=0).collect()
    assert_equal(got, ds.rename({"time": "t"}).isel(t=0))


def test_orthogonal_dataarray_select_records_a_select_and_replays_equal(ds):
    """An orthogonal ``isel(<DataArray>)`` records a positional ``Select`` (not ``Opaque``) and replays to eager.

    Notes
    -----
    The point is that the select isn't a barrier, so ``mean("lat").isel(...)`` optimises
    (the select moves in front of the reduce; the exact plan is checked in
    ``test_optimize``) and still equals the eager chain.
    """
    idx = xr.DataArray([0, 2], dims="time")
    chain = ds.plan.mean("lat").isel(time=idx)
    assert any(isinstance(n, Select) for n in chain._ops)
    assert not any(isinstance(n, Opaque) for n in chain._ops)
    assert_equal(chain.collect(), ds.mean("lat").isel(time=idx))


def test_vectorized_dataarray_select_records_opaque_and_replays_equal(ds):
    """A vectorized ``isel(<DataArray>)`` mints a dim, so it demotes to ``Opaque`` and replays verbatim to eager."""
    idx = xr.DataArray([0, 2], dims="points")
    chain = ds.plan.isel(time=idx).sum("lat")
    assert any(isinstance(n, Opaque) and n.name == "isel" for n in chain._ops)
    assert_equal(chain.collect(), ds.isel(time=idx).sum("lat"))


def test_invalid_dataarray_select_chain_raises_instead_of_laundering(ds):
    """An invalid ``isel(<DataArray>).mean().isel(time=0)`` chain is refused, not silently reordered.

    Notes
    -----
    Misfiled as a dim-dropping ``Scalar``, the advanced ``isel`` would make the bare
    ``mean()`` understate its ``consumes``, and the trailing ``isel(time=0)`` would be
    swapped in front — returning a value where the eager chain errors. The orthogonal
    indexer resizes ``time`` rather than dropping it, so ``time`` survives the fold and
    ``ALL_DIMS`` refuses the trailing select.
    """
    idx = xr.DataArray([0, 2], dims="time")
    with pytest.raises(InvalidExpressionError):
        ds.plan.isel(time=idx).mean().isel(time=0).collect()


def test_scalar_select_crosses_fillna_matches_eager(ds):
    """A scalar ``isel`` hops in front of a ``fillna`` and a ``mean``, and still equals eager.

    Notes
    -----
    Checks that the reorder happens (``isel`` comes first in the optimised plan), and
    that moving past the elementwise op and the reduce on a different dim changed no
    value, including the dropped ``time`` dim.
    """
    chain = ds.plan.mean("lat").fillna(0.0).isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["isel", "mean", "fillna"]
    assert_equal(chain.collect(), ds.mean("lat").fillna(0.0).isel(time=0))


def test_dim_keeping_select_crosses_fillna_matches_eager(ds):
    """A dim-keeping ``isel`` (a list of positions) also crosses a ``fillna`` and equals eager."""
    chain = ds.plan.fillna(0.0).isel(lon=[0, 2])
    assert [c.name for c in emit(chain._optimized())] == ["isel", "fillna"]
    assert_equal(chain.collect(), ds.fillna(0.0).isel(lon=[0, 2]))


def test_projection_crosses_astype_matches_eager(ds):
    """A projection hops past an ``astype`` (and a ``mean``) to the front, matching eager, coords included."""
    chain = ds.plan.mean("time").astype("float32")[["temperature"]]
    assert [c.name for c in emit(chain._optimized())] == [
        "__getitem__",
        "mean",
        "astype",
    ]
    assert_equal(chain.collect(), ds.mean("time").astype("float32")[["temperature"]])


def test_a_reduce_is_never_reordered_across_a_fillna_matches_eager():
    """A ``fillna`` before a reduce keeps its order, with a NaN present so a wrong reorder would show.

    Notes
    -----
    ``fillna`` and a reduce don't commute when there's a NaN: fill-then-average differs
    from average-then-fill. So this is the case a reduce-crossing bug would break. The
    shared fixtures have no NaN, which would make ``fillna`` do nothing and the equality
    check pointless, so this builds a dataset with one. A select is added so the
    optimiser actually rewrites something: it moves to the front past both ops, and the
    ``fillna``/``mean`` order has to survive.
    """
    t = np.arange(24.0).reshape(4, 3, 2)
    t[0, 0, 0] = np.nan
    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), t)},
        coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(2)},
    )
    chain = ds.plan.fillna(0.0).mean("lat").isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["isel", "fillna", "mean"]
    assert_equal(chain.collect(), ds.fillna(0.0).mean("lat").isel(time=0))


def test_blocked_projection_still_replays(ds):
    """A projection that can't move still replays to the eager result, since the plan is valid as written.

    Notes
    -----
    ``elevation`` has no ``time``, so the projection can't move -- but the plan is
    perfectly valid eagerly.
    """
    got = ds.plan.mean("time")[["elevation"]].collect()
    assert_equal(got, ds.mean("time")[["elevation"]])


def test_merged_projections_equal(ds):
    """Two projections merged into one collect to the eager result, auxiliary coords included.

    Notes
    -----
    ``assert_identical`` rather than ``assert_equal``. The merge relies on a
    projection's *coordinate* pruning composing correctly, so names and attrs are
    exactly what's under test. The ``area(lat, lon)`` coord this module's fixture adds
    is the one a sloppier rule would drop.
    """
    got = ds.plan[["temperature", "elevation"]][["temperature"]].collect()
    assert len(ds.plan[["temperature", "elevation"]][["temperature"]]._optimized()) == 1
    xr.testing.assert_identical(got, ds[["temperature", "elevation"]][["temperature"]])


def test_merged_projections_keep_the_dataarray_result(ds):
    """A list projection merged with a following bare name still yields the eager ``DataArray``."""
    got = ds.plan[["temperature", "elevation"]]["temperature"].collect()
    assert isinstance(got, xr.DataArray)
    xr.testing.assert_identical(got, ds[["temperature", "elevation"]]["temperature"])


def test_a_projection_feeding_a_coord_is_dropped_and_replays_equal(ds):
    """A projection whose only consumer names a surviving coordinate is dropped, replaying equal.

    Notes
    -----
    ``ds[["temperature"]]["lat"]`` builds ``temperature`` only to throw it away and
    return the ``lat`` coordinate. ``lat`` survives ``p1`` (``temperature`` spans it),
    so the best plan reads the coord from the base and never touches the data variable:
    ``eliminate_projection_before_coord`` drops ``p1``. Unlike the plain merge, this
    rewrite uses the schema to prove ``lat`` passes through ``p1`` unchanged.
    """
    chain = ds.plan[["temperature"]]["lat"]
    assert [type(n) for n in chain._optimized()] == [Project]
    xr.testing.assert_identical(chain.collect(), ds[["temperature"]]["lat"])


def test_a_projection_feeding_an_aux_coord_is_dropped_and_replays_equal(ds):
    """Dropping a projection extends to a non-dimension coordinate the survivor still spans.

    Notes
    -----
    ``area`` is an auxiliary ``(lat, lon)`` coordinate; ``temperature`` spans both dims,
    so ``area`` survives ``p1`` and the pair collapses to reading ``area`` from the
    base.
    """
    chain = ds.plan[["temperature"]]["area"]
    assert [type(n) for n in chain._optimized()] == [Project]
    xr.testing.assert_identical(chain.collect(), ds[["temperature"]]["area"])


def test_a_projection_feeding_a_coord_across_a_select_is_left(ds):
    """A select between the two projections separates them, so the pair is left and replays equal.

    Notes
    -----
    With a select in between, ``p1`` feeds the select, not the projection, and the two
    forms really are different (the chain reads ``lat`` from ``p1``'s *selected*
    result). The rule only looks at adjacent pairs, so it declines, and the plan still
    equals eager.
    """
    chain = ds.plan[["temperature"]].isel(time=0)["lat"]
    assert [type(n) for n in chain._optimized()].count(Project) == 2
    xr.testing.assert_identical(
        chain.collect(), ds[["temperature"]].isel(time=0)["lat"]
    )


def test_merging_projections_skips_an_error_from_a_discarded_name(ds):
    """Merging drops a first projection whose extra name doesn't exist (§8: discarded work may skip errors).

    Notes
    -----
    Eagerly this raises ``KeyError`` while building an intermediate the chain throws
    away one step later. The merged plan only asks for what the chain actually wanted,
    and succeeds. It's the same allowance ``pushdown_projections`` relies on
    (``test_projection_pushdown_skips_an_error_from_a_discarded_variable``): no value
    changes, and the skipped error came from work whose result is discarded.
    """
    with pytest.raises(KeyError):
        ds[["temperature", "nope"]][["temperature"]]

    got = ds.plan[["temperature", "nope"]][["temperature"]].collect()
    xr.testing.assert_identical(got, ds[["temperature"]])


def test_explain_shows_projection_pushdown(ds):
    """``explain()`` shows a pushed-down projection listed before the reduce it hopped past."""
    text = ds.plan.mean("time")[["temperature"]].explain()
    assert text.index("temperature") < text.index("mean")


def test_explain_shows_optimised_plan(ds):
    """``explain()`` shows the optimised plan, not the recorded one: the pushed-down ``isel`` appears before ``mean``."""
    text = ds.plan.mean("lat").isel(time=0).explain()
    assert text.startswith("plan (2 ops):")
    assert text.index("isel") < text.index("mean")


def test_explain_formats_getitem(ds):
    """``explain()`` formats a projection node using Python subscript syntax, e.g. ``['temperature']``."""
    assert "['temperature']" in ds.plan["temperature"].explain()


def test_explain_lists_one_line_per_lowered_node(ds):
    """``explain()`` lists exactly one line per lowered node, not per recorded call.

    Notes
    -----
    The listing is the *lowered* plan, so its lines correspond to nodes.
    """
    chain = ds.plan.groupby("time").mean().isel(lat=0)
    lowered = chain._optimized()
    lines = chain.explain().splitlines()[1:]
    assert len(lines) == len(lowered) == 2
    assert [type(n).__name__ for n in lowered] == [line.split()[1] for line in lines]


def test_explain_still_answers_what_will_run(ds):
    """``explain()`` lists every emitted call, in order.

    Notes
    -----
    ``explain()`` shows lowered nodes, but it must still show every call that will run,
    in order. The expected calls come from ``emit`` rather than being written out, so
    the test can't pass by matching a stale expectation.
    """
    chain = ds.plan.groupby("time.month").mean().isel(lat=0)
    text = chain.explain()
    positions = [text.index(c.name) for c in emit(chain._optimized())]
    assert positions == sorted(positions)


def test_explain_names_the_opaque_nodes_that_stopped_the_optimiser(ds):
    """``explain()`` names the ``Opaque`` nodes and says they're unmodelled, answering why nothing moved.

    Notes
    -----
    An unfusable context becomes two opaque nodes. ``explain()`` should make that
    visible, so a reader can see *why* nothing moved without knowing which methods are
    modelled.
    """
    text = ds.plan.groupby("lat").first().mean("time").explain()
    assert text.count("Opaque") == 2
    assert "not modelled" in text


def test_explain_empty_plan(ds):
    """``explain()`` on an empty plan reports zero ops."""
    assert ds.plan.explain() == "plan (0 ops)"


def test_explain_raises_on_invalid_plan(ds):
    """``explain()`` raises the same error ``collect()`` would, since it optimises the plan too."""
    with pytest.raises(InvalidExpressionError):
        ds.plan.mean(dim="lon").isel(lon=0).explain()


def test_explain_repr_is_unescaped_text(ds):
    """``repr()`` of an ``Explanation`` is the unescaped multi-line plan, not a quoted one-liner.

    Notes
    -----
    The REPL echoes ``repr()``; it must be the formatted multi-line plan, not the
    escaped one-line form a plain ``str`` would produce.
    """
    text = ds.plan.mean("lat").isel(time=0).explain()
    assert isinstance(text, Explanation)
    assert isinstance(text, str)
    assert repr(text) == str(text)
    assert "\\n" not in repr(text)


# --- selections push in front of rechunks ---------------------------------------


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
    return proxy._replay(emit(proxy._optimized()))


@requires_dask
def test_scalar_isel_past_rechunk_matches_eager(chunky_ds):
    """A scalar ``isel`` past a rechunk matches the eager (and computed) result."""
    ds = chunky_ds
    assert_equal(
        ds.plan.chunk({"time": 100}).isel(time=0).collect(),
        ds.chunk({"time": 100}).isel(time=0).compute(),
    )


@requires_dask
def test_spent_rechunk_leaves_a_single_op(chunky_ds):
    """A rechunk spent by a scalar ``isel`` leaves no ``chunk`` op in ``explain()``."""
    assert "chunk" not in chunky_ds.plan.chunk({"time": 100}).isel(time=0).explain()


@requires_dask
def test_stripped_rechunk_still_replays(chunky_ds):
    """A rechunk stripped of its consumed dim still replays, keeping the chunking of the surviving dim.

    Notes
    -----
    Leaving ``time`` in the spec after the select has dropped it would raise
    ``ValueError: chunks keys ... not found``.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": 100, "lat": 2}).isel(time=0)
    assert_equal(chain.collect(), ds.chunk({"time": 100, "lat": 2}).isel(time=0))
    assert _replayed(chain).chunks["lat"] == (2, 2)  # lat chunking preserved


@requires_dask
def test_slice_isel_past_rechunk_is_no_coarser(chunky_ds):
    """A slice ``isel`` pushed past a rechunk matches eager values while landing on fewer, more regular blocks.

    Notes
    -----
    Eager cuts across block boundaries -> ragged ``(50, 100, 50)``; pushed rechunks the
    already-selected data -> regular ``(100, 100)``. Same values, fewer blocks.
    """
    ds = chunky_ds
    indexer = {"time": slice(50, 250)}
    chain = ds.plan.chunk({"time": 100}).isel(**indexer)
    eager = ds.chunk({"time": 100}).isel(**indexer)

    assert_equal(chain.collect(), eager.compute())
    assert eager.chunks["time"] == (50, 100, 50)
    assert _replayed(chain).chunks["time"] == (100, 100)


@requires_dask
def test_explicit_block_tuple_chain_matches_eager(chunky_ds):
    """An explicit block-tuple rechunk is a barrier: nothing moves, and the plan still replays as written to match eager."""
    ds = chunky_ds
    blocks = (200, 300, 500)
    assert_equal(
        ds.plan.chunk({"time": blocks}).isel(time=slice(50, 250)).collect(),
        ds.chunk({"time": blocks}).isel(time=slice(50, 250)).compute(),
    )


@requires_dask
def test_unparseable_byte_target_fails_exactly_as_eager_does(chunky_ds):
    """A nonsense byte target raises dask's own error at ``collect``, as it does eagerly.

    Notes
    -----
    ``test_chunks.py`` checks that every string classifies as a ``ByteSize`` without
    being parsed. This is the other half: because it isn't parsed, the error is the same
    ``ValueError`` xarray raises eagerly, in dask's words.

    The error is deferred, not lost: recording succeeds where the eager chain would
    already have raised, and the error still has to come out at ``collect``. (A byte
    target is sized from the array, so it's a barrier and the select stays behind it.)
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": "banana"}).isel(time=slice(50, 250))  # records fine

    with pytest.raises(ValueError, match="byte unit"):
        ds.chunk({"time": "banana"}).isel(time=slice(50, 250)).compute()
    with pytest.raises(ValueError, match="byte unit"):
        chain.collect()


@requires_dask
def test_an_emptying_select_crosses_an_auto_rechunk_when_xarray_supports_it(chunky_ds):
    """An emptying select crosses an ``"auto"`` rechunk only with xarray's zero-dim fix.

    Notes
    -----
    ``isel(lat=[])`` keeps ``lat`` at length 0. Before xarray 2026.09.0, hoisting it
    hands the auto rechunk a zero-size array and dask raises ``ZeroDivisionError``.
    xarray's pydata/xarray#11486 replaces that auto request with the zero-byte full
    shape, so the same hop is safe on newer versions. A non-emptying select remains a
    barrier; see :func:`test_a_harmless_select_also_stays_behind_an_auto_rechunk`.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": "auto"}).isel(lat=[])

    assert_equal(chain.collect(), ds.chunk({"time": "auto"}).isel(lat=[]).compute())
    text = chain.explain()
    if AUTOCHUNK_EMPTY_DIMS_SAFE:
        assert text.index("isel") < text.index("chunk")
    else:
        assert text.index("chunk") < text.index("isel")


@requires_dask
def test_an_emptying_select_crosses_a_uniform_auto_rechunk_when_xarray_supports_it(
    chunky_ds,
):
    """The same version-gated exception applies to the uniform ``chunk("auto")`` form.

    Notes
    -----
    The uniform form is stored in ``args`` rather than ``chunks``, but xarray hands it
    to the same chunk manager. At a 1 kB target, older xarray raises after the hop;
    2026.09.0 and later use the same safe full-shape fallback as the per-dim form.
    """
    import dask

    ds = chunky_ds
    with dask.config.set({"array.chunk-size": "1kB"}):
        if AUTOCHUNK_EMPTY_DIMS_SAFE:
            assert ds.isel(lat=[]).chunk("auto").chunks["lat"] == (0,)
        else:
            with pytest.raises(ZeroDivisionError):
                ds.isel(lat=[]).chunk("auto")

        chain = ds.plan.chunk("auto").isel(lat=[])
        assert_equal(chain.collect(), ds.chunk("auto").isel(lat=[]).compute())
        text = chain.explain()
        if AUTOCHUNK_EMPTY_DIMS_SAFE:
            assert text.index("isel") < text.index("chunk")
        else:
            assert text.index("chunk") < text.index("isel")


@requires_dask
def test_a_harmless_select_also_stays_behind_an_auto_rechunk(chunky_ds):
    """Even a select that empties nothing stays behind an ``"auto"`` rechunk, and still matches eager.

    Notes
    -----
    This is the cost of deciding from the *spec* rather than from the pair.
    ``isel(lat=slice(0, 2))`` can't empty anything, but it's refused anyway, because
    whether ``"auto"`` survives the hop can't be decided from the plan: it depends on
    the configured ``array.chunk-size`` and the number of dims (see
    :class:`~xrexpr.chunks.Auto`).

    A size-*independent* spec in the same position still crosses; see
    :func:`test_a_select_crosses_a_sized_rechunk`.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": "auto"}).isel(lat=slice(0, 2))

    assert_equal(
        chain.collect(), ds.chunk({"time": "auto"}).isel(lat=slice(0, 2)).compute()
    )
    text = chain.explain()
    assert text.index("chunk") < text.index("isel")


@requires_dask
def test_a_select_crosses_a_sized_rechunk(chunky_ds):
    """A select crosses every spec dask doesn't size from the array, whether it empties a dim or not.

    Notes
    -----
    The other side of the line, end to end. A fixed size means the same on any array, so
    the hop happens and the result still matches eager. Without this test, the refusals
    above would also be consistent with a rule that had just stopped working.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": 100}).isel(lat=[])

    assert_equal(chain.collect(), ds.chunk({"time": 100}).isel(lat=[]).compute())
    text = chain.explain()
    assert text.index("isel") < text.index("chunk")


@requires_dask
def test_rechunk_and_reduce_pushdown_compose(chunky_ds):
    """Rechunk pushdown composes with reduce pushdown, end to end, matching the eager result."""
    ds = chunky_ds
    assert_equal(
        ds.plan.chunk({"time": 100}).mean("lat").isel(time=0).collect(),
        ds.chunk({"time": 100}).mean("lat").isel(time=0).compute(),
    )


@requires_dask
def test_adjacent_rechunks_merge_to_one_op_matching_eager(chunky_ds):
    """Two ``chunk`` calls over different dims merge into one ``chunk`` that matches eager.

    Notes
    -----
    ``chunk({"time": 100}).chunk({"lat": 50})`` is one request written as two calls. The
    merged node applies both at once and produces the same blocks dask does when
    applying the second call on top of the first.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": 100}).chunk({"lat": 2})
    eager = ds.chunk({"time": 100}).chunk({"lat": 2})

    assert_equal(chain.collect(), eager.compute())
    assert chain.explain().count("Rechunk") == 1  # two calls, one op
    replayed = _replayed(chain)
    assert replayed.chunks["time"] == eager.chunks["time"]
    assert replayed.chunks["lat"] == eager.chunks["lat"]


@requires_dask
def test_adjacent_rechunks_on_one_dim_are_later_wins_against_eager(chunky_ds):
    """A repeated dim across two ``chunk`` calls lands on the later spec's topology, matching eager."""
    ds = chunky_ds
    chain = ds.plan.chunk({"time": 100}).chunk({"time": 250})
    eager = ds.chunk({"time": 100}).chunk({"time": 250})

    assert_equal(chain.collect(), eager.compute())
    assert chain.explain().count("Rechunk") == 1
    assert _replayed(chain).chunks["time"] == eager.chunks["time"]  # later 250 wins


def test_compute_is_alias_for_collect(ds):
    """``.compute()`` is a synonym for ``.collect()``, not a recorded op: it returns the materialised result, not a proxy.

    Notes
    -----
    ``.compute()`` exists for xarray muscle memory.
    """
    chain = ds.plan.mean("lat").isel(time=0)
    assert_equal(chain.compute(), chain.collect())


# --- terminal operations -------------------------------------------------------------
# ``.plot`` / ``.to_*`` turn the plan into a figure, file, array, etc., rather than
# extending the chain. They're callable (``ds.plot`` is an accessor with ``__call__``),
# so without special handling they'd be *recorded* and never run. Instead they must call
# ``collect()`` (so the rewrite happens) and delegate to the computed object.


@pytest.mark.parametrize("term", sorted(_EAGER_ATTRS))
def test_every_terminal_is_routed_not_recorded(ds, term):
    """Every name in ``_EAGER_ATTRS`` is routed to the computed object, never recorded as a plan op.

    Notes
    -----
    Covers the whole ``_EAGER_ATTRS`` set. It catches a misspelled or nonexistent name
    in the set, and a terminal that gets recorded instead of delegated. Each name is
    reached through a chain whose result actually has it: some are Dataset-only
    (``to_array``, ``to_stacked_array``), some DataArray-only (``to_series``,
    ``to_numpy``). Choosing by ``hasattr`` avoids hand-written lists that would go stale
    if the set changed.
    """
    ds_chain = ds.plan.mean("lat")  # collects to a Dataset
    da_chain = ds.plan["temperature"].mean("lat")  # collects to a DataArray
    chain = ds_chain if hasattr(ds_chain.collect(), term) else da_chain

    # Access only -- never call, so no args / netcdf backend / matplotlib are needed.
    attr = getattr(chain, term)
    # A recorded terminal would come back as the recording wrapper (a plain function) or
    # a proxy; a routed one is the *materialised* object's own attribute.
    assert not isinstance(attr, LazyProxy)
    assert type(attr) is type(getattr(chain.collect(), term))


@pytest.mark.xfail(
    strict=True,
    reason="``to_nowhere`` is not a registered terminal, so it is never routed -- this "
    "pins that the routing guard above discriminates, passing only for names in "
    "``_EAGER_ATTRS`` rather than vacuously.",
)
def test_unregistered_terminal_is_not_routed(ds):
    """A name not in ``_EAGER_ATTRS`` isn't routed to the eager-delegate path (strict xfail).

    Notes
    -----
    The opposite of :func:`test_every_terminal_is_routed_not_recorded`. A name not in
    ``_EAGER_ATTRS`` must *not* reach the eager-delegate path. It goes to the
    record/property branch and, being bogus, raises ``AttributeError``, so the routing
    assertions never hold. The xfail is strict, so if ``to_nowhere`` were ever added to
    the set this would start passing and fail the suite, forcing a deliberate update.
    """
    term = "to_nowhere"
    assert term not in _EAGER_ATTRS
    chain = ds.plan["temperature"].mean("lat")
    attr = getattr(chain, term)
    assert not isinstance(attr, LazyProxy)
    assert type(attr) is type(getattr(chain.collect(), term))


# --- grouped / windowed contexts -----------------------------------------------------
# ``groupby``/``resample``/``rolling``/... return a builder, not a Dataset, so the call
# after them means something different from the same-named Dataset method. Lowering
# fuses a recognised opener/closer pair into one node. A pair it can't fuse becomes
# ``Opaque``, which no rule can fire on or move across: correct, but not optimised.


@pytest.fixture
def dated_ds() -> xr.Dataset:
    """A dataset with a real datetime ``time`` index, for the tests that need ``resample``.

    Returns
    -------
    xarray.Dataset
        ``temperature(time, lat, lon)`` with a datetime ``time`` coord (``ds`` has an
        integer one, and ``resample`` needs a datetime one). ``lon`` and the ``area``
        aux coord are for the fused-reduce pushdown tests: a third dim gives a select
        somewhere independent to go, and an aux coord spanning the selected dims checks
        that the reorder keeps it.
    """
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"temperature": (("time", "lat", "lon"), rng.random((8, 3, 4)))},
        coords={
            "time": pd.date_range("2000-01-01", periods=8, freq="D"),
            "lat": np.arange(3),
            "lon": np.arange(4),
            "area": (("lat", "lon"), rng.random((3, 4))),
        },
    )


def test_groupby_then_reduce_then_select_matches_eager(ds):
    """A ``groupby().mean()`` followed by a select on a surviving dim matches the eager result.

    Notes
    -----
    Valid eagerly: ``groupby('time').mean()`` reduces within groups, so ``lat`` is still
    there for the later ``isel(lat=0)``. Treated as a plain Dataset reduce (bare, so
    over every dim), the grouped ``mean()`` would drop ``lat`` from the schema and
    ``collect()`` would wrongly raise ``InvalidExpressionError``.
    """
    chain = ds.plan.groupby("time").mean().isel(lat=0)
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(lat=0))


def test_select_after_grouped_reduce_matches_eager(ds):
    """A select on the grouped dim itself matches eager, replaying in the order written.

    Notes
    -----
    The other way a Dataset-level reading of the closer would go wrong: a select on a
    dim outside the bogus ``consumes`` would be moved behind the reduce, and replay
    would then call ``isel`` on the ``DatasetGroupBy``. It replays in the order written.
    """
    chain = ds.plan.groupby("time").mean().isel(time=0)
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(time=0))


def test_context_chain_records_an_opener_then_lowers_to_one_node(ds):
    """A ``groupby().mean()`` context records an opener plus a provisional reduce, then lowers to one ``GroupedReduce`` node."""
    chain = ds.plan.groupby("time").mean().isel(time=0)

    # recorded: the opener is typed, the closer provisionally as an ordinary reduce
    assert [type(op) for op in chain._ops] == [ContextOpen, Reduce, Select]

    # lowered: the pair becomes one node, and no ContextOpen survives
    lowered = chain._optimized()
    assert [type(n) for n in lowered] == [GroupedReduce, Select]
    assert not any(isinstance(n, ContextOpen) for n in lowered)

    # ... which is what ``explain()`` shows: one op, still shown as the two calls it replays
    # as, in the order written. ``time -> time`` isn't a typo, and it's the reason the
    # annotation exists: ``groupby("time")`` *re-creates* the dim, so the ``time`` the
    # trailing ``isel`` indexes holds group labels, not the original values.
    assert chain.explain() == Explanation(
        "plan (2 ops):\n"
        "  1. GroupedReduce  groupby('time').mean()  [time -> time]\n"
        "  2. Select  isel({'time': 0})"
    )


@pytest.fixture
def regioned_ds() -> xr.Dataset:
    """A dataset with a non-dim coordinate grouper.

    Returns
    -------
    xarray.Dataset
        ``temperature(time, lat)`` with a ``region`` coord defined on ``lat``, so
        ``groupby("region")`` groups along ``lat`` and mints ``region`` -- everyday
        xarray. ``lat``'s three values give two groups.
    """
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"temperature": (("time", "lat"), rng.random((4, 3)))},
        coords={
            "time": np.arange(4),
            "lat": np.arange(3),
            "region": ("lat", ["a", "b", "b"]),
        },
    )


def test_a_non_dim_coordinate_grouper_does_not_fuse(regioned_ds):
    """A ``groupby`` on a non-dim coordinate doesn't fuse; it falls back to the opaque pair and matches eager.

    Notes
    -----
    The grouper's *name* isn't the dim it consumes. ``groupby("region")`` consumes
    ``lat`` and creates ``region``, but the call doesn't say so. Fusing based on the
    call alone would give ``group_dim="region"``: a schema claiming ``lat`` survived a
    grouping that removed it, which would mislead both the fold and ``dim_effect``.
    Refusing sends the pair to the opaque fallback: correct, just not optimised
    (``02-lowering.md`` §5.5).
    """
    chain = regioned_ds.plan.groupby("region").mean()

    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), regioned_ds.groupby("region").mean())


def test_explain_prints_no_group_arrow_for_a_coordinate_grouper(regioned_ds):
    """``explain()`` prints no ``[group_dim -> new_dim]`` arrow for an unfused coordinate grouper.

    Notes
    -----
    ``explain()`` shows a fused node's dims as ``[group_dim -> new_dim]``. If this pair
    were fused, it would show ``[region -> region]``, which is wrong about the data.
    Unfused, there's no arrow to get wrong.
    """
    text = str(regioned_ds.plan.groupby("region").mean().explain())

    assert "->" not in text
    assert "GroupedReduce" not in text


def test_ops_after_an_unfusable_context_are_modelled_again(ds):
    """A reduce after an unfusable ``groupby`` context is modelled as an ordinary ``Reduce``, not left opaque.

    Notes
    -----
    ``first`` closes the context and returns a Dataset, so the ``mean`` after it is an
    ordinary Dataset reduce, not opaque from the groupby onward.
    """
    chain = ds.plan.groupby("lat").first().mean("time")

    lowered = chain._optimized()
    assert [type(n) for n in lowered] == [Opaque, Opaque, Reduce]
    assert_equal(chain.collect(), ds.groupby("lat").first().mean("time"))


def test_select_on_a_grouped_nodes_own_dim_stays_put(ds):
    """A select on a grouped reduce's own (re-created) dim stays put, replaying to the eager result.

    Notes
    -----
    ``groupby("time")`` re-creates the dim it grouped over, so ``isel(time=0)`` indexes
    the *new* ``time``. That's valid eagerly, and the select can't move. Tested end to
    end because it's the case select pushdown must decline: moving it in front would
    index the original ``time`` instead of the group labels.
    """
    chain = ds.plan.groupby("time").mean().isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["groupby", "mean", "isel"]
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(time=0))


def test_climatology_select_is_pushed_in_front_of_the_grouping(dated_ds):
    """A select moves in front of a climatology grouping, so it runs over one latitude and still matches eager.

    Notes
    -----
    This is the main case fused grouped nodes exist for: the grouping runs over one
    latitude, instead of over all of them before discarding the rest. Both halves are
    checked: that the rewrite *happens* (so this can't pass trivially), and that the
    result still equals eager.
    """
    chain = dated_ds.plan.groupby("time.month").mean().isel(lat=0)

    assert [c.name for c in emit(chain._optimized())] == ["isel", "groupby", "mean"]
    assert_equal(chain.collect(), dated_ds.groupby("time.month").mean().isel(lat=0))


@pytest.mark.parametrize(
    "chain",
    [
        lambda o: o.groupby("time.month").mean().isel(lat=0),
        lambda o: o.groupby("time.month").mean().isel(lat=[0, 2]),
        lambda o: o.groupby("time.month").mean().sel(lon=slice(1, 3)),
        lambda o: o.groupby("time.month").mean().isel(lat=0, lon=1),
        # immovable, and valid: the select indexes the minted dim
        lambda o: o.groupby("time.month").mean().isel(month=0),
        lambda o: o.resample(time="2D").mean().isel(lat=0),
        lambda o: o.rolling(time=3).mean().isel(lat=0),
        lambda o: o.rolling(time=3, center=True, min_periods=1).mean().isel(lat=0),
        lambda o: o.coarsen(time=5, boundary="trim").mean().isel(lat=0),
        # immovable, and valid: rolling keeps its dim, so indexing it is fine
        lambda o: o.rolling(time=3).mean().isel(time=0),
        # the select empties a disjoint dim, which the grouping must survive
        lambda o: o.groupby("time.month").mean().isel(lat=[]),
    ],
)
def test_select_across_a_fused_reduce_matches_eager(dated_ds, chain):
    """A select around a fused reduce matches eager whether or not the rewrite moves it.

    Notes
    -----
    Moved or not, the answer must not change. That's what makes "move when the dims
    don't overlap, leave it otherwise" safe.
    """
    assert_equal(chain(dated_ds.plan).collect(), chain(dated_ds))


def test_projection_is_pushed_in_front_of_the_grouping(dated_ds):
    """A projection hops in front of a grouping, aggregating one variable instead of two and discarding one."""
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    chain = ds.plan.groupby("time.month").mean()[["temperature"]]

    assert [c.name for c in emit(chain._optimized())] == [
        "__getitem__",
        "groupby",
        "mean",
    ]
    assert_equal(chain.collect(), ds.groupby("time.month").mean()[["temperature"]])


@pytest.mark.parametrize(
    "chain",
    [
        lambda o: o.groupby("time.month").mean()[["temperature"]],
        lambda o: o.resample(time="2D").mean()[["temperature"]],
        lambda o: o.rolling(time=3).mean()[["temperature"]],
        lambda o: o.coarsen(time=2, boundary="trim").mean()[["temperature"]],
        # immovable: ``elevation`` has no ``time``, so projecting first would drop the
        # ``time`` coord and the fused op would raise
        lambda o: o.groupby("time.month").mean()[["elevation"]],
        lambda o: o.rolling(time=3).mean()[["elevation"]],
        # both variables kept: ``time`` survives the projection via ``temperature``
        lambda o: o.groupby("time.month").mean()[["temperature", "elevation"]],
        # projection and select both crossing
        lambda o: o.groupby("time.month").mean().isel(lat=0)[["temperature"]],
    ],
)
def test_projection_across_a_fused_reduce_matches_eager(dated_ds, chain):
    """A projection around a fused reduce matches eager whether or not the rewrite moves it, blocked and both-variable cases included."""
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    assert_equal(chain(ds.plan).collect(), chain(ds))


def test_projection_past_a_windowed_reduce_is_left_when_the_dim_would_go(dated_ds):
    """A projection that would drop the windowed dim is left alone, matching eager rather than raising.

    Notes
    -----
    Tested end to end because the failure would be an error, not a wrong value.
    Projecting first drops the ``time`` coord (no kept variable uses it), and
    ``rolling`` then raises ``Window dimensions ('time',) not found``. The rule must not
    cause that.
    """
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    chain = ds.plan.rolling(time=3).mean()[["elevation"]]

    assert [c.name for c in emit(chain._optimized())] == [
        "rolling",
        "mean",
        "__getitem__",
    ]
    assert_equal(chain.collect(), ds.rolling(time=3).mean()[["elevation"]])


def test_projection_pushdown_skips_an_error_from_a_discarded_variable(dated_ds):
    """Pushing a projection down skips work the plan discards, including errors.

    ``station`` is a *string* variable, so a Dataset ``std("time")`` raises while
    reducing it. Eagerly, the chain fails computing a standard deviation it then throws
    away. The rewrite never computes it. The optimised answer is the better one; see
    ``planning/roadmap/07-small-wins.md`` §8, which makes this an intended property
    rather than a difference to fix.

    It's safe because values can't change: the rule only fires once the kept variables
    are known to have the dims the crossed op names, so they reduce the same way either
    way. The only difference is a discarded variable's error.
    """
    # ------------------------------------------------------------------------------
    # WHY A *STRING* VARIABLE, AND NOT THE OBVIOUS FLOAT ONE. Read before "simplifying".
    #
    # This test is about *xrexpr*: ``pushdown_projections`` skips work on a discarded
    # variable, errors included. To test that we need eager to fail, and any failure
    # will do. The failure is a prop, not the subject.
    #
    # The tempting alternative gets the failure from a float variable without ``time``:
    # a Dataset reduce gives that variable an empty axis set, and ``std``/``var`` blow
    # up on it. That's a trap, because **the blow-up is a numbagg bug, not xarray
    # behaviour**. numbagg reads ``axis=()`` as ``axis=None`` and returns a scalar
    # (07-small-wins.md §8 has the details). Relying on it would make this test assert
    # that a third-party bug *exists*. It would fail under ``use_numbagg=False``, and
    # again whenever numbagg fixes it, both times reporting DID NOT RAISE as though
    # xrexpr had broken.
    #
    # numpy refusing to ``std`` a ``<U4`` array is xarray/numpy's own behaviour and
    # never reaches numbagg (numbagg only handles ``dtype.kind in "uif"``). So the prop
    # works with numbagg present, absent, or fixed. Switching back to a float variable
    # would tie this test to numbagg's releases.
    #
    # NOTE: this doesn't change the numbagg bug itself, which isn't ours to fix and is
    # still
    # present in this environment. See EMPTY_AXIS_UNSAFE_REDUCES in test_properties.
    # ------------------------------------------------------------------------------
    ds = dated_ds.assign(station=(("lat", "lon"), np.full((3, 4), "buoy")))

    with pytest.raises(TypeError, match="resolved dtypes"):
        ds.std(dim=["time"])[["temperature"]]

    planned = ds.plan.std(dim=["time"])[["temperature"]].collect()
    # the reference is the same chain with the discarded variable simply absent
    assert_equal(planned, ds[["temperature"]].std(dim=["time"]))


def test_projection_pushdown_introduces_no_error_of_its_own(dated_ds):
    """Projection pushdown adds no error of its own: when both orders succeed, they must agree exactly.

    Notes
    -----
    The other half of §8: skipping a discarded variable's error is allowed, adding one
    isn't. ``mean`` skips a variable that lacks the dim rather than raising, so both
    orders succeed and must give the same result.
    """
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    assert_equal(
        ds.plan.mean(dim=["time"])[["temperature"]].collect(),
        ds.mean(dim=["time"])[["temperature"]],
    )


def test_a_projection_hops_past_a_weighted_reduce_and_disarms_it(dated_ds):
    """A projection hops past a ``WeightedReduce`` and avoids an error eager would raise, end to end.

    Notes
    -----
    The clearest case of §8, end to end. Eagerly this chain *raises*: a weighted reduce
    runs per variable, and ``elevation`` has no ``time`` (a plain ``.mean("time")``
    would just do some wasted work on it). The projection moves in front, so
    ``elevation`` is gone before the reduce runs and the plan returns the value it was
    asked for. No error is hidden: it only ever came from discarded work.
    """
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean("time")[["temperature"]]

    assert [type(n) for n in chain._optimized()] == [Project, WeightedReduce]
    with pytest.raises(ValueError, match="do not exist"):
        ds.weighted(weights).mean("time")[["temperature"]]
    # ...and the value is exactly the one the hopped plan is supposed to compute
    xr.testing.assert_identical(
        chain.collect(), ds[["temperature"]].weighted(weights).mean("time")
    )


def test_a_projection_orphaning_a_weight_dim_is_left(dated_ds):
    """A projection that would drop a coord the weights align against is left alone, end to end.

    Notes
    -----
    The guard, end to end. ``elevation`` has no ``time``, so projecting to it drops the
    ``time`` *coordinate*. These weights have ``time``, so after the hop they'd
    broadcast a new ``time`` instead of inner-joining against the dataset's. That
    changes values, not just errors, which is why ``requires`` includes ``weight_dims``.
    """
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    weights = xr.DataArray(
        np.arange(1.0, 9.0), dims="time", coords={"time": ds["time"]}
    )
    chain = ds.plan.weighted(weights).mean("lat")[["elevation"]]

    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Project]
    xr.testing.assert_identical(
        chain.collect(), ds.weighted(weights).mean("lat")[["elevation"]]
    )


def test_select_on_a_consumed_group_dim_still_raises_from_xarray(dated_ds):
    """A select on a grouped reduce's consumed dim is left alone, and xarray raises at replay (not the optimiser).

    Notes
    -----
    Left rather than rejected, so the error comes from xarray at replay, in its own
    words, not an ``InvalidExpressionError`` the optimiser guessed at.
    ``pushdown_selects`` raises for a plain reduce because a reduce always removes what
    it names. A grouped reduce also creates a dim, so the same reasoning would reject
    valid chains.
    """
    chain = dated_ds.plan.groupby("time.month").mean().isel(time=0)
    with pytest.raises(ValueError, match="Dimensions"):
        chain.collect()


def test_builder_only_method_records_and_replays(ds):
    """A method that only exists on the builder (``first``) is recorded inside a context and replays to eager.

    Notes
    -----
    ``first`` is a ``DatasetGroupBy`` method that doesn't exist on Dataset. If the
    accessor asked the base object whether it's callable, it would take the property
    branch and compute eagerly, on a ``DatasetGroupBy`` that has no ``.compute()``.
    Inside a context that check is skipped, so it's recorded.
    """
    chain = ds.plan.groupby("lat").first()
    assert isinstance(chain, LazyProxy)
    assert_equal(chain.collect(), ds.groupby("lat").first())


def test_getitem_in_context_selects_a_group_not_a_variable(ds):
    """``__getitem__`` inside a ``groupby`` context selects a group, so it records ``Opaque``, not a ``Project``.

    Notes
    -----
    ``DatasetGroupBy.__getitem__`` selects a *group*; classifying it as a ``Project``
    would let projection pushdown treat the key as a variable name.
    """
    chain = ds.plan.groupby("lat")[0]
    assert isinstance(chain._ops[-1], Opaque)
    assert_equal(chain.collect(), ds.groupby("lat")[0])


def test_resample_matches_eager(dated_ds):
    """``resample().mean()`` collects to the same result as the eager chain."""
    chain = dated_ds.plan.resample(time="2D").mean()
    assert_equal(chain.collect(), dated_ds.resample(time="2D").mean())


def test_rolling_then_select_matches_eager(ds):
    """A select on a dim disjoint from a rolling window hops in front of it and still matches eager.

    Notes
    -----
    ``lat`` is disjoint from the window, so the rolling mean runs over one latitude
    instead of three.
    """
    chain = ds.plan.rolling(time=2).mean().isel(lat=0)
    assert [type(n) for n in chain._optimized()] == [Select, WindowedReduce]
    assert_equal(chain.collect(), ds.rolling(time=2).mean().isel(lat=0))


@pytest.mark.parametrize("boundary", ["trim", "pad"])
def test_coarsen_matches_eager(ds, boundary):
    """``coarsen().mean()`` followed by a select matches the eager result, for either boundary treatment."""
    chain = ds.plan.coarsen(time=2, boundary=boundary).mean().isel(lat=0)
    assert_equal(
        chain.collect(), ds.coarsen(time=2, boundary=boundary).mean().isel(lat=0)
    )


def test_rolling_closer_that_looks_like_a_dim_spec_fuses_and_matches_eager(ds):
    """A rolling closer whose positional arg looks like a dim name still fuses, and matches eager.

    Notes
    -----
    ``DatasetRolling.mean`` takes no dim argument, so this passes ``"lat"`` as
    *keep_attrs* and reduces nothing. The parsed dim has no effect, so lowering fuses
    the pair into one ``WindowedReduce`` (dropping ``to_opnode``'s wrong ``consumes``)
    rather than leaving an ``Opaque`` barrier, and the verbatim replay still matches
    xarray exactly.
    """
    chain = ds.plan.rolling(time=2).mean("lat")
    assert [type(n) for n in chain._optimized()] == [WindowedReduce]
    assert_equal(chain.collect(), ds.rolling(time=2).mean("lat"))


@pytest.mark.parametrize(
    "chain",
    [
        # an exponential weighting, not a fixed window -- ``window`` cannot describe it
        pytest.param(lambda o: o.rolling_exp(time=3).mean(), marks=requires_numbagg),
        pytest.param(
            lambda o: o.rolling_exp(time=3).mean().isel(lat=0), marks=requires_numbagg
        ),
        # a scan wearing a builder's clothes
        lambda o: o.cumulative("time").sum(),
    ],
)
def test_deliberately_unfused_openers_still_match_eager(ds, chain):
    """An opener deliberately not fused (``rolling_exp``, ``cumulative``) still replays verbatim and matches eager.

    Notes
    -----
    The other half of refusing to fuse: it has to *work*. These replay verbatim, so the
    pair stays opaque and the ops around it are unaffected.
    """
    plan = chain(ds.plan)
    assert [type(n) for n in plan._optimized()[:2]] == [Opaque, Opaque]
    assert_equal(plan.collect(), chain(ds))


def test_ops_after_a_windowed_reduce_are_modelled(ds):
    """A reduce after a windowed reduce is modelled as an ordinary ``Reduce`` and matches eager."""
    chain = ds.plan.rolling(time=2).mean().mean("time")
    assert [type(n) for n in chain._optimized()] == [WindowedReduce, Reduce]
    assert_equal(chain.collect(), ds.rolling(time=2).mean().mean("time"))


def test_weighted_matches_eager(ds):
    """``weighted().mean()`` lowers to one ``WeightedReduce`` node and matches the eager result."""
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean(("lat", "lon"))
    assert [type(n) for n in chain._optimized()] == [WeightedReduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean(("lat", "lon")))


def test_weighted_bare_closer_matches_eager(ds):
    """A bare ``weighted().mean()`` closer reduces every dim and matches eager.

    Notes
    -----
    A bare closer really does reduce every dim here (unlike a *grouped* bare closer,
    which only reduces along the group dim). ``DatasetWeighted.mean`` takes ``dim``, and
    ``dim=None`` means all of them, so the recorded ``ALL_DIMS`` carries over as is.
    """
    chain = ds.plan.weighted(ds["area"]).mean()
    assert [type(n) for n in chain._optimized()] == [WeightedReduce]
    assert_equal(chain.collect(), ds.weighted(ds["area"]).mean())


def test_select_after_a_weighted_reduce_does_not_move(ds):
    """A select after a ``WeightedReduce`` doesn't move, and the plan replays in the order written.

    Notes
    -----
    The rewrite the node exists to block, checked end to end: moving the ``isel`` left
    would leave the weights un-subset.
    """
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean("lat").isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["weighted", "mean", "isel"]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").isel(time=0))


def test_ops_after_a_weighted_reduce_are_modelled(ds):
    """A reduce after a ``WeightedReduce`` is modelled as an ordinary ``Reduce``, not left opaque.

    Notes
    -----
    Only the builder *pair* is affected by lowering, so the reduce after it is a normal
    ``Reduce``.
    """
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean("lat").mean("time")
    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Reduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").mean("time"))


@pytest.mark.parametrize("closer", ["sum_of_weights", "sum_of_squares"])
def test_weighted_only_closers_still_match_eager(ds, closer):
    """A weighted-only closer (``sum_of_weights``, ``sum_of_squares``) records ``Opaque`` and still matches eager.

    Notes
    -----
    The other half of refusing to fuse: the verbatim replay has to *work*. These aren't
    tabulated reductions, so they record ``Opaque`` and the pair stays opaque.
    """
    weights = ds["area"]
    chain = getattr(ds.plan.weighted(weights), closer)()
    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), getattr(ds.weighted(weights), closer)())


def test_weights_carrying_a_dim_the_dataset_lacks_match_eager(ds):
    """Weights with a dim the dataset lacks broadcast it onto every variable, and the plan still matches eager.

    Notes
    -----
    ``dot`` broadcasts, so this *adds* ``member`` to every variable, which is what
    ``weight_dims`` is for. Nothing moves across the node, so replay is verbatim. What's
    being checked is that the modelled schema didn't mislead a later rule.
    """
    weights = xr.DataArray([1.0, 2.0], dims="member", coords={"member": [0, 1]})
    chain = ds.plan.weighted(weights).mean("lat").mean("member")
    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Reduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").mean("member"))


def test_terminal_after_context_matches_eager(ds):
    """A terminal call after a ``groupby`` context still collects, rather than being recorded into the barrier.

    Notes
    -----
    ``_EAGER_ATTRS`` is checked before the context branch, so terminals still collect
    rather than being recorded and never run.
    """
    result = ds.plan.groupby("time").mean().to_dataframe()

    assert not isinstance(result, LazyProxy)
    assert result.equals(ds.groupby("time").mean().to_dataframe())


def test_plans_without_a_context_still_optimise(ds):
    """An ordinary chain with no ``groupby`` context is still rewritten; the barrier must not leak."""
    chain = ds.plan.mean("lat").isel(time=0)
    assert [n.name for n in chain._optimized()] == ["isel", "mean"]


def test_terminal_to_dataframe_triggers_collect_and_delegates(ds):
    """``.to_dataframe()`` runs collect and delegates, rather than returning a proxy that never runs.

    Notes
    -----
    A terminal like ``.to_dataframe()`` must compute rather than return a proxy
    (recorded, never run). This checks the rewrite ran and the call reached xarray.
    """
    chain = ds.plan["temperature"].isel(time=0)
    result = chain.to_dataframe()

    assert not isinstance(result, LazyProxy)
    eager = ds["temperature"].isel(time=0).to_dataframe()
    assert result.equals(eager)


def test_terminal_to_dict_matches_eager(ds):
    """``.to_dict()`` matches the eager chain's ``.to_dict()``."""
    chain = ds.plan.mean("lat")
    assert chain.to_dict() == ds.mean("lat").to_dict()


@requires_matplotlib
def test_plot_triggers_collect_and_delegates(ds):
    """``.plot`` delegates to the computed DataArray's plot accessor, not a recording wrapper.

    Notes
    -----
    ``.plot`` is xarray's plot accessor on the *computed* DataArray, so calling it draws
    rather than returning a proxy. Delegating the whole ``plot`` attribute (not just the
    call) also reaches the accessor's own methods, e.g. ``plot.line`` on a 1-D
    selection.
    """
    import matplotlib

    matplotlib.use("Agg")

    chain = ds.plan["temperature"].isel(time=0)

    assert not isinstance(chain.plot, LazyProxy)
    artist = chain.plot()
    assert not isinstance(artist, LazyProxy)
    assert type(artist) is type(ds["temperature"].isel(time=0).plot())

    line = ds.plan["temperature"].isel(time=0, lat=0).plot.line()
    assert not isinstance(line, LazyProxy)


# --- the DataArray accessor ----------------------------------------------------
#
# ``.plan`` is registered on both types and one class handles both, so this isn't a
# second suite. It tests the two ways a ``DataArray`` differs (``__getitem__`` means
# indexing, and there are no ``data_vars`` to project) and checks that everything else
# (recording, lowering, the rules, replay) works the same.


@pytest.fixture
def da(ds) -> xr.DataArray:
    """The shared fixture's ``temperature`` variable, as a standalone DataArray.

    Returns
    -------
    xarray.DataArray
        ``temperature(time, lat, lon)`` with its coords, the ``area`` aux coord
        included.
    """
    return ds["temperature"]


def test_plan_accessor_on_a_dataarray_returns_proxy(da):
    """``da.plan`` returns a ``LazyProxy`` over the array itself, with an empty op list."""
    assert isinstance(da.plan, LazyProxy)
    assert da.plan._ops == []
    assert da.plan._base is da


@pytest.mark.parametrize(
    "chain",
    [
        # the README pipeline, positional and keyword spellings
        lambda o: o.mean("lat").mean("lon").isel(time=0),
        lambda o: o.mean(dim="lat").mean(dim="lon").isel(time=0),
        lambda o: o.mean(dim=("lat", "lon")).isel(time=0),
        # a select the rewrite hoists in front of a disjoint reduce
        lambda o: o.sum("lat").isel(time=0),
        lambda o: o.mean("lat").sel(time=slice(0, 2)),
        # adjacent selects, which merge
        lambda o: o.isel(time=slice(0, 3)).isel(lat=0),
        # immovable: the select indexes the dim the reduce consumed's neighbour survives
        lambda o: o.mean("time").isel(lat=0),
        # a bare reduce, whose ``ALL_DIMS`` resolves against a coords-only schema
        lambda o: o.mean(),
        # a fused grouping, and a select around it
        lambda o: o.groupby("time").mean(),
        lambda o: o.rolling(time=3).mean().isel(lat=0),
        lambda o: o.weighted(xr.DataArray([1.0, 2.0, 3.0], dims="lat")).mean("lat"),
        # an unmodelled op, which barriers rather than misleads
        lambda o: o.rename({"time": "t"}).mean("t"),
    ],
)
def test_dataarray_chains_match_eager(da, chain):
    """Every README-style chain on a ``DataArray`` collects to the eager result."""
    assert_equal(chain(da.plan).collect(), chain(da))


def test_dataarray_rewrites_still_fire(da):
    """The optimiser rewrites a ``DataArray`` plan; it doesn't just pass it through.

    Notes
    -----
    A ``DataArray`` only affects the *projection* rules (it has no ``data_vars``).
    Select pushdown reasons about dims, which a ``DataArray`` has, so it fires just as
    it does on a ``Dataset``.
    """
    chain = da.plan.mean("lat").isel(time=0)
    assert [n.name for n in chain._optimized()] == ["isel", "mean"]


def test_dataarray_getitem_records_opaque_not_a_projection(da):
    """``da.plan["lat"]`` records an ``Opaque``: ``__getitem__`` on a ``DataArray`` is indexing.

    Notes
    -----
    The key *looks* like a projection to :func:`~xrexpr.schema.to_opnode`, which only
    sees the call. What decides it is the object being indexed, as for a builder's
    ``__getitem__``.
    """
    node = da.plan["lat"]._ops[0]
    assert isinstance(node, Opaque) and node.name == "__getitem__"
    assert_equal(da.plan["lat"].collect(), da["lat"])


def test_positional_getitem_pairs_on_a_dataarray_are_not_merged(da):
    """Two positional ``__getitem__`` calls stay two ops, so the selected rows are the eager ones.

    Notes
    -----
    The bug recording these as ``Opaque`` prevents, using the keys that would trigger
    it. ``da[[1, 2]]`` selects *positions along the first dim*. As two ``Project``
    nodes, ``da[[1, 2]][[1]]`` would pass ``merge_adjacent_projects``' subset test (the
    rule only looks at the two keys) and collapse to ``da[[1]]``: row 1, where the eager
    chain indexes *within* the selected pair and gives row 2. Checked by deliberately
    classifying the pair as projections: the merged plan returns the wrong row.
    """
    chain = da.plan[[1, 2]][[1]]

    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), da[[1, 2]][[1]])


def test_a_coordless_dataarray_still_replays_equal_to_eager(da):
    """A ``DataArray`` with no coords gives a schema with no dims, and every chain still matches eager.

    Notes
    -----
    ``SchemaState.from_dataset`` reads a ``DataArray``'s *coords*, so one without coords
    gives an empty schema: the model knows of no dims. That only removes facts, so any
    rule that needs one declines. Here ``groupby("time")`` fails
    ``lower._grouper_dims``' dim check and falls back to the opaque pair instead of
    fusing, which costs a rewrite but not correctness.
    """
    bare = xr.DataArray(da.values, dims=da.dims)
    assert not SchemaState.from_dataset(bare).dim_names

    assert_equal(
        bare.plan.mean("lat").isel(time=0).collect(), bare.mean("lat").isel(time=0)
    )
    assert_equal(
        bare.plan.groupby("time").mean().collect(), bare.groupby("time").mean()
    )


def test_dataarray_terminal_materialises(da):
    """A ``DataArray``-only terminal (``.to_index``) materialises the plan instead of recording it."""
    chain = da.plan.mean(("lat", "lon"))

    assert not isinstance(chain.to_index(), LazyProxy)
    assert chain.to_index().equals(da.mean(("lat", "lon")).to_index())


def test_dataarray_explain_reads_the_plan(da):
    """``explain()`` works off a ``DataArray`` base, showing the hoisted select."""
    text = da.plan.mean("lat").isel(time=0).explain()
    assert text.index("isel") < text.index("mean")


def test_getitem_after_a_bare_name_projection_is_indexing_too(ds):
    """A chain that *becomes* a DataArray records its later ``__getitem__`` calls as indexing, not projection.

    Notes
    -----
    The base is a ``Dataset`` here, so the base alone can't decide it. After a bare-name
    projection the object is a ``DataArray``, and ``ds["temperature"][[1, 2]][[1]]`` is
    the same wrong-row collapse as on a ``DataArray`` base, one op later.
    ``merge_adjacent_projects``' guard against a ``single`` first projection stops the
    *first* pair, not this one.
    """
    chain = ds.plan["temperature"][[1, 2]][[1]]

    assert [type(n) for n in chain._ops] == [Project, Opaque, Opaque]
    assert_equal(chain.collect(), ds["temperature"][[1, 2]][[1]])


@pytest.mark.parametrize(
    "escape",
    [
        # the two doors a recorded ``Opaque`` hands back a DataArray through
        lambda o: o.pipe(lambda d: d["temperature"]),
        lambda o: o.get("temperature"),
    ],
    ids=["pipe", "get"],
)
def test_getitem_after_an_opaque_is_not_a_projection(ds, escape):
    """An ``Opaque`` may return a ``DataArray``, so ``__getitem__`` after it is indexing too.

    Notes
    -----
    ``Opaque`` means unmodelled, and that includes its *result type*. Both ``pipe`` and
    ``get`` turn a ``Dataset`` into a ``DataArray`` with no modelled node involved.
    Classified by key alone, the pair would be two ``Project`` nodes, and
    ``merge_adjacent_projects`` (which would fire past an ``Opaque``) would collapse
    ``[[1, 2]][[1]]`` to ``[[1]]``. That returns valid data from the *wrong rows* rather
    than raising, so the equality assertion is the one that matters.
    """
    chain = escape(ds.plan)[[1, 2]][[1]]

    assert [type(n) for n in chain._ops] == [Opaque, Opaque, Opaque]
    assert_equal(chain.collect(), escape(ds)[[1, 2]][[1]])


def test_getitem_after_a_dataset_valued_opaque_is_demoted_too(ds):
    """An opaque that *does* return a Dataset still makes what follows opaque, losing one merge.

    Notes
    -----
    The accepted cost of treating an ``Opaque``'s result type as unknown, tested so it's
    intended rather than a surprise. ``transpose`` returns a ``Dataset``, so these two
    list projections could merge safely, but only the opaque's *name* tells you that,
    and a table of names that return Datasets would go stale as xarray grows. Projection
    pushdown loses nothing, because :func:`~xrexpr.optimize._trusted_prefix` already
    stops it at the first ``Opaque``. ``merge_adjacent_projects`` is the only thing
    lost. (``rename`` wouldn't work as the example: it's a modelled
    :class:`~xrexpr.ir.Rename`, so the merge across it fires.)
    """
    chain = ds.plan.transpose()[["temperature", "elevation"]][["temperature"]]

    assert [type(n) for n in chain._ops] == [Opaque, Opaque, Opaque]
    assert_equal(
        chain.collect(),
        ds.transpose()[["temperature", "elevation"]][["temperature"]],
    )


def test_projections_merge_across_a_modelled_rename(ds):
    """Two list projections merge across a modelled ``Rename``.

    Notes
    -----
    The counterpart of ``test_getitem_after_a_dataset_valued_opaque_is_demoted_too``.
    ``rename`` is a modelled :class:`~xrexpr.ir.Rename`, not an ``Opaque`` with unknown
    result type, so the ``__getitem__`` calls after it stay :class:`~xrexpr.ir.Project`
    and ``merge_adjacent_projects`` collapses the redundant pair.
    """
    chain = ds.plan.rename({"temperature": "t2m"})[["t2m", "elevation"]][["t2m"]]
    assert [type(n) for n in chain._ops] == [Rename, Project, Project]

    optimised = optimize(list(chain._ops), chain._base_schema())
    assert [type(n) for n in optimised] == [Rename, Project]
    assert optimised[1].variables == ("t2m",)
    assert_equal(
        chain.collect(),
        ds.rename({"temperature": "t2m"})[["t2m", "elevation"]][["t2m"]],
    )


def test_a_list_form_projection_keeps_the_chain_in_dataset_land(ds):
    """A list projection gives a Dataset, so later ``__getitem__`` calls are still recorded as projections.

    Notes
    -----
    The two forms are told apart by the *key*, not the node's ``variables``, which is
    ``("temperature",)`` for both. Deciding from the node alone would lose this case,
    and the projection rules would stop firing on ordinary ``ds[[...]][[...]]`` chains.
    """
    chain = ds.plan[["temperature", "elevation"]][["temperature"]]

    assert [type(n) for n in chain._ops] == [Project, Project]
    assert_equal(chain.collect(), ds[["temperature", "elevation"]][["temperature"]])
