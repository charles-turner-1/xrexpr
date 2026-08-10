"""Tests for the ``.plan`` accessor.

Covers recording behaviour and *equality* — ``ds.plan.<chain>.collect()`` matches
the eager ``ds.<chain>`` for the README pipelines, exercising record → optimise →
replay end to end. The golden op-list assertions that pin the optimiser itself live
in ``tests/test_optimize.py`` (it owns ``optimize``); here we only care that the
accessor records the right nodes and that the pipeline replays to the right result.
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
    Elementwise,
    GroupedReduce,
    Opaque,
    Project,
    Reduce,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.lower import emit
from xrexpr.optimize import _schemas
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

#: ``rolling_exp`` is the one opener xarray refuses outright without numbagg (an explicit
#: ``ImportError``, not a slow fallback), so only its *replay* needs this -- the lowering
#: half of those tests asserts fine either way. Nothing else in the suite needs numbagg
#: **installed**; the ``test-nonumbagg`` environment exists because numbagg changes
#: reduction *results*, not because it gates features.
requires_numbagg = pytest.mark.skipif(
    importlib.util.find_spec("numbagg") is None, reason="numbagg is not installed"
)


# A richer dataset than the shared ``conftest.ds``: this module shadows it to add an
# auxiliary (non-dimension) ``area`` coord, exercising that projection pushdown preserves
# aux coords when it reorders the plan.
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
    Recording is a pure append -- the proxy carries the base dataset and the plan and
    nothing else. The single logical-schema fold belongs to the optimiser, which plans from
    the front and so is the reader whose fold is exact.
    """
    chain = ds.plan.mean("lat").isel(time=0)
    assert not hasattr(chain, "_schema")

    entering = _schemas(chain._ops, chain._base_schema())
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
    The payoff of reading the dim spec from the *second* positional (#96): the node now
    carries a true ``consumes``, so ``.reduce`` chains optimise like any other reduction
    instead of being a barrier.
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


def test_keepdims_reduce_then_select_matches_eager(ds):
    """A ``keepdims=True`` reduce keeps its dim, so a select on that dim must replay, not raise.

    Notes
    -----
    The regression this pins was **live**: ``keepdims=True`` keeps ``time`` at size 1, but
    the node claimed ``consumes={"time"}``, so ``pushdown_selects`` classified the select
    as indexing a reduced-away dim and raised ``InvalidExpressionError`` on a chain that
    works eagerly. Recording ``Opaque`` makes the reduce a barrier instead -- unoptimised,
    which is the house's answer for a call it cannot model, and correct.
    """
    got = ds.plan.mean("time", keepdims=True).isel(time=0).collect()
    assert_equal(got, ds.mean("time", keepdims=True).isel(time=0))


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
    """A bare ``mean()`` followed by a select raises, end to end -- the empty-dim reorder bug is now an error, not a wrong swap."""
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
    because the scan runs independently at each ``lat``, and ``assert_equal`` is what proves
    the reorder changed no value.
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
    The rewrite reduces only ``temperature``; the result must still match the eager chain --
    coords included, which is what ``assert_equal`` checks.
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
    Records the variant and exercises record -> optimise -> replay end to end. Nothing
    reorders around the elementwise op yet (W5 PR 1 leaves it a barrier in ``dim_effect``);
    that the plan is still correct is what this pins.
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


def test_orthogonal_dataarray_select_records_a_select_and_replays_equal(ds):
    """An orthogonal ``isel(<DataArray>)`` records a positional ``Select`` (not ``Opaque``) and replays to eager.

    Notes
    -----
    The optimisation payoff: the select is no longer a barrier, so ``mean("lat").isel(...)``
    optimises (the select hops the disjoint reduce — pinned structurally in
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
    The bug this fixes: the advanced ``isel`` was misfiled as a dim-dropping ``Scalar``, so
    the bare ``mean()`` understated its ``consumes`` and the trailing ``isel(time=0)`` was
    swapped in front — returning a value where the eager chain errors. Now the orthogonal
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
    The reorder is what this pins — ``isel`` leads the optimised plan — and ``assert_equal``
    proves crossing the elementwise op (and the disjoint reduce) changed no value, the
    dropped ``time`` dim included.
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
    ``fillna`` and a reduce do not commute when a NaN is present -- fill-then-average
    differs from average-then-fill -- so this is the case a hypothetical reduce-crossing
    bug would break. The shared fixtures carry no NaN, which would make ``fillna`` a no-op
    and rob an equality check of its teeth, so this builds a dataset with one. A select is
    threaded through to make the optimiser actually rewrite: it hops to the front past both
    ops, and the ``fillna``/``mean`` order must survive intact.
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
    ``elevation`` has no ``time``, so the projection can't move -- but the plan is perfectly
    valid eagerly.
    """
    got = ds.plan.mean("time")[["elevation"]].collect()
    assert_equal(got, ds.mean("time")[["elevation"]])


def test_merged_projections_equal(ds):
    """Two projections merged into one collect to the eager result, auxiliary coords included.

    Notes
    -----
    ``assert_identical`` rather than ``assert_equal``: the claim the merge rests on is
    that a projection's *coordinate* pruning composes, so names and attrs are exactly
    what is under test -- the ``area(lat, lon)`` coord this module's fixture adds is the
    one a sloppier rule would drop.
    """
    got = ds.plan[["temperature", "elevation"]][["temperature"]].collect()
    assert len(ds.plan[["temperature", "elevation"]][["temperature"]]._optimized()) == 1
    xr.testing.assert_identical(got, ds[["temperature", "elevation"]][["temperature"]])


def test_merged_projections_keep_the_dataarray_result(ds):
    """A list projection merged with a following bare name still yields the eager ``DataArray``."""
    got = ds.plan[["temperature", "elevation"]]["temperature"].collect()
    assert isinstance(got, xr.DataArray)
    xr.testing.assert_identical(got, ds[["temperature", "elevation"]]["temperature"])


def test_a_projection_naming_a_coord_is_left_and_still_replays(ds):
    """A projection naming a *coordinate* fails the subset test, is left alone, and replays equal.

    Notes
    -----
    ``ds[["temperature"]]["lat"]`` is valid eagerly, because a projection keeps the coords
    its variables span. It is *not* a spelling of ``ds["lat"]``: the chain reads ``lat`` as
    it exists inside ``p1``'s result, which agrees with the base coord only by accident --
    a select or reduce between the two projections, or a ``p1`` that prunes or resizes it,
    and they differ. So this asserts the conservative leg, not a missed rewrite.

    There *is* a win here, and it is a different rule: ``p1`` materialises ``temperature``
    only to discard it, where the optimal plan reads the coord and never touches the data
    variable. That needs the schema (does ``lat`` survive ``p1`` untouched?), which puts it
    in ``pushdown_projections``' family rather than the syntactic merge -- filed as #115.
    """
    chain = ds.plan[["temperature"]]["lat"]
    assert [type(n) for n in chain._optimized()] == [Project, Project]
    xr.testing.assert_identical(chain.collect(), ds[["temperature"]]["lat"])


def test_merging_projections_skips_an_error_from_a_discarded_name(ds):
    """Merging drops a first projection whose extra name doesn't exist — §8's discarded-work clause.

    Notes
    -----
    Eager raises ``KeyError`` building an intermediate the chain throws away one node
    later; the merged plan asks only for what the chain actually wanted and succeeds. The
    same licence ``pushdown_projections`` runs on
    (``test_projection_pushdown_skips_an_error_from_a_discarded_variable``): no value
    moves, and the skipped error comes from work whose result is discarded.
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
    """``explain()`` names every emitted call, in order, even at the node-level listing.

    Notes
    -----
    The property the call-level listing had and the node-level one must not give up: every
    emitted call appears, in order. Derived from ``emit`` rather than restated, so it cannot
    pass by agreeing with a stale expectation.
    """
    chain = ds.plan.groupby("time.month").mean().isel(lat=0)
    text = chain.explain()
    positions = [text.index(c.name) for c in emit(chain._optimized())]
    assert positions == sorted(positions)


def test_explain_names_the_opaque_nodes_that_stopped_the_optimiser(ds):
    """``explain()`` names the ``Opaque`` nodes and states they are unmodelled, answering why nothing moved.

    Notes
    -----
    The question the call-level listing could not answer: *why* did nothing move? An
    unfusable context is two opaque nodes, and a reader should not have to know which methods
    are modelled to find that out.
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
    The REPL echoes ``repr()``; it must be the formatted multi-line plan, not the escaped
    one-line form a plain ``str`` would produce.
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
    The regression this rewrite exists to avoid: leaving ``time`` in the spec after the
    select has dropped it raises ``ValueError: chunks keys ... not found``.
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
    The other half of the contract ``test_chunks.py`` pins without dask, where every string
    classifies as a ``ByteSize`` unexamined. This is what unexamined buys: the same
    ``ValueError`` xarray raises eagerly, in dask's words rather than any xrexpr invented.

    Deferred, not lost. Recording the chain succeeds where the eager spelling has already
    raised, and the rechunk is pushable (a byte target re-picks against whatever data
    survives), so the select really does hop in front of it here. Neither the deferral nor
    the rewrite may swallow the failure.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": "banana"}).isel(time=slice(50, 250))  # records fine

    with pytest.raises(ValueError, match="byte unit"):
        ds.chunk({"time": "banana"}).isel(time=slice(50, 250)).compute()
    with pytest.raises(ValueError, match="byte unit"):
        chain.collect()


@requires_dask
def test_an_emptying_select_does_not_cross_an_auto_rechunk(chunky_ds):
    """A select that empties a dim stays behind an ``"auto"`` rechunk, and the chain matches eager.

    Notes
    -----
    Issue #121, end to end. ``isel(lat=[])`` keeps ``lat`` at length 0. Eagerly the rechunk
    runs first, on data that is still full, and only then is the dim emptied; hoisted, it
    would be handed a zero-size array and asked to size ``time``'s blocks against it —
    ``ZeroDivisionError`` where the eager chain succeeds.

    Both halves matter. Replaying equal to eager says the plan runs at all; the order
    assertion says it runs because the rule *refused the hop*, not because dask happened to
    tolerate it. The refusal is not specific to an emptying select — every select is
    refused, because ``"auto"`` is extent-dependent — which
    :func:`test_a_harmless_select_also_stays_behind_an_auto_rechunk` pins from the other
    side.
    """
    ds = chunky_ds
    chain = ds.plan.chunk({"time": "auto"}).isel(lat=[])

    assert_equal(chain.collect(), ds.chunk({"time": "auto"}).isel(lat=[]).compute())
    text = chain.explain()
    assert text.index("chunk") < text.index("isel")


@requires_dask
def test_an_emptying_select_does_not_cross_a_uniform_auto_rechunk(chunky_ds):
    """The same refusal for ``chunk("auto")``, which names no dim — and the crash it prevents.

    Notes
    -----
    ``chunk("auto")`` rides in ``args`` rather than ``chunks``, but it is the same request
    and carries the same hazard. It only *looks* safe on a small array: ``auto_chunks`` pins
    every dim below ``limit ** (1 / ndim)`` and returns before it can divide.

    So ``array.chunk-size`` is doing the work a multi-hundred-megabyte fixture otherwise
    would — at 1 kB this dataset is over the target, and hoisting ``isel(lat=[])`` in front
    raises where the eager order returns. The setting is load-bearing, not incidental:
    without it this test passes against a rule that makes the hop, which is what the
    ``pytest.raises`` below asserts it is not doing.
    """
    import dask

    ds = chunky_ds
    with dask.config.set({"array.chunk-size": "1kB"}):
        with pytest.raises(ZeroDivisionError):  # the divergence, if the hop were made
            ds.isel(lat=[]).chunk("auto")

        chain = ds.plan.chunk("auto").isel(lat=[])
        assert_equal(chain.collect(), ds.chunk("auto").isel(lat=[]).compute())
        text = chain.explain()
        assert text.index("chunk") < text.index("isel")


@requires_dask
def test_a_harmless_select_also_stays_behind_an_auto_rechunk(chunky_ds):
    """Even a select that empties nothing stays behind an ``"auto"`` rechunk, and still matches eager.

    Notes
    -----
    The cost of drawing the line on the *spec* rather than on the pair, stated as a test.
    ``isel(lat=slice(0, 2))`` cannot empty anything, and it is refused anyway, because
    whether ``"auto"`` survives a hop is not decidable from the plan: it depends on the
    configured ``array.chunk-size`` and on the rank — see :class:`~xrexpr.chunks.Auto`.

    An extent-*independent* spec in the same position still crosses, which
    :func:`test_a_select_crosses_a_sized_rechunk` pins.
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
    """A select crosses every spec dask does *not* size from the array, emptying or not.

    Notes
    -----
    The other side of the line, end to end: a fixed size is that size on any array, so the
    hop is made and the result still matches eager. Without this the refusals above would be
    consistent with a rule that had simply stopped working.
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


def test_compute_is_alias_for_collect(ds):
    """``.compute()`` is a synonym for ``.collect()``, not a recorded op: it returns the materialised result, not a proxy.

    Notes
    -----
    ``.compute()`` exists for xarray muscle memory.
    """
    chain = ds.plan.mean("lat").isel(time=0)
    assert_equal(chain.compute(), chain.collect())


# --- terminal operations -------------------------------------------------------------
# ``.plot`` / ``.to_*`` consume the plan into a figure/file/array rather than another
# link in the chain. They are callable (``ds.plot`` is an accessor with ``__call__``),
# so without special handling they would be *recorded* and silently never run. They must
# instead trigger ``collect()`` (firing the rewrite) and delegate to the realised object.


@pytest.mark.parametrize("term", sorted(_EAGER_ATTRS))
def test_every_terminal_is_routed_not_recorded(ds, term):
    """Every name in ``_EAGER_ATTRS`` is routed to the materialised object, never recorded as a plan op.

    Notes
    -----
    Coverage guard over the whole ``_EAGER_ATTRS`` set: catches a typo'd / nonexistent name
    in the set, or a terminal that gets silently recorded instead of delegated. Route through
    the chain whose collected result actually carries the terminal -- some are Dataset-only
    (``to_array``, ``to_stacked_array``), some DataArray-only (``to_series``, ``to_numpy``).
    Picking by ``hasattr`` keeps this declarative and avoids hardcoded lists that would rot
    if the set changes.
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
    """A name absent from ``_EAGER_ATTRS`` is not routed to the eager-delegate path, xfailed strictly.

    Notes
    -----
    The negative of :func:`test_every_terminal_is_routed_not_recorded`: a name absent from
    ``_EAGER_ATTRS`` must NOT reach the eager-delegate path. It falls through to the
    record / property branch and (being bogus) raises ``AttributeError`` -- so the routing
    assertions below never hold. Strict xfail, so if ``to_nowhere`` is ever added to the set
    (or otherwise made to route) this turns green and fails the suite, forcing a deliberate
    update.
    """
    term = "to_nowhere"
    assert term not in _EAGER_ATTRS
    chain = ds.plan["temperature"].mean("lat")
    attr = getattr(chain, term)
    assert not isinstance(attr, LazyProxy)
    assert type(attr) is type(getattr(chain.collect(), term))


# --- grouped / windowed contexts -----------------------------------------------------
# ``groupby``/``resample``/``rolling``/... return a builder, not a Dataset, so calls after
# them mean something different from the same-named Dataset ops. Lowering fuses a
# recognised opener/closer pair into one semantic node; a pair it can't fuse demotes to
# ``Opaque``, which no rewrite rule can fire on or across -- correct but unoptimised.


@pytest.fixture
def dated_ds() -> xr.Dataset:
    """A dataset with a real datetime ``time`` index, for the tests that need ``resample``.

    Returns
    -------
    xarray.Dataset
        ``temperature(time, lat, lon)`` with a datetime ``time`` coord -- ``ds`` has an
        integer one, and ``resample`` needs a real one. ``lon`` and the ``area`` aux coord
        are here for the fused-reduce pushdown tests: a third dim gives the select somewhere
        disjoint to go, and a non-dimension coord spanning the selected dims checks the
        reorder carries it along.
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
    Valid *eagerly* -- ``groupby('time').mean()`` reduces within groups, leaving ``lat`` for
    the later ``isel(lat=0)``. Before the barrier the proxy recorded the grouped ``mean()``
    as a full-dataset reduce (no dim -> every current dim), so its schema dropped ``lat`` and
    ``collect()`` raised ``InvalidExpressionError``.
    """
    chain = ds.plan.groupby("time").mean().isel(lat=0)
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(lat=0))


def test_select_after_grouped_reduce_matches_eager(ds):
    """A select on the grouped dim itself matches eager, replaying in the order written.

    Notes
    -----
    The other failure mode: a select on a dim *disjoint* from the bogus ``consumes`` was
    silently swapped behind the reduce, and replay then looked ``isel`` up on the
    ``DatasetGroupBy``. Barriered, it replays in the order written.
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

    # ... which is what ``explain()`` shows: one op, still rendered as the
    # two calls it replays as, in the order they were written. ``time -> time`` is not a
    # typo and is the reason the annotation exists -- ``groupby("time")`` *re-mints* the
    # dim, so the ``time`` the trailing ``isel`` indexes is group labels, not the original.
    assert chain.explain() == Explanation(
        "plan (2 ops):\n"
        "  1. GroupedReduce  groupby('time').mean()  [time -> time]\n"
        "  2. Select  isel(time=0)"
    )


@pytest.fixture
def regioned_ds() -> xr.Dataset:
    """A dataset with a non-dim coordinate grouper, the shape issue #90 was about.

    Returns
    -------
    xarray.Dataset
        ``temperature(time, lat)`` with a ``region`` coord defined on ``lat``, so
        ``groupby("region")`` groups along ``lat`` and mints ``region`` -- everyday xarray.
        ``lat``'s three values give two groups.
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
    """A ``groupby`` on a non-dim coordinate refuses to fuse and falls back to the opaque pair, matching eager.

    Notes
    -----
    Issue #90. The grouper's *name* is not the dim it consumes: ``groupby("region")``
    consumes ``lat`` and mints ``region``, but nothing in the call says so, and fusing on the
    call's own reading gave ``group_dim="region"`` — a schema claiming ``lat`` survived a
    grouping that removed it, read by the fold *and* by ``dim_effect``'s ``blocks`` and
    ``requires``. Refusing puts the pair on the opaque fallback: correct, unoptimised, and
    what ``02-lowering.md`` §5.5 specified from the start.
    """
    chain = regioned_ds.plan.groupby("region").mean()

    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), regioned_ds.groupby("region").mean())


def test_explain_prints_no_group_arrow_for_a_coordinate_grouper(regioned_ds):
    """``explain()`` prints no ``[group_dim -> new_dim]`` arrow for an unfused coordinate grouper.

    Notes
    -----
    The user-visible half of the issue #90 fix: ``explain()`` renders a fused node's
    inference as ``[group_dim -> new_dim]``, so pre-fix it stated ``[region -> region]`` --
    the tool asserting a wrong fact about the data. There is no arrow to get wrong now.
    """
    text = str(regioned_ds.plan.groupby("region").mean().explain())

    assert "->" not in text
    assert "GroupedReduce" not in text


def test_ops_after_an_unfusable_context_are_modelled_again(ds):
    """A reduce after an unfusable ``groupby`` context is modelled as an ordinary ``Reduce``, not left opaque.

    Notes
    -----
    The trailing barrier is retired. ``first`` closes the context and returns a Dataset, so
    the ``mean`` after it is an ordinary Dataset reduce -- a chain the barrier used to render
    permanently opaque from the groupby onward.
    """
    chain = ds.plan.groupby("lat").first().mean("time")

    lowered = chain._optimized()
    assert [type(n) for n in lowered] == [Opaque, Opaque, Reduce]
    assert_equal(chain.collect(), ds.groupby("lat").first().mean("time"))


def test_select_on_a_grouped_nodes_own_dim_stays_put(ds):
    """A select on a grouped reduce's own (re-minted) dim stays put, replaying to the eager result.

    Notes
    -----
    ``groupby("time")`` mints the dim it grouped over, so ``isel(time=0)`` indexes the
    *minted* ``time`` -- valid eagerly, and immovable. Pinned end to end because it is the
    case the select-pushdown rule must decline: hopping this in front would index the
    original ``time`` instead of the group labels.
    """
    chain = ds.plan.groupby("time").mean().isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["groupby", "mean", "isel"]
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(time=0))


def test_climatology_select_is_pushed_in_front_of_the_grouping(dated_ds):
    """A select hops in front of a climatology grouping, so it runs over one latitude and still matches eager.

    Notes
    -----
    The payoff the whole lowering workstream was built for: the grouping runs over one
    latitude instead of over all of them and then discarding the rest. Both halves are
    asserted -- that the rewrite *fires*, so this cannot pass vacuously, and that it still
    equals eager.
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
    Whether the select moved or not, the answer must not change -- which is the only thing
    that makes "hop when disjoint, leave otherwise" worth anything.
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
    Pinned end to end because the failure would be loud rather than wrong: projecting first
    drops the ``time`` coord (no surviving variable uses it) and ``rolling`` raises
    ``Window dimensions ('time',) not found``. The rule must not create that.
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
    """Pushing a projection down skips work the plan discards — errors included.

    ``station`` is a *string* variable, so a Dataset ``std("time")`` raises while reducing
    it. Eager therefore fails computing a standard deviation the chain explicitly throws
    away; the rewrite never computes it. The optimised answer is the better one — see
    ``planning/roadmap/07-small-wins.md`` §8, which settles this as a deliberate property
    rather than a divergence to be repaired.

    Safe because the values cannot move: the rule fires only once the projected variables
    are known to carry the dims the crossed op names, so they reduce identically either
    way, and the sole difference is a discarded variable's error.
    """
    # ------------------------------------------------------------------------------
    # WHY A *STRING* VARIABLE, AND NOT THE OBVIOUS FLOAT ONE. Read before "simplifying".
    #
    # What this test asserts is a fact about *xrexpr*: ``pushdown_projections`` skips work
    # on a discarded variable, errors included. To assert it we need eager to fail, and any
    # failure will do -- the failure is a *prop*, not the subject.
    #
    # The tempting alternative sources that failure from a float variable lacking ``time``:
    # a Dataset reduce hands such a variable an empty axis set, and ``std``/``var`` blow up
    # on it. That looks like the natural choice and it is a trap, because **the blow-up is a
    # numbagg bug, not xarray behaviour** -- numbagg reads ``axis=()`` as ``axis=None`` and
    # returns a scalar (07-small-wins.md §8 has the full account). Pinning it would make
    # this test assert that a third-party bug is *present*: it would fail under
    # ``use_numbagg=False``, and again the day numbagg fixed it -- in both
    # cases reporting DID NOT RAISE, as though xrexpr had regressed.
    #
    # numpy refusing to ``std`` a ``<U4`` array is xarray/numpy's own semantics and reaches
    # numbagg not at all -- its dispatch gate requires ``dtype.kind in "uif"``. So the prop
    # holds still with numbagg present, absent, or fixed. Swapping back to a float variable
    # re-couples this test to numbagg's release schedule.
    #
    # NOTE this changes nothing about the numbagg bug itself, which is not ours to fix and
    # is still live in this environment -- see EMPTY_AXIS_UNSAFE_REDUCES in test_properties.
    # ------------------------------------------------------------------------------
    ds = dated_ds.assign(station=(("lat", "lon"), np.full((3, 4), "buoy")))

    with pytest.raises(TypeError, match="resolved dtypes"):
        ds.std(dim=["time"])[["temperature"]]

    planned = ds.plan.std(dim=["time"])[["temperature"]].collect()
    # the reference is the same chain with the discarded variable simply absent
    assert_equal(planned, ds[["temperature"]].std(dim=["time"]))


def test_projection_pushdown_introduces_no_error_of_its_own(dated_ds):
    """Projection pushdown introduces no error of its own when both orders succeed; they must agree exactly.

    Notes
    -----
    The other half of §8's contract: skipping a discarded variable's failure is allowed,
    inventing one is not. ``mean`` skips a variable lacking the dim rather than refusing, so
    both orders succeed and must agree exactly.
    """
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    assert_equal(
        ds.plan.mean(dim=["time"])[["temperature"]].collect(),
        ds.mean(dim=["time"])[["temperature"]],
    )


def test_a_projection_hops_past_a_weighted_reduce_and_disarms_it(dated_ds):
    """A projection hops past a ``WeightedReduce`` and disarms an error that eager would raise, end to end.

    Notes
    -----
    §8's contract at its sharpest, end to end. Eagerly this chain *raises*: a weighted reduce
    maps per variable and ``elevation`` has no ``time``, where a plain ``.mean("time")``
    would merely waste effort on it. The projection hops in front, so ``elevation`` is gone
    before the reduce runs and the plan delivers the value it was asked for. Not a masked
    error -- an error that only ever came from discarded work.
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
    """A projection that would orphan a coord the weights align against is left alone, end to end.

    Notes
    -----
    The guard, end to end. ``elevation`` has no ``time``, so projecting to it drops the
    ``time`` *coordinate* -- and these weights carry ``time``, so after the hop they would
    broadcast a fresh ``time`` instead of inner-joining against the dataset's. That changes
    values, not just errors, which is why ``requires`` adds ``weight_dims``.
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
    """A select on a grouped reduce's consumed dim is left, and raises from xarray at replay, not from the optimiser.

    Notes
    -----
    Left rather than raised, so the error comes from xarray at replay in its own words -- not
    an ``InvalidExpressionError`` the optimiser guessed at. ``pushdown_selects`` raises for a
    plain reduce because a reduce always removes what it names; a grouped one mints as well,
    so the same inference would reject valid chains.
    """
    chain = dated_ds.plan.groupby("time.month").mean().isel(time=0)
    with pytest.raises(ValueError, match="Dimensions"):
        chain.collect()


def test_builder_only_method_records_and_replays(ds):
    """A method that exists only on the builder (``first``) records inside a context and replays to the eager result.

    Notes
    -----
    ``first`` is a ``DatasetGroupBy`` method that does not exist on Dataset, so before the
    barrier it fell to the property branch and collected eagerly -- onto a ``DatasetGroupBy``
    with no ``.compute()``. In context the callable check is skipped and it records.
    """
    chain = ds.plan.groupby("lat").first()
    assert isinstance(chain, LazyProxy)
    assert_equal(chain.collect(), ds.groupby("lat").first())


def test_getitem_in_context_selects_a_group_not_a_variable(ds):
    """``__getitem__`` inside a ``groupby`` context selects a group, so it records ``Opaque``, not a ``Project``.

    Notes
    -----
    ``DatasetGroupBy.__getitem__`` selects a *group*; classifying it as a ``Project`` would
    let projection pushdown treat the key as a variable name.
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
    ``lat`` is disjoint from the window, so the rolling mean runs over one latitude instead
    of three.
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


def test_rolling_closer_that_looks_like_a_dim_spec_still_matches_eager(ds):
    """A rolling closer whose positional arg looks like a dim name refuses to fuse and still matches eager.

    Notes
    -----
    ``DatasetRolling.mean`` takes no dim argument, so this passes ``"lat"`` as *keep_attrs*
    and reduces nothing. Lowering refuses to fuse it, and the verbatim pair replays as xarray
    reads it -- which is the whole point of refusing.
    """
    chain = ds.plan.rolling(time=2).mean("lat")
    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
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
    """An opener deliberately left unfused (``rolling_exp``, ``cumulative``) still replays verbatim to match eager.

    Notes
    -----
    The other half of a refusal to fuse: it has to *work*. These replay verbatim, so the pair
    stays opaque and the ops around it are unaffected.
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
    A bare closer really does reduce every dim here (unlike a *grouped* bare closer, which
    reduces only along the group dim) -- ``DatasetWeighted.mean`` takes ``dim`` and
    ``dim=None`` means all of them, so the recorded ``ALL_DIMS`` carries over untouched.
    """
    chain = ds.plan.weighted(ds["area"]).mean()
    assert [type(n) for n in chain._optimized()] == [WeightedReduce]
    assert_equal(chain.collect(), ds.weighted(ds["area"]).mean())


def test_select_after_a_weighted_reduce_does_not_move(ds):
    """A select after a ``WeightedReduce`` does not move, and the plan replays in the order written.

    Notes
    -----
    The rewrite the variant exists to block, checked end to end: pushing the ``isel`` left
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
    Lowering opaques only the *pair*, so the trailing reduce is a real ``Reduce`` again --
    the half of the barrier's retirement that ``weighted`` had been waiting for.
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
    The other half of a refusal to fuse: the verbatim replay has to *work*. These are not
    tabulated reductions, so they record ``Opaque`` and the pair stays opaque.
    """
    weights = ds["area"]
    chain = getattr(ds.plan.weighted(weights), closer)()
    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), getattr(ds.weighted(weights), closer)())


def test_weights_carrying_a_dim_the_dataset_lacks_match_eager(ds):
    """Weights carrying a dim the dataset lacks broadcast that dim onto every variable, and the plan still matches eager.

    Notes
    -----
    ``dot`` broadcasts, so this *adds* ``member`` to every variable -- the fact
    ``weight_dims`` exists for. Nothing moves across the node, so replay is verbatim; what is
    being checked is that the modelled schema did not mislead a later rule.
    """
    weights = xr.DataArray([1.0, 2.0], dims="member", coords={"member": [0, 1]})
    chain = ds.plan.weighted(weights).mean("lat").mean("member")
    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Reduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").mean("member"))


def test_terminal_after_context_matches_eager(ds):
    """A terminal call after a ``groupby`` context still collects, rather than being recorded into the barrier.

    Notes
    -----
    ``_EAGER_ATTRS`` is checked before the context branch, so terminals still collect rather
    than being recorded and never run.
    """
    result = ds.plan.groupby("time").mean().to_dataframe()

    assert not isinstance(result, LazyProxy)
    assert result.equals(ds.groupby("time").mean().to_dataframe())


def test_plans_without_a_context_still_optimise(ds):
    """An ordinary chain with no ``groupby`` context is still rewritten; the barrier must not leak."""
    chain = ds.plan.mean("lat").isel(time=0)
    assert [n.name for n in chain._optimized()] == ["isel", "mean"]


def test_terminal_to_dataframe_triggers_collect_and_delegates(ds):
    """``.to_dataframe()`` triggers collect and delegates, rather than returning a recorded-but-never-run proxy.

    Notes
    -----
    A terminal like ``.to_dataframe()`` must materialise rather than return a proxy
    (recorded, never run) -- proving the rewrite ran and delegation reached xarray.
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
    """``.plot`` delegates to the materialised DataArray's plot accessor, not a recording wrapper.

    Notes
    -----
    ``.plot`` is xarray's plot accessor off the *materialised* DataArray, not a recording
    wrapper -- so calling it draws instead of returning a proxy. Delegating the whole
    ``plot`` attribute (not just calling it) also reaches the accessor's own methods, e.g.
    ``plot.line`` on a 1-D selection.
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
# ``.plan`` is registered on both types and one class serves both, so what follows is
# not a second suite: it pins the two places a ``DataArray`` differs (``__getitem__``
# means indexing, and there are no ``data_vars`` to project) and checks that everything
# else -- recording, lowering, the rewrite rules, replay -- carries over unchanged.


@pytest.fixture
def da(ds) -> xr.DataArray:
    """The shared fixture's ``temperature`` variable, as a standalone DataArray.

    Returns
    -------
    xarray.DataArray
        ``temperature(time, lat, lon)`` with its coords, the ``area`` aux coord included.
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
    """The optimiser rewrites a ``DataArray`` plan; it is not a verbatim passthrough.

    Notes
    -----
    The narrowing a ``DataArray`` brings is confined to the *projection* rules (there are
    no ``data_vars`` for them to reason about). Select pushdown reasons about dims, which
    a ``DataArray`` has, so it fires exactly as it does on a ``Dataset``.
    """
    chain = da.plan.mean("lat").isel(time=0)
    assert [n.name for n in chain._optimized()] == ["isel", "mean"]


def test_dataarray_getitem_records_opaque_not_a_projection(da):
    """``da.plan["lat"]`` records an ``Opaque``: ``__getitem__`` on a ``DataArray`` is indexing.

    Notes
    -----
    The key *looks* like a projection to :func:`~xrexpr.schema.to_opnode`, which sees only
    the call -- the receiver is what settles it, exactly as it does for a builder's
    ``__getitem__``.
    """
    node = da.plan["lat"]._ops[0]
    assert isinstance(node, Opaque) and node.name == "__getitem__"
    assert_equal(da.plan["lat"].collect(), da["lat"])


def test_positional_getitem_pairs_on_a_dataarray_are_not_merged(da):
    """Two positional ``__getitem__``s stay two ops, so the rows selected are the eager ones.

    Notes
    -----
    The bug the ``Opaque`` classification exists to prevent, and the keys are the ones that
    trigger it rather than an arbitrary pair. ``da[[1, 2]]`` selects *positions along the
    first dim*, so as a pair of ``Project`` nodes ``da[[1, 2]][[1]]`` satisfies
    ``merge_adjacent_projects``' subset test -- the rule is deliberately schema-free,
    deciding on the two key sets alone -- and collapses to ``da[[1]]``: row 1, where the
    eager chain indexes *within* the selected pair and gives row 2. Verified by classifying
    the pair as projections on purpose: the merged plan returns the wrong row.
    """
    chain = da.plan[[1, 2]][[1]]

    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), da[[1, 2]][[1]])


def test_a_coordless_dataarray_still_replays_equal_to_eager(da):
    """A ``DataArray`` with no coords models no dims at all, and every chain still matches eager.

    Notes
    -----
    ``SchemaState.from_dataset`` reads a ``DataArray``'s *coords*, so an array without them
    yields an empty schema -- the model knows of no dims. That narrows in the safe
    direction: a rule that needs a fact it hasn't got refuses. ``groupby("time")`` here
    fails ``lower._grouper_dims``' dim check and lands on the opaque fallback rather than
    being fused, which costs a rewrite and no correctness.
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
    """A chain that *becomes* a DataArray records its later ``__getitem__``s as indexing, not projection.

    Notes
    -----
    The base is a ``Dataset`` here, so the classification cannot be read off the base
    alone: after a bare-name projection the live object is a ``DataArray``, and
    ``ds["temperature"][[1, 2]][[1]]`` is the same wrong-row collapse as on a
    ``DataArray`` base -- one op later, and reachable before this accessor existed.
    ``merge_adjacent_projects``' barrier on a single ``p1`` stops the *first* pair, not
    this one.
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
    """An ``Opaque`` may return a ``DataArray``, so the ``__getitem__``s after it index too.

    Notes
    -----
    ``Opaque`` is the node the IR does not model, and that covers its *result type*: both
    ``pipe`` and ``get`` take a ``Dataset`` to a ``DataArray`` without a modelled node in
    sight. Classified on the key alone the pair would be two ``Project``s, and
    ``merge_adjacent_projects`` -- which is licensed to fire past an ``Opaque`` -- would
    collapse ``[[1, 2]][[1]]`` to ``[[1]]``. That returns a valid array of the *wrong rows*
    rather than raising, so the equality assertion is the one that matters here.
    """
    chain = escape(ds.plan)[[1, 2]][[1]]

    assert [type(n) for n in chain._ops] == [Opaque, Opaque, Opaque]
    assert_equal(chain.collect(), escape(ds)[[1, 2]][[1]])


def test_getitem_after_a_dataset_valued_opaque_is_demoted_too(ds):
    """An opaque that *does* return a Dataset demotes what follows it anyway, losing one merge.

    Notes
    -----
    The acknowledged cost of reading ``Opaque`` as receiver-unknown, pinned as intended
    rather than left to surprise. ``rename`` returns a ``Dataset``, so these two list-form
    projections would merge safely -- but that is only knowable from the opaque's *name*,
    and a table of Dataset-preserving names would rot as xarray grows. Projection pushdown
    loses nothing, since :func:`~xrexpr.optimize._trusted_prefix` already stops it at the
    first ``Opaque``; ``merge_adjacent_projects`` is the whole of what is forfeited.
    """
    chain = ds.plan.rename({"temperature": "t2m"})[["t2m", "elevation"]][["t2m"]]

    assert [type(n) for n in chain._ops] == [Opaque, Opaque, Opaque]
    assert_equal(
        chain.collect(),
        ds.rename({"temperature": "t2m"})[["t2m", "elevation"]][["t2m"]],
    )


def test_a_list_form_projection_keeps_the_chain_in_dataset_land(ds):
    """A list-form projection yields a Dataset, so later ``__getitem__``s still record projections.

    Notes
    -----
    The two spellings are told apart by the *key*, not by the node's ``variables``, which
    reads ``("temperature",)`` for both -- so this is the case that would be lost by
    demoting on the node alone, and the projection rules would stop firing on ordinary
    ``ds[[...]][[...]]`` chains.
    """
    chain = ds.plan[["temperature", "elevation"]][["temperature"]]

    assert [type(n) for n in chain._ops] == [Project, Project]
    assert_equal(chain.collect(), ds[["temperature", "elevation"]][["temperature"]])
