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
from xrexpr.accessor import _EAGER_ATTRS, Explanation, LazyDatasetProxy
from xrexpr.exceptions import InvalidExpressionError
from xrexpr.ir import (
    ALL_DIMS,
    ContextOpen,
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


# --- recording builds Op variants ----------------------------------------------


def test_record_builds_op_variants(ds):
    from xrexpr.ir import Op, Reduce, Select

    ops = ds.plan.mean("lat").isel(time=0)._ops
    assert all(isinstance(n, Op) for n in ops)
    mean_node, isel_node = ops
    assert isinstance(mean_node, Reduce) and mean_node.consumes == frozenset({"lat"})
    assert isinstance(isel_node, Select) and isel_node.consumes == frozenset({"time"})


def test_recording_holds_no_schema_and_the_optimiser_folds_it(ds):
    # Recording is a pure append -- the proxy carries the base dataset and the plan and
    # nothing else. The single logical-schema fold belongs to the optimiser, which plans
    # from the front and so is the reader whose fold is exact.
    chain = ds.plan.mean("lat").isel(time=0)
    assert not hasattr(chain, "_schema")

    entering = _schemas(chain._ops, chain._base_schema())
    assert [dict(s.dims) for s in entering] == [
        {"time": 4, "lat": 3, "lon": 5},  # what mean("lat") sees
        {"time": 4, "lon": 5},  # what isel(time=0) sees
    ]


def test_bare_reduce_records_all_dims_symbolically(ds):
    # No dim named -> no dim set frozen in at record time. The expansion happens against
    # the optimiser's fold, so it can't bake in names an unmodelled op has since changed.
    node = ds.plan.mean()._ops[0]
    assert node.consumes is ALL_DIMS
    assert node.args == () and node.kwargs == frozendict()  # replays as bare mean()


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
    return proxy._replay(emit(proxy._optimized()))


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


# --- grouped / windowed contexts -----------------------------------------------------
# ``groupby``/``resample``/``rolling``/... return a builder, not a Dataset, so calls after
# them mean something different from the same-named Dataset ops. The accessor barriers
# them: everything from the context op on records as ``Opaque``, which no rewrite rule can
# fire on or across -- correct but unoptimised, until ``docs/roadmap/02-lowering.md``.


@pytest.fixture
def dated_ds() -> xr.Dataset:
    # ``ds`` has an integer ``time`` coord; ``resample`` needs a real datetime index.
    # ``lon`` and the ``area`` aux coord are here for the fused-reduce pushdown tests: a
    # third dim gives the select somewhere disjoint to go, and a non-dimension coord
    # spanning the selected dims checks the reorder carries it along.
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
    # Valid *eagerly* -- ``groupby('time').mean()`` reduces within groups, leaving ``lat``
    # for the later ``isel(lat=0)``. Before the barrier the proxy recorded the grouped
    # ``mean()`` as a full-dataset reduce (no dim -> every current dim), so its schema
    # dropped ``lat`` and ``collect()`` raised InvalidExpressionError.
    chain = ds.plan.groupby("time").mean().isel(lat=0)
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(lat=0))


def test_select_after_grouped_reduce_matches_eager(ds):
    # The other failure mode: a select on a dim *disjoint* from the bogus ``consumes`` was
    # silently swapped behind the reduce, and replay then looked ``isel`` up on the
    # DatasetGroupBy. Barriered, it replays in the order written.
    chain = ds.plan.groupby("time").mean().isel(time=0)
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(time=0))


def test_context_chain_records_an_opener_then_lowers_to_one_node(ds):
    chain = ds.plan.groupby("time").mean().isel(time=0)

    # recorded: the opener is typed, the closer provisionally as an ordinary reduce
    assert [type(op) for op in chain._ops] == [ContextOpen, Reduce, Select]

    # lowered: the pair becomes one node, and no ContextOpen survives
    lowered = chain._optimized()
    assert [type(n) for n in lowered] == [GroupedReduce, Select]
    assert not any(isinstance(n, ContextOpen) for n in lowered)

    # ... and it still replays as the two calls that were written, in order
    assert chain.explain() == Explanation(
        "plan (3 ops):\n  1. groupby('time')\n  2. mean()\n  3. isel(time=0)"
    )


def test_ops_after_an_unfusable_context_are_modelled_again(ds):
    # The trailing barrier is retired. ``first`` closes the context and returns a Dataset,
    # so the ``mean`` after it is an ordinary Dataset reduce -- a chain the barrier
    # rendered permanently opaque from the groupby onward.
    chain = ds.plan.groupby("lat").first().mean("time")

    lowered = chain._optimized()
    assert [type(n) for n in lowered] == [Opaque, Opaque, Reduce]
    assert_equal(chain.collect(), ds.groupby("lat").first().mean("time"))


def test_select_on_a_grouped_nodes_own_dim_stays_put(ds):
    # ``groupby("time")`` mints the dim it grouped over, so ``isel(time=0)`` indexes the
    # *minted* ``time`` -- valid eagerly, and immovable. Pinned end to end because it is
    # the case W2 PR 6's rule must decline: hopping this in front would index the original
    # ``time`` instead of the group labels. (Before that rule existed this test passed
    # because no rule touched a fused node at all; now it passes for the right reason.)
    chain = ds.plan.groupby("time").mean().isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["groupby", "mean", "isel"]
    assert_equal(chain.collect(), ds.groupby("time").mean().isel(time=0))


def test_climatology_select_is_pushed_in_front_of_the_grouping(dated_ds):
    # The payoff the whole lowering workstream was built for: the grouping runs over one
    # latitude instead of over all of them and then discarding the rest. Both halves are
    # asserted -- that the rewrite *fires*, so this cannot pass vacuously, and that it
    # still equals eager.
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
    # Whether the select moved or not, the answer must not change -- which is the only
    # thing that makes "hop when disjoint, leave otherwise" worth anything.
    assert_equal(chain(dated_ds.plan).collect(), chain(dated_ds))


def test_projection_is_pushed_in_front_of_the_grouping(dated_ds):
    # W2 PR 7: aggregate one variable instead of two and then discarding one.
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
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    assert_equal(chain(ds.plan).collect(), chain(ds))


def test_projection_past_a_windowed_reduce_is_left_when_the_dim_would_go(dated_ds):
    # Pinned end to end because the failure would be loud rather than wrong: projecting
    # first drops the ``time`` coord (no surviving variable uses it) and ``rolling``
    # raises ``Window dimensions ('time',) not found``. The rule must not create that.
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
    ``docs/roadmap/07-small-wins.md`` §8, which settles this as a deliberate property
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
    # This test used to source that failure from a float variable lacking ``time``: a
    # Dataset reduce hands such a variable an empty axis set, and ``std``/``var`` blow up on
    # it. That looks like the natural choice and it is a trap, because **the blow-up is a
    # numbagg bug, not xarray behaviour** -- numbagg reads ``axis=()`` as ``axis=None`` and
    # returns a scalar (07-small-wins.md §8 has the full account). Pinning it meant this
    # test asserted that a third-party bug was *present*: it failed under
    # ``use_numbagg=False``, and it would fail again the day numbagg fixed it -- in both
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
    # The other half of §8's contract: skipping a discarded variable's failure is allowed,
    # inventing one is not. ``mean`` skips a variable lacking the dim rather than refusing,
    # so both orders succeed and must agree exactly.
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    assert_equal(
        ds.plan.mean(dim=["time"])[["temperature"]].collect(),
        ds.mean(dim=["time"])[["temperature"]],
    )


def test_projection_past_a_weighted_reduce_is_left(dated_ds):
    # Pins the *current* behaviour rather than defending it. ``WeightedReduce`` is absent
    # from this rule's match, so nothing crosses it and both orders raise alike. The
    # argument that originally justified the exclusion -- that hopping the projection in
    # front would mask ``elevation``'s error -- has since been retired: by the test above
    # and ``07-small-wins.md`` §8, skipping a discarded variable's failure is the *better*
    # answer. Whether weighted should now join the rule is an open decision recorded in §8;
    # a projection, unlike a select, never has to subset the weights.
    ds = dated_ds.assign(elevation=(("lat", "lon"), np.zeros((3, 4))))
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean("time")[["temperature"]]

    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Project]
    with pytest.raises(ValueError, match="do not exist"):
        chain.collect()
    with pytest.raises(ValueError, match="do not exist"):
        ds.weighted(weights).mean("time")[["temperature"]]


def test_select_on_a_consumed_group_dim_still_raises_from_xarray(dated_ds):
    # Left rather than raised, so the error comes from xarray at replay in its own words
    # -- not an ``InvalidExpressionError`` the optimiser guessed at. ``pushdown_selects``
    # raises for a plain reduce because a reduce always removes what it names; a grouped
    # one mints as well, so the same inference would reject valid chains.
    chain = dated_ds.plan.groupby("time.month").mean().isel(time=0)
    with pytest.raises(ValueError, match="Dimensions"):
        chain.collect()


def test_builder_only_method_records_and_replays(ds):
    # ``first`` is a DatasetGroupBy method that does not exist on Dataset, so before the
    # barrier it fell to the property branch and collected eagerly -- onto a DatasetGroupBy
    # with no ``.compute()``. In context the callable check is skipped and it records.
    chain = ds.plan.groupby("lat").first()
    assert isinstance(chain, LazyDatasetProxy)
    assert_equal(chain.collect(), ds.groupby("lat").first())


def test_getitem_in_context_selects_a_group_not_a_variable(ds):
    # ``DatasetGroupBy.__getitem__`` selects a *group*; classifying it as a Project would
    # let projection pushdown treat the key as a variable name.
    chain = ds.plan.groupby("lat")[0]
    assert isinstance(chain._ops[-1], Opaque)
    assert_equal(chain.collect(), ds.groupby("lat")[0])


def test_resample_matches_eager(dated_ds):
    chain = dated_ds.plan.resample(time="2D").mean()
    assert_equal(chain.collect(), dated_ds.resample(time="2D").mean())


def test_rolling_then_select_matches_eager(ds):
    # The select now hops in front: ``lat`` is disjoint from the window, so the rolling
    # mean runs over one latitude instead of three. Expectation flipped deliberately in
    # W2 PR 6 -- it read ``[WindowedReduce, Select]`` while the fused nodes had no rules.
    chain = ds.plan.rolling(time=2).mean().isel(lat=0)
    assert [type(n) for n in chain._optimized()] == [Select, WindowedReduce]
    assert_equal(chain.collect(), ds.rolling(time=2).mean().isel(lat=0))


@pytest.mark.parametrize("boundary", ["trim", "pad"])
def test_coarsen_matches_eager(ds, boundary):
    chain = ds.plan.coarsen(time=2, boundary=boundary).mean().isel(lat=0)
    assert_equal(
        chain.collect(), ds.coarsen(time=2, boundary=boundary).mean().isel(lat=0)
    )


def test_rolling_closer_that_looks_like_a_dim_spec_still_matches_eager(ds):
    # ``DatasetRolling.mean`` takes no dim argument, so this passes "lat" as *keep_attrs*
    # and reduces nothing. Lowering refuses to fuse it, and the verbatim pair replays as
    # xarray reads it -- which is the whole point of refusing.
    chain = ds.plan.rolling(time=2).mean("lat")
    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), ds.rolling(time=2).mean("lat"))


@pytest.mark.parametrize(
    "chain",
    [
        # an exponential weighting, not a fixed window -- ``window`` cannot describe it
        lambda o: o.rolling_exp(time=3).mean(),
        lambda o: o.rolling_exp(time=3).mean().isel(lat=0),
        # a scan wearing a builder's clothes
        lambda o: o.cumulative("time").sum(),
    ],
)
def test_deliberately_unfused_openers_still_match_eager(ds, chain):
    # The other half of a refusal to fuse: it has to *work*. These replay verbatim, so
    # the pair stays opaque and the ops around it are unaffected.
    plan = chain(ds.plan)
    assert [type(n) for n in plan._optimized()[:2]] == [Opaque, Opaque]
    assert_equal(plan.collect(), chain(ds))


def test_ops_after_a_windowed_reduce_are_modelled(ds):
    chain = ds.plan.rolling(time=2).mean().mean("time")
    assert [type(n) for n in chain._optimized()] == [WindowedReduce, Reduce]
    assert_equal(chain.collect(), ds.rolling(time=2).mean().mean("time"))


def test_weighted_matches_eager(ds):
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean(("lat", "lon"))
    assert [type(n) for n in chain._optimized()] == [WeightedReduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean(("lat", "lon")))


def test_weighted_bare_closer_matches_eager(ds):
    # A bare closer really does reduce every dim here (unlike a *grouped* bare closer,
    # which reduces only along the group dim) -- ``DatasetWeighted.mean`` takes ``dim`` and
    # ``dim=None`` means all of them, so the recorded ALL_DIMS carries over untouched.
    chain = ds.plan.weighted(ds["area"]).mean()
    assert [type(n) for n in chain._optimized()] == [WeightedReduce]
    assert_equal(chain.collect(), ds.weighted(ds["area"]).mean())


def test_select_after_a_weighted_reduce_does_not_move(ds):
    # The rewrite the variant exists to block, checked end to end: pushing the ``isel``
    # left would leave the weights un-subset. The plan replays in the order written.
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean("lat").isel(time=0)
    assert [c.name for c in emit(chain._optimized())] == ["weighted", "mean", "isel"]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").isel(time=0))


def test_ops_after_a_weighted_reduce_are_modelled(ds):
    # Lowering opaques only the *pair*, so the trailing reduce is a real ``Reduce`` again --
    # the half of the barrier's retirement that ``weighted`` had been waiting for.
    weights = ds["area"]
    chain = ds.plan.weighted(weights).mean("lat").mean("time")
    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Reduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").mean("time"))


@pytest.mark.parametrize("closer", ["sum_of_weights", "sum_of_squares"])
def test_weighted_only_closers_still_match_eager(ds, closer):
    # The other half of a refusal to fuse: the verbatim replay has to *work*. These are not
    # tabulated reductions, so they record ``Opaque`` and the pair stays opaque.
    weights = ds["area"]
    chain = getattr(ds.plan.weighted(weights), closer)()
    assert [type(n) for n in chain._optimized()] == [Opaque, Opaque]
    assert_equal(chain.collect(), getattr(ds.weighted(weights), closer)())


def test_weights_carrying_a_dim_the_dataset_lacks_match_eager(ds):
    # ``dot`` broadcasts, so this *adds* ``member`` to every variable -- the fact
    # ``weight_dims`` exists for. Nothing moves across the node, so replay is verbatim;
    # what is being checked is that the modelled schema did not mislead a later rule.
    weights = xr.DataArray([1.0, 2.0], dims="member", coords={"member": [0, 1]})
    chain = ds.plan.weighted(weights).mean("lat").mean("member")
    assert [type(n) for n in chain._optimized()] == [WeightedReduce, Reduce]
    assert_equal(chain.collect(), ds.weighted(weights).mean("lat").mean("member"))


def test_terminal_after_context_matches_eager(ds):
    # ``_EAGER_ATTRS`` is checked before the context branch, so terminals still collect
    # rather than being recorded into the barrier and never run.
    result = ds.plan.groupby("time").mean().to_dataframe()

    assert not isinstance(result, LazyDatasetProxy)
    assert result.equals(ds.groupby("time").mean().to_dataframe())


def test_plans_without_a_context_still_optimise(ds):
    # The barrier must not leak: an ordinary chain is still rewritten.
    chain = ds.plan.mean("lat").isel(time=0)
    assert [n.name for n in chain._optimized()] == ["isel", "mean"]


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
