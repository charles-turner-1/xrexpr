"""Tests for the logical schema layer.

``SchemaState`` is snapshotted from a real dataset and then evolved by
``apply_schema`` using only :data:`~xrexpr.ir.Op` metadata — no array data is touched.
The nodes here are built by hand, which also documents the contract ``apply_schema``
relies on: a scalar select drops its dim (``Select.consumes``, derived from ``indexer``);
a non-scalar select leaves the dim in ``indexer`` only.

``variables`` (every variable, coordinate or not, → the dims it spans) is the store, and
``dims``/``data_vars``/``coords`` derive from it. So a schema **cannot** be built from dim
sizes alone — a dim exists only because some variable spans it, and sizes for unspanned
dims are pruned away. :func:`_dim_coords_only` builds the minimal schema with a given set
of dims for the tests that care about sizes rather than variables.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrexpr.ir import (
    ALL_DIMS,
    AllDims,
    GroupedReduce,
    Project,
    Reduce,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.schema import SchemaState, apply_schema


def _dim_coords_only(sizes, extra_coords=()):
    """Build the minimal schema with dims ``sizes``: one dim coordinate each, no data vars.

    Parameters
    ----------
    sizes : Mapping
        The dims and their extents, ``None`` for an unknown one.
    extra_coords : iterable of Hashable, optional
        Scalar (0-d) coordinate names, as ``ds.assign_coords(ref=1.0)`` would add.

    Returns
    -------
    SchemaState
        Equivalent to ``xr.Dataset(coords={d: range(n) for d, n in sizes.items()})``.

    Notes
    -----
    Spelled out because ``variables`` is the store: ``SchemaState(dims=sizes)`` alone would
    derive *no* dims and prune every size away, which is the phantom-dim state the shape
    forbids.
    """
    return SchemaState(
        variables={d: (d,) for d in sizes} | {c: () for c in extra_coords},
        coord_names=frozenset(sizes) | frozenset(extra_coords),
        dims=sizes,
    )


def test_from_dataset_snapshots_dims_and_coords(ds):
    """A snapshot of the canonical dataset reports its dims, sizes and coordinate names."""
    schema = SchemaState.from_dataset(ds)
    assert schema.dims == {"time": 4, "lat": 3, "lon": 5}
    assert schema.coords == {"time", "lat", "lon"}
    assert schema.dim_names == {"time", "lat", "lon"}


def test_from_dataarray_snapshots_dims(ds):
    """A ``DataArray`` snapshots its dims and sizes just as a ``Dataset`` does."""
    schema = SchemaState.from_dataset(ds["temperature"])
    assert schema.dims == {"time": 4, "lat": 3, "lon": 5}


def test_schema_is_immutable():
    """A snapshot rejects both item assignment inside ``dims`` and field assignment."""
    schema = _dim_coords_only({"time": 4})
    with pytest.raises(TypeError):
        schema.dims["time"] = 1
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        schema.dims = {}


def test_reduce_removes_dim_and_its_coord(ds):
    """An aggregating reduce removes the dim it names *and* the coordinate over it."""
    schema = SchemaState.from_dataset(ds)
    node = Reduce(name="mean", args=("lat",), consumes=["lat"])
    after = apply_schema(schema, node)
    assert after.dims == {"time": 4, "lon": 5}
    assert after.coords == {"time", "lon"}


def test_bare_reduce_removes_every_dim_and_coord(ds):
    """A bare reduce clears every dim, every coordinate and every variable's dims.

    Notes
    -----
    ``apply_schema`` is where the deferred ``ALL_DIMS`` expansion is finally cashed in,
    against a schema that is exact rather than the recorder's guess.
    """
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Reduce(name="mean", consumes=ALL_DIMS))
    assert after.dims == {}
    assert after.coords == set()
    assert all(dims == () for dims in after.data_vars.values())


def test_bare_reduce_expands_against_the_schema_it_is_given(ds):
    """``ALL_DIMS`` resolves against the schema entering the node, not the original dataset.

    Notes
    -----
    "Every dim *at this point*": fold a select in first and the sentinel resolves to what
    is left.
    """
    schema = apply_schema(
        SchemaState.from_dataset(ds), Select(name="isel", indexer={"time": 0})
    )
    assert apply_schema(schema, Reduce(name="mean", consumes=ALL_DIMS)).dims == {}


def test_scalar_isel_removes_dim_but_keeps_the_coord(ds):
    """``ds.isel(time=0)`` — the dim goes, the coordinate stays, demoted to 0-d.

    .. code-block:: python

        ds.isel(time=0).sizes    # {'lat': 3, 'lon': 5}      -- time is gone
        ds.isel(time=0).coords   # time, lat, lon            -- time survives, scalar

    The coordinate half was previously asserted the other way round, matching a schema
    that dropped any coordinate sharing a name with a removed dim because a name was all
    it had. Now that a coordinate carries dims it can lose ``time`` and remain, which is
    what xarray does (verified against 2026.7.0).
    """
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"time": 0})  # scalar -> consumes={"time"}
    after = apply_schema(schema, node)
    assert after.dims == {"lat": 3, "lon": 5}
    assert "time" in after.coords
    assert after.variables["time"] == ()  # 0-d: it lost its dim, not its existence


def test_scalar_isel_with_drop_removes_the_coord_too(ds):
    """``ds.isel(time=0, drop=True)`` — the dim *and* the coordinate go.

    .. code-block:: python

        ds.isel(time=0, drop=True).coords   # lat, lon  -- no time

    ``drop`` is the whole difference between the indexing and aggregating coordinate
    rules, and the only thing the ``Select`` arm consults to tell them apart.
    """
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", kwargs={"drop": True}, indexer={"time": 0})
    after = apply_schema(schema, node)
    assert after.dims == {"lat": 3, "lon": 5}
    assert "time" not in after.coords
    assert "time" not in after.variables


def test_scalar_sel_removes_dim(ds):
    """A scalar ``sel`` label removes its dim, as a scalar ``isel`` position does."""
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"lat": 1})
    after = apply_schema(schema, node)
    assert after.dims == {"time": 4, "lon": 5}


def test_slice_isel_resizes_kept_dim(ds):
    """A slice select keeps its dim at a new size, and keeps the coordinate with it."""
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"time": slice(0, 2)})
    after = apply_schema(schema, node)
    assert after.dims == {"time": 2, "lat": 3, "lon": 5}
    assert after.coords == {"time", "lat", "lon"}  # dim kept -> coord kept


def test_list_isel_resizes_kept_dim(ds):
    """A list select sizes its dim by the number of positions named."""
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"lon": [0, 2, 4]})
    after = apply_schema(schema, node)
    assert after.dims["lon"] == 3


def test_boolean_array_isel_resizes_by_true_count(ds):
    """A boolean-array mask sizes its dim by the number of ``True`` flags."""
    schema = SchemaState.from_dataset(ds)
    mask = np.array([True, False, True, True])
    node = Select(name="isel", indexer={"time": mask})
    after = apply_schema(schema, node)
    assert after.dims["time"] == 3


def test_integer_array_isel_resizes_by_length(ds):
    """An integer-array select sizes its dim by the array's length."""
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"lon": np.array([0, 1])})
    after = apply_schema(schema, node)
    assert after.dims["lon"] == 2


def test_boolean_list_isel_resizes_by_true_count(ds):
    """A boolean *list* sizes its dim like the array spelling — by ``True`` count."""
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"time": [True, False, True, True]})
    after = apply_schema(schema, node)
    assert after.dims["time"] == 3


def test_unsizable_sel_slice_is_unknown(ds):
    """A label slice leaves its dim present but of unknown extent.

    Notes
    -----
    It needs coord values to size, which this layer does not read — so the honest answer
    is "don't know", not a guess that keeps the current size.
    """
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"time": slice("a", "z")})
    after = apply_schema(schema, node)
    assert after.dims["time"] is None
    assert "time" in after.dims  # unknown extent, but the dim is still there


def test_integer_labelled_sel_slice_is_unknown_not_under_reported(ds):
    """An integer-bounded ``sel`` slice answers unknown rather than under-reporting a size.

    Notes
    -----
    The sharp case: integer bounds are indistinguishable from positional ones, so
    ``classify`` mints a ``ForwardSlice`` and its size reads them as positions. Against a
    length-4 dim, ``sel(time=slice(20, 30))`` sized as **0** — an under-report, the
    direction a size-driven rule could act on.
    """
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"time": slice(20, 30)})
    after = apply_schema(schema, node)
    assert after.dims["time"] is None


def test_positional_isel_slice_is_still_sized_exactly(ds):
    """An ``isel`` slice keeps its exact size, so the unknown above is confined to ``sel``."""
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Select(name="isel", indexer={"time": slice(1, 3)}))
    assert after.dims["time"] == 2


def test_unknown_size_propagates_through_a_later_select(ds):
    """Selecting on a dim of unknown size leaves it unknown, and never coerces it to zero.

    Notes
    -----
    Unknown in, unknown out — the failure mode the ``var_dims`` docstring warns about for
    the variable-level counterpart. A known size beside it is unaffected.
    """
    unknown = _dim_coords_only({"time": None, "lat": 3})
    after = apply_schema(unknown, Select(name="isel", indexer={"time": slice(0, 2)}))
    assert after.dims["time"] is None
    assert after.dims["time"] != 0
    assert after.dims["lat"] == 3  # a known size beside it is unaffected


def test_scalar_select_still_drops_a_dim_of_unknown_size(ds):
    """A scalar select drops a dim whose size is unknown — dropping needs no size at all.

    Notes
    -----
    The coordinate survives as 0-d, which is a separate rule (see
    ``test_scalar_isel_removes_dim_but_keeps_the_coord``) and is asserted here only so
    this test cannot pass by dropping too much.
    """
    unknown = _dim_coords_only({"time": None, "lat": 3})
    after = apply_schema(unknown, Select(name="isel", indexer={"time": 0}))
    assert "time" not in after.dims
    # The *dim* goes without needing a size; the coordinate survives as 0-d, which is a
    # separate rule (see ``test_scalar_isel_removes_dim_but_keeps_the_coord``) and is
    # asserted here only so this test cannot pass by dropping too much.
    assert after.coords == {"time", "lat"}
    assert after.variables["time"] == ()


# --- grouped reduces -----------------------------------------------------------------
# The arm was derived from what xarray actually does rather than from first principles,
# so it is checked the same way: fold the schema, run the chain, compare. Anything else
# would only re-assert the assumption the arm was written from.


@pytest.fixture
def dated_ds() -> xr.Dataset:
    """A dataset with a real datetime ``time`` coordinate, so groupby components resolve.

    Returns
    -------
    xarray.Dataset
        24 monthly steps of ``temperature(time, lat, lon)`` alongside
        ``elevation(lat, lon)``, which deliberately lacks ``time`` — it still gains the
        minted dim.
    """
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), rng.random((24, 3, 5))),
            # deliberately missing ``time``: it still gains the minted dim
            "elevation": (("lat", "lon"), rng.random((3, 5))),
        },
        coords={
            "time": pd.date_range("2000-01-31", periods=24, freq="ME"),
            "lat": np.arange(3),
            "lon": np.arange(5),
        },
    )


@pytest.mark.parametrize(
    ("calls", "node"),
    [
        (
            [("groupby", ("time.month",)), ("mean", ())],
            GroupedReduce(
                name="groupby", group_dim="time", new_dim="month", reduce="mean"
            ),
        ),
        (
            [("groupby", ("time.month",)), ("mean", (["time", "lat"],))],
            GroupedReduce(
                name="groupby",
                group_dim="time",
                new_dim="month",
                reduce="mean",
                consumes={"lat"},
            ),
        ),
        (
            [("groupby", ("lat",)), ("mean", ())],
            GroupedReduce(
                name="groupby", group_dim="lat", new_dim="lat", reduce="mean"
            ),
        ),
        (
            [("groupby_bins", ("lat", 2)), ("mean", ())],
            GroupedReduce(
                name="groupby_bins", group_dim="lat", new_dim="lat_bins", reduce="mean"
            ),
        ),
    ],
)
def test_grouped_reduce_arm_agrees_with_xarray(dated_ds, calls, node):
    """The folded schema agrees with evaluation on dims, coords and every variable's dims.

    Notes
    -----
    The arm was derived from what xarray actually does rather than from first principles,
    so it is checked the same way: fold the schema, run the chain, compare. Anything else
    would only re-assert the assumption the arm was written from.
    """
    eager = dated_ds
    for name, args in calls:
        eager = getattr(eager, name)(*args)

    after = apply_schema(SchemaState.from_dataset(dated_ds), node)

    assert after.dim_names == frozenset(eager.sizes)
    assert after.coords == frozenset(eager.coords)
    assert {k: set(v) for k, v in after.data_vars.items()} == {
        k: set(v.dims) for k, v in eager.data_vars.items()
    }


def test_grouped_reduce_mints_a_dim_of_unknown_size(dated_ds):
    """The minted dim has unknown extent, while untouched dims keep their known sizes.

    Notes
    -----
    The group count is a fact about coordinate *values*, which this layer does not read —
    exactly the gap ``int | None`` was added for.
    """
    node = GroupedReduce(
        name="groupby", group_dim="time", new_dim="month", reduce="mean"
    )
    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert after.dims["month"] is None
    assert after.dims["lat"] == 3  # untouched dims keep their known sizes


def test_grouped_reduce_adds_the_new_dim_to_a_variable_that_lacked_the_group_dim(
    dated_ds,
):
    """A variable that never carried the group dim still gains the minted one.

    Notes
    -----
    ``elevation(lat, lon)`` comes back as ``(month, lat, lon)``. Verified against xarray
    above; asserted directly here because it is the part a reader would most expect to be
    wrong.
    """
    node = GroupedReduce(
        name="groupby", group_dim="time", new_dim="month", reduce="mean"
    )
    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert set(after.data_vars["elevation"]) == {"month", "lat", "lon"}


# --- windowed reduces ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("calls", "node"),
    [
        (
            [("rolling", (), {"time": 3}), ("mean", (), {})],
            WindowedReduce(name="rolling", reduce="mean", window={"time": 3}),
        ),
        (
            [("rolling", (), {"time": 3, "center": True}), ("mean", (), {})],
            WindowedReduce(
                name="rolling",
                reduce="mean",
                window={"time": 3},
                kwargs={"time": 3, "center": True},
            ),
        ),
        (
            [("coarsen", (), {"time": 2, "boundary": "trim"}), ("mean", (), {})],
            WindowedReduce(
                name="coarsen",
                reduce="mean",
                window={"time": 2},
                kwargs={"time": 2, "boundary": "trim"},
            ),
        ),
        (
            [("coarsen", (), {"time": 2, "boundary": "pad"}), ("mean", (), {})],
            WindowedReduce(
                name="coarsen",
                reduce="mean",
                window={"time": 2},
                kwargs={"time": 2, "boundary": "pad"},
            ),
        ),
    ],
)
def test_windowed_reduce_arm_agrees_with_xarray(dated_ds, calls, node):
    """The folded sizes match evaluation for every rolling and coarsen spelling.

    Notes
    -----
    ``time`` is 24 here, so ``trim`` and ``pad`` agree; the odd-length case below is what
    actually tells the two roundings apart.
    """
    eager = dated_ds
    for name, args, kwargs in calls:
        eager = getattr(eager, name)(*args, **kwargs)

    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert dict(after.dims) == dict(eager.sizes)


@pytest.mark.parametrize(
    ("boundary", "expected"), [("trim", 3), ("pad", 4), ("exact", 3)]
)
def test_coarsen_rounding_follows_the_boundary_kwarg(boundary, expected):
    """Coarsening an odd-length dim rounds according to ``boundary``, matching xarray.

    Notes
    -----
    7 // 2 rounds three ways. ``exact`` would raise at replay rather than produce 3, so
    agreeing with ``trim`` here costs nothing: the plan never gets that far.
    """
    ds = xr.Dataset({"t": ("time", np.arange(7.0))}, coords={"time": np.arange(7)})
    node = WindowedReduce(
        name="coarsen",
        reduce="mean",
        window={"time": 2},
        kwargs={"time": 2, "boundary": boundary},
    )
    after = apply_schema(SchemaState.from_dataset(ds), node)
    assert after.dims["time"] == expected
    if boundary != "exact":  # exact raises on a length it cannot divide
        assert (
            after.dims["time"]
            == ds.coarsen(time=2, boundary=boundary).mean().sizes["time"]
        )


def test_rolling_leaves_every_size_alone(dated_ds):
    """Rolling changes no size and no coordinate — one output position per input position.

    Notes
    -----
    ``center``/``min_periods`` change values, never shape, which is what makes rolling's
    dim algebra simpler than groupby's.
    """
    node = WindowedReduce(name="rolling", reduce="mean", window={"time": 5})
    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert dict(after.dims) == dict(dated_ds.sizes)
    assert after.coords == frozenset(dated_ds.coords)


def test_unrecognised_boundary_marks_the_size_unknown():
    """An unrecognised ``boundary`` answers unknown: a future spelling costs precision only."""
    ds = xr.Dataset({"t": ("time", np.arange(7.0))}, coords={"time": np.arange(7)})
    node = WindowedReduce(
        name="coarsen",
        reduce="mean",
        window={"time": 2},
        kwargs={"time": 2, "boundary": "something-new"},
    )
    assert apply_schema(SchemaState.from_dataset(ds), node).dims["time"] is None


def test_windowing_an_unknown_size_stays_unknown():
    """Coarsening a dim of unknown size leaves it unknown rather than computing on ``None``."""
    unknown = _dim_coords_only({"time": None})
    node = WindowedReduce(
        name="coarsen", reduce="mean", window={"time": 2}, kwargs={"time": 2}
    )
    assert apply_schema(unknown, node).dims["time"] is None


# --- weighted reduces ----------------------------------------------------------------
# Derived from xarray the same way the two arms above were: the weights' dim effect is the
# part of this node that is *not* a plain reduce's, so it is compared against what
# evaluation produces rather than against the assumption the arm was written from.


def _weighted(consumes, weight_dims, weights):
    """Build a ``WeightedReduce`` whose closer header matches what ``consumes`` claims.

    Parameters
    ----------
    consumes : DimSet
        What the closing reduction removes. ``ALL_DIMS`` produces a bare closer.
    weight_dims : frozenset of Hashable
        The dims the weights carry.
    weights : xarray.DataArray
        The weights themselves, kept in the opener's verbatim header.

    Returns
    -------
    WeightedReduce
        The node, spelled so ``emit`` would reproduce the chain being compared against.
    """
    return WeightedReduce(
        name="weighted",
        reduce="mean",
        weight_dims=weight_dims,
        args=(weights,),
        reduce_args=(sorted(consumes),) if not isinstance(consumes, AllDims) else (),
        consumes=consumes,
    )


@pytest.mark.parametrize(
    ("weights", "consumes"),
    [
        # every weight dim consumed: nothing survives to be marked unknown, so the arm is
        # exactly a plain reduce's and the schema stays fully exact
        (lambda ds: ds["lat"] * ds["lon"], frozenset({"lat", "lon"})),
        # a weight dim *survives*: aligned here, but the arm cannot know that
        (lambda ds: ds["lat"] * ds["lon"], frozenset({"lat"})),
        # weights carrying a dim the dataset lacks -- broadcast onto every variable
        (
            lambda ds: xr.DataArray(
                [1.0, 2.0], dims="member", coords={"member": [0, 1]}
            ),
            frozenset({"lat"}),
        ),
        # a bare closer clears everything, weight dims included
        (lambda ds: ds["lat"] * ds["lon"], ALL_DIMS),
    ],
)
def test_weighted_reduce_arm_agrees_with_xarray(ds, weights, consumes):
    """The folded schema agrees with evaluation across every weights/closer combination.

    Notes
    -----
    Derived from xarray the same way the two arms above were: the weights' dim effect is
    the part of this node that is *not* a plain reduce's, so it is compared against what
    evaluation produces rather than against the assumption the arm was written from.
    Coords are asserted as a *subset*: a broadcast weight dim need carry no coord.
    """
    w = weights(ds)
    eager = ds.weighted(w)
    eager = (
        eager.mean() if isinstance(consumes, AllDims) else eager.mean(sorted(consumes))
    )
    node = _weighted(consumes, frozenset(w.dims), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    # dim *names* are exact, which is what every rule reasons about
    assert after.dim_names == frozenset(eager.sizes)
    # coords are only ever a subset: a broadcast weight dim need carry no coord
    assert after.coords <= frozenset(eager.coords)
    assert {k: set(v) for k, v in after.data_vars.items()} == {
        k: set(v.dims) for k, v in eager.data_vars.items()
    }


def test_a_surviving_weight_dim_is_sized_unknown_not_guessed(ds):
    """A weight dim that survives the reduce is present but unsized, not left at its old size.

    Notes
    -----
    The weights align with the dataset, so a shared dim can *shrink* — verified below. An
    extent this layer cannot compute is ``None`` rather than the current size, which would
    be an over-report of a dim a rewrite might act on.
    """
    w = ds["lat"] * ds["lon"]
    node = _weighted(frozenset({"lat"}), frozenset({"lat", "lon"}), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    assert after.dims["lon"] is None  # a weight dim, kept but not sized
    assert after.dims["time"] == 4  # untouched dims keep their known sizes
    assert "lat" not in after.dims


def test_misaligned_weights_really_do_shrink_a_shared_dim(ds):
    """Weights covering only part of a shared dim shrink it, and the schema says unknown.

    Notes
    -----
    The fact the unknown above exists for, pinned against xarray so it is a verified claim
    rather than a defensive guess: ``dot`` aligns, so weights covering two of three
    ``lat`` labels inner-join and the dim comes back shorter.
    """
    w = xr.DataArray([1.0, 2.0], dims="lat", coords={"lat": [0, 1]})
    eager = ds.weighted(w).mean("lon")
    assert eager.sizes["lat"] == 2 < ds.sizes["lat"]

    after = apply_schema(
        SchemaState.from_dataset(ds),
        _weighted(frozenset({"lon"}), frozenset({"lat"}), w),
    )
    assert after.dim_names == frozenset(eager.sizes)
    assert after.dims["lat"] is None


def test_a_broadcast_weight_dim_reaches_a_variable_that_could_not_have_had_it(ds):
    """A weight dim the dataset lacks is broadcast onto every variable, unsized.

    Notes
    -----
    The part a reader would most expect to be wrong: ``elevation`` has no ``member`` and
    no way to acquire one, yet comes back carrying it — because ``dot`` broadcasts.
    Asserted directly as well as via the derivation above.
    """
    w = xr.DataArray([1.0, 2.0], dims="member", coords={"member": [0, 1]})
    node = _weighted(frozenset({"lat"}), frozenset({"member"}), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    assert set(after.data_vars["elevation"]) == {"member", "lon"}
    assert after.dims["member"] is None


def test_several_minted_dims_land_in_a_deterministic_order(ds):
    """Several minted dims arrive sorted by ``str``, so two identical folds compare equal.

    Notes
    -----
    The minted set is a set, and ``SchemaState`` is a *value*: an iteration-order-dependent
    dim tuple would make two snapshots folded from identical inputs unequal between
    processes (set order over strings varies with ``PYTHONHASHSEED``). Dim order carries
    no meaning here, so a stable one is free — but it has to be asserted to stay stable.
    """
    w = xr.DataArray(np.ones((2, 2, 2)), dims=("run", "member", "ens"))
    node = _weighted(frozenset({"lat"}), frozenset(w.dims), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    assert after.data_vars["elevation"][:3] == ("ens", "member", "run")


def test_weighting_an_unknown_size_stays_unknown():
    """A weight dim whose size was already unknown stays unknown after the fold."""
    unknown = _dim_coords_only({"time": None, "lat": 3})
    w = xr.DataArray([1.0], dims="time")
    after = apply_schema(unknown, _weighted(frozenset({"lat"}), frozenset({"time"}), w))
    assert after.dims["time"] is None


def test_scan_leaves_schema_unchanged(ds):
    """A scan keeps every dim and coordinate: it is order-significant, not dim-changing."""
    schema = SchemaState.from_dataset(ds)
    node = Scan(name="cumsum", args=("time",))
    after = apply_schema(schema, node)
    assert after.dims == schema.dims
    assert after.coords == schema.coords


def test_schema_threads_through_a_chain(ds):
    """Two ops in sequence, one aggregating and one indexing — and they differ on coords.

    .. code-block:: python

        ds.mean("lat").isel(time=0).sizes    # {'lon': 5}
        ds.mean("lat").isel(time=0).coords   # lon, time

    Verified against xarray 2026.7.0, and the pair is worth having in one test: ``lat``'s
    coordinate is **dropped** by the aggregation, while ``time``'s survives the scalar
    select as a 0-d coordinate. Same dim removed either way, opposite coordinate outcome —
    the distinction only a coordinate carrying dims can express.
    """
    schema = SchemaState.from_dataset(ds)
    schema = apply_schema(schema, Reduce(name="mean", args=("lat",), consumes=["lat"]))
    schema = apply_schema(schema, Select(name="isel", indexer={"time": 0}))
    assert schema.dims == {"lon": 5}
    assert schema.coords == {"lon", "time"}
    assert schema.variables["time"] == ()  # survived, scalar
    assert "lat" not in schema.variables  # aggregated away, coordinate and all


def test_from_dataset_snapshots_variable_dims(ds):
    """A snapshot records each data variable's dims, which is what variable-level rules read."""
    schema = SchemaState.from_dataset(ds)
    assert schema.data_vars == {
        "temperature": ("time", "lat", "lon"),
        "elevation": ("lat", "lon"),
    }


def test_from_dataarray_has_no_data_vars(ds):
    """A ``DataArray`` has no data variables, so there is nothing to project over."""
    assert SchemaState.from_dataset(ds["temperature"]).data_vars == {}


def test_reduce_strips_the_dim_from_every_variable(ds):
    """A reduce removes its dim from every variable that carried it, not only from ``dims``."""
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Reduce(name="mean", consumes=["lat"]))
    assert after.data_vars == {"temperature": ("time", "lon"), "elevation": ("lon",)}


def test_scalar_select_strips_the_dim_from_every_variable(ds):
    """A scalar select removes its dim from every variable, as a reduce does."""
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Select(name="isel", indexer={"time": 0}))
    assert after.data_vars == {
        "temperature": ("lat", "lon"),
        "elevation": ("lat", "lon"),
    }


def test_project_drops_the_dims_the_survivors_no_longer_span(ds):
    """Projecting a variable that lacks ``time`` orphans ``time``, which xarray drops outright.

    Notes
    -----
    The tempting expectation is ``after.dims == schema.dims`` — that a projection narrows
    only the variables. It doesn't. Checked against evaluation, which is what caught the
    wrong version of this assertion.
    """
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("elevation",)))
    eager = ds[["elevation"]]

    assert after.data_vars == {"elevation": ("lat", "lon")}
    assert after.dim_names == frozenset(eager.sizes) == {"lat", "lon"}
    assert after.coords <= frozenset(eager.coords)


def test_project_keeping_every_dim_leaves_the_schema_alone(ds):
    """Projecting a variable that spans every dim orphans nothing, so dims and coords stand."""
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(
        schema, Project(name="__getitem__", variables=("temperature",))
    )
    assert after.dims == schema.dims
    assert after.coords == schema.coords


def test_project_of_an_unknown_name_is_declined_whole(ds):
    """An unmodellable projection gains no information, so the schema is left as it was.

    ``ds[["nope"]]`` raises ``KeyError`` in xarray, so this state is only reachable when
    something untracked introduced the name — an ``Opaque`` such as ``rename`` — which puts
    the plan past ``optimize._trusted_prefix`` and out of reach of the variable-level rules
    regardless.

    **This is the one behavioural change the derivation forced.** The old shape restricted
    ``data_vars`` to the known names (so ``{}``) while leaving ``dims`` untouched — a state
    where no variable spanned ``time`` yet ``time`` existed, i.e. exactly the phantom-dim
    contradiction deriving exists to forbid. It was chosen because over-reporting dims is
    the safe direction and knowing no variables is *also* safe, and only the two fields
    being independent made having both possible.

    Declining the whole projection keeps the safe direction on dims — the assertion below,
    and ``test_project_of_an_unknown_name_leaves_dims_alone`` — at the cost of reporting
    variables that a modellable projection would have removed. Sound here because an
    unmodellable ``Project`` is, at this layer, what an ``Opaque`` is: a node after which
    nothing is claimed.
    """
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("nope",)))
    assert after == schema


def test_project_of_an_unknown_name_leaves_dims_alone(ds):
    """Projecting an untracked name leaves every dim in place, the safe direction.

    Notes
    -----
    The guard on the arm: an untracked name (a coord, or something an unmodelled op
    introduced) makes the spanned-dims union an *under*-report, which is the unsafe
    direction. Unknown name -> keep every dim, as before.
    """
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("nope",)))
    assert after.dims == schema.dims


def test_var_dims_unions_known_names(ds):
    """``var_dims`` unions the dims spanned by each named variable."""
    schema = SchemaState.from_dataset(ds)
    assert schema.var_dims(["elevation"]) == {"lat", "lon"}
    assert schema.var_dims(["temperature", "elevation"]) == {"time", "lat", "lon"}
    assert schema.var_dims([]) == frozenset()


def test_var_dims_is_none_for_an_unknown_name(ds):
    """``var_dims`` returns ``None``, not a partial union, for any untracked name.

    Notes
    -----
    "don't know" -- a coord, or a variable an unmodelled op introduced. Callers must
    read this as "no rewrite", not as "no dims".
    """
    schema = SchemaState.from_dataset(ds)
    assert schema.var_dims(["lat"]) is None  # a coord, not a data variable
    assert schema.var_dims(["temperature", "nope"]) is None


def test_non_dimension_coord_survives_unrelated_removal():
    """A scalar coord that is not a dim must not be dropped when an unrelated dim goes."""
    schema = _dim_coords_only({"lat": 3, "lon": 5}, extra_coords=("ref",))
    node = Reduce(name="mean", consumes=["lat"])
    after = apply_schema(schema, node)
    assert after.coords == {"lon", "ref"}
