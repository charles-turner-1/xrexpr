"""Tests for the logical schema layer.

``SchemaState`` is snapshotted from a real dataset and then evolved by
``apply_schema`` using only :data:`~xrexpr.ir.Op` metadata — no array data is touched.
The nodes here are built by hand, which also documents the contract ``apply_schema``
relies on: a scalar select drops its dim (``Select.consumes``, derived from ``indexer``);
a non-scalar select leaves the dim in ``indexer`` only. ``data_vars`` (variable name
-> its dims) is tracked alongside, since that is what a projection rewrite consults.
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


def test_from_dataset_snapshots_dims_and_coords(ds):
    schema = SchemaState.from_dataset(ds)
    assert schema.dims == {"time": 4, "lat": 3, "lon": 5}
    assert schema.coords == {"time", "lat", "lon"}
    assert schema.dim_names == {"time", "lat", "lon"}


def test_from_dataarray_snapshots_dims(ds):
    schema = SchemaState.from_dataset(ds["temperature"])
    assert schema.dims == {"time": 4, "lat": 3, "lon": 5}


def test_schema_is_immutable():
    schema = SchemaState(dims={"time": 4}, coords={"time"})
    with pytest.raises(TypeError):
        schema.dims["time"] = 1
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        schema.dims = {}


def test_reduce_removes_dim_and_its_coord(ds):
    schema = SchemaState.from_dataset(ds)
    node = Reduce(name="mean", args=("lat",), consumes=["lat"])
    after = apply_schema(schema, node)
    assert after.dims == {"time": 4, "lon": 5}
    assert after.coords == {"time", "lon"}


def test_bare_reduce_removes_every_dim_and_coord(ds):
    # ``apply_schema`` is where the deferred ALL_DIMS expansion is finally cashed in,
    # against a schema that is exact rather than the recorder's guess.
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Reduce(name="mean", consumes=ALL_DIMS))
    assert after.dims == {}
    assert after.coords == set()
    assert all(dims == () for dims in after.data_vars.values())


def test_bare_reduce_expands_against_the_schema_it_is_given(ds):
    # "every dim *at this point*": fold a select in first and the sentinel resolves to
    # what is left, not to the original dataset's dims.
    schema = apply_schema(
        SchemaState.from_dataset(ds), Select(name="isel", indexer={"time": 0})
    )
    assert apply_schema(schema, Reduce(name="mean", consumes=ALL_DIMS)).dims == {}


def test_scalar_isel_removes_dim(ds):
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"time": 0})  # scalar -> consumes={"time"}
    after = apply_schema(schema, node)
    assert after.dims == {"lat": 3, "lon": 5}
    assert "time" not in after.coords


def test_scalar_sel_removes_dim(ds):
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"lat": 1})
    after = apply_schema(schema, node)
    assert after.dims == {"time": 4, "lon": 5}


def test_slice_isel_resizes_kept_dim(ds):
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"time": slice(0, 2)})
    after = apply_schema(schema, node)
    assert after.dims == {"time": 2, "lat": 3, "lon": 5}
    assert after.coords == {"time", "lat", "lon"}  # dim kept -> coord kept


def test_list_isel_resizes_kept_dim(ds):
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"lon": [0, 2, 4]})
    after = apply_schema(schema, node)
    assert after.dims["lon"] == 3


def test_boolean_array_isel_resizes_by_true_count(ds):
    schema = SchemaState.from_dataset(ds)
    mask = np.array([True, False, True, True])
    node = Select(name="isel", indexer={"time": mask})
    after = apply_schema(schema, node)
    assert after.dims["time"] == 3


def test_integer_array_isel_resizes_by_length(ds):
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"lon": np.array([0, 1])})
    after = apply_schema(schema, node)
    assert after.dims["lon"] == 2


def test_boolean_list_isel_resizes_by_true_count(ds):
    schema = SchemaState.from_dataset(ds)
    node = Select(name="isel", indexer={"time": [True, False, True, True]})
    after = apply_schema(schema, node)
    assert after.dims["time"] == 3


def test_unsizable_sel_slice_is_unknown(ds):
    # A label slice needs coord values to size. This used to keep the *current* size --
    # a guess in the safe direction, but still a guess; "don't know" is now sayable.
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"time": slice("a", "z")})
    after = apply_schema(schema, node)
    assert after.dims["time"] is None
    assert "time" in after.dims  # unknown extent, but the dim is still there


def test_integer_labelled_sel_slice_is_unknown_not_under_reported(ds):
    # The sharp case: integer bounds are indistinguishable from positional ones, so
    # ``classify`` mints a ForwardSlice and its size reads them as positions. Against a
    # length-4 dim, ``sel(time=slice(20, 30))`` sized as **0** -- an under-report, the
    # direction a size-driven rule could act on. Now unknown.
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"time": slice(20, 30)})
    after = apply_schema(schema, node)
    assert after.dims["time"] is None


def test_positional_isel_slice_is_still_sized_exactly(ds):
    # The unknown is confined to ``sel``: an ``isel`` slice really is positional, so it
    # keeps its exact size and nothing is lost by the change above.
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Select(name="isel", indexer={"time": slice(1, 3)}))
    assert after.dims["time"] == 2


def test_unknown_size_propagates_through_a_later_select(ds):
    # Unknown in, unknown out -- and never silently coerced to 0, the failure mode the
    # ``var_dims`` docstring warns about for the variable-level counterpart.
    unknown = SchemaState(dims={"time": None, "lat": 3}, coords={"time", "lat"})
    after = apply_schema(unknown, Select(name="isel", indexer={"time": slice(0, 2)}))
    assert after.dims["time"] is None
    assert after.dims["time"] != 0
    assert after.dims["lat"] == 3  # a known size beside it is unaffected


def test_scalar_select_still_drops_a_dim_of_unknown_size(ds):
    # Dropping needs no size at all, so the unknown must not block it.
    unknown = SchemaState(dims={"time": None, "lat": 3}, coords={"time", "lat"})
    after = apply_schema(unknown, Select(name="isel", indexer={"time": 0}))
    assert "time" not in after.dims
    assert "time" not in after.coords


# --- grouped reduces -----------------------------------------------------------------
# The arm was derived from what xarray actually does rather than from first principles,
# so it is checked the same way: fold the schema, run the chain, compare. Anything else
# would only re-assert the assumption the arm was written from.


@pytest.fixture
def dated_ds() -> xr.Dataset:
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
    # The group count is a fact about coordinate *values*, which this layer does not read
    # -- exactly the gap ``int | None`` was added for. Unknown, never a guess.
    node = GroupedReduce(
        name="groupby", group_dim="time", new_dim="month", reduce="mean"
    )
    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert after.dims["month"] is None
    assert after.dims["lat"] == 3  # untouched dims keep their known sizes


def test_grouped_reduce_adds_the_new_dim_to_a_variable_that_lacked_the_group_dim(
    dated_ds,
):
    # The non-obvious half: ``elevation(lat, lon)`` comes back as ``(month, lat, lon)``
    # even though it never carried ``time``. Verified against xarray above; asserted
    # directly here because it is the part a reader would most expect to be wrong.
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
    # ``time`` is 24 here, so ``trim`` and ``pad`` agree; the odd-length case below is
    # what actually tells the two roundings apart.
    eager = dated_ds
    for name, args, kwargs in calls:
        eager = getattr(eager, name)(*args, **kwargs)

    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert dict(after.dims) == dict(eager.sizes)


@pytest.mark.parametrize(
    ("boundary", "expected"), [("trim", 3), ("pad", 4), ("exact", 3)]
)
def test_coarsen_rounding_follows_the_boundary_kwarg(boundary, expected):
    # 7 // 2 rounds three ways. ``exact`` would raise at replay rather than produce 3,
    # so agreeing with ``trim`` here costs nothing: the plan never gets that far.
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
    # One output position per input position: ``center``/``min_periods`` change values,
    # never shape -- which is what makes rolling's dim algebra simpler than groupby's.
    node = WindowedReduce(name="rolling", reduce="mean", window={"time": 5})
    after = apply_schema(SchemaState.from_dataset(dated_ds), node)
    assert dict(after.dims) == dict(dated_ds.sizes)
    assert after.coords == frozenset(dated_ds.coords)


def test_unrecognised_boundary_marks_the_size_unknown():
    # A future spelling should cost precision, not correctness.
    ds = xr.Dataset({"t": ("time", np.arange(7.0))}, coords={"time": np.arange(7)})
    node = WindowedReduce(
        name="coarsen",
        reduce="mean",
        window={"time": 2},
        kwargs={"time": 2, "boundary": "something-new"},
    )
    assert apply_schema(SchemaState.from_dataset(ds), node).dims["time"] is None


def test_windowing_an_unknown_size_stays_unknown():
    unknown = SchemaState(dims={"time": None}, coords={"time"})
    node = WindowedReduce(
        name="coarsen", reduce="mean", window={"time": 2}, kwargs={"time": 2}
    )
    assert apply_schema(unknown, node).dims["time"] is None


# --- weighted reduces ----------------------------------------------------------------
# Derived from xarray the same way the two arms above were: the weights' dim effect is the
# part of this node that is *not* a plain reduce's, so it is compared against what
# evaluation produces rather than against the assumption the arm was written from.


def _weighted(consumes, weight_dims, weights):
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
    # The weights align with the dataset, so a shared dim can *shrink* -- verified below.
    # An extent this layer cannot compute is ``None``, W3's answer, rather than the current
    # size, which would be an over-report of a dim a rewrite might act on.
    w = ds["lat"] * ds["lon"]
    node = _weighted(frozenset({"lat"}), frozenset({"lat", "lon"}), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    assert after.dims["lon"] is None  # a weight dim, kept but not sized
    assert after.dims["time"] == 4  # untouched dims keep their known sizes
    assert "lat" not in after.dims


def test_misaligned_weights_really_do_shrink_a_shared_dim(ds):
    # The fact the unknown above exists for, pinned against xarray so it is a verified
    # claim rather than a defensive guess: ``dot`` aligns, so weights covering two of
    # three ``lat`` labels inner-join and the dim comes back shorter.
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
    # The other half, and the part a reader would most expect to be wrong: ``elevation``
    # has no ``member`` and no way to acquire one, yet comes back carrying it -- because
    # ``dot`` broadcasts. Asserted directly as well as via the derivation above.
    w = xr.DataArray([1.0, 2.0], dims="member", coords={"member": [0, 1]})
    node = _weighted(frozenset({"lat"}), frozenset({"member"}), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    assert set(after.data_vars["elevation"]) == {"member", "lon"}
    assert after.dims["member"] is None


def test_several_minted_dims_land_in_a_deterministic_order(ds):
    # ``minted`` is a set, and ``SchemaState`` is a *value*: an iteration-order-dependent
    # dim tuple would make two snapshots folded from identical inputs unequal between
    # processes (set order over strings varies with PYTHONHASHSEED). Dim order carries no
    # meaning here, so a stable one is free -- but it has to be asserted to stay stable.
    w = xr.DataArray(np.ones((2, 2, 2)), dims=("run", "member", "ens"))
    node = _weighted(frozenset({"lat"}), frozenset(w.dims), w)

    after = apply_schema(SchemaState.from_dataset(ds), node)

    assert after.data_vars["elevation"][:3] == ("ens", "member", "run")


def test_weighting_an_unknown_size_stays_unknown():
    unknown = SchemaState(dims={"time": None, "lat": 3}, coords={"time", "lat"})
    w = xr.DataArray([1.0], dims="time")
    after = apply_schema(unknown, _weighted(frozenset({"lat"}), frozenset({"time"}), w))
    assert after.dims["time"] is None


def test_scan_leaves_schema_unchanged(ds):
    schema = SchemaState.from_dataset(ds)
    node = Scan(name="cumsum", args=("time",))
    after = apply_schema(schema, node)
    assert after.dims == schema.dims
    assert after.coords == schema.coords


def test_schema_threads_through_a_chain(ds):
    schema = SchemaState.from_dataset(ds)
    schema = apply_schema(schema, Reduce(name="mean", args=("lat",), consumes=["lat"]))
    schema = apply_schema(schema, Select(name="isel", indexer={"time": 0}))
    assert schema.dims == {"lon": 5}
    assert schema.coords == {"lon"}


def test_from_dataset_snapshots_variable_dims(ds):
    schema = SchemaState.from_dataset(ds)
    assert schema.data_vars == {
        "temperature": ("time", "lat", "lon"),
        "elevation": ("lat", "lon"),
    }


def test_from_dataarray_has_no_data_vars(ds):
    assert SchemaState.from_dataset(ds["temperature"]).data_vars == {}


def test_reduce_strips_the_dim_from_every_variable(ds):
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Reduce(name="mean", consumes=["lat"]))
    assert after.data_vars == {"temperature": ("time", "lon"), "elevation": ("lon",)}


def test_scalar_select_strips_the_dim_from_every_variable(ds):
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Select(name="isel", indexer={"time": 0}))
    assert after.data_vars == {
        "temperature": ("lat", "lon"),
        "elevation": ("lat", "lon"),
    }


def test_project_drops_the_dims_the_survivors_no_longer_span(ds):
    # This test used to assert ``after.dims == schema.dims`` -- that a projection narrows
    # only the variables. It doesn't: ``elevation`` has no ``time``, and xarray drops an
    # orphaned dim outright rather than keeping it empty. Corrected against evaluation,
    # and found by the property suite once its generated datasets grew a second variable.
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("elevation",)))
    eager = ds[["elevation"]]

    assert after.data_vars == {"elevation": ("lat", "lon")}
    assert after.dim_names == frozenset(eager.sizes) == {"lat", "lon"}
    assert after.coords <= frozenset(eager.coords)


def test_project_keeping_every_dim_leaves_the_schema_alone(ds):
    # The other side of it: ``temperature`` spans every dim, so nothing is orphaned.
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(
        schema, Project(name="__getitem__", variables=("temperature",))
    )
    assert after.dims == schema.dims
    assert after.coords == schema.coords


def test_project_of_an_unknown_name_yields_nothing_known(ds):
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("nope",)))
    assert after.data_vars == {}


def test_project_of_an_unknown_name_leaves_dims_alone(ds):
    # The guard on the arm: an untracked name (a coord, or something an unmodelled op
    # introduced) makes the spanned-dims union an *under*-report, which is the unsafe
    # direction. Unknown name -> keep every dim, as before.
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("nope",)))
    assert after.dims == schema.dims


def test_var_dims_unions_known_names(ds):
    schema = SchemaState.from_dataset(ds)
    assert schema.var_dims(["elevation"]) == {"lat", "lon"}
    assert schema.var_dims(["temperature", "elevation"]) == {"time", "lat", "lon"}
    assert schema.var_dims([]) == frozenset()


def test_var_dims_is_none_for_an_unknown_name(ds):
    # "don't know" -- a coord, or a variable an unmodelled op introduced. Callers must
    # read this as "no rewrite", not as "no dims".
    schema = SchemaState.from_dataset(ds)
    assert schema.var_dims(["lat"]) is None  # a coord, not a data variable
    assert schema.var_dims(["temperature", "nope"]) is None


def test_non_dimension_coord_survives_unrelated_removal():
    # a scalar coord ("ref") that is not a dim must not be dropped when "lat" goes
    schema = SchemaState(dims={"lat": 3, "lon": 5}, coords={"lat", "lon", "ref"})
    node = Reduce(name="mean", consumes=["lat"])
    after = apply_schema(schema, node)
    assert after.coords == {"lon", "ref"}
