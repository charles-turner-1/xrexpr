"""Tests for the record-time logical schema (PR 4).

``SchemaState`` is snapshotted from a real dataset and then evolved by
``apply_schema`` using only :data:`~xrexpr.ir.Op` metadata — no array data is touched.
The nodes here are built by hand, which also documents the contract ``apply_schema``
relies on: a scalar select drops its dim (``Select.consumes``, derived from ``indexer``);
a non-scalar select leaves the dim in ``indexer`` only. ``data_vars`` (variable name
-> its dims) is tracked alongside, since that is what a projection rewrite consults.
"""

import numpy as np
import pytest

from xrexpr.ir import ALL_DIMS, Project, Reduce, Scan, Select
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


def test_project_restricts_variables_but_not_dims_or_coords(ds):
    # xarray's ``ds[[...]]`` keeps coords and indexes; only the variables narrow
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("elevation",)))
    assert after.data_vars == {"elevation": ("lat", "lon")}
    assert after.dims == schema.dims
    assert after.coords == schema.coords


def test_project_of_an_unknown_name_yields_nothing_known(ds):
    schema = SchemaState.from_dataset(ds)
    after = apply_schema(schema, Project(name="__getitem__", variables=("nope",)))
    assert after.data_vars == {}


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
