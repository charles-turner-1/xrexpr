"""Tests for the record-time logical schema (PR 4).

``SchemaState`` is snapshotted from a real dataset and then evolved by
``apply_schema`` using only :data:`~xrexpr.ir.Op` metadata — no array data is touched.
The nodes here are built by hand, which also documents the contract ``apply_schema``
relies on: a scalar select drops its dim (``Select.consumes``, derived from ``indexer``);
a non-scalar select leaves the dim in ``indexer`` only.
"""

import numpy as np
import pytest
import xarray as xr

from xrexpr.ir import Reduce, Scan, Select
from xrexpr.schema import SchemaState, apply_schema


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


def test_unsizable_sel_slice_keeps_current_size(ds):
    # a label slice would need coord values to size -> conservatively unchanged
    schema = SchemaState.from_dataset(ds)
    node = Select(name="sel", indexer={"time": slice("a", "z")})
    after = apply_schema(schema, node)
    assert after.dims["time"] == 4


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


def test_non_dimension_coord_survives_unrelated_removal():
    # a scalar coord ("ref") that is not a dim must not be dropped when "lat" goes
    schema = SchemaState(dims={"lat": 3, "lon": 5}, coords={"lat", "lon", "ref"})
    node = Reduce(name="mean", consumes=["lat"])
    after = apply_schema(schema, node)
    assert after.coords == {"lon", "ref"}
