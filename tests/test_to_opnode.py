"""Tests for ``to_opnode`` (PR 5): record-time normalisation of a raw call.

These are golden-node assertions: each recorded call, in every dim spelling, must
resolve to the same :data:`~xrexpr.ir.Op` variant and normalised metadata
(``consumes`` / ``indexer``) while keeping ``args``/``kwargs`` verbatim for replay.
The headline case is that a
no-dim ``mean()`` consumes *every current dim*, which is what fixes the empty-dim
reorder bug downstream.
"""

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.ir import Opaque, Rechunk, Reduce, Scan, Select
from xrexpr.schema import SchemaState, apply_schema, to_opnode


@pytest.fixture
def schema() -> SchemaState:
    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.zeros((4, 3, 5)))},
        coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(5)},
    )
    return SchemaState.from_dataset(ds)


def test_reduce_positional_dim(schema):
    node = to_opnode(schema, "mean", ("lat",), {})
    assert isinstance(node, Reduce)
    assert node.consumes == frozenset({"lat"})
    assert node.args == ("lat",)


def test_reduce_keyword_dim(schema):
    node = to_opnode(schema, "mean", (), {"dim": "lat"})
    assert node.consumes == frozenset({"lat"})
    assert node.kwargs == frozendict({"dim": "lat"})


def test_reduce_tuple_dims(schema):
    node = to_opnode(schema, "mean", (("lat", "lon"),), {})
    assert node.consumes == frozenset({"lat", "lon"})


def test_reduce_list_dims_kwarg(schema):
    node = to_opnode(schema, "sum", (), {"dim": ["lat", "lon"]})
    assert node.consumes == frozenset({"lat", "lon"})


def test_reduce_no_dim_consumes_all_current_dims(schema):
    node = to_opnode(schema, "mean", (), {})
    assert node.consumes == frozenset({"time", "lat", "lon"})
    assert node.args == () and node.kwargs == frozendict()  # replayed as bare mean()


def test_reduce_dim_none_consumes_all_current_dims(schema):
    node = to_opnode(schema, "mean", (), {"dim": None})
    assert node.consumes == frozenset({"time", "lat", "lon"})


def test_reduce_non_string_hashable_dim(schema):
    # xarray dim names are Hashable, not only str; a bare hashable is a single dim
    node = to_opnode(schema, "mean", (0,), {})
    assert node.consumes == frozenset({0})


def test_reduce_keeps_non_dim_kwargs_verbatim(schema):
    node = to_opnode(schema, "mean", ("lat",), {"skipna": True})
    assert node.consumes == frozenset({"lat"})
    assert node.kwargs == frozendict({"skipna": True})


def test_isel_scalar_kwarg_drops_dim(schema):
    node = to_opnode(schema, "isel", (), {"time": 0})
    assert isinstance(node, Select)
    assert node.indexer == frozendict({"time": 0})
    assert node.consumes == frozenset({"time"})


def test_isel_positional_dict(schema):
    node = to_opnode(schema, "isel", ({"time": 0},), {})
    assert node.indexer == frozendict({"time": 0})
    assert node.consumes == frozenset({"time"})


def test_isel_slice_keeps_dim(schema):
    node = to_opnode(schema, "isel", (), {"time": slice(0, 2)})
    assert node.indexer == frozendict({"time": slice(0, 2)})
    assert node.consumes == frozenset()


def test_isel_list_keeps_dim(schema):
    node = to_opnode(schema, "isel", (), {"lon": [0, 2]})
    assert node.consumes == frozenset()


def test_isel_option_kwarg_not_treated_as_dim(schema):
    node = to_opnode(schema, "isel", (), {"time": 0, "drop": True})
    assert node.indexer == frozendict({"time": 0})  # drop excluded from indexer
    assert node.consumes == frozenset({"time"})
    assert node.kwargs == frozendict({"time": 0, "drop": True})  # verbatim for replay


def test_sel_scalar_label_drops_dim(schema):
    node = to_opnode(schema, "sel", (), {"lat": 1})
    assert node.indexer == frozendict({"lat": 1})
    assert node.consumes == frozenset({"lat"})


def test_sel_label_slice_keeps_dim(schema):
    node = to_opnode(schema, "sel", (), {"time": slice("a", "z")})
    assert node.consumes == frozenset()


def test_sel_option_kwarg_excluded(schema):
    node = to_opnode(schema, "sel", (), {"lat": 1, "method": "nearest"})
    assert node.indexer == frozendict({"lat": 1})
    assert node.consumes == frozenset({"lat"})


def test_scan_carries_no_resolved_dims(schema):
    node = to_opnode(schema, "cumsum", ("time",), {})
    assert isinstance(node, Scan)  # a Scan has no consumes/indexer at all
    assert node.args == ("time",)  # dim kept in args for replay


def test_rechunk_positional_mapping(schema):
    node = to_opnode(schema, "chunk", ({"time": 100},), {})
    assert isinstance(node, Rechunk)
    assert node.chunks == frozendict({"time": 100})


def test_rechunk_kwarg_mapping(schema):
    node = to_opnode(schema, "chunk", (), {"time": 100, "lat": 1})
    assert node.chunks == frozendict({"time": 100, "lat": 1})


@pytest.mark.parametrize("spec", [100, "auto", -1])
def test_uniform_rechunk_names_no_dim(schema, spec):
    # a uniform spec has no dim key a later select could invalidate, so ``chunks``
    # stays empty and the spec is replayed verbatim from ``args``
    node = to_opnode(schema, "chunk", (spec,), {})
    assert node.chunks == frozendict()
    assert node.args == (spec,)


def test_bare_rechunk_names_no_dim(schema):
    node = to_opnode(schema, "chunk", (), {})
    assert node.chunks == frozendict()


def test_rechunk_option_kwarg_not_treated_as_a_dim(schema):
    node = to_opnode(schema, "chunk", ({"time": 100},), {"token": "t"})
    assert node.chunks == frozendict({"time": 100})
    assert node.kwargs == frozendict({"token": "t"})  # verbatim for replay


def test_rechunk_leaves_the_schema_unchanged(schema):
    node = to_opnode(schema, "chunk", ({"time": 100},), {})
    assert apply_schema(schema, node) == schema


def test_untabulated_op_is_opaque(schema):
    node = to_opnode(schema, "__getitem__", ("temperature",), {})
    assert isinstance(node, Opaque)
    assert node.args == ("temperature",)


def test_unknown_method_is_opaque(schema):
    node = to_opnode(schema, "where", ("cond",), {})
    assert isinstance(node, Opaque)


def test_to_opnode_then_apply_schema_threads(schema):
    node = to_opnode(schema, "mean", (), {})  # bare mean -> consumes all
    after = apply_schema(schema, node)
    assert after.dims == frozendict()  # every dim reduced away

    schema2 = apply_schema(schema, to_opnode(schema, "mean", ("lat",), {}))
    node2 = to_opnode(schema2, "isel", (), {"time": 0})
    after2 = apply_schema(schema2, node2)
    assert after2.dims == frozendict({"lon": 5})


def test_returns_the_matching_variant(schema):
    assert isinstance(to_opnode(schema, "mean", ("lat",), {}), Reduce)
