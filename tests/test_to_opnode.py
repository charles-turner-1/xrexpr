"""Tests for ``to_opnode``: record-time normalisation of a raw call.

These are golden-node assertions: each recorded call, in every dim spelling, must
resolve to the same :data:`~xrexpr.ir.Op` variant and normalised metadata
(``consumes`` / ``indexer``) while keeping ``args``/``kwargs`` verbatim for replay.
The headline case is that a no-dim ``mean()`` consumes :data:`~xrexpr.ir.ALL_DIMS` — the
symbolic set, *not* an expansion against any schema — which is what fixes the empty-dim
reorder bug downstream without freezing dim names the recorder may have wrong.

``to_opnode`` takes no schema: these tests pass none, and the ``schema`` fixture appears
only where a node is then folded through ``apply_schema``.
"""

import pytest
from frozendict import frozendict

from xrexpr.indexers import ForwardSlice, Scalar
from xrexpr.ir import ALL_DIMS, Opaque, Project, Rechunk, Reduce, Scan, Select
from xrexpr.schema import apply_schema, to_opnode


def test_reduce_positional_dim(schema):
    node = to_opnode("mean", ("lat",), {})
    assert isinstance(node, Reduce)
    assert node.consumes == frozenset({"lat"})
    assert node.args == ("lat",)


def test_reduce_keyword_dim(schema):
    node = to_opnode("mean", (), {"dim": "lat"})
    assert node.consumes == frozenset({"lat"})
    assert node.kwargs == frozendict({"dim": "lat"})


def test_reduce_tuple_dims(schema):
    node = to_opnode("mean", (("lat", "lon"),), {})
    assert node.consumes == frozenset({"lat", "lon"})


def test_reduce_list_dims_kwarg(schema):
    node = to_opnode("sum", (), {"dim": ["lat", "lon"]})
    assert node.consumes == frozenset({"lat", "lon"})


def test_reduce_no_dim_consumes_all_dims_symbolically():
    node = to_opnode("mean", (), {})
    assert node.consumes is ALL_DIMS  # not expanded here -- see ``ir``'s docstring
    assert node.args == () and node.kwargs == frozendict()  # replayed as bare mean()


def test_reduce_dim_none_consumes_all_dims_symbolically():
    node = to_opnode("mean", (), {"dim": None})
    assert node.consumes is ALL_DIMS


def test_all_dims_is_resolved_by_apply_schema(schema):
    # the deferred expansion, cashed in where the schema is exact: every dim goes
    after = apply_schema(schema, to_opnode("mean", (), {}))
    assert after.sizes == frozendict()


def test_all_dims_survives_construction_uncoerced():
    # ``Reduce.__post_init__`` coerces a dim *set*; the sentinel must pass through it
    assert Reduce(name="mean", consumes=ALL_DIMS).consumes is ALL_DIMS


def test_reduce_non_string_hashable_dim(schema):
    # xarray dim names are Hashable, not only str; a bare hashable is a single dim
    node = to_opnode("mean", (0,), {})
    assert node.consumes == frozenset({0})


def test_reduce_keeps_non_dim_kwargs_verbatim(schema):
    node = to_opnode("mean", ("lat",), {"skipna": True})
    assert node.consumes == frozenset({"lat"})
    assert node.kwargs == frozendict({"skipna": True})


def test_isel_scalar_kwarg_drops_dim(schema):
    node = to_opnode("isel", (), {"time": 0})
    assert isinstance(node, Select)
    assert node.indexer == frozendict({"time": Scalar(0)})
    assert node.consumes == frozenset({"time"})


def test_isel_positional_dict(schema):
    node = to_opnode("isel", ({"time": 0},), {})
    assert node.indexer == frozendict({"time": Scalar(0)})
    assert node.consumes == frozenset({"time"})


def test_isel_slice_keeps_dim(schema):
    node = to_opnode("isel", (), {"time": slice(0, 2)})
    assert node.indexer == frozendict({"time": ForwardSlice(0, 2)})
    assert node.consumes == frozenset()


def test_isel_list_keeps_dim(schema):
    node = to_opnode("isel", (), {"lon": [0, 2]})
    assert node.consumes == frozenset()


def test_isel_option_kwarg_not_treated_as_dim(schema):
    node = to_opnode("isel", (), {"time": 0, "drop": True})
    assert node.indexer == frozendict({"time": Scalar(0)})  # drop excluded from indexer
    assert node.consumes == frozenset({"time"})
    assert node.kwargs == frozendict({"time": 0, "drop": True})  # verbatim for replay


def test_sel_scalar_label_drops_dim(schema):
    node = to_opnode("sel", (), {"lat": 1})
    assert node.indexer == frozendict({"lat": Scalar(1)})
    assert node.consumes == frozenset({"lat"})


def test_sel_label_slice_keeps_dim(schema):
    node = to_opnode("sel", (), {"time": slice("a", "z")})
    assert node.consumes == frozenset()


def test_sel_option_kwarg_excluded(schema):
    node = to_opnode("sel", (), {"lat": 1, "method": "nearest"})
    assert node.indexer == frozendict({"lat": Scalar(1)})
    assert node.consumes == frozenset({"lat"})


def test_scan_carries_no_resolved_dims(schema):
    node = to_opnode("cumsum", ("time",), {})
    assert isinstance(node, Scan)  # a Scan has no consumes/indexer at all
    assert node.args == ("time",)  # dim kept in args for replay


def test_rechunk_positional_mapping(schema):
    node = to_opnode("chunk", ({"time": 100},), {})
    assert isinstance(node, Rechunk)
    assert node.chunks == frozendict({"time": 100})


def test_rechunk_kwarg_mapping(schema):
    node = to_opnode("chunk", (), {"time": 100, "lat": 1})
    assert node.chunks == frozendict({"time": 100, "lat": 1})


@pytest.mark.parametrize("spec", [100, "auto", -1])
def test_uniform_rechunk_names_no_dim(schema, spec):
    # a uniform spec has no dim key a later select could invalidate, so ``chunks``
    # stays empty and the spec is replayed verbatim from ``args``
    node = to_opnode("chunk", (spec,), {})
    assert node.chunks == frozendict()
    assert node.args == (spec,)


def test_bare_rechunk_names_no_dim(schema):
    node = to_opnode("chunk", (), {})
    assert node.chunks == frozendict()


def test_rechunk_option_kwarg_not_treated_as_a_dim(schema):
    node = to_opnode("chunk", ({"time": 100},), {"token": "t"})
    assert node.chunks == frozendict({"time": 100})
    assert node.kwargs == frozendict({"token": "t"})  # verbatim for replay


def test_rechunk_leaves_the_schema_unchanged(schema):
    node = to_opnode("chunk", ({"time": 100},), {})
    assert apply_schema(schema, node) == schema


def test_getitem_of_a_name_is_a_projection(schema):
    node = to_opnode("__getitem__", ("temperature",), {})
    assert isinstance(node, Project)
    assert node.variables == ("temperature",)
    assert node.single  # ``ds["temperature"]`` -> DataArray
    assert node.args == ("temperature",)  # key kept verbatim for replay


def test_getitem_of_a_list_is_a_multi_projection(schema):
    node = to_opnode("__getitem__", (["temperature", "elevation"],), {})
    assert isinstance(node, Project)
    assert node.variables == ("temperature", "elevation")
    assert not node.single  # ``ds[[...]]`` -> Dataset


def test_getitem_names_are_not_validated_at_record_time(schema):
    # an unknown name still records as a projection; whether it may *move* is the
    # optimiser's call, made against ``data_vars`` at that point in the plan
    node = to_opnode("__getitem__", (["nope"],), {})
    assert isinstance(node, Project) and node.variables == ("nope",)


def test_mask_style_getitem_is_opaque(schema):
    # a dict key is xarray's ``isel`` spelling, not a projection
    node = to_opnode("__getitem__", ({"lat": 0},), {})
    assert isinstance(node, Opaque)


def test_unknown_method_is_opaque(schema):
    node = to_opnode("where", ("cond",), {})
    assert isinstance(node, Opaque)


def test_to_opnode_then_apply_schema_threads(schema):
    node = to_opnode("mean", (), {})  # bare mean -> consumes all
    after = apply_schema(schema, node)
    assert after.sizes == frozendict()  # every dim reduced away

    schema2 = apply_schema(schema, to_opnode("mean", ("lat",), {}))
    node2 = to_opnode("isel", (), {"time": 0})
    after2 = apply_schema(schema2, node2)
    assert after2.sizes == frozendict({"lon": 5})


def test_returns_the_matching_variant(schema):
    assert isinstance(to_opnode("mean", ("lat",), {}), Reduce)
