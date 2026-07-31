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
    """``mean("lat")`` records a ``Reduce`` consuming ``lat``, with the arg kept verbatim."""
    node = to_opnode("mean", ("lat",), {})
    assert isinstance(node, Reduce)
    assert node.consumes == frozenset({"lat"})
    assert node.args == ("lat",)


def test_reduce_keyword_dim(schema):
    """``mean(dim="lat")`` resolves to the same ``consumes`` as the positional spelling."""
    node = to_opnode("mean", (), {"dim": "lat"})
    assert node.consumes == frozenset({"lat"})
    assert node.kwargs == frozendict({"dim": "lat"})


def test_reduce_tuple_dims(schema):
    """A tuple dim spec collapses to the set of both dims."""
    node = to_opnode("mean", (("lat", "lon"),), {})
    assert node.consumes == frozenset({"lat", "lon"})


def test_reduce_list_dims_kwarg(schema):
    """A list dim spec passed by keyword collapses to the same set as a tuple would."""
    node = to_opnode("sum", (), {"dim": ["lat", "lon"]})
    assert node.consumes == frozenset({"lat", "lon"})


def test_reduce_no_dim_consumes_all_dims_symbolically():
    """A bare ``mean()`` records the ``ALL_DIMS`` sentinel and stays replayable as written.

    Notes
    -----
    The headline case: the set is left symbolic rather than expanded against a record-time
    schema, which is what fixes the empty-dim reorder bug without freezing dim names the
    recorder may have wrong. See ``ir``'s module docstring.
    """
    node = to_opnode("mean", (), {})
    assert node.consumes is ALL_DIMS
    assert node.args == () and node.kwargs == frozendict()  # replayed as bare mean()


def test_reduce_dim_none_consumes_all_dims_symbolically():
    """An explicit ``dim=None`` is the same "every dim" as naming no dim at all."""
    node = to_opnode("mean", (), {"dim": None})
    assert node.consumes is ALL_DIMS


def test_all_dims_is_resolved_by_apply_schema(schema):
    """Folding a bare ``mean()`` through the schema clears every dim.

    Notes
    -----
    The deferred expansion, cashed in where the schema is exact.
    """
    after = apply_schema(schema, to_opnode("mean", (), {}))
    assert after.sizes == frozendict()


def test_all_dims_survives_construction_uncoerced():
    """Constructing a ``Reduce`` with ``ALL_DIMS`` keeps the sentinel, not a frozenset.

    Notes
    -----
    ``Reduce.__post_init__`` coerces a dim *set*; the sentinel must pass through it.
    """
    assert Reduce(name="mean", consumes=ALL_DIMS).consumes is ALL_DIMS


def test_reduce_non_string_hashable_dim(schema):
    """A non-string hashable dim name records as one dim, not as an iterable of them.

    Notes
    -----
    xarray dim names are ``Hashable``, not only ``str``.
    """
    node = to_opnode("mean", (0,), {})
    assert node.consumes == frozenset({0})


def test_reduce_keeps_non_dim_kwargs_verbatim(schema):
    """A reduction's non-dim kwargs survive into the node untouched, for faithful replay."""
    node = to_opnode("mean", ("lat",), {"skipna": True})
    assert node.consumes == frozenset({"lat"})
    assert node.kwargs == frozendict({"skipna": True})


# --- ``.reduce``, whose first positional is a function ----------------------------
#
# ``Dataset.reduce(func, dim, ...)`` is the one tabulated reduction that does not take its
# dim spec first. Reading ``args[0]`` as one recorded the *function* as a dim (#96) -- a
# node whose ``consumes`` was nonsense, and a written claim that was untrue.


def test_reduce_method_reads_its_dim_from_the_second_positional(schema):
    """``.reduce(func, "lat")`` consumes ``lat``: the dim spec is second, the function first."""
    node = to_opnode("reduce", (sum, "lat"), {})
    assert isinstance(node, Reduce)
    assert node.consumes == frozenset({"lat"})
    assert node.args == (sum, "lat")  # verbatim, function included, for replay


def test_reduce_method_keyword_dim(schema):
    """``.reduce(func, dim="lat")`` resolves the same way; the kwarg spelling was never affected."""
    node = to_opnode("reduce", (sum,), {"dim": "lat"})
    assert node.consumes == frozenset({"lat"})


def test_reduce_method_tuple_dims(schema):
    """``.reduce(func, ("lat", "lon"))`` collapses to both dims, as any other spelling does."""
    node = to_opnode("reduce", (sum, ("lat", "lon")), {})
    assert node.consumes == frozenset({"lat", "lon"})


def test_bare_reduce_method_consumes_all_dims_symbolically(schema):
    """``.reduce(func)`` names no dim, so it consumes ``ALL_DIMS`` like any other bare reduce.

    Notes
    -----
    The regression this guards: with the dim read off ``args[0]``, the lone function
    argument made the call look like a *named* reduce over ``frozenset({func})`` rather
    than a bare one -- the empty-dim reorder bug, reintroduced through a side door.
    """
    node = to_opnode("reduce", (sum,), {})
    assert node.consumes is ALL_DIMS


# --- ``keepdims=True`` keeps the dims it "reduces" --------------------------------


def test_keepdims_records_opaque(schema):
    """``mean("lat", keepdims=True)`` records ``Opaque``: the dim survives, so no ``consumes`` is true.

    Notes
    -----
    Verified against xarray 2026.7.0: ``ds.mean("time", keepdims=True)`` keeps ``time`` at
    size 1. Recorded as a ``Reduce`` it claimed ``consumes={"time"}``, which made
    ``dim_effect`` classify a following select as *invalid* and the optimiser **reject a
    valid chain** -- see ``test_accessor.test_keepdims_reduce_then_select_matches_eager``.
    ``consumes=frozenset()`` would be honest about the dims but leave the schema claiming
    the dim's original *size*, so the conservative variant is the right answer.
    """
    node = to_opnode("mean", ("lat",), {"keepdims": True})
    assert isinstance(node, Opaque)
    assert node.kwargs == frozendict({"keepdims": True})  # verbatim, for replay


def test_keepdims_false_is_an_ordinary_reduce(schema):
    """An explicit ``keepdims=False`` is the default, so it records an ordinary ``Reduce``."""
    node = to_opnode("mean", ("lat",), {"keepdims": False})
    assert isinstance(node, Reduce)
    assert node.consumes == frozenset({"lat"})


def test_keepdims_guard_covers_the_reduce_method_too(schema):
    """``.reduce(func, "lat", keepdims=True)`` is refused on the same grounds as any other reduction."""
    node = to_opnode("reduce", (sum, "lat"), {"keepdims": True})
    assert isinstance(node, Opaque)


def test_isel_scalar_kwarg_drops_dim(schema):
    """``isel(time=0)`` records a ``Select`` whose scalar indexer drops ``time``."""
    node = to_opnode("isel", (), {"time": 0})
    assert isinstance(node, Select)
    assert node.indexer == frozendict({"time": Scalar(0)})
    assert node.consumes == frozenset({"time"})


def test_isel_positional_dict(schema):
    """The positional-dict spelling of ``isel`` resolves like the keyword one."""
    node = to_opnode("isel", ({"time": 0},), {})
    assert node.indexer == frozendict({"time": Scalar(0)})
    assert node.consumes == frozenset({"time"})


def test_isel_slice_keeps_dim(schema):
    """A forward slice classifies as ``ForwardSlice`` and keeps its dim."""
    node = to_opnode("isel", (), {"time": slice(0, 2)})
    assert node.indexer == frozendict({"time": ForwardSlice(0, 2)})
    assert node.consumes == frozenset()


def test_isel_list_keeps_dim(schema):
    """A list of positions keeps its dim, at a new size."""
    node = to_opnode("isel", (), {"lon": [0, 2]})
    assert node.consumes == frozenset()


def test_isel_option_kwarg_not_treated_as_dim(schema):
    """``drop=True`` is excluded from the indexer but kept in kwargs for replay."""
    node = to_opnode("isel", (), {"time": 0, "drop": True})
    assert node.indexer == frozendict({"time": Scalar(0)})  # drop excluded from indexer
    assert node.consumes == frozenset({"time"})
    assert node.kwargs == frozendict({"time": 0, "drop": True})  # verbatim for replay


def test_sel_scalar_label_drops_dim(schema):
    """A scalar ``sel`` label drops its dim, exactly as a scalar ``isel`` position does."""
    node = to_opnode("sel", (), {"lat": 1})
    assert node.indexer == frozendict({"lat": Scalar(1)})
    assert node.consumes == frozenset({"lat"})


def test_sel_label_slice_keeps_dim(schema):
    """A label slice keeps its dim, even though its extent is not statically knowable."""
    node = to_opnode("sel", (), {"time": slice("a", "z")})
    assert node.consumes == frozenset()


def test_sel_option_kwarg_excluded(schema):
    """``method="nearest"`` is an option, not a dim, so it stays out of the indexer."""
    node = to_opnode("sel", (), {"lat": 1, "method": "nearest"})
    assert node.indexer == frozendict({"lat": Scalar(1)})
    assert node.consumes == frozenset({"lat"})


def test_scan_carries_no_resolved_dims(schema):
    """A scan records as ``Scan`` with its dim left in ``args``, and no resolved metadata.

    Notes
    -----
    A ``Scan`` has no ``consumes``/``indexer`` at all — its scanned-dim metadata arrives
    with the first scan-aware rule (W6).
    """
    node = to_opnode("cumsum", ("time",), {})
    assert isinstance(node, Scan)
    assert node.args == ("time",)  # dim kept in args for replay


def test_rechunk_positional_mapping(schema):
    """A positional chunk mapping records as ``Rechunk`` with that mapping in ``chunks``."""
    node = to_opnode("chunk", ({"time": 100},), {})
    assert isinstance(node, Rechunk)
    assert node.chunks == frozendict({"time": 100})


def test_rechunk_kwarg_mapping(schema):
    """Dim keywords on ``chunk`` collect into the same ``chunks`` mapping."""
    node = to_opnode("chunk", (), {"time": 100, "lat": 1})
    assert node.chunks == frozendict({"time": 100, "lat": 1})


@pytest.mark.parametrize("spec", [100, "auto", -1])
def test_uniform_rechunk_names_no_dim(schema, spec):
    """A uniform spec leaves ``chunks`` empty and stays in ``args``.

    Notes
    -----
    A uniform spec has no dim key a later select could invalidate, so it is replayed
    verbatim rather than modelled.
    """
    node = to_opnode("chunk", (spec,), {})
    assert node.chunks == frozendict()
    assert node.args == (spec,)


def test_bare_rechunk_names_no_dim(schema):
    """A bare ``chunk()`` names no dim, so ``chunks`` is empty."""
    node = to_opnode("chunk", (), {})
    assert node.chunks == frozendict()


def test_rechunk_option_kwarg_not_treated_as_a_dim(schema):
    """``token`` is an option, not a dim spec, so it stays out of ``chunks``."""
    node = to_opnode("chunk", ({"time": 100},), {"token": "t"})
    assert node.chunks == frozendict({"time": 100})
    assert node.kwargs == frozendict({"token": "t"})  # verbatim for replay


def test_rechunk_leaves_the_schema_unchanged(schema):
    """A rechunk changes chunk topology only, so the folded schema comes back identical."""
    node = to_opnode("chunk", ({"time": 100},), {})
    assert apply_schema(schema, node) == schema


def test_getitem_of_a_name_is_a_projection(schema):
    """``ds["temperature"]`` records a single-variable ``Project``, key kept verbatim."""
    node = to_opnode("__getitem__", ("temperature",), {})
    assert isinstance(node, Project)
    assert node.variables == ("temperature",)
    assert node.single  # ``ds["temperature"]`` -> DataArray
    assert node.args == ("temperature",)  # key kept verbatim for replay


def test_getitem_of_a_list_is_a_multi_projection(schema):
    """``ds[[...]]`` records a ``Project`` over several names, and is not ``single``."""
    node = to_opnode("__getitem__", (["temperature", "elevation"],), {})
    assert isinstance(node, Project)
    assert node.variables == ("temperature", "elevation")
    assert not node.single  # ``ds[[...]]`` -> Dataset


def test_getitem_names_are_not_validated_at_record_time(schema):
    """An unknown variable name still records as a ``Project``.

    Notes
    -----
    Whether it may *move* is the optimiser's call, made against ``data_vars`` at that
    point in the plan — recording is a pure function of the call.
    """
    node = to_opnode("__getitem__", (["nope"],), {})
    assert isinstance(node, Project) and node.variables == ("nope",)


def test_mask_style_getitem_is_opaque(schema):
    """A dict key is xarray's ``isel`` spelling, not a projection, so it records opaque."""
    node = to_opnode("__getitem__", ({"lat": 0},), {})
    assert isinstance(node, Opaque)


def test_unknown_method_is_opaque(schema):
    """An untabulated method records as ``Opaque``, so no rule can fire on it."""
    node = to_opnode("where", ("cond",), {})
    assert isinstance(node, Opaque)


def test_to_opnode_then_apply_schema_threads(schema):
    """Recorded nodes fold through ``apply_schema`` in sequence, each seeing the last result.

    Notes
    -----
    Two chains: a bare ``mean()`` clears every dim, and a named reduce followed by a
    scalar select leaves only ``lon``.
    """
    node = to_opnode("mean", (), {})  # bare mean -> consumes all
    after = apply_schema(schema, node)
    assert after.sizes == frozendict()  # every dim reduced away

    schema2 = apply_schema(schema, to_opnode("mean", ("lat",), {}))
    node2 = to_opnode("isel", (), {"time": 0})
    after2 = apply_schema(schema2, node2)
    assert after2.sizes == frozendict({"lon": 5})


def test_returns_the_matching_variant(schema):
    """``to_opnode`` returns the ``Op`` variant its ``OP_TABLE`` spec selects."""
    assert isinstance(to_opnode("mean", ("lat",), {}), Reduce)
