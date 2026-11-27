"""Tests for the plan optimiser: the fixpoint loop and its rules.

These assert the exact node list :func:`~xrexpr.optimize.optimize` returns. Nodes are
built with ``to_opnode``, which gives the same metadata the real recorder produces and
is a pure function of the call, so these test the optimiser without going through the
accessor. The ``schema`` fixture is passed to :func:`~xrexpr.optimize.optimize`, not
used to build nodes: it's the *base* schema the optimiser folds forward.

What's covered:

- merging adjacent selects, and moving selects left past reductions. A select moves past
  any reduce on other dims, and the two rules combine through the fixpoint loop (move,
  then merge). When the dims overlap, the result is one of three outcomes: swap, leave,
  or raise.
- projection pushdown, the variable-level rule, which really uses the ``schema``
  argument. Whether ``[["temperature"]]`` may cross an op depends on the dims that
  variable has at that point, so the fixture has a second variable (``elevation``)
  *without* ``time``. That's the case that must not be reordered, and must not raise
  either.
"""

import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr._xrexprs.ir import ALL_DIMS
from xrexpr.chunks import NoChange, SingleSize
from xrexpr.exceptions import InvalidExpressionError
from xrexpr.indexers import classify
from xrexpr.ir import ALL_DIMS, GroupedReduce, WeightedReduce, WindowedReduce
from xrexpr.lower import Call, emit
from xrexpr.optimize import AUTOCHUNK_EMPTY_DIMS_SAFE, dim_effect, optimize
from xrexpr.schema import SchemaState, to_opnode


def _node(name, *args, **kwargs):
    """The ``OpNode`` ``to_opnode`` would build for this call, args and kwargs verbatim."""
    return to_opnode(name, args, kwargs)


def _ix(**dims):
    """The ``indexer`` a ``Select`` would hold for these raw values (each classified)."""
    return frozendict({d: classify(v) for d, v in dims.items()})


def test_merge_consecutive_isel_kwargs(schema):
    """Two consecutive keyword ``isel`` calls on different dims merge into one indexer node."""
    plan = [_node("isel", time=0), _node("isel", lat=1)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].name == "isel"
    assert out[0].indexer == _ix(time=0, lat=1)
    assert out[0].consumes == frozenset({"time", "lat"})
    # Replayable: the merged indexer emits as one canonical positional dict.
    assert emit(out) == [Call("isel", ({"time": 0, "lat": 1},))]


def test_merge_consecutive_isel_positional_dict(schema):
    """A positional dict-form ``isel`` merges with a following one, just as the kwarg form does."""
    plan = [_node("isel", {"time": 0}), _node("isel", {"lat": 1})]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=0, lat=1)


def test_merge_run_of_three_isel(schema):
    """A run of three consecutive ``isel`` calls, each on a different dim, merges into one node."""
    plan = [
        _node("isel", time=0),
        _node("isel", lat=1),
        _node("isel", lon=2),
    ]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=0, lat=1, lon=2)


def test_merge_consecutive_sel(schema):
    """Two consecutive ``sel`` calls on different dims merge into a single indexer node."""
    plan = [_node("sel", lat=1), _node("sel", lon=2)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].name == "sel"
    assert out[0].indexer == _ix(lat=1, lon=2)


def test_isel_keeps_slice_dim_when_merging(schema):
    """Merging a slice ``isel`` with a scalar ``isel`` keeps the slice's dim, consuming only the scalar's."""
    plan = [_node("isel", time=slice(0, 2)), _node("isel", lat=1)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=slice(0, 2), lat=1)
    assert out[0].consumes == frozenset({"lat"})


def test_isel_and_sel_not_merged(schema):
    """``isel`` and ``sel`` are never merged together, since they carry different indexing semantics."""
    plan = [_node("isel", time=0), _node("sel", lat=1)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "sel"]


def test_option_kwarg_select_is_a_barrier(schema):
    """A ``drop=True`` option kwarg can't be carried by a bare indexer, so it blocks the merge."""
    plan = [_node("isel", time=0, drop=True), _node("isel", lat=1)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "isel"]
    assert out[0].kwargs == frozendict({"time": 0, "drop": True})  # verbatim, unmerged


def test_non_select_plan_unchanged(schema):
    """A plan of non-select ops (two means) is left unchanged by the optimiser."""
    plan = [_node("mean", "lat"), _node("mean", "lon")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "mean"]


def test_pushdown_isel_past_mean(schema):
    """A disjoint ``isel`` hops in front of a ``mean`` reduce that doesn't touch its dim."""
    plan = [_node("mean", "lat"), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=0)


def test_pushdown_generalises_to_sum(schema):
    """Select pushdown generalises beyond ``mean``: a disjoint ``isel`` also hops past a ``sum`` reduce."""
    plan = [_node("sum", "lat"), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "sum"]


def test_pushdown_generalises_to_any_reduce(schema):
    """Every reduce (``std``, ``max``, ``median``, ...) pushes a disjoint select in front of it, not just ``mean``/``sum``."""
    for reduce_op in ("std", "max", "median"):
        plan = [_node(reduce_op, "lat"), _node("isel", time=0)]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["isel", reduce_op]


def test_pushdown_sel_past_reduce(schema):
    """A disjoint ``sel`` hops in front of a reduce too, not just ``isel``."""
    plan = [_node("mean", "lat"), _node("sel", lon=2)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["sel", "mean"]


def test_select_on_reduced_dim_raises(schema):
    """Selecting a dim a preceding reduce already removed is unreplayable and must raise."""
    plan = [_node("mean", "lat"), _node("isel", lat=0)]
    with pytest.raises(InvalidExpressionError, match="lat"):
        optimize(plan, schema)


def test_keepdims_reduce_admits_a_disjoint_select(schema):
    """A disjoint ``isel`` hops in front of a ``keepdims=True`` reduce, just as it does past a plain one."""
    plan = [_node("mean", "lat", keepdims=True), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=0)


def test_keepdims_reduce_leaves_a_select_on_a_kept_dim(schema):
    """A ``keepdims=True`` reduce keeps its dim at size 1, so a select on it is valid but *left*, not raised.

    Notes
    -----
    Compare :func:`test_select_on_reduced_dim_raises`. A plain ``mean("lat")`` removes
    ``lat``, so a select on it is invalid. ``mean("lat", keepdims=True)`` keeps it, so
    the chain replays, and the ``immovable`` dim effect leaves the order alone rather
    than rejecting it.
    """
    plan = [_node("mean", "lat", keepdims=True), _node("isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "isel"]


def test_keepdims_dim_effect_blocks_its_dims_immovably(schema):
    """A ``keepdims`` reduce's ``dim_effect`` blocks and requires its named dims, ``immovable`` not ``invalid``.

    Notes
    -----
    Same as ``WindowedReduce``: the dims survive (resized to 1), so a select on one must
    not be reordered, but the chain is valid. A plain reduce's arm is ``invalid``
    instead.
    """
    kept = dim_effect(_node("mean", "lat", keepdims=True))
    assert kept.blocks == frozenset({"lat"})
    assert kept.requires == frozenset({"lat"})
    assert kept.on_conflict == "immovable"
    assert dim_effect(_node("mean", "lat")).on_conflict == "invalid"


def test_bare_mean_then_select_raises_empty_dim_bug(schema):
    """A bare ``mean()`` consumes every dim, so a following ``isel`` is invalid and must raise.

    Notes
    -----
    This is the empty-dim reorder bug: it must be caught, not silently swapped to the
    front.
    """
    plan = [_node("mean"), _node("isel", time=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, schema)


def test_an_orthogonal_dataarray_select_hops_past_a_disjoint_reduce(schema):
    """An orthogonal ``DataArray`` select is positional, so it reorders like any other select.

    Notes
    -----
    ``isel(time=<DataArray dims=(time,)>)`` becomes ``Positions``, so the select moves
    past ``mean("lat")`` (a different dim) exactly as a plain-list select would. That's
    the benefit of modelling it rather than treating it as a barrier.
    """
    da = xr.DataArray([0, 2], dims="time")
    plan = [_node("mean", "lat"), _node("isel", time=da)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]


def test_a_vectorized_dataarray_select_does_not_cross_a_following_select(schema):
    """A vectorized ``DataArray`` select is ``Opaque``, so a following select can't move past it.

    Notes
    -----
    If the first node were a plain ``reduce("lat")``, ``isel(lon=0)`` would move to the
    front (different dims). The vectorized select is ``Opaque`` and blocks it instead,
    so the dims it creates (which aren't modelled) are never relied on.
    """
    da = xr.DataArray([0, 1], dims="points")
    plan = [_node("isel", time=da), _node("isel", lon=0)]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Opaque", "Select"]


def test_dataarray_select_then_bare_mean_then_select_raises(schema):
    """An invalid chain through an orthogonal DataArray select is rejected, not laundered.

    Notes
    -----
    Misfiled as a dim-dropping ``Scalar``, ``isel(time=da)`` would drop ``time`` from
    the schema; the bare ``mean()`` would then understate ``consumes`` and the trailing
    ``isel(time=0)`` would be silently swapped in front. ``time`` survives the select
    (the orthogonal indexer resizes it), and ``ALL_DIMS`` rejects the trailing select
    outright.
    """
    da = xr.DataArray([0, 2], dims="time")
    plan = [_node("isel", time=da), _node("mean"), _node("isel", time=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, schema)


def test_bare_mean_needs_no_schema_to_reject_a_following_select():
    """A bare ``mean()`` rejects a following select even against an empty schema.

    Notes
    -----
    ``ALL_DIMS`` makes the rejection independent of any dim names: whatever dims exist
    when the reduce runs, it removes all of them, so *every* select dim intersects.
    Expanded eagerly against this empty schema, ``consumes`` would be ``frozenset()``
    and the select would be swapped in front.
    """
    plan = [_node("mean"), _node("isel", whatever=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, SchemaState())


def test_bare_mean_after_a_rename_rejects_a_select_on_the_new_name(schema):
    """A bare ``mean()`` after a ``rename`` still rejects a select on the renamed dim.

    Notes
    -----
    Shows why a bare reduce records ``ALL_DIMS`` instead of a list of names. The dims
    before the rename are ``{time, lat, lon}``, which doesn't include ``t2``. A bare
    ``mean()`` expanded against those names would look disjoint from ``isel(t2=0)``, the
    select would be moved to the front, and the plan would return data where the eager
    chain raises (``mean()`` leaves no ``t2`` to index). ``ALL_DIMS`` has no names to go
    stale. (``rename`` is modelled now, so the fold would get ``t2`` right anyway, but
    the rejection doesn't depend on that.)
    """
    plan = [_node("rename", time="t2"), _node("mean"), _node("isel", t2=0)]
    with pytest.raises(InvalidExpressionError):
        optimize(plan, schema)


def test_bare_mean_still_admits_an_empty_select(schema):
    """An empty ``isel()`` still crosses a bare ``mean()``, being the one select that can.

    Notes
    -----
    ``isel()`` names no dim, so it intersects nothing even against ``ALL_DIMS``.
    """
    plan = [_node("mean"), _node("isel")]
    assert [n.name for n in optimize(plan, schema)] == ["isel", "mean"]


def test_scan_then_select_on_scan_dim_left_untouched(schema):
    """A select on a scan's own dim is left in place, never raised: ``cumsum`` is order-sensitive, not a reduce."""
    plan = [_node("cumsum", "time"), _node("isel", time=5)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["cumsum", "isel"]


def test_scan_then_disjoint_select_hops_in_front(schema):
    """A select on a dim disjoint from a scan's hops in front: the scan runs per-position along its own dim."""
    plan = [_node("cumsum", "time"), _node("isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "cumsum"]


def test_select_reaches_the_front_past_a_reduce_and_a_scan(schema):
    """A disjoint select reaches the front past both a reduce and a scan, via the fixpoint composing both hops."""
    plan = [_node("cumsum", "time"), _node("mean", "lat"), _node("isel", lon=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "cumsum", "mean"]


def test_projection_hops_past_a_scan(schema):
    """A projection hops in front of a scan when the kept variable still carries the scanned dim."""
    plan = [_node("cumsum", "time"), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "cumsum"]
    assert out[0].variables == ("temperature",)


def test_projection_not_pushed_past_a_scan_on_missing_dim(schema):
    """A projection to a variable without the scanned dim is left behind the scan, not moved.

    Notes
    -----
    ``elevation`` has no ``time``, so putting it first would leave ``cumsum("time")``
    scanning a dim the variable doesn't have. It can't move, so it's left as written
    (not raised).
    """
    plan = [_node("cumsum", "time"), _node("__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["cumsum", "__getitem__"]


def test_select_hops_past_an_elementwise(schema):
    """A select hops in front of an elementwise op: it is per-element, so the two commute."""
    plan = [_node("fillna", 0), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "fillna"]


def test_select_composes_past_an_elementwise_and_a_reduce(schema):
    """The fixpoint moves a select past a ``fillna`` *and* a ``mean`` on another dim, combining two rules.

    Notes
    -----
    ``mean("lat") -> fillna(0) -> isel(time=0)`` ends with the select first: it crosses
    ``fillna`` (the elementwise arm) and then ``mean`` (the reduce arm). The ``fillna``
    node itself never moves; only the select passes it.
    """
    plan = [_node("mean", "lat"), _node("fillna", 0), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean", "fillna"]


def test_projection_hops_past_an_elementwise(schema):
    """A projection hops in front of an elementwise op — it needs no dim to cross one."""
    plan = [_node("astype", "float32"), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "astype"]
    assert out[0].variables == ("temperature",)


def test_projection_composes_past_an_elementwise_and_a_reduce(schema):
    """A projection reaches the front past an ``astype`` and a ``mean``, reducing only the kept variable."""
    plan = [
        _node("mean", "time"),
        _node("astype", "float32"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean", "astype"]


def test_select_hops_past_a_drop(schema):
    """A select hops in front of a ``drop_vars``: dropping a variable and selecting commute."""
    plan = [_node("drop_vars", ["elevation"]), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "drop_vars"]


def test_generic_pushdown_does_not_cross_a_drop():
    """``Drop`` answers ``requires=None``, so the generic projection pushdown never crosses it.

    Notes
    -----
    A projection moved in front of a drop could remove the very name ``drop_vars``
    targets, turning a valid chain into a ``KeyError``. So a blind swap is unsafe, and
    the generic rule refuses. The useful rewrites here (a projection that excludes the
    dropped data vars makes the drop redundant; one that keeps a dropped coord may move
    in front) depend on the folded schema, so they're in the dedicated
    :func:`~xrexpr.optimize.push_projection_past_drop` rule, not in ``dim_effect``.
    """
    assert dim_effect(_node("drop_vars", ["elevation"])).requires is None


def test_select_on_an_untouched_dim_hops_a_rename(schema):
    """A select on a dim the rename leaves alone commutes with it: untouched dims stay transparent."""
    plan = [_node("rename", {"time": "t"}), _node("isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "rename"]


def test_select_on_a_renamed_dim_is_left_behind_a_rename(schema):
    """A select on the *renamed* dim doesn't move past the rename: before it, the dim has its old name.

    Notes
    -----
    ``rename({"time": "t"}).isel(t=0)`` names ``t``, which doesn't exist before the
    rename, so the select stays where it is. The dim survives (just relabelled), so this
    is ``immovable``: left, never raised.
    """
    plan = [_node("rename", {"time": "t"}), _node("isel", t=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["rename", "isel"]


def test_generic_pushdown_does_not_cross_a_rename():
    """``Rename`` answers ``requires=None``: the generic projection pushdown never crosses it.

    Notes
    -----
    The danger is renaming a *variable*: a projection naming the new name can't go
    before the rename that creates it, and the dim-based pushdown can't see that. So no
    projection crosses via the generic rule. A schema-aware rename/projection rule, and
    translating a select's dim names across a rename (``11-relabel-rename-drop.md`` §4),
    aren't written.
    """
    assert dim_effect(_node("rename", {"tas": "temp"})).requires is None


def test_projection_hops_a_reduce_behind_a_rename(schema):
    """Because ``rename`` is modelled, the trusted prefix continues past it and a projection still crosses a reduce.

    Notes
    -----
    An ``Opaque`` would be a trust boundary that switches off the projection rules for
    the rest of the plan. A modelled ``Rename`` is inside the prefix, so the projection
    reaches the front past the ``mean``.
    """
    plan = [
        _node("rename", {"time": "t"}),
        _node("mean", "lat"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["rename", "__getitem__", "mean"]
    assert out[1].variables == ("temperature",)


def test_projection_hops_a_reduce_behind_a_drop(schema):
    """Because ``drop_vars`` is modelled, the trusted prefix continues past it and a projection still crosses a reduce.

    Notes
    -----
    As an ``Opaque``, ``drop_vars`` would be a trust boundary: ``_trusted_prefix`` would
    stop there and the schema-reading projection rules would do nothing after it. A
    modelled ``Drop`` is inside the prefix, so the projection reaches the front past the
    ``mean``. Once it's in front of the drop,
    :func:`~xrexpr.optimize.push_projection_past_drop` removes the drop of
    ``elevation``, which the projection excludes anyway. So the whole
    ``drop_vars("elevation").mean("time")[["temperature"]]`` becomes
    ``[["temperature"]].mean("time")``.
    """
    plan = [
        _node("drop_vars", ["elevation"]),
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].variables == ("temperature",)


_AUX_SCHEMA = SchemaState(
    variables={
        "temperature": ("time", "lat", "lon"),
        "elevation": ("lat", "lon"),
        "time": ("time",),
        "lat": ("lat",),
        "lon": ("lon",),
        "area": ("lat", "lon"),
    },
    coord_names=frozenset({"time", "lat", "lon", "area"}),
    sizes={"time": 4, "lat": 3, "lon": 5},
)


def test_a_redundant_drop_is_eliminated_by_a_projection(schema):
    """A ``drop_vars`` of a data variable the following projection excludes anyway is removed."""
    plan = [_node("drop_vars", ["elevation"]), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__"]
    assert out[0].variables == ("temperature",)


def test_a_drop_of_a_surviving_coord_hops_the_projection():
    """A projection hops in front of a ``drop_vars`` of a coord it retains; the drop stays after it.

    Notes
    -----
    ``area`` is a coordinate ``temperature`` still spans, so it survives
    ``[["temperature"]]`` -- the projection may go first and the ``drop_vars("area")``
    follow it, same values.
    """
    plan = [_node("drop_vars", ["area"]), _node("__getitem__", ["temperature"])]
    out = optimize(plan, _AUX_SCHEMA)
    assert [n.name for n in out] == ["__getitem__", "drop_vars"]
    assert out[0].variables == ("temperature",)
    assert out[1].variables == ("area",)


def test_a_mixed_drop_is_trimmed_to_its_surviving_coords():
    """A ``drop_vars`` of a redundant data var *and* a surviving coord hops the projection, trimmed to the coord.

    Notes
    -----
    ``elevation`` (a data variable the projection excludes) is redundant and drops out;
    ``area`` (a retained coordinate) survives, so the hopped plan keeps
    ``drop_vars("area")`` -- and ``elevation`` is never computed.
    """
    plan = [
        _node("drop_vars", ["elevation", "area"]),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, _AUX_SCHEMA)
    assert [n.name for n in out] == ["__getitem__", "drop_vars"]
    assert out[1].variables == ("area",)


def test_a_drop_then_projecting_the_dropped_name_is_left_alone(schema):
    """A chain that drops a variable and then projects it is an eager error, so the rule declines.

    Notes
    -----
    ``drop_vars("temperature")[["temperature"]]`` raises eagerly. Running the projection
    first would turn that error into a value, which the rule must never do.
    """
    plan = [_node("drop_vars", ["temperature"]), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["drop_vars", "__getitem__"]


def test_a_drop_of_an_absent_name_before_a_projection_is_elided(schema):
    """A ``drop_vars`` naming something absent is elided when the projection excludes it."""
    plan = [_node("drop_vars", ["nope"]), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__"]


def test_a_mixed_absent_drop_is_trimmed_to_its_surviving_coords():
    """Absent dropped names disappear, while retained coordinates still hop behind the projection."""
    plan = [
        _node("drop_vars", ["nope", "area"]),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, _AUX_SCHEMA)
    assert [n.name for n in out] == ["__getitem__", "drop_vars"]
    assert out[1].variables == ("area",)


def test_a_single_name_projection_does_not_absorb_a_drop(schema):
    """A bare-name projection (a ``DataArray``) is out of scope, so a preceding drop is left alone."""
    plan = [_node("drop_vars", ["elevation"]), _node("__getitem__", "temperature")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["drop_vars", "__getitem__"]


def test_nothing_reorders_around_an_unsafe_elementwise(schema):
    """A ``fillna`` with a per-variable ``dict`` records ``Opaque``, so the select stays behind it.

    Notes
    -----
    The guard records the dict-argument call as :class:`~xrexpr.ir.Opaque`, and an
    ``Opaque`` blocks everything. So only the *safe* form lets selects through; an
    unsafe one blocks them.
    """
    plan = [_node("fillna", {"temperature": 0}), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["fillna", "isel"]


def test_a_reduce_and_an_elementwise_keep_their_order(schema):
    """A reduce and an elementwise op never swap, in either order.

    Notes
    -----
    A ``mean`` doesn't commute with a ``fillna`` (filling then averaging differs from
    averaging then filling), so the optimiser must never swap them. It can't: **no rule
    moves a** :class:`~xrexpr.ir.Reduce` (a reduce is only ever the node being *crossed*
    in ``dim_effect``, never the one moving), and nothing moves an
    :class:`~xrexpr.ir.Elementwise` either. Elementwise ops let *selects and
    projections* cross them; that doesn't allow reordering them against a reduce.
    """
    assert [
        n.name for n in optimize([_node("fillna", 0), _node("mean", "lat")], schema)
    ] == [
        "fillna",
        "mean",
    ]
    assert [
        n.name for n in optimize([_node("mean", "lat"), _node("fillna", 0)], schema)
    ] == [
        "mean",
        "fillna",
    ]


def test_a_select_threads_past_a_reduce_and_elementwise_without_reordering_them(schema):
    """A select reaches the front past both a ``fillna`` and a disjoint ``mean``, leaving their order intact."""
    plan = [_node("fillna", 0), _node("mean", "lat"), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "fillna", "mean"]


def test_pushdown_composes_past_two_reduces(schema):
    """The fixpoint hops a select left one reduce at a time until it reaches the front."""
    plan = [
        _node("mean", "lat"),
        _node("mean", "lon"),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean", "mean"]


def test_pushdown_then_merge_across_a_reduce(schema):
    """A trailing ``isel`` hops past ``mean`` and then merges with the leading ``isel``."""
    plan = [
        _node("isel", time=0),
        _node("mean", "lat"),
        _node("isel", lon=2),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=0, lon=2)


def test_pushdown_projection_past_reduce(schema):
    """A projection hops in front of a reduce, so only the projected variable is reduced."""
    plan = [
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].variables == ("temperature",)


def test_pushdown_projection_past_bare_reduce(schema):
    """A projection may go before a bare ``mean()`` when the variable spans every dim the reduce would resolve to.

    Notes
    -----
    ``ALL_DIMS`` resolves against the schema going *into* the reduce, ``{time, lat,
    lon}``, which ``temperature`` spans. So the projection may go first, and the
    replayed bare ``mean()`` still reduces the same dims.
    """
    plan = [_node("mean"), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]


def test_projection_not_pushed_past_bare_reduce_on_missing_dim(schema):
    """A projection to a variable missing a dim the bare reduce would touch is left alone, never raised.

    Notes
    -----
    ``elevation`` has no ``time``, so putting it first would make the bare ``mean()``
    reduce fewer dims than it does in the plan as written. The chain is valid, it just
    can't move.
    """
    plan = [_node("mean"), _node("__getitem__", ["elevation"])]
    assert optimize(plan, schema) == plan


def test_pushdown_single_variable_projection(schema):
    """A single-variable projection (``ds["temperature"]``, a DataArray result) pushes down just the same."""
    plan = [_node("mean", "time"), _node("__getitem__", "temperature")]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].single


def test_pushdown_projection_past_select(schema):
    """A projection hops in front of a select as well as a reduce."""
    plan = [
        _node("isel", time=0),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "isel"]


def test_projection_not_pushed_past_reduce_on_missing_dim(schema):
    """A projection to a variable missing the reduced dim is left alone, and not raised either.

    Notes
    -----
    ``elevation`` has no ``time``, so ``ds[["elevation"]].mean("time")`` would raise if
    the projection moved. Unlike a select on a reduced dim, this doesn't raise; it just
    can't move.
    """
    plan = [_node("mean", "time"), _node("__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_not_pushed_past_select_on_missing_dim(schema):
    """A projection to a variable missing the selected dim is left behind the select, not hopped."""
    plan = [_node("isel", time=0), _node("__getitem__", ["elevation"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "__getitem__"]


def test_projection_of_unknown_name_left_alone(schema):
    """Projecting an untracked name (a coord, or something unmodelled) triggers no rewrite."""
    plan = [_node("mean", "time"), _node("__getitem__", ["lat"])]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_behind_an_opaque_left_alone(schema):
    """A projection is left alone once an ``Opaque`` op stands between it and the front.

    Notes
    -----
    Past an unmodelled op, the schema's ``data_vars`` is only a guess, so the rewrite
    stays out.
    """
    plan = [
        _node("rename", {"temperature": "t2m"}),
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["rename", "mean", "__getitem__"]


def test_mask_style_getitem_is_not_a_projection(schema):
    """A dict-keyed ``__getitem__`` is xarray's ``isel`` spelling, not a projection, so it stays opaque and unmoved."""
    plan = [_node("mean", "time"), _node("__getitem__", {"lat": 0})]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["mean", "__getitem__"]


def test_projection_composes_past_two_reduces(schema):
    """A projection composes past two consecutive reduces via the fixpoint."""
    plan = [
        _node("mean", "lat"),
        _node("mean", "lon"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean", "mean"]


def test_projection_and_select_pushdown_compose(schema):
    """Select and projection pushdown compose: the select hops past the reduce, the projection past both."""
    plan = [
        _node("mean", "lat"),
        _node("isel", time=0),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "isel", "mean"]


# --- merge adjacent projections -------------------------------------------------


def test_merge_adjacent_projects(schema):
    """A projection subsumed by the one before it collapses the pair to itself alone."""
    plan = [
        _node("__getitem__", ["temperature", "elevation"]),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__"]
    assert out[0].variables == ("temperature",)


def test_merge_adjacent_projects_keeps_the_single_form(schema):
    """A list projection followed by a bare name collapses to the bare name, DataArray result and all."""
    plan = [
        _node("__getitem__", ["temperature", "elevation"]),
        _node("__getitem__", "temperature"),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__"]
    assert out[0].single


def test_a_single_projection_is_a_merge_barrier(schema):
    """A *first* projection by bare name blocks the merge: what follows it indexes a ``DataArray``.

    Notes
    -----
    ``ds["temperature"]["temperature"]`` raises ``KeyError`` (``DataArray.__getitem__``
    is positional indexing, not projection), so collapsing the pair would turn an error
    into a value. Checked against xarray 2026.7.0.
    """
    plan = [
        _node("__getitem__", "temperature"),
        _node("__getitem__", "temperature"),
    ]
    assert optimize(plan, schema) == plan


def test_projections_that_are_not_a_subset_are_left_alone(schema):
    """A second projection naming a data variable the first didn't leaves the pair alone.

    Notes
    -----
    ``ds[["temperature"]][["elevation"]]`` isn't a subset (so the merge declines), and
    ``elevation`` is a data variable, not a coordinate (so
    ``eliminate_projection_before_coord`` declines). The optimiser leaves the pair
    alone: it doesn't reject chains, even though this one raises eagerly.
    """
    plan = [
        _node("__getitem__", ["temperature"]),
        _node("__getitem__", ["elevation"]),
    ]
    assert optimize(plan, schema) == plan


def test_projection_feeding_a_surviving_coord_is_dropped(schema):
    """A projection whose only consumer names a coordinate its variables span is dropped.

    Notes
    -----
    ``ds[["temperature"]]["lat"]``: ``temperature`` spans ``lat``, so the coordinate
    survives ``p1`` and reads the same from the base. ``p1`` is pure waste.
    """
    plan = [
        _node("__getitem__", ["temperature"]),
        _node("__getitem__", "lat"),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__"]
    assert out[0].variables == ("lat",)


def test_projection_feeding_a_dropped_coord_is_left(schema):
    """A projection is kept when the next one names a coordinate the kept variables don't span.

    Notes
    -----
    ``ds[["elevation"]]["time"]``: ``elevation`` only spans ``lat``/``lon``, so ``p1``
    drops ``time`` and the eager chain raises. The rule declines rather than turning
    that error into a value.
    """
    plan = [
        _node("__getitem__", ["elevation"]),
        _node("__getitem__", "time"),
    ]
    assert optimize(plan, schema) == plan


def test_projection_feeding_a_coord_past_opaque_is_left(schema):
    """Past an ``Opaque`` the schema is a guess, so the coordinate-elimination rule declines.

    Notes
    -----
    The rule reads the folded schema to prove the coordinate survives, so it stays
    inside the trusted prefix, like ``pushdown_projections``.
    """
    plan = [
        _node("pipe", lambda d: d),
        _node("__getitem__", ["temperature"]),
        _node("__getitem__", "lat"),
    ]
    assert optimize(plan, schema) == plan


def test_projection_feeding_a_coord_after_bare_name_is_left(schema):
    """A first projection by bare name gives a DataArray, so the pair is left alone.

    Notes
    -----
    After ``ds["temperature"]`` the object is a ``DataArray``, where ``["lat"]`` is
    indexing, not projection. Same ``single`` guard as the merge rule.
    """
    plan = [
        _node("__getitem__", "temperature"),
        _node("__getitem__", "lat"),
    ]
    assert optimize(plan, schema) == plan


def test_a_run_of_three_projections_collapses_to_the_last(schema):
    """The fixpoint collapses a run of three nested projections down to the innermost."""
    plan = [
        _node("__getitem__", ["temperature", "elevation"]),
        _node("__getitem__", ["temperature", "elevation"]),
        _node("__getitem__", ["elevation"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__"]
    assert out[0].variables == ("elevation",)


def test_merge_and_projection_pushdown_compose(schema):
    """A projection hops past a reduce onto the projection in front of it, and the pair then merges."""
    plan = [
        _node("__getitem__", ["temperature", "elevation"]),
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["__getitem__", "mean"]
    assert out[0].variables == ("temperature",)


def test_merging_projections_is_idempotent(schema):
    """Optimizing a plan whose projections merged is a fixpoint."""
    plan = [
        _node("__getitem__", ["temperature", "elevation"]),
        _node("__getitem__", ["temperature"]),
        _node("mean", "time"),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_projections_merge_past_an_opaque(schema):
    """The merge is syntactic, so it still fires behind an unmodelled op the schema can't see through."""
    plan = [
        _node("rename", {"temperature": "t2m"}),
        _node("__getitem__", ["t2m", "elevation"]),
        _node("__getitem__", ["t2m"]),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["rename", "__getitem__"]
    assert out[0].name == "rename"


def test_scalar_isel_past_rechunk_drops_the_spent_rechunk(schema):
    """A scalar ``isel`` past a rechunk drops the rechunk once its only named dim is gone.

    Notes
    -----
    ``chunk({time: 100}).isel(time=0)`` becomes ``isel(time=0)``: the rechunk's only
    named dim is gone, and ``chunk({})`` would buy nothing but a single-chunk array.
    """
    plan = [_node("chunk", {"time": 100}), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel"]
    assert out[0].indexer == _ix(time=0)


def test_scalar_isel_past_rechunk_strips_only_the_dropped_dim(schema):
    """A scalar ``isel`` past a rechunk strips only the dim it consumes; a surviving dim keeps its chunk spec.

    Notes
    -----
    ``lat`` survives the select, so the rechunk stays -- minus the dim that no longer
    exists.
    """
    plan = [
        _node("chunk", {"time": 100, "lat": 50}),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"lat": SingleSize(50)})
    # Replayable, with no stale ``time`` key — the header is derived from ``chunks``.
    assert emit(out)[1] == Call("chunk", ({"lat": 50},))


def test_slice_isel_pushes_with_the_spec_intact(schema):
    """A slice ``isel`` pushes past a rechunk with the chunk spec intact, since a slice keeps its dim.

    Notes
    -----
    Pushing means the rechunk sees less data *and* lands on regular blocks instead of
    ragged ones.
    """
    plan = [
        _node("chunk", {"time": 100}),
        _node("isel", time=slice(0, 2)),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"time": SingleSize(100)})


def test_select_on_unchunked_dim_is_a_plain_swap(schema):
    """A select on a dim the rechunk doesn't name is a plain swap to the front."""
    plan = [_node("chunk", {"lat": 2}), _node("isel", time=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"lat": SingleSize(2)})


def test_rechunk_kwarg_form_pushes(schema):
    """The keyword form of ``chunk`` pushes past a select just as the dict form does."""
    plan = [_node("chunk", time=100), _node("isel", lat=0)]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "chunk"]
    assert out[1].chunks == frozendict({"time": SingleSize(100)})


def test_uniform_rechunk_forms_push_and_are_kept(schema):
    """A size-independent uniform rechunk (``chunk()``, ``chunk(100)``, ``chunk(-1)``) lets selects through and is kept.

    Notes
    -----
    These forms name no dim, so there's nothing to strip and nothing to use up. xarray
    reads a uniform spec as ``dict.fromkeys(self.dims, spec)``, so it already means
    "whatever dims there are". The size-*dependent* uniform forms are the other half:
    see :func:`test_extent_dependent_uniform_rechunk_forms_are_barriers`.
    """
    for args in ((), (100,), (-1,)):
        plan = [_node("chunk", *args), _node("isel", time=0)]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["isel", "chunk"]
        assert out[1].args == args


def test_extent_dependent_uniform_rechunk_forms_are_barriers(schema):
    """``chunk("auto")``, ``chunk("10MB")`` and ``chunk((100, 400, 500))`` are barriers.

    Notes
    -----
    A uniform spec goes into ``Rechunk.uniform`` through the same
    :func:`~xrexpr.chunks.classify_chunk` as the per-dim form. So these reach
    :func:`~xrexpr.optimize._pushable_rechunk`'s ``match`` as
    :class:`~xrexpr.chunks.Auto`, :class:`~xrexpr.chunks.ByteSize` and
    :class:`~xrexpr.chunks.BlockSeq`, the same arms as their ``chunk({"time": ...})``
    forms. One set of variants covers both spellings.
    """
    for arg in ("auto", "10MB", (100, 400, 500)):
        plan = [_node("chunk", arg), _node("isel", time=0)]
        assert [n.name for n in optimize(plan, schema)] == ["chunk", "isel"]


def test_an_emptying_select_can_cross_an_auto_rechunk_after_xarray_fix(schema):
    """The 2026.09.0 gate permits only a select that provably empties an auto rechunk.

    ``[]`` is an exact empty positional indexer without consulting the source shape.
    Other extent-dependent specs and non-emptying selects remain barriers; the accessor
    suite verifies both the eager replay and the uniform ``chunk("auto")`` spelling.
    """
    plan = [_node("chunk", {"time": "auto"}), _node("isel", lat=[])]
    expected = ["isel", "chunk"] if AUTOCHUNK_EMPTY_DIMS_SAFE else ["chunk", "isel"]
    assert [node.name for node in optimize(plan, schema)] == expected


def test_a_uniform_float_size_is_an_opaque_barrier(schema):
    """``chunk(100.0)`` isn't modelled, so it's a barrier, as ``chunk({"time": 100.0})`` is.

    Notes
    -----
    Dask accepts a whole float, but xarray's API documents an int. Not modelling a
    spelling xarray doesn't offer costs one optimisation and never correctness, which is
    what :class:`~xrexpr.chunks.OpaqueChunk` is for. This checks it applies to the
    uniform form too.
    """
    plan = [_node("chunk", 100.0), _node("isel", time=0)]
    assert [n.name for n in optimize(plan, schema)] == ["chunk", "isel"]


def test_explicit_block_tuple_is_a_barrier(schema):
    """An explicit block-tuple chunk spec is a barrier no select may cross, scalar or slice.

    Notes
    -----
    The blocks must add up to the dim's length, so a select must never cross, not even a
    scalar one (for which the key would just be removed).
    """
    for indexer in ({"time": 0}, {"time": slice(0, 2)}):
        plan = [
            _node("chunk", {"time": (1, 1, 2)}),
            _node("isel", **indexer),
        ]
        out = optimize(plan, schema)
        assert [n.name for n in out] == ["chunk", "isel"]


def test_rechunk_option_kwarg_is_a_barrier(schema):
    """A rechunk carrying an option kwarg is a barrier, since a rebuilt spec couldn't carry the option faithfully."""
    plan = [
        _node("chunk", {"time": 100}, chunked_array_type="dask"),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk", "isel"]


def test_rechunk_never_raises_on_a_reduced_dim(schema):
    """Unlike a reduce, a rechunk can never make a following select unreplayable, so it never raises."""
    plan = [_node("chunk", {"time": 100}), _node("sel", time=0)]
    assert [n.name for n in optimize(plan, schema)] == ["sel"]


def test_select_reaches_the_front_past_rechunk_and_reduce(schema):
    """The fixpoint composes both pushdown rules: a select reaches the front past a rechunk and a reduce."""
    plan = [
        _node("chunk", {"time": 100}),
        _node("mean", "lat"),
        _node("isel", time=0),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]


def test_rechunk_pushdown_is_idempotent(schema):
    """Rechunk pushdown is idempotent: optimizing an already-optimized plan changes nothing."""
    plan = [
        _node("chunk", {"time": 100, "lat": 50}),
        _node("isel", time=0),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_adjacent_rechunks_over_disjoint_dims_merge(schema):
    """Two mapping-form ``chunk`` calls over disjoint dims fuse into one union node."""
    plan = [
        _node("chunk", {"time": 100}),
        _node("chunk", {"lat": 50}),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk"]
    assert out[0].chunks == frozendict({"time": SingleSize(100), "lat": SingleSize(50)})
    # Replayable mapping form, derived from the merged ``chunks``.
    assert emit(out) == [Call("chunk", ({"time": 100, "lat": 50},))]


def test_adjacent_rechunks_on_one_dim_are_later_wins(schema):
    """A repeated dim across two ``chunk`` calls takes the later spec, as dask applies it on top."""
    plan = [
        _node("chunk", {"time": 100}),
        _node("chunk", {"time": 50}),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk"]
    assert out[0].chunks == frozendict({"time": SingleSize(50)})


def test_a_later_no_change_does_not_override_an_earlier_spec(schema):
    """A later ``chunk({dim: None})`` leaves an earlier spec on that dim in place.

    Notes
    -----
    ``None`` asks nothing of the dim ("leave it as it is"), so ``chunk({"time": 100})``
    followed by ``chunk({"time": None})`` keeps ``time`` at 100. A plain ``{**r1,
    **r2}`` would wrongly replace it with :class:`~xrexpr.chunks.NoChange` (checked
    against dask).
    """
    plan = [
        _node("chunk", {"time": 100}),
        _node("chunk", {"time": None}),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk"]
    assert out[0].chunks == frozendict({"time": SingleSize(100)})


def test_a_leading_no_change_survives_when_the_dim_is_unspecified_earlier(schema):
    """A ``NoChange`` on a dim no earlier node names is carried into the merged node."""
    plan = [
        _node("chunk", {"time": 100}),
        _node("chunk", {"lat": None}),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk"]
    assert out[0].chunks == frozendict({"time": SingleSize(100), "lat": NoChange()})


def test_a_uniform_rechunk_in_either_position_blocks_the_merge(schema):
    """A uniform ``chunk(100)`` next to a per-dim ``chunk`` leaves the pair alone.

    Notes
    -----
    A uniform spec rechunks *every* dim and can't be combined by dict union, so neither
    order may merge. Same line :func:`test_uniform_rechunk_forms_push_and_are_kept`
    draws for pushdown.
    """
    for plan in (
        [_node("chunk", 100), _node("chunk", {"time": 50})],
        [_node("chunk", {"time": 50}), _node("chunk", 100)],
    ):
        assert [n.name for n in optimize(plan, schema)] == ["chunk", "chunk"]


def test_an_extent_dependent_rechunk_blocks_the_merge(schema):
    """An ``"auto"`` spec in either node barriers the merge, as it does the select pushdown."""
    for plan in (
        [_node("chunk", {"time": "auto"}), _node("chunk", {"lat": 50})],
        [_node("chunk", {"time": 100}), _node("chunk", {"lat": "auto"})],
    ):
        assert [n.name for n in optimize(plan, schema)] == ["chunk", "chunk"]


def test_a_rechunk_option_kwarg_blocks_the_merge(schema):
    """A rechunk carrying an option kwarg cannot merge, since the rebuilt node drops it."""
    plan = [
        _node("chunk", {"time": 100}, chunked_array_type="dask"),
        _node("chunk", {"lat": 50}),
    ]
    assert [n.name for n in optimize(plan, schema)] == ["chunk", "chunk"]


def test_a_run_of_three_rechunks_collapses_to_one(schema):
    """The fixpoint fuses a run of three adjacent mapping-form ``chunk`` calls into a single node."""
    plan = [
        _node("chunk", {"time": 100}),
        _node("chunk", {"lat": 50}),
        _node("chunk", {"lon": 25}),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["chunk"]
    assert out[0].chunks == frozendict(
        {"time": SingleSize(100), "lat": SingleSize(50), "lon": SingleSize(25)}
    )


def test_rechunk_merge_is_idempotent(schema):
    """Merging adjacent rechunks is idempotent: a second pass over the merged plan changes nothing."""
    plan = [
        _node("chunk", {"time": 100}),
        _node("chunk", {"lat": 50}),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_empty_plan(schema):
    """An empty plan optimizes to an empty plan."""
    assert optimize([], schema) == []


def test_optimize_is_idempotent(schema):
    """Optimizing a general plan is idempotent: a second pass changes nothing."""
    plan = [
        _node("isel", time=0),
        _node("isel", lat=1),
        _node("mean", "lon"),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


def test_optimize_is_idempotent_with_a_projection(schema):
    """Optimizing a plan that includes a projection is idempotent too."""
    plan = [
        _node("mean", "lat"),
        _node("__getitem__", ["temperature"]),
        _node("isel", time=0),
    ]
    once = optimize(plan, schema)
    assert optimize(once, schema) == once


# --- same-dim composition -------------------------------------------------------
#
# What these check: merging a run of selects with ``dict.update`` would let a second
# indexer on the same dim *replace* the first instead of composing with it. The second
# indexes positions within the first one's result, so ``isel(time=slice(100,
# 1000)).isel(time=slice(10, 20))`` is ``slice(110, 120)``, not ``slice(10, 20)``. Cases
# where composition can't be proven must end the run (two nodes) rather than merge into
# a wrong one.


@pytest.mark.parametrize(
    "outer, inner, expected",
    [
        (slice(100, 1000), slice(10, 20), slice(110, 120, 1)),
        (slice(100, None), slice(10, 20), slice(110, 120, 1)),
        (slice(None, 50), slice(10, None), slice(10, 50, 1)),
        (slice(None), slice(2, 5), slice(2, 5, 1)),
        (slice(10, 100, 2), slice(1, 4), slice(12, 18, 2)),  # every 2nd, then 3 of them
        (slice(10, 100), slice(0, 5, 3), slice(10, 15, 3)),
        (slice(100, 1000), 5, 105),  # slice then scalar -> scalar (dim drops)
        (slice(10, 20), slice(50, 60), slice(60, 20, 1)),  # past the end -> empty
        ([10, 20, 30], 1, 20),  # concrete positions: just index them
        ([10, 20, 30], slice(1, 3), [20, 30]),
        ([10, 20, 30], [2, 0], [30, 10]),
        ([10, 20, 30], -1, 30),  # negatives are exact against a known sequence
    ],
)
def test_same_dim_selects_compose(schema, outer, inner, expected):
    """Two ``isel`` calls on the same dim compose into one indexer when the composition can be proven.

    Notes
    -----
    A ``dict.update`` merge would let the second indexer *replace* the first. The second
    one indexes positions within the first one's result, so ``isel(time=slice(100,
    1000)).isel(time=slice(10, 20))`` is ``slice(110, 120)``, not ``slice(10, 20)``.
    """
    plan = [_node("isel", time=outer), _node("isel", time=inner)]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=expected)


@pytest.mark.parametrize(
    "outer, inner",
    [
        (slice(0, 10), slice(-3, None)),  # negative bound needs the dim length
        (slice(0, 10), slice(None, None, -1)),  # reversal needs the dim length
        (slice(-5, None), slice(0, 2)),
        (slice(0, 10), -1),  # negative scalar into a bounded slice
        (slice(0, 10), 20),  # out of the outer slice's range: xarray should raise
        (0, 0),  # scalar outer already dropped the dim
        ([10, 20, 30], 7),  # out of range against a known sequence
    ],
)
def test_uncomposable_same_dim_selects_are_left_separate(schema, outer, inner):
    """Two same-dim ``isel`` calls with no provable composition are left as two separate nodes, not folded wrongly."""
    plan = [_node("isel", time=outer), _node("isel", time=inner)]
    out = optimize(plan, schema)
    assert [n.indexer for n in out] == [_ix(time=outer), _ix(time=inner)]


def test_same_dim_sel_is_never_composed(schema):
    """Two same-dim ``sel`` calls are never composed, since label indexers would need coordinate values we don't have."""
    plan = [
        _node("sel", time=slice(0, 10)),
        _node("sel", time=slice(2, 4)),
    ]
    assert len(optimize(plan, schema)) == 2


def test_uncomposable_dim_abandons_the_whole_merge(schema):
    """One dim that can't be composed abandons the merge for every dim in the pair.

    Notes
    -----
    ``lat`` *could* merge here, but ``time`` can't, so neither does. Otherwise the plan
    would represent neither select correctly.
    """
    plan = [
        _node("isel", time=0, lat=slice(0, 3)),
        _node("isel", time=0, lat=1),
    ]
    out = optimize(plan, schema)
    assert len(out) == 2
    assert out[0].indexer == _ix(time=0, lat=slice(0, 3))


def test_composed_run_of_three_on_one_dim(schema):
    """A run of three ``isel`` calls on the same dim composes into one: slice, then slice, then scalar."""
    plan = [
        _node("isel", time=slice(100, 1000)),
        _node("isel", time=slice(10, 20)),
        _node("isel", time=2),
    ]
    out = optimize(plan, schema)
    assert len(out) == 1
    assert out[0].indexer == _ix(time=112)
    assert out[0].consumes == frozenset({"time"})


def test_composition_survives_pushdown_past_a_reduce(schema):
    """A trailing ``isel`` hops past ``mean``, then composes with the leading one on the same dim."""
    plan = [
        _node("isel", time=slice(1, 4)),
        _node("mean", "lat"),
        _node("isel", time=1),
    ]
    out = optimize(plan, schema)
    assert [n.name for n in out] == ["isel", "mean"]
    assert out[0].indexer == _ix(time=2)


# --- weighted reduces: the node no select may cross (02-lowering §8.1) ----------------


def _weighted(
    reduce="mean", consumes=frozenset({"lat"}), weight_dims=frozenset({"lat"})
):
    """A ``WeightedReduce`` as ``to_lower_ir`` would build it, weights payload included."""
    weights = xr.DataArray([1.0, 2.0, 3.0], dims="lat")
    return WeightedReduce(
        name="weighted",
        reduce=reduce,
        weight_dims=weight_dims,
        args=(weights,),
        reduce_args=(sorted(consumes),),
        consumes=consumes,
    )


@pytest.mark.parametrize(
    "trailing",
    [
        # the rewrite the variant exists to block: hopping this left would run the
        # weighted mean over one latitude with the *weights* left un-subset
        lambda: _node("isel", time=0),
        lambda: _node("chunk", {"time": 2}),
    ],
)
def test_no_select_or_rechunk_crosses_a_weighted_reduce(schema, trailing):
    """Neither a select nor a rechunk ever crosses a ``WeightedReduce``, unlike a plain reduce.

    Notes
    -----
    ``WeightedReduce.consumes`` is exactly a plain reduce's, so as a ``Reduce`` it would
    look the same and ``pushdown_selects`` would match it. It's a separate kind so that
    no select can. Subsetting the weights too would be a rewrite that changes array
    data, which needs its own design (§8.1). A *projection* does cross it (see
    :func:`test_a_projection_hops_past_a_weighted_reduce`), because it removes variables
    rather than subsetting them, so the weights don't need rewriting.
    """
    plan = [_weighted(), trailing()]
    assert optimize(plan, schema) == plan


def test_a_select_on_a_weighted_reduces_own_dim_is_left_not_raised(schema):
    """A select on a ``WeightedReduce``'s own dim is left in place, not raised.

    Notes
    -----
    ``pushdown_selects`` raises on an overlapping ``(Reduce, Select)`` because that plan
    can never run. That's deliberately not reachable here: the ``invalid`` answer is
    only given for ``Reduce``, so xarray reports the bad selection at replay in its own
    words, instead of the optimiser guessing about a node it models less exactly.
    """
    plan = [_weighted(consumes=frozenset({"lat"})), _node("isel", lat=0)]
    assert optimize(plan, schema) == plan


def test_a_bare_weighted_closer_is_left_alone_too(schema):
    """A bare (``ALL_DIMS``) ``WeightedReduce`` is left alone by select pushdown too.

    Notes
    -----
    ``ALL_DIMS`` is the case most likely to tempt a rule to fire, since it's what
    ``pushdown_selects``'s schema-free reasoning keys on.
    """
    plan = [
        WeightedReduce(name="weighted", reduce="mean", consumes=ALL_DIMS),
        _node("isel", time=0),
    ]
    assert optimize(plan, schema) == plan


def test_ops_after_a_weighted_reduce_are_still_optimised(schema):
    """Ops after a ``WeightedReduce`` still rewrite normally, since the fused node isn't an opaque barrier.

    Notes
    -----
    Fusing the pair into one node means there's no ``Opaque`` at the weighted step, so
    the plan after it is inside the trusted prefix and rewrites normally. The projection
    doesn't stop at the trailing reduce either: ``temperature`` has both the consumed
    and the weight dims, so it reaches the front and the weighted mean runs over one
    variable instead of two.
    """
    plan = [
        _weighted(consumes=frozenset({"lat"})),
        _node("mean", "time"),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "WeightedReduce", "Reduce"]


# --- select pushdown past the fused reduces -------------------------------------------


def _grouped(group_dim="time", new_dim="month", consumes=frozenset()):
    """A ``GroupedReduce`` as ``to_lower_ir`` would build it from ``groupby(...).mean()``."""
    return GroupedReduce(
        name="groupby",
        group_dim=group_dim,
        new_dim=new_dim,
        reduce="mean",
        args=(f"{group_dim}.{new_dim}" if group_dim != new_dim else group_dim,),
        consumes=consumes,
    )


def _windowed(name="rolling", window=frozendict({"time": 3})):
    """A ``WindowedReduce`` as ``to_lower_ir`` would build it from ``rolling``/``coarsen``."""
    return WindowedReduce(name=name, reduce="mean", window=window, kwargs=dict(window))


@pytest.mark.parametrize(
    ("fused", "select"),
    [
        # the headline: a climatology over one latitude instead of all of them
        (_grouped(), lambda: _node("isel", lat=0)),
        (_grouped(), lambda: _node("isel", lat=[0, 2])),
        (_grouped(), lambda: _node("sel", lon=slice(1, 3))),
        (_grouped(group_dim="lat", new_dim="lat"), lambda: _node("isel", lon=0)),
        (_windowed(), lambda: _node("isel", lat=0)),
        (
            _windowed(name="coarsen", window=frozendict({"time": 2})),
            lambda: _node("isel", lat=0),
        ),
    ],
)
def test_select_hops_past_a_fused_reduce_on_disjoint_dims(schema, fused, select):
    """A select on a dim disjoint from a fused reduce (grouped, windowed, coarsened) hops in front of it."""
    out = optimize([fused, select()], schema)
    assert [type(n).__name__ for n in out] == [
        "Select",
        type(fused).__name__,
    ]


@pytest.mark.parametrize(
    ("fused", "select", "why"),
    [
        (_grouped(), lambda: _node("isel", month=0), "the minted dim"),
        (_grouped(), lambda: _node("isel", time=0), "the consumed group dim"),
        (
            _grouped(consumes=frozenset({"lat"})),
            lambda: _node("isel", lat=0),
            "a dim the closer also reduced",
        ),
        (
            _grouped(group_dim="lat", new_dim="lat"),
            lambda: _node("isel", lat=0),
            "group dim and minted dim coincide",
        ),
        (_windowed(), lambda: _node("isel", time=0), "a windowed dim"),
        (
            _windowed(name="coarsen", window=frozendict({"time": 2})),
            lambda: _node("isel", time=0),
            "a coarsened dim, whose size the node changes",
        ),
        (_grouped(), lambda: _node("isel", lat=0, month=1), "one dim of several"),
    ],
)
def test_intersecting_select_is_left_alone_never_raised(schema, fused, select, why):
    """A select overlapping a fused reduce's dims is left alone, never raised, whichever dim it names.

    Notes
    -----
    This follows the scan behaviour, not ``pushdown_selects``' reduce behaviour. Three
    of these cases are *valid* eagerly (selecting the new dim, or a dim a window kept),
    so raising would reject a working chain. The group-dim case is invalid, and leaving
    it lets xarray report that at replay rather than the optimiser guessing. Either way:
    leave, never raise.
    """
    plan = [fused, select()]
    assert optimize(plan, schema) == plan, why


def test_select_reaches_the_front_past_a_fused_reduce_and_a_reduce(schema):
    """A select reaches the front past both a grouped reduce and a plain reduce, via the fixpoint composing both rules."""
    plan = [_grouped(), _node("mean", "lat"), _node("isel", lon=0)]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Select", "GroupedReduce", "Reduce"]


def test_a_hopped_select_then_merges_with_the_one_in_front_of_it(schema):
    """A select that moves past a grouped reduce then merges with the select already in front of it.

    Notes
    -----
    This combines with ``merge_adjacent_selects``, which is the point of running small
    rules in a fixpoint loop.
    """
    plan = [_node("isel", lat=0), _grouped(), _node("isel", lon=1)]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Select", "GroupedReduce"]
    assert out[0].indexer == _ix(lat=0, lon=1)


def test_no_select_hops_past_a_weighted_reduce(schema):
    """No select moves past a ``WeightedReduce``, even when the dims alone would allow it.

    Notes
    -----
    This is exactly what the node kind exists to block: the dims would allow the move,
    but the weights would be left un-subset. ``dim_effect`` gives a weighted reduce
    ``blocks=None``, so the rule can't fire.
    """
    plan = [_weighted(consumes=frozenset({"lat"})), _node("isel", time=0)]
    assert optimize(plan, schema) == plan


# --- projection pushdown past the fused reduces ---------------------------------------


@pytest.mark.parametrize(
    "fused",
    [
        _grouped(),
        _grouped(group_dim="lat", new_dim="lat"),
        _windowed(),
        _windowed(name="coarsen", window=frozendict({"time": 2})),
    ],
)
def test_projection_hops_past_a_fused_reduce_when_the_dims_survive(schema, fused):
    """A projection moves past a fused reduce when the kept variable has every dim the reduce names.

    Notes
    -----
    ``temperature(time, lat, lon)`` has every dim these name, so projecting first
    aggregates one variable instead of aggregating two and throwing one away.
    """
    plan = [fused, _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", type(fused).__name__]


@pytest.mark.parametrize(
    ("fused", "why"),
    [
        (_grouped(), "the group dim is not carried by the projected variable"),
        (
            _grouped(group_dim="time", new_dim="time"),
            "same, for a self-minting grouper",
        ),
        (_windowed(), "the window dim is not carried by the projected variable"),
        (
            _grouped(group_dim="lat", new_dim="lat", consumes=frozenset({"time"})),
            "the group dim survives but the extra consumes does not",
        ),
    ],
)
def test_projection_is_left_when_the_fused_node_would_lose_its_dim(schema, fused, why):
    """A projection that would drop a dim the fused reduce needs is left alone, never raised.

    Notes
    -----
    ``elevation(lat, lon)`` has no ``time``. Projecting first would drop the ``time``
    coord entirely (no remaining variable uses it), and every fused kind then *raises*
    rather than quietly doing nothing. Left alone, not raised: the chain is valid
    eagerly, just not reorderable.
    """
    plan = [fused, _node("__getitem__", ["elevation"])]
    assert optimize(plan, schema) == plan, why


def test_projection_needed_set_excludes_the_minted_dim(schema):
    """A projection's required dims exclude a grouped reduce's new dim, unlike a select's blocked dims.

    Notes
    -----
    A grouped reduce *creates* ``month``, but the projection would go before it, where
    ``month`` doesn't exist yet. Requiring the kept variables to have it would block
    every move. ``DimEffect.blocks`` includes it, because a select comes from the other
    side; ``DimEffect.requires`` must not.
    """
    plan = [_grouped(new_dim="month"), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "GroupedReduce"]


def test_a_projection_hops_past_a_weighted_reduce(schema):
    """A projection moves past a ``WeightedReduce``, turning a plan that would raise into one that runs.

    Notes
    -----
    The clearest case of §8. As written this chain *raises*: a weighted reduce runs per
    variable, and ``elevation`` has no ``time`` (a plain ``.mean("time")`` would just do
    wasted work on it). Moving the projection first removes ``elevation`` before the
    reduce sees it, so the plan returns the value it asked for instead of an error about
    work it was discarding.
    """
    plan = [
        _weighted(consumes=frozenset({"time"}), weight_dims=frozenset({"lat"})),
        _node("__getitem__", ["temperature"]),
    ]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "WeightedReduce"]


def test_a_projection_is_left_when_it_would_orphan_a_weight_dim(schema):
    """A projection that would drop a coord the weights align against is left alone.

    Notes
    -----
    The guard that makes :func:`test_a_projection_hops_past_a_weighted_reduce` safe, and
    the one case where a weighted reduce requires more than a plain reduce.
    ``elevation`` has no ``time``, so projecting to it drops the ``time`` *coordinate*.
    The weights have ``time``, so they'd switch from inner-joining against that coord to
    broadcasting a new one. Checked against xarray: that changes values, not just
    errors. Remove the ``weight_dims`` term from ``requires`` and this plan wrongly
    moves.
    """
    plan = [
        _weighted(consumes=frozenset({"lat"}), weight_dims=frozenset({"time"})),
        _node("__getitem__", ["elevation"]),
    ]
    assert optimize(plan, schema) == plan


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        # ``temperature`` spans every dim, so the replayed bare closer reduces the same
        # ones and no coord can go missing -- a bare closer clears the weight dims too
        (["temperature"], ["Project", "WeightedReduce"]),
        # ``elevation`` does not span ``time``, so the bare closer would reduce a
        # different set of dims after the hop
        (["elevation"], ["WeightedReduce", "Project"]),
    ],
)
def test_a_bare_weighted_closer_admits_a_projection_spanning_every_dim(
    schema, variables, expected
):
    """A bare (``ALL_DIMS``) ``WeightedReduce`` lets through a projection spanning every dim, without a ``weight_dims`` term.

    Notes
    -----
    ``ALL_DIMS`` needs no ``weight_dims`` term: a bare closer removes every dim, new
    weight dims included, so no coordinate is left to be missing. It couldn't have one
    anyway: a weight dim the dataset lacks can never be a subset of the kept variables'
    dims, so requiring it would block every move.
    """
    plan = [
        WeightedReduce(name="weighted", reduce="mean", consumes=ALL_DIMS),
        _node("__getitem__", variables),
    ]
    assert [type(n).__name__ for n in optimize(plan, schema)] == expected


def test_projection_and_select_both_cross_a_fused_reduce(schema):
    """Select and projection pushdown compose through the fixpoint, both ending up in front of a grouped reduce."""
    plan = [_grouped(), _node("isel", lat=0), _node("__getitem__", ["temperature"])]
    out = optimize(plan, schema)
    assert [type(n).__name__ for n in out] == ["Project", "Select", "GroupedReduce"]


# --- the derived dim effect -----------------------------------------------------------


@pytest.mark.parametrize(
    ("node", "blocks", "requires"),
    [
        # a reduce removes exactly what it names, so both answers coincide
        (_node("mean", "lat"), frozenset({"lat"}), frozenset({"lat"})),
        (_node("mean"), ALL_DIMS, ALL_DIMS),
        # a select is never crossed by another select, but is by a projection
        (_node("isel", time=0), None, frozenset({"time"})),
        # the asymmetry the two fields exist for: ``new_dim`` blocks a select coming
        # from the right, but is *not* required of the input a projection would leave
        # behind
        (_grouped(), frozenset({"time", "month"}), frozenset({"time"})),
        (
            _grouped(consumes=frozenset({"lat"})),
            frozenset({"time", "month", "lat"}),
            frozenset({"time", "lat"}),
        ),
        # a window resizes what it names and needs the same dims present
        (_windowed(), frozenset({"time"}), frozenset({"time"})),
        # a scan keeps its dims but is order-significant: a disjoint select hops
        # (blocks), a projection needing those dims hops too (requires), an intersecting
        # select stays
        (_node("cumsum", "time"), frozenset({"time"}), frozenset({"time"})),
        # nothing crosses these, in either direction
        (_node("where", "cond"), None, None),
        (_node("chunk", {"time": 2}), None, None),
        (_node("__getitem__", ["temperature"]), None, None),
        # A select never crosses a weighted reduce (the weights would need subsetting),
        # but a projection does, and it must cover the weight dims as well as the
        # consumed ones, or it could drop the coord the weights align against.
        (_weighted(consumes=frozenset({"time"})), None, frozenset({"time", "lat"})),
        (
            WeightedReduce(name="weighted", reduce="mean", consumes=ALL_DIMS),
            None,
            ALL_DIMS,
        ),
    ],
)
def test_dim_effect_table(node, blocks, requires):
    """``dim_effect`` reports the ``blocks``/``requires`` pair the parametrize table expects for each node kind."""
    effect = dim_effect(node)
    assert effect.blocks == blocks
    assert effect.requires == requires


def test_only_a_plain_reduce_calls_a_conflict_invalid():
    """Of every lowered node kind, only a plain ``Reduce`` treats an overlapping select as invalid.

    Notes
    -----
    A reduce *removes* what it names, so a select on one of those dims can never run,
    and the optimiser may reject the chain. Every other kind either creates or keeps the
    dims it blocks, so an overlapping select there just can't move. That split is what
    lets one rule handle both, and a new kind has to get it right.
    """
    invalid = [
        node
        for node in (
            _node("mean", "lat"),
            _grouped(),
            _windowed(),
            _weighted(),
            _node("cumsum", "time"),
        )
        if dim_effect(node).on_conflict == "invalid"
    ]
    assert invalid == [_node("mean", "lat")]


def test_dim_effect_covers_every_lowered_variant():
    """``dim_effect`` covers every member of the ``LoweredOp`` union, checked at runtime as well as by ``assert_never``.

    Notes
    -----
    ``assert_never`` already makes a missing kind a type error. This checks it at
    runtime too, so the table above can't quietly stop covering the whole union.
    """
    from typing import get_args

    from xrexpr.ir import LoweredOp

    covered = {
        type(node)
        for node in (
            _node("mean", "lat"),
            _node("isel", time=0),
            _node("cumsum", "time"),
            _node("fillna", 0),
            _node("__getitem__", ["temperature"]),
            _node("drop_vars", ["region"]),
            _node("rename", {"time": "t"}),
            _node("chunk", {"time": 2}),
            _node("where", "cond"),
            _grouped(),
            _windowed(),
            _weighted(),
        )
    }
    assert covered == set(get_args(LoweredOp))
