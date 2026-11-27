"""What ``SchemaState.sizes`` means, and what it's for.

``sizes`` is the one stored field no rewrite may read. Every rule in ``optimize``
reasons about dim *names*, and ``test_rewrites_survive_unknown_dim_sizes`` checks that
by setting every size to ``None`` and requiring identical output. So a wrong size can't
cause a wrong answer: no code path goes from a size to a rewrite.

So why store sizes at all? **Because a size is a check on the arm that produced it.** An
arm's job is to describe what an op does, and "what shape comes out" is the one part of
that you can check against reality cheaply and exactly. A schema that only tracked names
couldn't be checked at all for ops that keep every dim but change a length, which is the
case ``test_diff_shrinks_its_dim`` (below) covers.

So sizes catch modelling mistakes early; they don't make rewrites correct. There are two
separate checks, catching different things:

===========================  ====================================  ====================
check                        catches                               how
===========================  ====================================  ====================
this module + size property  the schema **getting a shape wrong**  tracked == real sizes
``optimised == eager``       a rule **reordering wrongly**         run both, compare
===========================  ====================================  ====================

Sizes say nothing about whether ops commute. ``diff("time")`` doesn't commute with a
prefix slice while ``cumsum("time")`` does, and nothing here would notice. That's the
equality property's job.

**Structure.** The per-indexer size arithmetic (how a mask, a list or a slice is sized)
is in ``test_schema.py`` with the other ``apply_schema`` arms. This module is about the
*model*: which ops are sized exactly, which decline to answer and why, how the field is
normalised, and what it catches. Every case shows the xarray code it stands for, because
"the schema says ``time`` is 2" can only be checked against a real dataset.
"""

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.lower import to_lower_ir
from xrexpr.schema import SchemaState, apply_schema, to_opnode


@pytest.fixture
def ds() -> xr.Dataset:
    """``time`` is a real datetime index, which ``resample`` and ``time.month`` need.

    .. code-block:: python

        xr.Dataset(
            {"tas": (("time", "lat"), ...),   # 4 x 3
             "elev": (("lat",), ...)},        # 3     -- deliberately missing ``time``
            coords={"time": <4 daily dates>, "lat": [0, 1, 2],
                    "region": ("lat", [0.0, 1.0, 1.0])},   # non-dim coord on ``lat``
        )
    """
    return xr.Dataset(
        {
            "tas": (("time", "lat"), np.arange(12.0).reshape(4, 3)),
            "elev": (("lat",), np.arange(3.0)),
        },
        coords={
            "time": xr.date_range("2000-01-01", periods=4, freq="D"),
            "lat": np.arange(3),
            "region": ("lat", [0.0, 1.0, 1.0]),
        },
    )


def _fold(ds, calls):
    """Fold the schema over ``calls`` the way the pipeline does, and replay them eagerly.

    Returns ``(tracked_sizes, real_sizes)``. Goes through ``to_opnode`` and
    ``to_lower_ir`` rather than hand-building nodes so that a builder *pair* is one case
    like any other, and so the cases below exercise the same path ``collect()`` takes.
    """
    plan = [to_opnode(name, args, kwargs) for name, args, kwargs in calls]
    schema = SchemaState.from_dataset(ds)
    for node in to_lower_ir(plan, schema.dim_names):
        schema = apply_schema(schema, node)

    obj = ds
    for name, args, kwargs in calls:
        obj = getattr(obj, name)(*args, **kwargs)
    return dict(schema.sizes), dict(obj.sizes)


# --- sized exactly -------------------------------------------------------------------
# The size depends only on the *call*, so the schema can compute it without reading any
# values. A failure in this table would mean the schema has an op's effect on shape
# wrong.

EXACT = [
    # (label -- the xarray code this stands for, calls)
    ("ds.mean('lat')", [("mean", ("lat",), {})]),
    ("ds.mean()", [("mean", (), {})]),
    ("ds.isel(time=0)", [("isel", (), {"time": 0})]),
    ("ds.isel(time=slice(0, 2))", [("isel", (), {"time": slice(0, 2)})]),
    ("ds.isel(lat=[0, 2])", [("isel", (), {"lat": [0, 2]})]),
    (
        "ds.isel(time=[True, False, True, True])",
        [("isel", (), {"time": [True, False, True, True]})],
    ),
    ("ds[['tas']]", [("__getitem__", (["tas"],), {})]),
    ("ds[['elev']]", [("__getitem__", (["elev"],), {})]),
    ("ds.chunk({'time': 2})", [("chunk", ({"time": 2},), {})]),
    ("ds.cumsum('time')", [("cumsum", ("time",), {})]),
    (
        "ds.coarsen(time=2, boundary='trim').mean()",
        [("coarsen", (), {"time": 2, "boundary": "trim"}), ("mean", (), {})],
    ),
    (
        # boundary='pad' ceils: lat 3 → 2, where 'trim' would floor to 1 — the ceil
        # branch.
        "ds.coarsen(lat=2, boundary='pad').mean()",
        [("coarsen", (), {"lat": 2, "boundary": "pad"}), ("mean", (), {})],
    ),
    ("ds.rolling(time=2).mean()", [("rolling", (), {"time": 2}), ("mean", (), {})]),
    (
        "ds.mean('lat').isel(time=0)",
        [("mean", ("lat",), {}), ("isel", (), {"time": 0})],
    ),
    (
        "ds.isel(time=slice(0, 2)).mean('lat')",
        [("isel", (), {"time": slice(0, 2)}), ("mean", ("lat",), {})],
    ),
]


@pytest.mark.parametrize(("label", "calls"), EXACT, ids=[c[0] for c in EXACT])
def test_sizes_are_exact_when_the_call_determines_them(ds, label, calls):
    """Tracked extents equal a real evaluation's, for every op whose shape the call fixes.

    The ``label`` *is* the xarray expression, so a failure names the code that broke
    rather than a node repr. Read the parametrisation table as the specification: those
    are the ops whose output shape this layer claims to know exactly.
    """
    tracked, real = _fold(ds, calls)
    assert tracked == real


# --- declines to answer --------------------------------------------------------------
# The size depends on coordinate *values*, which this layer never reads. ``None`` is the
# correct answer, and the reason sizes are ``int | None``.

UNKNOWN = [
    (
        "ds.groupby('lat').mean()",
        [("groupby", ("lat",), {}), ("mean", (), {})],
        "lat",
        "the group count is a fact about coordinate values",
    ),
    (
        "ds.groupby('time.month').mean()",
        [("groupby", ("time.month",), {}), ("mean", (), {})],
        "month",
        "likewise: how many distinct months is a fact about the index",
    ),
    (
        "ds.resample(time='2D').mean()",
        [("resample", (), {"time": "2D"}), ("mean", (), {})],
        "time",
        "the bin count depends on the index's span, not on the frequency string",
    ),
    (
        "ds.sel(time=slice('2000-01-01', '2000-01-03'))",
        [("sel", (), {"time": slice("2000-01-01", "2000-01-03")})],
        "time",
        "a label slice needs the coordinate's values to resolve to positions",
    ),
]


@pytest.mark.parametrize(
    ("label", "calls", "dim", "why"), UNKNOWN, ids=[c[0] for c in UNKNOWN]
)
def test_sizes_decline_to_guess_when_the_extent_needs_values(
    ds, label, calls, dim, why
):
    """``None`` where the extent is a fact about data, and the dim *name* still exact.

    Each case has a real, knowable answer — the reason it is ``None`` is that finding it
    would mean reading coordinate values, which this layer is defined not to do. What
    must still hold is the part the rewrites actually use: the surviving dim names.
    """
    tracked, real = _fold(ds, calls)
    assert tracked[dim] is None, why
    assert real[dim] is not None  # anti-vacuity: there *was* an answer to decline
    assert set(tracked) == set(real)  # names exact even where extents are not


def test_unknown_is_never_reported_as_zero(ds):
    """The under-report ``None`` exists to prevent.

    .. code-block:: python

        ds.sel(lat=slice(20, 30))   # lat is [0, 1, 2] -- selects nothing, legally

    ``lat``'s coordinate is integers, so the bounds look the same as positions, and
    sizing by arithmetic would give **0** against a length-3 dim. Zero is the one wrong
    answer a size-based rule could act on (dropping the whole computation as provably
    empty), which is why the contract is "unknown", never "empty". (``time`` can't show
    this: pandas rejects integer bounds on a ``DatetimeIndex``, so the ambiguity only
    exists where the labels really are numbers.)
    """
    tracked, _ = _fold(ds, [("sel", (), {"lat": slice(20, 30)})])
    assert tracked["lat"] is None
    assert tracked["lat"] != 0
    assert "lat" in tracked  # the dim exists; only its extent is unknown


# --- what sizes catch ----------------------------------------------------------------


def test_diff_shrinks_its_dim(ds):
    """``ds.diff('time')`` drops one element along ``time``, and the schema says so.

    .. code-block:: python

        ds.diff("time").sizes    # {'time': 3, 'lat': 3}   -- one shorter, tracked right

    **This is the kind of bug sizes exist to catch.** ``diff`` and ``cumsum`` are the
    same node type (``Scan``), and a names-only schema can't tell them apart: neither
    removes a dim, so ``set(tracked) == set(real)`` passes for both however wrong the
    arm is. Only comparing sizes tells them apart.

    What this does *not* catch, since the two are easy to mix up: ``diff`` also doesn't
    commute with a prefix slice, where ``cumsum`` does. That's about *order*, and is
    caught by ``optimised == eager``, not by anything here.
    """
    tracked, real = _fold(ds, [("diff", ("time",), {})])
    assert tracked == real


def test_diff_shrinks_its_dim_by_n(ds):
    """``ds.diff('time', n=2)`` drops two elements along ``time``, tracked to match.

    The size arm reads ``n`` (the schema shrinks ``time`` by exactly ``n``), so a
    second-difference is distinguished from a first, not merely from a shape-transparent
    ``cumsum``.
    """
    tracked, real = _fold(ds, [("diff", ("time",), {"n": 2})])
    assert tracked == real


def test_cumsum_really_is_shape_transparent(ds):
    """The other half of the pair: ``cumsum`` really doesn't change any size.

    .. code-block:: python

        ds.cumsum("time").sizes   # {'time': 4, 'lat': 3}  -- unchanged, and tracked
        right

    The ``Scan`` arm's size-preserving ``pass`` is *correct* for ``cumsum``. Only
    ``diff`` changes a length.
    """
    tracked, real = _fold(ds, [("cumsum", ("time",), {})])
    assert tracked == real


# --- the field's own contract --------------------------------------------------------


def test_sizes_are_pruned_to_the_dims_that_exist():
    """An extent for a dim no variable spans is dropped on construction.

    ``dim_names`` derives from ``variables``, so a stray extent is already unreachable
    through it. Pruning matters for a different reason: ``SchemaState`` is a *value*,
    and a stale key would make two semantically identical snapshots compare unequal.
    """
    schema = SchemaState(
        variables={"tas": ("time",), "time": ("time",)},
        coord_names=frozenset({"time"}),
        sizes={"time": 4, "ghost": 99},  # nothing spans ``ghost``
    )
    assert dict(schema.sizes) == {"time": 4}
    assert schema.dim_names == frozenset({"time"})


def test_equal_schemas_compare_equal_despite_stale_extents():
    """Why the pruning is not merely tidiness.

    ``test_rewrites_survive_unknown_dim_sizes`` compares two *folded* schemas for
    equality, so any path that left a stale extent behind would break an assertion about
    rewrites while having nothing to do with rewrites.
    """
    without = SchemaState(
        variables={"time": ("time",)},
        coord_names=frozenset({"time"}),
        sizes={"time": 4},
    )
    with_stale = SchemaState(
        variables={"time": ("time",)},
        coord_names=frozenset({"time"}),
        sizes={"time": 4, "ghost": 1},
    )
    assert without == with_stale


def test_blanking_every_extent_changes_no_dim_name(ds):
    """The rule that rewrites can't read sizes, shown on a small example.

    The generated version (``test_rewrites_survive_unknown_dim_sizes``) checks the
    strong form: ``optimize`` gives identical output with every size blanked. This
    checks the same thing one step earlier: blanking sizes can't even change the *fold*,
    because no arm branches on a size's value.
    """
    base = SchemaState.from_dataset(ds)
    blanked = SchemaState(
        variables=base.variables,
        coord_names=base.coord_names,
        sizes=frozendict(dict.fromkeys(base.sizes)),
    )
    node = to_opnode("mean", ("lat",), {})
    assert apply_schema(blanked, node).dim_names == apply_schema(base, node).dim_names
    assert all(size is None for size in apply_schema(blanked, node).sizes.values())
