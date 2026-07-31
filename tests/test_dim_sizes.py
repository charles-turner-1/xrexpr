"""What ``SchemaState.sizes`` means, and what it is for.

``sizes`` is the one stored field that no rewrite may read. Every rule in ``optimize``
reasons about dim *names*, and ``test_rewrites_survive_unknown_dim_sizes`` pins that by
blanking every extent to ``None`` and demanding byte-identical output. So a wrong size
cannot produce a wrong answer — there is no code path from an extent to a rewrite. Even
the one rule that reasons about *emptiness* (``pushdown_selects_past_rechunks``, issue
#121) reads the indexer rather than the extent.

Which raises the obvious question this module exists to answer: **why store extents at
all?** Because an extent is a *checksum on the arm that produced it*. An arm's real job is
to state what an op does, and "what shape comes out" is the only part of that statement
which can be checked against reality cheaply and exactly. A schema that tracked names alone
would be unfalsifiable for the whole class of ops that keep every dim and change an extent —
which is where the model has actually been wrong (see ``test_diff_shrinks_its_dim`` below).

So sizes buy early detection, not correctness. The two guards are separate and catch
different things:

===========================  ====================================  ====================
guard                        catches                               how
===========================  ====================================  ====================
this module + size property  the schema **mismodelling a shape**   tracked == real sizes
``optimised == eager``       a rule **mis-deciding an order**      run both, compare
===========================  ====================================  ====================

Sizes say nothing about commutativity. ``diff("time")`` does not commute with a prefix
slice while ``cumsum("time")`` does, and no assertion here would notice; that is the
equality property's job.

**Structure.** The per-indexer size arithmetic (how a boolean mask, a list, a slice each
size) lives in ``test_schema.py`` alongside the other ``apply_schema`` arms. This module is
about the *model*: which ops are sized exactly, which decline to answer and why, how the
field is canonicalised, and what it catches. Every case carries the xarray code it stands
for, because "the schema says ``time`` is 2" is only checkable against a real dataset.
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

    Returns ``(tracked_sizes, real_sizes)``. Goes through ``to_opnode`` and ``to_lower_ir``
    rather than hand-building nodes so that a builder *pair* is one case like any other,
    and so the cases below exercise the same path ``collect()`` takes.
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
# The extent is a function of the *call* alone, so the schema can compute it without
# reading a single value. Anything in this table that regressed would be a real
# mismodelling of what the op does to shape.

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

    The ``label`` *is* the xarray expression, so a failure names the code that broke rather
    than a node repr. Read the parametrisation table as the specification: those are the ops
    whose output shape this layer claims to know exactly.
    """
    tracked, real = _fold(ds, calls)
    assert tracked == real


# --- declines to answer --------------------------------------------------------------
# The extent depends on coordinate *values*, which this layer never reads. ``None`` is the
# honest answer and the whole reason sizes are ``int | None``.

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
    would mean reading coordinate values, which this layer is defined not to do. What must
    still hold is the part the rewrites actually use: the surviving dim names.
    """
    tracked, real = _fold(ds, calls)
    assert tracked[dim] is None, why
    assert real[dim] is not None  # anti-vacuity: there *was* an answer to decline
    assert set(tracked) == set(real)  # names exact even where extents are not


def test_unknown_is_never_reported_as_zero(ds):
    """The under-report that ``None`` exists to prevent.

    .. code-block:: python

        ds.sel(lat=slice(20, 30))   # lat is [0, 1, 2] -- selects nothing, legally

    On ``lat``, whose coordinate is integers, the bounds are indistinguishable from
    positional ones, so sizing this by arithmetic once produced **0** against a length-3
    dim. Zero is the one wrong answer a size-driven rule could act on — drop the whole
    computation as provably empty — which is why the contract is "unknown", never "empty".
    (``time`` cannot show this: pandas rejects integer bounds on a ``DatetimeIndex``
    outright, so the ambiguity only exists where the labels really are numbers.)
    """
    tracked, _ = _fold(ds, [("sel", (), {"lat": slice(20, 30)})])
    assert tracked["lat"] is None
    assert tracked["lat"] != 0
    assert "lat" in tracked  # the dim exists; only its extent is unknown


# --- what sizes catch ----------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="the Scan arm is a blanket pass, which is exact for cumsum/cumprod and wrong "
    "for diff -- scheduled with W6 (issue #100), which gives Scan its dims. Kept xfail "
    "rather than deleted because it is the worked example of what tracking extents buys: "
    "flip it to passing when #100 lands.",
)
def test_diff_shrinks_its_dim(ds):
    """``ds.diff('time')`` drops one element along ``time``; the schema says it doesn't.

    .. code-block:: python

        ds.diff("time").sizes    # {'time': 3, 'lat': 3}   -- one shorter
        # tracked:               # {'time': 4, 'lat': 3}   -- wrong

    **This is the class of bug extents exist to catch.** ``diff`` and ``cumsum`` are the
    same node type (``Scan``), and a names-only schema cannot tell them apart: neither
    removes a dim, so ``set(tracked) == set(real)`` passes for both no matter how wrong the
    arm is. Only comparing extents distinguishes them.

    Note what this does *not* catch, since the two are easy to conflate: ``diff`` also fails
    to commute with a prefix slice where ``cumsum`` commutes fine. That is a fact about
    *order*, caught by ``optimised == eager``, not by anything here.
    """
    tracked, real = _fold(ds, [("diff", ("time",), {})])
    assert tracked == real


def test_cumsum_really_is_shape_transparent(ds):
    """The other half of the pair, so the xfail above reads as specific rather than vague.

    .. code-block:: python

        ds.cumsum("time").sizes   # {'time': 4, 'lat': 3}  -- unchanged, and tracked right

    The ``Scan`` arm's blanket ``pass`` is *correct* here. The gap is `diff`, not scans.
    """
    tracked, real = _fold(ds, [("cumsum", ("time",), {})])
    assert tracked == real


# --- the field's own contract --------------------------------------------------------


def test_sizes_are_pruned_to_the_dims_that_exist():
    """An extent for a dim no variable spans is dropped on construction.

    ``dim_names`` derives from ``variables``, so a stray extent is already unreachable
    through it. Pruning matters for a different reason: ``SchemaState`` is a *value*, and a
    stale key would make two semantically identical snapshots compare unequal.
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

    ``test_rewrites_survive_unknown_dim_sizes`` compares two *folded* schemas for equality,
    so any path that left a stale extent behind would break an assertion about rewrites
    while having nothing to do with rewrites.
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
    """The invariant that makes extents unreadable by rewrites, stated at example scale.

    The generated version (``test_rewrites_survive_unknown_dim_sizes``) asserts the strong
    form — ``optimize`` gives byte-identical output against blanked extents. This is the
    same claim one step earlier: blanking cannot even perturb the *fold*, because no arm
    branches on an extent's value.
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
