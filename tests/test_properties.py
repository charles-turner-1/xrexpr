"""Property-based tests for rewrite correctness and schema tracking.

The hand-written suites pin the optimiser against examples somebody thought of. These
generate the awkward combinations nobody wrote down: small datasets, short chains of
selects and reductions, and indexers that interact badly.

The generators are deliberately narrow (see ``_calls``), because a chain that is
*invalid* is not interesting here — ``InvalidExpressionError`` is a real failure signal
in this module, never expected noise. Two narrowings are worth naming:

- **No dim is indexed by two selects anywhere in the chain** (the ``selected_dims``
  bookkeeping in ``_calls``). ``merge_adjacent_selects`` *composes* same-dim indexers
  today — and safely splits the run where composition isn't statically provable — so
  the original reason for this restriction is gone; it survives as a candidate widening
  that nobody has done. The restriction is chain-wide rather than run-local, which is
  worth stating because it is not obvious: the two selects need never be adjacent in
  the chain the user wrote. ``pushdown_selects`` hops a select left past a reduce, so
  ``isel(time=[1]).all(dim=['lat']).isel(time=0)`` becomes a run before it folds.
- **No ``chunk``/rechunk ops.** ``Rechunk`` and its pushdown rule exist and are pinned
  by the hand-written suites; generating rechunk chains is scheduled with the chunk
  taxonomy (``docs/roadmap/07-small-wins.md`` §7, after W4).

``reduce`` is also excluded: it takes a *function* first, which ``_reduce_dims`` misreads
as a dim spec, so a generated ``.reduce(...)`` would fail for reasons unrelated to the
rewrites under test.

**Builder chains** (``groupby``/``resample``/``rolling``/``coarsen``/``weighted``) are
generated too, by :func:`builder_plans`. They feed the *contract* properties below
(optimised equals eager, lowering idempotent, the emit round-trip, tracked names) but not
the size-exactness one, which builder nodes are entitled to answer "unknown" to by design.
A builder chain is a **pair** of calls, so :func:`_builder_pair` draws both at once, and
each kind constrains its closer differently — see that function for the per-kind facts,
every one of which was verified against xarray 2026.7.0 rather than assumed.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from frozendict import frozendict
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st
from xarray.testing import assert_equal

import xrexpr  # noqa: F401 -- registers the ``.plan`` accessor

# aliased: this module's own ``Call`` is a generated *recorded* call (name + args/kwargs),
# a different thing from the emitted call header ``lower.Call`` denotes.
from xrexpr.ir import ContextOpen, Opaque
from xrexpr.lower import Call as Lowered
from xrexpr.lower import emit, to_lower_ir
from xrexpr.operations import CONTEXT_METHODS, OP_TABLE
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, apply_schema, to_opnode

# Reductions worth generating: every tabulated reduce except ``reduce`` itself.
REDUCE_NAMES = tuple(
    sorted(n for n, s in OP_TABLE.items() if s.kind == "reduce" and n != "reduce")
)

#: Reductions with no identity element, which numpy refuses to apply to an empty axis
#: ("zero-size array to reduction operation fmax which has no identity"). An empty
#: selection is a case worth generating, so these are skipped when one is in play — the
#: chain would raise in eager xarray too, making it uninteresting here.
NO_IDENTITY_REDUCES = frozenset({"max", "min"})

DIM_NAMES = ("time", "lat", "lon")

# xarray op timing is jittery on small arrays, and generation applies ops eagerly, so a
# per-example deadline just produces flakes. Function-scoped fixtures are not used here
# (the dataset is generated), so that health check is irrelevant rather than suppressed.
SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

# Empty selections are generated on purpose, and reducing an empty (or single-element)
# axis makes numpy warn about degenerate statistics. That is expected here rather than a
# signal, and left unfiltered it would train the reader to skim past warnings. Scoped to
# the exact messages so an unrelated RuntimeWarning still surfaces.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning"),
]


class Call(dict):
    """One recorded call: a method name plus its positional and keyword arguments.

    A ``dict`` subclass (over the *kwargs*) so Hypothesis shrinks and prints it readably —
    a failing example reports ``isel(time=0)`` rather than a nest of tuples.

    Positional args are carried alongside because some calls only mean what they should
    when spelled positionally: ``groupby("lat")``'s grouper is read off ``args[0]``, so a
    generated ``groupby(group="lat")`` would refuse to fuse and the builder widening would
    quietly test nothing. (That refusal is safe but avoidable — ``lower._grouper_dims``
    could read the ``group=`` keyword too, which xarray accepts. Noted, not fixed here.)
    """

    def __init__(self, name: str, *args: object, **kwargs: object) -> None:
        super().__init__(kwargs)
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        parts = [repr(a) for a in self.args]
        parts += [f"{k}={v!r}" for k, v in self.items()]
        return f"{self.name}({', '.join(parts)})"


def _apply(obj, calls):
    """Replay ``calls`` against ``obj`` — a real Dataset, a builder, or a ``.plan`` proxy."""
    for call in calls:
        obj = getattr(obj, call.name)(*call.args, **call)
    return obj


def _dims(ds):
    """What ``to_lower_ir`` needs of the base dataset: its dim names, and nothing else.

    Spelled out rather than inlined because it is the same argument at every call site, and
    because getting it from the *base* dataset is the point — see ``to_lower_ir``'s
    docstring on why base dims rather than dims-at-the-opener.
    """
    return frozenset(ds.sizes)


def _build_plan(ds, calls):
    """Normalise ``calls`` the way the recorder would, and fold the schema over the result.

    Returns the *fluent* plan — one node per call, as recorded — and the **final** schema,
    so a caller can compare tracked metadata against what evaluation actually produced.

    The fold runs over the **lowered** plan, which is where it belongs and not merely a
    convenience: ``apply_schema`` takes a ``LoweredOp``, and a ``ContextOpen`` is not one.
    Folding the fluent plan directly would hit ``assert_never`` the moment a generated
    chain contained a builder pair.
    """
    plan = [to_opnode(call.name, call.args, dict(call)) for call in calls]
    schema = SchemaState.from_dataset(ds)
    for node in to_lower_ir(plan, _dims(ds)):
        schema = apply_schema(schema, node)
    return plan, schema


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #


@st.composite
def datasets(draw, dated=False):
    """A tiny dataset with 2-3 dims, monotonic integer coords and readable values.

    ``dated`` gives ``time`` a **datetime** index instead, which two builders need and
    nothing else does: ``resample`` requires one outright, and ``groupby("time.month")``
    has no component to read off an integer coord. It is off by default so the plain
    properties keep the coords they were written against — a ``sel`` label there is an
    ``int``, and shrinking reports one.

    The resolution of the index here is **not pinned**, deliberately: it is whatever the
    installed pandas produces (``us`` on pandas 3, ``ns`` on pandas 2), so both are
    exercised across the supported matrix rather than only the newest. That makes
    :func:`_label` load-bearing — see its docstring for the trap, which was live and
    invisible until CI ran the older environment.
    """
    ndim = draw(st.integers(min_value=2, max_value=3))
    dims = DIM_NAMES[:ndim]
    sizes = [draw(st.integers(min_value=1, max_value=5)) for _ in dims]
    values = np.arange(int(np.prod(sizes)), dtype=float).reshape(sizes)
    coords = {d: np.arange(s) for d, s in zip(dims, sizes)}
    # A **second variable missing the first dim**, mirroring the hand-written fixtures'
    # ``elevation(lat, lon)``. Without it the projection rules cannot be exercised at all:
    # their whole question is whether the projected *subset* still carries the dims the
    # crossed op names, which is vacuous when every variable carries every dim.
    elevation = np.arange(int(np.prod(sizes[1:])), dtype=float).reshape(sizes[1:])
    # A **non-dim coordinate** on the last dim — the docs' ``region(lat)``, and everyday
    # xarray. It is what the builder generators were blind to until issue #90: a grouper's
    # *name* does not say which dim it consumes, because ``groupby("region")`` groups along
    # the dim ``region`` is defined on and mints ``region`` in its place. Two labels
    # whatever the size, so the grouping really aggregates rather than being the identity.
    # Numeric region *ids* rather than the docs' ``"a"``/``"b"`` strings, because the coord
    # has to survive everything else the generators draw: ``coarsen`` reduces coordinate
    # values with ``coord_func="mean"``, which raises on a ``<U1`` dtype. The dtype is
    # incidental to what this coordinate is here to exercise, which is its *dims*.
    coords["region"] = (dims[-1], np.array([0.0, 1.0] * sizes[-1])[: sizes[-1]])
    if dated:
        # daily, so a 2-5 point index spans one month: ``time.month`` yields one group and
        # ``resample(time="2D")`` halves it. Both are legal, which is all that is needed.
        coords["time"] = pd.date_range("2000-01-01", periods=sizes[0], freq="D")
    return xr.Dataset(
        {"temperature": (dims, values), "elevation": (dims[1:], elevation)},
        coords=coords,
    )


def _label(value):
    """A ``sel``-able Python label for one coordinate value.

    ``.item()`` is the obvious spelling and is **wrong for datetimes**, in a way that only
    shows up on some dependency sets: ``np.datetime64`` unpacks to a ``datetime.datetime``
    at microsecond resolution but to a bare ``int`` of nanoseconds-since-epoch at
    nanosecond resolution, because ``datetime`` cannot represent the latter. Which one a
    ``pd.date_range`` produces depends on the pandas version — pandas 3 gives ``us``,
    pandas 2 gives ``ns`` — so ``sel(time=946684800000000000)`` reached xarray on the older
    pinned environment and raised ``KeyError``, while every newer one passed.

    ``pd.Timestamp`` is exact at either resolution and is a label ``sel`` accepts, so it
    sidesteps the version split rather than pinning around it. Numbers keep ``.item()``,
    which is what makes a shrunk example report ``lat=0`` rather than ``np.int64(0)``.
    """
    return (
        pd.Timestamp(value)
        if np.issubdtype(value.dtype, np.datetime64)
        else value.item()
    )


@st.composite
def indexers(draw, obj, dim, name):
    """One indexer for ``dim`` of ``obj``, valid for ``isel`` or ``sel`` respectively.

    The two cannot share a generator: ``isel`` addresses *positions* ``0..size-1``, while
    ``sel`` addresses *coordinate labels*, which stop being ``0..size-1`` as soon as an
    earlier op subsets the dim. Drawing sel indexers from the live coordinate values is
    what keeps a generated chain replayable.

    Negative bounds and reversed slices are excluded on purpose: they count from the end,
    so composing them needs the dim length that the optimiser deliberately does not
    carry. They are the uncomposable cases, and belong with the rule that handles them
    rather than here.
    """
    size = obj.sizes[dim]
    if name == "isel":
        values = list(range(size))
        bounds = st.integers(min_value=0, max_value=size)
    else:
        values = [_label(v) for v in obj[dim].values]
        # a label slice is inclusive of both ends, and needs labels that exist
        bounds = st.sampled_from(values) if values else st.none()

    strategies = [
        st.builds(slice, bounds, bounds, st.one_of(st.none(), st.integers(1, 2))),
        st.just([]),  # empty selection
    ]
    if values:
        strategies += [
            st.sampled_from(values),  # scalar: drops the dim
            st.lists(
                st.sampled_from(values), min_size=1, max_size=min(3, size), unique=True
            ).map(sorted),
        ]
    return draw(st.one_of(strategies))


#: Which tabulated reductions each builder actually *has*, verified against xarray
#: 2026.7.0. Drawing from :data:`REDUCE_NAMES` for all of them would generate
#: ``ds.weighted(w).max()`` — an ``AttributeError`` about a method that does not exist,
#: which is noise rather than a finding about the rewrites under test.
BUILDER_CLOSERS = {
    "groupby": REDUCE_NAMES,
    "resample": REDUCE_NAMES,
    # DatasetRolling has no ``all``/``any``; the rest it has
    "rolling": tuple(n for n in REDUCE_NAMES if n not in {"all", "any"}),
    "coarsen": REDUCE_NAMES,
    # DatasetWeighted has only these four (plus the untabulated ``sum_of_weights``,
    # ``sum_of_squares`` and ``quantile``, which record ``Opaque`` and so never fuse)
    "weighted": ("mean", "std", "sum", "var"),
}


#: Reductions the **windowed** builders cannot apply to **boolean** data. Mapped
#: exhaustively against xarray 2026.7.0 rather than guessed, and the shape is tidier than
#: it sounds: only ``bool`` is affected (``int`` — what ``count`` yields — is fine
#: everywhere), and only ``rolling``/``coarsen``, whose kernels stride and NaN-pad; a
#: boolean array cannot hold the pad value, and the moving-window kernels are float-only.
#: ``groupby``/``resample``/``weighted`` take all of theirs on bool.
WINDOWED_FLOAT_ONLY_REDUCES = frozenset({"median", "std", "var"})

#: Reductions that need a float array once it is **empty**: ``median`` of a zero-size
#: ``bool``/``int`` array raises ``cannot reshape array of size 0``, where a float one
#: answers NaN. Note the empty dim need not be one of the *reduced* ones — it is the
#: array being empty that does it, so this is a wider condition than
#: :data:`NO_IDENTITY_REDUCES`'.
NON_FLOAT_EMPTY_UNSAFE_REDUCES = frozenset({"median"})


#: Reductions that break when a *variable* ends up reducing over **no** dims. A Dataset
#: reduce passes ``dim=[]`` down to any variable that lacks every dim it named, and
#: ``std``/``var`` then build a 0-d result while still claiming the variable's dims:
#: ``ValueError: dimensions ('lat',) must have the same length as ... ndim=0``. Reachable
#: only with a variable that lacks a dim — i.e. only since the generator grew one; and
#: ``rolling``/``coarsen`` are unaffected (they only touch variables having the window dim).
#:
#: **This is a numbagg bug, not an xarray one** (``07-small-wins.md`` §8): numbagg reads
#: ``axis=()`` as ``axis=None`` and hands back a scalar, and xarray raises on being given
#: one where an array was promised. ``mean`` and the rest are spared not because xarray
#: skips such a variable — it does not, they all take the same empty-axis reduce — but
#: because ``duck_array_ops`` short-circuits ``axis == ()`` for every reduce that *is* the
#: identity there (``invariant_0d``), returning before dispatch. ``std``/``var`` are
#: excluded from that shortcut **correctly**, their empty-axis answer being zeros rather
#: than the input, which is exactly why they are the only two that reach the bug.
#:
#: So expect this filter to go **inert**: it is vacuous without numbagg and would be
#: retired outright by a numbagg fix. Keep it regardless — CI has numbagg (``rolling_exp``
#: refuses to run without it) and the suite must be green there. Nothing else is coupled to
#: the bug: the hand-written test named below deliberately sources its eager failure from a
#: *string* variable instead, so it does not track numbagg's release schedule.
#:
#: Filtered because such a chain has **no usable eager reference**, not to hide anything:
#: the properties below assert ``optimised == eager``, and eager here *raises*. What the
#: optimiser does with one is deliberate and good — ``pushdown_projections`` skips the
#: failing computation precisely because the plan discards it — but that is the sharpened
#: contract in ``docs/roadmap/07-small-wins.md`` §8, which an equality assertion cannot
#: express. It is pinned by a hand-written test instead.
EMPTY_AXIS_UNSAFE_REDUCES = frozenset({"std", "var"})


def _drawable_reduces(obj, names, dims=None, windowed=False, reduced=None):
    """``names`` minus the reductions xarray cannot actually apply to ``obj`` as it stands.

    Every exclusion is an xarray/numpy limitation rather than a finding about the rewrites
    — a chain that raises for one of these reasons would be noise — and every one is
    verified against xarray 2026.7.0. They are gathered here rather than inlined at the two
    draw sites because they overlap: non-float data (``all``/``any`` yield bool, ``count``
    yields int) and empty dims (an empty selection is generated on purpose) both arise from
    the *plain* chain and then flow into whatever a builder does next.

    ``dims`` names the axes being reduced, for the identity check that only cares about
    those; a builder closer passes ``None``, meaning "any empty dim disqualifies", since it
    chooses its own axes. ``windowed`` marks a ``rolling``/``coarsen`` closer, which has
    :data:`WINDOWED_FLOAT_ONLY_REDUCES` to avoid on boolean data.
    """
    reducing_empty = (
        any(obj.sizes[d] == 0 for d in dims)
        if dims is not None
        else any(size == 0 for size in obj.sizes.values())
    )
    if reducing_empty:
        names = tuple(n for n in names if n not in NO_IDENTITY_REDUCES)
    if any(size == 0 for size in obj.sizes.values()) and any(
        v.dtype.kind != "f" for v in obj.data_vars.values()
    ):
        names = tuple(n for n in names if n not in NON_FLOAT_EMPTY_UNSAFE_REDUCES)
    if windowed and any(v.dtype == bool for v in obj.data_vars.values()):
        names = tuple(n for n in names if n not in WINDOWED_FLOAT_ONLY_REDUCES)
    if reduced is not None and any(
        not (reduced & set(var.dims)) for var in obj.data_vars.values()
    ):
        names = tuple(n for n in names if n not in EMPTY_AXIS_UNSAFE_REDUCES)
    return names


def _has_datetime_index(obj, dim="time"):
    """Whether ``dim`` carries a datetime index — what ``resample`` and ``.month`` need."""
    return dim in obj.indexes and isinstance(obj.indexes[dim], pd.DatetimeIndex)


@st.composite
def _builder_pair(draw, obj, kind=None):
    """An ``(opener, closer)`` call pair legal against ``obj``, or ``None`` if none is.

    A builder chain is one semantic operation spelled as **two** calls, so it has to be
    drawn as a unit — a loop drawing one call at a time could never produce a fusable
    adjacency. Each kind constrains the pair differently, and every constraint below is a
    verified fact about xarray 2026.7.0, not a precaution:

    - **the dim must be non-empty.** ``groupby`` raises ``lat must not be empty`` and
      ``rolling`` rejects every window on a zero-length dim. An empty selection is worth
      generating (and is, elsewhere), just not underneath a builder.
    - **``rolling``'s window must be ``1..size``** — beyond that it raises ``window not in
      valid range``, which is a fact about the *call*, not about the rewrites.
    - **``coarsen``'s default ``boundary="exact"`` requires divisibility**, so it is only
      offered when the window divides; ``trim`` and ``pad`` always are.
    - **``rolling``/``coarsen`` closers must be bare.** ``DatasetRolling.mean`` is
      ``(keep_attrs=None, **kwargs)`` — no dim argument at all — so a ``dim=`` would be
      swallowed as ``keep_attrs``. That is the misparse lowering refuses to fuse, and it is
      pinned by example in ``test_lower.py``; generating it here would only re-test the
      fallback.
    - **a grouped closer may name dims or not**, and the two mean different things: bare or
      naming the group dim is an aggregation (which fuses), naming *other* dims is a
      per-group map (which does not). Both are legal — for ``resample`` as much as for
      ``groupby`` — so both are drawn, for both kinds.
    - **``weighted`` needs a real ``DataArray``**, aligned so nothing is reindexed away.

    ``max``/``min`` are dropped whenever any dim is empty, for the reason
    :data:`NO_IDENTITY_REDUCES` gives.
    """
    dims = sorted(d for d, size in obj.sizes.items() if size > 0)
    if not dims:
        return None

    kinds = ["groupby", "rolling", "coarsen", "weighted"]
    if "time" in dims and _has_datetime_index(obj):
        # ``dims`` is the non-empty ones: resampling an empty ``time`` raises
        # ``__resample_dim__ must not be empty``, the same rule ``groupby`` states.
        kinds.append("resample")
    if kind is None:
        kind = draw(st.sampled_from(kinds))
    elif kind not in kinds:
        return None

    dim = "time" if kind == "resample" else draw(st.sampled_from(dims))
    size = obj.sizes[dim]

    coord_grouper = False
    if kind == "groupby":
        groupers = [dim]
        if dim == "time" and _has_datetime_index(obj):
            groupers += ["time.month", "time.day"]
        # The non-dim coordinate groupers defined on *this* dim, so the pair's group dim is
        # still ``dim`` and the ``reduced`` bookkeeping below stays correct. This is the
        # widening issue #90 asked for: pre-fix these fused with ``group_dim="region"`` and
        # ``test_tracked_schema_agrees_with_evaluation`` catches it; post-fix they refuse and
        # the equality property covers them through the opaque fallback.
        coord_groupers = _coord_groupers(obj, dim)
        groupers += coord_groupers
        # positional: the grouper is read off ``args[0]`` (see ``Call``)
        grouper = draw(st.sampled_from(groupers))
        coord_grouper = grouper in coord_groupers
        opener = Call("groupby", grouper)
    elif kind == "resample":
        opener = Call("resample", time=draw(st.sampled_from(["1D", "2D", "3D"])))
    elif kind == "rolling":
        opener = Call("rolling", **{dim: draw(st.integers(1, size))})
    elif kind == "coarsen":
        window = draw(st.integers(1, size))
        boundaries = ["trim", "pad"] + (["exact"] if size % window == 0 else [])
        opener = Call(
            "coarsen", boundary=draw(st.sampled_from(boundaries)), **{dim: window}
        )
    else:
        opener = Call("weighted", _weights(obj, dim))

    # The closer's dim spec is drawn *first*, because which reductions are legal depends on
    # it: a grouped reduce removes the group dim plus whatever the closer named, and a
    # variable left with none of those trips :data:`EMPTY_AXIS_UNSAFE_REDUCES`.
    # A coordinate grouper takes a **bare** closer only. The map/aggregation distinction
    # ``_closer_dims`` draws over is stated in terms of the group dim, and under a coordinate
    # grouper the closer cannot name it — ``groupby("region").mean("region")`` is not the
    # aggregation case, and naming the underlying dim instead is a shape whose eager
    # behaviour would have to be established before it could be a reference. Bare is the
    # case #90 is about, and unambiguously legal.
    spec = {} if coord_grouper else draw(_closer_dims(obj, kind, dim))
    reduced = None
    if kind in {"groupby", "resample"}:
        # What each *variable* ends up reducing over, which is not the same as what the
        # node removes: the group dim is consumed by the grouping mechanism, so a closer
        # that named dims reduces exactly those within each group (the map case), while a
        # bare one reduces along the group dim. ``groupby("lat").std(dim=["time"])`` is
        # the case that tells them apart -- a ``lat``-less variable reduces over nothing.
        reduced = set(spec["dim"]) if "dim" in spec else {dim}
    names = _drawable_reduces(
        obj,
        BUILDER_CLOSERS[kind],
        windowed=kind in {"rolling", "coarsen"},
        reduced=reduced,
    )
    closer = Call(draw(st.sampled_from(names)), **spec)
    return [opener, closer]


def _coord_groupers(obj, dim):
    """Non-dim coordinate names defined on ``dim`` alone.

    Groupers whose name is not a dim, but which group along ``dim`` all the same — the
    class ``lower._grouper_dims`` cannot read off the call, and so refuses to fuse.
    Restricted to one-dim coordinates: a multi-dim one stacks and consumes *every* dim it
    spans, which no single ``group_dim`` describes.
    """
    return [
        name
        for name, coord in obj.coords.items()
        if name not in obj.sizes and coord.dims == (dim,)
    ]


@st.composite
def _closer_dims(draw, obj, kind, dim):
    """The closer's kwargs: a ``dim=`` spec where the builder takes one, else nothing.

    ``dim`` is the builder's own dim, needed only by the ``weighted`` arm below — the
    grouped kinds offer every dim regardless of which one they group over.
    """
    if kind in {"rolling", "coarsen"}:
        return {}  # takes no dim argument at all -- see ``_builder_pair``
    if kind == "weighted":
        # A weighted reduce is applied per variable and *raises* when one lacks a named
        # dim, where a plain reduce skips it -- but ``dot`` broadcasts the weights in
        # first, so what each variable ``v`` needs is ``named <= v.dims | w.dims``. The
        # weights here are over ``dim`` alone (see ``_weights``), which makes the safe
        # candidate set the dims *every* variable carries, plus ``dim`` itself.
        #
        # Note what this excludes, and why that is not a gap: the raising case is exactly
        # where ``pushdown_projections`` *skips* an error, so eager and optimised
        # legitimately differ and ``optimised == eager`` could not express it. Same reason
        # as :data:`EMPTY_AXIS_UNSAFE_REDUCES` -- the chains excluded are the ones whose
        # eager *reference* is broken. ``07-small-wins.md`` §8 states the sharpened
        # contract; the behaviour is pinned by hand in ``test_optimize``/``test_accessor``.
        # The hop itself is still well covered here: it fires on ~21% of generated
        # weighted-plus-projection plans, all of which assert equality with eager.
        shared = set(obj.sizes)
        for var in obj.data_vars.values():
            shared &= set(var.dims)
        candidates = sorted(shared | {dim})
    else:
        # Both grouped kinds offer every dim, and the choice is semantic: bare or naming
        # the group dim is an aggregation (which fuses), naming only other dims is a
        # per-group map (which does not). That holds for ``resample`` exactly as for
        # ``groupby`` -- verified against xarray 2026.7.0, where
        # ``resample(time="2D").mean(dim=["lat"])`` comes back on the *original* ``time``,
        # not the resampled one. Drawing only the group dim here, as this once did for
        # ``resample``, left the map case reachable through ``groupby`` alone.
        candidates = sorted(obj.sizes)
    spec = draw(
        st.one_of(
            st.none(),  # bare
            st.lists(st.sampled_from(candidates), min_size=1, unique=True).map(sorted),
        )
        if candidates
        else st.none()
    )
    return {} if spec is None else {"dim": spec}


def _weights(obj, dim):
    """Positive weights over ``dim``, aligned with ``obj`` so nothing is reindexed away.

    Misaligned weights would *shrink* the dim (xarray inner-joins), which is modelled but
    deliberately not generated: it is pinned by example in ``test_schema.py``, and here it
    would only add a second reason for a size to be unknown.
    """
    values = np.arange(1.0, obj.sizes[dim] + 1.0)
    coords = {dim: obj[dim]} if dim in obj.coords else None
    return xr.DataArray(values, dims=dim, coords=coords)


@st.composite
def _calls(draw, ds, max_ops=4, builders=False):
    """A chain of ops that is legal against ``ds`` by construction.

    Legality is guaranteed by *evaluating as we generate*: each drawn call is applied to
    a running dataset, so the next call sees the real post-op dims and sizes rather than
    a reimplementation of them. The datasets are tiny, so this is cheap and exact.
    """
    calls = []
    current = ds
    selected_dims: set[str] = set()  # every dim any select has already indexed

    # "builder" twice, so the ``assume`` in ``builder_plans`` discards fewer examples —
    # a chain with no builder pair in it is just a plain chain, already covered.
    kinds = ["isel", "sel", "reduce", "project"] + (["builder"] * 2 if builders else [])

    for _ in range(draw(st.integers(min_value=0, max_value=max_ops))):
        if not current.sizes:
            break  # everything has been reduced away; nothing legal is left

        kind = draw(st.sampled_from(kinds))

        if kind == "builder":
            pair = draw(_builder_pair(current))
            if pair is None:
                break
            current = _apply(current, pair)
            calls.extend(pair)
            continue

        if kind == "project":
            # A **list** key only, so the result stays a Dataset. A bare name yields a
            # DataArray, whose downstream algebra is a different thing entirely (no
            # ``data_vars``, and ``__getitem__`` becomes indexing rather than projection);
            # that case is covered by the hand-written suites, and ``.plan`` is not even
            # registered for DataArray yet (``07-small-wins.md`` §5).
            names = sorted(map(str, current.data_vars))
            if not names:
                break
            call = Call(
                "__getitem__",
                draw(
                    st.lists(st.sampled_from(names), min_size=1, unique=True).map(
                        sorted
                    )
                ),
            )
            current = _apply(current, [call])
            calls.append(call)
            continue

        if kind == "reduce":
            dims = draw(
                st.lists(
                    st.sampled_from(sorted(current.sizes)),
                    min_size=1,
                    max_size=len(current.sizes),
                    unique=True,
                ).map(sorted)
            )
            names = _drawable_reduces(
                current, REDUCE_NAMES, dims=dims, reduced=set(dims)
            )
            call = Call(draw(st.sampled_from(names)), dim=dims)
        else:
            # No dim is indexed twice anywhere in the chain — not merely twice in a row.
            # Adjacency is not a property of the chain the user wrote: ``pushdown_selects``
            # hops a select left past a reduce, so two selects with a reduction between
            # them become a run that ``merge_adjacent_selects`` then folds. Restricting
            # only within an existing run leaves that path wide open (see module docstring).
            available = sorted(set(current.sizes) - selected_dims)
            if not available:
                break
            dim = draw(st.sampled_from(available))
            call = Call(kind, **{dim: draw(indexers(current, dim, kind))})
            selected_dims.add(dim)

        current = _apply(current, [call])
        calls.append(call)

    return calls


@st.composite
def plans(draw):
    """A dataset paired with a legal chain of calls against it."""
    ds = draw(datasets())
    return ds, draw(_calls(ds))


@st.composite
def builder_plans(draw):
    """A dataset paired with a legal chain containing at least one **builder pair**.

    The chain is generated by the same loop as :func:`plans`, so ops land on both sides of
    the pair — which is the interesting shape, since a fused node is exactly what no rule
    may reorder across.
    """
    ds = draw(datasets(dated=draw(st.booleans())))
    calls = draw(_calls(ds, builders=True))
    assume(any(call.name in CONTEXT_METHODS for call in calls))
    return ds, calls


def any_plans():
    """Plain and builder chains alike — for the properties that hold of every plan."""
    return st.one_of(plans(), builder_plans())


@st.composite
def select_runs(draw):
    """A dataset paired with a run of >=2 adjacent same-name selects on distinct dims."""
    ds = draw(datasets())
    name = draw(st.sampled_from(["isel", "sel"]))
    dims = draw(
        st.lists(
            st.sampled_from(sorted(ds.sizes)),
            min_size=2,
            max_size=len(ds.sizes),
            unique=True,
        )
    )
    calls = []
    current = ds
    for dim in dims:
        call = Call(name, **{dim: draw(indexers(current, dim, name))})
        current = _apply(current, [call])
        calls.append(call)
    return ds, calls


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #


@SETTINGS
@given(any_plans())
def test_optimised_plan_matches_eager_evaluation(case):
    """The headline property: optimising must not change the answer.

    This exercises the whole optimiser without encoding any individual rule, which is
    what makes it worth generating rather than enumerating.

    The fused-node mix is reported as Hypothesis events rather than asserted, since which
    nodes a random chain happens to contain is not a property — but a run whose statistics
    show no fused nodes at all would mean the builder widening had stopped biting.
    """
    ds, calls = case
    for node in to_lower_ir(_build_plan(ds, calls)[0], _dims(ds)):
        if type(node).__name__.endswith("Reduce") and type(node).__name__ != "Reduce":
            event(f"fused: {type(node).__name__}")
    assert_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@SETTINGS
@given(any_plans())
def test_optimize_is_idempotent(case):
    """A second pass changes nothing — the fixpoint really is a fixed point.

    Reaching this assertion at all is half the property: ``optimize`` loops until no rule
    fires, so a rule pair that undid each other's work would hang here rather than fail.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    schema = SchemaState.from_dataset(ds)
    once = optimize(to_lower_ir(plan, _dims(ds)), schema)
    assert optimize(once, schema) == once


@SETTINGS
@given(any_plans())
def test_lowering_is_idempotent(case):
    """Lowering an already-lowered plan returns it unchanged.

    The stage's other contract (the analogue of ``test_optimize_is_idempotent``, and the
    reason lowering can't be a rewrite rule: it runs *once*, so re-running it must be a
    no-op rather than keep shrinking a measure).
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    once = to_lower_ir(plan, _dims(ds))
    assert to_lower_ir(once, _dims(ds)) == once


@SETTINGS
@given(builder_plans())
def test_no_context_open_survives_lowering(case):
    """The invariant ``LoweredOp`` enforces at type-check time, asserted at runtime.

    Cheap and worth having generated: every builder pair either fuses or is demoted
    *together with its closer*, so an opener can only survive by way of a shape nobody
    thought of — which is what generation is for.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    assert not any(
        isinstance(node, ContextOpen) for node in to_lower_ir(plan, _dims(ds))
    )


@SETTINGS
@given(any_plans())
def test_emit_after_lowering_reproduces_the_recorded_calls(case):
    """A lowered plan emits exactly the calls that were recorded.

    The round-trip half of lowering's contract, pinned node-for-node rather than only
    through the result: nothing in the pipeline may quietly re-spell a call it did not
    need to touch. Compared as ``Call`` headers, not as nodes — emit's output is codegen,
    and one node need not stay one call.

    That last part is why this holds for builder chains too, which is not obvious: a fused
    pair is *one* node emitting *two* calls, and because both headers are kept verbatim the
    two are exactly the two that were recorded. The unfusable fallback lands in the same
    place from the other direction — two ``Opaque`` nodes carrying the same headers.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    assert emit(to_lower_ir(plan, _dims(ds))) == [
        Lowered(name=node.name, args=node.args, kwargs=node.kwargs) for node in plan
    ]


@SETTINGS
@given(select_runs())
def test_adjacent_selects_collapse_without_changing_meaning(case):
    """A run of same-name selects folds to one node that means the same thing.

    Asserting the fold *fires* keeps this from passing vacuously: an over-cautious merge
    rule that never merged would satisfy the equality half on its own.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    optimised = optimize(to_lower_ir(plan, _dims(ds)), SchemaState.from_dataset(ds))

    assert len(optimised) == 1, "a run of selects on distinct dims should fold to one"
    assert_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@SETTINGS
@given(any_plans())
def test_tracked_schema_agrees_with_evaluation(case):
    """Tracked dim *names* and coords must describe what evaluation actually produces.

    Sizes are asserted separately (below), because one case gets them wrong today and the
    fused nodes decline to answer at all. What holds universally — builder chains
    included, which is the whole reason their arms mark sizes unknown rather than
    dropping dims — is:

    - **dims**: the same dim names survive. This is what the rewrites actually reason
      about — ``consumes`` and the pushdown conflict check are name-based.
    - **coords and variables**: exactly the result's, not merely a subset. A scalar select
      drops the dim while xarray keeps a scalar coordinate behind, and an aggregating op
      removes that coordinate outright — both are modelled now that a coordinate is a
      variable with dims, so there is nothing left to be conservative about.

    Dim *order* is deliberately not asserted: ``Dataset.sizes`` does not promise the
    insertion order the schema threads, so comparing as sets is the honest check.

    **Confined to plans with no unmodelled op**, which until builder chains were generated
    was an assumption the generator happened to guarantee rather than one anybody stated.
    ``apply_schema`` models an :class:`~xrexpr.ir.Opaque` as dim- and variable-preserving,
    which is not true of ``rename``/``drop_vars`` — or, now, of an unfusable builder pair:
    ``groupby("time").all(dim=["lat"])`` is a per-group *map*, refuses to fuse, and really
    does remove ``lat`` while the tracked schema says it stayed. That is the documented
    trust boundary (``optimize._trusted_prefix``), not a tracking bug, and the rules
    respect it — dim-level ones read no schema, and ``pushdown_projections`` confines
    itself to the prefix. So the ``assume`` states the boundary rather than papering over
    it, and the equality property above still covers the opaque case, where it matters.
    """
    ds, calls = case
    plan, schema = _build_plan(ds, calls)
    assume(not any(isinstance(node, Opaque) for node in to_lower_ir(plan, _dims(ds))))
    result = _apply(ds, calls)

    assert set(schema.sizes) == set(result.sizes)

    # Every tracked coordinate, not just the dim coordinates — the restriction issue #109
    # asked for is gone, because coordinates are variables now and their lifetimes are
    # modelled per-op: an aggregating op drops one over the dim it aggregates, an indexing
    # one demotes it to 0-d. Asserted **exactly** rather than as a subset, which the bare-name
    # ``coords`` set could never have supported.
    assert set(schema.coords) == set(result.coords)
    assert set(schema.variables) == set(result.variables)


@SETTINGS
@given(plans())
def test_sizes_are_tracked_exactly_without_label_slices(case):
    """Away from label slices, tracked sizes are exact — not merely conservative.

    Every indexer except a ``sel`` slice is sized from the indexer alone: positions,
    lengths of sequences, boolean counts. This pins that the imprecision below is
    confined to the label-slice case rather than lurking generally.

    Deliberately kept on the **plain** generator. The fused nodes are entitled to answer
    "unknown": a grouped reduce's minted extent is the group count, and a surviving weight
    dim's is a post-alignment length — both facts about coordinate *values*, which this
    layer does not read. Widening this one would assert the opposite of that contract.
    """
    ds, calls = case
    assume(not any(_has_label_slice(call) for call in calls))

    _, schema = _build_plan(ds, calls)
    assert dict(schema.sizes) == dict(_apply(ds, calls).sizes)


def _has_label_slice(call):
    """Whether ``call`` is a ``sel`` carrying a slice indexer."""
    return call.name == "sel" and any(isinstance(v, slice) for v in call.values())


@pytest.mark.xfail(
    strict=True,
    reason="a sel label slice is sized 'unknown', not exactly -- deliberate, see the "
    "docstring: the schema trades an unsafe under-report for an honest None, and "
    "computing the exact answer needs coordinate values, which is a different decision.",
)
def test_sel_label_slice_size_is_tracked_correctly():
    """A ``sel`` label slice is sized ``None``, so it is not tracked *exactly*.

    Found by ``test_tracked_schema_agrees_with_evaluation``, which shrank it to a one-op
    chain. ``sel`` bounds are *labels* and the slice is inclusive of both ends, but where
    the bounds are integers ``classify`` cannot tell them from positions and mints a
    ``ForwardSlice``. With labels ``[10, 20, 30, 40]``, ``sel(lat=slice(20, 30))`` really
    yields 2 elements while the positional reading gave **0** — under-reporting, the
    *unsafe* direction, since it is the error a size-driven rule would act on.

    **Kept xfail deliberately** (``docs/roadmap/03-schema-sizes.md`` §5.5). The size
    is ``None`` rather than ``0``: the unsafe answer is gone, and this test's stricter
    claim — that the size is tracked *exactly* — remains unmet, so the marker stays.
    Reaching the exact 2 means reading ``ds.indexes["lat"]`` at ``apply_schema`` time,
    which is a separate call about how far the schema layer may consult coordinate
    values; ``03-schema-sizes.md`` §2 opens that door for lowering's own use but does
    not walk through it here. Still latent either way: no rewrite consults tracked sizes.
    """
    ds = xr.Dataset({"t": ("lat", np.arange(4.0))}, coords={"lat": [10, 20, 30, 40]})
    call = Call("sel", lat=slice(20, 30))

    _, schema = _build_plan(ds, [call])
    assert dict(schema.sizes) == dict(_apply(ds, [call]).sizes)


def test_sel_label_slice_size_is_unknown_rather_than_wrong():
    """The half of the case above that *is* now settled: unknown, never under-reported.

    The companion to the xfail — it pins the improvement so that regressing to a
    positional reading fails a green test rather than merely leaving an xfail xfailing.
    """
    ds = xr.Dataset({"t": ("lat", np.arange(4.0))}, coords={"lat": [10, 20, 30, 40]})

    _, schema = _build_plan(ds, [Call("sel", lat=slice(20, 30))])
    assert schema.sizes["lat"] is None
    assert schema.sizes["lat"] != 0  # the under-report this replaced


@SETTINGS
@given(any_plans())
def test_rewrites_survive_unknown_dim_sizes(case):
    """Optimising against a schema whose sizes are all unknown still matches eager.

    The property the ``int | None`` sizes exist to license: a ``GroupedReduce`` mints dims
    whose extent comes from coordinate *values* and a ``WeightedReduce`` marks its
    surviving weight dims unknown, so the plans the rules see carry ``None`` sizes
    routinely — builder chains are generated here, so this is not a hypothetical. Every
    rule reasons about dim *names*, so blanking every size must change nothing.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    base = SchemaState.from_dataset(ds)
    blanked = SchemaState(
        variables=base.variables,  # the store is untouched; only the extents are blanked
        coord_names=base.coord_names,
        sizes=frozendict(dict.fromkeys(base.sizes)),  # every size -> None
    )
    lowered = to_lower_ir(plan, _dims(ds))
    assert optimize(lowered, blanked) == optimize(lowered, base)


@pytest.mark.parametrize("kind", sorted(BUILDER_CLOSERS))
@SETTINGS
# keyword, not positional: a positional strategy binds to the *trailing* parameter, which
# is ``kind`` -- pytest's, not Hypothesis's to fill.
@given(data=st.data())
def test_every_builder_kind_is_generated_and_replays_equal_to_eager(data, kind):
    """Anti-vacuity, per kind: each is reachable, and each pair works.

    The properties above would pass just as happily if :func:`_builder_pair` had quietly
    stopped emitting, say, ``resample`` — a random chain contains what it contains, so
    nothing there can notice. This asks for each kind by name instead.

    Fusion is asserted too, with exactly one exception: a grouped closer naming dims that
    *exclude* the group dim is a per-group **map**, not an aggregation, and deliberately
    does not fuse. Everything else the generator emits must, or the widening is exercising
    the opaque fallback while looking like it exercises the fused nodes.
    """
    ds = data.draw(datasets(dated=True))
    pair = data.draw(_builder_pair(ds, kind=kind))

    assert pair is not None and pair[0].name == kind
    lowered = to_lower_ir(_build_plan(ds, pair)[0], _dims(ds))
    event(f"{kind}: {'fused' if len(lowered) == 1 else 'opaque pair'}")
    assert len(lowered) == 1 or kind in {"groupby", "resample"}

    assert_equal(_apply(ds.plan, pair).collect(), _apply(ds, pair))
