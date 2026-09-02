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
- **No ``chunk`` op unless dask is installed.** Replaying a rechunk needs a chunk
  manager, so :data:`HAS_DASK` gates the kind out rather than letting a drawn chain
  raise. Where it is installed, both call forms are generated — see :func:`_rechunks`.

``.reduce`` **is** generated, as of #96. It used to be excluded because ``_dim_spec``
read its first positional — a *function* — as a dim spec; now that the dim spec is read
from the second, the node is honest and the chains are worth generating. It is drawn
separately from :data:`REDUCE_NAMES` rather than added to it, because its call shape
differs (the function comes first) and the builder closers are generated as
``name(**spec)``.

**Scans** (``cumsum``/``cumprod``/``diff``, :data:`SCAN_NAMES`) are generated too, as of
W6. They are order-significant but dim-keeping, so the equality-vs-eager property is what
proves the optimiser hops a *disjoint* select or projection across a scan while leaving an
intersecting one put — the leave-don't-raise leg the hand-written goldens pin by example.

**Elementwise ops** (``astype``/``fillna``/``clip``/``round``) are generated too, as of
W5, with scalar arguments only — so they record :class:`~xrexpr.ir.Elementwise` and the
equality-vs-eager property covers a select or projection crossing them in arbitrary
interleavings. The unsafe-argument path (a data- or per-variable-shaped argument demoting
to :class:`~xrexpr.ir.Opaque`) is left to the hand-written suite; the generator draws only
the safe forms, so a drawn chain never turns an elementwise op into a barrier by accident.

**Advanced (``DataArray``) indexers** are not drawn: :func:`indexers` produces only slices,
positions, labels and scalars. An *orthogonal* one would add nothing anyway — the schema
layer normalises it to the very positional indexer already generated (``schema``'s
``_orthogonal_advanced``), so it is exactly a positional select the suite covers. The
*vectorized* form, which demotes to :class:`~xrexpr.ir.Opaque`, is pinned by the
hand-written suite as the unsafe-elementwise path above is; its full dim modelling is still
open (#60).

**Scattered NaNs** in the float data (see :func:`datasets`) are what give the
equality-vs-eager property its teeth for the elementwise ops above: ``fillna`` and every
skipna reduce treat a NaN specially, so a value-level reorder bug changes an answer only
where a NaN sits, and a NaN-free array would let it pass. The comparison is done on
*materialised* values (:func:`_assert_replays_equal`) so it does not run through a dask
reduction, which raises on a degenerate chunk a strided slice can leave — see that helper.

**Builder chains** (``groupby``/``resample``/``rolling``/``coarsen``/``weighted``) are
generated too, by :func:`builder_plans`. They feed the *contract* properties below
(optimised equals eager, lowering idempotent, the emit round-trip, tracked names) but not
the size-exactness one, which builder nodes are entitled to answer "unknown" to by design.
A builder chain is a **pair** of calls, so :func:`_builder_pair` draws both at once, and
each kind constrains its closer differently — see that function for the per-kind facts,
every one of which was verified against xarray 2026.7.0 rather than assumed.

**Coordinate projections** — a projection naming a *coordinate*, which
``eliminate_projection_before_coord`` collapses when its input is a projection whose
variables the coordinate survives — get their own generator, :func:`coord_projection_plans`,
rather than a widening of ``_calls``. The pair must come last (a coordinate projection
leaves a coord-only dataset) and its chains cannot feed the schema-exactness property
(``apply_schema`` over-reports a coordinate projection), so keeping them separate is what
lets a single dedicated test pin both that the rule fires and that collapsing preserves the
answer — see :func:`test_a_coord_projection_drops_its_input_and_replays_equal`.
"""

import importlib.util

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
from xrexpr.chunks import Auto, BlockSeq, ByteSize
from xrexpr.interned import deintern, intern
from xrexpr.interner import Interner
from xrexpr.ir import ContextOpen, Opaque, Rechunk, Select
from xrexpr.lower import Call as Lowered
from xrexpr.lower import emit, to_lower_ir
from xrexpr.operations import CONTEXT_METHODS, OP_TABLE, ReduceSpec, ScanSpec
from xrexpr.optimize import (
    eliminate_projection_before_coord,
    optimize,
    push_projection_past_drop,
)
from xrexpr.schema import SchemaState, apply_schema, to_opnode

# Reductions spelled ``name(dim=...)``: every tabulated reduce except ``reduce``, whose
# first positional is a *function*. Kept out of this tuple because of that call shape, not
# because its node is untrustworthy — the builder closers below are generated as
# ``name(**spec)`` and have nowhere to put a function.
REDUCE_NAMES = tuple(
    sorted(
        n for n, s in OP_TABLE.items() if isinstance(s, ReduceSpec) and n != "reduce"
    )
)

#: Order-significant scans, spelled ``name(dim)`` positionally. ``diff`` shrinks its dim,
#: ``cumsum``/``cumprod`` keep it; all three are drawn so the equality-vs-eager property
#: proves a disjoint select or projection may hop while an intersecting one stays put.
SCAN_NAMES = tuple(sorted(n for n, s in OP_TABLE.items() if isinstance(s, ScanSpec)))

#: The function generated ``.reduce`` calls pass. ``np.mean`` deliberately: the
#: drawability constraints of a ``.reduce`` call are those of the function it is given,
#: and ``mean`` is in none of the exclusion sets :func:`_drawable_reduces` applies — so
#: ``"reduce"`` needs no entry in any of them, and would need one if this changed.
REDUCE_METHOD_FUNC = np.mean

#: Whether ``chunk`` calls are generated at all. xrexpr never needs dask, but *replaying* a
#: rechunk does, so without a chunk manager the drawn chain would raise rather than chunk.
HAS_DASK = importlib.util.find_spec("dask") is not None

#: What may follow a ``chunk`` in a generated chain. Dask-backed data is what narrows this,
#: not the optimiser: several ops xarray accepts eagerly it refuses on a chunked array
#: (``median`` over more than one axis, ``groupby`` on a chunked coord, ``rolling`` past a
#: zero-length dim), and a chain that raises *in xarray* tests nothing here. Selects and
#: projections are exactly what the rechunk rules are about, so nothing under test is lost —
#: and a reduce or builder *before* a chunk is still drawn freely.
AFTER_CHUNK = ["isel", "sel", "project", "elementwise", "drop_vars", "rename", "chunk"]

#: Fresh names a generated ``rename`` may relabel a variable to. Disjoint from every name
#: :func:`datasets` produces, so a rename never collides with an existing variable or dim
#: (which xarray would reject); a handful, since a short chain renames at most a few times.
RENAME_TARGETS = ("v_a", "v_b", "v_c", "v_d", "v_e", "v_f")

#: Reductions with no identity element, which numpy refuses to apply to an empty axis
#: ("zero-size array to reduction operation fmax which has no identity"). An empty
#: selection is a case worth generating, so these are skipped when one is in play — the
#: chain would raise in eager xarray too, making it uninteresting here.
NO_IDENTITY_REDUCES = frozenset({"max", "min"})

DIM_NAMES = ("time", "lat", "lon")

# xarray op timing is jittery on small arrays, and generation applies ops eagerly, so a
# per-example deadline just produces flakes. Function-scoped fixtures are not used here
# (the dataset is generated), so that health check is irrelevant rather than suppressed.
#
# ``max_examples`` is raised above Hypothesis's default of 100: the input space widened as
# the generators grew (builder pairs, scans, elementwise ops, and now scattered NaNs), and
# the interesting interactions -- a ``fillna`` and a skipna reduce reordered around a NaN,
# a select threading a specific fused pair -- are a small fraction of draws, so more of them
# is what turns "could generate that" into "did". The suite is cheap (tiny arrays), so this
# stays well within CI's budget.
SETTINGS = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    max_examples=250,
)

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
        """Record the method name and its positional args alongside the keyword mapping.

        Parameters
        ----------
        name : str
            The method to call.
        *args : object
            Positional arguments, kept because some calls only mean what they should when
            spelled positionally — see the class docstring.
        **kwargs : object
            Keyword arguments, which are what the ``dict`` base holds.
        """
        super().__init__(kwargs)
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        """Render the call as source, so a failing example reads as the chain that broke.

        Returns
        -------
        str
            ``name(arg, kw=value)`` — the whole reason this class exists over a tuple.
        """
        parts = [repr(a) for a in self.args]
        parts += [f"{k}={v!r}" for k, v in self.items()]
        return f"{self.name}({', '.join(parts)})"


def _apply(obj, calls):
    """Replay ``calls`` against ``obj`` — a real Dataset, a builder, or a ``.plan`` proxy."""
    for call in calls:
        obj = getattr(obj, call.name)(*call.args, **call)
    return obj


def _assert_replays_equal(optimised, eager):
    """Assert a replayed plan equals its eager spelling, comparing **materialised** values.

    Parameters
    ----------
    optimised : xarray.Dataset or xarray.DataArray
        The result of collecting the ``.plan`` chain.
    eager : xarray.Dataset or xarray.DataArray
        The result of the same chain applied eagerly.

    Notes
    -----
    The generated chains contain ``chunk`` calls, so a side may be dask-backed.
    :func:`~xarray.testing.assert_equal` compares via ``array_equiv``, which computes
    ``(a == b).all()`` — and on dask operands that ``.all()`` is built as a *dask reduction*,
    whose graph construction raises ``AxisError`` when the array carries a degenerate
    ``(1, 0)`` chunk (an empty block), as strided-slicing a block-tuple chunk leaves behind:
    ``chunk(lat=(1, 2))`` then ``isel(lat=slice(0, 2, 2))``. A dask limitation reproducible in
    pure xarray+dask, not a rewrite defect — the values agree.

    Calling ``.compute()`` first does **not** skip the reduction; it moves it off dask.
    Materialising each side is a plain concatenation (which handles the empty block fine),
    so ``assert_equal`` then runs the same ``(a == b).all()`` on *numpy* operands, where it
    is well-defined and the dask-only graph error cannot arise. The chunking under test still
    happened during replay, which is what these properties exercise.
    """
    assert_equal(optimised.compute(), eager.compute())


def _assert_chunking_equal(proxy, eager):
    """Assert an optimised ``.plan`` chain replays to the same block topology as eager.

    Parameters
    ----------
    proxy : xrexpr.accessor.LazyProxy
        The uncollected ``.plan`` chain, replayed through the optimiser but *not* computed,
        so its chunking survives to be inspected.
    eager : xarray.Dataset or xarray.DataArray
        The same chain applied eagerly, dask-backed.

    Notes
    -----
    :func:`_assert_replays_equal` computes both sides, which materialises the chunking away —
    right where the claim is a value, useless where it is the topology. A rechunk preserves
    values whatever blocks it lands on, so only ``.chunks`` can tell a correct merge from one
    that dropped a spec or resolved a repeated dim the wrong way. Replaying without computing
    is what keeps ``.chunks`` populated, mirroring ``test_accessor``'s ``_replayed`` helper.
    """
    replayed = proxy._replay(emit(proxy._optimized()))
    assert dict(replayed.chunks) == dict(eager.chunks)


def _dim_names(ds):
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
    for node in to_lower_ir(plan, _dim_names(ds)):
        schema = apply_schema(schema, node)
    return plan, schema


# --------------------------------------------------------------------------- #
# strategies
# --------------------------------------------------------------------------- #


@st.composite
def datasets(draw, dated=False):
    """A tiny dataset with 2-3 dims, monotonic integer coords and readable values.

    The float data occasionally carries scattered NaNs (see the body): the equality-vs-eager
    property is only as sharp as the data is, and a value-level reorder bug -- a ``fillna``
    hopped past a skipna reduce, say -- changes an answer only where a NaN sits, so a
    NaN-free array would let it pass. The coords stay NaN-free and monotonic, so ``sel`` and
    the size tracking are unaffected.

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
    # Occasionally scatter NaNs into the float data. A NaN is what gives the
    # equality-vs-eager property teeth against a *value* reorder: ``fillna`` and every
    # skipna reduce treat it specially, so fill-then-reduce differs from reduce-then-fill,
    # and a bug that hopped one past the other would change a value only when a NaN is
    # present. Gated on a coin and capped below each array's size (so a reduce still sees
    # real data) -- both so a genuine failure shrinks back to the clean arrays. ``astype``
    # is generated to float dtypes only, so a NaN never meets an integer cast that would
    # raise (see ``_calls``).
    for arr in (values, elevation):
        if arr.size > 1 and draw(st.booleans()):
            holes = draw(
                st.sets(
                    st.integers(min_value=0, max_value=arr.size - 1),
                    min_size=1,
                    max_size=arr.size - 1,  # leave at least one real value
                )
            )
            arr.flat[list(holes)] = np.nan
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
#: contract in ``planning/roadmap/07-small-wins.md`` §8, which an equality assertion cannot
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
    - **``rolling``/``coarsen`` closers are drawn bare.** ``DatasetRolling.mean`` is
      ``(keep_attrs=None, **kwargs)`` — no dim parameter at all. A *positional*
      (``mean("lat")``) binds to ``keep_attrs``; a ``dim=`` *keyword* falls into ``**kwargs``
      and is ignored (with a warning). Either way no dim is reduced. Lowering now *fuses* such
      a closer (the inert dim is dropped, see ``test_lower.py`` / ``test_accessor.py``). It is
      not drawn here because the ``WindowedReduce`` schema and dim effects read only
      ``window`` and ignore ``reduce_args``, so a named-dim closer exercises the same schema
      arm as a bare one, and carrying ``reduce_args`` verbatim through replay is already
      covered — no new path. (The node is *not* identical to the bare case: its
      ``reduce_args`` differ, and a truthy positional ``keep_attrs="lat"`` can even keep attrs
      a bare ``mean()`` would drop.)
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
def _block_lengths(draw, n):
    """Explicit block lengths that add up to ``n`` exactly.

    Parameters
    ----------
    n : int
        The dim's current length.

    Returns
    -------
    tuple of int
        One to three blocks, summing to ``n`` — the only shape dask accepts, since a
        sequence that adds up to anything else raises ``Chunks do not add up to shape``.
    """
    cuts = (
        draw(st.lists(st.integers(1, n - 1), max_size=2, unique=True).map(sorted))
        if n > 1
        else []
    )
    bounds = [0, *cuts, n]
    return tuple(hi - lo for lo, hi in zip(bounds, bounds[1:]))


def _dim_chunk_specs(n):
    """Every chunk spec one dim of length ``n`` can legally be given.

    Parameters
    ----------
    n : int
        The dim's current length, which is at least 1 — see :func:`_rechunks`.

    Returns
    -------
    SearchStrategy
        A strategy over raw specs covering each modelled variant: a block size, the two
        symbolic forms, and an explicit sequence of block lengths.
    """
    return st.one_of(
        st.sampled_from([-1, None, "auto"]),
        st.integers(min_value=1, max_value=n),
        _block_lengths(n),
    )


@st.composite
def _rechunks(draw, obj):
    """A ``chunk`` call that is legal against ``obj``, in a mapping or a uniform form.

    Parameters
    ----------
    obj : xr.Dataset
        The dataset the call will be applied to, drawn against its *current* sizes. Every
        dim is non-empty — the caller checks, because dask's auto-chunking divides by the
        largest block and so raises ``ZeroDivisionError`` on an array with a zero-length
        dim. Empty selections are generated on purpose here, so that combination has to be
        avoided rather than assumed away.

    Returns
    -------
    Call
        The generated call.

    Notes
    -----
    Both forms are drawn because the optimiser treats them differently: a mapping names
    dims, so a select that drops one has to strip the spec (and may empty it out
    entirely), while a uniform spec names none and rides in ``args`` untouched. Within the
    mapping form every :data:`~xrexpr.chunks.ChunkSpec` variant a legal call can carry is
    reachable — including the explicit block lengths that make the rechunk a *barrier*,
    which is the arm no rewrite may cross.
    """
    dims = sorted(map(str, obj.sizes))
    if (
        not dims or draw(st.integers(0, 3)) == 0
    ):  # a quarter uniform, three quarters named
        return draw(
            st.sampled_from(
                [
                    Call("chunk"),
                    Call("chunk", 2),
                    Call("chunk", "auto"),
                    Call("chunk", -1),
                ]
            )
        )
    spec = {
        dim: draw(_dim_chunk_specs(obj.sizes[dim]))
        for dim in draw(st.lists(st.sampled_from(dims), min_size=1, unique=True))
    }
    # Positionally or as dim keywords: two spellings of one mapping, and ``to_opnode``
    # folds them into the same ``chunks`` -- which is the claim being exercised.
    return Call("chunk", spec) if draw(st.booleans()) else Call("chunk", **spec)


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
    chunked = False  # whether a ``chunk`` has already been drawn — see AFTER_CHUNK

    # "builder" twice, so the ``assume`` in ``builder_plans`` discards fewer examples —
    # a chain with no builder pair in it is just a plain chain, already covered.
    kinds = [
        "isel",
        "sel",
        "reduce",
        "scan",
        "project",
        "elementwise",
        "drop_vars",
        "rename",
    ] + (["builder"] * 2 if builders else [])
    if HAS_DASK:
        kinds.append("chunk")

    for _ in range(draw(st.integers(min_value=0, max_value=max_ops))):
        if not current.sizes:
            break  # everything has been reduced away; nothing legal is left

        kind = draw(st.sampled_from(AFTER_CHUNK if chunked else kinds))

        if kind == "chunk":
            if not all(current.sizes.values()):
                continue  # an empty dim: dask cannot auto-chunk it -- see ``_rechunks``
            call = draw(_rechunks(current))
            current = _apply(current, [call])
            calls.append(call)
            chunked = True
            continue

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

        if kind == "drop_vars":
            # ``drop_vars`` removes named variables by key. Draw only names that are *not*
            # dimension coordinates: dropping a dim-coord leaves a coordinate-less dim, on
            # which a later generated ``sel`` cannot draw a label (``indexers`` reads the
            # coord's values during generation). That case -- and the dim-survives-its-coord
            # schema fact it exercises -- is pinned by the hand-written suite instead.
            droppable = sorted(
                str(n) for n in current.variables if n not in current.dims
            )
            if not droppable:
                continue
            names = draw(
                st.lists(st.sampled_from(droppable), min_size=1, unique=True).map(
                    sorted
                )
            )
            call = Call("drop_vars", names)
            current = _apply(current, [call])
            calls.append(call)
            continue

        if kind == "rename":
            # Rename only *variables* (data vars and auxiliary coords), never a dimension or
            # dim-coord: relabelling a dim mid-chain changes dim names, which the
            # ``selected_dims`` bookkeeping and a later ``sel``'s coordinate read would have
            # to track. The dim and dim-coord rename paths -- where ``_relabelled`` also moves
            # ``sizes`` and rewrites dim tuples -- are pinned by the hand-written suite.
            renameable = sorted(
                str(n) for n in current.variables if n not in current.dims
            )
            fresh = [
                n
                for n in RENAME_TARGETS
                if n not in current.variables and n not in current.dims
            ]
            if not renameable or not fresh:
                continue
            call = Call(
                "rename",
                {draw(st.sampled_from(renameable)): draw(st.sampled_from(fresh))},
            )
            current = _apply(current, [call])
            calls.append(call)
            continue

        if kind == "elementwise":
            # A per-element op keeps every dim, size and variable, so it composes anywhere
            # in the chain. Scalar arguments only, so it records ``Elementwise`` -- a
            # data-shaped argument would demote to ``Opaque``, a path the hand-written suite
            # pins instead. ``astype`` casts to *float* only: an int cast raises on a NaN a
            # degenerate reduce upstream can produce (``mean`` over an empty selection),
            # while a float cast never does.
            names = ["astype", "fillna", "clip"]
            if not any(v.dtype == bool for v in current.data_vars.values()):
                names.append("round")  # numpy round refuses a boolean array
            name = draw(st.sampled_from(names))
            if name == "astype":
                call = Call("astype", draw(st.sampled_from(["float32", "float64"])))
            elif name == "round":
                call = Call("round", draw(st.integers(min_value=0, max_value=2)))
            elif name == "clip":
                call = Call("clip", 0.0, 1.0)
            else:
                call = Call("fillna", 0.0)
            current = _apply(current, [call])
            calls.append(call)
            continue

        if kind == "scan":
            scans = SCAN_NAMES
            if any(v.dtype == bool for v in current.data_vars.values()):
                # ``diff`` subtracts, which numpy refuses on booleans (an upstream
                # ``all``/``any`` produces them); ``cumsum``/``cumprod`` cast and are fine.
                scans = tuple(n for n in scans if n != "diff")
            name = draw(st.sampled_from(scans))
            if name == "diff":
                # ``diff`` shrinks its dim by one, so draw a dim long enough to stay
                # non-empty -- an empty dim is legal but noise here, not the point.
                candidates = sorted(d for d, s in current.sizes.items() if s >= 2)
                if not candidates:
                    continue
                dim = draw(st.sampled_from(candidates))
            else:
                dim = draw(st.sampled_from(sorted(current.sizes)))
            call = Call(name, dim)
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
                current, (*REDUCE_NAMES, "reduce"), dims=dims, reduced=set(dims)
            )
            name = draw(st.sampled_from(names))
            # ``keepdims=True`` keeps the named dims at size 1 rather than removing them
            # (#117). The reduce stays a ``Reduce`` and its chains optimise, so the tracked
            # schema must report those dims at size 1 exactly -- which the size-exactness
            # property is where a wrong arm would surface. The same xarray limitations still
            # apply (``max`` over an empty dim raises with or without ``keepdims``), so the
            # ``_drawable_reduces`` filter above needs no adjustment.
            extra = {"keepdims": True} if draw(st.booleans()) else {}
            # ``.reduce`` is spelled ``reduce(func, dim)``, and generated **positionally**
            # on purpose: that is the shape #96 fixed, where the function used to be read
            # as the dim spec. The kwarg spelling was never affected and is pinned by the
            # hand-written suite instead.
            call = (
                Call(name, REDUCE_METHOD_FUNC, dims, **extra)
                if name == "reduce"
                else Call(name, dim=dims, **extra)
            )
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
def coord_projection_plans(draw):
    """A plain chain ending in the pair ``eliminate_projection_before_coord`` targets.

    A list-form data-variable projection ``p1`` followed by a list-form projection ``p2``
    naming coordinates ``p1``'s variables still span. ``p2`` reads those coordinates
    identically from the base, so the optimiser drops ``p1``.

    Its own strategy, and **terminal**, for two reasons the inline :func:`_calls` draw
    could not satisfy. A coordinate projection yields a coord-only dataset, with no data
    variable left to draw a legal downstream op against -- so the pair must come last. And
    ``apply_schema``'s ``Project`` arm *declines* (over-reports) a projection naming a
    coordinate, so a chain carrying one cannot feed :func:`test_tracked_schema_agrees_with_evaluation`;
    keeping it separate keeps it out of the schema-exactness properties while still driving
    the value and anti-vacuity check below. The prefix is builder-free, so both projections
    land in the trusted prefix where the rule looks.
    """
    ds = draw(datasets(dated=draw(st.booleans())))
    calls = draw(_calls(ds))
    current = _apply(ds, calls)
    data_vars = sorted(map(str, current.data_vars))
    assume(bool(data_vars))
    p1 = draw(st.lists(st.sampled_from(data_vars), min_size=1, unique=True).map(sorted))
    # Coordinates present *after* ``p1`` are exactly those that survive it, so drawing ``p2``
    # from them guarantees the rule's survival test passes and it fires.
    survivors = sorted(map(str, _apply(current, [Call("__getitem__", p1)]).coords))
    assume(bool(survivors))
    p2 = draw(st.lists(st.sampled_from(survivors), min_size=1, unique=True).map(sorted))
    return ds, [*calls, Call("__getitem__", p1), Call("__getitem__", p2)]


@st.composite
def drop_projection_plans(draw):
    """A plain chain ending in the ``drop_vars(names)`` then ``[[keep]]`` pair #176 rewrites.

    ``keep`` is a list-form data-variable projection; ``names`` is drawn from the names that
    projection excludes -- other data variables (redundant, the drop is eliminated) and
    coordinates the kept variables still span (survivors, the drop hops behind the projection
    trimmed to them). So :func:`~xrexpr.optimize.push_projection_past_drop` always fires.

    Its own strategy, mirroring :func:`coord_projection_plans`: the random :func:`_calls`
    draw hits this ``(Drop, Project)`` shape too rarely to rely on, and asking for it by name
    is the anti-vacuity guarantee. The prefix is builder-free, so the pair lands in the
    trusted prefix where the rule looks.
    """
    ds = draw(datasets(dated=draw(st.booleans())))
    calls = draw(_calls(ds))
    current = _apply(ds, calls)
    data_vars = sorted(map(str, current.data_vars))
    assume(bool(data_vars))
    keep = draw(
        st.lists(st.sampled_from(data_vars), min_size=1, unique=True).map(sorted)
    )
    # Names the projection excludes and the rule therefore fires on: the other data variables
    # (redundant), plus the coordinates present *after* the projection (survivors).
    survivors = sorted(map(str, _apply(current, [Call("__getitem__", keep)]).coords))
    candidates = sorted({v for v in data_vars if v not in keep} | set(survivors))
    assume(bool(candidates))
    names = draw(
        st.lists(st.sampled_from(candidates), min_size=1, unique=True).map(sorted)
    )
    return ds, [*calls, Call("drop_vars", names), Call("__getitem__", keep)]


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


@st.composite
def rechunk_runs(draw):
    """A dataset paired with a run of >=2 adjacent mapping-form ``chunk`` calls that merge.

    Every drawn spec is one :func:`~xrexpr.optimize._pushable_rechunk` admits -- a block
    size, ``-1`` or ``None`` -- so each node is mapping form *and* pushable, the two
    conditions :func:`~xrexpr.optimize._mergeable_rechunk` needs. The barrier specs
    (``"auto"``, byte targets, explicit block sequences) are drawn out, since a run
    containing one would not fully fold and the anti-vacuity assertion below would fail; the
    goldens in ``test_optimize.py`` cover the barriers instead.
    """
    ds = draw(datasets())
    dims = sorted(map(str, ds.sizes))
    n = draw(st.integers(min_value=2, max_value=4))
    calls = []
    for _ in range(n):
        spec = {
            dim: draw(
                st.one_of(st.integers(1, ds.sizes[dim]), st.sampled_from([-1, None]))
            )
            for dim in draw(st.lists(st.sampled_from(dims), min_size=1, unique=True))
        }
        calls.append(
            Call("chunk", spec) if draw(st.booleans()) else Call("chunk", **spec)
        )
    return ds, calls


# --------------------------------------------------------------------------- #
# properties
# --------------------------------------------------------------------------- #


def _extent_sized(node):
    """Report whether ``node`` carries a spec dask resolves by measuring the array.

    Notes
    -----
    Spelled out here rather than imported because the optimiser has no such predicate to
    import: :func:`~xrexpr.optimize._pushable_rechunk` decides the whole node in one
    ``match`` and never names this subset. Restating it keeps the property a statement about
    *behaviour* — the adjacency the plan must not contain — rather than a re-run of the
    implementation against itself.

    :class:`~xrexpr.chunks.BlockSeq` is included with the two auto-sizing specs: all three
    are resolved against the extent, which is what makes the adjacency hazardous.
    """
    specs = (*node.chunks.values(), *((node.uniform,) if node.uniform else ()))
    return any(isinstance(spec, Auto | ByteSize | BlockSeq) for spec in specs)


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
    for node in to_lower_ir(_build_plan(ds, calls)[0], _dim_names(ds)):
        if type(node).__name__.endswith("Reduce") and type(node).__name__ != "Reduce":
            event(f"fused: {type(node).__name__}")
    _assert_replays_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@SETTINGS
@given(any_plans())
def test_roundtrip_intern_schema_unchanged(case):
    ds, calls = case
    _, schema = _build_plan(ds, calls)

    interner = Interner()
    interned_schema = schema.to_interned(interner)
    roundtrip_schema = SchemaState.from_interned(interned_schema, interner)

    assert roundtrip_schema == schema


@SETTINGS
@given(any_plans())
def test_roundtrip_intern_ops_unchanged(case):
    """Every op survives ``deintern(intern(op)) == op``.

    Covers both the fluent plan — which may carry a ``ContextOpen`` — and its lowered form,
    which carries the fused ``*Reduce`` nodes, so all op variants are exercised.
    """
    ds, calls = case
    Interner()._clear()

    fluent = _build_plan(ds, calls)[0]
    lowered = to_lower_ir(fluent, _dim_names(ds))
    for op in (*fluent, *lowered):
        assert deintern(intern(op)) == op


@pytest.mark.skipif(not HAS_DASK, reason="rechunk replay needs a chunk manager")
@SETTINGS
@given(any_plans())
def test_optimised_plan_matches_eager_evaluation_under_a_tiny_chunk_target(case):
    """The headline property again, with ``array.chunk-size`` small enough to make ``"auto"`` bite.

    Notes
    -----
    This is what makes the generated chains a real test of issue #121 rather than a vacuous
    one. Dask's ``auto_chunks`` pins every dim below ``limit ** (1 / ndim)`` and returns
    *before* it can divide, so at the default 128 MiB target these 1-to-5-long dims survive
    an emptied neighbour by arithmetic — a rule that made the hop would pass here all day.
    At 1 kB they do not.

    The setting stands in for a multi-hundred-megabyte fixture, and it names the condition
    the crash actually depends on, which a fixture would only imply. It is also why
    ``_calls`` no longer refuses to draw an emptying select once a ``chunk`` is in the
    chain: that exclusion was #121's, and this is the property that would report it back.
    """
    import dask

    ds, calls = case
    with dask.config.set({"array.chunk-size": "1kB"}):
        _assert_replays_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@SETTINGS
@given(any_plans())
def test_no_rule_moves_a_select_in_front_of_an_extent_sized_rechunk(case):
    """No rule moves a select from *after* an extent-dependent rechunk to *before* it.

    Notes
    -----
    ``pushdown_selects_past_rechunks`` refuses to make this move, but refusing is only half
    the claim: any rule that reorders nodes could make it, and the plan would then resolve an
    extent-dependent spec against data the user never rechunked. So the assertion is about
    the optimised plan as a whole, quantified over every generated chain rather than over one
    rule's inputs.

    The count is over **ordered pairs, not adjacent ones**, and that is the whole content of
    the property. Adjacency is not an invariant of anything: ``pushdown_projections`` closing
    the gap in ``isel(...) → project → chunk("auto")`` creates an adjacent pair without
    moving the select, which was in front of the rechunk as the user wrote it. Counting
    ``i < j`` instead asks the question that matters — did a select *cross* a rechunk — and
    leaves a chain the user wrote that way alone, since replaying it faithfully means
    chunking exactly what they asked to chunk.

    ``<=`` rather than ``==`` because merging is free to shrink the count:
    ``merge_adjacent_selects`` folds two selects into one, and a select spent entirely on a
    dropped dim disappears.
    """
    ds, calls = case
    lowered = to_lower_ir(_build_plan(ds, calls)[0], _dim_names(ds))

    def crossings(nodes):
        rechunks = [
            i
            for i, n in enumerate(nodes)
            if isinstance(n, Rechunk) and _extent_sized(n)
        ]
        return sum(
            i < j
            for i, n in enumerate(nodes)
            if isinstance(n, Select)
            for j in rechunks
        )

    event(
        f"selects in front of an extent-sized rechunk as written: {crossings(lowered)}"
    )
    assert crossings(optimize(lowered, SchemaState.from_dataset(ds))) <= crossings(
        lowered
    )


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
    once = optimize(to_lower_ir(plan, _dim_names(ds)), schema)
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
    once = to_lower_ir(plan, _dim_names(ds))
    assert to_lower_ir(once, _dim_names(ds)) == once


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
        isinstance(node, ContextOpen) for node in to_lower_ir(plan, _dim_names(ds))
    )


@SETTINGS
@given(any_plans())
def test_emit_after_lowering_is_a_reparseable_canonical_fixed_point(case):
    """Emitted calls re-parse and re-lower to the same calls — a canonical fixed point.

    ``emit`` derives each header from the node's *semantic fields* (the inverse of
    ``to_opnode``), so it emits the **canonical** spelling of a call, not necessarily the
    recorded one. The round-trip half of the contract is therefore not byte-identity with
    what was recorded, but *stability*: parse the emitted calls back with ``to_opnode``,
    lower them again, and emit — and the calls are identical. That pins two things at once —
    every emitted call is one ``to_opnode`` accepts, and canonicalisation has a fixed point
    (a second pass changes nothing).

    Holds for builder chains too, which is why re-lowering is part of the loop rather than a
    bare ``to_opnode``: a fused node emits *two* calls, whose re-parse is an opener plus a
    closer that ``to_lower_ir`` re-fuses. The literal replay equivalence — that these calls
    evaluate as eager does — is :func:`test_optimised_plan_matches_eager_evaluation`.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    dims = _dim_names(ds)

    emitted = emit(to_lower_ir(plan, dims))
    reparsed = [to_opnode(c.name, c.args, dict(c.kwargs)) for c in emitted]
    assert emit(to_lower_ir(reparsed, dims)) == emitted


@SETTINGS
@given(select_runs())
def test_adjacent_selects_collapse_without_changing_meaning(case):
    """A run of same-name selects folds to one node that means the same thing.

    Asserting the fold *fires* keeps this from passing vacuously: an over-cautious merge
    rule that never merged would satisfy the equality half on its own.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    optimised = optimize(
        to_lower_ir(plan, _dim_names(ds)), SchemaState.from_dataset(ds)
    )

    assert len(optimised) == 1, "a run of selects on distinct dims should fold to one"
    _assert_replays_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@pytest.mark.skipif(not HAS_DASK, reason="rechunk replay needs a chunk manager")
@SETTINGS
@given(rechunk_runs())
def test_adjacent_rechunks_fuse_without_changing_the_chunking(case):
    """A run of mapping-form ``chunk`` calls folds to one node that lands on the eager blocks.

    Two claims, and the second is the load-bearing one: the fold *fires* (so this cannot
    pass vacuously against a merge that never merged), and the single node reaches the exact
    block topology dask does by applying each call on top of the last —
    :func:`_assert_chunking_equal`, not :func:`_assert_replays_equal`, because rechunking
    preserves values whatever blocks it picks, so only ``.chunks`` can catch a dropped spec
    or a repeated dim resolved the wrong way.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    optimised = optimize(
        to_lower_ir(plan, _dim_names(ds)), SchemaState.from_dataset(ds)
    )

    assert len(optimised) == 1, "a run of mergeable rechunks should fold to one"
    _assert_chunking_equal(_apply(ds.plan, calls), _apply(ds, calls))


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
    - **coords**: exactly the result's, not merely a subset. A scalar select drops the dim
      while xarray keeps a scalar coordinate behind, and an aggregating op removes that
      coordinate outright — both are modelled now that a coordinate is a variable with
      dims, so there is nothing left to be conservative about.
    - **variables**: each with **the dims it spans**, not merely by name. The store is
      per-variable, so a name-only comparison cannot see a variable whose dims are wrong
      — and the dataset-level dim set cannot either, as long as some *sibling* still
      carries the dim. That is exactly how issue #125 survived here: the ``WeightedReduce``
      arm skipped minting ``time`` onto ``elevation`` because ``temperature`` still had it,
      and every dataset-level fact stayed true. It became visible only through a trailing
      ``[["elevation"]]``, a conjunction this generator draws roughly once in 20,000 plans.

    Dim *order* is deliberately not asserted: ``Dataset.sizes`` does not promise the
    insertion order the schema threads, and ``_minted`` sorts minted dims by ``str`` to
    keep a ``SchemaState`` a value across ``PYTHONHASHSEED`` (see
    ``test_several_minted_dims_land_in_a_deterministic_order``), which is its own order
    rather than xarray's. Comparing as sets is the honest check; the ordering that *is*
    promised is pinned by example instead.

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
    assume(
        not any(isinstance(node, Opaque) for node in to_lower_ir(plan, _dim_names(ds)))
    )
    result = _apply(ds, calls)

    assert set(schema.sizes) == set(result.sizes)

    # Every tracked coordinate, not just the dim coordinates — the restriction issue #109
    # asked for is gone, because coordinates are variables now and their lifetimes are
    # modelled per-op: an aggregating op drops one over the dim it aggregates, an indexing
    # one demotes it to 0-d. Asserted **exactly** rather than as a subset, which the
    # bare-name ``coords`` set could never have supported.
    #
    # Checked unconditionally on every version. ``cumsum``/``cumprod`` drop the scanned
    # dim's coordinates on xarray before the 2026.04.0 retention fix (pull 10987) and keep
    # them after, and ``apply_schema`` reproduces that split bug-for-bug
    # (``schema.SCAN_DROPS_SCANNED_COORDS``) — so the schema is exact on both sides of the
    # fix and this compares against the real evaluation with no version relaxation.
    assert set(schema.coords) == set(result.coords)

    # Per variable, which subsumes the name-set comparison this replaced *and* the size
    # keys above — ``sizes`` is pruned to ``dim_names``, which is derived from these very
    # dims. Both are kept anyway: they fail first and say less when they do, so a shrunk
    # counterexample reads as "a dim went missing" before it reads as a variables diff.
    assert {k: set(v) for k, v in schema.variables.items()} == {
        k: set(v.dims) for k, v in result.variables.items()
    }


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

    **Kept xfail deliberately** (``planning/roadmap/03-schema-sizes.md`` §5.5). The size
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
    lowered = to_lower_ir(plan, _dim_names(ds))
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

    The **per-variable schema** is asserted here as well as in
    :func:`test_tracked_schema_agrees_with_evaluation`, and for the reason this test
    exists: each fused arm mints or resizes dims on its own terms, and asking for the kind
    by name is what makes that arm's schema checked *every run* rather than on the ~9% of
    random chains that happen to contain one. Issue #125 — a weight dim minted onto the
    dataset instead of onto each variable — is caught by ~12% of the ``weighted`` pairs
    drawn here, so this parametrisation fails essentially every run, where the general
    property found it in none of 20,000 plans.
    """
    ds = data.draw(datasets(dated=True))
    pair = data.draw(_builder_pair(ds, kind=kind))

    assert pair is not None and pair[0].name == kind
    plan, schema = _build_plan(ds, pair)
    lowered = to_lower_ir(plan, _dim_names(ds))
    event(f"{kind}: {'fused' if len(lowered) == 1 else 'opaque pair'}")
    assert len(lowered) == 1 or kind in {"groupby", "resample"}

    eager = _apply(ds, pair)
    assert_equal(_apply(ds.plan, pair).collect(), eager)

    # Only where the schema claims to know: an unfused pair lowers to ``Opaque``, which
    # ``apply_schema`` models as dim- and variable-preserving and a per-group map is not.
    # The same trust boundary the ``assume`` above states, spelled as a guard because a
    # kind that *stopped* fusing should still be asserted about, not discarded.
    if len(lowered) == 1:
        assert {k: set(v) for k, v in schema.variables.items()} == {
            k: set(v.dims) for k, v in eager.variables.items()
        }


@SETTINGS
@given(coord_projection_plans())
def test_a_coord_projection_drops_its_input_and_replays_equal(case):
    """Anti-vacuity plus value: the coord-only projection pair is generated, fires, and preserves the answer.

    ``eliminate_projection_before_coord`` fires on no random chain -- :func:`_calls` draws
    only data-variable projections -- so it is asked for by name here, the way
    :func:`test_every_builder_kind_is_generated_and_replays_equal_to_eager` asks for each
    builder kind. :func:`coord_projection_plans` ends every chain in the ``[[data_vars]]``,
    ``[[coords]]`` pair the rule collapses, and both facts are pinned: that the rule fires
    (its own return on the lowered plan is not ``None``, checked order-independently rather
    than by inferring a drop that :func:`merge_adjacent_projects` could also cause), and
    that the full optimiser's answer still equals eager.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    lowered = to_lower_ir(plan, _dim_names(ds))
    assert (
        eliminate_projection_before_coord(lowered, SchemaState.from_dataset(ds))
        is not None
    )
    _assert_replays_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@given(drop_projection_plans())
def test_a_drop_before_a_projection_is_absorbed_and_replays_equal(case):
    """Anti-vacuity plus value: the ``drop_vars`` then projection pair fires #176 and preserves the answer.

    ``push_projection_past_drop`` fires on no random chain reliably -- the ``(Drop, Project)``
    adjacency with the right schema shape is too rare -- so it is asked for by name here, like
    :func:`test_a_coord_projection_drops_its_input_and_replays_equal`. Both facts are pinned:
    that the rule fires (its return on the lowered plan is not ``None``), and that the full
    optimiser's answer still equals eager, whether the drop was eliminated or hopped and
    trimmed.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    lowered = to_lower_ir(plan, _dim_names(ds))
    assert push_projection_past_drop(lowered, SchemaState.from_dataset(ds)) is not None
    _assert_replays_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))
