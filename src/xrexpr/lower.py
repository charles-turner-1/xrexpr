"""Lowering — what the user *wrote* → what it *means* → the calls that reproduce it.

The two pure translation stages either side of the optimiser. :func:`to_lower_ir` fuses
the builder pairs xarray spells as two calls (``ds.groupby("time.month").mean()``) into
the three fused kinds, and :func:`emit` maps one lowered node back to the call sequence
that reproduces it — which is what makes a node standing for *two* calls ordinary rather
than a special case the replay loop has to know about.

Fusion is a stage rather than a rewrite rule for two reasons, and both matter when editing
here. :func:`~xrexpr.schema.to_opnode` is a *per-call* function and cannot see a pair;
``optimize``'s rules run to a shared fixpoint under a strictly-decreasing measure, whereas
fusion runs **once** and is a *precondition* of those rules rather than one of them. The
contract is that :func:`to_lower_ir` is **semantics-preserving** — ``emit(to_lower_ir(p))``
replays as ``p`` does — and **idempotent**, which is what lets it run without first asking
whether it has.

A builder pair no rule claims (``rolling_exp``, ``cumulative``) demotes to
:class:`~xrexpr.ir.Opaque`, taking *both* halves with it.

See ``docs/internals/lowering.md``.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr.ir import (
    AllDims,
    ContextOpen,
    DimSet,
    Drop,
    Elementwise,
    FluentOp,
    GroupedReduce,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Rename,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.operations import CHUNK_OPTION_KWARGS, SELECT_OPTION_KWARGS, ReduceSpec
from xrexpr.operations import spec as op_spec

__all__ = ["Call", "emit", "to_lower_ir"]

#: ``rolling``/``coarsen`` keyword arguments that are *options*, not per-dim windows
#: (xarray 2026.7.0's signatures, unioned — the two share this shape).
_WINDOW_OPTION_KWARGS = frozenset(
    {"min_periods", "center", "boundary", "side", "coord_func", "keep_attrs"}
)

#: ``resample`` keyword arguments that are *options*, not the dim being resampled
#: (xarray 2026.7.0's signature). Drift is safe in one direction only, and it is the
#: right one: an option this misses leaves two candidate keys, which refuses to fuse.
_RESAMPLE_OPTION_KWARGS = frozenset(
    {"closed", "label", "offset", "origin", "restore_coord_dims", "skipna"}
)


@dataclass(frozen=True)
class Call:
    """One xarray method invocation — the unit replay actually performs.

    Attributes
    ----------
    name : str
        The method to invoke, e.g. ``"mean"`` or ``"__getitem__"``.
    args : tuple
        Positional arguments, verbatim.
    kwargs : frozendict
        Keyword arguments, verbatim.

    Notes
    -----
    Deliberately *not* an :data:`~xrexpr.ir.Op`: a node carries the normalised metadata
    the optimiser reasons about, whereas a ``Call`` is codegen output and carries only
    what ``getattr(ds, name)(*args, **kwargs)`` needs. Keeping them distinct is what lets
    one node emit several calls, and stops replay from growing an arm per variant.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


def to_lower_ir(
    nodes: list[FluentOp], dim_names: frozenset[Hashable]
) -> list[LoweredOp]:
    """Translate a recorded plan into what it means, fusing builder chains.

    Parameters
    ----------
    nodes : list of FluentOp
        The recorded plan, one node per call as the fluent API spelled it.
    dim_names : frozenset of Hashable
        The base dataset's dim names — the one thing this pass cannot read off the calls.
        See the notes for why base dims, and why names rather than a schema.

    Returns
    -------
    list of LoweredOp
        The same plan with every builder pair resolved: fused into one node where a
        rule claims it, demoted to :class:`~xrexpr.ir.Opaque` where none does. Contains
        no :class:`~xrexpr.ir.ContextOpen`, which its return type enforces.

    Notes
    -----
    Where the fluent API is one-to-one with semantics the recorded node is already right,
    so it passes through untouched — this is the *same* vocabulary plus the nodes a single
    call cannot express, not a translation into a poorer language. Only the multi-call
    spellings are rewritten.

    ``dim_names`` is the **base dataset's dim names**, and it is the one thing this pass cannot
    read off the calls: whether ``groupby("region")`` groups along a dim or along a
    coordinate defined on one is a fact about the data, not the call (see
    ``_grouper_dims``). Dim names rather than a ``SchemaState`` because that is the
    whole of what lowering needs to know — a set keeps this module's dependencies at
    ``xrexpr.ir`` alone, and makes the requirement legible in the signature.

    Base dims rather than the dims *at* each opener, which would mean folding the schema
    forward through lowering. The gap is one-sided and pessimising: a dim minted mid-plan
    and then grouped over (``groupby("time.month").mean().groupby("month")``) refuses and
    falls to the fallback below. The converse — a name that is a base dim but no longer one
    by the time of the groupby — needs an intervening ``stack``/``rename``, which is
    ``Opaque``, so the plan is already past ``optimize._trusted_prefix`` and the tracked
    schema is a guess by contract rather than by this shortcut.

    Contexts are **pairs, not runs** (verified against xarray 2026.7.0): there is no
    builder→builder middle call to skip, because ``DatasetGroupBy.__getitem__`` selects a
    *group* and closes the context, and Rolling/Coarsen/Weighted reject ``__getitem__``
    outright. So this matches adjacent pairs only.

    Every :class:`~xrexpr.ir.ContextOpen` must leave, and one that no fusion rule claims
    is **demoted to** :class:`~xrexpr.ir.Opaque` together with its closer — the pair then
    replays verbatim and no rule can fire on it. That fallback is what makes the closer's
    provisional typing safe: a closer recorded under a shape no rule expects simply fails
    to match, so the failure mode is pessimisation, never wrongness. It is also what the
    openers no fused node describes (``rolling_exp``, ``cumulative``) rely on permanently,
    rather than only until their own rule lands.

    Idempotent and semantics-preserving (see the module docstring). Idempotence is not
    incidental: the output contains no ``ContextOpen`` at all, so a second pass has
    nothing left to match.
    """
    out: list[LoweredOp] = []
    i, n = 0, len(nodes)
    while i < n:
        node = nodes[i]
        if not isinstance(node, ContextOpen):
            out.append(node)
            i += 1
            continue

        closer = nodes[i + 1] if i + 1 < n else None
        fused = (
            _fuse_grouped(node, closer, dim_names)
            or _fuse_windowed(node, closer)
            or _fuse_weighted(node, closer)
            if closer is not None
            else None
        )
        if fused is not None:
            out.append(fused)
            i += 2
            continue

        # The mandatory fallback. The *closer* must be demoted with the opener, not just
        # the opener: a closer left as the node ``to_opnode`` provisionally typed it is a
        # Dataset-level reading of a call that was never Dataset-level --
        # ``rolling(time=2).mean()`` would keep a ``Reduce`` whose bare dim spec means
        # "every dim", and a following select would be rejected against dims the grouped
        # mean never removed. Demoting the pair is what makes the provisional typing safe.
        out.append(Opaque(name=node.name, args=node.args, kwargs=node.kwargs))
        if closer is not None:
            out.append(Opaque(name=closer.name, args=closer.args, kwargs=closer.kwargs))
            i += 2
        else:
            i += 1  # an unclosed context: the opener is the whole of it
    return out


def _fuse_grouped(
    opener: ContextOpen, closer: FluentOp, dim_names: frozenset[Hashable]
) -> GroupedReduce | None:
    """Fuse a groupby-family pair into a :class:`~xrexpr.ir.GroupedReduce`, or refuse.

    Parameters
    ----------
    opener : ContextOpen
        The builder-returning call.
    closer : FluentOp
        The node recorded immediately after it.
    dim_names : frozenset of Hashable
        The base dataset's dim names, passed through to ``_grouper_dims`` — a grouper
        that does not name one refuses (issue #90).

    Returns
    -------
    GroupedReduce or None
        The fused node, or ``None`` to refuse — which the caller turns into a verbatim
        opaque pair.

    Notes
    -----
    Refusing is always safe (the caller demotes to ``Opaque``), so every condition below
    is a *narrowing* of what v1 claims to understand rather than a correctness risk.

    - **The opener must be groupby-family.** ``rolling``/``coarsen`` and ``weighted`` have
      their own nodes, fused by the two functions below, so they refuse here.
    - **The grouper must be a string naming a dim** — a dim name (``groupby("lat")``) or a
      component of a dim coordinate (``groupby("time.month")``), from which ``group_dim``
      reads off directly. A ``DataArray`` grouper or a ``Grouper`` object has no single
      ``group_dim``, so it refuses; so does a *non-dim coordinate* name, which is what
      ``dim_names`` is threaded here to establish (``_grouper_dims``).
    - **The closer must be an aggregating reduce.** This is the subtle one, and it is a
      correction to the obvious reading: a grouped reduce over dims that *exclude* the
      group dim is not an aggregation at all. ``ds.groupby("time.month").mean("lat")``
      returns dims ``{time, lon}`` — a per-group *map*, reassembled along the original
      dim, minting no ``month``. Only a bare closer or one naming the group dim
      aggregates, and only that case fuses.
    """
    match opener.name:
        # Spelled as a match rather than a set membership so mypy narrows the opener's
        # name to ``GroupedReduce``'s own (narrower) Literal -- the set would need a cast,
        # and a cast is exactly the check being skipped.
        case "groupby" | "groupby_bins" | "resample":
            kind = opener.name
        case _:
            return None
    if not isinstance(closer, Reduce) or closer.keepdims:
        # A grouped ``keepdims=True`` reduce is nothing ``GroupedReduce`` models. Before
        # pydata/xarray#11521 it *retained the group dim* (sized to the group count, and
        # left without a coordinate) instead of minting ``new_dim`` --
        # ``groupby("time.month").mean(keepdims=True)`` kept a coordinate-less ``time`` and
        # never minted ``month`` (pydata/xarray#11519); #11521 stops groupby/resample
        # honouring ``keepdims`` at all, so it now raises instead. ``GroupedReduce``
        # describes neither (it removes the group dim and mints ``new_dim``), so fusing
        # would misdescribe it: refuse and let the pair replay verbatim, exactly as an
        # opaque closer did before #117.
        return None

    grouped = _grouper_dims(opener, dim_names)
    if grouped is None:
        return None
    group_dim, new_dim = grouped

    # A bare closer consumes nothing extra: unlike a Dataset-level ``mean()``, a grouped
    # one reduces only along the group dim, so ``lat``/``lon`` survive.
    match closer.consumes:
        case AllDims():
            extra: frozenset[Hashable] = frozenset()
        case frozenset() as named if group_dim in named:
            extra = named - {group_dim}
        case _:
            return None  # names dims but not the group dim: the map case

    return GroupedReduce(
        name=kind,
        group_dim=group_dim,
        new_dim=new_dim,
        reduce=closer.name,
        args=opener.args,
        kwargs=opener.kwargs,
        reduce_args=closer.args,
        reduce_kwargs=closer.kwargs,
        consumes=extra,
    )


def _fuse_windowed(opener: ContextOpen, closer: FluentOp) -> WindowedReduce | None:
    """Fuse a rolling/coarsen pair into a :class:`~xrexpr.ir.WindowedReduce`, or refuse.

    Parameters
    ----------
    opener : ContextOpen
        The builder-returning call.
    closer : FluentOp
        The node recorded immediately after it.

    Returns
    -------
    WindowedReduce or None
        The fused node, or ``None`` to refuse — which the caller turns into a verbatim
        opaque pair.

    Notes
    -----
    ``rolling_exp`` and ``cumulative`` are deliberately not here: the first is an
    exponential weighting rather than a fixed window, and the second is a scan wearing a
    builder's clothes. Neither is described by ``window``, so both keep replaying
    verbatim rather than being squeezed into a node that would misdescribe them.

    **The closer must have named no dim** — and that is a stronger condition than it
    looks. ``DatasetRolling.mean`` is ``(keep_attrs=None, **kwargs)``: it takes no dim
    argument at all, so ``ds.rolling(time=3).mean("lat")`` passes ``"lat"`` as
    *keep_attrs* and reduces nothing. ``to_opnode``, which cannot know the call is
    windowed, records that as ``consumes={"lat"}``. Fusing it would import a dim effect
    invented by a misparse, so a closer carrying any parsed dim spec is refused and the
    pair replays verbatim.
    """
    match opener.name:
        case "rolling" | "coarsen":
            kind = opener.name
        case _:
            return None
    if not isinstance(closer, Reduce) or closer.keepdims:
        # ``keepdims=True`` still refuses: ``coarsen.mean(keepdims=True)`` raises
        # ``TypeError`` and rolling's is an inert no-op -- neither a size effect a
        # ``WindowedReduce`` describes -- so both stay ``Opaque`` to replay verbatim.
        #
        # A closer that *named a dim* no longer refuses. A windowed reduce has no dim
        # parameter (``mean(keep_attrs=None, **kwargs)``): a positional (``mean("lat")``)
        # binds to ``keep_attrs``, a ``dim=`` keyword falls into ``**kwargs`` and is ignored
        # (with a warning). Either way the closer reduces no dim, so ``rolling(time=12)
        # .mean("lat")`` reduces the same dims as ``.mean()`` (differing only in attrs). The
        # ``to_opnode`` misparse in ``closer.consumes`` is dropped here rather than carried,
        # and the node reads neither it nor ``reduce_args`` -- its schema and dim effects
        # depend only on ``window``. ``reduce_args`` rides along verbatim, so replay
        # reproduces the closer (``keep_attrs`` truthiness included) exactly.
        return None

    window = _window_spec(opener)
    if not window:
        return None  # no statically-known window: nothing to model

    return WindowedReduce(
        name=kind,
        reduce=closer.name,
        window=window,
        args=opener.args,
        kwargs=opener.kwargs,
        reduce_args=closer.args,
        reduce_kwargs=closer.kwargs,
    )


def _window_spec(opener: ContextOpen) -> frozendict[Hashable, int]:
    """Extract the ``{dim: window}`` mapping of a ``rolling``/``coarsen`` call.

    Parameters
    ----------
    opener : ContextOpen
        The ``rolling`` or ``coarsen`` call.

    Returns
    -------
    frozendict
        The windowed dims and their integer window sizes. Empty when no window is
        statically known, which the caller reads as a refusal to fuse.

    Notes
    -----
    Both spell it the same two ways — a positional mapping or dim keywords — alongside
    option kwargs that are not dims, exactly as ``isel`` and ``chunk`` do; this is
    ``schema._select_indexer``'s shape for a third caller. A window that isn't a plain
    ``int`` is dropped rather than modelled, which (an empty result being a refusal to
    fuse) means an unrecognised spelling replays verbatim instead of being guessed at.
    """
    raw: dict[Hashable, Any] = {}
    if opener.args and isinstance(opener.args[0], dict):
        raw.update(opener.args[0])
    raw.update(
        {k: v for k, v in opener.kwargs.items() if k not in _WINDOW_OPTION_KWARGS}
    )
    return frozendict({dim: size for dim, size in raw.items() if isinstance(size, int)})


def _fuse_weighted(opener: ContextOpen, closer: FluentOp) -> WeightedReduce | None:
    """Fuse a weighted pair into a :class:`~xrexpr.ir.WeightedReduce`, or refuse.

    Parameters
    ----------
    opener : ContextOpen
        The builder-returning call.
    closer : FluentOp
        The node recorded immediately after it.

    Returns
    -------
    WeightedReduce or None
        The fused node, or ``None`` to refuse — which the caller turns into a verbatim
        opaque pair.

    Notes
    -----
    Unlike ``_fuse_windowed``, a closer that named dims is **fused, not refused**, and
    the asymmetry is a fact about the two signatures rather than a judgement:
    ``DatasetWeighted.mean`` is ``(dim=None, *, skipna=None, keep_attrs=None)`` — it really
    does take a dim — whereas ``DatasetRolling.mean`` takes none, which is what makes
    ``rolling(time=3).mean("lat")`` a misparse. So ``to_opnode``'s parsed dim spec is
    meaningful here and carries over untouched, :data:`~xrexpr.ir.ALL_DIMS` included.

    The one thing that must be read off the opener is the **weights' dims**, because they
    have a dim effect of their own (see :class:`~xrexpr.ir.WeightedReduce`). Reading
    ``w.dims`` materialises nothing — it is metadata, the same class of access
    ``SchemaState.from_dataset`` already makes. Duck-typed rather than
    ``isinstance(w, xr.DataArray)`` so this module keeps needing no xarray import; anything
    with no ``dims`` tuple refuses, which covers the weights xarray would itself reject
    (it raises ``ValueError: `weights` must be a DataArray``) and costs only a fusion the
    plan could not have replayed anyway.

    An untabulated closer refuses for free by not being a ``Reduce``, which is the right
    answer for the weighted-only methods: ``sum_of_weights``/``sum_of_squares``/``quantile``
    are not in ``OP_TABLE``, and ``quantile`` in particular takes ``q`` first positionally,
    which a reduce's dim-spec parse would misread.
    """
    if opener.name != "weighted" or not isinstance(closer, Reduce) or closer.keepdims:
        # A ``keepdims=True`` closer keeps its dims at size 1: fusing it into a
        # ``WeightedReduce`` (which removes the weighted dims) would misdescribe it, so
        # refuse and replay verbatim.
        return None

    weight_dims = _weight_dims(opener)
    if weight_dims is None:
        return None

    return WeightedReduce(
        name="weighted",
        reduce=closer.name,
        weight_dims=weight_dims,
        args=opener.args,
        kwargs=opener.kwargs,
        reduce_args=closer.args,
        reduce_kwargs=closer.kwargs,
        consumes=closer.consumes,
    )


def _weight_dims(opener: ContextOpen) -> frozenset[Hashable] | None:
    """Read the dims the weights carry off a ``weighted`` call.

    Parameters
    ----------
    opener : ContextOpen
        The ``weighted`` call, holding the weights in its verbatim header.

    Returns
    -------
    frozenset of Hashable or None
        The weights' dim names, or ``None`` if the weights aren't a shape this can read —
        which the caller reads as a refusal to fuse.

    Notes
    -----
    ``Dataset.weighted`` takes exactly one argument, so the weights are ``args[0]`` or the
    ``weights=`` keyword. A 0-d weights array answers the empty set, which is honest: it
    broadcasts nothing and aligns nothing, so such a node's dim effect really is a plain
    reduce's.
    """
    weights = opener.args[0] if opener.args else opener.kwargs.get("weights")
    dims = getattr(weights, "dims", None)
    return frozenset(dims) if isinstance(dims, tuple) else None


def _grouper_dims(
    opener: ContextOpen, dim_names: frozenset[Hashable]
) -> tuple[Hashable, Hashable] | None:
    """Read ``(group_dim, new_dim)`` off a groupby-family opener.

    Parameters
    ----------
    opener : ContextOpen
        The ``groupby``, ``groupby_bins`` or ``resample`` call.
    dim_names : frozenset of Hashable
        The base dataset's dim names. Every reading below assumes the grouper names a
        dim, so every one is checked against this — see the notes.

    Returns
    -------
    tuple of Hashable or None
        The dim grouped along and the dim minted, or ``None`` when neither is statically
        known — which the caller reads as a refusal to fuse.

    Notes
    -----
    Read off the call, since v1 fuses only string groupers:

    - ``groupby("time.month")`` groups along ``time`` and mints ``month``;
    - ``groupby("lat")`` mints the dim it grouped over, now holding the distinct values;
    - ``groupby_bins("lat", 2)`` mints ``lat_bins`` — xarray's own naming convention;
    - ``resample(time="2D")`` takes its dim from the (single) keyword, and mints it back.

    **Every one of those reads assumes the name is a dim, so every one is checked against**
    ``dim_names``. The name alone cannot tell them apart: ``groupby("region")`` where ``region``
    is a coordinate *on* ``lat`` groups along ``lat`` and mints ``region``, so reading
    ``group_dim`` off the call gives ``region`` — a dim the grouping never consumed, while
    the one it did consume is reported as surviving. That was issue #90, and it reached
    ``dim_effect``'s ``blocks``/``requires`` as well as the schema fold. Refusing puts these
    pairs on the opaque fallback, which is what ``02-lowering.md`` §5.5 specified all along.

    Verified against xarray 2026.7.0, since the guard's reach is wider than the issue's
    title suggests — all three kinds share the assumption:

    - ``resample`` accepts a non-dim coordinate too (``resample(t2="2D")`` with ``t2`` on
      ``lat`` yields ``{t2: 2, time: 4}``), so the keyword is checked, not trusted;
    - ``groupby_bins("time.month", 2)`` yields ``{lat: 3, month_bins: 2}`` — group dim
      ``time``, minting ``month_bins``, whereas this function's pre-guard reading was
      ``("time.month", "time.month_bins")``, wrong twice. The dotted bins form is therefore
      refused rather than corrected: fusing it needs its own node-level goldens, and
      refusing is the safe half of the fix. Future work.
    - a multi-dim coordinate grouper stacks (``cell`` on ``(time, lat)`` yields
      ``{cell: 3}``, consuming both), which is why modelling non-dim groupers properly
      needs a ``group_dim`` that is a *set* — the larger question #90 leaves open.

    A ``Grouper`` object or a keyword grouper (``groupby(x=UniqueGrouper())``) already
    refuses by leaving ``args`` empty or non-``str``; the guard does not change that.
    """
    if opener.name == "resample":
        keys = [k for k in opener.kwargs if k not in _RESAMPLE_OPTION_KWARGS]
        if len(keys) != 1 or keys[0] not in dim_names:
            return None
        return keys[0], keys[0]

    if not opener.args or not isinstance(grouper := opener.args[0], str):
        return None
    if opener.name == "groupby_bins":
        # Guarded on the whole grouper, not its head: the dotted form's real group dim
        # *is* the head, but its minted name is not ``f"{grouper}_bins"``, so a head-only
        # check would let the wrong ``new_dim`` through under a correct ``group_dim``.
        return (grouper, f"{grouper}_bins") if grouper in dim_names else None
    group_dim, _, component = grouper.partition(".")
    if group_dim not in dim_names:
        return None
    return group_dim, (component or group_dim)


def emit(nodes: list[LoweredOp]) -> list[Call]:
    """Generate the call sequence that reproduces a lowered plan.

    Parameters
    ----------
    nodes : list of LoweredOp
        The plan, after lowering and optimisation.

    Returns
    -------
    list of Call
        The calls to perform, in order — what
        ``_replay`` invokes against the dataset.

    Notes
    -----
    The inverse direction of :func:`to_lower_ir`: a node that stands for two calls emits
    two. A pure function of the plan — no dataset in sight — so codegen is unit-testable
    on its own, and ``_replay`` stays the short
    ``getattr`` loop it has always been instead of growing an arm per variant. That is
    what keeps per-call verbatim-ness a property of the whole pipeline rather than
    something each node kind has to be trusted with.
    """
    return [call for node in nodes for call in _emit_node(node)]


def _canon_dims(dims: frozenset[Hashable]) -> Hashable | list[Hashable]:
    """Render a concrete dim set as a reduce/scan ``dim`` argument.

    Parameters
    ----------
    dims : frozenset of Hashable
        The dims the op names — never :data:`~xrexpr.ir.ALL_DIMS` (the caller handles that).

    Returns
    -------
    Hashable or list of Hashable
        A single *string* dim bare (``"time"``), matching xarray's str-is-one-name
        convention and the common spelling; anything else a list, ordered by ``repr`` for a
        deterministic header. A single *non*-string dim is still wrapped in a list, because a
        bare iterable dim name (a tuple) would otherwise re-parse as several dims through
        ``schema._as_dim_set``.
    """
    ordered = sorted(dims, key=repr)
    if len(ordered) == 1 and isinstance(ordered[0], str):
        return ordered[0]
    return ordered


def _dim_arg(name: str) -> int:
    """Where a reduction's dim spec sits among its positionals — 0, or 1 for ``reduce``.

    Parameters
    ----------
    name : str
        The reduction's method name.

    Returns
    -------
    int
        The :attr:`~xrexpr.operations.ReduceSpec.dim_arg` position — 0 for every reduction
        but ``reduce`` (``reduce(func, dim, ...)``), and 0 for a non-tabulated name.
    """
    op = op_spec(name)
    return op.dim_arg if isinstance(op, ReduceSpec) else 0


def _dim_call(
    name: str,
    args: tuple[Any, ...],
    kwargs: frozendict[str, Any],
    dims: DimSet,
    position: int,
) -> Call:
    """Rebuild a reduce/scan call header from its parsed ``dims``, keeping every option.

    The inverse of ``schema._dim_spec``: drop whatever spelled the dim (the positional at
    ``position``, or a ``dim=`` kwarg), then re-emit the dim positionally from ``dims`` — or
    emit no dim at all for :data:`~xrexpr.ir.ALL_DIMS`, the bare ``mean()``/``cumsum()`` case.
    Every other positional (``reduce``'s function, ``diff``'s ``n``) and every option kwarg
    (``skipna``, ``keepdims``, ...) rides along untouched, so a Rust-mutated ``dims`` flows
    into the header while the options that live nowhere else are preserved.

    Parameters
    ----------
    name : str
        The method name.
    args, kwargs : tuple, frozendict
        The recorded call header.
    dims : DimSet
        The op's parsed dims, or :data:`~xrexpr.ir.ALL_DIMS`.
    position : int
        Where the dim sits among the positionals — :func:`_dim_arg` for a reduce, 0 for a scan.

    Returns
    -------
    Call
        The reconstructed call.
    """
    positionals = list(args)
    if "dim" not in kwargs and len(positionals) > position:
        del positionals[position]
    options = {k: v for k, v in kwargs.items() if k != "dim"}
    if not isinstance(dims, AllDims):
        positionals.insert(position, _canon_dims(dims))
    return Call(name=name, args=tuple(positionals), kwargs=frozendict(options))


def _rechunk_args(rechunk: Rechunk) -> tuple[Any, ...]:
    """Rebuild the positional header of a ``chunk`` call from its spec fields.

    Parameters
    ----------
    rechunk : Rechunk
        The node whose ``chunk`` call to rebuild.

    Returns
    -------
    tuple
        The uniform form (``chunk("auto")``) when :attr:`~xrexpr.ir.Rechunk.uniform` is set,
        the mapping form (``chunk({dim: spec})``) when :attr:`~xrexpr.ir.Rechunk.chunks` is,
        and an empty header for a bare ``chunk()`` — the three cases ``schema._chunk_spec``
        parses.
    """
    if rechunk.uniform is not None:
        return (rechunk.uniform.to_raw(),)
    if rechunk.chunks:
        return ({dim: spec.to_raw() for dim, spec in rechunk.chunks.items()},)
    return ()


def _emit_node(node: LoweredOp) -> tuple[Call, ...]:
    """Generate the calls one lowered node stands for.

    Parameters
    ----------
    node : LoweredOp
        The node to emit.

    Returns
    -------
    tuple of Call
        One call for most kinds; two for a fused node, reassembled from the headers it
        fused.

    Notes
    -----
    The header is **derived from the node's semantic fields**, not copied from the recorded
    ``args``/``kwargs`` — the inverse of ``schema.to_opnode``'s parse
    (``planning/roadmap/02-lowering.md`` §7). A select's header is rebuilt from its
    ``indexer`` (via each value's ``to_raw()``), a rechunk's from its ``chunks``/``uniform``,
    a reduce's from its ``consumes``, and so on; the *option* kwargs a call carries
    (``skipna``, ``drop``, ...) — which live nowhere else — are preserved verbatim. This is
    what lets a rule (or a Rust optimiser) rewrite a *semantic field* and have the replayed
    call follow, instead of each rule hand-rebuilding its own header. It also means the
    emitted call is the **canonical** spelling of the selection, not necessarily the recorded
    one: ``isel(time=0)`` re-emits as ``isel({"time": 0})``. The two replay identically —
    equivalence, not byte-identity, is the contract (see ``test_lower.py``).

    Two families keep their recorded header instead. :class:`~xrexpr.ir.Elementwise` and
    :class:`~xrexpr.ir.Opaque` carry no semantic spec — the header *is* the content — and
    neither is ever reordered. And the fused nodes
    (:class:`~xrexpr.ir.GroupedReduce`/:class:`~xrexpr.ir.WindowedReduce`/:class:`~xrexpr.ir.WeightedReduce`)
    reassemble from the two headers they fused: deriving them would have to invent
    ``groupby("time.month")`` back out of ``group_dim``/``new_dim``, which §7 rejects.

    ``assert_never`` closes the match: a new lowered variant fails type-check here until
    someone says what it replays as, which is exactly the question a fused node exists to
    answer. A :class:`~xrexpr.ir.ContextOpen` cannot reach this function at all — it is
    not in :data:`~xrexpr.ir.LoweredOp` — so lowering's central invariant is enforced by
    the type checker rather than by a runtime check here.
    """
    match node:
        case GroupedReduce() as grouped:
            return (
                Call(name=grouped.name, args=grouped.args, kwargs=grouped.kwargs),
                Call(
                    name=grouped.reduce,
                    args=grouped.reduce_args,
                    kwargs=grouped.reduce_kwargs,
                ),
            )
        case WindowedReduce() as windowed:
            return (
                Call(name=windowed.name, args=windowed.args, kwargs=windowed.kwargs),
                Call(
                    name=windowed.reduce,
                    args=windowed.reduce_args,
                    kwargs=windowed.reduce_kwargs,
                ),
            )
        case WeightedReduce() as weighted:
            return (
                Call(name=weighted.name, args=weighted.args, kwargs=weighted.kwargs),
                Call(
                    name=weighted.reduce,
                    args=weighted.reduce_args,
                    kwargs=weighted.reduce_kwargs,
                ),
            )
        case Reduce() as reduce:
            # Looks like args/kwargs are replayed verbatim and could drift out of sync,
            # but _dim_call strips the dim out of the recorded header and rebuilds it from
            # the node's semantic field (``consumes``); only the non-dim payload rides the
            # header through, and that has no semantic twin to disagree with. So this is safe
            # - see https://github.com/charles-turner-1/xrexpr/pull/191 and
            # tests/test_lower.py::test_emit_follows_the_reduce_dim_not_a_stale_header.
            # Leans on the dim sitting at _dim_arg(name): test_operations pins the
            # leading-positional exceptions (quantile, ...) out of OP_TABLE.
            return (
                _dim_call(
                    reduce.name,
                    reduce.args,
                    reduce.kwargs,
                    reduce.consumes,
                    _dim_arg(reduce.name),
                ),
            )
        case Scan() as scan:
            return (_dim_call(scan.name, scan.args, scan.kwargs, scan.dims, 0),)
        case Select() as select:
            indexer = {dim: idx.to_raw() for dim, idx in select.indexer.items()}
            return (
                Call(
                    name=select.name,
                    args=(indexer,) if indexer else (),
                    kwargs=frozendict(
                        {
                            k: v
                            for k, v in select.kwargs.items()
                            if k in SELECT_OPTION_KWARGS
                        }
                    ),
                ),
            )
        case Rechunk() as rechunk:
            return (
                Call(
                    name=rechunk.name,
                    args=_rechunk_args(rechunk),
                    kwargs=frozendict(
                        {
                            k: v
                            for k, v in rechunk.kwargs.items()
                            if k in CHUNK_OPTION_KWARGS
                        }
                    ),
                ),
            )
        case Project() as project:
            key = project.variables[0] if project.single else list(project.variables)
            return (Call(name=project.name, args=(key,)),)
        case Drop() as drop:
            return (
                Call(
                    name=drop.name,
                    args=(list(drop.variables),),
                    kwargs=frozendict(
                        {k: v for k, v in drop.kwargs.items() if k != "names"}
                    ),
                ),
            )
        case Rename() as rename:
            return (Call(name=rename.name, args=(dict(rename.mapping),)),)
        case Elementwise() | Opaque():
            # No semantic spec to derive from — the verbatim header *is* the content, and
            # neither op is ever reordered or restructured, so it replays exactly as recorded.
            return (Call(name=node.name, args=node.args, kwargs=node.kwargs),)
        case _:
            assert_never(node)
