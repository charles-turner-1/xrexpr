"""Lowering: turn what the user *wrote* into what it *means*, and back into calls.

Two pure translation steps sit either side of the optimiser:

- :func:`to_lower_ir` fuses the builder pairs xarray spells as two calls
  (``ds.groupby("time.month").mean()``) into single fused nodes.
- :func:`emit` turns each lowered node back into the calls that reproduce it. That's
  what lets one node stand for two calls without replay needing a special case.

Fusion is a separate stage, not a rewrite rule, for two reasons:

- :func:`~xrexpr.schema.to_opnode` sees one call at a time, so it can't see a pair.
- The optimiser's rules run repeatedly under a termination measure. Fusion runs once,
  and the rules assume it has already happened.

The contract: :func:`to_lower_ir` preserves semantics (``emit(to_lower_ir(p))`` replays
the same as ``p``) and is idempotent, so it's safe to run without checking whether it
already has.

A builder pair no fusion rule handles (``rolling_exp``, ``cumulative``) is demoted to
:class:`~xrexpr.ir.Opaque`, both halves together.

See ``docs/internals/lowering.md``.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr._xrexprs.ir import AllDims
from xrexpr.ir import (
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
#: (the union of both signatures in xarray 2026.7.0).
_WINDOW_OPTION_KWARGS = frozenset(
    {"min_periods", "center", "boundary", "side", "coord_func", "keep_attrs"}
)

#: ``resample`` keyword arguments that are *options*, not the dim being resampled
#: (xarray 2026.7.0's signature). If xarray adds an option and this list misses it, the
#: call ends up with two candidate dim keys and refuses to fuse. That failure is safe.
_RESAMPLE_OPTION_KWARGS = frozenset(
    {"closed", "label", "offset", "origin", "restore_coord_dims", "skipna"}
)


@dataclass(frozen=True)
class Call:
    """One xarray method call: the unit replay actually performs.

    Attributes
    ----------
    name : str
        The method to call, e.g. ``"mean"`` or ``"__getitem__"``.
    args : tuple
        Positional arguments, verbatim.
    kwargs : frozendict
        Keyword arguments, verbatim.

    Notes
    -----
    Deliberately not an :data:`~xrexpr.ir.Op`. A node carries the metadata the optimiser
    reasons about; a ``Call`` is codegen output and carries only what ``getattr(ds,
    name)(*args, **kwargs)`` needs. Keeping them separate is what lets one node emit
    several calls, and keeps replay from needing a branch per node kind.
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
    """Translate a recorded plan into what it means, fusing builder pairs.

    Parameters
    ----------
    nodes : list of FluentOp
        The recorded plan, one node per call as the user wrote it.
    dim_names : frozenset of Hashable
        The base dataset's dim names. This is the one fact lowering can't read off the
        calls. See the notes for why it's the base dims, and why only names.

    Returns
    -------
    list of LoweredOp
        The same plan with every builder pair resolved: fused into one node where a rule
        handles it, or demoted to :class:`~xrexpr.ir.Opaque` where none does. Contains
        no :class:`~xrexpr.ir.ContextOpen`, and the return type enforces that.

    Notes
    -----
    Most nodes pass through unchanged. The lowered IR is the same vocabulary as the
    fluent IR, plus nodes for operations that take two calls to spell. Only those
    multi-call spellings get rewritten.

    **Why ``dim_names``.** Whether ``groupby("region")`` groups along a dim, or along a
    coordinate defined on one, is a fact about the data, not the call (see
    ``_grouper_dims``). Only names are needed, so this takes a set rather than a
    ``SchemaState``. That keeps this module depending on ``xrexpr.ir`` alone.

    **Why the base dims** rather than the dims at each opener (which would mean folding
    the schema through lowering). The shortcut can only lose fusions, never cause wrong
    ones:

    - A dim created mid-plan and then grouped over
      (``groupby("time.month").mean().groupby("month")``) isn't in the base dims, so it
      refuses to fuse and falls back to ``Opaque``.
    - The reverse, a base dim that's gone by the time of the groupby, needs a ``stack``
      or similar in between. That's ``Opaque``, so the plan is already past
      ``optimize._trusted_prefix`` and the schema is a guess anyway.

    **Contexts are pairs, not runs** (checked against xarray 2026.7.0). There's no call
    that goes from one builder to another: ``DatasetGroupBy.__getitem__`` selects a
    group and ends the context, and Rolling/Coarsen/Weighted reject ``__getitem__``. So
    only adjacent pairs are matched.

    **Every** :class:`~xrexpr.ir.ContextOpen` **must go.** One that no rule fuses is
    demoted to :class:`~xrexpr.ir.Opaque` together with its closer, and the pair replays
    verbatim. This fallback is what makes it safe for ``to_opnode`` to type the closer
    provisionally: a closer with an unexpected shape just doesn't fuse, so the worst
    case is a missed optimisation, never a wrong result. Openers with no fused node at
    all (``rolling_exp``, ``cumulative``) always take this path.

    Idempotent because the output contains no ``ContextOpen``, so a second pass has
    nothing to match.
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

        # The fallback. The *closer* must be demoted along with the opener. Left as
        # ``to_opnode`` typed it, it would be a Dataset-level reading of a call that
        # wasn't Dataset-level: ``rolling(time=2).mean()`` would keep a ``Reduce`` whose
        # bare dim spec means "every dim", and a following select would be rejected
        # against dims the rolling mean never removed.
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
        The node recorded right after it.
    dim_names : frozenset of Hashable
        The base dataset's dim names, passed to ``_grouper_dims``. A grouper that
        doesn't name a dim refuses.

    Returns
    -------
    GroupedReduce or None
        The fused node, or ``None`` to refuse (the caller then keeps the pair as opaque
        calls).

    Notes
    -----
    Refusing is always safe, so every condition below just narrows what gets fused. None
    of them is a correctness risk.

    - **The opener must be groupby-family.** ``rolling``/``coarsen`` and ``weighted``
      have their own fused nodes, handled by the functions below.
    - **The grouper must be a string naming a dim**, either a dim name
      (``groupby("lat")``) or a component of one (``groupby("time.month")``).
      ``group_dim`` is read straight off it. A ``DataArray`` grouper or a ``Grouper``
      object has no single ``group_dim``, so it refuses. So does the name of a *non-dim
      coordinate*, which is why ``dim_names`` is passed in (``_grouper_dims``).
    - **The closer must be an aggregating reduce.** This one is easy to get wrong. A
      grouped reduce over dims that *exclude* the group dim isn't an aggregation at all:
      ``ds.groupby("time.month").mean("lat")`` returns dims ``{time, lon}``. It's a
      per-group *map*, reassembled along the original dim, and creates no ``month``.
      Only a bare closer, or one that names the group dim, aggregates, and only that
      case fuses.
    """
    match opener.name:
        # A ``match`` rather than ``in {...}`` so mypy narrows the opener's name to
        # ``GroupedReduce``'s own narrower ``Literal``. A set test would need a
        # ``cast``, which would skip exactly the check we want.
        case "groupby" | "groupby_bins" | "resample":
            kind = opener.name
        case _:
            return None
    if not isinstance(closer, Reduce) or closer.keepdims:
        # ``GroupedReduce`` doesn't model a grouped ``keepdims=True`` reduce:
        #
        # - Before pydata/xarray#11521, it kept the group dim (sized to the number of
        #   groups, with no coordinate) instead of creating ``new_dim``. For example
        #   ``groupby("time.month").mean(keepdims=True)`` kept a coordinate-less
        #   ``time`` and no ``month`` (pydata/xarray#11519).
        # - Since #11521, groupby/resample ignore ``keepdims`` entirely, so it raises.
        #
        # ``GroupedReduce`` (removes the group dim, creates ``new_dim``) describes
        # neither, so refuse and let the pair replay verbatim.
        return None

    grouped = _grouper_dims(opener, dim_names)
    if grouped is None:
        return None
    group_dim, new_dim = grouped

    # A bare closer consumes nothing extra. Unlike a Dataset-level ``mean()``, a grouped
    # one only reduces along the group dim, so ``lat``/``lon`` survive.
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
        The node recorded right after it.

    Returns
    -------
    WindowedReduce or None
        The fused node, or ``None`` to refuse (the caller then keeps the pair as opaque
        calls).

    Notes
    -----
    ``rolling_exp`` and ``cumulative`` aren't handled here. The first is an exponential
    weighting, not a fixed window; the second is really a scan. ``window`` can't
    describe either, so both replay verbatim.

    **A closer's parsed dim is ignored.** ``DatasetRolling.mean`` is ``(keep_attrs=None,
    **kwargs)`` and takes no dim at all. So in ``ds.rolling(time=3).mean("lat")``,
    ``"lat"`` is actually ``keep_attrs``, and nothing is reduced. ``to_opnode`` can't
    know the call is windowed, so it records ``consumes={"lat"}``. The fused node drops
    that wrong ``consumes``: its dim effect depends only on ``window``, and
    ``reduce_args`` replays the closer verbatim.
    """
    match opener.name:
        case "rolling" | "coarsen":
            kind = opener.name
        case _:
            return None
    if not isinstance(closer, Reduce) or closer.keepdims:
        # ``keepdims=True`` refuses. ``coarsen.mean(keepdims=True)`` raises
        # ``TypeError``, and on rolling it does nothing. Neither is a size effect
        # ``WindowedReduce`` describes, so both stay ``Opaque`` and replay verbatim.
        #
        # A closer that *names a dim* still fuses (see the notes). A positional dim
        # (``mean("lat")``) binds to ``keep_attrs``, and a ``dim=`` keyword lands in
        # ``**kwargs`` and is ignored with a warning. Either way no dim is reduced.
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
        The windowed dims and their integer window sizes. Empty when no window is known,
        which the caller treats as a refusal to fuse.

    Notes
    -----
    Both calls take the window either as a positional dict or as dim keywords, mixed
    with option kwargs that aren't dims. That's the same shape ``isel`` and ``chunk``
    use (see ``schema._select_indexer``). A window that isn't a plain ``int`` is
    dropped, so an unrecognised spelling ends up with an empty result and replays
    verbatim rather than being guessed at.
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
        The node recorded right after it.

    Returns
    -------
    WeightedReduce or None
        The fused node, or ``None`` to refuse (the caller then keeps the pair as opaque
        calls).

    Notes
    -----
    Unlike ``_fuse_windowed``, a closer that names dims is fused *with* those dims.
    That's because the signatures differ: ``DatasetWeighted.mean`` is ``(dim=None, *,
    skipna=None, keep_attrs=None)``, so it really does take a dim, while
    ``DatasetRolling.mean`` doesn't. So ``to_opnode``'s parsed dim is correct here and
    is kept as is, :data:`~xrexpr.ir.ALL_DIMS` included.

    The one thing read off the opener is the **weights' dims**, because they have a dim
    effect of their own (see :class:`~xrexpr.ir.WeightedReduce`). Reading ``w.dims`` is
    metadata only and computes nothing, like the reads ``SchemaState.from_dataset``
    already makes. It's duck-typed rather than ``isinstance(w, xr.DataArray)`` so this
    module needs no xarray import. Anything without a ``dims`` tuple refuses. That
    covers weights xarray would reject anyway (``ValueError: `weights` must be a
    DataArray``), so the only cost is a fusion that couldn't have replayed.

    A closer that isn't a tabulated reduce refuses automatically, because it isn't a
    ``Reduce``. That's the right answer for the weighted-only methods:
    ``sum_of_weights``/``sum_of_squares``/``quantile`` aren't in ``OP_TABLE``, and
    ``quantile`` takes ``q`` first, which a reduce's dim parse would misread.
    """
    if opener.name != "weighted" or not isinstance(closer, Reduce) or closer.keepdims:
        # A ``keepdims=True`` closer keeps its dims at size 1. A ``WeightedReduce``
        # removes them, so fusing would misdescribe it. Refuse and replay verbatim.
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
    """Read the dims of the weights off a ``weighted`` call.

    Parameters
    ----------
    opener : ContextOpen
        The ``weighted`` call, with the weights in its verbatim header.

    Returns
    -------
    frozenset of Hashable or None
        The weights' dim names, or ``None`` if the weights aren't something this can
        read (the caller treats that as a refusal to fuse).

    Notes
    -----
    ``Dataset.weighted`` takes exactly one argument, so the weights are ``args[0]`` or
    the ``weights=`` keyword. 0-d weights give the empty set, which is correct: they
    broadcast and align nothing, so the node behaves like a plain reduce.
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
        The base dataset's dim names. Every rule below assumes the grouper names a dim,
        and this is what checks it.

    Returns
    -------
    tuple of Hashable or None
        The dim grouped along and the dim created, or ``None`` if they can't be read off
        the call (the caller treats that as a refusal to fuse).

    Notes
    -----
    Only string groupers fuse. The readings are:

    - ``groupby("time.month")`` groups along ``time`` and creates ``month``;
    - ``groupby("lat")`` creates the dim it grouped over, now holding the distinct
      values;
    - ``groupby_bins("lat", 2)`` creates ``lat_bins`` (xarray's naming);
    - ``resample(time="2D")`` takes its dim from the single keyword and creates it
      again.

    **Each reading assumes the name is a dim, so each is checked against**
    ``dim_names``. The name alone can't tell. If ``region`` is a coordinate *on*
    ``lat``, then ``groupby("region")`` groups along ``lat`` and creates ``region``.
    Reading ``group_dim`` off the call would give ``region``: a dim the grouping never
    consumed, while ``lat``, which it did consume, would be reported as surviving. That
    would mislead both ``dim_effect`` and the schema fold. Refusing sends these pairs to
    the opaque fallback (``02-lowering.md`` §5.5).

    Checked against xarray 2026.7.0. All three kinds share the assumption:

    - ``resample`` accepts a non-dim coordinate too (``resample(t2="2D")`` with ``t2``
      on ``lat`` gives ``{t2: 2, time: 4}``), so the keyword is checked rather than
      trusted.
    - ``groupby_bins("time.month", 2)`` gives ``{lat: 3, month_bins: 2}``: group dim
      ``time``, creating ``month_bins``. The naive reading, ``("time.month",
      "time.month_bins")``, gets both wrong. The dotted bins form is refused rather than
      modelled.
    - A grouper that's a multi-dim coordinate stacks (``cell`` on ``(time, lat)`` gives
      ``{cell: 3}``, consuming both). Modelling non-dim groupers properly would need
      ``group_dim`` to be a *set*.

    A ``Grouper`` object or a keyword grouper (``groupby(x=UniqueGrouper())``) already
    refuses, because ``args`` is empty or not a ``str``.
    """
    if opener.name == "resample":
        keys = [k for k in opener.kwargs if k not in _RESAMPLE_OPTION_KWARGS]
        if len(keys) != 1 or keys[0] not in dim_names:
            return None
        return keys[0], keys[0]

    if not opener.args or not isinstance(grouper := opener.args[0], str):
        return None
    if opener.name == "groupby_bins":
        # Check the whole grouper, not just the part before the dot. For the dotted form
        # the real group dim *is* that head, but the created name isn't
        # ``f"{grouper}_bins"``, so a head-only check would accept a wrong ``new_dim``
        # alongside a correct ``group_dim``.
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
        The calls to perform, in order. This is what ``_replay`` runs against the
        dataset.

    Notes
    -----
    The reverse of :func:`to_lower_ir`: a node that stands for two calls emits two. It's
    a pure function of the plan (no dataset involved), so codegen can be unit-tested on
    its own, and ``_replay`` stays a short ``getattr`` loop instead of needing a branch
    per node kind.
    """
    return [call for node in nodes for call in _emit_node(node)]


def _canon_dims(dims: frozenset[Hashable]) -> Hashable | list[Hashable]:
    """Render a concrete dim set as a reduce/scan ``dim`` argument.

    Parameters
    ----------
    dims : frozenset of Hashable
        The dims the op names. Never :data:`~xrexpr.ir.ALL_DIMS` (the caller handles
        that).

    Returns
    -------
    Hashable or list of Hashable
        A single *string* dim on its own (``"time"``), following xarray's "a str is one
        name" convention. Anything else as a list, sorted by ``repr`` so the header is
        deterministic. A single *non*-string dim still goes in a list, because a bare
        iterable dim name (e.g. a tuple) would otherwise be re-parsed as several dims by
        ``schema._as_dim_set``.
    """
    ordered = sorted(dims, key=repr)
    if len(ordered) == 1 and isinstance(ordered[0], str):
        return ordered[0]
    return ordered


def _dim_arg(name: str) -> int:
    """Where a reduction's dim sits among its positional args: 0, or 1 for ``reduce``.

    Parameters
    ----------
    name : str
        The reduction's method name.

    Returns
    -------
    int
        The :attr:`~xrexpr.operations.ReduceSpec.dim_arg` position. 0 for every
        reduction except ``reduce`` (``reduce(func, dim, ...)``), and 0 for an
        untabulated name.
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
    """Rebuild a reduce/scan call from its parsed ``dims``, keeping every other argument.

    The reverse of ``schema._dim_spec``. Remove whatever spelled the dim (the positional
    at ``position``, or a ``dim=`` kwarg), then put the dim back positionally from
    ``dims``. For :data:`~xrexpr.ir.ALL_DIMS` (the bare ``mean()``/``cumsum()`` case) no
    dim is emitted. Every other positional (``reduce``'s function, ``diff``'s ``n``) and
    option kwarg (``skipna``, ``keepdims``, ...) is kept as is. So a rewritten ``dims``
    (from a rule or a Rust backend) ends up in the call, and the options, which live
    nowhere else, are kept.

    Parameters
    ----------
    name : str
        The method name.
    args, kwargs : tuple, frozendict
        The recorded call header.
    dims : DimSet
        The op's parsed dims, or :data:`~xrexpr.ir.ALL_DIMS`.
    position : int
        Where the dim sits among the positionals: :func:`_dim_arg` for a reduce, 0 for a
        scan.

    Returns
    -------
    Call
        The rebuilt call.
    """
    positionals = list(args)
    if "dim" not in kwargs and len(positionals) > position:
        del positionals[position]
    options = {k: v for k, v in kwargs.items() if k != "dim"}
    if not isinstance(dims, AllDims):
        positionals.insert(position, _canon_dims(dims))
    return Call(name=name, args=tuple(positionals), kwargs=frozendict(options))


def _rechunk_args(rechunk: Rechunk) -> tuple[Any, ...]:
    """Rebuild the positional arguments of a ``chunk`` call from the node's spec fields.

    Parameters
    ----------
    rechunk : Rechunk
        The node whose ``chunk`` call to rebuild.

    Returns
    -------
    tuple
        One of the three forms ``schema._chunk_spec`` parses: the uniform form
        (``chunk("auto")``) when :attr:`~xrexpr.ir.Rechunk.uniform` is set, the per-dim
        form (``chunk({dim: spec})``) when :attr:`~xrexpr.ir.Rechunk.chunks` is, or
        empty for a bare ``chunk()``.
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
        One call for most kinds; two for a fused node, rebuilt from the two calls it
        fused.

    Notes
    -----
    **The call is rebuilt from the node's semantic fields**, not copied from the
    recorded ``args``/``kwargs``. This is the reverse of ``schema.to_opnode``'s parse
    (``planning/roadmap/02-lowering.md`` §7):

    - a select's call comes from its ``indexer`` (each value's ``to_raw()``);
    - a rechunk's from ``chunks``/``uniform``;
    - a reduce's from ``consumes``; and so on.

    Option kwargs (``skipna``, ``drop``, ...) live nowhere else, so they're copied
    verbatim. This is what lets a rule (or a Rust optimiser) change a semantic field and
    have the replayed call follow, without each rule rebuilding calls itself.

    It also means the emitted call is the *canonical* spelling, which may differ from
    what was recorded: ``isel(time=0)`` comes back as ``isel({"time": 0})``. They replay
    the same. The contract is equivalence, not byte-identical calls (see
    ``test_lower.py``).

    Two kinds keep their recorded call instead:

    - :class:`~xrexpr.ir.Elementwise` and :class:`~xrexpr.ir.Opaque` have no semantic
      fields (the call *is* the content), and neither is ever reordered.
    - The fused nodes (:class:`~xrexpr.ir.GroupedReduce`,
      :class:`~xrexpr.ir.WindowedReduce`, :class:`~xrexpr.ir.WeightedReduce`) replay the
      two calls they fused. Rebuilding them would mean inventing
      ``groupby("time.month")`` from ``group_dim``/``new_dim``.

    The ``match`` is closed with ``assert_never``, so a new lowered kind fails
    type-checking here until someone says how it replays. A
    :class:`~xrexpr.ir.ContextOpen` can't reach this function: it isn't in
    :data:`~xrexpr.ir.LoweredOp`, so the type checker enforces that lowering removed it.
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
            # but _dim_call strips the dim out of the recorded header and rebuilds it
            # from the node's semantic field (``consumes``); only the non-dim payload
            # rides the header through, and that has no semantic twin to disagree with.
            # So this is safe
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
            # No semantic fields to rebuild from: the recorded call *is* the content,
            # and neither kind is ever reordered, so it replays exactly as recorded.
            return (Call(name=node.name, args=node.args, kwargs=node.kwargs),)
        case _:
            assert_never(node)
