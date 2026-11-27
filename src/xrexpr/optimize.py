"""The plan optimiser: rewrite a lowered plan into a cheaper, equivalent one.

:func:`optimize` runs the rewrite rules in ``_RULES`` until none of them changes the
plan (a fixpoint). Each rule is local: it looks at one or two adjacent nodes, rewrites
them, and returns ``None`` when it has nothing to do. Big rewrites come from small ones
repeating. For example, a select reaches the front of a plan by hopping left one node at
a time, and no rule ever reasons about the whole chain.

Two rules for anyone adding a rule:

- **Termination.** The loop terminates because every rule decreases
  ``(len(plan), sum of the indices of the Select and Project nodes)``. So a rule may
  shorten the plan or move a select/projection left, but it must never lengthen the plan
  or move a node right. A rule that needs to do either needs a new termination argument.
- **One place for dim facts.** What each node kind does to dims is answered in one
  ``match``, :func:`dim_effect`, closed with ``assert_never``. Rules read that rather
  than matching node kinds themselves, so a new node kind has to state its answer there
  (even if the answer is "nothing may cross me").

The schema fold (tracking each node's input schema) lives here rather than in recording,
because only the optimiser knows how far it can be trusted.
:func:`~xrexpr.schema.apply_schema` assumes an :class:`~xrexpr.ir.Opaque` keeps every
variable, which is false for ops like ``assign``. So rules that read ``data_vars`` stop
at the first ``Opaque`` (``_trusted_prefix``). Rules that only reason about dims don't
need that limit: an opaque node blocks them outright, so it costs rewrites but never
correctness.

See ``docs/internals/optimiser.md``.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard

import xarray as xr
from frozendict import frozendict
from packaging.version import Version
from typing_extensions import assert_never

from xrexpr._xrexprs.ir import ALL_DIMS, AllDims
from xrexpr.chunks import (
    Auto,
    BlockSeq,
    ByteSize,
    ChunkSpec,
    FullDim,
    NoChange,
    OpaqueChunk,
    SingleSize,
)
from xrexpr.exceptions import InvalidExpressionError
from xrexpr.indexers import (
    Advanced,
    ForwardSlice,
    GeneralSlice,
    Indexer,
    Label,
    Mask,
    Positions,
    Scalar,
)
from xrexpr.ir import (
    DimSet,
    Drop,
    Elementwise,
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
from xrexpr.schema import SchemaState, apply_schema, resolve_dims

__all__ = ["optimize"]

#: ``xarray`` 2026.09.0 works around dask's divide-by-zero when a rechunk containing
#: ``"auto"`` sees an empty dimension (pydata/xarray#11486). On older versions a select
#: must not be moved in front of such a rechunk.
AUTOCHUNK_EMPTY_DIMS_SAFE = Version(xr.__version__) >= Version("2026.9.0")

#: What the rules rewrite: a **lowered** plan. Typed as :data:`~xrexpr.ir.LoweredOp`
#: rather than :data:`~xrexpr.ir.Op` so that a node lowering should have fused away (a
#: ``ContextOpen``) is a type error here.
Plan = list[LoweredOp]
#: A rule takes a plan and the schema its first node sees, and returns the rewritten
#: plan, or ``None`` if it changed nothing. ``None`` lets :func:`optimize` spot the
#: fixpoint without comparing whole plans. Rules that only reason about dims ignore the
#: schema.
Rule = Callable[[Plan, SchemaState[Hashable]], Plan | None]


def optimize(nodes: Plan, schema: SchemaState[Hashable]) -> Plan:
    """Rewrite ``nodes`` into an equivalent plan, applying every rule until none fires.

    Parameters
    ----------
    nodes : Plan
        The lowered plan to rewrite.
    schema : SchemaState[Hashable]
        The schema of the dataset the plan starts from. This is the *base* schema, not
        the one at the end of the plan: rules need the schema each node sees, and
        rewriting changes that.

    Returns
    -------
    Plan
        An equivalent plan. It replays to the same result as ``nodes``, more cheaply.

    Raises
    ------
    InvalidExpressionError
        If a rule proves the plan can never replay, e.g. a select on a dim that a
        preceding reduce removed. See :func:`pushdown_selects`.

    Notes
    -----
    Every rule preserves the plan's result, so the whole loop does too.

    The loop terminates because every rule strictly decreases the measure
    ``(len(plan), sum of the indices of the Select and Project nodes)``:

    - merges and eliminations shorten the plan;
    - pushdowns keep the length but move a select or projection one step left.

    Neither part of the measure can grow, so a new rule must never move a node right or
    lengthen the plan.
    """
    plan = list(nodes)
    while True:
        changed = False
        for rule in _RULES:
            rewritten = rule(plan, schema)
            if rewritten is not None:
                plan, changed = rewritten, True
        if not changed:
            return plan


def _schemas(nodes: Plan, base: SchemaState[Hashable]) -> list[SchemaState[Hashable]]:
    """Fold the base schema forward through a plan.

    Parameters
    ----------
    nodes : Plan
        The plan to fold over.
    base : SchemaState[Hashable]
        The schema of the dataset the plan starts from.

    Returns
    -------
    list of SchemaState
        The schema each node *sees*: ``out[i]`` is the schema going into ``nodes[i]``.
    """
    if not nodes:
        return []
    out = [base]
    for node in nodes[:-1]:
        out.append(apply_schema(out[-1], node))
    return out


def _trusted_prefix(nodes: Plan) -> int:
    """Find how far into the plan the folded schema is exact.

    Parameters
    ----------
    nodes : Plan
        The plan to scan.

    Returns
    -------
    int
        The index of the first :class:`~xrexpr.ir.Opaque` node, or the plan's length if
        there is none.

    Notes
    -----
    :class:`~xrexpr.ir.Opaque` covers anything the IR doesn't model, including ops that
    add, rename or drop variables. After the first one, ``data_vars`` is a guess. Any
    rule that reasons about variables must only look at
    ``nodes[:_trusted_prefix(nodes)]``.
    """
    for i, node in enumerate(nodes):
        if isinstance(node, Opaque):
            return i
    return len(nodes)


@dataclass(frozen=True)
class DimEffect:
    """What one node does to dims, in the terms the rules need.

    Attributes
    ----------
    blocks : DimSet or None
        Dims a select coming from the right may not hop over. ``None`` means no select
        may hop at all.
    requires : DimSet or None
        Dims the node needs from its input. A projection may only hop in front of the
        node if the kept variables still carry these. ``None`` means no projection may
        hop.
    on_conflict : {"immovable", "invalid"}
        What it means when a select names one of the :attr:`blocks` dims:

        - ``"invalid"``: the node *removed* those dims, so the chain can never replay,
          and the optimiser raises.
        - ``"immovable"``: the dims survive in some form, so the chain is probably fine
          and just can't be reordered. Leave it.

        Only meaningful when ``blocks`` is not ``None``.

    Notes
    -----
    Never stored on a node: :func:`dim_effect` computes it on demand. This follows the
    same "derive, don't store" rule as ``Select.consumes`` and ``Project.single``.

    There are two fields because the two pushdown rules approach a node from opposite
    sides:

    - :attr:`blocks` is read when moving a **select** left, from *after* the node. The
      select must avoid every dim the node consumed, minted or resized. That includes a
      grouped reduce's newly minted dim, because a select after the reduce may name it.
    - :attr:`requires` is read when moving a **projection** to *before* the node. It's
      what the node needs from its input, so it excludes minted dims, which don't exist
      yet at that point.

    ``None`` means *don't know*, and must be read as "no rewrite", never as "no dims".
    """

    blocks: DimSet | None
    requires: DimSet | None
    on_conflict: Literal["invalid", "immovable"] = "immovable"


#: Nothing crosses this node in either direction. Used for ``Opaque`` (could do
#: anything), ``Rechunk`` (has its own rule) and ``Project`` (handled by the projection
#: rules).
_OPAQUE_EFFECT = DimEffect(blocks=None, requires=None)


def dim_effect(node: LoweredOp) -> DimEffect:
    """Compute what ``node`` does to dims. The single place this is decided.

    Parameters
    ----------
    node : LoweredOp
        The node a rule wants to move something across.

    Returns
    -------
    DimEffect
        What a select coming from the right must avoid, and what a projection moving to
        the left must supply. Kinds the optimiser doesn't model answer ``None`` to both.

    Notes
    -----
    All the rules get their dim facts from here, and the ``match`` is closed with
    ``assert_never``. So when a node kind is added, mypy forces a decision here about
    both questions at once, instead of each rule silently falling back to "leave it".

    Most kinds that the optimiser doesn't reason about answer ``None`` to both fields,
    which blocks every hop. That's always safe; it just costs optimisation.
    """
    match node:
        case Reduce(consumes=consumes) as reduce if reduce.keepdims:
            # ``keepdims=True`` removes nothing: every named dim stays, at size 1. A
            # later select on one is valid but must not move in front (slicing a size-1
            # dim then filling is not the same as filling then slicing). Same shape as
            # ``WindowedReduce``: the dims are both blocked and required, and the
            # default ``immovable`` leaves the chain alone rather than rejecting it.
            return DimEffect(blocks=consumes, requires=consumes)
        case Reduce(consumes=consumes):
            # A reduce removes exactly the dims it names, so a later select on one of
            # them can never replay. This is the one case where the optimiser rejects
            # the chain itself.
            return DimEffect(blocks=consumes, requires=consumes, on_conflict="invalid")
        case Select(indexer=indexer):
            # A select never hops another select (``merge_adjacent_selects`` merges them
            # instead). A projection can cross a select, and needs the select's indexed
            # dims to still exist.
            return DimEffect(blocks=None, requires=frozenset(indexer))
        case GroupedReduce(group_dim=group_dim, new_dim=new_dim, consumes=consumes):
            return DimEffect(
                blocks=frozenset({group_dim, new_dim}) | consumes,
                requires=frozenset({group_dim}) | consumes,
            )
        case WindowedReduce(window=window):
            # A window consumes and mints nothing. It can only resize the dims it names,
            # so those dims are both the blocked set and the required set.
            return DimEffect(blocks=frozenset(window), requires=frozenset(window))
        case WeightedReduce(consumes=consumes, weight_dims=weight_dims):
            # ``blocks=None``: a select never crosses a weighted reduce, because the
            # weights would have to be subset too. That would be a rewrite that changes
            # array data, and it hasn't been done (``planning/roadmap/02-lowering.md``
            # §8.1).
            #
            # ``requires`` lets projections cross, and the ``weight_dims`` term is what
            # keeps that safe. A projection drops any dim *coordinate* that no kept
            # variable uses. The weights behave differently depending on whether that
            # coordinate is present: if it is, they're aligned (inner-joined) against
            # it; if not, they're broadcast as a fresh dim. Checked against xarray
            # 2026.7.0: with weights on an uncoordinated ``lat``,
            # ``ds.weighted(w).mean("time")[["ts"]]`` keeps ``lat``'s coord, but the
            # hopped form has no ``lat`` coord at all. With weights on a *disjoint*
            # ``lat`` the two give different values. Requiring the kept variables to
            # carry every weight dim keeps the coordinate alive and avoids both
            # problems.
            match consumes:
                case AllDims():
                    # A bare closer removes every dim, minted weight dims included, so
                    # there's nothing left for a coordinate to be missing from.
                    # ``ALL_DIMS`` alone is enough. It has to be: a weight dim the
                    # dataset lacks could never be a subset of the kept variables' dims.
                    return DimEffect(blocks=None, requires=ALL_DIMS)
                case frozenset() as named:
                    return DimEffect(blocks=None, requires=named | weight_dims)
                case _:
                    assert_never(consumes)
        case Scan(dims=dims):
            # A scan keeps its dims but order matters along them.
            #
            # - ``blocks=dims``: a select on a *different* dim commutes with the scan
            #   (the scan runs independently at each position of the other dims). A
            #   select on a scanned dim is left where it is, not rejected, because the
            #   dim survives.
            # - ``requires=dims``: a projection may go before the scan if the kept
            #   variables still carry the scanned dims. Same condition as for a reduce
            #   (``07-small-wins.md`` §3).
            #
            # (A *prefix* forward slice on the scanned dim would also commute with
            # ``cumsum`` and ``cumprod``, though not ``diff``. Not implemented.)
            return DimEffect(blocks=dims, requires=dims)
        case Elementwise():
            # A per-element op keeps every dim, so any select commutes with it (empty
            # ``blocks``) and a projection needs nothing from it (empty ``requires``).
            # This is different from the ``None``/``None`` barrier answer. The arguments
            # that would break commutation (an array, or a per-variable dict) never get
            # here: ``schema._elementwise_safe`` records those calls as ``Opaque``.
            # ``apply_schema`` is exact for this node, so the trusted prefix spans it.
            return DimEffect(blocks=frozenset(), requires=frozenset())
        case Rename(mapping=mapping):
            # A rename changes dim *names* mid-plan, and the pushdown rules don't
            # translate names across it.
            #
            # - ``blocks`` is the set of new names. A select names dims by their current
            #   name, so a select on a renamed dim is blocked (``immovable``: the dim
            #   survives, so leave the chain alone). A select on an untouched dim
            #   passes. Renamed variables in the set are harmless, since a select never
            #   names a variable.
            # - ``requires=None``: no projection crosses via the generic rule. The
            #   danger is a renamed *variable*: a projection naming the new name can't
            #   go before the rename that creates it, and the dim-based check can't see
            #   that.
            #
            # A schema-aware rename/projection rule, and translating a select's dim
            # names across a rename (``11-relabel-rename-drop.md`` §4), are not written.
            # The schema stays exact across the node either way, so the trusted prefix
            # continues past it.
            return DimEffect(blocks=frozenset(mapping.values()), requires=None)
        case Drop():
            # ``drop_vars`` changes no dim, so any select commutes with it (empty
            # ``blocks``).
            #
            # A projection may not cross it blindly (``requires=None``). Moved in front,
            # the projection could remove the very name ``drop_vars`` targets, turning a
            # valid chain into a ``KeyError``. The safe, schema-aware version of that
            # hop is :func:`push_projection_past_drop`. The schema stays exact across
            # the node, so the trusted prefix continues past it.
            return DimEffect(blocks=frozenset(), requires=None)
        case Project() | Rechunk() | Opaque():
            return _OPAQUE_EFFECT
        case _:
            assert_never(node)


def merge_adjacent_selects(nodes: Plan, schema: SchemaState[Hashable]) -> Plan | None:
    """Merge each run of consecutive selects of the same kind into one node.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema entering the plan. Unused: this rule only reasons about dims.

    Returns
    -------
    Plan or None
        The plan with every mergeable run merged, or ``None`` if nothing was merged.

    Notes
    -----
    ``ds.isel(time=0).isel(lat=1)`` becomes ``isel({time: 0, lat: 1})``. Selects on
    different dims just combine. Selects on the *same* dim have to be composed (see
    ``_compose_into``), because the second one indexes into the first one's result, not
    into the original dim.

    A run ends, leaving two correct nodes rather than one wrong one, at:

    - a switch between ``isel`` and ``sel`` (different indexing semantics);
    - a select with option kwargs (``drop``, ``method``, ...), which a plain merged
      indexer dict can't carry;
    - a same-dim pair whose composition can't be proven without knowing the dim's
      length.
    """
    out: Plan = []
    folded = False
    i, n = 0, len(nodes)
    while i < n:
        node = nodes[i]
        if not _mergeable_select(node):
            out.append(node)
            i += 1
            continue

        j = i + 1
        indexer = dict(node.indexer)
        # Only ``indexer`` is accumulated. ``consumes`` is derived from it, and
        # ``lower._emit_node`` rebuilds the call from it, so neither can disagree with
        # the merged indexer.
        while j < n:
            nxt = nodes[j]
            if nxt.name != node.name or not _mergeable_select(nxt):
                break
            merged = _compose_into(node.name, indexer, nxt.indexer)
            if merged is None:  # a dim we can't compose — end the run here
                break
            indexer = merged
            j += 1

        if j - i > 1:  # at least two selects folded
            folded = True
            out.append(Select(name=node.name, indexer=frozendict(indexer)))
        else:
            out.append(node)
        i = j
    return out if folded else None


def _mergeable_select(node: LoweredOp) -> TypeGuard[Select]:
    """Report whether ``node`` is a select fully described by its ``indexer``.

    Parameters
    ----------
    node : LoweredOp
        The candidate node.

    Returns
    -------
    bool
        ``True`` for a :class:`~xrexpr.ir.Select` with no option kwargs. A
        ``TypeGuard``, so callers can treat ``node`` as a ``Select``.

    Notes
    -----
    A select whose kwargs include anything other than indexed dims (``drop``,
    ``method``, ``missing_dims``, ``tolerance``) can't be folded into a plain indexer
    dict. It ends the merge run, rather than having its options silently dropped.
    """
    return isinstance(node, Select) and all(k in node.indexer for k in node.kwargs)


def _compose_into(
    name: str,
    outer: dict[Hashable, Indexer],
    inner: Mapping[Hashable, Indexer],
) -> dict[Hashable, Indexer] | None:
    """Merge ``inner``'s indexers into ``outer``'s, composing any dim both index.

    Parameters
    ----------
    name : str
        The select kind, ``"isel"`` or ``"sel"``. Only ``isel`` composes: composing
        ``sel`` needs coordinate values.
    outer : dict
        The indexers applied first, accumulated so far.
    inner : Mapping
        The indexers applied second. They index positions *within* ``outer``'s result.

    Returns
    -------
    dict or None
        The merged mapping, or ``None`` if the two selects must not be merged.

    Notes
    -----
    A dim indexed by only one side carries over unchanged.

    A dim indexed by both is why this isn't a plain ``dict.update``.
    ``isel(time=slice(100, 1000)).isel(time=slice(10, 20))`` is *not*
    ``isel(time=slice(10, 20))``: the second slice indexes into the first's result, so
    they compose to ``slice(110, 120)``. ``_compose_indexer`` handles the cases it can
    prove.

    ``None`` from either function means "don't merge these two selects", and the caller
    ends the run. It's all or nothing: if one dim can't be composed, the whole merge is
    abandoned, because a half-applied ``inner`` would match neither select.
    """
    if name != "isel":  # ``sel`` composition needs coordinate values; positions only
        return None if set(outer) & set(inner) else {**outer, **inner}

    merged = dict(outer)
    for dim, index in inner.items():
        if dim not in merged:
            merged[dim] = index
            continue
        composed = _compose_indexer(merged[dim], index)
        if composed is None:  # no statically provable composition — end the run
            return None
        merged[dim] = composed
    return merged


def _compose_indexer(outer: Indexer, inner: Indexer) -> Indexer | None:
    """Compose two positional indexers on the same dim: ``outer`` first, then ``inner``.

    Parameters
    ----------
    outer : Indexer
        The indexer applied first, to the original dim.
    inner : Indexer
        The indexer applied second, to ``outer``'s result.

    Returns
    -------
    Indexer or None
        The single equivalent indexer, or ``None`` if the composition can't be proven
        without the dim's length (which the optimiser doesn't track).

    Notes
    -----
    Composition works when ``outer``'s selected positions are known *without* the dim's
    length. Three variants qualify:

    - :class:`~xrexpr.indexers.Positions`: already a list of positions, so composing is
      just indexing into it. ``[10, 20, 30]`` then ``1`` is ``20``.
    - :class:`~xrexpr.indexers.Mask`: the same thing in another form. The kept positions
      are the indices of the ``True`` flags.
    - :class:`~xrexpr.indexers.ForwardSlice`: arithmetic on the bounds.
      ``slice(100, 1000)`` then ``slice(10, 20)`` is ``slice(110, 120)``.

    The rest return ``None``, each for its own reason:

    - ``GeneralSlice``: negative or reversed bounds count from an end we can't locate.
    - ``Label``: not positional at all.
    - ``Scalar``: drops the dim, so there's nothing left for ``inner`` to index.

    These are listed explicitly and the ``match`` ends in ``assert_never``. This
    function is the policy for which compositions the optimiser will attempt, so a new
    ``Indexer`` variant should fail type-checking here until someone decides where it
    belongs.
    """
    match outer:
        case Positions(values=values):
            return _index_sequence(values, inner)
        case Mask(values=flags):
            # A mask is a list of positions in another form: it keeps ``[k for k, flag
            # in enumerate(flags) if flag]``, which needs no dim length. So it composes
            # the same way ``Positions`` does.
            return _index_sequence(
                tuple(k for k, flag in enumerate(flags) if flag), inner
            )
        case ForwardSlice():
            return _compose_slice(outer, inner)
        case Scalar() | GeneralSlice() | Label():
            return None
        case Advanced():
            # Never reached in practice: a select with an advanced indexer is recorded
            # as ``Opaque``, so two of them are never composed. Answered for
            # exhaustiveness.
            return None
        case _:
            assert_never(outer)


def _index_sequence(positions: tuple[int, ...], inner: Indexer) -> Indexer | None:
    """Apply ``inner`` to a concrete tuple of positions, i.e. ``positions[inner]``.

    Parameters
    ----------
    positions : tuple of int
        The positions ``outer`` selected, fully listed.
    inner : Indexer
        The indexer applied to that list.

    Returns
    -------
    Indexer or None
        The composed indexer, or ``None`` if the composition is refused.

    Notes
    -----
    Exact, because the outer selection is already a concrete list. If ``inner`` is out
    of range this returns ``None`` instead of raising, so the error comes from xarray at
    replay, in its own words.
    """
    seq = list(positions)
    try:
        match inner:
            case Scalar() as s if s.position is not None:
                return Scalar(seq[s.position])
            case ForwardSlice() | GeneralSlice() as s:
                # ``seq`` is already concrete, so a reversed/negative slice is fine
                # here.
                return Positions(tuple(seq[s.to_raw()]))
            case Positions(values=idx):
                return Positions(tuple(seq[i] for i in idx))
            case Mask(values=flags):
                # Both sides are concrete, so this is exact too. A wrong-length mask
                # isn't ours to diagnose: xarray requires a boolean mask to match the
                # dim's length and will raise at replay. Refuse and let it.
                if len(flags) != len(seq):
                    return None
                return Positions(tuple(p for p, keep in zip(seq, flags) if keep))
            case _:
                return None
    except IndexError:
        return None


def _compose_slice(outer: ForwardSlice, inner: Indexer) -> Indexer | None:
    """Compose a forward, non-negative ``outer`` slice with ``inner``.

    Parameters
    ----------
    outer : ForwardSlice
        The slice applied first. Forward and non-negative by construction.
    inner : Indexer
        The indexer applied to its result.

    Returns
    -------
    Indexer or None
        The composed indexer, or ``None`` if the composition is refused.

    Notes
    -----
    Element ``k`` of the result is ``outer_start + (inner_start + k * inner_step) *
    outer_step``. That's an arithmetic progression, so slice-then-slice is a slice and
    slice-then-scalar is a scalar.

    The composed stop is the tighter of two limits: ``inner`` can't run past its own
    stop, or past the end of what ``outer`` produced.
    """
    start, step = outer.start or 0, outer.step or 1

    match inner:
        case Scalar() as s if s.position is not None and s.position >= 0:
            position = start + s.position * step
            # Out of ``outer``'s range: the original chain raises, but the composed
            # position might be valid in the full dim and would silently return data
            # instead.
            if outer.stop is not None and position >= outer.stop:
                return None
            return Scalar(position)

        case ForwardSlice(start=inner_start, stop=inner_stop, step=inner_step):
            inner_start, inner_step = inner_start or 0, inner_step or 1
            candidates = (outer.stop, _scaled_stop(inner_stop, start, step))
            stops = [s for s in candidates if s is not None]
            return ForwardSlice(
                start + inner_start * step,
                min(stops) if stops else None,
                step * inner_step,
            )

        case Positions(values=idx):
            # ``outer`` is an arithmetic progression, so its element ``k`` is ``start +
            # k * step``, the same arithmetic as the ``Scalar`` arm. A *negative* entry
            # counts back from the end of ``outer``'s result, which needs a length we
            # don't have, so it refuses the whole merge.
            if any(k < 0 for k in idx):
                return None
            return _mapped_positions(outer, start, step, addressed=idx, kept=idx)

        case Mask(values=flags):
            # This works because xarray requires a boolean mask to be exactly as long as
            # the dim it indexes. So ``len(flags)`` *is* the number of elements
            # ``outer`` produced, which is the length we otherwise don't know. That lets
            # us compute the positions without the dim size, which ``GeneralSlice``
            # can't offer.
            return _mapped_positions(
                outer,
                start,
                step,
                addressed=range(len(flags)),
                kept=[k for k, keep in enumerate(flags) if keep],
            )

        case _:
            return None


def _mapped_positions(
    outer: ForwardSlice,
    start: int,
    step: int,
    addressed: Iterable[int],
    kept: Iterable[int],
) -> Positions | None:
    """Map ``inner``'s element indices onto ``outer``'s positions, or refuse.

    Parameters
    ----------
    outer : ForwardSlice
        The slice applied first. Its ``stop`` is the bound that matters here.
    start : int
        ``outer``'s start, defaulted to 0.
    step : int
        ``outer``'s step, defaulted to 1.
    addressed : iterable of int
        Every element of ``outer`` that ``inner`` reads.
    kept : iterable of int
        The subset ``inner`` actually keeps. Only differs from ``addressed`` for a mask.

    Returns
    -------
    Positions or None
        The mapped positions, or ``None`` if any addressed element is out of ``outer``'s
        range.

    Notes
    -----
    The range check is over ``addressed``, not ``kept``. Same trap as the ``Scalar``
    arm: reading past ``outer``'s stop makes the original chain raise, but the composed
    position may be valid in the full dim and would quietly return data. Reading is what
    raises, so every element read has to be in range, kept or not.
    """
    if outer.stop is not None and any(
        start + k * step >= outer.stop for k in addressed
    ):
        return None
    return Positions(tuple(start + k * step for k in kept))


def _scaled_stop(inner_stop: int | None, start: int, step: int) -> int | None:
    """Express ``inner``'s stop as a position in the *original* dim.

    Parameters
    ----------
    inner_stop : int or None
        ``inner``'s stop bound, in ``outer``'s coordinates.
    start : int
        ``outer``'s start, defaulted to 0.
    step : int
        ``outer``'s step, defaulted to 1.

    Returns
    -------
    int or None
        The equivalent position in the original dim, or ``None`` for an unbounded stop.
    """
    return None if inner_stop is None else start + inner_stop * step


def merge_adjacent_projects(nodes: Plan, schema: SchemaState[Hashable]) -> Plan | None:
    """Drop the first of two adjacent projections when the second one covers it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema entering the plan. Unused: this rule decides from the two nodes' key
        sets alone (see the notes for why that's enough, and what it costs).

    Returns
    -------
    Plan or None
        The plan with one projection dropped, or ``None`` if no pair merges.

    Notes
    -----
    ``ds[["tas", "pr"]][["tas"]]`` builds a Dataset holding ``pr`` only to throw it
    away. When ``set(p2.variables) <= set(p1.variables)``, the pair collapses to ``p2``
    alone. The second node is kept as is: its key already describes the result.

    Coordinates work out too. A projection keeps the coordinates its variables still
    span. Because ``p2``'s variables are a subset of ``p1``'s, every coordinate ``p2``
    would keep on its own already survived ``p1``. Checked against xarray 2026.7.0:
    ``ds[["temperature", "elevation"]][["temperature"]]`` is ``identical`` to
    ``ds[["temperature"]]``, auxiliary coordinates included.

    Two guards:

    - **The subset test.** ``p2`` may name a *coordinate*:
      ``ds[["temperature"]]["lat"]`` works, because projection keeps coords. But
      ``ds["lat"]`` is a different expression. It reads the coord from the base, where
      the chain reads it from ``p1``'s result. The two only agree by accident. The
      subset test is exactly the case where the collapse is provable without a schema,
      so when it fails the rule leaves the plan alone (it never raises).
    - **``p1`` must not be** ``single``. After ``ds["temperature"]`` the object is a
      ``DataArray``, where ``__getitem__`` means indexing, not projection.
      ``ds["temperature"]["temperature"]`` raises ``KeyError``, but the collapsed
      ``ds["temperature"]`` would return data. Turning an error into a value is
      forbidden (``planning/roadmap/07-small-wins.md`` §8), and the subset test alone
      wouldn't catch this.

    Deciding without the schema does skip one kind of error, on purpose. If ``p1`` names
    a variable the dataset lacks, the eager chain raises ``KeyError`` while the merged
    plan succeeds: ``ds.plan[["temperature", "nope"]][["temperature"]]``. That's allowed
    by the same §8 rule :func:`pushdown_projections` relies on: the error came from work
    whose result was discarded, and no returned value changes. Not using the schema also
    lets the rule handle a ``p1`` that names a coordinate (``ds[["lat",
    "temperature"]][["lat"]]``), which a check against ``data_vars`` couldn't.

    The rule would also fire past an :class:`~xrexpr.ir.Opaque`, but recording never
    produces such a pair. An ``Opaque`` may return a DataArray (``ds.plan.pipe(lambda d:
    d["tas"])``), where a list key indexes rows. So ``accessor._getitem_is_projection``
    records every ``__getitem__`` after an ``Opaque`` as ``Opaque`` too. Without that,
    ``[[1, 2]][[1]]`` would pass the subset test and collapse to the wrong row.

    One merge per call; the fixpoint handles runs of three or more. The rule shortens
    the plan, so termination holds.
    """
    for i in range(len(nodes) - 1):
        match nodes[i], nodes[i + 1]:
            case (Project() as first, Project() as second) if not first.single and set(
                second.variables
            ) <= set(first.variables):
                return list(nodes[:i]) + list(nodes[i + 1 :])
    return None


def eliminate_projection_before_coord(
    nodes: Plan, schema: SchemaState[Hashable]
) -> Plan | None:
    """Drop a projection when the next one only reads coordinates that survive it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema of the dataset the plan starts from. Needed: whether ``p2``'s
        coordinate survives ``p1`` is something only the folded schema knows.

    Returns
    -------
    Plan or None
        The plan with the first projection of a qualifying pair dropped, or ``None`` if
        no pair qualifies.

    Notes
    -----
    ``ds[["temperature"]]["lat"]`` builds ``temperature`` (``p1``), then throws it away
    and returns the ``lat`` coordinate (``p2``). The best plan reads ``lat`` from the
    base and never touches ``temperature``.

    :func:`merge_adjacent_projects` can't do this, and shouldn't: its subset test fails,
    and ``ds["lat"]`` is a different expression from the pair unless the schema proves
    ``lat`` passes through ``p1`` unchanged. Proving that is this rule's job, which is
    why it reads the folded schema and stays inside the trusted prefix.

    The pair collapses to ``p2`` when, in the schema entering ``p1``:

    - ``p1`` is not ``single``. A bare-name ``p1`` gives a DataArray, where ``p2`` would
      be indexing, not projection (the same guard as :func:`merge_adjacent_projects`).
    - every name ``p1`` selects is a tracked data variable, so ``p1`` is a real,
      modelled projection and the dims it spans are known.
    - every name ``p2`` selects is a coordinate that survives ``p1``: its dims are a
      subset of the dims ``p1``'s variables span. This is the same survival test
      :func:`~xrexpr.schema.apply_schema` uses for ``Project``. A projection never
      changes a coordinate's values, so a surviving coordinate reads the same from
      ``p1``'s result as from the base.

    The survival test also stops the rule changing errors into values. If the coordinate
    doesn't survive ``p1`` (``ds[["elevation"]]["time"]``, where ``elevation`` has no
    ``time``), the eager chain raises, and the rule declines (see
    ``planning/roadmap/07-small-wins.md`` §8). A ``p2`` that names a data variable is
    left to :func:`merge_adjacent_projects`, or is an eager error left alone.

    Dropping ``p1`` can't orphan anything downstream: after ``p2`` only coordinates
    remain, so ``p1``'s data variables are gone whether or not ``p1`` ran. The rule
    shortens the plan, so termination holds. One drop per call; the fixpoint repeats it.
    """
    if not any(isinstance(node, Project) for node in nodes):
        return None  # nothing to drop: don't fold the schema for a projection-free plan

    limit = _trusted_prefix(nodes)
    schemas = _schemas(nodes[:limit], schema)
    for i in range(limit - 1):
        first, second = nodes[i], nodes[i + 1]
        if not isinstance(first, Project) or not isinstance(second, Project):
            continue
        if first.single:
            continue  # a bare-name p1 indexes; p2 is not a projection of its result

        entering = schemas[i]
        if not all(n in entering.data_vars for n in first.variables):
            continue  # p1 isn't a modelled projection; its spanned dims are a guess

        spanned = {d for n in first.variables for d in entering.variables[n]}
        survives = all(
            c in entering.coord_names and set(entering.variables[c]) <= spanned
            for c in second.variables
        )
        if not survives:
            continue

        return list(nodes[:i]) + list(nodes[i + 1 :])
    return None


def push_projection_past_drop(
    nodes: Plan, schema: SchemaState[Hashable]
) -> Plan | None:
    """Move a projection in front of a preceding ``drop_vars``, trimming the drop.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema of the dataset the plan starts from. Needed: which dropped names
        would *survive* the projection is something only the folded schema knows.

    Returns
    -------
    Plan or None
        The plan with one ``(Drop, Project)`` pair rewritten, or ``None`` if none
        qualifies.

    Notes
    -----
    Two cases, one rewrite:

    - ``ds.drop_vars("elevation")[["temperature"]]``: the projection removes
      ``elevation`` anyway, so the drop is redundant. The best plan is
      ``ds[["temperature"]]``.
    - ``ds.drop_vars("area")[["temperature"]]``, where ``area`` is a coordinate
      ``temperature`` spans: ``area`` survives the projection, so the projection can go
      first and the drop stays after it. Same result.

    In general::

        [Drop(names), Project(keep)] -> [Project(keep), Drop(names that survive)]

    and the trailing ``Drop`` disappears if nothing survives.

    A dropped name *survives* the projection if it's a coordinate whose dims the kept
    variables still span. A dropped data variable never survives (the projection
    excludes it), which is the redundant case that shortens the plan.

    The pair is rewritten when, in the schema entering the ``Drop``:

    - the projection is list-form (not ``single``). A bare-name projection gives a
      DataArray, where ``drop_vars`` behaves differently. Like the other projection
      rules, this only handles Datasets.
    - every kept name is a tracked data variable (so the projection is modelled and its
      dims are known), and none of them is also dropped. Dropping then projecting the
      same name is an eager ``KeyError``, and the rule must not turn that into a value.
    - dropped names missing from the schema are treated like any other non-survivor:
      they can't affect the result, and they go with the redundant part of the drop.

    This rule can skip an error, under the same §8 rule :func:`pushdown_projections`
    uses. Dropped data variables are never computed, and a missing dropped name is never
    checked, so a chain that only fails in discarded work returns the projection-first
    answer. No kept value changes.

    Stays inside the trusted prefix, so the receiver is a known Dataset and the fold is
    exact. The redundant case shortens the plan; the hop case moves the projection one
    step left (a ``Drop`` isn't counted by the measure). Either way termination holds.
    One rewrite per call.

    ``drop_vars`` only touches metadata, so removing it saves little runtime. Whether
    the rule is worth the schema fold it triggers is open (#180); it's correct either
    way.
    """
    if not any(isinstance(node, Drop) for node in nodes):
        return None  # nothing to rewrite: don't fold the schema for a drop-free plan

    limit = _trusted_prefix(nodes)
    schemas = _schemas(nodes[:limit], schema)
    for i in range(limit - 1):
        first, second = nodes[i], nodes[i + 1]
        if not isinstance(first, Drop) or not isinstance(second, Project):
            continue
        if second.single:
            continue  # a bare-name projection is DataArray indexing, out of scope

        entering = schemas[i]
        dropped = set(first.variables)
        keep = second.variables
        if not all(n in entering.data_vars for n in keep) or dropped & set(keep):
            continue  # projection isn't a modelled data-var projection, or drops-then-keeps

        spanned = {d for n in keep for d in entering.variables[n]}
        survivors = tuple(
            n
            for n in first.variables
            if n in entering.coord_names and set(entering.variables[n]) <= spanned
        )
        moved: Plan = [second]
        if survivors:
            moved.append(Drop(name=first.name, variables=survivors))
        return list(nodes[:i]) + moved + list(nodes[i + 2 :])
    return None


def pushdown_selects(nodes: Plan, schema: SchemaState[Hashable]) -> Plan | None:
    """Move a select one step left, past the preceding node, when the dims allow it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema entering the plan. Unused: this rule only reasons about dims, and
        even :data:`~xrexpr.ir.ALL_DIMS` needs no schema here (see the notes).

    Returns
    -------
    Plan or None
        The plan with one select moved one step left, or ``None`` if nothing moved.

    Raises
    ------
    InvalidExpressionError
        If the select indexes a dim the preceding node *removed*
        (:attr:`DimEffect.on_conflict` is ``"invalid"``). The chain can never replay.

    Notes
    -----
    The decision is set arithmetic: compare the select's dims with the preceding node's
    :attr:`DimEffect.blocks`, then use the node's ``on_conflict`` to decide what an
    overlap means.

    - **No overlap**: the select touches nothing the node consumed, minted or resized,
      so swapping is valid and cheaper. ``ds.mean("lat").isel(time=0)`` becomes
      ``ds.isel(time=0).mean("lat")``, and
      ``ds.groupby("time.month").mean().isel(lat=0)`` runs the climatology over one
      latitude. That's the pattern this project exists for.
    - **Overlap, ``"invalid"``**: a plain reduce *removed* those dims, so the chain can
      never replay (``mean("lon").isel(lon=0)``, or ``mean().isel(time=0)``). Raise
      :class:`~xrexpr.InvalidExpressionError` instead of emitting a wrong reorder.
    - **Overlap, ``"immovable"``**: the dims survive in some form, so the chain is
      probably valid, just not reorderable. ``.mean().isel(month=0)`` indexes a minted
      dim; ``rolling(time=3).mean().isel(time=0)`` indexes one the window kept. Leave
      it. If the chain really is invalid, xarray will say so at replay.
    - **``blocks is None``**: don't know, so don't move. Scans are the main example:
      ``cumsum("time").isel(time=5)`` stays put because order matters.

    That split is what lets one generic rule cover every node kind.

    A select can't disturb window boundaries either (``02-lowering.md`` §11.2). A window
    runs along its own dims independently at each position of the others, ``center`` and
    ``min_periods`` included. So a select on a different dim can't change what a window
    sees, and a select on a windowed dim never moves.

    ``ALL_DIMS`` needs no schema: whatever the dims turn out to be, a reduce over all of
    them leaves nothing to index, so every select dim overlaps.

    One step per call. The fixpoint repeats it so a select reaches the front of a run of
    reductions (where adjacent selects then merge).
    """
    for i in range(len(nodes) - 1):
        crossed, select = nodes[i], nodes[i + 1]
        if not isinstance(select, Select):
            continue

        effect = dim_effect(crossed)
        blocks = effect.blocks
        if blocks is None:
            continue

        select_dims = frozenset(select.indexer)
        shared = select_dims if isinstance(blocks, AllDims) else select_dims & blocks
        if not shared:
            swapped = list(nodes)
            swapped[i], swapped[i + 1] = select, crossed
            return swapped

        if effect.on_conflict == "invalid":
            raise InvalidExpressionError(
                f"{select.name}() indexes {sorted(str(d) for d in shared)}, "
                f"which {crossed.name}() has already reduced away"
            )
    return None


def pushdown_projections(nodes: Plan, schema: SchemaState[Hashable]) -> Plan | None:
    """Move a projection one step left, past a preceding node whose dims allow it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema of the dataset the plan starts from. Needed here, unlike in the
        dim-only rules: whether a projection may cross an op depends on which dims the
        kept variables carry at that point, which only the fold knows.

    Returns
    -------
    Plan or None
        The plan with one projection moved one step left, or ``None`` if nothing moved.

    Notes
    -----
    ``ds.mean("time")[["tas"]]`` reduces every variable, then throws away all but
    ``tas``. ``ds[["tas"]].mean("time")`` reduces one. The rule swaps the pair when it's
    safe, and that comes down to one question: **do the kept variables still carry the
    dims the preceding op needs?** (:attr:`DimEffect.requires`). With ``tas(time, lat)``
    and ``elevation(lat)``:

    - ``ds.mean("time")[["tas"]]`` swaps, because ``tas`` has ``time``;
    - ``ds.mean("time")[["elevation"]]`` is left alone, because ``mean("time")`` on a
      dataset whose only variable lacks ``time`` raises. Unlike
      :func:`pushdown_selects`, this rule never raises: the original plan is valid, it
      just can't be reordered.

    The dims come from the schema *entering* the preceding op. The schema after it would
    say ``tas`` has no ``time``, and block the very case this rule is for. Two
    conservative cases block the hop: a name that isn't a known data variable (a
    coordinate, or something an unmodelled op created), and a required dim carried only
    by a coordinate.

    Reductions and selections act per variable, so a swap doesn't change the kept
    variables' values. A bare ``mean()`` carries :data:`~xrexpr.ir.ALL_DIMS`, which is
    resolved against the entering schema via :func:`~xrexpr.schema.resolve_dims`. So the
    hop only happens when the kept variables span every dim, which is exactly when the
    replayed bare ``mean()`` reduces the same dims. That's exact because the rule stays
    in the trusted prefix.

    **Fused reduces** cross on the same terms, and the check matters even more for them.
    A projection drops any dim coordinate no kept variable uses, so
    ``ds[["elevation"]]`` (no ``time``) has no ``time`` at all, and
    ``groupby("time.month")``, ``resample``, ``rolling`` and ``coarsen`` would then
    raise. What each needs from its input is ``{group_dim} | consumes`` for a grouped
    reduce and the ``window`` keys for a windowed one. For a grouped reduce that's
    narrower than :attr:`DimEffect.blocks`; see :func:`dim_effect`.

    **This rule can skip an error, on purpose.** Moving a projection earlier means
    discarded variables are never computed. Usually that's just cheaper. Occasionally it
    also avoids an error: ``ds.std("time")[["temperature"]]`` raises while reducing
    ``elevation``, which the chain throws away anyway, and the rewritten plan never
    touches it. The values can't change, because the check above already ensured the
    kept variables reduce the same way. The contract is: keep every value the plan asks
    for, never add an error, but it's fine to skip an error raised by discarded work.
    It's written up in ``planning/roadmap/07-small-wins.md`` §8.

    **Weighted reduces cross too**, under that contract. A weighted reduce *raises* on a
    variable missing a named dim, where a plain reduce skips it. So the hop can skip an
    error, which is the clearest example of §8:
    ``ds.plan.weighted(w).mean("time")[["temperature"]]`` raises eagerly because of
    ``elevation``, which is discarded, and succeeds once the projection goes first. A
    projection never needs to subset the weights, which is why projections may cross a
    weighted reduce but selects may not.

    For weighted reduces, ``requires`` is ``consumes | weight_dims``, and the extra term
    matters. A projection drops any dim coordinate no kept variable uses, and the
    weights behave differently depending on whether that coordinate exists (inner-join
    if it does, broadcast if not). Checked against xarray 2026.7.0: without the extra
    term, two chains give different *values*. See the :func:`dim_effect` arm.

    One step per call. The fixpoint repeats it so a projection moves to the front of the
    plan (where #43 would like to turn it into a backend read plan).
    """
    if not any(isinstance(node, Project) for node in nodes):
        return None  # nothing to move: don't fold the schema for a projection-free plan

    limit = _trusted_prefix(nodes)
    schemas = _schemas(nodes[:limit], schema)
    for i in range(limit - 1):
        project = nodes[i + 1]
        if not isinstance(project, Project):
            continue

        crossed = nodes[i]
        requires = dim_effect(crossed).requires
        if requires is None:  # don't know what it needs, so don't move anything past it
            continue

        # ``ALL_DIMS`` (a bare ``mean()``) resolves against the schema *entering* the
        # preceding node. That's exact because this loop stays inside the trusted
        # prefix. The subset test then only passes when the kept variables span every
        # dim, i.e. when the replayed bare reduce reduces the same dims.
        needed = resolve_dims(requires, schemas[i].dim_names)
        available = schemas[i].var_dims(project.variables)
        if available is None or not needed <= available:
            continue

        swapped = list(nodes)
        swapped[i], swapped[i + 1] = project, crossed
        return swapped
    return None


def pushdown_selects_past_rechunks(
    nodes: Plan, schema: SchemaState[Hashable]
) -> Plan | None:
    """Move a select in front of a preceding ``chunk`` so the rechunk handles less data.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema entering the plan. Unused: this rule has no dim check at all.

    Returns
    -------
    Plan or None
        The plan with one select moved in front of a rechunk (the chunk spec trimmed or
        dropped as needed), or ``None`` if nothing moved.

    Notes
    -----
    A rechunk changes no dim, size or value, only how the data is split into blocks. So
    a select and a rechunk always commute as far as the result goes. What this rule
    protects is the chunking itself. Selecting first leaves dask less data to shuffle,
    and the blocks are never coarser than in the eager order. ``chunk({time: 100})``
    then ``isel(time=slice(50, 250))`` gives ragged ``(50, 100, 50)`` blocks; pushed, it
    gives regular ``(100, 100)`` ones.

    It isn't a plain swap. A dim the select *drops* must also leave the chunk spec, or
    xarray raises ``ValueError: chunks keys ('time',) not found in data dimensions``.
    So, checking the select's ``consumes``:

    - **No named dim dropped**: swap as is. This covers the uniform forms that get here
      (``chunk()``, ``chunk(100)``, ``chunk(-1)``), which name no dim. xarray reads a
      uniform spec as ``dict.fromkeys(self.dims, spec)``, so it already means "whatever
      dims there are", and a select in front just leaves fewer of them.
    - **Some named dims dropped**: swap, and rebuild the rechunk from the surviving
      keys.
    - **Every named dim dropped**: swap and *remove* the rechunk. What's left would be
      ``chunk({})``, a single-chunk dask array: no parallelism, no out-of-core benefit.
      (This is not the same as ``open_dataset(chunks={})``, which uses on-disk chunks;
      the method has no such knowledge.) A rechunk that named no dim to begin with is
      kept, because there converting to dask is the point.

    The rule never repairs a spec it moves past. Specs that would need repairing (byte
    targets, and ``"auto"`` in general) change meaning with the array's extent, and
    ``_pushable_rechunk`` refuses them. One narrow exception: a select whose positional
    indexer is provably empty, on xarray 2026.09.0 or later. There, `pydata/xarray#11486
    <https://github.com/pydata/xarray/pull/11486>`_ replaces the auto request with the
    full zero-byte shape before dask sees it. See :func:`_empty_autochunk_is_pushable`.

    Unlike :func:`pushdown_selects`, this never raises: a rechunk can't make a select
    fail, only slower. One step per call.
    """
    for i in range(len(nodes) - 1):
        match nodes[i], nodes[i + 1]:
            case (Rechunk() as rechunk, Select() as select) if _pushable_rechunk(
                rechunk
            ) or _empty_autochunk_is_pushable(rechunk, select):
                kept = {
                    dim: spec
                    for dim, spec in rechunk.chunks.items()
                    if dim not in select.consumes
                }
                if len(kept) == len(rechunk.chunks):  # nothing named was dropped
                    moved: Plan = [select, rechunk]
                elif kept:
                    moved = [
                        select,
                        Rechunk(name=rechunk.name, chunks=frozendict(kept)),
                    ]
                else:  # the spec is spent
                    moved = [select]
                return list(nodes[:i]) + moved + list(nodes[i + 2 :])
    return None


def _empty_autochunk_is_pushable(rechunk: Rechunk, select: Select) -> bool:
    """Report whether a select that provably empties a dim may cross an auto rechunk.

    ``"auto"`` normally blocks the hop, because dask sizes it from the array it
    receives. A select that is *provably empty* is the one exception: xarray 2026.09.0
    and later replace auto chunks with the full zero-byte shape before dask can divide
    by zero (pydata/xarray#11486). Byte targets and explicit block sequences are still
    refused, because their replay still depends on the extent before the select.

    Parameters
    ----------
    rechunk : Rechunk
        The rechunk the select would move in front of.
    select : Select
        The select immediately after it.

    Returns
    -------
    bool
        Whether the swap is safe: xarray has the fix, ``select`` provably empties a dim,
        and every chunk spec is ``"auto"`` or extent-independent, with at least one
        ``"auto"``.
    """
    if not AUTOCHUNK_EMPTY_DIMS_SAFE or not _select_provably_empties(select):
        return False
    if any(key not in rechunk.chunks for key in rechunk.kwargs):
        return False

    specs: list[ChunkSpec] = list(rechunk.chunks.values())
    if rechunk.uniform is not None:
        specs.append(rechunk.uniform)

    has_auto = False
    for spec in specs:
        match spec:
            case Auto():
                has_auto = True
            case SingleSize() | FullDim() | NoChange():
                continue
            case BlockSeq() | ByteSize() | OpaqueChunk():
                return False
            case _:
                assert_never(spec)
    return has_auto


def _select_provably_empties(select: Select) -> bool:
    """Report whether one positional indexer leaves its dim with zero elements.

    Reads only the recorded indexer, never data or schema sizes. An empty position list,
    an all-false mask, and a forward slice ending at 0 are empty for every input length.
    Anything else answers ``False``.

    Parameters
    ----------
    select : Select
        The select whose recorded indexer to read.

    Returns
    -------
    bool
        Whether some indexer selects zero elements whatever the dim's length.
    """
    for indexer in select.indexer.values():
        match indexer:
            case Positions(values=()):
                return True
            case Mask(values=values) if not any(values):
                return True
            case ForwardSlice(stop=0):
                return True
    return False


def _pushable_rechunk(node: Rechunk) -> bool:
    """Report whether a select may move in front of ``node``.

    Parameters
    ----------
    node : Rechunk
        The rechunk the select would move in front of.

    Returns
    -------
    bool
        ``True`` if the select may cross, ``False`` if the rechunk blocks it.

    Notes
    -----
    First, the call itself: any **option kwarg** (``token``, ``chunked_array_type``,
    ...) blocks, because a rebuilt spec couldn't carry it faithfully. Same reason
    ``_mergeable_select`` refuses option kwargs.

    Then the specs. Both the per-dim and the uniform spec go through one ``match`` over
    :data:`~xrexpr.chunks.ChunkSpec`, closed with ``assert_never``, so a new variant
    fails mypy here until someone decides which side it belongs on. ``chunk("auto")``
    arrives as :class:`~xrexpr.chunks.Auto` and ``chunk((100, 400, 500))`` as
    :class:`~xrexpr.chunks.BlockSeq`, the same as their per-dim spellings.

    The dividing line is whether **dask sizes the spec by measuring the array it's
    given**:

    - **Extent-dependent specs block.** A block sequence must sum to the dim's length,
      so after a select it can't replay at all. ``"auto"`` and byte targets divide a
      byte budget by the array's extent, so a select in front changes what they mean. If
      the select empties a dim, that becomes a ``ZeroDivisionError`` on xarray before
      2026.09.0 (see the notes on :class:`~xrexpr.chunks.Auto`).
      :func:`_empty_autochunk_is_pushable` handles the one safe auto case separately.
    - **Extent-independent specs cross.** A fixed size means the same on any array,
      ``-1`` is always one block, and ``None`` keeps what's there. No select can
      invalidate them.
    - **An unmodelled spec** (:class:`~xrexpr.chunks.OpaqueChunk`) blocks, just as an
      :class:`~xrexpr.ir.Opaque` node does.

    Nothing here reads a size or the schema. The decision comes from the spec alone.

    ``specs`` has a type annotation, and it matters. Built as ``(*node.chunks.values(),
    *uniform)`` it would infer ``Any``, and then ``assert_never`` accepts anything and
    the exhaustiveness check silently stops working. If you rewrite these lines, delete
    a ``case`` arm and check that mypy still fails.
    """
    if any(key not in node.chunks for key in node.kwargs):
        return False
    specs: list[ChunkSpec] = list(node.chunks.values())
    if node.uniform is not None:
        specs.append(node.uniform)
    for spec in specs:
        match spec:
            case BlockSeq() | Auto() | ByteSize() | OpaqueChunk():
                return False  # dask sizes these from the array it is handed
            case SingleSize() | FullDim() | NoChange():
                continue  # stated in terms no select can invalidate
            case _:
                assert_never(spec)
    return True


def _mergeable_rechunk(node: Rechunk) -> bool:
    """Report whether ``node`` is a per-dim rechunk that can be merged with another.

    Parameters
    ----------
    node : Rechunk
        The rechunk to test.

    Returns
    -------
    bool
        ``True`` when ``node`` names its dims and every spec it carries can be merged;
        ``False`` for the uniform, empty and blocking forms.

    Notes
    -----
    Two conditions:

    - **Per-dim form**: ``node.uniform is None`` and ``node.chunks`` is non-empty. A
      uniform spec (``chunk(100)``, ``chunk("auto")``, ``chunk((100, 400, 500))``)
      rechunks every dim, so it can't be merged into a per-dim dict. A bare ``chunk()``
      names no dim but still touches all of them, so a dict merge can't express it
      either.
    - **Pushable**: :func:`_pushable_rechunk` holds. That rules out option kwargs (which
      the rebuilt node would drop) and the extent-dependent specs. What's left is
      ``SingleSize``/``FullDim``/``NoChange``, whose
      :meth:`~xrexpr.chunks.SingleSize.to_raw` values round-trip through
      :func:`~xrexpr.chunks.classify_chunk`.
    """
    return node.uniform is None and bool(node.chunks) and _pushable_rechunk(node)


def merge_adjacent_rechunks(nodes: Plan, schema: SchemaState[Hashable]) -> Plan | None:
    """Merge two adjacent per-dim ``chunk`` calls into one.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState[Hashable]
        The schema entering the plan. Unused: this rule decides from the two nodes'
        specs.

    Returns
    -------
    Plan or None
        The plan with one adjacent ``(Rechunk, Rechunk)`` pair replaced by a merged
        node, or ``None`` if no pair merges.

    Notes
    -----
    ``chunk({"time": 100}).chunk({"lat": 50})`` becomes ``chunk({"time": 100, "lat":
    50})``. That's shorter, and a better input for
    :func:`pushdown_selects_past_rechunks`. Both nodes must pass
    :func:`_mergeable_rechunk`; other forms are left alone.

    Dask applies a later ``chunk`` per dim on top of an earlier one, so the later spec
    wins, with one exception. A later :class:`~xrexpr.chunks.NoChange` (``chunk({dim:
    None})``, "leave this dim alone") must **not** override an earlier spec on the same
    dim. ``chunk({"time": 100}).chunk({"time": None})`` keeps ``time`` at 100-element
    blocks, where a plain ``{**r1, **r2}`` would reset it to the base's single block
    (checked against dask). So a later ``NoChange`` only fills a dim the earlier node
    didn't specify.

    The merged node is built the same way :func:`pushdown_selects_past_rechunks`
    rebuilds a trimmed one: ``chunks`` only, no ``args`` or ``kwargs``. That's safe
    because both inputs passed the pushable check and so had no option kwargs. With no
    ``args`` the result's ``uniform`` is ``None``, so it can be merged again.

    One merge per call; the fixpoint handles longer runs. The rule shortens the plan, so
    termination holds.
    """
    for i in range(len(nodes) - 1):
        match nodes[i], nodes[i + 1]:
            case (Rechunk() as r1, Rechunk() as r2) if _mergeable_rechunk(
                r1
            ) and _mergeable_rechunk(r2):
                merged = dict(r1.chunks)
                for dim, spec in r2.chunks.items():
                    if isinstance(spec, NoChange):
                        merged.setdefault(dim, spec)
                    else:
                        merged[dim] = spec
                node = Rechunk(name=r1.name, chunks=frozendict(merged))
                return list(nodes[:i]) + [node] + list(nodes[i + 2 :])
    return None


_RULES: tuple[Rule, ...] = (
    merge_adjacent_selects,
    merge_adjacent_projects,
    merge_adjacent_rechunks,
    eliminate_projection_before_coord,
    push_projection_past_drop,
    pushdown_selects,
    pushdown_projections,
    pushdown_selects_past_rechunks,
)
