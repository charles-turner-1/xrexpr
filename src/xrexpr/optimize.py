"""The plan optimiser — rewrite a linear lowered plan to a cheaper equivalent.

:func:`optimize` runs the local rewrite rules in ``_RULES`` to a **fixpoint**: each maps a
plan to an equivalent plan and returns ``None`` when it changes nothing. Local rules plus
a fixpoint let a small rewrite compose into a large one: a select reaching the front of a
plan is one rule firing repeatedly, and nothing reasons about the whole chain.

Two things to keep when adding a rule. Termination rests on a lexicographic measure,
``(len(plan), sum of the indices of the Select and Project nodes)``, so **no rule may push
a node right, and none may lengthen a plan**; one that wants to needs a new measure, not
an exception. And the dim algebra has a single dispatch site, :func:`dim_effect`, a
``match`` closed with ``assert_never`` — a new node kind may still take the conservative
answer, but no longer *by accident*.

The schema fold lives here, not in recording, because only the optimiser knows how far it
can be trusted. :func:`~xrexpr.schema.apply_schema` models :class:`~xrexpr.ir.Opaque` as
variable-preserving, which ``rename`` and ``drop_vars`` are not, so ``_trusted_prefix``
bounds the rules that consult ``data_vars``. Dim-level rules are unaffected — an opaque
node costs rewrites, not correctness.

See ``docs/internals/optimiser.md``.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard

from frozendict import frozendict
from typing_extensions import assert_never

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
    ALL_DIMS,
    AllDims,
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

#: What the rules rewrite — a **lowered** plan. Spelled with
#: :data:`~xrexpr.ir.LoweredOp` rather than :data:`~xrexpr.ir.Op` because that is the
#: level ``optimize`` is contracted to run at — a node that should have been fused away
#: by ``lower.to_lower_ir`` becomes a type error here rather than a convention.
Plan = list[LoweredOp]
#: A rule maps a plan (and the schema its first node sees) to a rewritten plan, or
#: returns ``None`` when it changes nothing (letting :func:`optimize` detect the fixpoint
#: without a full-plan equality compare). Dim-level rules ignore the schema argument.
Rule = Callable[[Plan, SchemaState], Plan | None]


def optimize(nodes: Plan, schema: SchemaState) -> Plan:
    """Rewrite ``nodes`` into an equivalent plan, applying every rule to a fixpoint.

    Parameters
    ----------
    nodes : Plan
        The lowered plan to rewrite.
    schema : SchemaState
        The schema of the dataset the plan starts from — the *base*, not the one left at
        the end of recording — since rules need to know the shape each node sees, and
        rewriting changes that.

    Returns
    -------
    Plan
        An equivalent plan: it replays to the same dataset as ``nodes``, only cheaper.

    Raises
    ------
    InvalidExpressionError
        If a rule proves the plan can never replay — a select on a dim a preceding
        reduce removed. See :func:`pushdown_selects`.

    Notes
    -----
    Each rule preserves the plan's result, which is what makes the guarantee above hold
    of the whole loop. A rule returns ``None`` when it changes nothing, so the loop
    detects the fixpoint from that signal rather than by comparing whole plans each pass.

    **Termination.** Every rule strictly decreases the lexicographic measure
    ``(len(plan), sum of the indices of the Select and Project nodes)``. Merging and
    dropping a spent rechunk shrink the plan; the three pushdown rules leave the length
    alone but move a select or projection strictly left. Neither component can grow, so
    no rule may ever push a node *right* or lengthen a plan — the invariant a new rule
    has to preserve. (The pushdown rules fire on disjoint adjacencies — ``(*, select)``,
    ``(*, project)`` and ``(rechunk, select)`` — so they can't undo one another.)
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


def _schemas(nodes: Plan, base: SchemaState) -> list[SchemaState]:
    """Fold the base schema forward through a plan.

    Parameters
    ----------
    nodes : Plan
        The plan to fold over.
    base : SchemaState
        The schema of the dataset the plan starts from.

    Returns
    -------
    list of SchemaState
        The schema each node *sees*: ``out[i]`` is the schema entering ``nodes[i]``.
    """
    if not nodes:
        return []
    out = [base]
    for node in nodes[:-1]:
        out.append(apply_schema(out[-1], node))
    return out


def _trusted_prefix(nodes: Plan) -> int:
    """Find how far the folded schema is exact.

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
    rename, add or drop variables — so past the first one, ``data_vars`` is a guess. A
    rule reasoning about variables must confine itself to ``nodes[:_trusted_prefix(nodes)]``.
    """
    for i, node in enumerate(nodes):
        if isinstance(node, Opaque):
            return i
    return len(nodes)


@dataclass(frozen=True)
class DimEffect:
    """What one node does to dims, as the *rules* need to know it.

    Attributes
    ----------
    blocks : DimSet or None
        Dims a select may not hop over, or ``None`` to refuse every hop.
    requires : DimSet or None
        Dims the node needs of its input, or ``None`` to refuse every projection hop.
    on_conflict : {"immovable", "invalid"}
        What a select hopping over :attr:`blocks` means. ``"invalid"`` — the node
        *removed* those dims, so the chain can never replay and the optimiser should say
        so. ``"immovable"`` — the dims survive in some form, so the chain is very likely
        valid and merely un-reorderable; leave it. Only meaningful when ``blocks`` is not
        ``None``.

    Notes
    -----
    A **derived** view, never stored: :func:`dim_effect` computes it from a node with one
    ``match``, which is the house's "derive, don't store" discipline (``Select.consumes``,
    ``Project.single``) applied one level up — at the plan rather than the node. One
    dispatch site instead of a partial match per rule, so every node kind answers both
    rules' questions in one place.

    Two fields rather than one dim set, because the pushdown rules approach a node from
    **opposite sides** and that turned out to be the load-bearing fact:

    - :attr:`blocks` is read by a rule moving a **select** left, i.e. from *after* the
      node. It must avoid everything the node consumed, minted or resized — a grouped
      reduce's minted dim included, since a select is entitled to name it.
    - :attr:`requires` is read by a rule moving a **projection** left, i.e. to *before*
      the node. It is what the node needs of its **input**, which excludes a minted dim
      outright: that dim does not exist yet where the projection would land.

    ``None`` means *don't know*, the contract ``SchemaState.var_dims`` states, with the
    same discipline: it must be read as **no rewrite**, never as "no dims". Nothing crosses
    a node that answers ``None``.
    """

    blocks: DimSet | None
    requires: DimSet | None
    on_conflict: Literal["invalid", "immovable"] = "immovable"


#: Nothing crosses this node in either direction — the answer for every kind whose dim
#: effect the optimiser does not model (``Opaque``, which could do anything; ``Rechunk``,
#: which has its own rule) and for ``Project``, which the *other* rules handle.
_OPAQUE_EFFECT = DimEffect(blocks=None, requires=None)


def dim_effect(node: LoweredOp) -> DimEffect:
    """Compute the dim effect of ``node``: one dispatch site, closed with ``assert_never``.

    Parameters
    ----------
    node : LoweredOp
        The node a rule wants to move something across.

    Returns
    -------
    DimEffect
        What a select coming from the right must avoid, and what a projection going to
        the left must supply. Kinds the optimiser does not model answer ``None`` to both.

    Notes
    -----
    The point of gathering these here is that a **new variant must answer every question at
    once**. Before this existed the same facts were spread over three partial matches — one
    in ``pushdown_selects``, one in ``_fused_dims``, four arms in
    :func:`pushdown_projections` — each with its own silent "anything else is left"
    fallback, so a new node kind would quietly get the conservative answer from all three
    without anyone deciding it should.

    A conservative answer is still available, and is what most kinds take: ``None`` on both
    fields. What is no longer available is taking it *by accident*.
    """
    match node:
        case Reduce(consumes=consumes) as reduce if reduce.keepdims:
            # ``keepdims=True`` removes nothing: it keeps every named dim at size 1, so a
            # later select on one is valid but must not reorder across (fill-then-slice on
            # a size-1 dim is not slice-then-fill). Same shape as ``WindowedReduce`` -- the
            # dims are both blocked and required, and the default ``immovable`` leaves the
            # chain rather than rejecting it.
            return DimEffect(blocks=consumes, requires=consumes)
        case Reduce(consumes=consumes):
            # A reduce removes exactly what it names, so a later select on one of those
            # dims can never replay -- the empty-dim reorder bug, and the one case where
            # the optimiser is entitled to reject the chain itself.
            return DimEffect(blocks=consumes, requires=consumes, on_conflict="invalid")
        case Select(indexer=indexer):
            # Nothing hops a select over a select -- ``merge_adjacent_selects`` folds those
            # instead -- but a *projection* crosses one, and needs its indexed dims.
            return DimEffect(blocks=None, requires=frozenset(indexer))
        case GroupedReduce(group_dim=group_dim, new_dim=new_dim, consumes=consumes):
            return DimEffect(
                blocks=frozenset({group_dim, new_dim}) | consumes,
                requires=frozenset({group_dim}) | consumes,
            )
        case WindowedReduce(window=window):
            # A window consumes nothing and mints nothing; it can only *resize*, and only
            # the dims it names. So the keys are both answers.
            return DimEffect(blocks=frozenset(window), requires=frozenset(window))
        case WeightedReduce(consumes=consumes, weight_dims=weight_dims):
            # ``blocks=None``: a select still never crosses one, because the hop would
            # need the *weights* subset alongside -- a data-touching rewrite, deferred
            # (``planning/roadmap/02-lowering.md`` §8.1).
            #
            # ``requires`` admits projections, and the ``weight_dims`` term is what makes
            # that sound. A projection drops a dim *coordinate* no surviving variable uses,
            # and the weights' behaviour toward a dim depends on whether that coord is
            # there: aligned (inner-joined) if so, freshly broadcast if not. Verified
            # against xarray 2026.7.0 -- with weights on an uncoordinated ``lat``,
            # ``ds.weighted(w).mean("time")[["ts"]]`` keeps ``lat``'s coord where the
            # hopped form has no ``lat`` coord at all, and with weights on a *disjoint*
            # ``lat`` the two disagree outright. Requiring the projected variables to carry
            # every weight dim keeps that coord alive, and closes both.
            match consumes:
                case AllDims():
                    # A bare closer clears every dim, minted weight dims included, so
                    # nothing survives for a coord to be missing from. ``ALL_DIMS`` alone
                    # is enough -- and it must be, since a weight dim the dataset lacks
                    # could never be a subset of the projected variables' dims.
                    return DimEffect(blocks=None, requires=ALL_DIMS)
                case frozenset() as named:
                    return DimEffect(blocks=None, requires=named | weight_dims)
                case _:
                    assert_never(consumes)
        case Scan(dims=dims):
            # A scan keeps its dims but is order-significant. ``blocks=dims``: a select on
            # a *disjoint* dim commutes (the scan acts independently at each position of
            # the others), an intersecting one is left, never raised -- ``on_conflict``
            # stays the default ``"immovable"``, since the dims survive. ``requires=dims``:
            # a projection may go before the scan iff the kept variables still carry the
            # scanned dims, exactly a reduce's condition -- ``07-small-wins.md`` §3's
            # ``Scan`` arm. (A *prefix* forward-slice on the scanned dim also commutes with
            # ``cumsum``/``cumprod`` but not ``diff``; a later refinement, out of scope.)
            return DimEffect(blocks=dims, requires=dims)
        case Elementwise():
            # A per-element op keeps every dim, so a select on *any* dim commutes with it
            # (empty ``blocks`` -- every select is disjoint) and a projection needs no dim
            # to cross it (empty ``requires``). This is not the ``None``/``None`` of a
            # barrier: the shapes that would break commutation -- a data- or per-variable-
            # shaped argument -- never reach here, having recorded ``Opaque`` at
            # ``schema._elementwise_safe``. ``apply_schema`` is exact for the node too, so
            # the trusted prefix rightly spans it with no change to ``_trusted_prefix``.
            return DimEffect(blocks=frozenset(), requires=frozenset())
        case Rename(mapping=mapping):
            # A rename relabels dim *names* mid-plan, and the name-keyed pushdowns do not
            # translate keys across it. So a select on a renamed dim must not hop -- above
            # the rename that dim has its old name -- while a select on an *untouched* dim
            # still commutes. ``blocks`` is the set of new names: a select names dims by
            # their current (post-rename) name, so blocking those keeps a renamed-dim select
            # put (``immovable``: the dim survives, so leave the chain, never reject it) and
            # lets an untouched-dim select through. Variable renames in the set are inert --
            # a select never names a variable. ``requires=None``: the projection pushdown is
            # dim-keyed, but the hazard here is a *variable*-key rename (a projection naming
            # a renamed variable cannot precede the rename that mints its new name), which
            # that machinery cannot see, so no projection crosses via the generic rule. A
            # schema-aware rename/projection rule, and the name-translating select cross
            # (``11-relabel-rename-drop.md`` §4), are follow-ups. The schema is exact for the
            # node regardless, which is what re-opens the trusted prefix past it.
            return DimEffect(blocks=frozenset(mapping.values()), requires=None)
        case Drop():
            # ``drop_vars`` remaps no dim, so a select commutes with it freely (empty
            # ``blocks`` -- every select is disjoint). But a *projection* may not cross it
            # blindly: moved before the drop it could remove the very name ``drop_vars``
            # targets, turning a valid chain into a ``KeyError`` -- so ``requires=None``. The
            # profitable schema-aware rewrites next to it (a projection excluding the dropped
            # data vars makes the drop redundant; one retaining a dropped coord may hop it)
            # want a dedicated trusted-prefix rule -- issue #176. The schema is exact for the
            # node regardless (``apply_schema``), which is what re-opens the trusted prefix
            # past it; crossability is the secondary question.
            return DimEffect(blocks=frozenset(), requires=None)
        case Project() | Rechunk() | Opaque():
            return _OPAQUE_EFFECT
        case _:
            assert_never(node)


def merge_adjacent_selects(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Fold each run of consecutive same-op selects into one node.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema entering the plan. Unused — this is a dim-level rule.

    Returns
    -------
    Plan or None
        The plan with every foldable run folded, or ``None`` when no run was folded.

    Notes
    -----
    Consecutive ``isel``s (or ``sel``s) compose, so ``ds.isel(time=0).isel(lat=1)``
    becomes a single ``isel({time: 0, lat: 1})``. Selects on *different* dims simply
    union; selects on the **same** dim are composed by ``_compose_into``, because
    the later indexer addresses positions within the earlier one's result rather than
    the original dim.

    Three things act as barriers, each ending the run so the plan keeps two correct
    nodes instead of collapsing to one wrong one: a mixed ``isel``/``sel`` run
    (different indexing semantics), a select carrying *option* kwargs
    (``drop``/``method``/...) that a bare merged indexer couldn't carry faithfully, and
    a same-dim collision with no statically provable composition.
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
        # Only the ``indexer`` is accumulated: ``consumes`` is a derived property of it, and
        # the replay header is derived from it by ``lower._emit_node`` (§7), so neither can
        # drift from the merged indexer — this rule touches the semantic field alone.
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
        ``True`` for a :class:`~xrexpr.ir.Select` carrying no option kwargs. A
        ``TypeGuard``, so callers narrow ``node`` to ``Select``.

    Notes
    -----
    A select whose kwargs carry keys beyond the indexed dims (``drop``, ``method``,
    ``missing_dims``, ``tolerance``) can't be folded into a bare indexer dict, so it
    acts as a merge barrier rather than being silently stripped of those options.
    """
    return isinstance(node, Select) and all(k in node.indexer for k in node.kwargs)


def _compose_into(
    name: str,
    outer: dict[Hashable, Indexer],
    inner: Mapping[Hashable, Indexer],
) -> dict[Hashable, Indexer] | None:
    """Merge ``inner``'s indexers into ``outer``'s, composing where dims collide.

    Parameters
    ----------
    name : str
        The select kind, ``"isel"`` or ``"sel"``. Only ``isel`` composes — ``sel``
        composition needs coordinate values.
    outer : dict
        The indexers applied first, accumulated so far.
    inner : Mapping
        The indexers applied second, addressing positions *within* ``outer``'s result.

    Returns
    -------
    dict or None
        The merged mapping, or ``None`` when the two must not be merged at all.

    Notes
    -----
    A dim in only one of the two carries over untouched. A dim in **both** is the case
    the plain ``dict.update`` got wrong: ``isel(time=slice(100,1000)).isel(time=slice(10,20))``
    is *not* ``isel(time=slice(10,20))`` — the second indexer addresses positions
    **within the first's result**, so the two must compose (here to ``slice(110,120)``).
    ``_compose_indexer`` does that for the cases it can prove; ``None`` from either
    function means "don't merge these two selects at all" (the caller ends the run), so
    an uncomposable collision degrades to a correct two-node plan rather than a wrong
    one-node plan.

    Composition is all-or-nothing: a single uncomposable dim abandons the whole merge,
    since a half-applied ``inner`` would be neither select.
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
    """Compose two positional indexers applied to the same dim, ``outer`` then ``inner``.

    Parameters
    ----------
    outer : Indexer
        The indexer applied first, to the original dim.
    inner : Indexer
        The indexer applied second, to ``outer``'s result.

    Returns
    -------
    Indexer or None
        The single equivalent indexer, or ``None`` when no composition is provable
        without the dim's length (which the optimiser doesn't carry).

    Notes
    -----
    The dividing line is whether ``outer``'s selected positions are knowable *without* that
    length, which three shapes satisfy:

    - **:class:`~xrexpr.indexers.Positions`** — already a concrete list, so the answer is
      just indexing it: ``[10, 20, 30]`` then ``1`` is ``20``.
    - **:class:`~xrexpr.indexers.Mask`** — the same thing spelled differently; the kept
      positions are the indices of its ``True`` flags.
    - **:class:`~xrexpr.indexers.ForwardSlice`** — arithmetic on the bounds, e.g.
      ``slice(100, 1000)`` then ``slice(10, 20)`` → ``slice(110, 120)``.

    The rest yield ``None`` for reasons that are *not* interchangeable: a ``GeneralSlice``
    has negative or reversed bounds that count from an end the composer cannot locate; a
    ``Label`` is not positional at all (§3.2); and a ``Scalar`` drops the dim outright, so
    ``inner`` has nothing left to apply to.

    Those three are spelled out rather than caught by a wildcard, and ``assert_never`` closes
    the match: this is the *policy* site — which shapes the optimiser is willing to prove a
    composition for — so a seventh :data:`~xrexpr.indexers.Indexer` variant should fail
    type-check here until someone decides which side of that line it falls on, rather than
    defaulting to "uncomposable" unnoticed. The same discipline ``apply_schema`` uses over
    ``Op``.
    """
    match outer:
        case Positions(values=values):
            return _index_sequence(values, inner)
        case Mask(values=flags):
            # A mask is a position enumeration written differently: the elements it keeps
            # are ``[k for k, flag in enumerate(flags) if flag]``, known without the dim
            # length. So it composes exactly as ``Positions`` does, via the same helper.
            return _index_sequence(
                tuple(k for k, flag in enumerate(flags) if flag), inner
            )
        case ForwardSlice():
            return _compose_slice(outer, inner)
        case Scalar() | GeneralSlice() | Label():
            return None
        case Advanced():
            # Never actually reached: a select carrying an advanced indexer is ``Opaque``,
            # so no two of them are ever composed. Answered for exhaustiveness -- its select
            # is a barrier, so "uncomposable" is the only sound reply.
            return None
        case _:
            assert_never(outer)


def _index_sequence(positions: tuple[int, ...], inner: Indexer) -> Indexer | None:
    """Apply ``inner`` to a concrete tuple of positions, i.e. ``positions[inner]``.

    Parameters
    ----------
    positions : tuple of int
        The positions ``outer`` selected, fully enumerated.
    inner : Indexer
        The indexer applied to that enumeration.

    Returns
    -------
    Indexer or None
        The composed indexer, or ``None`` when the composition is refused.

    Notes
    -----
    Exact by construction — the outer selection is already fully enumerated, so there
    is nothing to reason about. An out-of-range ``inner`` would raise here; that is
    reported as uncomposable (``None``) so the error surfaces from xarray at replay, in
    its own words, rather than from the optimiser.
    """
    seq = list(positions)
    try:
        match inner:
            case Scalar() as s if s.position is not None:
                return Scalar(seq[s.position])
            case ForwardSlice() | GeneralSlice() as s:
                # ``seq`` is already concrete, so a reversed/negative slice is fine here.
                return Positions(tuple(seq[s.to_raw()]))
            case Positions(values=idx):
                return Positions(tuple(seq[i] for i in idx))
            case Mask(values=flags):
                # Both sides are concrete, so this is the same exactness as the arm above.
                # A length mismatch is not ours to diagnose: xarray requires a boolean mask
                # to match the dim it indexes, so a wrong length would raise at replay —
                # refuse, and let it say so.
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
        The slice applied first — forward and non-negative by construction.
    inner : Indexer
        The indexer applied to its result.

    Returns
    -------
    Indexer or None
        The composed indexer, or ``None`` when the composition is refused.

    Notes
    -----
    Element ``k`` of the result is ``outer_start + (inner_start + k * inner_step) *
    outer_step``, which is itself an arithmetic progression — so a slice composed with a
    slice is a slice, and with a scalar is a scalar. The stop bound is the *tighter* of
    the two constraints: ``inner`` cannot run past its own stop, nor past the end of
    what ``outer`` produced. ``outer`` needs no forward-check — it is a
    :class:`~xrexpr.indexers.ForwardSlice`, forward by construction.
    """
    start, step = outer.start or 0, outer.step or 1

    match inner:
        case Scalar() as s if s.position is not None and s.position >= 0:
            position = start + s.position * step
            # Out of ``outer``'s range: xarray raises, but the composed scalar might well
            # be a valid position in the full dim, which would silently return data instead.
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
            # ``outer`` is an arithmetic progression, so its element ``k`` sits at
            # ``start + k * step`` — the ``Scalar`` arm's arithmetic, mapped over ``idx``.
            # A *negative* entry counts back from the end of what ``outer`` produced, which
            # is the length the composer does not carry, so it disqualifies the whole merge.
            if any(k < 0 for k in idx):
                return None
            return _mapped_positions(outer, start, step, addressed=idx, kept=idx)

        case Mask(values=flags):
            # Sound because of a fact xarray enforces rather than one the composer carries:
            # a boolean mask must be exactly as long as the dim it indexes, so ``len(flags)``
            # *is* how many elements ``outer`` produced — the very length the optimiser
            # otherwise refuses to guess. That makes the positions computable without knowing
            # the dim size, which is precisely what ``GeneralSlice`` cannot offer.
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
        The slice applied first; its ``stop`` is the bound that matters here.
    start : int
        ``outer``'s start, defaulted to 0.
    step : int
        ``outer``'s step, defaulted to 1.
    addressed : iterable of int
        Every element of ``outer`` that ``inner`` reads.
    kept : iterable of int
        The subset ``inner`` actually selects. Differs from ``addressed`` only for a mask.

    Returns
    -------
    Positions or None
        The mapped positions, or ``None`` when any addressed element is out of ``outer``'s
        range.

    Notes
    -----
    The check runs over *addressed*
    because it is the same trap the ``Scalar`` arm guards: reaching past ``outer``'s stop
    made the original two-step chain raise, but the composed position is often still valid
    in the full dim and would quietly return data instead. Reading is what raises, so
    reading is what has to be in range — whether or not the element is then kept.
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


def merge_adjacent_projects(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Drop the first of an adjacent pair of projections when the second subsumes it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema entering the plan. Unused — this rule is *syntactic*, deciding on the
        two nodes' own key sets (see the notes on why that is enough, and what it costs).

    Returns
    -------
    Plan or None
        The plan with one projection dropped, or ``None`` when no pair merges.

    Notes
    -----
    ``ds[["tas", "pr"]][["tas"]]`` records two :class:`~xrexpr.ir.Project` nodes and
    builds an intermediate Dataset holding ``pr`` purely to throw it away one node
    later. The pair collapses to the second alone **iff ``set(p2.variables) <=
    set(p1.variables)``**, and the second node is emitted verbatim — its key already
    describes the result exactly, so unlike :func:`merge_adjacent_selects` there is
    nothing to rebuild.

    **Why the subset test is enough for the coordinates too.** A projection keeps the
    coordinates its selected variables still span, so composing two of them prunes
    twice — but the second prune is against ``needed_dims(p2)``, which is a subset of
    ``needed_dims(p1)``, so every coordinate the lone ``p2`` would keep already survived
    ``p1``. Verified against xarray 2026.7.0: ``ds[["temperature", "elevation"]]
    [["temperature"]]`` is ``identical`` to ``ds[["temperature"]]``, auxiliary coords
    included, and so is the list-then-bare-name form.

    Two guards, each load-bearing:

    - **The subset test itself.** ``p2`` may legally name a *coordinate* —
      ``ds[["temperature"]]["lat"]`` works eagerly, because projection keeps coords — and
      the collapsed ``ds["lat"]`` is then a *different expression*, not a cheaper
      spelling of the same one: it reads the coord from the base, where the chain reads
      it as it exists inside ``p1``'s result. The two agree only by accident; put a
      select or a reduce between the projections, or let ``p1`` prune a coord ``p2``
      names, and they diverge. The subset test is exactly the condition under which the
      collapse is provable *without* the schema, so when it fails, **leave, never
      raise**: the chain is very likely valid as written, and this rule has no standing
      to rewrite it, let alone reject it.
    - **``p1`` must not be** ``single``. After a bare-name projection the object is a
      ``DataArray``, on which ``__getitem__`` is *indexing* rather than projection:
      ``ds["temperature"]["temperature"]`` raises ``KeyError`` where the collapsed
      ``ds["temperature"]`` returns data. Turning an error into a value is the one thing
      the contract in ``planning/roadmap/07-small-wins.md`` §8 forbids outright, so a
      ``single`` first node is a hard barrier — the subset test alone would not catch it.

    **Deciding this without the schema does skip one error, deliberately.** A name in
    ``p1`` that the dataset lacks makes the eager chain raise ``KeyError`` while the
    merged plan succeeds — ``ds.plan[["temperature", "nope"]][["temperature"]]``. That is
    §8's middle clause, the same licence :func:`pushdown_projections` runs on: the plan
    says outright that the extra name is not wanted, and the error comes from work whose
    result it discards. No value moves, because the surviving key is unchanged. Taking
    the schema-free reading is also what lets the rule admit a ``p1`` that names a
    coordinate (``ds[["lat", "temperature"]][["lat"]]``, which composes correctly), which
    a ``data_vars`` guard confined to ``_trusted_prefix`` could not do.

    It also lets the rule fire past an :class:`~xrexpr.ir.Opaque` — but nothing recorded
    supplies such a pair any more. An ``Opaque`` may *return a DataArray*
    (``ds.plan.pipe(lambda d: d["tas"])``), on which a list key indexes rather than
    projects, so ``_getitem_is_projection`` demotes every
    ``__getitem__`` after one; without that, ``[[1, 2]][[1]]`` would satisfy the subset
    test and collapse to the wrong row. The behaviour stays for plans built by hand or by
    a future producer that *can* prove the receiver — the rule's contract is unchanged,
    only its supply.

    One merge per call; :func:`optimize`'s fixpoint collapses a run of three. The rule
    shrinks the plan, so the termination measure is satisfied on its first component.
    """
    for i in range(len(nodes) - 1):
        match nodes[i], nodes[i + 1]:
            case (Project() as first, Project() as second) if not first.single and set(
                second.variables
            ) <= set(first.variables):
                return list(nodes[:i]) + list(nodes[i + 1 :])
    return None


def eliminate_projection_before_coord(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Drop a projection whose only consumer names coordinates that outlive it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema of the dataset the plan starts from. Load-bearing: whether ``p2``'s
        coordinate survives ``p1`` is a fact only the fold knows.

    Returns
    -------
    Plan or None
        The plan with the first projection of a qualifying pair dropped, or ``None`` when
        none qualifies.

    Notes
    -----
    ``ds[["temperature"]]["lat"]`` records two :class:`~xrexpr.ir.Project` nodes: the
    first (``p1``) materialises ``temperature``, the second (``p2``) throws it away and
    returns the ``lat`` *coordinate*. The optimal plan reads ``lat`` from the base and
    never touches ``temperature``. :func:`merge_adjacent_projects` cannot do this — its
    subset test ``set(p2) <= set(p1)`` fails, and it must, because that rule is syntactic
    and ``ds["lat"]`` is a *different expression* from the pair unless the schema proves
    ``lat`` passes through ``p1`` untouched. Proving that is this rule's whole job, which
    is why it lives in :func:`pushdown_projections`' family (confined to the trusted
    prefix, reading the folded schema) rather than in the syntactic merge.

    The pair collapses to ``p2`` alone when, in the schema entering ``p1``:

    - **``p1`` is not** ``single``. A bare-name ``p1`` yields a ``DataArray`` on which
      ``p2``'s ``__getitem__`` is *indexing*, not projection — the same hard barrier
      :func:`merge_adjacent_projects` carries.
    - **every name ``p1`` selects is a tracked data variable**, so ``p1`` is a genuine
      modelled projection and the dims it spans are known rather than guessed.
    - **every name ``p2`` selects is a coordinate that survives ``p1``** — its dims are a
      subset of the dims ``p1``'s selected variables span, which is exactly the survival
      test :func:`~xrexpr.schema.apply_schema`'s ``Project`` arm applies. A projection
      never touches a coordinate's *values*, so a surviving coordinate reads identically
      from ``p1``'s result and from the base, and dropping ``p1`` preserves the answer.

    The survival test is load-bearing, not an optimisation: it fires only when the
    coordinate is present in ``p1``'s output, i.e. exactly when the eager chain succeeds.
    Where the coordinate does *not* survive ``p1`` (``ds[["elevation"]]["time"]``, with
    ``elevation`` lacking ``time``), the eager chain raises and the rule **declines** —
    never turning that error into a value, the one thing ``planning/roadmap/07-small-wins.md``
    §8's contract forbids outright. A ``p2`` naming a data variable is left to
    :func:`merge_adjacent_projects` (its subset case) or is an eager error left alone.

    Dropping ``p1`` cannot orphan a variable downstream: ``p2`` is a coordinate
    projection, so ``p1``'s data variables are gone after it whether or not ``p1`` ran,
    and nothing past ``p2`` can read them. The rule shrinks the plan, so the termination
    measure is satisfied on its first component. One drop per call; :func:`optimize`'s
    fixpoint composes them.
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


def push_projection_past_drop(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a projection in front of a preceding ``drop_vars``, trimming the drop to survivors.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema of the dataset the plan starts from. Load-bearing: which dropped names
        *survive* the projection is a fact only the fold knows.

    Returns
    -------
    Plan or None
        The plan with one ``(Drop, Project)`` pair rewritten, or ``None`` when none
        qualifies.

    Notes
    -----
    ``ds.drop_vars("elevation")[["temperature"]]`` drops a variable and then projects a
    disjoint one — so the drop is *redundant*, the projection removes ``elevation`` anyway,
    and the optimal plan is ``ds[["temperature"]]``, which never touches ``elevation``.
    ``ds.drop_vars("area")[["temperature"]]``, with ``area`` a coordinate ``temperature``
    still spans, is the other half: ``area`` *survives* the projection, so the projection may
    go first and the ``drop_vars("area")`` stay after it, same values. Both are the one
    rewrite: for a ``(Drop(names), Project(keep))`` pair, put the projection first and keep a
    ``Drop`` of only the names that survive it —

    ``[Drop(names), Project(keep)] -> [Project(keep), Drop(names & survivors)]`` (the trailing
    ``Drop`` dropped entirely when nothing survives).

    A **survivor** is a dropped name a projection would keep: a coordinate whose dims the kept
    variables still span (a projection keeps the coords its selected variables span). A
    dropped **data variable** is never a survivor — the projection excludes it — so it is the
    redundant case that shrinks the plan. The pair rewrites when, in the schema entering the
    ``Drop``:

    - the projection is **list-form** (``not single``): a bare-name projection yields a
      ``DataArray``, whose ``drop_vars`` drops coordinates with different reach; kept in
      Dataset land, as the other projection rules are.
    - **every name the projection keeps is a tracked data variable**, so it is a genuine
      modelled projection and its spanned dims are known, and it **names nothing the drop
      removes** (a chain that drops then projects the same name is an eager ``KeyError`` this
      rule must not launder into a value).
    - dropped names that are absent from the schema are treated like other non-survivors:
      they cannot affect the projected value and disappear with the redundant part of the
      ``Drop``.

    **This rule may skip an error, and that is the same §8 licence** :func:`pushdown_projections`
    runs on: the dropped data variables are never computed, and an absent dropped name that
    the projection excludes is never validated, so chains that fail only in discarded pieces
    return the projection-first answer instead. No surviving value moves.

    Confined to the trusted prefix, so the receiver is a known ``Dataset`` and the fold is
    exact. The elimination case shrinks the plan; the hop case moves the projection one left
    (and a ``Drop`` is neither a ``Select`` nor a ``Project``, so it does not enter the
    measure) -- either way the termination measure strictly decreases. One rewrite per call;
    :func:`optimize`'s fixpoint composes them. Issue #176.

    ``drop_vars`` is a metadata-only op, so eliminating it saves little *runtime*; whether
    this rule earns the schema fold it triggers is a cost/benefit question to be profiled,
    tracked in #180 (it is confined and correct either way).
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


def pushdown_selects(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a select left past a preceding node whose dims permit it.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema entering the plan. Unused — this is a dim-level rule, and even
        :data:`~xrexpr.ir.ALL_DIMS` needs no schema here (see the notes).

    Returns
    -------
    Plan or None
        The plan with one select moved one hop left, or ``None`` when no adjacency swaps.

    Raises
    ------
    InvalidExpressionError
        When the select indexes a dim the crossed node *removed*
        (:attr:`DimEffect.on_conflict` is ``"invalid"``), so the chain can never replay.

    Notes
    -----
    The structural test is only *which* adjacency fires; what happens once it does is
    **set algebra** on the select's dims against the crossed node's
    :attr:`DimEffect.blocks`, plus that node's answer to what an overlap *means*:

    - **disjoint dims** — the select touches nothing the node consumed, minted or resized,
      so the swap is valid and strictly cheaper. ``ds.mean("lat").isel(time=0)`` becomes
      ``ds.isel(time=0).mean("lat")``, and ``ds.groupby("time.month").mean().isel(lat=0)``
      runs the climatology over one latitude — the pattern this project exists for.
    - **overlapping dims, ``on_conflict="invalid"``** — a plain reduce *removed* those
      dims, so the chain can never replay (``mean("lon").isel(lon=0)``, and the all-dims
      ``mean().isel(time=0)``): raise :class:`~xrexpr.InvalidExpressionError` rather than emit a
      silently-wrong reorder.
    - **overlapping dims, ``on_conflict="immovable"``** — the dims survive in some form, so
      the chain is very likely *valid* and merely un-reorderable. ``.mean().isel(month=0)``
      indexes a minted dim; ``rolling(time=3).mean().isel(time=0)`` indexes one the window
      kept. Leave it. Raising here would reject working chains, and where such a chain
      really is invalid xarray reports it at replay in its own words.
    - **``blocks is None``** — don't know, so don't move. Scans are the salient case:
      ``cumsum("time").isel(time=5)`` is left untouched because order matters there.

    That last distinction is what makes this one generic rule rather than one per node
    kind. A
    pushed select also cannot disturb window *boundaries* (``02-lowering.md`` §11.2):
    windows run along their own dims independently at each position of the others,
    ``center``/``min_periods`` included, so a select on a **disjoint** dim cannot change
    which elements a window sees — and an intersecting one never moves.

    ``ALL_DIMS`` needs no schema: whatever the dims turn out to be, a reduce over every one
    of them leaves nothing for a select to index, so every select dim overlaps.

    One hop per call. :func:`optimize`'s fixpoint composes hops so a select reaches the
    front of a run of reductions (and adjacent selects then merge).
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


def pushdown_projections(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a variable projection left past a preceding reduce, select or fused reduce.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema of the dataset the plan starts from. Load-bearing here, unlike in the
        dim-level rules: whether a projection may cross an op depends on which dims the
        projected *variables* carry at that point, which only the fold knows.

    Returns
    -------
    Plan or None
        The plan with one projection moved one hop left, or ``None`` when nothing moves.

    Notes
    -----
    ``ds.mean("time")[["tas"]]`` reduces every variable in the dataset and then throws
    all but ``tas`` away; ``ds[["tas"]].mean("time")`` reduces one. The rule fires on a
    ``(reduce | select, project)`` adjacency and swaps it when the projection can safely
    go first — which turns on a single question the nodes alone can't answer: **do the
    projected variables still carry the dims the crossed op names?** ``mean("time")`` on
    a dataset whose only variable has no ``time`` dim raises, so with
    ``tas(time, lat)`` and ``elevation(lat)``:

    - ``ds.mean("time")[["tas"]]`` → swap (``tas`` has ``time``);
    - ``ds.mean("time")[["elevation"]]`` → **left alone**. Unlike
      :func:`pushdown_selects` this rule never raises: that plan is perfectly valid
      eagerly, it simply can't be reordered.

    The dims come from ``data_vars`` in the schema entering the crossed op (``_schemas``),
    i.e. the variables' dims *before* it — the post-op dims would report ``tas`` as lacking
    ``time`` and block the very case this rule exists for. Two conservative edges:
    a name that isn't a known data variable (a coordinate, or something an unmodelled op
    introduced) blocks the hop, as does a dim carried only by a coordinate rather than by
    a projected variable.

    Reductions and selections act per variable, so a swap leaves the surviving variables'
    values untouched. A bare ``mean()`` carries :data:`~xrexpr.ir.ALL_DIMS`, resolved by
    :func:`~xrexpr.schema.resolve_dims` against that same entering schema — so the subset
    test only passes when the projected variables span every dim, which is exactly when the
    verbatim replayed ``mean()`` reduces the same ones. Resolving *here* is what makes it
    exact: this rule already confines itself to the trusted prefix, where the fold is not
    a guess.

    **The fused reduces** join on the same terms, and the guard is if anything
    more load-bearing for them: projecting drops a dim coordinate that no surviving
    variable uses, so ``ds[["elevation"]]`` (no ``time``) has no ``time`` *at all*, and
    every one of ``groupby("time.month")``/``resample``/``rolling``/``coarsen`` then raises
    rather than quietly doing nothing. The ``needed`` sets say what each requires of its
    **input**: ``{group_dim} | consumes`` for a grouped reduce, the ``window`` keys for a
    windowed one. That is :attr:`DimEffect.requires`, and for a grouped reduce it is
    deliberately *narrower* than :attr:`DimEffect.blocks` — see :func:`dim_effect`.

    **This rule may skip an error, and that is deliberate.** Moving a projection earlier
    means the discarded variables are never computed — usually just cheaper, occasionally
    also error-free: ``ds.std("time")[["temperature"]]`` raises while reducing
    ``elevation``, which the chain throws away, whereas the rewritten plan never touches
    it. The optimised answer is the better one, and it is safe because the *values* cannot
    move: the guard above has already established that the projected variables carry the
    dims the crossed op names, so they reduce identically either way. The sharpened
    contract — preserve every value the plan asks for, never introduce an error, but feel
    free to skip one raised by discarded work — is written up in
    ``planning/roadmap/07-small-wins.md`` §8.

    **Weighted reduces are admitted too**, on the strength of that same contract. The
    argument that once excluded them — a weighted reduce *refuses* a variable lacking a
    named dim where a plain one merely skips it, so the hop would mask an error — is
    precisely what §8 reclassifies as a win, and it is the sharpest instance of it:
    ``ds.plan.weighted(w).mean("time")[["temperature"]]`` raises eagerly over
    ``elevation``, which the chain discards, and succeeds once the projection runs first.
    A projection also never has to subset the weights, which is what keeps
    :func:`pushdown_selects` away from this variant.

    Their ``needed`` set carries an extra term — ``consumes | weight_dims`` — and it is
    load-bearing rather than decorative. Projecting drops a dim *coordinate* no surviving
    variable uses, and what the weights do with a dim turns on whether its coord is
    present: inner-join against it if so, broadcast a fresh one if not. Verified against
    xarray 2026.7.0, and without the extra term two chains diverge in *values*, not just
    in errors — see the :func:`dim_effect` arm. With it, a sweep of 512 weighted chains
    (278 of them hopped) changed no value, no coordinate and raised nothing new; the 36
    that skipped an error each returned exactly the projection-first answer.

    One hop per call. :func:`optimize`'s fixpoint composes hops so a projection walks to
    the front of the plan (where issue #43 wants to eventually turn it into a backend
    read plan).
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
        # crossed node, which is exact here because this loop stays inside the trusted
        # prefix. The subset test below then passes only when the projected variables span
        # every dim — exactly when the replayed bare reduce reduces the same ones.
        needed = resolve_dims(requires, schemas[i].dim_names)
        available = schemas[i].var_dims(project.variables)
        if available is None or not needed <= available:
            continue

        swapped = list(nodes)
        swapped[i], swapped[i + 1] = project, crossed
        return swapped
    return None


def pushdown_selects_past_rechunks(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a select left past a preceding ``chunk`` so the rechunk moves less data.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema entering the plan. Unused — this rule has no disjointness test at all.

    Returns
    -------
    Plan or None
        The plan with one select moved in front of a rechunk — the spec rebuilt or
        dropped as needed — or ``None`` when nothing moved.

    Notes
    -----
    A rechunk changes no dim, no size and no value — only chunk topology — so a select
    and a rechunk *always* commute as far as results go. What the rule protects is the
    chunking itself: selecting first can only leave dask with less data to shuffle, and
    the topology it lands on is no coarser than the eager order's. ``chunk({time: 100})``
    then ``isel(time=slice(50, 250))`` yields ragged ``(50, 100, 50)`` blocks; pushed, it
    yields fresh regular ``(100, 100)`` ones.

    The rewrite is not a plain swap, because a dim the select *drops* must also leave the
    chunk spec — xarray raises ``ValueError: chunks keys ('time',) not found in data
    dimensions`` otherwise. So, against the select's ``consumes``:

    - **no named dim dropped** — swap as-is. This covers the uniform forms that get this
      far, ``chunk()`` and ``chunk(100)`` / ``chunk(-1)``, which name no dim: xarray reads
      a uniform spec as ``dict.fromkeys(self.dims, spec)``, so it already means "whatever
      dims there are" and a select in front simply leaves fewer of them.
    - **some named dims dropped** — swap, rebuilding the rechunk from the surviving keys.
    - **every named dim dropped** — swap and *drop the rechunk*. What is left would be
      ``chunk({})``: a single-chunk dask array, so no parallelism and no out-of-core
      benefit — it would preserve dask-ness and nothing of value. (Note this is not the
      disk-chunk-aware ``open_dataset(chunks={})``; the *method* has no such knowledge.)
      A rechunk that named no dim to begin with is kept, since there the conversion is
      the stated purpose rather than a leftover.

    What the rule does **not** do is repair a spec it moved past. The specs that would need
    repairing — ``"auto"`` and a byte target, whose meaning changes with the extent — are
    refused by ``_pushable_rechunk`` instead, because measurement says no repair exists:
    naming the emptied dim ``"auto"``, xarray's own fix in `pydata/xarray#11486
    <https://github.com/pydata/xarray/pull/11486>`_, does not help, since dask's
    ``auto_chunks`` pins a zero-length auto dim to ``(0,)`` and *recurses*, and on the second
    pass it is no longer ``"auto"``. See :class:`~xrexpr.chunks.Auto`.

    Unlike :func:`pushdown_selects` this never raises: a rechunk cannot make a select
    unreplayable, only slower. One hop per call.
    """
    for i in range(len(nodes) - 1):
        match nodes[i], nodes[i + 1]:
            case (Rechunk() as rechunk, Select() as select) if _pushable_rechunk(
                rechunk
            ):
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


def _pushable_rechunk(node: Rechunk) -> bool:
    """Report whether a select may cross ``node``, or it acts as a barrier.

    Parameters
    ----------
    node : Rechunk
        The rechunk the select would hop in front of.

    Returns
    -------
    bool
        ``True`` when the select may cross, ``False`` for the barrier forms below.

    Notes
    -----
    **One check about the call header**, which no taxonomy can answer: **option kwargs**
    (``token``, ``chunked_array_type``, ...) barrier, because a rebuilt spec couldn't carry
    the option faithfully — the same reason ``_mergeable_select`` bails. It is a fact
    about how the call was written, not about what any spec is.

    **Then the specs**, both spellings through one ``match`` over
    :data:`~xrexpr.chunks.ChunkSpec` closed with ``assert_never``, so this being the *policy*
    site is enforced: a new variant fails mypy here until someone decides which side of the
    line it falls on. ``Rechunk`` classifies its uniform form too, so ``chunk("auto")``
    reaches this match as an :class:`~xrexpr.chunks.Auto` and ``chunk((100, 400, 500))`` as a
    :class:`~xrexpr.chunks.BlockSeq` — the same arms their mapping spellings take, rather
    than a pair of ``isinstance`` checks on ``args`` restating the answer.

    The line is whether **dask resolves the spec by measuring the array it is handed**:

    - **extent-dependent** specs barrier. An explicit block sequence must sum to the dim's
      length, so a select in front leaves a spec that cannot replay at all; ``"auto"`` and a
      byte target are resolved by *dividing* a byte budget by the array's own extent, so a
      select in front changes what they mean — and if it empties a dim, into a
      ``ZeroDivisionError`` where the eager chain succeeds (issue #121, and the notes on
      :class:`~xrexpr.chunks.Auto`). Someone writing either is already reasoning about
      chunking against the data as it stands, so nothing crosses such a call, scalar selects
      included.
    - **extent-independent** specs cross. A fixed size is that size on any array, ``-1`` is
      one block whatever its length, and ``None`` keeps whatever is there. None of the three
      can be invalidated by a select, which is what makes moving it in front safe.
    - an **unmodelled spec** barriers because the optimiser has said outright it cannot
      reason about the value; :class:`~xrexpr.chunks.OpaqueChunk` means the same here as
      :class:`~xrexpr.ir.Opaque` does one level up.

    Nothing here reads an extent, a size or the schema — the barrier follows from the spec
    alone, so :class:`~xrexpr.schema.SchemaState`'s "no rewrite reads a size" holds
    unqualified.

    ``specs`` is annotated, and that is load-bearing rather than decorative: gathering the
    two sites as ``(*node.chunks.values(), *uniform)`` infers ``Any``, which makes
    ``assert_never`` accept anything and voids the exhaustiveness guarantee **silently** —
    the probe technique reports success. Any rewrite of these three lines has to re-run that
    proof.
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
    """Report whether ``node`` is a pure mapping-form rechunk two of which may merge.

    Parameters
    ----------
    node : Rechunk
        The rechunk to test.

    Returns
    -------
    bool
        ``True`` when ``node`` names its dims (mapping form) and every spec it carries is
        one a merge can compose; ``False`` for the uniform, empty and barrier forms.

    Notes
    -----
    Two conditions, both load-bearing:

    - **Mapping form** — ``node.uniform is None`` *and* ``node.chunks`` is non-empty. A
      uniform positional spec (``chunk(100)``, ``chunk("auto")``, ``chunk((100, 400,
      500))``) rechunks *every* dim and does not compose by dict union, so it cannot merge
      with a per-dim mapping; and a bare ``chunk()`` names no dim while still touching all
      of them, so a union cannot capture its effect either. Both are left alone.
    - **Pushable** — :func:`_pushable_rechunk` holds, reused here as the single policy site.
      It barriers **option kwargs** (``token``, ...), which a rebuilt node cannot carry —
      the same faithfulness limit that makes the merged node drop ``kwargs`` — and the
      **extent-dependent specs** (``BlockSeq``/``Auto``/``ByteSize``/``OpaqueChunk``), so
      the specs a merge composes are confined to ``SingleSize``/``FullDim``/``NoChange``,
      whose :meth:`~xrexpr.chunks.SingleSize.to_raw` values round-trip through
      :func:`~xrexpr.chunks.classify_chunk`.
    """
    return node.uniform is None and bool(node.chunks) and _pushable_rechunk(node)


def merge_adjacent_rechunks(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Fuse an adjacent pair of mapping-form ``chunk`` calls into one.

    Parameters
    ----------
    nodes : Plan
        The plan to rewrite.
    schema : SchemaState
        The schema entering the plan. Unused — this rule is *syntactic*, deciding on the
        two nodes' own specs.

    Returns
    -------
    Plan or None
        The plan with one adjacent ``(Rechunk, Rechunk)`` pair replaced by a single merged
        node, or ``None`` when no pair merges.

    Notes
    -----
    ``chunk({"time": 100}).chunk({"lat": 50})`` records two :class:`~xrexpr.ir.Rechunk`
    nodes where one ``chunk({"time": 100, "lat": 50})`` suffices — shorter, and a strictly
    better input to :func:`pushdown_selects_past_rechunks` than two adjacent ones. Both
    nodes must be :func:`_mergeable_rechunk` (mapping form, pushable); uniform, empty and
    barrier forms are left where they are.

    Dask applies a later ``chunk`` per dim *on top of* an earlier one, so the merge is
    **later-wins** — with one exception. A later :class:`~xrexpr.chunks.NoChange`
    (``chunk({dim: None})``, "leave this dim as it is") must **not** override an earlier
    concrete spec on the same dim: ``chunk({"time": 100}).chunk({"time": None})`` keeps
    ``time`` at 100-blocks, where a plain ``{**r1, **r2}`` union would drop it to the base's
    single block (verified against dask). So a later ``NoChange`` fills a dim only when the
    earlier node left it unspecified, while every concrete later spec wins outright.

    The merged node is rebuilt exactly as :func:`pushdown_selects_past_rechunks` rebuilds a
    trimmed one — mapping form, ``kwargs`` dropped (safe, since both inputs were pushable
    and so carried no option kwarg). A leading ``Mapping`` in ``args`` makes
    ``Rechunk.__post_init__`` derive ``uniform = None``, so the result is again a clean
    mapping-form node the rule can merge further.

    One merge per call; :func:`optimize`'s fixpoint collapses a run of three. The rewrite
    replaces two nodes with one, shrinking the plan, so the termination measure holds on its
    first component — the same footing as :func:`merge_adjacent_projects`.
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
