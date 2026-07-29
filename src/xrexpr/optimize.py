"""The plan optimiser: rewrite a linear :data:`~xrexpr.ir.LoweredOp` plan to a cheaper equivalent.

:func:`optimize` runs a set of local rewrite *rules* to a **fixpoint** — each rule
maps a plan to an equivalent plan, and the loop reapplies the whole set until the
plan stops changing. Local rules + a fixpoint let a small rewrite (a select moving
one hop) compose into a global one (the select reaching the front) without any rule
having to reason about the whole chain.

Rules (in ``_RULES``):

- :func:`merge_adjacent_selects` — fold a run of consecutive ``isel``/``isel`` (or
  ``sel``/``sel``) into a single indexer, *composing* rather than overwriting when both
  select the same dim.
- :func:`pushdown_selects` — hop a select left past **any** node whose dims permit it, so
  the work behind it runs on less data: past a plain reduce, and past a grouped or
  windowed one, which is what lowering's fused nodes exist to make expressible —
  ``groupby("time.month").mean().isel(lat=0)`` groups one latitude instead of all of them.
- :func:`pushdown_projections` — hop a variable projection left past a preceding reduce,
  select or fused reduce (weighted included), so only the needed variables flow through
  the plan. This also disarms a footgun: the discarded variables can no longer raise for
  lacking a dim they were never asked about.
- :func:`pushdown_selects_past_rechunks` — hop a select left past a preceding ``chunk``
  so the rechunk moves less data.

Each is a *local, single-step* rewrite; the fixpoint composes them (a select bubbles
past a whole run of reductions, and newly-adjacent selects then merge).

**One dispatch site for dim algebra.** The two pushdown rules above ask different
questions of the node they cross — what a *select* coming from the right must avoid, and
what a *projection* going to its left must supply — and both read the answer from
:func:`dim_effect`, a single ``match`` closed with ``assert_never``, so a new node kind
must state its dim effect before either rule will compile against it.
:func:`pushdown_selects_past_rechunks` deliberately stays outside it: it has no
disjointness test at all — a rechunk changes no dim, so a select *always* commutes with
one — and instead rewrites the spec it crosses.

**The schema, and how far it can be trusted.** Dim-level rules read almost everything they
need off the nodes themselves, but a *variable*-level rule can't: whether a projection may
cross an op depends on which dims the projected variables carry at that point in the plan.
So :func:`optimize` takes the **base** schema and :func:`_schemas` folds it forward to give
the schema each node sees — the plan's single fold, and also where a symbolic
:data:`~xrexpr.ir.ALL_DIMS` is resolved (:func:`_resolve_dims`).
:func:`~xrexpr.schema.apply_schema` models
:class:`~xrexpr.ir.Opaque` as variable-preserving, which is not true of ``rename`` or
``drop_vars``, so those folded schemas are exact only up to the first opaque node —
:func:`_trusted_prefix` marks that boundary and rules that consult ``data_vars`` stay
inside it.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard

from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.indexers import (
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
    GroupedReduce,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.schema import SchemaState, apply_schema

__all__ = ["optimize"]

#: What the rules rewrite: a **lowered** plan. Spelled with
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

    ``schema`` is the schema of the dataset the plan starts from — the *base*, not the
    one left at the end of recording — since rules need to know the shape each node
    sees, and rewriting changes that.

    Each rule preserves the plan's result, so the returned plan replays to the same
    dataset as ``nodes`` — only cheaper. A rule returns ``None`` when it changes nothing,
    so the loop detects the fixpoint from that signal rather than by comparing whole
    plans each pass.

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
    """The schema each node *sees*: ``out[i]`` is the schema entering ``nodes[i]``."""
    if not nodes:
        return []
    out = [base]
    for node in nodes[:-1]:
        out.append(apply_schema(out[-1], node))
    return out


def _resolve_dims(consumes: DimSet, entering: SchemaState) -> frozenset[Hashable]:
    """``consumes`` made concrete against the schema *entering* the node that carries it.

    :data:`~xrexpr.ir.ALL_DIMS` — a bare ``mean()`` — means every dim the op finds when
    it runs, which is exactly what ``entering`` records. Callers must be inside the
    trusted prefix, where that fold is exact rather than a guess.
    """
    return entering.dim_names if isinstance(consumes, AllDims) else consumes


def _trusted_prefix(nodes: Plan) -> int:
    """How far the folded schema is exact: the index of the first unmodelled op.

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

    #: Dims a select may not hop over, or ``None`` to refuse every hop.
    blocks: DimSet | None
    #: Dims the node needs of its input, or ``None`` to refuse every projection hop.
    requires: DimSet | None
    #: What a select hopping over :attr:`blocks` means. ``"invalid"`` — the node *removed*
    #: those dims, so the chain can never replay and the optimiser should say so.
    #: ``"immovable"`` — the dims survive in some form, so the chain is very likely valid
    #: and merely un-reorderable; leave it. Only meaningful when ``blocks`` is not ``None``.
    on_conflict: Literal["invalid", "immovable"] = "immovable"


#: Nothing crosses this node in either direction — the answer for every kind whose dim
#: effect the optimiser does not model (``Scan``, whose order matters; ``Opaque``, which
#: could do anything; ``Rechunk``, which has its own rule) and for ``Project``, which the
#: *other* rules handle.
_OPAQUE_EFFECT = DimEffect(blocks=None, requires=None)


def dim_effect(node: LoweredOp) -> DimEffect:
    """The dim effect of ``node``: one dispatch site, closed with ``assert_never``.

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
            # (``docs/roadmap/02-lowering.md`` §8.1).
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
        case Scan() | Project() | Rechunk() | Opaque():
            return _OPAQUE_EFFECT
        case _:
            assert_never(node)


def merge_adjacent_selects(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Fold each run of consecutive same-op selects into one node.

    Consecutive ``isel``s (or ``sel``s) compose, so ``ds.isel(time=0).isel(lat=1)``
    becomes a single ``isel({time: 0, lat: 1})``. Selects on *different* dims simply
    union; selects on the **same** dim are composed by :func:`_compose_into`, because
    the later indexer addresses positions within the earlier one's result rather than
    the original dim.

    Three things act as barriers, each ending the run so the plan keeps two correct
    nodes instead of collapsing to one wrong one: a mixed ``isel``/``sel`` run
    (different indexing semantics), a select carrying *option* kwargs
    (``drop``/``method``/...) that a bare merged indexer couldn't carry faithfully, and
    a same-dim collision with no statically provable composition.

    Returns ``None`` when no run was folded (nothing to change).
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
        # ``consumes`` is not accumulated here — it is a derived property of the
        # merged ``indexer`` on :class:`~xrexpr.ir.Select`, so it cannot drift from it.
        # ``args`` does mirror ``indexer`` and is
        # rebuilt from it below, which stays the merge rule's local responsibility.
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
            out.append(
                Select(
                    name=node.name,
                    args=({dim: v.to_raw() for dim, v in indexer.items()},),
                    indexer=frozendict(indexer),
                )
            )
        else:
            out.append(node)
        i = j
    return out if folded else None


def _mergeable_select(node: LoweredOp) -> TypeGuard[Select]:
    """Whether ``node`` is a select fully described by its ``indexer`` (no option kwargs).

    A select whose kwargs carry keys beyond the indexed dims (``drop``, ``method``,
    ``missing_dims``, ``tolerance``) can't be folded into a bare indexer dict, so it
    acts as a merge barrier rather than being silently stripped of those options. A
    ``TypeGuard`` so callers narrow ``node`` to :class:`~xrexpr.ir.Select`.
    """
    return isinstance(node, Select) and all(k in node.indexer for k in node.kwargs)


def _compose_into(
    name: str,
    outer: dict[Hashable, Indexer],
    inner: Mapping[Hashable, Indexer],
) -> dict[Hashable, Indexer] | None:
    """Merge ``inner``'s indexers into ``outer``'s, composing where dims collide.

    A dim in only one of the two carries over untouched. A dim in **both** is the case
    the plain ``dict.update`` got wrong: ``isel(time=slice(100,1000)).isel(time=slice(10,20))``
    is *not* ``isel(time=slice(10,20))`` — the second indexer addresses positions
    **within the first's result**, so the two must compose (here to ``slice(110,120)``).
    :func:`_compose_indexer` does that for the cases it can prove; ``None`` from either
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

    Returns the single equivalent :data:`~xrexpr.indexers.Indexer`, or ``None`` when no
    composition is provable without the dim's length (which the optimiser doesn't carry).
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
        case _:
            assert_never(outer)


def _index_sequence(positions: tuple[int, ...], inner: Indexer) -> Indexer | None:
    """Apply ``inner`` to a concrete tuple of positions, i.e. ``positions[inner]``.

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

    ``addressed`` is every element of ``outer`` that ``inner`` reads; ``kept`` is the subset
    it actually selects (they differ only for a mask). The check runs over *addressed*
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
    """``inner``'s stop expressed as a position in the *original* dim, or ``None``."""
    return None if inner_stop is None else start + inner_stop * step


def pushdown_selects(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a select left past a preceding node whose dims permit it.

    The structural test is only *which* adjacency fires; what happens once it does is
    **set algebra** on the select's dims against the crossed node's
    :attr:`DimEffect.blocks`, plus that node's answer to what an overlap *means*:

    - **disjoint dims** — the select touches nothing the node consumed, minted or resized,
      so the swap is valid and strictly cheaper. ``ds.mean("lat").isel(time=0)`` becomes
      ``ds.isel(time=0).mean("lat")``, and ``ds.groupby("time.month").mean().isel(lat=0)``
      runs the climatology over one latitude — the pattern this project exists for.
    - **overlapping dims, ``on_conflict="invalid"``** — a plain reduce *removed* those
      dims, so the chain can never replay (``mean("lon").isel(lon=0)``, and the all-dims
      ``mean().isel(time=0)``): raise :class:`InvalidExpressionError` rather than emit a
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

    One hop per call, returning the rewritten plan; ``None`` when no adjacency swaps.
    :func:`optimize`'s fixpoint composes hops so a select reaches the front of a run of
    reductions (and adjacent selects then merge).
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

    The dims come from ``data_vars`` in the schema entering the crossed op (:func:`_schemas`),
    i.e. the variables' dims *before* it — the post-op dims would report ``tas`` as lacking
    ``time`` and block the very case this rule exists for. Two conservative edges:
    a name that isn't a known data variable (a coordinate, or something an unmodelled op
    introduced) blocks the hop, as does a dim carried only by a coordinate rather than by
    a projected variable.

    Reductions and selections act per variable, so a swap leaves the surviving variables'
    values untouched. A bare ``mean()`` carries :data:`~xrexpr.ir.ALL_DIMS`, resolved by
    :func:`_resolve_dims` against that same entering schema — so the subset test only
    passes when the projected variables span every dim, which is exactly when the
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
    ``docs/roadmap/07-small-wins.md`` §8.

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

    One hop per call, returning the rewritten plan; ``None`` when nothing moves.
    :func:`optimize`'s fixpoint composes hops so a projection walks to the front of the
    plan (where issue #43 wants to eventually turn it into a backend read plan).
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
        needed = _resolve_dims(requires, schemas[i])
        available = schemas[i].var_dims(project.variables)
        if available is None or not needed <= available:
            continue

        swapped = list(nodes)
        swapped[i], swapped[i + 1] = project, crossed
        return swapped
    return None


def pushdown_selects_past_rechunks(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a select left past a preceding ``chunk`` so the rechunk moves less data.

    A rechunk changes no dim, no size and no value — only chunk topology — so a select
    and a rechunk *always* commute as far as results go. What the rule protects is the
    chunking itself: selecting first can only leave dask with less data to shuffle, and
    the topology it lands on is no coarser than the eager order's. ``chunk({time: 100})``
    then ``isel(time=slice(50, 250))`` yields ragged ``(50, 100, 50)`` blocks; pushed, it
    yields fresh regular ``(100, 100)`` ones.

    The rewrite is not a plain swap, because a dim the select *drops* must also leave the
    chunk spec — xarray raises ``ValueError: chunks keys ('time',) not found in data
    dimensions`` otherwise. So, against the select's ``consumes``:

    - **no named dim dropped** — swap as-is (this covers the uniform forms, ``chunk()`` /
      ``chunk(100)`` / ``chunk("auto")``, which name no dim at all; ``"auto"`` simply
      re-picks block sizes against whatever survives, which is the point of asking for it).
    - **some named dims dropped** — swap, rebuilding the rechunk from the surviving keys.
    - **every named dim dropped** — swap and *drop the rechunk*. What is left would be
      ``chunk({})``: a single-chunk dask array, so no parallelism and no out-of-core
      benefit — it would preserve dask-ness and nothing of value. (Note this is not the
      disk-chunk-aware ``open_dataset(chunks={})``; the *method* has no such knowledge.)
      A rechunk that named no dim to begin with is kept, since there the conversion is
      the stated purpose rather than a leftover.

    Unlike :func:`pushdown_selects` this never raises: a rechunk cannot make a select
    unreplayable, only slower. One hop per call; ``None`` when nothing moved.
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
                        Rechunk(
                            name=rechunk.name,
                            args=(dict(kept),),
                            chunks=frozendict(kept),
                        ),
                    ]
                else:  # the spec is spent
                    moved = [select]
                return list(nodes[:i]) + moved + list(nodes[i + 2 :])
    return None


def _pushable_rechunk(node: Rechunk) -> bool:
    """Whether a select may cross ``node``, or it acts as a barrier.

    Two forms are barriers:

    - an **explicit block sequence** (``chunk({"time": (100, 400, 500)})``, or a
      positional sequence), whose blocks must sum to the dim's length. A slice select
      moving in front would shrink the dim and leave a spec that cannot replay at all —
      and someone writing explicit block sizes is already reasoning about chunking, so
      nothing crosses such a call, scalar selects included.
    - **option kwargs** (``token``, ``chunked_array_type``, ...), which a rebuilt spec
      couldn't carry faithfully — the same reason :func:`_mergeable_select` bails.
    """
    if any(key not in node.chunks for key in node.kwargs):
        return False
    if node.args and isinstance(node.args[0], list | tuple):
        return False
    return not any(isinstance(spec, list | tuple) for spec in node.chunks.values())


_RULES: tuple[Rule, ...] = (
    merge_adjacent_selects,
    pushdown_selects,
    pushdown_projections,
    pushdown_selects_past_rechunks,
)
