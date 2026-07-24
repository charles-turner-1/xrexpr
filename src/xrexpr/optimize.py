"""The plan optimiser: rewrite a linear :data:`~xrexpr.ir.Op` plan to a cheaper equivalent.

:func:`optimize` runs a set of local rewrite *rules* to a **fixpoint** — each rule
maps a plan to an equivalent plan, and the loop reapplies the whole set until the
plan stops changing. Local rules + a fixpoint let a small rewrite (a select moving
one hop) compose into a global one (the select reaching the front) without any rule
having to reason about the whole chain.

Rules (in ``_RULES``):

- :func:`merge_adjacent_selects` — fold a run of consecutive ``isel``/``isel`` (or
  ``sel``/``sel``) into a single indexer, *composing* rather than overwriting when both
  select the same dim (PR 7, #33).
- :func:`pushdown_selects` — hop a select left past a preceding reduce when their dims
  are disjoint, so the reduction scans a smaller array (PR 8, #11).
- :func:`pushdown_projections` — hop a variable projection left past a preceding reduce
  or select, so only the needed variables flow through the plan (#32).
- :func:`pushdown_selects_past_rechunks` — hop a select left past a preceding ``chunk``
  so the rechunk moves less data (#57).

Each is a *local, single-step* rewrite; the fixpoint composes them (a select bubbles
past a whole run of reductions, and newly-adjacent selects then merge). ``pushdown_selects``
handles each ``(reduce, select)`` adjacency by set algebra — disjoint dims → swap,
intersecting dims → :class:`InvalidExpressionError` — and leaves anything else (PR 9, #12).
See ``docs/pr-plan.md``.

**The schema, and how far it can be trusted.** Dim-level rules read everything they need
off the nodes themselves, but a *variable*-level rule can't: whether a projection may cross
an op depends on which dims the projected variables carry at that point in the plan. So
:func:`optimize` takes the **base** schema and :func:`_schemas` folds it forward to give
the schema each node sees. :func:`~xrexpr.schema.apply_schema` models
:class:`~xrexpr.ir.Opaque` as variable-preserving, which is not true of ``rename`` or
``drop_vars``, so those folded schemas are exact only up to the first opaque node —
:func:`_trusted_prefix` marks that boundary and rules that consult ``data_vars`` stay
inside it.
"""

from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import TypeGuard

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
from xrexpr.ir import Op, Opaque, Project, Rechunk, Reduce, Select
from xrexpr.schema import SchemaState, apply_schema

__all__ = ["optimize"]

Plan = list[Op]
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
    has to preserve. (The pushdown rules fire on disjoint adjacencies — ``(reduce,
    select)``, ``(reduce | select, project)`` and ``(rechunk, select)`` — so they can't
    undo one another.)
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
        # ``consumes`` is no longer accumulated here — it is a derived property of the
        # merged ``indexer`` on :class:`~xrexpr.ir.Select`, so it cannot drift from it
        # (the desync the flat record risked). ``args`` still mirrors ``indexer`` and is
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


def _mergeable_select(node: Op) -> TypeGuard[Select]:
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
    """Hop a select left past a preceding reduce when their dims permit.

    The rule only *fires* on a `(reduce, select)` adjacency — that structural test is
    the one thing this shares with pattern matching. What to do once it fires is decided
    by **set algebra** on the select's dims vs the reduce's ``consumes``, not by any
    further dispatch:

    - **disjoint dims** — the select touches no reduced dim, so the swap is valid:
      ``ds.mean("lat").isel(time=0)`` → ``ds.isel(time=0).mean("lat")`` (select first, a
      smaller array to reduce). Matching the :class:`~xrexpr.ir.Reduce` variant
      generalises this to *any* reduce (``std``/``max``/...), not just a hard-coded
      ``mean``/``sum``/``prod`` list (#1).
    - **intersecting dims** — the select indexes a dim the reduce already removed, so the
      expression can never replay (``mean("lon").isel(lon=0)``, and the all-dims
      ``mean().isel(time=0)``): raise :class:`InvalidExpressionError` rather than emit a
      silently-wrong reorder (the empty-dim reorder bug).

    Any other adjacency is simply left. Scans are the salient case: a ``cumsum``/``diff``
    is a :class:`~xrexpr.ir.Scan`, not a :class:`~xrexpr.ir.Reduce`, so the pattern never
    matches it — ``cumsum("time").isel(time=5)`` is left untouched (order matters),
    neither swapped nor raised.

    One hop per call, returning the rewritten plan; ``None`` when no adjacency swaps.
    :func:`optimize`'s fixpoint composes hops so a select reaches the front of a run of
    reductions (and adjacent selects then merge).
    """
    for i in range(len(nodes) - 1):
        match nodes[i], nodes[i + 1]:
            case (
                Reduce(consumes=consumes) as reduce_node,
                Select(indexer=indexer) as select_node,
            ):
                select_dims = frozenset(indexer)
                if select_dims.isdisjoint(consumes):
                    swapped = list(nodes)
                    swapped[i], swapped[i + 1] = select_node, reduce_node
                    return swapped

                shared = sorted(str(d) for d in select_dims & consumes)
                raise InvalidExpressionError(
                    f"{select_node.name}() indexes {shared}, "
                    f"which {reduce_node.name}() has already reduced away"
                )
    return None


def pushdown_projections(nodes: Plan, schema: SchemaState) -> Plan | None:
    """Hop a variable projection left past a preceding reduce or select.

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
    values untouched. A bare ``mean()`` needs no special case: its ``consumes`` is every
    current dim, so the subset test only passes when the projected variables span all of
    them — exactly when the verbatim replayed ``mean()`` reduces the same dims.

    One hop per call, returning the rewritten plan; ``None`` when nothing moves.
    :func:`optimize`'s fixpoint composes hops so a projection walks to the front of the
    plan (where #43 will eventually turn it into a backend read plan).
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
        needed: frozenset[Hashable]
        match crossed:
            case Reduce(consumes=consumes):
                needed = consumes
            case Select(indexer=indexer):
                needed = frozenset(indexer)
            case _:
                continue

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
