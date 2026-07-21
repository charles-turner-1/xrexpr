"""The plan optimiser: rewrite a linear :data:`~xrexpr.ir.Op` plan to a cheaper equivalent.

:func:`optimize` runs a set of local rewrite *rules* to a **fixpoint** — each rule
maps a plan to an equivalent plan, and the loop reapplies the whole set until the
plan stops changing. Local rules + a fixpoint let a small rewrite (a select moving
one hop) compose into a global one (the select reaching the front) without any rule
having to reason about the whole chain.

Rules (in ``_RULES``):

- :func:`merge_adjacent_selects` — fold a run of consecutive ``isel``/``isel`` (or
  ``sel``/``sel``) into a single indexer (PR 7).
- :func:`pushdown_selects` — hop a select left past a preceding reduce when their dims
  are disjoint, so the reduction scans a smaller array (PR 8, #11).
- :func:`pushdown_selects_past_rechunks` — hop a select left past a preceding ``chunk``
  so the rechunk moves less data (#57).

Each is a *local, single-step* rewrite; the fixpoint composes them (a select bubbles
past a whole run of reductions, and newly-adjacent selects then merge). ``pushdown_selects``
handles each ``(reduce, select)`` adjacency by set algebra — disjoint dims → swap,
intersecting dims → :class:`InvalidExpressionError` — and leaves anything else (PR 9, #12).
See ``docs/pr-plan.md``.
"""

from collections.abc import Callable
from typing import TypeGuard

from frozendict import frozendict

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.ir import Op, Rechunk, Reduce, Select

__all__ = ["optimize"]

Plan = list[Op]
#: A rule maps a plan to a rewritten one, or returns ``None`` when it changes nothing
#: (letting :func:`optimize` detect the fixpoint without a full-plan equality compare).
Rule = Callable[[Plan], Plan | None]


def optimize(nodes: Plan) -> Plan:
    """Rewrite ``nodes`` into an equivalent plan, applying every rule to a fixpoint.

    Each rule preserves the plan's result, so the returned plan replays to the same
    dataset as ``nodes`` — only cheaper. A rule returns ``None`` when it changes nothing,
    so the loop detects the fixpoint from that signal rather than by comparing whole
    plans each pass.

    **Termination.** Every rule strictly decreases the lexicographic measure
    ``(len(plan), sum of the indices of the Select nodes)``. Merging and dropping a
    spent rechunk shrink the plan; the pushdown rules leave the length alone but move a
    select strictly left. Neither component can grow, so no rule may ever push a select
    *right* or lengthen a plan — the invariant a new rule has to preserve.
    """
    plan = list(nodes)
    while True:
        changed = False
        for rule in _RULES:
            rewritten = rule(plan)
            if rewritten is not None:
                plan, changed = rewritten, True
        if not changed:
            return plan


def merge_adjacent_selects(nodes: Plan) -> Plan | None:
    """Fold each run of consecutive same-op selects into one node.

    Consecutive ``isel``s (or ``sel``s) commute and compose, so
    ``ds.isel(time=0).isel(lat=1)`` becomes a single ``isel({time: 0, lat: 1})``.
    A mixed ``isel``/``sel`` run is left alone (different indexing semantics), and a
    select carrying *option* kwargs (``drop``/``method``/...) is treated as a barrier
    — it isn't folded, since a single merged indexer couldn't carry those faithfully.

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
            indexer.update(nxt.indexer)
            j += 1

        if j - i > 1:  # at least two selects folded
            folded = True
            out.append(
                Select(
                    name=node.name,
                    args=(dict(indexer),),
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


def pushdown_selects(nodes: Plan) -> Plan | None:
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


def pushdown_selects_past_rechunks(nodes: Plan) -> Plan | None:
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
    pushdown_selects_past_rechunks,
)
