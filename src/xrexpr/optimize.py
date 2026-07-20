"""The plan optimiser: rewrite a linear ``OpNode`` plan to a cheaper equivalent.

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

Each is a *local, single-step* rewrite; the fixpoint composes them (a select bubbles
past a whole run of reductions, and newly-adjacent selects then merge). ``pushdown_selects``
classifies each adjacency by the report's validity **trichotomy** — disjoint → swap,
consumed dim → :class:`InvalidExpressionError`, scan dim → leave (PR 9, #12). See
``docs/pr-plan.md``.
"""

from collections.abc import Callable

from frozendict import frozendict

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.ir import OpNode

__all__ = ["optimize"]

Plan = list[OpNode]
#: A rule maps a plan to a rewritten one, or returns ``None`` when it changes nothing
#: (letting :func:`optimize` detect the fixpoint without a full-plan equality compare).
Rule = Callable[[Plan], Plan | None]

#: ``isel``/``sel`` — the two select ops the merge rule folds.
_SELECTS = ("isel", "sel")


def optimize(nodes: Plan) -> Plan:
    """Rewrite ``nodes`` into an equivalent plan, applying every rule to a fixpoint.

    Each rule preserves the plan's result, so the returned plan replays to the same
    dataset as ``nodes`` — only cheaper. Termination relies on every rule being
    non-growing (each either shrinks the plan or leaves it unchanged). A rule returns
    ``None`` when it changes nothing, so the loop detects the fixpoint from that signal
    rather than by comparing whole plans each pass.
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
        # ``consumes`` is accumulated in step with ``indexer`` and must stay consistent
        # with it: ``indexer.update`` keeps the last value for a dim while this unions,
        # so a re-indexed dim could disagree. The two hold today only because xarray
        # rejects re-indexing a dim a prior scalar select already dropped, upstream of
        # recording — a non-local invariant. (The sum-type refactor makes ``consumes``
        # a derived property of ``indexer``, so it can no longer drift.)
        consumes = set(node.consumes)
        while j < n and nodes[j].name == node.name and _mergeable_select(nodes[j]):
            indexer.update(nodes[j].indexer)
            consumes |= set(nodes[j].consumes)
            j += 1

        if j - i > 1:  # at least two selects folded
            folded = True
            out.append(
                OpNode(
                    name=node.name,
                    kind="select",
                    args=(dict(indexer),),
                    indexer=frozendict(indexer),
                    consumes=frozenset(consumes),
                )
            )
        else:
            out.append(node)
        i = j
    return out if folded else None


def _mergeable_select(node: OpNode) -> bool:
    """Whether ``node`` is a select fully described by its ``indexer`` (no option kwargs).

    A select whose kwargs carry keys beyond the indexed dims (``drop``, ``method``,
    ``missing_dims``, ``tolerance``) can't be folded into a bare indexer dict, so it
    acts as a merge barrier rather than being silently stripped of those options.
    """
    return node.name in _SELECTS and all(k in node.indexer for k in node.kwargs)


def pushdown_selects(nodes: Plan) -> Plan | None:
    """Hop a select left past a preceding reduce when their dims permit.

    The rule only *fires* on a `(reduce, select)` adjacency — that structural test is
    the one thing this shares with pattern matching. What to do once it fires is decided
    by **set algebra** on the select's dims vs the reduce's ``consumes``, not by any
    further dispatch:

    - **disjoint dims** — the select touches no reduced dim, so the swap is valid:
      ``ds.mean("lat").isel(time=0)`` → ``ds.isel(time=0).mean("lat")`` (select first, a
      smaller array to reduce). Keying off ``kind`` generalises this to *any* reduce
      (``std``/``max``/...), not just a hard-coded ``mean``/``sum``/``prod`` list (#1).
    - **intersecting dims** — the select indexes a dim the reduce already removed, so the
      expression can never replay (``mean("lon").isel(lon=0)``, and the all-dims
      ``mean().isel(time=0)``): raise :class:`InvalidExpressionError` rather than emit a
      silently-wrong reorder (the empty-dim reorder bug).

    Any other adjacency is simply left. Scans are the salient case: a ``cumsum``/``diff``
    is ``kind == "scan"``, not ``reduce``, so this rule never fires on it —
    ``cumsum("time").isel(time=5)`` is left untouched (order matters), neither swapped nor
    raised.

    One hop per call, returning the rewritten plan; ``None`` when no adjacency swaps.
    :func:`optimize`'s fixpoint composes hops so a select reaches the front of a run of
    reductions (and adjacent selects then merge).
    """
    for i in range(len(nodes) - 1):
        reduce_node, select_node = nodes[i], nodes[i + 1]
        if reduce_node.kind != "reduce" or select_node.name not in _SELECTS:
            continue

        select_dims = frozenset(select_node.indexer)
        if select_dims.isdisjoint(reduce_node.consumes):
            swapped = list(nodes)
            swapped[i], swapped[i + 1] = select_node, reduce_node
            return swapped

        shared = sorted(str(d) for d in select_dims & reduce_node.consumes)
        raise InvalidExpressionError(
            f"{select_node.name}() indexes {shared}, "
            f"which {reduce_node.name}() has already reduced away"
        )
    return None


_RULES: tuple[Rule, ...] = (merge_adjacent_selects, pushdown_selects)
