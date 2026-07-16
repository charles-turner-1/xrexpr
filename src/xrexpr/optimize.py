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
past a whole run of reductions, and newly-adjacent selects then merge). Classifying the
non-disjoint conflict — invalid select vs. a scan the select must stay behind — lands in
PR 9 (#12). See ``docs/pr-plan.md``.
"""

from collections.abc import Callable

from frozendict import frozendict

from xrexpr.ir import OpNode

__all__ = ["optimize"]

Plan = list[OpNode]
Rule = Callable[[Plan], Plan]

#: ``isel``/``sel`` — the two select ops the merge rule folds.
_SELECTS = ("isel", "sel")


def optimize(nodes: Plan) -> Plan:
    """Rewrite ``nodes`` into an equivalent plan, applying every rule to a fixpoint.

    Each rule preserves the plan's result, so the returned plan replays to the same
    dataset as ``nodes`` — only cheaper. Termination relies on every rule being
    non-growing (each either shrinks the plan or leaves it unchanged).
    """
    plan = list(nodes)
    while True:
        rewritten = plan
        for rule in _RULES:
            rewritten = rule(rewritten)
        if rewritten == plan:
            return rewritten
        plan = rewritten


def merge_adjacent_selects(nodes: Plan) -> Plan:
    """Fold each run of consecutive same-op selects into one node.

    Consecutive ``isel``s (or ``sel``s) commute and compose, so
    ``ds.isel(time=0).isel(lat=1)`` becomes a single ``isel({time: 0, lat: 1})``.
    A mixed ``isel``/``sel`` run is left alone (different indexing semantics), and a
    select carrying *option* kwargs (``drop``/``method``/...) is treated as a barrier
    — it isn't folded, since a single merged indexer couldn't carry those faithfully.
    """
    out: Plan = []
    i, n = 0, len(nodes)
    while i < n:
        node = nodes[i]
        if not _mergeable_select(node):
            out.append(node)
            i += 1
            continue

        j = i + 1
        indexer = dict(node.indexer)
        consumes = set(node.consumes)
        while j < n and nodes[j].name == node.name and _mergeable_select(nodes[j]):
            indexer.update(nodes[j].indexer)
            consumes |= set(nodes[j].consumes)
            j += 1

        if j - i > 1:  # at least two selects folded
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
    return out


def _mergeable_select(node: OpNode) -> bool:
    """Whether ``node`` is a select fully described by its ``indexer`` (no option kwargs).

    A select whose kwargs carry keys beyond the indexed dims (``drop``, ``method``,
    ``missing_dims``, ``tolerance``) can't be folded into a bare indexer dict, so it
    acts as a merge barrier rather than being silently stripped of those options.
    """
    return node.name in _SELECTS and all(k in node.indexer for k in node.kwargs)


def pushdown_selects(nodes: Plan) -> Plan:
    """Hop a select left past an immediately-preceding reduce when their dims are disjoint.

    ``ds.mean("lat").isel(time=0)`` becomes ``ds.isel(time=0).mean("lat")`` — selecting
    *before* reducing shrinks the array the reduction scans. The swap is valid whenever
    the select touches no dim the reduce consumes (``time`` vs ``lat`` here). This reads
    ``reduce.kind``/``consumes`` off the ``OpNode``, so it generalises the demo's
    ``mean``/``sum``/``prod`` special-case to *any* reduce (``std``, ``max``, ...) —
    fixing the mean-only limitation (#1).

    One hop per call: the :func:`optimize` fixpoint re-applies it so a select bubbles
    past a whole run of reductions. A select sharing a dim with the reduce is left in
    place — whether that is an invalid expression or a scan the select must stay behind
    is classified in PR 9 (#12).
    """
    for i in range(len(nodes) - 1):
        reduce_node, select_node = nodes[i], nodes[i + 1]
        if (
            reduce_node.kind == "reduce"
            and select_node.name in _SELECTS
            and frozenset(select_node.indexer).isdisjoint(reduce_node.consumes)
        ):
            swapped = list(nodes)
            swapped[i], swapped[i + 1] = select_node, reduce_node
            return swapped
    return nodes


_RULES: tuple[Rule, ...] = (merge_adjacent_selects, pushdown_selects)
