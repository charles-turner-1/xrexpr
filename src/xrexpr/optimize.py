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

Each is a *local, single-step* rewrite; the fixpoint composes them (a select bubbles
past a whole run of reductions, and newly-adjacent selects then merge). ``pushdown_selects``
handles each ``(reduce, select)`` adjacency by set algebra — disjoint dims → swap,
intersecting dims → :class:`InvalidExpressionError` — and leaves anything else (PR 9, #12).
See ``docs/pr-plan.md``.
"""

from collections.abc import Callable, Hashable, Mapping
from typing import Any, TypeGuard

import numpy as np
from frozendict import frozendict

from xrexpr.exceptions import InvalidExpressionError
from xrexpr.ir import Op, Reduce, Select

__all__ = ["optimize"]

Plan = list[Op]
#: A rule maps a plan to a rewritten one, or returns ``None`` when it changes nothing
#: (letting :func:`optimize` detect the fixpoint without a full-plan equality compare).
Rule = Callable[[Plan], Plan | None]


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


def _compose_into(
    name: str,
    outer: dict[Hashable, Any],
    inner: Mapping[Hashable, Any],
) -> dict[Hashable, Any] | None:
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
        if composed is _UNCOMPOSABLE:
            return None
        merged[dim] = composed
    return merged


#: Sentinel for "these two indexers have no statically-known composition" — distinct
#: from ``None``, which is itself a legitimate slice bound.
_UNCOMPOSABLE: Any = object()


def _compose_indexer(outer: Any, inner: Any) -> Any:
    """Compose two positional indexers applied to the same dim, ``outer`` then ``inner``.

    Returns the single equivalent indexer, or :data:`_UNCOMPOSABLE` when no composition
    is provable without the dim's length (which the optimiser doesn't carry). Handled:

    - **sequence then anything** — ``outer`` is a concrete list of positions, so the
      answer is just indexing it: ``[10, 20, 30]`` then ``1`` is ``20``.
    - **forward slice then forward slice / non-negative scalar** — arithmetic on the
      bounds, e.g. ``slice(100, 1000)`` then ``slice(10, 20)`` → ``slice(110, 120)``.

    Deliberately *not* handled (all yield :data:`_UNCOMPOSABLE`): negative starts, stops
    or steps, which index from the end and so need the length; boolean masks; and a
    scalar ``outer``, which drops the dim entirely so that ``inner`` cannot be replayed
    against it at all.
    """
    if isinstance(outer, np.ndarray) and outer.dtype != bool:
        outer = outer.tolist()
    if isinstance(outer, list | tuple) and all(isinstance(x, int) for x in outer):
        return _index_sequence(list(outer), inner)
    if isinstance(outer, slice):
        return _compose_slice(outer, inner)
    return _UNCOMPOSABLE


def _index_sequence(positions: list[int], inner: Any) -> Any:
    """Apply ``inner`` to a concrete list of positions, i.e. ``positions[inner]``.

    Exact by construction — the outer selection is already fully enumerated, so there
    is nothing to reason about. An out-of-range ``inner`` would raise here; that is
    reported as uncomposable so the error surfaces from xarray at replay, in its own
    words, rather than from the optimiser.
    """
    try:
        if isinstance(inner, slice):
            return positions[inner]
        if isinstance(inner, int) and not isinstance(inner, bool):
            return positions[inner]
        if isinstance(inner, list | tuple) and all(isinstance(x, int) for x in inner):
            return [positions[i] for i in inner]
    except IndexError:
        return _UNCOMPOSABLE
    return _UNCOMPOSABLE


def _compose_slice(outer: slice, inner: Any) -> Any:
    """Compose a forward, non-negative ``outer`` slice with ``inner``.

    Element ``k`` of the result is ``outer_start + (inner_start + k * inner_step) *
    outer_step``, which is itself an arithmetic progression — so a slice composed with a
    slice is a slice, and with a scalar is a scalar. The stop bound is the *tighter* of
    the two constraints: ``inner`` cannot run past its own stop, nor past the end of
    what ``outer`` produced.
    """
    if not _is_forward_slice(outer):
        return _UNCOMPOSABLE
    start, step = outer.start or 0, outer.step or 1

    if isinstance(inner, int) and not isinstance(inner, bool):
        if inner < 0:
            return _UNCOMPOSABLE
        position = start + inner * step
        # Out of ``outer``'s range: xarray raises, but the composed scalar might well
        # be a valid position in the full dim, which would silently return data instead.
        if outer.stop is not None and position >= outer.stop:
            return _UNCOMPOSABLE
        return position

    if isinstance(inner, slice) and _is_forward_slice(inner):
        inner_start, inner_step = inner.start or 0, inner.step or 1
        candidates = (outer.stop, _scaled_stop(inner.stop, start, step))
        stops = [s for s in candidates if s is not None]
        return slice(
            start + inner_start * step,
            min(stops) if stops else None,
            step * inner_step,
        )
    return _UNCOMPOSABLE


def _is_forward_slice(s: slice) -> bool:
    """Whether ``s`` steps forward from non-negative bounds, so it needs no dim length.

    A negative bound or step counts from the end of the dim, which the optimiser cannot
    resolve without knowing how long the dim is.
    """
    if s.step is not None and (not isinstance(s.step, int) or s.step < 1):
        return False
    return all(b is None or (isinstance(b, int) and b >= 0) for b in (s.start, s.stop))


def _scaled_stop(inner_stop: int | None, start: int, step: int) -> int | None:
    """``inner``'s stop expressed as a position in the *original* dim, or ``None``."""
    return None if inner_stop is None else start + inner_stop * step


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


_RULES: tuple[Rule, ...] = (merge_adjacent_selects, pushdown_selects)
