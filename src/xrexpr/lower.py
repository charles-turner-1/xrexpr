"""Lowering: what the user *wrote* → what it *means* → the calls that reproduce it.

Two pure stages either side of the optimiser, so the pipeline reads:

.. code-block:: text

    xarray calls → to_opnode → fluent IR → to_lower_ir → lowered IR
                → optimize → lowered IR → emit → [Call] → _replay → xr.Dataset

**Why a stage and not a rewrite rule.** :func:`~xrexpr.schema.to_opnode` is a *per-call*
function: handed one recorded call, it must return one node with no knowledge of what
follows. That is right for almost every xarray method — ``ds.mean("lat")`` is a
:class:`~xrexpr.ir.Reduce` and nothing later can change that — but xarray spells some
single semantic operations as **two** calls via builder objects
(``ds.groupby("time.month").mean()``), and no per-call function can see the pair.
:func:`to_lower_ir` runs over the *finished* plan, so it has the lookahead none of the
earlier workarounds had.

It cannot be one of ``optimize``'s rules either. Those run to a shared fixpoint under a
strictly-decreasing measure; fusion runs **once**, and is a *precondition* of the rules
rather than one of them — a rule matching a fused node cannot be correct on a plan where
that node is still two. So :func:`to_lower_ir` is sequenced before ``optimize``, with its
own contract:

    :func:`to_lower_ir` is **semantics-preserving** — ``emit(to_lower_ir(p))`` replays to
    the same result as ``p`` — and **idempotent**: applied to an already-lowered plan it
    returns it unchanged.

**Today both stages are identities**, because no fused node kinds exist yet: this module
is the seam they will arrive through, and the pipeline is rewired around it first so that
each fused node is built once rather than twice. :func:`emit` is the piece that makes that
possible — a lowered node may stand for *several* calls, which a one-node-one-call replay
loop could never express.
"""

from dataclasses import dataclass, field
from typing import Any

from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr.ir import (
    FluentOp,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
)

__all__ = ["Call", "emit", "to_lower_ir"]


@dataclass(frozen=True)
class Call:
    """One xarray method invocation — the unit replay actually performs.

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


def to_lower_ir(nodes: list[FluentOp]) -> list[LoweredOp]:
    """Translate a recorded plan into what it means, fusing builder chains.

    Where the fluent API is one-to-one with semantics the recorded node is already
    right, so it passes through untouched — this is the *same* vocabulary plus the nodes
    a single call cannot express, not a translation into a poorer language. Only the
    multi-call spellings need rewriting, and there are none yet, so this is currently the
    identity.

    Idempotent and semantics-preserving (see the module docstring). The identity trivially
    satisfies both; the property tests assert them anyway, so the contract is already
    pinned when the first fusion rule lands.
    """
    return list(nodes)


def emit(nodes: list[LoweredOp]) -> list[Call]:
    """Generate the call sequence that reproduces a lowered plan.

    The inverse direction of :func:`to_lower_ir`: a node that stands for two calls emits
    two. A pure function of the plan — no dataset in sight — so codegen is unit-testable
    on its own, and :meth:`~xrexpr.accessor.LazyDatasetProxy._replay` stays the short
    ``getattr`` loop it has always been instead of growing an arm per variant. That is
    what keeps per-call verbatim-ness a property of the whole pipeline rather than
    something each node kind has to be trusted with.
    """
    return [call for node in nodes for call in _emit_node(node)]


def _emit_node(node: LoweredOp) -> tuple[Call, ...]:
    """The calls one lowered node stands for.

    Every node in today's vocabulary is one call, and reproduces it from the verbatim
    header the recorder kept — so an unmodified plan emits exactly the calls the user
    wrote, spelling included. A rule that *synthesises* a node is responsible for its
    header, as ``merge_adjacent_selects`` already is; moving that reconstruction here is
    a follow-up (``docs/roadmap/02-lowering.md`` §7), deliberately not taken in the PR
    that rewires the pipeline, since it would change rewrite output rather than leave it
    identical.

    ``assert_never`` closes the match: a new lowered variant fails type-check here until
    someone says what it replays as, which is exactly the question a fused node exists to
    answer.
    """
    match node:
        case Reduce() | Select() | Scan() | Project() | Rechunk() | Opaque():
            return (Call(name=node.name, args=node.args, kwargs=node.kwargs),)
        case _:
            assert_never(node)
