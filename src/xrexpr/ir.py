"""The expression IR: a single flat record type, :class:`OpNode`.

A Dataset method chain is *linear* (each op has exactly one input — the previous
dataset), so the IR is a **list** of ``OpNode``, not a tree. Each node carries
enough normalised metadata for the optimiser to decide whether two ops commute,
without re-inspecting raw call ``args``/``kwargs``.

PR 2 introduces the type only — nothing records or consumes ``OpNode`` yet.
``to_opnode`` (record-time resolution) and the optimiser land in later PRs;
see ``docs/pr-plan.md``.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any

from frozendict import frozendict

__all__ = ["KINDS", "OpNode", "frozendict"]

#: The op kinds the optimiser distinguishes (see the report's validity trichotomy).
KINDS = frozenset({"reduce", "scan", "select", "elementwise", "opaque"})


@dataclass(frozen=True)
class OpNode:
    """One recorded operation in a linear plan.

    ``args``/``kwargs``/``consumes``/``indexer`` are coerced to immutable
    containers (``frozendict`` for the mappings) on construction, so a node is
    fully hashable and safe to share between the recorded plan and its optimised
    copy — optimiser rules cannot mutate recorded call metadata in place.
    """

    name: str  # "mean", "isel", "sel", "cumsum", "__getitem__", ...
    kind: str  # one of KINDS
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    # dim names use xarray's ``Hashable`` (usually str); kwargs *names* are str.
    consumes: frozenset[Hashable] = frozenset()  # dims removed, vs the current schema
    indexer: frozendict[Hashable, Any] = field(
        default_factory=frozendict
    )  # selects: {dim: indexer}

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown op kind {self.kind!r}; expected one of {sorted(KINDS)}"
            )
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "consumes", frozenset(self.consumes))
        object.__setattr__(self, "indexer", frozendict(self.indexer))
