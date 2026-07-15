"""The expression IR: a single flat record type, :class:`OpNode`.

A Dataset method chain is *linear* (each op has exactly one input — the previous
dataset), so the IR is a **list** of ``OpNode``, not a tree. Each node carries
enough normalised metadata for the optimiser to decide whether two ops commute,
without re-inspecting raw call ``args``/``kwargs``.

PR 2 introduces the type only — nothing records or consumes ``OpNode`` yet.
``to_opnode`` (record-time resolution) and the optimiser land in later PRs;
see ``docs/pr-plan.md``.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

#: The op kinds the optimiser distinguishes (see the report's validity trichotomy).
KINDS = frozenset({"reduce", "scan", "select", "elementwise", "opaque"})


class frozendict(Mapping):
    """A minimal immutable, hashable mapping.

    Backs :attr:`OpNode.kwargs` / :attr:`OpNode.indexer` so recorded call
    metadata cannot be mutated in place by optimiser rules, and so ``OpNode``
    stays hashable (needed for golden-list equality and set/dict membership).
    """

    __slots__ = ("_data", "_hash")

    def __init__(self, *args: Any, **kwargs: Any):
        self._data: dict = dict(*args, **kwargs)
        self._hash: int | None = None

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __repr__(self) -> str:
        return f"frozendict({self._data!r})"


@dataclass(frozen=True)
class OpNode:
    """One recorded operation in a linear plan.

    ``args``/``kwargs``/``consumes``/``indexer`` are coerced to immutable
    containers on construction, so a node is fully hashable and safe to share
    between the recorded plan and its optimised copy.
    """

    name: str  # "mean", "isel", "sel", "cumsum", "__getitem__", ...
    kind: str  # one of KINDS
    args: tuple = ()
    kwargs: frozendict = field(default_factory=frozendict)
    consumes: frozenset[str] = frozenset()  # dims removed, vs the current schema
    indexer: frozendict = field(default_factory=frozendict)  # for selects: {dim: indexer}

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown op kind {self.kind!r}; expected one of {sorted(KINDS)}")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "consumes", frozenset(self.consumes))
        object.__setattr__(self, "indexer", frozendict(self.indexer))
