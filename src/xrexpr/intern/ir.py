"""The *interned* IR records: every op's *names* replaced by handles, its structure kept.

This is the form a plan takes when it's handed to a Rust backend. Rust can't hash or
compare arbitrary Python objects, so dim and variable **names** are replaced by
:class:`~xrexpr.intern.interner.InternedVal` handles. But the Rust side needs the value
types (:data:`~xrexpr.indexers.Indexer`, :data:`~xrexpr.chunks.ChunkSpec`) as real
structure, since they map onto Rust enums, and it does arithmetic on the numbers inside
them. So interning is **selective**:

- **Names** (dims, variables) become :class:`~xrexpr.intern.interner.InternedVal`.
- **Values** (positions, bounds, sizes, labels, bools) and the ``Indexer``/``ChunkSpec``
  wrappers are kept as they are. The one name inside a value is ``Advanced.dims``, which
  :func:`~xrexpr.intern.converters._intern_indexer` interns.
- **Derived flags** the fluent node computes from its verbatim call (``keepdims``,
  ``single``, ``uniform``) are stored on the interned record as typed fields, so the Rust
  side never parses ``args``/``kwargs``.
- **``args``/``kwargs``** stay verbatim. They're the Python replay header
  (``lower._emit_node`` reuses them, and they carry options like ``skipna`` that live
  nowhere else). They're marked :data:`~xrexpr.payload.Payload`: Rust carries them but
  never reads or changes them.

A handle is its own *type*, so it can't be confused with a plain ``int``: de-interning
looks up each ``InternedVal`` and leaves a bare ``int`` alone. :class:`~xrexpr.ir.AllDims`
is kept as the sentinel.

Every ``FluentOp | LoweredOp`` kind has a record here. :func:`~xrexpr.intern.intern` and
:func:`~xrexpr.intern.deintern` close their ``match`` with ``assert_never``, so a new op
kind is a type error until both directions handle it.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from frozendict import frozendict

from xrexpr.chunks import ChunkSpec
from xrexpr.indexers import Indexer
from xrexpr.intern.interner import InternedVal
from xrexpr.ir import AllDims, ContextOpenName
from xrexpr.payload import Payload

# Mirror the Rust Enum DimSet for readability.
DimSet = frozenset[InternedVal] | AllDims

__all__ = [
    "InternedContextOpen",
    "InternedDrop",
    "InternedElementwise",
    "InternedFluentOp",
    "InternedGroupedReduce",
    "InternedLoweredOp",
    "InternedOp",
    "InternedOpaque",
    "InternedProject",
    "InternedRechunk",
    "InternedReduce",
    "InternedRename",
    "InternedScan",
    "InternedSelect",
    "InternedWeightedReduce",
    "InternedWindowedReduce",
]


@dataclass(frozen=True)
class InternedReduce:
    """Interned :class:`~xrexpr.ir.Reduce` — ``consumes`` relabeled, ``keepdims`` materialized."""

    name: str
    consumes: DimSet = frozenset()
    keepdims: bool = False
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedSelect:
    """Interned :class:`~xrexpr.ir.Select` — ``indexer`` keys relabeled, values kept structural."""

    name: Literal["isel", "sel"]
    indexer: frozendict[InternedVal, Indexer] = field(default_factory=frozendict)
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedScan:
    """Interned :class:`~xrexpr.ir.Scan` — ``dims`` relabeled, ``AllDims`` preserved."""

    name: Literal["cumsum", "cumprod", "diff"]
    dims: DimSet = frozenset()
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedElementwise:
    """Interned :class:`~xrexpr.ir.Elementwise` — no names; just the replay header."""

    name: str
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedProject:
    """Interned :class:`~xrexpr.ir.Project` — ``variables`` relabeled, ``single`` materialized."""

    name: Literal["__getitem__"]
    variables: tuple[InternedVal, ...] = ()
    single: bool = False
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedRechunk:
    """Interned :class:`~xrexpr.ir.Rechunk` — ``chunks`` keys relabeled, ``uniform`` materialized."""

    name: Literal["chunk"]
    chunks: frozendict[InternedVal, ChunkSpec] = field(default_factory=frozendict)
    uniform: ChunkSpec | None = None
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedOpaque:
    """Interned :class:`~xrexpr.ir.Opaque` — no names; the replay header may hold arrays."""

    name: str
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedDrop:
    """Interned :class:`~xrexpr.ir.Drop` — ``variables`` relabeled per name."""

    name: Literal["drop_vars"]
    variables: tuple[InternedVal, ...] = ()
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedRename:
    """Interned :class:`~xrexpr.ir.Rename` — both sides of ``mapping`` relabeled.

    We have no need for args or kwargs when we drop to rust, as mapping is interned
    and contains all the info we need. We keep them purely so that we can replay the
    original call verbatim.
    """

    name: Literal["rename"]
    mapping: frozendict[InternedVal, InternedVal] = field(default_factory=frozendict)
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedContextOpen:
    """Interned :class:`~xrexpr.ir.ContextOpen` — fluent-only; never survives lowering."""

    name: ContextOpenName
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedGroupedReduce:
    """Interned :class:`~xrexpr.ir.GroupedReduce` — ``group_dim``/``new_dim``/``consumes`` relabeled."""

    name: Literal["groupby", "groupby_bins", "resample"]
    group_dim: InternedVal
    new_dim: InternedVal
    reduce: str
    consumes: frozenset[InternedVal] = frozenset()
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)
    reduce_args: Payload[tuple[Any, ...]] = ()
    reduce_kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedWindowedReduce:
    """Interned :class:`~xrexpr.ir.WindowedReduce` — ``window`` keys relabeled, sizes kept."""

    name: Literal["rolling", "coarsen"]
    reduce: str
    window: frozendict[InternedVal, int] = field(default_factory=frozendict)
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)
    reduce_args: Payload[tuple[Any, ...]] = ()
    reduce_kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedWeightedReduce:
    """Interned :class:`~xrexpr.ir.WeightedReduce` — ``weight_dims``/``consumes`` relabeled, weights verbatim."""

    name: Literal["weighted"]
    reduce: str
    weight_dims: frozenset[InternedVal] = frozenset()
    consumes: DimSet = frozenset()
    args: Payload[tuple[Any, ...]] = ()
    kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)
    reduce_args: Payload[tuple[Any, ...]] = ()
    reduce_kwargs: Payload[frozendict[str, Any]] = field(default_factory=frozendict)


#: The interned counterpart of :data:`~xrexpr.ir.Op`.
InternedOp = (
    InternedReduce
    | InternedSelect
    | InternedScan
    | InternedElementwise
    | InternedProject
    | InternedDrop
    | InternedRename
    | InternedRechunk
    | InternedOpaque
)

#: The interned counterpart of :data:`~xrexpr.ir.FluentOp`.
InternedFluentOp = InternedOp | InternedContextOpen

#: The interned counterpart of :data:`~xrexpr.ir.LoweredOp`.
InternedLoweredOp = (
    InternedOp | InternedGroupedReduce | InternedWindowedReduce | InternedWeightedReduce
)
