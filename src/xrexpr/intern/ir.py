"""The *interned* IR — every op's *names* relabeled to handles, its *structure* preserved.

The optimiser reasons over the fluent/lowered IR (``ir.py``); this module is the last step
before that plan crosses to a Rust backend that does the reasoning itself. Rust cannot hash
or compare arbitrary Python objects, so dim and variable **names** are relabeled to
:class:`~xrexpr.intern.interner.InternedVal` handles. But the reasoner needs the closed value
sum-types — :data:`~xrexpr.indexers.Indexer`, :data:`~xrexpr.chunks.ChunkSpec` — as
*structure* (they exist to map onto Rust enums), and it does arithmetic on the literal
values inside them (slice bounds, positions, block sizes). So interning is **selective**:

- **Names** (dims, variables) become :class:`~xrexpr.intern.interner.InternedVal`.
- **Values** (positions, bounds, sizes, labels, bools) and the ``Indexer``/``ChunkSpec``
  wrappers are left exactly as they are. The one name buried inside a value is
  ``Advanced.dims``, which :func:`_intern_indexer` relabels in place.
- **Derived reasoning flags** the fluent node computes from its verbatim call header
  (``keepdims``, ``single``, ``uniform``) are *materialized* onto the interned record as
  typed fields, so the reasoner never parses ``args``/``kwargs``.
- **``args``/``kwargs``** stay verbatim — the Python replay header (``lower._emit_node``
  re-invokes them, and they carry options like ``skipna`` that live nowhere else). They are
  the one loosely-typed slot; the Rust reasoning struct does not extract them.

Because a handle is a distinct *type* (``InternedVal``), it is never confused with a literal
``int`` value: de-interning looks up an ``InternedVal`` and leaves a bare ``int`` alone.
:class:`~xrexpr.ir.AllDims` is preserved as the sentinel it is.

Every ``FluentOp | LoweredOp`` has an interned counterpart by construction: :func:`intern`
and :func:`deintern` close their ``match`` with ``assert_never``, so a new op variant is a
type error here until both directions handle it.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from frozendict import frozendict

from xrexpr.chunks import ChunkSpec
from xrexpr.indexers import Indexer
from xrexpr.intern.interner import InternedVal
from xrexpr.ir import AllDims, ContextOpenName

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

_V = TypeVar("_V")


@dataclass(frozen=True)
class InternedReduce:
    """Interned :class:`~xrexpr.ir.Reduce` — ``consumes`` relabeled, ``keepdims`` materialized."""

    name: str
    consumes: frozenset[InternedVal] | AllDims = frozenset()
    keepdims: bool = False
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedSelect:
    """Interned :class:`~xrexpr.ir.Select` — ``indexer`` keys relabeled, values kept structural."""

    name: Literal["isel", "sel"]
    indexer: frozendict[InternedVal, Indexer] = field(default_factory=frozendict)
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedScan:
    """Interned :class:`~xrexpr.ir.Scan` — ``dims`` relabeled, ``AllDims`` preserved."""

    name: Literal["cumsum", "cumprod", "diff"]
    dims: frozenset[InternedVal] | AllDims = frozenset()
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedElementwise:
    """Interned :class:`~xrexpr.ir.Elementwise` — no names; just the replay header."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedProject:
    """Interned :class:`~xrexpr.ir.Project` — ``variables`` relabeled, ``single`` materialized."""

    name: Literal["__getitem__"]
    variables: tuple[InternedVal, ...] = ()
    single: bool = False
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedRechunk:
    """Interned :class:`~xrexpr.ir.Rechunk` — ``chunks`` keys relabeled, ``uniform`` materialized."""

    name: Literal["chunk"]
    chunks: frozendict[InternedVal, ChunkSpec] = field(default_factory=frozendict)
    uniform: ChunkSpec | None = None
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedOpaque:
    """Interned :class:`~xrexpr.ir.Opaque` — no names; the replay header may hold arrays."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedDrop:
    """Interned :class:`~xrexpr.ir.Drop` — ``variables`` relabeled per name."""

    name: Literal["drop_vars"]
    variables: tuple[InternedVal, ...] = ()
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedRename:
    """Interned :class:`~xrexpr.ir.Rename` — both sides of ``mapping`` relabeled."""

    name: Literal["rename"]
    mapping: frozendict[InternedVal, InternedVal] = field(default_factory=frozendict)
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedContextOpen:
    """Interned :class:`~xrexpr.ir.ContextOpen` — fluent-only; never survives lowering."""

    name: ContextOpenName
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedGroupedReduce:
    """Interned :class:`~xrexpr.ir.GroupedReduce` — ``group_dim``/``new_dim``/``consumes`` relabeled."""

    name: Literal["groupby", "groupby_bins", "resample"]
    group_dim: InternedVal
    new_dim: InternedVal
    reduce: str
    consumes: frozenset[InternedVal] = frozenset()
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedWindowedReduce:
    """Interned :class:`~xrexpr.ir.WindowedReduce` — ``window`` keys relabeled, sizes kept."""

    name: Literal["rolling", "coarsen"]
    reduce: str
    window: frozendict[InternedVal, int] = field(default_factory=frozendict)
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)


@dataclass(frozen=True)
class InternedWeightedReduce:
    """Interned :class:`~xrexpr.ir.WeightedReduce` — ``weight_dims``/``consumes`` relabeled, weights verbatim."""

    name: Literal["weighted"]
    reduce: str
    weight_dims: frozenset[InternedVal] = frozenset()
    consumes: frozenset[InternedVal] | AllDims = frozenset()
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)


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
