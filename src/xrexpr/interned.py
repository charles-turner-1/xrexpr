"""The *interned* IR — every op's *names* relabeled to handles, its *structure* preserved.

The optimiser reasons over the fluent/lowered IR (``ir.py``); this module is the last step
before that plan crosses to a Rust backend that does the reasoning itself. Rust cannot hash
or compare arbitrary Python objects, so dim and variable **names** are relabeled to
:class:`~xrexpr.interner.InternedVal` handles. But the reasoner needs the closed value
sum-types — :data:`~xrexpr.indexers.Indexer`, :data:`~xrexpr.chunks.ChunkSpec` — as
*structure* (they exist to map onto Rust enums), and it does arithmetic on the literal
values inside them (slice bounds, positions, block sizes). So interning is **selective**:

- **Names** (dims, variables) become :class:`~xrexpr.interner.InternedVal`.
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

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, assert_never, cast

from frozendict import frozendict

from xrexpr.chunks import ChunkSpec
from xrexpr.indexers import Advanced, Indexer
from xrexpr.interner import InternedVal, Interner
from xrexpr.ir import (
    AllDims,
    ContextOpen,
    ContextOpenName,
    DimSet,
    Drop,
    Elementwise,
    FluentOp,
    GroupedReduce,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Rename,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)

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
    "deintern",
    "intern",
]

_V = TypeVar("_V")


# --------------------------------------------------------------------------------------
# Name-atom converters. Names -> InternedVal and back; everything else is left untouched.
# --------------------------------------------------------------------------------------


def _iname(name: Hashable) -> InternedVal:
    """Relabel one name to its handle."""
    it: Interner[Hashable] = Interner()
    return InternedVal(it(name))


def _dename(val: InternedVal) -> Hashable:
    """Look one handle back up to its name."""
    it: Interner[Hashable] = Interner()
    return it[val.handle]


def _inames(names: tuple[Hashable, ...]) -> tuple[InternedVal, ...]:
    return tuple(_iname(n) for n in names)


def _dnames(names: tuple[InternedVal, ...]) -> tuple[Hashable, ...]:
    return tuple(_dename(n) for n in names)


def _inameset(names: frozenset[Hashable]) -> frozenset[InternedVal]:
    return frozenset(_iname(n) for n in names)


def _dnameset(names: frozenset[InternedVal]) -> frozenset[Hashable]:
    return frozenset(_dename(n) for n in names)


def _idims(dims: DimSet) -> frozenset[InternedVal] | AllDims:
    """Intern a dim set, preserving the :data:`~xrexpr.ir.ALL_DIMS` sentinel."""
    if isinstance(dims, AllDims):
        return dims
    return frozenset(_iname(d) for d in dims)


def _ddims(dims: frozenset[InternedVal] | AllDims) -> DimSet:
    if isinstance(dims, AllDims):
        return dims
    return frozenset(_dename(d) for d in dims)


def _inamemap(
    mapping: Mapping[Hashable, Hashable],
) -> frozendict[InternedVal, InternedVal]:
    return frozendict({_iname(k): _iname(v) for k, v in mapping.items()})


def _dnamemap(
    mapping: Mapping[InternedVal, InternedVal],
) -> frozendict[Hashable, Hashable]:
    return frozendict({_dename(k): _dename(v) for k, v in mapping.items()})


def _ikeys(mapping: Mapping[Hashable, _V]) -> frozendict[InternedVal, _V]:
    """Intern the *keys* of a ``{dim: value}`` map (chunks, window); values untouched."""
    return frozendict({_iname(k): v for k, v in mapping.items()})


def _dkeys(mapping: Mapping[InternedVal, _V]) -> frozendict[Hashable, _V]:
    return frozendict({_dename(k): v for k, v in mapping.items()})


def _intern_indexer(idx: Indexer) -> Indexer:
    """Relabel the one name inside an indexer value — ``Advanced.dims`` — leaving the
    variant and every other value atom (positions, bounds, labels) exactly as they are.

    Moot on a live plan: an ``Advanced`` select is recorded :class:`~xrexpr.ir.Opaque`, so
    it never reaches here. Handled anyway so the transform is total over the variant.
    """
    if isinstance(idx, Advanced):
        return Advanced(dims=_inames(idx.dims))
    return idx


def _deintern_indexer(idx: Indexer) -> Indexer:
    if isinstance(idx, Advanced):
        # ``Advanced.dims`` is declared ``tuple[Hashable]`` in both worlds; after interning
        # it holds ``InternedVal``s, which the reader here knows but the type cannot.
        return Advanced(dims=_dnames(cast(tuple[InternedVal, ...], idx.dims)))
    return idx


def _iindexer(mapping: Mapping[Hashable, Indexer]) -> frozendict[InternedVal, Indexer]:
    return frozendict({_iname(k): _intern_indexer(v) for k, v in mapping.items()})


def _dindexer(mapping: Mapping[InternedVal, Indexer]) -> frozendict[Hashable, Indexer]:
    return frozendict({_dename(k): _deintern_indexer(v) for k, v in mapping.items()})


# --------------------------------------------------------------------------------------
# The interned op records. Name fields hold ``InternedVal``; value fields keep the real
# sum-types; reasoning flags are materialized; ``args``/``kwargs`` are the replay header.
# --------------------------------------------------------------------------------------


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


def intern(op: FluentOp | LoweredOp) -> InternedFluentOp | InternedLoweredOp:
    """Relabel ``op``'s names to handles, materialize its reasoning flags, keep its structure.

    Total over every op variant: the ``assert_never`` makes a new one a type error until it
    is handled. ``args``/``kwargs`` and every value atom pass through untouched.
    """
    match op:
        case Reduce():
            return InternedReduce(
                op.name, _idims(op.consumes), op.keepdims, op.args, op.kwargs
            )
        case Select():
            return InternedSelect(op.name, _iindexer(op.indexer), op.args, op.kwargs)
        case Scan():
            return InternedScan(op.name, _idims(op.dims), op.args, op.kwargs)
        case Elementwise():
            return InternedElementwise(op.name, op.args, op.kwargs)
        case Project():
            return InternedProject(
                op.name, _inames(op.variables), op.single, op.args, op.kwargs
            )
        case Drop():
            return InternedDrop(op.name, _inames(op.variables), op.args, op.kwargs)
        case Rename():
            return InternedRename(op.name, _inamemap(op.mapping), op.args, op.kwargs)
        case Rechunk():
            return InternedRechunk(
                op.name, _ikeys(op.chunks), op.uniform, op.args, op.kwargs
            )
        case Opaque():
            return InternedOpaque(op.name, op.args, op.kwargs)
        case ContextOpen():
            return InternedContextOpen(op.name, op.args, op.kwargs)
        case GroupedReduce():
            return InternedGroupedReduce(
                op.name,
                _iname(op.group_dim),
                _iname(op.new_dim),
                op.reduce,
                _inameset(op.consumes),
                op.args,
                op.kwargs,
                op.reduce_args,
                op.reduce_kwargs,
            )
        case WindowedReduce():
            return InternedWindowedReduce(
                op.name,
                op.reduce,
                _ikeys(op.window),
                op.args,
                op.kwargs,
                op.reduce_args,
                op.reduce_kwargs,
            )
        case WeightedReduce():
            return InternedWeightedReduce(
                op.name,
                op.reduce,
                _inameset(op.weight_dims),
                _idims(op.consumes),
                op.args,
                op.kwargs,
                op.reduce_args,
                op.reduce_kwargs,
            )
        case _:
            assert_never(op)


def deintern(op: InternedFluentOp | InternedLoweredOp) -> FluentOp | LoweredOp:
    """Inverse of :func:`intern`: look names back up and rebuild the fluent/lowered op.

    Rebuilds through the *fluent* constructors, so ``__post_init__`` re-runs every
    normalisation and derivation (``keepdims``, ``single``, ``uniform``, ``classify``) from
    the verbatim replay header — the materialized flags are Rust-facing only and ignored
    here. Total, with the same ``assert_never`` guarantee as :func:`intern`.
    """
    match op:
        case InternedReduce():
            return Reduce(
                name=op.name,
                args=op.args,
                kwargs=op.kwargs,
                consumes=_ddims(op.consumes),
            )
        case InternedSelect():
            return Select(
                name=op.name,
                args=op.args,
                kwargs=op.kwargs,
                indexer=_dindexer(op.indexer),
            )
        case InternedScan():
            return Scan(
                name=op.name, args=op.args, kwargs=op.kwargs, dims=_ddims(op.dims)
            )
        case InternedElementwise():
            return Elementwise(name=op.name, args=op.args, kwargs=op.kwargs)
        case InternedProject():
            return Project(
                name=op.name,
                args=op.args,
                kwargs=op.kwargs,
                variables=_dnames(op.variables),
            )
        case InternedDrop():
            return Drop(
                name=op.name,
                args=op.args,
                kwargs=op.kwargs,
                variables=_dnames(op.variables),
            )
        case InternedRename():
            return Rename(
                name=op.name,
                args=op.args,
                kwargs=op.kwargs,
                mapping=_dnamemap(op.mapping),
            )
        case InternedRechunk():
            return Rechunk(
                name=op.name, args=op.args, kwargs=op.kwargs, chunks=_dkeys(op.chunks)
            )
        case InternedOpaque():
            return Opaque(name=op.name, args=op.args, kwargs=op.kwargs)
        case InternedContextOpen():
            return ContextOpen(name=op.name, args=op.args, kwargs=op.kwargs)
        case InternedGroupedReduce():
            return GroupedReduce(
                name=op.name,
                group_dim=_dename(op.group_dim),
                new_dim=_dename(op.new_dim),
                reduce=op.reduce,
                args=op.args,
                kwargs=op.kwargs,
                reduce_args=op.reduce_args,
                reduce_kwargs=op.reduce_kwargs,
                consumes=_dnameset(op.consumes),
            )
        case InternedWindowedReduce():
            return WindowedReduce(
                name=op.name,
                reduce=op.reduce,
                window=_dkeys(op.window),
                args=op.args,
                kwargs=op.kwargs,
                reduce_args=op.reduce_args,
                reduce_kwargs=op.reduce_kwargs,
            )
        case InternedWeightedReduce():
            return WeightedReduce(
                name=op.name,
                reduce=op.reduce,
                weight_dims=_dnameset(op.weight_dims),
                args=op.args,
                kwargs=op.kwargs,
                reduce_args=op.reduce_args,
                reduce_kwargs=op.reduce_kwargs,
                consumes=_ddims(op.consumes),
            )
        case _:
            assert_never(op)
