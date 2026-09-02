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

from collections.abc import Hashable, Mapping
from typing import TypeVar, assert_never, cast

from frozendict import frozendict

from xrexpr.indexers import Advanced, Indexer
from xrexpr.intern.interner import InternedVal, Interner
from xrexpr.intern.ir import (
    InternedContextOpen,
    InternedDrop,
    InternedElementwise,
    InternedFluentOp,
    InternedGroupedReduce,
    InternedLoweredOp,
    InternedOpaque,
    InternedProject,
    InternedRechunk,
    InternedReduce,
    InternedRename,
    InternedScan,
    InternedSelect,
    InternedWeightedReduce,
    InternedWindowedReduce,
)
from xrexpr.ir import (
    AllDims,
    ContextOpen,
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
    "deintern",
    "intern",
]

_V = TypeVar("_V")


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
