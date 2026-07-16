"""Record-time logical schema tracking (no data materialisation).

The ``.plan`` proxy starts from the *real* ``self._base_ds``, so as it records a
chain it can maintain a cheap **logical** schema — the current dims, their sizes,
and the coordinate names — and evolve it after each op **without ever touching
array data**. That is the record-time win the CST path could never have: it lets
``to_opnode`` (PR 5) resolve, e.g., a no-dim ``mean()`` to "every dim that exists
*right now*" rather than blindly against the original dataset.

:class:`SchemaState` is an immutable snapshot; :func:`apply_schema` returns the
next snapshot after an :class:`~xrexpr.ir.OpNode` is applied; :func:`to_opnode`
normalises a raw recorded call into that ``OpNode`` against the current schema.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr
from frozendict import frozendict

from xrexpr.ir import OpNode
from xrexpr.operations import spec as op_spec

__all__ = ["SchemaState", "apply_schema", "to_opnode"]


@dataclass(frozen=True)
class SchemaState:
    """An immutable snapshot of a dataset's logical shape at one point in a plan.

    Holds only metadata — ``dims`` (name → size) and ``coords`` (coordinate names)
    — never array data. ``dims``/``coords`` are coerced to immutable containers on
    construction, so a snapshot is hashable and safe to thread through the plan.
    """

    dims: frozendict[Hashable, int] = field(default_factory=frozendict)
    coords: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", frozendict(self.dims))
        object.__setattr__(self, "coords", frozenset(self.coords))

    @classmethod
    def from_dataset(cls, ds: xr.Dataset | xr.DataArray) -> "SchemaState":
        """Snapshot ``ds``'s logical schema, reading only ``.sizes``/``.coords``."""
        return cls(dims=frozendict(ds.sizes), coords=frozenset(ds.coords))

    @property
    def dim_names(self) -> frozenset[Hashable]:
        """The set of current dimension names."""
        return frozenset(self.dims)


def apply_schema(schema: SchemaState, node: OpNode) -> SchemaState:
    """Return the schema resulting from applying ``node`` to ``schema``.

    - dims in ``node.consumes`` are removed (reductions, and scalar selects that
      drop their dim);
    - a select that *keeps* a dim (a slice/sequence indexer) resizes it;
    - a coordinate sharing a name with a removed dim disappears.

    Ops touching neither ``consumes`` nor ``indexer`` (scans, elementwise) leave
    the schema unchanged — ``apply_schema`` is driven entirely by node metadata,
    never by re-inspecting raw call args.
    """
    dims = dict(schema.dims)

    for dim in node.consumes:
        dims.pop(dim, None)

    # A non-scalar select keeps its dim but changes its size (scalar selects are
    # already gone via ``consumes``).
    for dim, indexer in node.indexer.items():
        if dim not in node.consumes and dim in dims:
            dims[dim] = _indexer_size(indexer, dims[dim])

    removed = frozenset(schema.dims) - frozenset(dims)
    coords = frozenset(c for c in schema.coords if c not in removed)
    return SchemaState(dims=frozendict(dims), coords=coords)


def _indexer_size(indexer: Any, current: int) -> int:
    """Best-effort new size of a kept dim under a non-scalar ``isel``/``sel`` indexer.

    Handles the cheap, unambiguous cases (slices, integer/boolean sequences and
    arrays). For anything that would need data to size — e.g. a ``sel`` label slice
    needing coordinate values — it conservatively returns the current size. Being
    conservative is always *safe*: an imprecise size never changes the result of a
    replayed plan, it only leaves a potential size-driven optimisation on the table.
    """
    if isinstance(indexer, slice):
        bounds = (indexer.start, indexer.stop, indexer.step)
        if all(b is None or isinstance(b, int) for b in bounds):
            return len(range(*indexer.indices(current)))
        return current  # label slice (e.g. sel) — needs coords to size

    if isinstance(indexer, np.ndarray):
        return int(indexer.sum()) if indexer.dtype == bool else int(indexer.size)
    if isinstance(indexer, list | tuple):
        seq = list(indexer)
        if seq and all(isinstance(x, bool | np.bool_) for x in seq):
            return int(sum(bool(x) for x in seq))
        return len(seq)
    return current


#: ``isel``/``sel`` keyword arguments that are *options*, not dim indexers.
_SELECT_OPTION_KWARGS = frozenset({"drop", "missing_dims", "method", "tolerance"})


def to_opnode(
    schema: SchemaState,
    name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> OpNode:
    """Normalise one recorded call into a resolved :class:`~xrexpr.ir.OpNode`.

    Resolution is against the *current* ``schema`` (not the original dataset) — the
    record-time win the proxy unlocks:

    - **reduce** (``mean``/``sum``/...): the dim spec — positional (``mean("lat")``),
      keyword (``mean(dim="lat")``) or tuple (``mean(("lat", "lon"))``) — collapses to
      one ``consumes`` frozenset; a **no-dim ``mean()`` consumes every dim in the
      schema right now**, fixing the empty-dim reorder bug (``ds.mean().isel(...)``).
    - **select** (``isel``/``sel``): the indexer (a positional dict and/or kwargs,
      minus option kwargs like ``drop``) becomes ``indexer``; a dim given a *scalar*
      index is dropped and so also lands in ``consumes`` (a slice/list/array keeps it).
    - **scan** / untabulated ops: ``kind`` only; no dims resolved.

    ``args``/``kwargs`` are kept verbatim for faithful replay; ``consumes``/``indexer``
    are the derived metadata the optimiser reasons about.
    """
    op = op_spec(name)
    kind = op.kind if op is not None else "opaque"
    kw = frozendict(kwargs)

    if kind == "reduce":
        return OpNode(
            name=name,
            kind=kind,
            args=args,
            kwargs=kw,
            consumes=_reduce_dims(schema, args, kwargs),
        )
    if kind == "select":
        indexer = _select_indexer(args, kwargs)
        consumes = frozenset(d for d, v in indexer.items() if _is_scalar_index(v))
        return OpNode(
            name=name,
            kind=kind,
            args=args,
            kwargs=kw,
            consumes=consumes,
            indexer=indexer,
        )
    return OpNode(name=name, kind=kind, args=args, kwargs=kw)


def _reduce_dims(
    schema: SchemaState, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> frozenset[Hashable]:
    """Dims a reduction removes: its ``dim`` spec, or *all* current dims if unspecified."""
    if "dim" in kwargs:
        dim = kwargs["dim"]
    elif args:
        dim = args[0]  # reductions take ``dim`` first (``.reduce(func, dim)`` aside)
    else:
        dim = None
    if dim is None:  # bare ``mean()`` / ``mean(dim=None)`` → every current dim
        return frozenset(schema.dim_names)
    return _as_dim_set(dim)


def _as_dim_set(dim: Any) -> frozenset[Hashable]:
    """A single dim name or an iterable of them → a frozenset (xarray's dim convention)."""
    if isinstance(dim, str) or not isinstance(dim, Iterable):
        return frozenset({dim})
    return frozenset(dim)


def _select_indexer(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> frozendict[Hashable, Any]:
    """The ``{dim: index}`` mapping of an ``isel``/``sel`` call (option kwargs dropped)."""
    indexer: dict[Hashable, Any] = {}
    if args and isinstance(args[0], dict):
        indexer.update(args[0])
    for key, value in kwargs.items():
        if key not in _SELECT_OPTION_KWARGS:
            indexer[key] = value
    return frozendict(indexer)


def _is_scalar_index(value: Any) -> bool:
    """Whether an indexer drops its dim (a scalar) rather than keeping it.

    A ``slice``/``list``/``tuple``/array keeps the dim (and resizes it); anything
    else is treated as a scalar that removes it. (A tuple MultiIndex *label* is the
    known exception — niche, and left for later.)
    """
    return not isinstance(value, slice | list | tuple | np.ndarray)
