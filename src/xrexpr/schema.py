"""Record-time logical schema tracking (no data materialisation).

The ``.plan`` proxy starts from the *real* ``self._base_ds``, so as it records a
chain it can maintain a cheap **logical** schema — the current dims, their sizes,
and the coordinate names — and evolve it after each op **without ever touching
array data**. That is the record-time win the CST path could never have: it lets
``to_opnode`` (PR 5) resolve, e.g., a no-dim ``mean()`` to "every dim that exists
*right now*" rather than blindly against the original dataset.

:class:`SchemaState` is an immutable snapshot; :func:`apply_schema` returns the
next snapshot after an :data:`~xrexpr.ir.Op` node is applied; :func:`to_opnode`
normalises a raw recorded call into that ``Op`` variant against the current schema.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import xarray as xr
from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr.ir import Op, Opaque, Project, Reduce, Scan, Select
from xrexpr.operations import spec as op_spec

__all__ = ["SchemaState", "apply_schema", "to_opnode"]


@dataclass(frozen=True)
class SchemaState:
    """An immutable snapshot of a dataset's logical shape at one point in a plan.

    Holds only metadata — ``dims`` (name → size), ``coords`` (coordinate names) and
    ``data_vars`` (variable name → its dims) — never array data. All three are coerced
    to immutable containers on construction, so a snapshot is hashable and safe to
    thread through the plan.

    ``data_vars`` is what makes *variable*-level reasoning possible: whether a
    projection may hop left past an op depends on whether the projected subset still
    carries the dims that op names.
    """

    dims: frozendict[Hashable, int] = field(default_factory=frozendict)
    coords: frozenset[Hashable] = frozenset()
    data_vars: frozendict[Hashable, tuple[Hashable, ...]] = field(
        default_factory=frozendict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", frozendict(self.dims))
        object.__setattr__(self, "coords", frozenset(self.coords))
        object.__setattr__(
            self,
            "data_vars",
            frozendict({k: tuple(v) for k, v in self.data_vars.items()}),
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset | xr.DataArray) -> "SchemaState":
        """Snapshot ``ds``'s logical schema, reading only ``.sizes``/``.coords``/dims."""
        data_vars = (
            {k: tuple(v.dims) for k, v in ds.data_vars.items()}
            if isinstance(ds, xr.Dataset)
            else {}  # a DataArray has no data_vars to project over
        )
        return cls(
            dims=frozendict(ds.sizes),
            coords=frozenset(ds.coords),
            data_vars=frozendict(data_vars),
        )

    def var_dims(self, names: Iterable[Hashable]) -> frozenset[Hashable] | None:
        """The dims carried by ``names`` collectively, or ``None`` if any is unknown.

        ``None`` is the *don't know* answer — a name that isn't a tracked data
        variable (a coordinate, or something an unmodelled op introduced) — and
        callers must treat it as "no rewrite", never as "no dims".
        """
        dims: set[Hashable] = set()
        for name in names:
            if name not in self.data_vars:
                return None
            dims.update(self.data_vars[name])
        return frozenset(dims)

    @property
    def dim_names(self) -> frozenset[Hashable]:
        """The set of current dimension names."""
        return frozenset(self.dims)


def apply_schema(schema: SchemaState, node: Op) -> SchemaState:
    """Return the schema resulting from applying ``node`` to ``schema``.

    Each variant affects the schema differently, so this dispatches with ``match``:

    - :class:`~xrexpr.ir.Reduce` removes its ``consumes`` dims;
    - :class:`~xrexpr.ir.Select` removes the dims it drops (scalar indices) and
      *resizes* the dims it keeps (slice/sequence indices);
    - :class:`~xrexpr.ir.Project` restricts the variables, leaving dims alone;
    - :class:`~xrexpr.ir.Scan`/:class:`~xrexpr.ir.Opaque` leave dims untouched;
    - a coordinate sharing a name with a removed dim disappears (all cases), and a
      removed dim likewise disappears from every variable's dim tuple.

    An :class:`~xrexpr.ir.Opaque` is assumed variable-preserving, which is *not* true
    in general (``rename``/``drop_vars``/``assign``). That makes the schema exact only
    up to the first opaque op — a trust boundary the optimiser enforces rather than
    this function (see ``optimize._trusted_prefix``).

    ``assert_never`` on the final arm makes the union exhaustive: a new variant fails
    type-check here until handled.
    """
    dims = dict(schema.dims)
    data_vars = dict(schema.data_vars)

    match node:
        case Reduce(consumes=consumes):
            for dim in consumes:
                dims.pop(dim, None)
        case Select(indexer=indexer) as select:
            for dim in select.consumes:
                dims.pop(dim, None)
            # A non-scalar select keeps its dim but changes its size (scalar selects
            # are already gone via ``consumes``).
            for dim, index in indexer.items():
                if dim not in select.consumes and dim in dims:
                    dims[dim] = _indexer_size(index, dims[dim])
        case Project(variables=variables):
            data_vars = {v: data_vars[v] for v in variables if v in data_vars}
        case Scan() | Opaque():
            pass
        case _:
            assert_never(node)

    removed = frozenset(schema.dims) - frozenset(dims)
    coords = frozenset(c for c in schema.coords if c not in removed)
    data_vars = {
        name: tuple(d for d in var_dims if d not in removed)
        for name, var_dims in data_vars.items()
    }
    return SchemaState(
        dims=frozendict(dims), coords=coords, data_vars=frozendict(data_vars)
    )


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
) -> Op:
    """Normalise one recorded call into a resolved :data:`~xrexpr.ir.Op` variant.

    Resolution is against the *current* ``schema`` (not the original dataset) — the
    record-time win the proxy unlocks:

    - **reduce** (``mean``/``sum``/...): the dim spec — positional (``mean("lat")``),
      keyword (``mean(dim="lat")``) or tuple (``mean(("lat", "lon"))``) — collapses to
      one ``consumes`` frozenset; a **no-dim ``mean()`` consumes every dim in the
      schema right now**, fixing the empty-dim reorder bug (``ds.mean().isel(...)``).
    - **select** (``isel``/``sel``): the indexer (a positional dict and/or kwargs,
      minus option kwargs like ``drop``) becomes ``indexer``; a dim given a *scalar*
      index is dropped and so also lands in ``consumes`` (a slice/list/array keeps it).
    - **project** (``ds["tas"]`` / ``ds[["tas", "pr"]]``): the key's variable names
      become ``variables``. This is the one kind the ``OP_TABLE`` can't settle —
      ``__getitem__`` is a projection only when its key *names variables*, so a
      mask-style key (a boolean ``DataArray``, a dict) stays ``Opaque``.
    - **scan** / untabulated ops: no dims resolved (name/args/kwargs only).

    ``args``/``kwargs`` are kept verbatim for faithful replay; ``consumes``/``indexer``
    are the derived metadata the optimiser reasons about. The ``OP_TABLE`` kind selects
    the variant; a ``Select``/``Scan`` name is ``cast`` to its ``Literal`` because the
    table guarantees it is one of the closed set (the ``Literal`` still guards
    hand-written construction elsewhere).
    """
    op = op_spec(name)
    kind = op.kind if op is not None else "opaque"
    kw = frozendict(kwargs)

    if kind == "reduce":
        return Reduce(
            name=name,
            args=args,
            kwargs=kw,
            consumes=_reduce_dims(schema, args, kwargs),
        )
    if kind == "select":
        return Select(
            name=cast(Literal["isel", "sel"], name),
            args=args,
            kwargs=kw,
            indexer=_select_indexer(args, kwargs),
        )
    if kind == "scan":
        return Scan(
            name=cast(Literal["cumsum", "cumprod", "diff"], name),
            args=args,
            kwargs=kw,
        )
    if name == "__getitem__" and (variables := _projected_names(args)) is not None:
        return Project(name="__getitem__", args=args, kwargs=kw, variables=variables)
    return Opaque(name=name, args=args, kwargs=kw)


def _projected_names(args: tuple[Any, ...]) -> tuple[Hashable, ...] | None:
    """The variable names a ``__getitem__`` key selects, or ``None`` if it isn't one.

    Recognises the two projection spellings, splitting them exactly where xarray's own
    ``Dataset.__getitem__`` does: a **hashable** key is one variable name
    (``ds["tas"]``, and a tuple counts — xarray reads it as a single name, not a list),
    while a **list** is a sequence of them (``ds[["tas", "pr"]]``). Everything else — a
    dict-like key (which xarray routes to ``isel``), a boolean ``DataArray`` mask —
    returns ``None`` and so falls through to :class:`~xrexpr.ir.Opaque`, never to be
    reordered. Names are taken verbatim; they are checked against the schema's
    ``data_vars`` by the rule that would move the node, not here.
    """
    if len(args) != 1:
        return None
    key = args[0]
    if isinstance(key, Hashable):
        return (key,)
    if isinstance(key, list) and all(isinstance(k, Hashable) for k in key):
        return tuple(key)
    return None


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
