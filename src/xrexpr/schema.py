"""Logical schema tracking (no data materialisation).

The ``.plan`` proxy starts from the *real* base dataset, so a cheap **logical** schema
— the current dims, their sizes, and the coordinate names — can be folded forward
through a plan **without ever touching array data**. That is what lets a rule know the
shape each node sees, and what a dim set like ``ds.mean()``'s
:data:`~xrexpr.ir.ALL_DIMS` is finally resolved against.

The fold is owned by the *optimiser* (``optimize._schemas``), not by recording:
:func:`to_opnode` is a pure function of one call, so nothing is resolved against a
schema that is only a guess about where the op will run.

:class:`SchemaState` is an immutable snapshot; :func:`apply_schema` returns the
next snapshot after an :data:`~xrexpr.ir.Op` node is applied; :func:`to_opnode`
normalises a raw recorded call into that ``Op`` variant.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import xarray as xr
from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr.indexers import Indexer
from xrexpr.ir import (
    ALL_DIMS,
    AllDims,
    ContextOpen,
    ContextOpenName,
    DimSet,
    FluentOp,
    GroupedReduce,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.operations import CONTEXT_METHODS
from xrexpr.operations import spec as op_spec

__all__ = ["SchemaState", "apply_schema", "to_opnode"]


@dataclass(frozen=True)
class SchemaState:
    """An immutable snapshot of a dataset's logical shape at one point in a plan.

    Attributes
    ----------
    dims : frozendict
        Dim name → size, where a size of ``None`` means **don't know** (see the notes).
    coords : frozenset of Hashable
        The coordinate names.
    data_vars : frozendict
        Data variable name → the tuple of dims it carries.

    Notes
    -----
    Holds only metadata — never array data. All three fields are coerced to immutable
    containers on construction, so a snapshot is hashable and safe to thread through the
    plan.

    A size of ``None`` means **don't know** — the dim exists, but its extent is not
    statically evident. The same contract ``var_dims`` states, and the same warning
    applies with one substitution: callers must treat it as "no rewrite", never as
    *size zero*. Under-reporting a size is the unsafe direction, because it is the one
    a rewrite could act on. Nothing computes a size but :func:`apply_schema`'s select
    arm, and no optimiser rule reads one at all — every rule reasons about dim *names*
    — which is what keeps the unknown from having to propagate anywhere else.

    ``data_vars`` is what makes *variable*-level reasoning possible: whether a
    projection may hop left past an op depends on whether the projected subset still
    carries the dims that op names.
    """

    dims: frozendict[Hashable, int | None] = field(default_factory=frozendict)
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
        """Snapshot a dataset's logical schema, materialising nothing.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            The object to read. Only ``.sizes``, ``.coords`` and each variable's dims are
            touched — all metadata, so a dask-backed dataset computes nothing.

        Returns
        -------
        SchemaState
            The snapshot. A ``DataArray`` has no ``data_vars`` to project over, so that
            field comes back empty.
        """
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
        """Return the dims carried by ``names`` collectively.

        Parameters
        ----------
        names : iterable of Hashable
            Data variable names, e.g. the ones a :class:`~xrexpr.ir.Project` requests.

        Returns
        -------
        frozenset of Hashable or None
            The union of those variables' dims, or ``None`` if any name is not a tracked
            data variable.

        Notes
        -----
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
        """The set of current dimension names.

        Returns
        -------
        frozenset of Hashable
            The keys of :attr:`dims`, which is what a symbolic
            :data:`~xrexpr.ir.ALL_DIMS` resolves to.
        """
        return frozenset(self.dims)


def apply_schema(schema: SchemaState, node: LoweredOp) -> SchemaState:
    """Return the schema resulting from applying ``node`` to ``schema``.

    Parameters
    ----------
    schema : SchemaState
        The schema *entering* the node.
    node : LoweredOp
        The node to apply.

    Returns
    -------
    SchemaState
        The next snapshot: the schema the following node sees.

    Notes
    -----
    Each variant affects the schema differently, so this dispatches with ``match``:

    - :class:`~xrexpr.ir.Reduce` removes its ``consumes`` dims — *every* dim when that
      is :data:`~xrexpr.ir.ALL_DIMS`, which is where the deferred bare-reduce expansion
      is finally cashed in against a schema that is exact;
    - :class:`~xrexpr.ir.Select` removes the dims it drops (scalar indices) and
      *resizes* the dims it keeps (slice/sequence indices);
    - :class:`~xrexpr.ir.Project` restricts the variables, and drops any dim the survivors
      no longer span — xarray orphans it rather than keeping it empty;
    - :class:`~xrexpr.ir.GroupedReduce` removes its ``group_dim`` and any extra
      ``consumes``, and mints ``new_dim`` — of *unknown* size, since the group count is a
      fact about coordinate values — onto every variable;
    - :class:`~xrexpr.ir.WindowedReduce` keeps every dim and *resizes* the windowed ones
      (``coarsen`` only);
    - :class:`~xrexpr.ir.WeightedReduce` removes its ``consumes`` like a plain reduce, and
      then marks every surviving ``weight_dims`` entry unknown — the weights broadcast in
      the dims the dataset lacks and align (so possibly *shrink*) the ones it shares;
    - :class:`~xrexpr.ir.Scan`/:class:`~xrexpr.ir.Rechunk`/:class:`~xrexpr.ir.Opaque`
      leave dims untouched (a rechunk changes only chunk topology);
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
    # Dims this node *adds*. Collected by the arms and applied to every variable once
    # below, because two arms mint now and both mint onto every variable alike.
    minted: set[Hashable] = set()

    match node:
        case Reduce(consumes=consumes):
            # Nested rather than two ``Reduce`` arms: splitting the variant across arms
            # leaves mypy unable to see it as exhausted, and this way ``DimSet`` gets an
            # ``assert_never`` of its own -- a third dim-set shape fails type-check here
            # too, not just a seventh ``Op``.
            match consumes:
                case AllDims():
                    dims.clear()  # a bare ``mean()`` -- whatever is here now, it goes
                case frozenset() as named:
                    for dim in named:
                        dims.pop(dim, None)
                case _:
                    assert_never(consumes)
        case Select(indexer=indexer) as select:
            for dim in select.consumes:
                dims.pop(dim, None)
            # A non-scalar select keeps its dim but changes its size (scalar selects
            # are already gone via ``consumes``).
            for dim, index in indexer.items():
                if dim not in select.consumes and dim in dims:
                    dims[dim] = _selected_size(select.name, index, dims[dim])
        case Project(variables=variables):
            data_vars = {v: data_vars[v] for v in variables if v in data_vars}
            if all(v in schema.data_vars for v in variables):
                # A projection **orphans** dims: xarray keeps
                # only the dims the selected variables span, so ``ds[["elevation"]]``
                # (``lat``, ``lon``) drops ``time`` outright rather than leaving it empty.
                # Coordinates do not keep one alive either -- an aux coord over a dropped
                # dim goes with it (verified against xarray 2026.7.0), so the ``removed``
                # tail below handles coords with no extra work.
                #
                # Guarded on every projected name being a *tracked* variable: a name this
                # layer doesn't know (a coord, or something an unmodelled op introduced)
                # would make the union an under-report, and under-reporting dims is the
                # unsafe direction. Unknown name → leave ``dims`` alone, as before.
                spanned = {d for var_dims in data_vars.values() for d in var_dims}
                dims = {d: size for d, size in dims.items() if d in spanned}
        case GroupedReduce() as grouped:
            for dim in (grouped.group_dim, *grouped.consumes):
                dims.pop(dim, None)
            # The minted dim's extent is the number of groups -- a fact about coordinate
            # *values*, which this layer does not read. ``None`` rather than a guess.
            dims[grouped.new_dim] = None
            minted.add(grouped.new_dim)
        case WeightedReduce() as weighted:
            match weighted.consumes:
                case AllDims():
                    dims.clear()  # a bare weighted closer, like a bare ``mean()``
                case frozenset() as named:
                    for dim in named:
                        dims.pop(dim, None)
                    for dim in weighted.weight_dims - named:
                        # The weights have a dim effect of their own, via ``dot``'s
                        # broadcast and alignment -- see ``WeightedReduce``. A weight dim
                        # the dataset lacks is *minted*; one it shares may be *resized*
                        # (misaligned weights inner-join and shrink it). Either way the
                        # name is known and the extent is not, which is the answer the
                        # ``int | None`` sizes exist for. Under ``ALL_DIMS`` above there is
                        # nothing to do: a bare closer clears the weight dims too.
                        if dim not in dims:
                            minted.add(dim)
                        dims[dim] = None
                case _:
                    assert_never(weighted.consumes)
        case WindowedReduce() as windowed:
            for dim, window in windowed.window.items():
                if dim in dims:
                    dims[dim] = _windowed_size(windowed, dims[dim], window)
        case Scan() | Rechunk() | Opaque():
            pass
        case _:
            assert_never(node)

    removed = frozenset(schema.dims) - frozenset(dims)
    coords = frozenset(c for c in schema.coords if c not in removed)
    data_vars = {
        name: tuple(d for d in var_dims if d not in removed)
        for name, var_dims in data_vars.items()
    }
    if minted:
        # Verified against xarray 2026.7.0, and the non-obvious half of the arms above:
        # *every* variable comes back carrying a minted dim, including ones that could not
        # have had it -- ``elevation(lat, lon)`` under ``groupby("time.month")`` is
        # ``(month, lat, lon)``, and under ``weighted(w)`` with ``w(lat, member)`` it is
        # ``(lon, member)``. So this is an addition to each variable, not a substitution of
        # one dim for another.
        #
        # Sorted because ``minted`` is a set and :class:`SchemaState` is a *value*: an
        # iteration-order-dependent dim tuple would make two snapshots folded from identical
        # inputs unequal between processes. ``str`` is the key for the same reason
        # ``pushdown_selects`` uses it — a dim name is only ``Hashable``, so it is the one
        # total order available. Dim order is not otherwise meaningful here (``var_dims``
        # unions into a set, and ``Dataset.sizes`` promises no order either).
        ordered = sorted(minted, key=str)
        data_vars = {
            name: (*ordered, *(d for d in var_dims if d not in minted))
            for name, var_dims in data_vars.items()
        }
    if isinstance(node, GroupedReduce):
        # A minted *coordinate*, which is grouped-specific: the group labels become one.
        # A weight dim broadcast in need not have a coord at all, and under-reporting one
        # is allowed where under-reporting a dim is not -- ``coords`` is only ever asserted
        # to be a *subset* of the result's (``test_tracked_schema_agrees_with_evaluation``),
        # because a scalar select already leaves a coord this layer does not model.
        coords = coords | {node.new_dim}
    return SchemaState(
        dims=frozendict(dims), coords=coords, data_vars=frozendict(data_vars)
    )


def _windowed_size(
    node: WindowedReduce, current: int | None, window: int
) -> int | None:
    """Return the length of a windowed dim after ``node``.

    Parameters
    ----------
    node : WindowedReduce
        The windowed node being applied.
    current : int or None
        The dim's length entering the node, or ``None`` if that is already unknown.
    window : int
        The window size ``node`` names for this dim.

    Returns
    -------
    int or None
        The new length, or ``None`` when it is not statically known.

    Notes
    -----
    ``rolling`` yields one output position per input position, so its dims keep their
    length outright — ``center`` and ``min_periods`` change values, never shape.

    ``coarsen`` divides, and how it rounds is an *option* rather than a property of the
    op: ``boundary="trim"`` discards the short final block (floor), ``"pad"`` fills it
    (ceil), and the default ``"exact"`` requires divisibility — so where it is exact the
    two agree, and where it is not, xarray raises at replay rather than producing the
    size computed here. A ``boundary`` this does not recognise answers ``None``: a
    future spelling should cost precision, not correctness.
    """
    if current is None or node.name == "rolling":
        return current
    match node.kwargs.get("boundary", "exact"):
        case "trim" | "exact":
            return current // window
        case "pad":
            return -(-current // window)  # ceil
        case _:
            return None


def _selected_size(
    name: Literal["isel", "sel"], index: Indexer, current: int | None
) -> int | None:
    """Return the size a kept dim has after ``index`` is applied to it.

    Parameters
    ----------
    name : {"isel", "sel"}
        Which selection the indexer belongs to. A ``sel`` label slice is unsizable here.
    index : Indexer
        The dim's indexer. Must be one that *keeps* its dim — a scalar has already been
        dropped via ``Select.consumes``.
    current : int or None
        The dim's length entering the select, or ``None`` if that is already unknown.

    Returns
    -------
    int or None
        The new length, or ``None`` when it is not statically known.

    Notes
    -----
    The only place a size is *computed*, and so the only place the unknown has to be
    handled. Two cases answer ``None``:

    - **an unknown input size.** Every :meth:`~xrexpr.indexers.Indexer.size` bar the
      concrete enumerations resolves against the current length, so unknown in means
      unknown out. Guarding here keeps ``Indexer.size``'s signature ``int``-only rather
      than pushing ``| None`` through six variant implementations, and the uniform rule
      costs only the exactness ``Positions``/``Mask`` could have kept.
    - **a label slice.** ``sel(lat=slice(20, 30))`` names *labels*, and both ends are
      inclusive, so its extent is a fact about coordinate values. Where the bounds happen
      to be integers :func:`~xrexpr.indexers.classify` cannot tell it from a positional
      slice and mints a ``ForwardSlice``, whose size then reads the bounds as positions —
      with labels ``[10, 20, 30, 40]`` that reports **0** where the answer is 2. Sizing a
      dim *smaller* than it is is the unsafe direction: it is the error a size-driven rule
      would act on. "Don't know" is the honest answer and is now available, so this stops
      being an under-report. (Non-integer label slices took ``Label.size``'s
      keep-the-current-size fallback, an over-report; they answer ``None`` here too, since
      a guess in the safe direction is still a guess.)
    """
    if current is None:
        return None
    if name == "sel" and isinstance(index.to_raw(), slice):
        return None
    return index.size(current)


#: ``isel``/``sel`` keyword arguments that are *options*, not dim indexers.
_SELECT_OPTION_KWARGS = frozenset({"drop", "missing_dims", "method", "tolerance"})

#: ``chunk`` keyword arguments that are *options*, not per-dim chunk specs.
_CHUNK_OPTION_KWARGS = frozenset(
    {
        "name_prefix",
        "token",
        "lock",
        "inline_array",
        "chunked_array_type",
        "from_array_kwargs",
    }
)


def to_opnode(
    name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> FluentOp:
    """Normalise one recorded call into a resolved :data:`~xrexpr.ir.Op` variant.

    Parameters
    ----------
    name : str
        The method name as called, e.g. ``"mean"``, ``"isel"`` or ``"__getitem__"``.
    args : tuple
        The call's positional arguments.
    kwargs : Mapping
        The call's keyword arguments.

    Returns
    -------
    FluentOp
        The variant this call is: a :class:`~xrexpr.ir.Reduce`,
        :class:`~xrexpr.ir.Select`, :class:`~xrexpr.ir.Project`,
        :class:`~xrexpr.ir.Rechunk`, :class:`~xrexpr.ir.Scan`,
        :class:`~xrexpr.ir.ContextOpen`, or :class:`~xrexpr.ir.Opaque` for anything
        untabulated.

    Notes
    -----
    A pure function of the call itself: every kind below is settled by the method name
    and the shape of its arguments, so nothing here reads a schema. That is deliberate
    — the one case that would otherwise need one, a bare ``mean()``, records the symbolic
    :data:`~xrexpr.ir.ALL_DIMS` instead of an eagerly expanded dim set, leaving the
    expansion to a reader whose schema is exact.

    - **reduce** (``mean``/``sum``/...): the dim spec — positional (``mean("lat")``),
      keyword (``mean(dim="lat")``) or tuple (``mean(("lat", "lon"))``) — collapses to
      one ``consumes`` frozenset; a **no-dim ``mean()`` consumes** :data:`~xrexpr.ir.ALL_DIMS`,
      which is what fixes the empty-dim reorder bug (``ds.mean().isel(...)``).
    - **select** (``isel``/``sel``): the indexer (a positional dict and/or kwargs,
      minus option kwargs like ``drop``) becomes ``indexer``; a dim given a *scalar*
      index is dropped and so also lands in ``consumes`` (a slice/list/array keeps it).
    - **project** (``ds["tas"]`` / ``ds[["tas", "pr"]]``): the key's variable names
      become ``variables``. This is the one kind the ``OP_TABLE`` can't settle —
      ``__getitem__`` is a projection only when its key *names variables*, so a
      mask-style key (a boolean ``DataArray``, a dict) stays ``Opaque``.
    - **rechunk** (``chunk``): the *mapping* form (a positional dict and/or dim kwargs,
      minus option kwargs like ``token``) becomes ``chunks``. The uniform forms
      (``chunk()``, ``chunk(100)``, ``chunk("auto")``) name no dim, so ``chunks`` is
      empty and the spec stays in ``args``.
    - **context opener** (``groupby``/``rolling``/``weighted``/...): a
      :class:`~xrexpr.ir.ContextOpen`, which says only *a context opens here*. Checked
      before the table, and the one kind whose meaning this function deliberately does
      not settle — the call is half an operation, and the pair is
      ``lower.to_lower_ir``'s to read.
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

    if name in CONTEXT_METHODS:
        # Decidable per-call even though what it *means* is not: a builder-returning
        # method opens a context whatever follows it. Pairing is lowering's job.
        return ContextOpen(
            name=cast(ContextOpenName, name),
            args=args,
            kwargs=kw,
        )
    if kind == "reduce":
        return Reduce(
            name=name,
            args=args,
            kwargs=kw,
            consumes=_reduce_dims(args, kwargs),
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
    if kind == "rechunk":
        return Rechunk(
            name=cast(Literal["chunk"], name),
            args=args,
            kwargs=kw,
            chunks=_chunk_spec(args, kwargs),
        )
    if name == "__getitem__" and (variables := _projected_names(args)) is not None:
        return Project(name="__getitem__", args=args, kwargs=kw, variables=variables)
    return Opaque(name=name, args=args, kwargs=kw)


def _projected_names(args: tuple[Any, ...]) -> tuple[Hashable, ...] | None:
    """Return the variable names a ``__getitem__`` key selects.

    Parameters
    ----------
    args : tuple
        The ``__getitem__`` call's positional arguments — one key, if it is a projection
        at all.

    Returns
    -------
    tuple of Hashable or None
        The requested names in order, or ``None`` when the key is not a projection.

    Notes
    -----
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


def _reduce_dims(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> DimSet:
    """Parse the dims a reduction removes out of its call.

    Parameters
    ----------
    args : tuple
        The reduction's positional arguments. Reductions take ``dim`` first
        (``.reduce(func, dim)`` aside).
    kwargs : Mapping
        The reduction's keyword arguments, where a ``dim=`` spec takes precedence.

    Returns
    -------
    DimSet
        The named dims as a frozenset, or :data:`~xrexpr.ir.ALL_DIMS` when the call named
        none.

    Notes
    -----
    An unspecified ``dim`` is left symbolic rather than expanded here: which dims exist
    depends on where in the plan the reduce ends up running, and this function is not
    the place that knows.
    """
    if "dim" in kwargs:
        dim = kwargs["dim"]
    elif args:
        dim = args[0]  # reductions take ``dim`` first (``.reduce(func, dim)`` aside)
    else:
        dim = None
    if dim is None:  # bare ``mean()`` / ``mean(dim=None)`` → every dim, resolved later
        return ALL_DIMS
    return _as_dim_set(dim)


def _as_dim_set(dim: Any) -> frozenset[Hashable]:
    """Normalise a dim spec to a set of dim names.

    Parameters
    ----------
    dim : Any
        A single dim name or an iterable of them — xarray's dim convention, where a
        ``str`` is one name rather than a sequence of characters.

    Returns
    -------
    frozenset of Hashable
        The named dims.
    """
    if isinstance(dim, str) or not isinstance(dim, Iterable):
        return frozenset({dim})
    return frozenset(dim)


def _select_indexer(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> frozendict[Hashable, Any]:
    """Extract the ``{dim: index}`` mapping of an ``isel``/``sel`` call.

    Parameters
    ----------
    args : tuple
        The select's positional arguments; a leading dict is the mapping form.
    kwargs : Mapping
        The select's keyword arguments, minus the options in
        :data:`_SELECT_OPTION_KWARGS` (``drop``, ``method``, ...).

    Returns
    -------
    frozendict
        The indexed dims and their raw indexer values.
    """
    indexer: dict[Hashable, Any] = {}
    if args and isinstance(args[0], dict):
        indexer.update(args[0])
    for key, value in kwargs.items():
        if key not in _SELECT_OPTION_KWARGS:
            indexer[key] = value
    return frozendict(indexer)


def _chunk_spec(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> frozendict[Hashable, Any]:
    """Extract the ``{dim: chunksize}`` mapping of a ``chunk()`` call.

    Parameters
    ----------
    args : tuple
        The rechunk's positional arguments; a leading dict is the mapping form.
    kwargs : Mapping
        The rechunk's keyword arguments, minus the options in
        :data:`_CHUNK_OPTION_KWARGS` (``token``, ``lock``, ...).

    Returns
    -------
    frozendict
        The named dims and their chunk specs; empty for the uniform forms.

    Notes
    -----
    Only the mapping form contributes. A uniform positional spec (``chunk(100)``,
    ``chunk("auto")``) names no dim, so it yields an empty mapping and is left to be
    replayed verbatim from ``args`` — which is exactly right, since a uniform spec has
    no dim key that a later select could invalidate.
    """
    chunks: dict[Hashable, Any] = {}
    if args and isinstance(args[0], dict):
        chunks.update(args[0])
    for key, value in kwargs.items():
        if key not in _CHUNK_OPTION_KWARGS:
            chunks[key] = value
    return frozendict(chunks)
