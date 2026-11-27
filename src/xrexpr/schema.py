"""Logical schema tracking, without touching any array data.

The ``.plan`` proxy starts from the real base dataset, so we can cheaply fold a
*logical* schema forward through a plan: every variable and the dims it spans, which of
them are coordinates, and the dim sizes. All of that is metadata, so no data is ever
read. Rules use it to know the shape each node sees, and a bare ``ds.mean()``'s
:data:`~xrexpr.ir.ALL_DIMS` is resolved against it.

Variables are the stored fact and dims are derived from them, the same way
``xr.Dataset`` stores ``_variables``/``_coord_names`` and derives ``.dims``. See
:class:`SchemaState` for why that direction matters.

The optimiser owns the fold (``optimize._schemas``), not recording. :func:`to_opnode` is
a pure function of one call, so nothing gets resolved against a schema that's only a
guess about where the op will end up running.

- :class:`SchemaState` is an immutable snapshot.
- :func:`apply_schema` returns the snapshot after applying one node.
- :func:`to_opnode` turns a raw recorded call into its ``Op`` node.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import xarray as xr
from frozendict import frozendict
from packaging.version import Version
from typing_extensions import assert_never

from xrexpr._xrexprs.ir import ALL_DIMS, AllDims
from xrexpr.indexers import Advanced, Indexer, classify
from xrexpr.intern import InternedVal, Interner
from xrexpr.ir import (
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
from xrexpr.operations import CHUNK_OPTION_KWARGS as _CHUNK_OPTION_KWARGS
from xrexpr.operations import SELECT_OPTION_KWARGS as _SELECT_OPTION_KWARGS
from xrexpr.operations import (
    ContextSpec,
    DropSpec,
    ElementwiseSpec,
    ProjectSpec,
    RechunkSpec,
    ReduceSpec,
    RenameSpec,
    ScanSpec,
    SelectSpec,
)
from xrexpr.operations import spec as op_spec

__all__ = ["SchemaState", "apply_schema", "resolve_dims", "to_opnode"]

#: Before xarray 2026.4.0, ``cumsum``/``cumprod`` dropped every coordinate spanning the
#: scanned dim (fixed by pydata/xarray#10987); ``diff`` never did. The ``Scan`` arm of
#: :func:`apply_schema` reproduces the old behaviour on old versions, so the tracked
#: schema is exact on every supported version.
SCAN_DROPS_SCANNED_COORDS = Version(xr.__version__) < Version("2026.4.0")

#: Before pydata/xarray#11562 (issue #11560), ``Dataset.weighted(...).<reduction>`` kept
#: only the coordinates that some *data variable* carried. It ran ``Dataset.map`` and
#: rebuilt coordinates from the per-variable results, so a coordinate no data variable
#: spans was dropped. Examples: ``time`` once ``drop_vars`` has removed the last
#: variable with ``time``, or an aux ``region(lat)`` next to only ``temperature(time)``.
#: A dim left with nothing but that coordinate went too. The fix keeps every coordinate
#: the reduction doesn't touch, like a plain reduce.
#:
#: The gate is ``2026.9.1.dev86``, the version setuptools-scm gives the merge commit
#: (``dev85`` still drops). So builds from xarray ``main`` are modelled correctly too,
#: and the gate won't need editing at the next release, whatever it's called. The
#: ``WeightedReduce`` arm of :func:`apply_schema` reproduces the old behaviour on older
#: versions, as ``SCAN_DROPS_SCANNED_COORDS`` does for scans.
WEIGHTED_DROPS_UNCARRIED_COORDS = Version(xr.__version__) < Version("2026.9.1.dev86")

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class SchemaState(Generic[T]):
    """An immutable snapshot of a dataset's logical shape at one point in a plan.

    Attributes
    ----------
    variables : frozendict
        **Every** variable, coordinate or not, mapped to the dims it spans. This is the
        stored fact the rest is derived from.
    coord_names : frozenset of Hashable
        Which of those names are coordinates. That's a *role*, not a kind of variable:
        ``xr.Dataset`` keeps the same split (``_variables`` plus ``_coord_names``), and
        ``reset_coords`` moves a name between them without changing the variable.
    sizes : frozendict
        Dim sizes only, ``{dim: size}``, for the dims :attr:`dim_names` derives. It
        answers *how big*, never *which dims exist*. There's deliberately no ``dims``
        attribute, because that name could mean either. A size of ``None`` means **don't
        know**; see the notes.

    Notes
    -----
    All three fields are converted to immutable containers on construction, so a
    snapshot is hashable and safe to share across the plan. It holds metadata only,
    never data.

    **Which dims exist is derived, not stored** (:attr:`dim_names`). A dim exists
    exactly when some variable spans it. ``Dataset.dims`` is derived upstream for the
    same reason. Storing it separately would store the same fact twice, and a schema
    could then say ``dims={"time": 4}`` while a variable spans ``("time", "lat")``: a
    phantom ``lat`` that ``var_dims`` would report to any rule that asked. Deriving it
    makes that state impossible to build.

    :attr:`data_vars` is derived too (:attr:`variables` minus :attr:`coord_names`). It's
    what makes variable-level rules possible: whether a projection may move left past an
    op depends on whether the kept variables still carry the dims that op names.

    Storing coordinates as variables with dims, rather than as bare names, is what lets
    :func:`apply_schema` track their lifetimes. An aggregating op drops a coordinate
    over the dim it aggregates, while an indexing op keeps it (as 0-d). You can't
    express that if a coordinate has no dims to lose.

    A size of ``None`` means **don't know**: the dim exists, but its length isn't known
    statically. Callers must treat it as "no rewrite", never as size zero.
    Under-reporting a size is the dangerous direction, because a rule could act on it.
    Currently no rule reads sizes at all (every rule reasons about dim *names*), and
    ``test_rewrites_survive_unknown_dim_sizes`` checks that by blanking every size and
    requiring the same output.
    """

    variables: frozendict[T, tuple[T, ...]] = field(default_factory=frozendict)
    coord_names: frozenset[T] = frozenset()
    sizes: frozendict[T, int | None] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "variables",
            frozendict({k: tuple(v) for k, v in self.variables.items()}),
        )
        object.__setattr__(self, "coord_names", frozenset(self.coord_names))
        # Drop sizes for dims nothing spans. They'd be unreachable through ``dim_names``
        # anyway, but they would make two equal snapshots compare unequal, and tests
        # compare folded schemas directly (``test_rewrites_survive_unknown_dim_sizes``).
        # Normalising here keeps ``SchemaState`` a proper value.
        spanned = self.dim_names
        object.__setattr__(
            self,
            "sizes",
            frozendict({d: s for d, s in self.sizes.items() if d in spanned}),
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset | xr.DataArray) -> "SchemaState[Hashable]":
        """Snapshot a dataset's logical schema, reading no data.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            The object to read. Only ``.sizes``, ``.coords`` and each variable's dims
            are read. All of that is metadata, so a dask-backed dataset computes
            nothing.

        Returns
        -------
        SchemaState
            The snapshot.

        Notes
        -----
        Coordinates and data variables go into one mapping, as ``Dataset._variables``
        holds them. A ``DataArray`` contributes only its coords: it has no
        ``data_vars``, so there's nothing to project (see ``07-small-wins.md`` §5).
        """
        variables = {k: tuple(v.dims) for k, v in ds.coords.items()}
        if isinstance(ds, xr.Dataset):
            variables |= {k: tuple(v.dims) for k, v in ds.data_vars.items()}
        return SchemaState[Hashable](
            variables=frozendict(variables),
            coord_names=frozenset(ds.coords),
            sizes=frozendict(ds.sizes),
        )

    def var_dims(self, names: Iterable[T]) -> frozenset[T] | None:
        """Return the union of the dims carried by ``names``.

        Parameters
        ----------
        names : iterable of T
            Data variable names, e.g. the ones a :class:`~xrexpr.ir.Project` keeps.

        Returns
        -------
        frozenset of T or None
            The union of those variables' dims, or ``None`` if any name isn't a tracked
            data variable.

        Notes
        -----
        ``None`` means *don't know*. The name might be a coordinate, or something an
        unmodelled op created. Callers must treat it as "no rewrite", never as "no
        dims".

        Only :attr:`data_vars` count, on purpose. The callers are the projection rules,
        and they only model projections of data variables. A coordinate answers ``None``
        even though its dims are known.
        """
        dims: set[T] = set()
        data_vars = self.data_vars
        for name in names:
            if name not in data_vars:
                return None
            dims.update(data_vars[name])
        return frozenset(dims)

    @property
    def dim_names(self) -> frozenset[T]:
        """The dims that exist: every dim some variable spans. Derived, never stored.

        Returns
        -------
        frozenset of T
            The union of every variable's dims. This is what :data:`~xrexpr.ir.ALL_DIMS`
            resolves to.
        """
        return frozenset(d for var_dims in self.variables.values() for d in var_dims)

    @property
    def data_vars(self) -> frozendict[T, tuple[T, ...]]:
        """The non-coordinate variables and their dims, derived from :attr:`variables`.

        Returns
        -------
        frozendict
            :attr:`variables` minus :attr:`coord_names`.
        """
        return frozendict(
            {k: v for k, v in self.variables.items() if k not in self.coord_names}
        )

    @property
    def coords(self) -> frozenset[T]:
        """The coordinate names. An alias for :attr:`coord_names`.

        Returns
        -------
        frozenset of T
            :attr:`coord_names`, unchanged.
        """
        return self.coord_names

    def to_interned(self, interner: Interner[T]) -> "SchemaState[InternedVal]":
        """Return a copy with every *name* replaced by an :class:`InternedVal` handle.

        Names become handles: variable names, the dims in each variable's tuple,
        coordinate names, and the keys of ``sizes``. The *values* of ``sizes`` stay
        plain ``int``, because they're lengths, not names. This is the same name/value
        split :mod:`xrexpr.intern.converters` makes for ops.

        Parameters
        ----------
        interner : Interner[T]
            The interner that maps each name to its handle.

        Returns
        -------
        SchemaState[InternedVal]
            The schema with every name replaced, values unchanged.
        """
        int_variables = frozendict(
            {
                InternedVal(interner(v)): tuple(InternedVal(interner(d)) for d in dims)
                for v, dims in self.variables.items()
            }
        )
        int_coord_names = frozenset(InternedVal(interner(c)) for c in self.coord_names)
        int_sizes = frozendict(
            {InternedVal(interner(d)): s for d, s in self.sizes.items()}
        )
        return SchemaState[InternedVal](
            variables=int_variables, coord_names=int_coord_names, sizes=int_sizes
        )

    @classmethod
    def from_interned(
        cls, interned_schema: "SchemaState[InternedVal]", interner: Interner[Hashable]
    ) -> "SchemaState[Hashable]":
        """Turn an interned schema back into one with real names.

        Parameters
        ----------
        interned_schema : SchemaState[InternedVal]
            The interned schema.
        interner : Interner[Hashable]
            The interner that maps each handle back to its name.

        Returns
        -------
        SchemaState[Hashable]
            The schema with every handle replaced by its name.
        """
        deint_variables = frozendict(
            {
                interner[v]: tuple(interner[d] for d in dims)
                for v, dims in interned_schema.variables.items()
            }
        )
        deint_coord_names = frozenset(interner[c] for c in interned_schema.coord_names)
        deint_sizes = frozendict(
            {interner[d]: s for d, s in interned_schema.sizes.items()}
        )
        return SchemaState[Hashable](
            variables=deint_variables, coord_names=deint_coord_names, sizes=deint_sizes
        )


def resolve_dims(
    consumes: DimSet, dim_names: frozenset[Hashable]
) -> frozenset[Hashable]:
    """Expand a dim spec into the concrete set of dim names it stands for.

    A :data:`~xrexpr.ir.DimSet` is either an explicit ``frozenset`` of dim names, or the
    :data:`~xrexpr.ir.ALL_DIMS` sentinel meaning *every dim present at this point in the
    plan*. This turns the sentinel into a set, and returns an explicit set unchanged.

    Parameters
    ----------
    consumes : DimSet
        The dim spec to expand, as stored on a node.
    dim_names : frozenset of Hashable
        The dims that exist where ``consumes`` applies: :attr:`SchemaState.dim_names` of
        the schema *going into* the node.

    Returns
    -------
    frozenset of Hashable
        ``dim_names`` when ``consumes`` is :data:`~xrexpr.ir.ALL_DIMS`, otherwise
        ``consumes``.

    See Also
    --------
    xrexpr.ir.AllDims : The sentinel, and why it stays unexpanded until something reads
    it.

    Notes
    -----
    ``ds.mean()`` names no dim, and what it means depends on where it runs: every dim
    the dataset has *at that point*. The recorder can't know that, so ``to_opnode``
    records the sentinel instead of guessing. It stays in the plan until something with
    an exact schema expands it. This function is that something.

    Only call it where the schema is exact. ``optimize``'s rules stay inside the trusted
    prefix (``optimize._trusted_prefix``) for this reason. Past the first
    :class:`~xrexpr.ir.Opaque` the folded schema is a guess, and expanding against a
    guess would silently widen or narrow what a bare reduce consumes.

    It takes dim *names* rather than a :class:`SchemaState` because that's all it needs.
    It also keeps the ``DimSet`` ``assert_never`` in one place: a third kind of dim set
    fails type-checking here once, rather than at every site that matches on it.

    Examples
    --------
    An explicit set is returned as is:

    >>> resolve_dims(frozenset({"lat"}), frozenset({"time", "lat"}))
    frozenset({'lat'})

    The sentinel becomes whatever dims are present:

    >>> resolve_dims(ALL_DIMS, frozenset({"time", "lat"})) == frozenset({"time", "lat"})
    True

    So the same node resolves differently at different points in a plan, which is why
    it's deferred:

    >>> resolve_dims(ALL_DIMS, frozenset({"time"}))
    frozenset({'time'})
    """
    match consumes:
        case AllDims():
            return dim_names
        case frozenset() as named:
            return named
        case _:
            assert_never(consumes)


def apply_schema(
    schema: SchemaState[Hashable], node: LoweredOp
) -> SchemaState[Hashable]:
    """Return the schema after applying ``node`` to ``schema``.

    Parameters
    ----------
    schema : SchemaState[Hashable]
        The schema *going into* the node.
    node : LoweredOp
        The node to apply.

    Returns
    -------
    SchemaState[Hashable]
        The next snapshot: the schema the following node sees.

    Notes
    -----
    Each arm says what happens to the **variables**. The dims follow automatically,
    because :attr:`SchemaState.dim_names` is derived. No arm removes a dim directly, so
    no arm can leave dims and variables disagreeing.

    Two helpers do most of the work. Which one an op uses is *the* difference between op
    families (both checked against xarray 2026.7.0; see each helper):

    - ``_aggregated``: data variables lose the dims, and coordinates spanning them are
      **dropped**. Used by :class:`~xrexpr.ir.Reduce`,
      :class:`~xrexpr.ir.GroupedReduce`, :class:`~xrexpr.ir.WeightedReduce`, and a
      ``Select`` with ``drop=True``.
    - ``_indexed``: every variable loses the dims and **nothing is dropped**, so a
      coordinate left with no dims becomes 0-d. Used by an ordinary
      :class:`~xrexpr.ir.Select`.

    Per node kind:

    - :class:`~xrexpr.ir.Reduce` aggregates over ``consumes``, or over every dim when
      that is :data:`~xrexpr.ir.ALL_DIMS`. This is where a bare reduce finally gets
      expanded, against a schema that's exact. With ``keepdims=True`` the dims stay, at
      size 1.
    - :class:`~xrexpr.ir.Select` also *resizes* the dims it keeps (slices and
      sequences).
    - :class:`~xrexpr.ir.Project` keeps only the named variables, plus the coordinates
      they still span. Dims nothing spans any more simply stop being derived. An unknown
      name leaves the whole schema alone; see the comment on that arm.
    - :class:`~xrexpr.ir.GroupedReduce` aggregates over ``group_dim`` plus any extra
      ``consumes``, adds ``new_dim`` to every data variable, and usually adds
      ``new_dim`` as a coordinate of *unknown* size (the number of groups depends on
      coordinate values).
    - :class:`~xrexpr.ir.WindowedReduce` keeps every dim and resizes the windowed ones
      (``coarsen`` only).
    - :class:`~xrexpr.ir.WeightedReduce` aggregates over ``consumes`` like a plain
      reduce, then adds each surviving weight dim to every variable that lacks it and
      marks every weight dim's size unknown (the weights broadcast into dims a variable
      lacks, and align, so possibly shrink, the ones it has). On xarray before
      pydata/xarray#11562 (:data:`WEIGHTED_DROPS_UNCARRIED_COORDS`) it first drops every
      coordinate no data variable spans, reproducing that version's behaviour.
    - :class:`~xrexpr.ir.Scan` mostly changes nothing. ``diff`` shrinks its dim. On
      xarray before 2026.4.0 (:data:`SCAN_DROPS_SCANNED_COORDS`), ``cumsum``/``cumprod``
      drop every coordinate over the scanned dim while keeping the dim, reproducing that
      bug.
    - :class:`~xrexpr.ir.Rename` relabels dims and/or variables (``_relabelled``).
    - :class:`~xrexpr.ir.Drop` removes the named variables and coordinates.
    - :class:`~xrexpr.ir.Elementwise` changes nothing, and that's exact: it's
      per-element.
    - :class:`~xrexpr.ir.Rechunk` changes nothing (only chunk layout changes).
    - :class:`~xrexpr.ir.Opaque` is *assumed* to change nothing.

    That last assumption is often false (``assign``, ``stack``, ...). So the schema is
    only exact up to the first opaque op. The optimiser enforces that boundary, not this
    function (see ``optimize._trusted_prefix``). Inside it the schema really is exact,
    coordinates included: ``test_tracked_schema_agrees_with_evaluation`` checks the
    tracked variables and coordinates against a real evaluation for equality.

    ``assert_never`` on the last arm means a new node kind fails type-checking here
    until it's handled.
    """
    variables = dict(schema.variables)
    coord_names = set(schema.coord_names)
    sizes = dict(schema.sizes)

    match node:
        case Reduce(consumes=consumes) as reduce if reduce.keepdims:
            # ``keepdims=True`` keeps every named dim, at size 1. Data variables keep
            # their dims; only coordinates spanning a reduced dim are dropped (a mean
            # over ``time`` has no meaningful ``time`` label; checked against xarray
            # 2026.7.0). The plain reduce below removes the dims from the variables too.
            over = resolve_dims(consumes, schema.dim_names)
            variables, coord_names = _keepdims_reduced(variables, coord_names, over)
            for dim in over:
                sizes[dim] = 1
        case Reduce(consumes=consumes):
            variables, coord_names = _aggregated(
                variables, coord_names, resolve_dims(consumes, schema.dim_names)
            )
        case Select(indexer=indexer) as select:
            # ``drop=True`` decides which coordinate rule applies, and this is the only
            # place the arm needs to know. Without it, a scalar select keeps the coords
            # over its dim as 0-d (``isel(lat=0)`` keeps ``lat`` and ``region``, both
            # scalar). With it, they're dropped (checked against xarray 2026.7.0). This
            # is modelled exactly because keeping a coord xarray drops would
            # over-report, and the tests compare coords to the evaluated result.
            if select.kwargs.get("drop", False):
                variables, coord_names = _aggregated(
                    variables, coord_names, select.consumes
                )
            else:
                variables = _indexed(variables, select.consumes)
            # A non-scalar select keeps its dim but changes its size. (Scalar-indexed
            # dims are already gone, via ``consumes``.)
            for dim, index in indexer.items():
                if dim not in select.consumes and dim in sizes:
                    sizes[dim] = _selected_size(select.name, index, sizes[dim])
        case Project(variables=names):
            # Only apply the projection if every name is a *tracked* data variable. A
            # name this layer doesn't know (a coord, or something an unmodelled op
            # created) could make the surviving dims an under-report, which is the
            # dangerous direction. So with an unknown name, leave the schema alone
            # entirely. That over-reports instead.
            #
            # A projection that names a *coordinate* also lands here, even though the
            # coord is known. The tracked schema then over-reports (it claims data
            # variables survive that evaluation drops). That's safe, because no rule
            # reads the schema after a coordinate projection, but it isn't exact.
            # Modelling it is #162.
            if all(n in schema.data_vars for n in names):
                kept = {n: variables[n] for n in names}
                # xarray keeps only the coordinates the *selected* variables still span,
                # so ``ds[["elevation"]]`` (dims ``lat``, ``lon``) drops the ``time``
                # coordinate along with the ``time`` dim. A scalar coord (no dims) is
                # spanned by anything, so it stays. Checked against xarray 2026.7.0.
                spanned = {d for var_dims in kept.values() for d in var_dims}
                coord_names = {c for c in coord_names if set(variables[c]) <= spanned}
                variables = {c: variables[c] for c in coord_names} | kept
        case GroupedReduce() as grouped:
            variables, coord_names = _aggregated(
                variables,
                coord_names,
                frozenset({grouped.group_dim}) | grouped.consumes,
            )
            variables = _minted(variables, coord_names, frozenset({grouped.new_dim}))
            # The group labels usually become a coordinate of their own. That's specific
            # to grouped reduces, and it's why this arm adds a *variable*, not just a
            # dim (a broadcast weight dim, below, needn't have a coord at all).
            #
            # But xarray only creates that coordinate when the grouper had one:
            #
            # - A coordinate or component grouper always does. ``groupby("region")`` and
            #   ``groupby("time.month")`` both give a ``new_dim`` different from
            #   ``group_dim``.
            # - Grouping a bare *dimension* with no coordinate (``new_dim ==
            #   group_dim``) creates no coordinate; xarray leaves the dim without one
            #   (checked on 2025.6.1 and 2026.7.0). That state can arise after a
            #   pre-2026.4.0 scan drops the scanned dim's coordinate
            #   (:data:`SCAN_DROPS_SCANNED_COORDS`).
            if (
                grouped.new_dim != grouped.group_dim
                or grouped.group_dim in schema.coords
            ):
                variables[grouped.new_dim] = (grouped.new_dim,)
                coord_names = coord_names | {grouped.new_dim}
            # The new dim's size is the number of groups. That depends on coordinate
            # *values*, which this layer doesn't read, so it's ``None`` rather than a
            # guess.
            sizes[grouped.new_dim] = None
        case WeightedReduce() as weighted:
            if WEIGHTED_DROPS_UNCARRIED_COORDS:
                # Use the schema *going into* the node: ``Dataset.map`` passes each data
                # variable over as a DataArray carrying the coordinates its own dims
                # span, and the result's coordinates are rebuilt from those alone. See
                # the constant.
                uncarried = coord_names - _carried_coords(variables, coord_names)
                coord_names -= uncarried
                variables = {k: v for k, v in variables.items() if k not in uncarried}
            match weighted.consumes:
                case AllDims():
                    # A bare weighted closer, like a bare ``mean()``, removes every dim,
                    # weight dims included.
                    variables, coord_names = _aggregated(
                        variables, coord_names, schema.dim_names
                    )
                case frozenset() as named:
                    variables, coord_names = _aggregated(variables, coord_names, named)
                    # The weights have a dim effect of their own, via ``dot``'s
                    # broadcasting and alignment (see ``WeightedReduce``). A weight dim
                    # a variable lacks is added to it. One it has may be *resized*,
                    # because misaligned weights inner-join and shrink it. Either way
                    # the name is known and the size isn't, so the size is ``None``.
                    #
                    # Added per variable, not per dataset, because ``dot`` broadcasts
                    # each variable separately. Skipping the add because some *other*
                    # variable already has the dim would under-report ``elevation``
                    # after ``weighted(w(time)).mean("lat")`` on a dataset that also
                    # holds ``temperature(time, lat)``. ``_minted`` works per variable
                    # and is idempotent, so it just gets the whole set.
                    from_weights = frozenset(weighted.weight_dims) - named
                    variables = _minted(variables, coord_names, from_weights)
                    for dim in from_weights:
                        sizes[dim] = None
                case _:
                    assert_never(weighted.consumes)
        case WindowedReduce() as windowed:
            for dim, window in windowed.window.items():
                if dim in sizes:
                    sizes[dim] = _windowed_size(windowed, sizes[dim], window)
        case Scan(name="diff", dims=frozenset() as dims) as scan:
            # ``diff(dim, n=1)`` shrinks ``dim`` by ``n``; ``label`` only picks which
            # end. ``diff`` requires its dim positionally, so ``dims`` is always a
            # concrete set here. A bare ``cumsum()`` (``ALL_DIMS``) falls through to the
            # size-preserving ``pass`` below, as do ``cumsum``/``cumprod`` generally. No
            # rule reads sizes, but the schema shouldn't carry a known wrong value.
            n = scan.kwargs.get("n", scan.args[1] if len(scan.args) > 1 else 1)
            for dim in dims:
                size = sizes.get(dim)
                if size is not None:  # an unknown length stays unknown, never guessed
                    sizes[dim] = max(size - n, 0)
        case Scan(name="cumsum" | "cumprod", dims=dims) if SCAN_DROPS_SCANNED_COORDS:
            # Before 2026.4.0, xarray ran ``cumsum``/``cumprod`` through
            # ``Dataset.reduce``, which keeps a coordinate only if it spans *none* of
            # the reduced dims. So every coordinate over the scanned dim is dropped
            # (index coord or not), while data variables keep the dim (a scan preserves
            # shape). The dim then survives as a "dimension without coordinates".
            #
            # pydata/xarray#10987 (the flox migration) changed it to keep coords. On
            # those versions ``SCAN_DROPS_SCANNED_COORDS`` is false, this arm is
            # skipped, and the op falls through to the ``pass`` below. ``diff`` never
            # had the bug; its arm above keeps coords on every version. Unlike
            # ``_aggregated``, this only drops coordinate variables and leaves data
            # variables and sizes alone.
            over = resolve_dims(dims, schema.dim_names)
            dropped = {c for c in coord_names if not over.isdisjoint(variables[c])}
            coord_names -= dropped
            variables = {k: v for k, v in variables.items() if k not in dropped}
        case Rename(mapping=renames):
            # ``rename`` relabels a dim, a variable/coordinate, or (for a dimension
            # coordinate) both. ``_relabelled`` does it as one shape-preserving relabel,
            # so it's exact, coordinates included.
            variables, coord_names, sizes = _relabelled(
                variables, coord_names, sizes, renames
            )
        case Drop(variables=names):
            # ``drop_vars`` removes variables/coordinates by *name*, not by dim, so this
            # isn't ``_aggregated``. A dim nothing spans afterwards (its last variable
            # dropped, no coordinate on it) is removed by ``SchemaState.__post_init__``.
            # If you drop a dimension coordinate while a data variable still spans the
            # dim, the dim survives without a coordinate, as in xarray. Checked against
            # xarray 2026.7.0 for a data variable, an auxiliary coordinate and a
            # dimension coordinate.
            dropped = set(names)
            variables = {k: v for k, v in variables.items() if k not in dropped}
            coord_names -= dropped
        case Elementwise() | Scan() | Rechunk() | Opaque():
            pass
        case _:
            assert_never(node)

    return SchemaState[Hashable](
        variables=frozendict(variables),
        coord_names=frozenset(coord_names),
        sizes=frozendict(sizes),
    )


def _aggregated(
    variables: dict[Hashable, tuple[Hashable, ...]],
    coord_names: set[Hashable],
    over: frozenset[Hashable],
) -> tuple[dict[Hashable, tuple[Hashable, ...]], set[Hashable]]:
    """Apply an *aggregating* op: variables lose the dims, spanning coordinates go.

    Parameters
    ----------
    variables : dict
        The schema's variables going into the op, name to dims.
    coord_names : set of Hashable
        Which of those names are coordinates.
    over : frozenset of Hashable
        The dims being aggregated away.

    Returns
    -------
    tuple of (dict, set)
        The surviving variables with ``over`` removed from their dims, and which of them
        are coordinates.

    Notes
    -----
    This asymmetry is xarray's (checked against 2026.7.0). ``ds.mean("lat")`` reduces
    ``tas(time, lat)`` to ``tas(time)`` but *removes* a ``region(lat)`` coordinate,
    since there's no meaningful average of a coordinate's labels. Compare ``_indexed``,
    where nothing is dropped.

    A coordinate that spans none of ``over`` is untouched. That's why ``time`` survives
    ``groupby("lat").mean()`` while ``region(lat)`` doesn't.
    """
    kept = {
        name: tuple(d for d in var_dims if d not in over)
        for name, var_dims in variables.items()
        if not (name in coord_names and not over.isdisjoint(var_dims))
    }
    return kept, coord_names & set(kept)


def _carried_coords(
    variables: dict[Hashable, tuple[Hashable, ...]], coord_names: set[Hashable]
) -> set[Hashable]:
    """Name the coordinates some data variable carries (as ``ds[name]`` attaches them).

    Parameters
    ----------
    variables : dict
        The schema's variables, name to dims.
    coord_names : set of Hashable
        Which of those names are coordinates.

    Returns
    -------
    set of Hashable
        Every coordinate whose dims are a subset of at least one data variable's dims. A
        0-d coordinate is carried by any data variable, and by nothing if there are
        none.

    Notes
    -----
    This is the coordinate set ``Dataset.map`` rebuilds its result from, which is what
    weighted reductions returned before pydata/xarray#11562 (see
    :data:`WEIGHTED_DROPS_UNCARRIED_COORDS`). Checked against xarray 2026.7.0.
    """
    data_dims = [set(d) for n, d in variables.items() if n not in coord_names]
    return {
        c for c in coord_names if any(set(variables[c]) <= dims for dims in data_dims)
    }


def _indexed(
    variables: dict[Hashable, tuple[Hashable, ...]], over: frozenset[Hashable]
) -> dict[Hashable, tuple[Hashable, ...]]:
    """Apply an *indexing* op: every variable loses the dims, and nothing is dropped.

    Parameters
    ----------
    variables : dict
        The schema's variables going into the op, name to dims.
    over : frozenset of Hashable
        The dims being indexed away.

    Returns
    -------
    dict
        Every variable, with ``over`` removed from its dims.

    Notes
    -----
    A coordinate left with no dims is a 0-d coordinate, which is what xarray gives for
    ``isel(lat=0)``: ``lat`` and ``region`` both survive as scalars. This treats
    coordinates and data variables the same way, which is why they share one store.
    """
    return {
        name: tuple(d for d in var_dims if d not in over)
        for name, var_dims in variables.items()
    }


def _keepdims_reduced(
    variables: dict[Hashable, tuple[Hashable, ...]],
    coord_names: set[Hashable],
    over: frozenset[Hashable],
) -> tuple[dict[Hashable, tuple[Hashable, ...]], set[Hashable]]:
    """Apply a ``keepdims`` reduction: dims stay at size 1, spanning coordinates go.

    Parameters
    ----------
    variables : dict
        The schema's variables going into the op, name to dims.
    coord_names : set of Hashable
        Which of those names are coordinates.
    over : frozenset of Hashable
        The dims being reduced and kept at size 1.

    Returns
    -------
    tuple of (dict, set)
        Every variable with its dims **unchanged**, minus coordinates spanning ``over``,
        and which of the survivors are coordinates.

    Notes
    -----
    Unlike ``_aggregated``, data variables keep their dims; the caller sets those dims'
    sizes to 1. Coordinates go the same way as in ``_aggregated``: one spanning a
    reduced dim is dropped, since its labels mean nothing after the reduction. For
    example, ``time`` and a non-dim ``ref(time)`` both go from ``mean("time",
    keepdims=True)`` (checked against xarray 2026.7.0).
    """
    kept = {
        name: var_dims
        for name, var_dims in variables.items()
        if not (name in coord_names and not over.isdisjoint(var_dims))
    }
    return kept, coord_names & set(kept)


def _minted(
    variables: dict[Hashable, tuple[Hashable, ...]],
    coord_names: set[Hashable],
    new_dims: frozenset[Hashable],
) -> dict[Hashable, tuple[Hashable, ...]]:
    """Add ``new_dims`` to every **data** variable.

    Parameters
    ----------
    variables : dict
        The schema's variables after the op's own dim removal, name to dims.
    coord_names : set of Hashable
        Which of those names are coordinates. Coordinates don't get the new dims.
    new_dims : frozenset of Hashable
        The dims the op creates.

    Returns
    -------
    dict
        The variables, with each data variable now starting with the new dims in ``str``
        order.

    Notes
    -----
    Checked against xarray 2026.7.0. The surprising part: *every* data variable gets the
    new dim, including ones that couldn't have had the original. ``elev(lat)`` under
    ``groupby("time.month")`` becomes ``(month, lat)``, and under ``weighted(w)`` with
    ``w(lat, member)`` it becomes ``(member,)``. So this adds a dim to each variable
    rather than swapping one dim for another.

    Coordinates don't get it. Existing ones keep their dims (``region`` stays ``(lat,)``
    through ``groupby("time.month")``), and the coordinate a grouped reduce adds is a
    new variable spanning only itself.

    Sorted because ``new_dims`` is a set and :class:`SchemaState` is compared by value.
    A dim tuple that depended on set iteration order could make two identical folds
    unequal across processes. ``str`` is the sort key because dim names are only
    ``Hashable``, and ``str`` is the one total order available. Dim order doesn't
    otherwise matter here (``var_dims`` returns a set, and ``Dataset.sizes`` promises no
    order).
    """
    if not new_dims:
        return variables
    ordered = sorted(new_dims, key=str)
    return {
        name: (
            var_dims
            if name in coord_names
            else (*ordered, *(d for d in var_dims if d not in new_dims))
        )
        for name, var_dims in variables.items()
    }


def _relabelled(
    variables: dict[Hashable, tuple[Hashable, ...]],
    coord_names: set[Hashable],
    sizes: dict[Hashable, int | None],
    mapping: Mapping[Hashable, Hashable],
) -> tuple[
    dict[Hashable, tuple[Hashable, ...]], set[Hashable], dict[Hashable, int | None]
]:
    """Apply a ``rename``: relabel dims and/or variable names, keeping every shape.

    Parameters
    ----------
    variables : dict
        The schema's variables going into the op, name to dims.
    coord_names : set of Hashable
        Which of those names are coordinates.
    sizes : dict
        The dim sizes going into the op, dim to size or ``None``.
    mapping : Mapping
        The ``{old: new}`` mapping, as ``rename`` was called.

    Returns
    -------
    tuple of (dict, set, dict)
        The variables, coordinate names and sizes after the rename.

    Notes
    -----
    This is the third schema primitive alongside :func:`_aggregated` (drop) and
    :func:`_minted` (add), and the only one that changes the *keys* of ``variables`` and
    ``coord_names``. It can't be built from the other two: ``_minted`` adds a dim to
    every data variable and ``_aggregated`` drops the coordinate, so drop-then-add isn't
    a rename. Doing it as one relabel copies each shape across, so the shape is
    preserved by construction.

    A rename can target three things, and a **dimension coordinate** is all of them at
    once (checked against xarray 2026.7.0):

    - a **dim** (``old`` appears in some variable's dims): rename it in every variable's
      dims and move ``sizes[old]`` to ``sizes[new]``. An unknown size (``None``) stays
      unknown; ``test_rewrites_survive_unknown_dim_sizes`` checks that no size is
      invented.
    - a **variable or coordinate** (``old in variables``): rename the key, and its
      ``coord_names`` membership with it.
    - a **dimension coordinate**: both, because its name is a dim *and* a variable, so
      ``rename({month: time})`` gives ``time`` as a dimension coordinate.

    Everything is renamed in one pass through an ``old -> new`` lookup, not entry by
    entry, so a chained or swapping mapping (``{a: b, b: c}``) never sees a half-renamed
    state. Only mapping entries whose key is an actual dim rename dims; renaming an
    auxiliary coordinate leaves every dims tuple alone.
    """
    dim_names = {d for var_dims in variables.values() for d in var_dims}
    dim_renames = {old: new for old, new in mapping.items() if old in dim_names}

    def _rename_dims(var_dims: tuple[Hashable, ...]) -> tuple[Hashable, ...]:
        return tuple(dim_renames.get(d, d) for d in var_dims)

    relabelled = {
        mapping.get(name, name): _rename_dims(var_dims)
        for name, var_dims in variables.items()
    }
    coords = {mapping.get(c, c) for c in coord_names}
    resized = {dim_renames.get(d, d): s for d, s in sizes.items()}
    return relabelled, coords, resized


def _windowed_size(
    node: WindowedReduce, current: int | None, window: int
) -> int | None:
    """Return the length of a windowed dim after ``node``.

    Parameters
    ----------
    node : WindowedReduce
        The windowed node being applied.
    current : int or None
        The dim's length going in, or ``None`` if already unknown.
    window : int
        The window size ``node`` uses for this dim.

    Returns
    -------
    int or None
        The new length, or ``None`` if it can't be known statically.

    Notes
    -----
    ``rolling`` gives one output per input position, so its dims keep their length.
    ``center`` and ``min_periods`` change values, not shape.

    ``coarsen`` divides, and how it rounds depends on an option. ``boundary="trim"``
    drops the short last block (floor), ``"pad"`` fills it (ceil), and the default
    ``"exact"`` requires the length to divide evenly. Where it does, floor and ceil
    agree; where it doesn't, xarray raises at replay, so the size here never matters.
    The node precomputes this as :attr:`~xrexpr.ir.WindowedReduce.rounds_up` (``True``
    ceil, ``False`` floor, ``None`` unknown), so this function doesn't parse the raw
    call. A ``boundary`` value the node doesn't recognise gives ``None``, so a new
    spelling costs precision, not correctness.
    """
    if current is None or node.name == "rolling":
        return current
    match node.rounds_up:
        case False:
            return current // window
        case True:
            return -(-current // window)  # ceil
        case None:
            return None


def _selected_size(
    name: Literal["isel", "sel"], index: Indexer, current: int | None
) -> int | None:
    """Return a kept dim's size after ``index`` is applied to it.

    Parameters
    ----------
    name : {"isel", "sel"}
        Which kind of select the indexer belongs to. A ``sel`` label slice can't be
        sized here.
    index : Indexer
        The dim's indexer. Must be one that *keeps* its dim; a scalar's dim has already
        been dropped via ``Select.consumes``.
    current : int or None
        The dim's length going in, or ``None`` if already unknown.

    Returns
    -------
    int or None
        The new length, or ``None`` if it can't be known statically.

    Notes
    -----
    The only place a size is *computed*, so the only place that has to handle "unknown".
    Two cases return ``None``:

    - **The input size is unknown.** Every :meth:`~xrexpr.indexers.Indexer.size` except
      the concrete lists needs the current length, so unknown in means unknown out.
      Checking here keeps ``Indexer.size`` typed as ``int`` only, instead of adding ``|
      None`` to six implementations. The only cost is exactness ``Positions``/``Mask``
      could have kept.
    - **A label slice.** ``sel(lat=slice(20, 30))`` names *labels*, inclusive at both
      ends, so its size depends on coordinate values. With integer bounds,
      :func:`~xrexpr.indexers.classify` can't tell it from a positional slice and makes
      a ``ForwardSlice``, whose size reads the bounds as positions. With labels ``[10,
      20, 30, 40]`` that gives **0**, where the real answer is 2. Reporting a dim
      *smaller* than it is is the dangerous direction (a size-based rule would act on
      it), so this returns "don't know". Non-integer label slices also return ``None``,
      rather than ``Label.size``'s keep-the-current-size guess.
    """
    if current is None:
        return None
    if name == "sel" and isinstance(index.to_raw(), slice):
        return None
    return index.size(current)


def to_opnode(
    name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> FluentOp:
    """Turn one recorded call into its :data:`~xrexpr.ir.Op` node.

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
        The node for this call: a :class:`~xrexpr.ir.Reduce`,
        :class:`~xrexpr.ir.Select`, :class:`~xrexpr.ir.Project`,
        :class:`~xrexpr.ir.Rechunk`, :class:`~xrexpr.ir.Scan`,
        :class:`~xrexpr.ir.Elementwise`, :class:`~xrexpr.ir.Rename`,
        :class:`~xrexpr.ir.Drop`, :class:`~xrexpr.ir.ContextOpen`, or
        :class:`~xrexpr.ir.Opaque` for anything untabulated.

    Notes
    -----
    A pure function of the call. Every node kind is decided by the method name and the
    shape of its arguments, so nothing here reads a schema. The one case that would
    otherwise need one, a bare ``mean()``, records the symbolic
    :data:`~xrexpr.ir.ALL_DIMS` instead, and leaves expanding it to code that has an
    exact schema.

    There's one ``match`` over the method's :data:`~xrexpr.operations.OpSpec`, closed
    with ``assert_never``. The spec *is* the node kind, so adding a spec without an arm
    here is a type error rather than a call silently recorded as
    :class:`~xrexpr.ir.Opaque`. Each spec carries its name already typed as the
    ``Literal`` its node expects, so no ``cast`` is needed.

    What each kind records:

    - **reduce** (``mean``/``sum``/...): the dim spec, whether positional
      (``mean("lat")``), keyword (``mean(dim="lat")``) or a tuple (``mean(("lat",
      "lon"))``), becomes one ``consumes`` set. A bare ``mean()`` consumes
      :data:`~xrexpr.ir.ALL_DIMS`, so ``ds.mean().isel(...)`` is rejected rather than
      reordered. Which positional holds the dim comes from
      :attr:`~xrexpr.operations.ReduceSpec.dim_arg`, because ``.reduce(func, dim)`` puts
      a *function* first. ``keepdims=True`` is still a ``Reduce``; the derived
      :attr:`~xrexpr.ir.Reduce.keepdims` says the dims stay at size 1.
    - **select** (``isel``/``sel``): the indexer (a positional dict and/or kwargs, minus
      options like ``drop``) becomes ``indexer``. A dim indexed by a *scalar* is
      dropped, so it also ends up in ``consumes`` (a slice, list or array keeps the
      dim).
    - **project** (``ds["tas"]``, ``ds[["tas", "pr"]]``): the key's names become
      ``variables``. The table can only *nominate* this kind, because ``__getitem__`` is
      a projection only when its key names variables. So this arm has a guard, and a
      mask-like key (a boolean ``DataArray``, a dict) falls through to ``Opaque``.
    - **rechunk** (``chunk``): the per-dim form (a positional dict and/or dim kwargs,
      minus options like ``token``) becomes ``chunks``. The uniform forms (``chunk()``,
      ``chunk(100)``, ``chunk("auto")``) name no dim, so ``chunks`` is empty and the
      spec stays in ``args``.
    - **context opener** (``groupby``/``rolling``/``weighted``/...): a
      :class:`~xrexpr.ir.ContextOpen`, which only says "a context opens here". The call
      is half an operation, and ``lower.to_lower_ir`` decides what the pair means.
    - **elementwise** (``fillna``/``astype``/``round``/...): an
      :class:`~xrexpr.ir.Elementwise`, but **only if every argument is a plain value**
      (``_elementwise_safe``). ``fillna(other_da)`` and ``fillna({"tas": 0})`` don't
      commute with a projection, so they fall through to ``Opaque``.
    - **scan** (``cumsum``/``cumprod``/``diff``): the dim spec, parsed like a reduce's,
      becomes ``dims``.
    - **rename** / **drop_vars**: the ``{old: new}`` mapping, or the dropped names.
    - anything untabulated: :class:`~xrexpr.ir.Opaque`, with just name/args/kwargs.

    ``args``/``kwargs`` are always kept verbatim for replay. The other fields are the
    parsed metadata the optimiser reasons about.
    """
    kw = frozendict(kwargs)

    match op_spec(name):
        case ReduceSpec(dim_arg=position):
            # ``keepdims=True`` is modelled, not refused. The named dims are kept at
            # size 1, which the derived ``Reduce.keepdims`` records. ``dim_effect`` and
            # ``apply_schema`` read it alongside ``consumes`` (the same named dims).
            return Reduce(
                name=name,
                args=args,
                kwargs=kw,
                consumes=_dim_spec(args, kwargs, position),
            )
        case SelectSpec(name=select) if not _select_has_advanced(args, kwargs):
            return Select(
                name=select,
                args=args,
                kwargs=kw,
                indexer=_select_indexer(args, kwargs),
            )
        case ScanSpec(name=scan):
            # Scans take the dim first (``cumsum(dim)``, ``diff(dim)``), so position 0.
            # That's why ``ScanSpec`` has no ``dim_arg`` field.
            return Scan(
                name=scan, args=args, kwargs=kw, dims=_dim_spec(args, kwargs, 0)
            )
        case RechunkSpec(name=rechunk):
            return Rechunk(
                name=rechunk,
                args=args,
                kwargs=kw,
                chunks=_chunk_spec(args, kwargs),
            )
        case RenameSpec(name=rename):
            # ``rename`` is modelled rather than ``Opaque`` so the schema fold stays
            # exact past it. ``_rename_map`` collects the ``{old: new}`` mapping from
            # the positional dict and/or the keyword arguments.
            return Rename(
                name=rename,
                args=args,
                kwargs=kw,
                mapping=frozendict(_rename_map(args, kwargs)),
            )
        case DropSpec(name=drop):
            # ``drop_vars`` is modelled rather than ``Opaque`` so the schema fold stays
            # exact past it. ``_dropped_names`` splits its argument the same way xarray
            # does.
            return Drop(
                name=drop,
                args=args,
                kwargs=kw,
                variables=_dropped_names(args, kwargs),
            )
        case ContextSpec(name=opener):
            # Decidable from this call alone, even though what it means isn't: a builder
            # method opens a context whatever follows it. Pairing it with its closer is
            # lowering's job.
            return ContextOpen(name=opener, args=args, kwargs=kw)
        case ElementwiseSpec(name=ew) if _elementwise_safe(args, kwargs):
            # ``fillna(0)``, ``astype("float32")`` etc. with plain-valued arguments
            # only. Per-element, so every dim, size and variable is kept. An array or
            # per-variable argument fails the guard and falls through to ``Opaque``
            # below.
            return Elementwise(name=ew, args=args, kwargs=kw)
        case ProjectSpec(name=getitem) if (
            variables := _projected_names(args)
        ) is not None:
            return Project(name=getitem, args=args, kwargs=kw, variables=variables)
        case SelectSpec() | ElementwiseSpec() | ProjectSpec() | None:
            # Everything here replays verbatim and is never reordered:
            #
            # - a select with a *vectorized* ``DataArray``/``Variable`` indexer (an
            #   orthogonal one is converted to plain values earlier and stays a
            #   ``Select``). It blocks because a vectorized indexer drops its dim and
            #   creates new ones; see ``_select_has_advanced``;
            # - an elementwise call with an array or per-variable argument;
            # - a ``__getitem__`` whose key names no variable (a mask, a dict);
            # - a method with no table row at all.
            return Opaque(name=name, args=args, kwargs=kw)
        case unreachable:
            assert_never(unreachable)


def _elementwise_safe(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """Report whether every argument to an elementwise call is a plain value.

    Parameters
    ----------
    args : tuple
        The call's positional arguments.
    kwargs : Mapping
        The call's keyword arguments.

    Returns
    -------
    bool
        ``True`` if no argument is array- or variable-shaped, so the call is per-element
        and commutes with any select or projection. ``False`` if one is, so the call
        must be recorded as :class:`~xrexpr.ir.Opaque`.

    Notes
    -----
    A blocklist, not an allowlist. An argument is unsafe only if it's:

    - an ``xr.DataArray``/``xr.Dataset`` (``fillna(other)`` fills from another array's
      values);
    - a ``dict`` (``fillna({"tas": 0})`` is per-variable);
    - a ``list``/``tuple``/``ndarray`` (array-shaped).

    Everything else passes. That's what lets ``astype``'s ``np.dtype`` instances and
    ``type`` objects, and scalar ``clip``/``fillna`` bounds, record as
    :class:`~xrexpr.ir.Elementwise`. An allowlist would have to list every way of
    spelling a dtype; the blocklist only needs the shapes that break commutation.
    ``_projected_names`` uses an argument's shape to decide the node kind in the same
    way (:class:`~xrexpr.ir.Project` vs :class:`~xrexpr.ir.Opaque`).
    """
    unsafe = (xr.DataArray, xr.Dataset, dict, list, tuple, np.ndarray)
    return not any(isinstance(v, unsafe) for v in (*args, *kwargs.values()))


def _select_has_advanced(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """Report whether any of a select's indexers is an advanced (``DataArray``) indexer.

    Parameters
    ----------
    args : tuple
        The select's positional arguments.
    kwargs : Mapping
        The select's keyword arguments.

    Returns
    -------
    bool
        ``True`` if at least one indexer classifies as
        :class:`~xrexpr.indexers.Advanced`, so the whole select must be recorded as
        :class:`~xrexpr.ir.Opaque`. ``False`` if every indexer is one the schema layer
        can handle exactly.

    Notes
    -----
    Only a **vectorized** advanced indexer gets here as
    :class:`~xrexpr.indexers.Advanced`. :func:`_select_indexer` has already converted
    the orthogonal ones (an array named after the dim it indexes) to plain ``.values``,
    so they classify as ordinary positional indexers and their select optimises normally
    (see :func:`_orthogonal_advanced`).

    A vectorized indexer drops its dim and adds the array's dims. No positional indexer
    can express that and this layer doesn't model it, so the select must block. Only an
    :class:`~xrexpr.ir.Opaque` node can do that, because it ends the trusted prefix; a
    ``Select`` is always inside it.

    The decision goes through :func:`~xrexpr.indexers.classify`, the one function that
    builds indexer values, so "what counts as an advanced indexer" is defined in one
    place.
    """
    return any(
        isinstance(classify(v), Advanced)
        for v in _select_indexer(args, kwargs).values()
    )


def _rename_map(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[Hashable, Hashable]:
    """Collect a ``rename`` call's ``{old: new}`` mapping from its args and kwargs.

    Parameters
    ----------
    args : tuple
        The ``rename`` call's positional arguments: a ``{old: new}`` dict, if given
        positionally.
    kwargs : Mapping
        The call's keyword arguments: name mappings given as ``old=new``.

    Returns
    -------
    dict
        The merged ``{old: new}`` mapping.

    Notes
    -----
    ``Dataset.rename`` is ``rename(name_dict=None, **names)``, so a call can give the
    mapping positionally (``rename({"month": "time"})``), as kwargs
    (``rename(month="time")``), or both. ``rename`` has no option kwargs, so every
    keyword is a mapping entry. Taken as is: whether each ``old`` is a dim, a variable
    or both is decided by :func:`_relabelled` against the schema, not here.
    """
    mapping: dict[Hashable, Hashable] = {}
    if args and isinstance(args[0], Mapping):
        mapping.update(args[0])
    for old, new in kwargs.items():  # str keys widen to Hashable one at a time
        mapping[old] = new
    return mapping


def _dropped_names(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Hashable, ...]:
    """Return the names a ``drop_vars`` call removes, in order.

    Parameters
    ----------
    args : tuple
        The ``drop_vars`` call's positional arguments: the names, if given positionally.
    kwargs : Mapping
        The call's keyword arguments: the names may be given as ``names=`` instead.

    Returns
    -------
    tuple of Hashable
        The names being dropped.

    Notes
    -----
    Split the same way ``Dataset.drop_vars`` splits its argument: a ``str`` or any other
    non-iterable is one name (``drop_vars("region")``), and any other iterable is
    several (``drop_vars(["a", "b"])``). That's different from :func:`_projected_names`,
    which follows ``__getitem__``'s rule (a bare tuple is one name there, but several
    here), so the two don't share code.
    """
    names = args[0] if args else kwargs["names"]
    if isinstance(names, str) or not isinstance(names, Iterable):
        return (names,)
    return tuple(names)


def _projected_names(args: tuple[Any, ...]) -> tuple[Hashable, ...] | None:
    """Return the variable names a ``__getitem__`` key selects.

    Parameters
    ----------
    args : tuple
        The ``__getitem__`` call's positional arguments: one key, if it's a projection
        at all.

    Returns
    -------
    tuple of Hashable or None
        The requested names in order, or ``None`` if the key isn't a projection.

    Notes
    -----
    Recognises the two projection spellings, split the same way xarray's
    ``Dataset.__getitem__`` splits them:

    - a **hashable** key is one variable name (``ds["tas"]``). A tuple counts: xarray
      reads it as a single name, not a list.
    - a **list** is several names (``ds[["tas", "pr"]]``).

    Anything else (a dict-like key, which xarray sends to ``isel``, or a boolean
    ``DataArray`` mask) returns ``None``, so the call becomes
    :class:`~xrexpr.ir.Opaque`. Names are taken as given; the rule that would move the
    node checks them against the schema's ``data_vars``.
    """
    if len(args) != 1:
        return None
    key = args[0]
    if isinstance(key, Hashable):
        return (key,)
    if isinstance(key, list) and all(isinstance(k, Hashable) for k in key):
        return tuple(key)
    return None


def _dim_spec(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], position: int
) -> DimSet:
    """Parse the dim spec out of a reduce or scan call.

    Used for both reduces (which *remove* the dims) and scans (which *keep* them). The
    parsing is the same: a ``dim=`` kwarg, otherwise the positional at ``position``,
    otherwise every current dim.

    Parameters
    ----------
    args : tuple
        The call's positional arguments.
    kwargs : Mapping
        The call's keyword arguments. A ``dim=`` here takes precedence.
    position : int
        Where the dim spec sits in ``args``: the reduction's
        :attr:`~xrexpr.operations.ReduceSpec.dim_arg` (0 for every reduction except
        ``.reduce``, whose first positional is a *function*), and 0 for every scan.

    Returns
    -------
    DimSet
        The named dims as a frozenset, or :data:`~xrexpr.ir.ALL_DIMS` if the call named
        none or passed xarray's ``...`` (every dim).

    Notes
    -----
    An unspecified dim is left symbolic rather than expanded here: which dims exist
    depends on where in the plan the op ends up, and this function doesn't know that.
    That covers ``.reduce(func)`` and a bare ``cumsum()`` as well as ``mean()``.
    (``diff`` requires its dim, so it can't be bare.)

    A bare ``...`` (``mean(...)``, ``cumsum(dim=...)``) is xarray's other way of saying
    "every dim", so it records :data:`~xrexpr.ir.ALL_DIMS` too. Read as a name, it would
    become ``{Ellipsis}``: a reduce over no real dim, and the plan would replay the
    wrong call.

    An ``...`` *inside* a collection (``mean(dim=["lat", ...])``) isn't special-cased;
    it stays in the set as a literal element. That matches xarray, which ignores it on a
    Dataset (it reduces nothing) but rejects it on a DataArray (``Ellipsis not found in
    array dimensions``). Keeping it means the replayed call carries it and behaves as it
    would eagerly. Stripping it would quietly turn the DataArray error into a success.

    ``keepdims=True`` needs nothing special here. The named dims are the same set
    whether they're removed or kept at size 1. The derived
    :attr:`~xrexpr.ir.Reduce.keepdims` tells ``dim_effect``/``apply_schema`` which one
    happens.
    """
    if "dim" in kwargs:
        dim = kwargs["dim"]
    elif len(args) > position:
        dim = args[position]
    else:
        dim = None
    # bare ``mean()`` / ``mean(dim=None)`` / ``mean(...)`` → every dim, resolved later
    if dim is None or dim is ...:
        return ALL_DIMS
    return _as_dim_set(dim)


def _as_dim_set(dim: Any) -> frozenset[Hashable]:
    """Normalise a dim spec to a set of dim names.

    Parameters
    ----------
    dim : Any
        A single dim name or an iterable of them. Following xarray, a ``str`` is one
        name, not a sequence of characters.

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
        The select's positional arguments. A leading dict is the mapping form.
    kwargs : Mapping
        The select's keyword arguments. The options in ``_SELECT_OPTION_KWARGS``
        (``drop``, ``method``, ...) are skipped.

    Returns
    -------
    frozendict
        The indexed dims and their raw indexer values, except that an *orthogonal*
        advanced indexer is replaced by its ``.values`` (see
        :func:`_orthogonal_advanced`).

    Notes
    -----
    The recorded call itself is untouched (``to_opnode`` keeps the original
    ``args``/``kwargs`` for replay). This only shapes the parsed ``indexer``: an
    orthogonal ``DataArray`` becomes the ``ndarray`` it indexes identically to, and
    ``Select.__post_init__`` then classifies it as an ordinary positional indexer.
    """
    raw: dict[Hashable, Any] = {}
    if args and isinstance(args[0], dict):
        raw.update(args[0])
    for key, value in kwargs.items():
        if key not in _SELECT_OPTION_KWARGS:
            raw[key] = value
    return frozendict(
        {
            dim: value.values if _orthogonal_advanced(dim, value) else value
            for dim, value in raw.items()
        }
    )


def _orthogonal_advanced(dim: Hashable, value: Any) -> bool:
    """Report whether a ``DataArray``/``Variable`` indexer of ``dim`` is orthogonal.

    Parameters
    ----------
    dim : Hashable
        The dim this value indexes.
    value : Any
        The raw indexer for ``dim``.

    Returns
    -------
    bool
        ``True`` if ``value`` is an ``xr.DataArray``/``xr.Variable`` whose own dims are
        a subset of ``{dim}``: a 0-d array (scalar-like) or a 1-d array named ``dim``.

    Notes
    -----
    Such an indexer creates no new dim, so ``ds.isel(d=arr)`` is *identical* to
    ``ds.isel(d=arr.values)`` (checked against xarray 2026.7.0, for ``isel`` and
    ``sel``, with integer, boolean and label values). So :func:`_select_indexer`
    replaces it with ``.values``, and the normal taxonomy
    (:class:`~xrexpr.indexers.Positions`/``Scalar``/ ``Mask``/``Label``) classifies it.
    It then sizes, composes and reorders like any positional select.

    A *vectorized* indexer (its dims aren't a subset of ``{dim}``) is left alone. It
    drops ``dim`` and adds the array's dims, which no positional indexer can express, so
    it stays :class:`~xrexpr.indexers.Advanced` and its select blocks
    (``_select_has_advanced``).
    """
    return isinstance(value, xr.DataArray | xr.Variable) and set(value.dims) <= {dim}


def _chunk_spec(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> frozendict[Hashable, Any]:
    """Extract the ``{dim: chunksize}`` mapping of a ``chunk()`` call.

    Parameters
    ----------
    args : tuple
        The rechunk's positional arguments. A leading dict is the mapping form.
    kwargs : Mapping
        The rechunk's keyword arguments. The options in ``_CHUNK_OPTION_KWARGS``
        (``token``, ``lock``, ...) are skipped.

    Returns
    -------
    frozendict
        The named dims and their chunk specs. Empty for the uniform forms.

    Notes
    -----
    Only the per-dim form contributes. A uniform spec (``chunk(100)``,
    ``chunk("auto")``) names no dim, so this returns an empty mapping, and
    ``Rechunk.uniform`` reads the spec from ``args[0]`` instead.

    **A uniform spec silences the dim kwargs**, so they're dropped rather than recorded.
    That's what xarray does: ``Dataset.chunk`` takes the ``dict.fromkeys(self.dims,
    chunks)`` path for any non-mapping ``chunks`` and never looks at the kwargs. So
    ``ds.chunk("auto", lat=5)`` chunks every dim ``"auto"`` and ignores ``lat``, whereas
    the mapping form of the same clash, ``ds.chunk({"lat": 4}, lon=5)``, raises.
    Recording the ignored kwarg would let a rewrite rebuild the call from it and replay
    something the user didn't write.

    Values stay **raw** here. ``Rechunk.__post_init__`` sorts them into
    :data:`~xrexpr.chunks.ChunkSpec` variants, so a hand-built node gets normalised too,
    not just a recorded one. :func:`_select_indexer` and ``Select.__post_init__`` split
    the work the same way.
    """
    if args and not isinstance(args[0], Mapping):
        return frozendict()
    chunks: dict[Hashable, Any] = {}
    if args and isinstance(args[0], dict):
        chunks.update(args[0])
    for key, value in kwargs.items():
        if key not in _CHUNK_OPTION_KWARGS:
            chunks[key] = value
    return frozendict(chunks)
