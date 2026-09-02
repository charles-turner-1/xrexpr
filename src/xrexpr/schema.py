"""Logical schema tracking (no data materialisation).

The ``.plan`` proxy starts from the *real* base dataset, so a cheap **logical** schema
— every variable and the dims it spans, which of those are coordinates, and the dims'
sizes — can be folded forward through a plan **without ever touching array data**. That
is what lets a rule know the shape each node sees, and what a dim set like ``ds.mean()``'s
:data:`~xrexpr.ir.ALL_DIMS` is finally resolved against.

The variables are the store and the dims are **derived** from them, mirroring
``xr.Dataset``'s own ``_variables``/``_coord_names`` split and its derived ``.dims``. See
:class:`SchemaState` for why that direction is load-bearing rather than cosmetic.

The fold is owned by the *optimiser* (``optimize._schemas``), not by recording:
:func:`to_opnode` is a pure function of one call, so nothing is resolved against a
schema that is only a guess about where the op will run.

:class:`SchemaState` is an immutable snapshot; :func:`apply_schema` returns the
next snapshot after an :data:`~xrexpr.ir.Op` node is applied; :func:`to_opnode`
normalises a raw recorded call into that ``Op`` variant.
"""

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import xarray as xr
from frozendict import frozendict
from packaging.version import Version
from typing_extensions import assert_never

from xrexpr.indexers import Advanced, Indexer, classify
from xrexpr.intern import InternedVal, Interner
from xrexpr.ir import (
    ALL_DIMS,
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

#: ``cumsum``/``cumprod`` dropped every coordinate spanning the scanned dim before xarray
#: 2026.4.0 (the coordinate-retention fix, pull 10987); ``diff`` never did. The ``Scan``
#: arm in :func:`apply_schema` reproduces that bug-for-bug so the tracked schema stays
#: *exact* on every supported version rather than modelling only the fixed behaviour.
SCAN_DROPS_SCANNED_COORDS = Version(xr.__version__) < Version("2026.4.0")

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class SchemaState(Generic[T]):
    """An immutable snapshot of a dataset's logical shape at one point in a plan.

    Attributes
    ----------
    variables : frozendict
        **Every** variable, coordinate or not, mapped to the dims it spans. This is the
        store.
    coord_names : frozenset of Hashable
        Which of those names are coordinates. A *role*, not a type: ``xr.Dataset`` keeps
        the same distinction as ``_variables`` plus ``_coord_names``, and
        ``reset_coords`` moves a name between the two without touching the variable.
    sizes : frozendict
        Extents only, ``{dim: size}``, for the dims :attr:`dim_names` derives. Named for
        what it holds: it answers *how big*, never *which*. There is deliberately no
        ``dims`` attribute, because that name meant both before the variables store
        existed and is the ambiguity this class is now free of. A size of ``None`` means
        **don't know** — see the notes.

    Notes
    -----
    All three fields are coerced to immutable containers on construction, so a snapshot
    is hashable and safe to thread through the plan.

    The shape above holds only metadata — never array data — and mirrors how
    ``xr.Dataset`` itself is built, because that is what makes the model's invariants hold
    by construction rather than by maintenance.

    **Dim existence is derived, not stored** (:attr:`dim_names`). A dim exists exactly
    when some variable spans it — true of the domain, and ``Dataset.dims`` is a derived
    property upstream for the same reason. Storing it separately would mean storing one
    fact twice, and a schema could then say ``dims={"time": 4}`` while a variable spanned
    ``("time", "lat")``: a phantom ``lat``, reported by ``var_dims`` to any rule that
    asked. Deriving makes that state unconstructible rather than merely unlikely — the
    same trade as the ``assert_never`` discipline elsewhere.

    :attr:`data_vars` is likewise derived (:attr:`variables` minus :attr:`coord_names`).
    It is what makes *variable*-level reasoning possible: whether a projection may hop
    left past an op depends on whether the projected subset still carries the dims that
    op names.

    Modelling coordinates as variables-with-dims — rather than as bare names — is what
    lets :func:`apply_schema` state their lifetimes at all. An aggregating op drops a
    coordinate over the dim it aggregates while an indexing op keeps it (demoted to 0-d),
    and that distinction is inexpressible if a coordinate has no dims to lose.

    A size of ``None`` means **don't know** — the dim exists, but its extent is not
    statically evident. The same contract ``var_dims`` states, and the same warning
    applies with one substitution: callers must treat it as "no rewrite", never as
    *size zero*. Under-reporting a size is the unsafe direction, because it is the one
    a rewrite could act on. No optimiser rule reads a size at all — every rule reasons
    about dim *names*, and ``test_rewrites_survive_unknown_dim_sizes`` pins that by
    blanking every size and demanding the same output.
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
        # Pruned to the derived dims, which is the one place the two fields have to meet:
        # an extent for a dim nothing spans is unreachable through ``dim_names``, but it
        # would still make two semantically equal snapshots compare unequal, and equality
        # is load-bearing (``test_rewrites_survive_unknown_dim_sizes`` compares two folded
        # schemas directly). Canonicalising here is what keeps ``SchemaState`` a *value*.
        spanned = self.dim_names
        object.__setattr__(
            self,
            "sizes",
            frozendict({d: s for d, s in self.sizes.items() if d in spanned}),
        )

    @classmethod
    def from_dataset(cls, ds: xr.Dataset | xr.DataArray) -> "SchemaState[Hashable]":
        """Snapshot a dataset's logical schema, materialising nothing.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            The object to read. Only ``.sizes``, ``.coords`` and each variable's dims are
            touched — all metadata, so a dask-backed dataset computes nothing.

        Returns
        -------
        SchemaState
            The snapshot.

        Notes
        -----
        Coordinates and data variables are read into one mapping, exactly as
        ``Dataset._variables`` holds them. A ``DataArray`` contributes only its coords:
        it has no ``data_vars``, so there is nothing to project over — see
        ``07-small-wins.md`` §5.
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
        """Return the dims carried by ``names`` collectively.

        Parameters
        ----------
        names : iterable of T
            Data variable names, e.g. the ones a :class:`~xrexpr.ir.Project` requests.

        Returns
        -------
        frozenset of T or None
            The union of those variables' dims, or ``None`` if any name is not a tracked
            data variable.

        Notes
        -----
        ``None`` is the *don't know* answer — a name that isn't a tracked data
        variable (a coordinate, or something an unmodelled op introduced) — and
        callers must treat it as "no rewrite", never as "no dims".

        Restricted to :attr:`data_vars` deliberately, unchanged by the variables store:
        the callers are the projection rules, and a projection names data variables. A
        coordinate answers ``None`` here even though its dims are now known, because
        naming one in a projection is not something those rules model.
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
        """The dims that exist — every dim some variable spans. Derived, never stored.

        Returns
        -------
        frozenset of T
            The union of every variable's dims, which is what a symbolic
            :data:`~xrexpr.ir.ALL_DIMS` resolves to.
        """
        return frozenset(d for var_dims in self.variables.values() for d in var_dims)

    @property
    def data_vars(self) -> frozendict[T, tuple[T, ...]]:
        """The non-coordinate variables and their dims. Derived from the store.

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
        """The coordinate names. Alias for :attr:`coord_names`, kept for readers.

        Returns
        -------
        frozenset of T
            :attr:`coord_names`, unchanged.
        """
        return self.coord_names

    def to_interned(self, interner: Interner[T]) -> "SchemaState[InternedVal]":
        """Return a schema with every *name* relabeled to an :class:`InternedVal` handle.

        Names — variable names, their dim tuples, coord names, and the dim keys of
        ``sizes`` — become handles; ``sizes`` *values* stay bare ``int`` (they are extents,
        not names), the same name/value split :mod:`xrexpr.intern.converters` makes for the ops.
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
        """Take an interned schema and return a schema with every name de-interned."""
        deint_variables = frozendict(
            {
                interner[v.handle]: tuple(interner[d.handle] for d in dims)
                for v, dims in interned_schema.variables.items()
            }
        )
        deint_coord_names = frozenset(
            interner[c.handle] for c in interned_schema.coord_names
        )
        deint_sizes = frozendict(
            {interner[d.handle]: s for d, s in interned_schema.sizes.items()}
        )
        return SchemaState[Hashable](
            variables=deint_variables, coord_names=deint_coord_names, sizes=deint_sizes
        )


def resolve_dims(
    consumes: DimSet, dim_names: frozenset[Hashable]
) -> frozenset[Hashable]:
    """Expand a dim spec into the concrete set of dim names it stands for.

    A :data:`~xrexpr.ir.DimSet` is one of two things: an explicit ``frozenset`` of dim
    names, or the :data:`~xrexpr.ir.ALL_DIMS` sentinel standing for *every dim present at
    this point in the plan*. This turns the second into the first, and passes the first
    through unchanged.

    Parameters
    ----------
    consumes : DimSet
        The dim spec to expand, as recorded on a node — a ``frozenset`` of dim names, or
        :data:`~xrexpr.ir.ALL_DIMS`.
    dim_names : frozenset of Hashable
        The dims that exist where ``consumes`` applies: :attr:`SchemaState.dim_names` of
        the schema *entering* the node carrying it.

    Returns
    -------
    frozenset of Hashable
        ``dim_names`` when ``consumes`` is :data:`~xrexpr.ir.ALL_DIMS`, otherwise
        ``consumes`` unchanged.

    See Also
    --------
    xrexpr.ir.AllDims : The sentinel, and why it stays unexpanded until something reads it.

    Notes
    -----
    ``ds.mean()`` names no dim, and what it means depends on where it runs: every dim the
    dataset has *at that point*, which the recorder cannot know. So ``to_opnode`` records
    the sentinel rather than a guess, and it survives in the plan until a reader with an
    exact schema expands it. This is that reader.

    Callers must be somewhere the schema is exact. ``optimize``'s rules confine themselves
    to the trusted prefix (``optimize._trusted_prefix``) for this reason: past the first
    :class:`~xrexpr.ir.Opaque` the folded schema is a guess, and expanding against a guess
    would silently widen or narrow what a bare reduce claims to consume.

    Takes dim *names* rather than a :class:`SchemaState` because that is the whole of what
    expanding needs, and it keeps the ``DimSet`` ``assert_never`` in one place: a third
    dim-set shape fails type-check here, once, rather than at each site that spells the
    match out for itself.

    Examples
    --------
    An explicit set is returned as it stands:

    >>> resolve_dims(frozenset({"lat"}), frozenset({"time", "lat"}))
    frozenset({'lat'})

    The sentinel becomes whatever dims are present:

    >>> resolve_dims(ALL_DIMS, frozenset({"time", "lat"})) == frozenset({"time", "lat"})
    True

    So the *same* recorded node resolves differently earlier and later in a plan, which is
    the whole point of deferring it:

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
    """Return the schema resulting from applying ``node`` to ``schema``.

    Parameters
    ----------
    schema : SchemaState[Hashable]
        The schema *entering* the node.
    node : LoweredOp
        The node to apply.

    Returns
    -------
    SchemaState[Hashable]
        The next snapshot: the schema the following node sees.

    Notes
    -----
    Each variant affects the schema differently, so this dispatches with ``match``.

    Every arm says what happened to the **variables**; the dims follow, because
    :attr:`SchemaState.dim_names` is derived. That direction is the whole point — nothing
    here removes a dim as a primary action, so no arm can leave the dims and the variables
    disagreeing, and the reconciliation tails an independently-stored ``dims`` needed are
    gone with it.

    Two variable rules do the work, and which one an op takes is *the* distinction between
    the op families (both verified against xarray 2026.7.0 — see each helper's docstring):

    - ``_aggregated`` — data variables lose the dims, coordinates spanning them are
      **dropped**. Taken by :class:`~xrexpr.ir.Reduce`,
      :class:`~xrexpr.ir.GroupedReduce`, :class:`~xrexpr.ir.WeightedReduce`, and by a
      ``Select`` carrying ``drop=True``.
    - ``_indexed`` — every variable loses the dims and **nothing is dropped**, so a
      coordinate left dimensionless is a 0-d coordinate. Taken by an ordinary
      :class:`~xrexpr.ir.Select`.

    On top of that, per variant:

    - :class:`~xrexpr.ir.Reduce` aggregates over its ``consumes`` — *every* dim when that
      is :data:`~xrexpr.ir.ALL_DIMS`, which is where the deferred bare-reduce expansion
      is finally cashed in against a schema that is exact;
    - :class:`~xrexpr.ir.Select` also *resizes* the dims it keeps (slice/sequence indices);
    - :class:`~xrexpr.ir.Project` restricts the variables and prunes coordinates to those
      the survivors still span; dims it orphans simply stop being derived. An unknown name
      declines the whole projection — see the guard's comment;
    - :class:`~xrexpr.ir.GroupedReduce` aggregates over ``group_dim`` plus any extra
      ``consumes``, mints ``new_dim`` onto every data variable, and adds ``new_dim`` as a
      coordinate of its own — of *unknown* size, since the group count is a fact about
      coordinate values;
    - :class:`~xrexpr.ir.WindowedReduce` keeps every dim and *resizes* the windowed ones
      (``coarsen`` only);
    - :class:`~xrexpr.ir.WeightedReduce` aggregates over its ``consumes`` like a plain
      reduce, then mints every surviving weight dim onto each **variable** that lacks it
      and marks every weight dim's extent unknown — the weights broadcast in the dims a
      variable lacks and align (so possibly *shrink*) the ones it shares;
    - :class:`~xrexpr.ir.Scan` mostly preserves the schema, with two exceptions: ``diff``
      *shrinks* its dim, and on xarray before 2026.4.0 (:data:`SCAN_DROPS_SCANNED_COORDS`)
      ``cumsum``/``cumprod`` **drop** every coordinate over the scanned dim while keeping
      the dim itself — the version-gated bug this arm reproduces so the schema stays exact;
    - :class:`~xrexpr.ir.Elementwise` changes nothing — it is per-element, so every dim,
      size and variable survives; this arm is *exact* for it, unlike the assumption below
      for :class:`~xrexpr.ir.Opaque`;
    - :class:`~xrexpr.ir.Rechunk`/:class:`~xrexpr.ir.Opaque` change nothing (a rechunk
      changes only chunk topology).

    An :class:`~xrexpr.ir.Opaque` is assumed variable-preserving, which is *not* true
    in general (``rename``/``drop_vars``/``assign``). That makes the schema exact only
    up to the first opaque op — a trust boundary the optimiser enforces rather than
    this function (see ``optimize._trusted_prefix``). Within that boundary it really is
    exact, coordinates included: ``test_tracked_schema_agrees_with_evaluation`` compares
    the tracked variables and coordinates to a real evaluation's for equality, not
    containment.

    ``assert_never`` on the final arm makes the union exhaustive: a new variant fails
    type-check here until handled.
    """
    variables = dict(schema.variables)
    coord_names = set(schema.coord_names)
    sizes = dict(schema.sizes)

    match node:
        case Reduce(consumes=consumes) as reduce if reduce.keepdims:
            # ``keepdims=True`` keeps every named dim at size 1 rather than removing it:
            # data variables keep their dims, only coordinates spanning a resized dim are
            # dropped (a mean over ``time`` has no meaningful ``time`` label -- verified
            # against xarray 2026.7.0). Contrast the plain reduce below, which drops the
            # dim from the variables outright.
            over = resolve_dims(consumes, schema.dim_names)
            variables, coord_names = _keepdims_reduced(variables, coord_names, over)
            for dim in over:
                sizes[dim] = 1
        case Reduce(consumes=consumes):
            variables, coord_names = _aggregated(
                variables, coord_names, resolve_dims(consumes, schema.dim_names)
            )
        case Select(indexer=indexer) as select:
            # ``drop=True`` is the difference between the two coordinate rules, and it is
            # the only place this arm needs to know which it is: without it a scalar select
            # *demotes* the coords over its dim to 0-d and keeps them
            # (``isel(lat=0)`` -> ``lat``, ``region`` both present, scalar); with it they go
            # (verified against xarray 2026.7.0). Modelled rather than approximated because
            # the conservative reading is now the *unsafe* one -- coords are asserted to be
            # a subset of the result's, so keeping one xarray dropped is an over-report.
            if select.kwargs.get("drop", False):
                variables, coord_names = _aggregated(
                    variables, coord_names, select.consumes
                )
            else:
                variables = _indexed(variables, select.consumes)
            # A non-scalar select keeps its dim but changes its size (scalar selects
            # are already gone via ``consumes``).
            for dim, index in indexer.items():
                if dim not in select.consumes and dim in sizes:
                    sizes[dim] = _selected_size(select.name, index, sizes[dim])
        case Project(variables=names):
            # Guarded on every projected name being a *tracked* data variable: a name this
            # layer doesn't know (a coord, or something an unmodelled op introduced) would
            # make the surviving dims an under-report, and under-reporting dims is the
            # unsafe direction. Unknown name -> leave the store alone entirely.
            #
            # That "entirely" is the one behavioural difference from the pre-derivation
            # shape, which restricted ``data_vars`` while leaving ``dims`` untouched -- i.e.
            # it produced exactly the phantom-dim state deriving exists to forbid. Declining
            # the whole projection over-reports in the same direction it used to, and stays
            # constructible.
            #
            # A projection naming a *coordinate* falls into that "decline" leg even though
            # the coord is known, so the tracked schema over-reports it (claims a data var
            # survives that evaluation drops). Safe -- no rule reads the post-projection
            # schema of a coord projection -- but not exact; modelling it is #162.
            if all(n in schema.data_vars for n in names):
                kept = {n: variables[n] for n in names}
                # xarray prunes coordinates to those the *selected* variables still span,
                # so ``ds[["elevation"]]`` (``lat``, ``lon``) drops ``time`` and the ``time``
                # coordinate together. A scalar coord (no dims) is spanned by anything and
                # survives. Verified against xarray 2026.7.0.
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
            # The group labels usually become a coordinate of their own -- grouped-specific,
            # and the reason this arm mints a *variable* rather than only a dim (a broadcast
            # weight dim, below, need not have a coord at all). But xarray mints that
            # coordinate only when the grouper *had* one. A coordinate or component grouper
            # always does -- ``groupby("region")`` and ``groupby("time.month")`` both give a
            # ``new_dim`` distinct from ``group_dim`` -- so those always mint. Grouping a
            # bare *dimension* (``new_dim == group_dim`` and no coordinate over it) mints no
            # coordinate: xarray leaves the dim without one (verified on 2025.6.1 and
            # 2026.7.0), which is exactly what a pre-2026.4.0 scan leaves behind when it
            # drops the scanned dim's coordinate (:data:`SCAN_DROPS_SCANNED_COORDS`).
            if (
                grouped.new_dim != grouped.group_dim
                or grouped.group_dim in schema.coords
            ):
                variables[grouped.new_dim] = (grouped.new_dim,)
                coord_names = coord_names | {grouped.new_dim}
            # The minted dim's extent is the number of groups -- a fact about coordinate
            # *values*, which this layer does not read. ``None`` rather than a guess.
            sizes[grouped.new_dim] = None
        case WeightedReduce() as weighted:
            match weighted.consumes:
                case AllDims():
                    # A bare weighted closer, like a bare ``mean()``: the weight dims go too.
                    variables, coord_names = _aggregated(
                        variables, coord_names, schema.dim_names
                    )
                case frozenset() as named:
                    variables, coord_names = _aggregated(variables, coord_names, named)
                    # The weights have a dim effect of their own, via ``dot``'s broadcast
                    # and alignment -- see ``WeightedReduce``. A weight dim a *variable*
                    # lacks is minted onto it; one it shares may be *resized* (misaligned
                    # weights inner-join and shrink it). Either way the name is known and
                    # the extent is not, which is the answer the ``int | None`` sizes exist
                    # for.
                    #
                    # Minted per variable, not per dataset: the broadcast is ``dot``'s, so
                    # it reaches each variable on its own terms. Skipping the mint when some
                    # *other* variable already carried the dim under-reported ``elevation``
                    # after ``weighted(w(time)).mean("lat")`` on a dataset that also held
                    # ``temperature(time, lat)`` -- issue #125. ``_minted`` is per-variable
                    # and idempotent, so handing it the whole set is both correct and the
                    # simpler statement.
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
            # ``diff(dim, n=1)`` shrinks ``dim`` by ``n``; ``label`` only picks which end,
            # not the size. ``diff``'s dim is a required positional, so ``dims`` is always
            # this concrete frozenset, never ``ALL_DIMS`` -- an ``ALL_DIMS`` scan (a bare
            # ``cumsum()``) falls through to the size-preserving ``pass`` arm below.
            # ``cumsum``/``cumprod`` keep every size and go there too. Latent today
            # (nothing downstream of a scan reads sizes), but the schema should not carry a
            # known lie once the dims exist to fix it.
            n = scan.kwargs.get("n", scan.args[1] if len(scan.args) > 1 else 1)
            for dim in dims:
                size = sizes.get(dim)
                if size is not None:  # an unknown length stays unknown, never guessed
                    sizes[dim] = max(size - n, 0)
        case Scan(name="cumsum" | "cumprod", dims=dims) if SCAN_DROPS_SCANNED_COORDS:
            # Pre-2026.4.0 xarray routed ``cumsum``/``cumprod`` through ``Dataset.reduce``,
            # which keeps a coordinate only when it spans *none* of the reduced dims -- so
            # every coordinate over the scanned dim is dropped (index coord or not), while
            # the data variables keep the dim (the scan preserves shape). The dim then
            # survives as a "dimension without coordinates". PR 10987 (the flox migration)
            # fixed it to retain coords; on those versions ``SCAN_DROPS_SCANNED_COORDS`` is
            # false, this arm is skipped, and the op falls through to the coord- and
            # size-preserving ``pass`` below. ``diff`` never had the bug -- its own arm
            # above keeps coords on every version. Unlike ``_aggregated`` this drops the
            # coordinate *variables* only and leaves the data variables (and sizes) alone.
            over = resolve_dims(dims, schema.dim_names)
            dropped = {c for c in coord_names if not over.isdisjoint(variables[c])}
            coord_names -= dropped
            variables = {k: v for k, v in variables.items() if k not in dropped}
        case Rename(mapping=renames):
            # ``rename`` relabels a dim, a variable/coordinate, or -- for a dimension
            # coordinate -- both at once. ``_relabelled`` applies it as a single
            # shape-preserving relabel, so it is exact, coordinates included.
            variables, coord_names, sizes = _relabelled(
                variables, coord_names, sizes, renames
            )
        case Drop(variables=names):
            # ``drop_vars`` removes named variables/coordinates by *key* -- not by dim, so
            # this is not ``_aggregated``. A dimension left spanned by nothing afterwards
            # (its last variable dropped, no coordinate holding it) is pruned by
            # ``SchemaState.__post_init__``; a dim-coordinate's dim survives while a data
            # variable still spans it, exactly as xarray leaves a dimension without
            # coordinates. Verified against xarray 2026.7.0 for a data var, an auxiliary
            # coordinate and a dimension coordinate.
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
    """Apply an *aggregating* op: variables lose the dims, spanning coordinates are dropped.

    Parameters
    ----------
    variables : dict
        The schema's variables entering the op, name → dims.
    coord_names : set of Hashable
        Which of those names are coordinates.
    over : frozenset of Hashable
        The dims being aggregated away.

    Returns
    -------
    tuple of (dict, set)
        The surviving variables with ``over`` removed from their dims, and the coordinate
        names among them.

    Notes
    -----
    The asymmetry is xarray's, not a modelling choice (verified against xarray 2026.7.0):
    ``ds.mean("lat")`` reduces ``tas(time, lat)`` to ``tas(time)`` but *removes* a
    ``region(lat)`` coordinate rather than aggregating it, because there is no meaningful
    average of a coordinate's labels. Compare ``_indexed``, where nothing is dropped.

    A coordinate that spans none of ``over`` is untouched, which is what keeps ``time``
    alive through ``groupby("lat").mean()`` while ``region(lat)`` goes.
    """
    kept = {
        name: tuple(d for d in var_dims if d not in over)
        for name, var_dims in variables.items()
        if not (name in coord_names and not over.isdisjoint(var_dims))
    }
    return kept, coord_names & set(kept)


def _indexed(
    variables: dict[Hashable, tuple[Hashable, ...]], over: frozenset[Hashable]
) -> dict[Hashable, tuple[Hashable, ...]]:
    """Apply an *indexing* op: every variable loses the dims, and **nothing is dropped**.

    Parameters
    ----------
    variables : dict
        The schema's variables entering the op, name → dims.
    over : frozenset of Hashable
        The dims being indexed away.

    Returns
    -------
    dict
        Every variable, with ``over`` removed from its dims.

    Notes
    -----
    A coordinate left with no dims is a 0-d coordinate, which is exactly what xarray
    produces for ``isel(lat=0)`` — ``lat`` and ``region`` both survive, scalar. Uniform
    over coordinates and data variables, which is the whole reason they share a store.
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
    """Apply a ``keepdims`` reduction: dims are kept (resized to 1), spanning coords dropped.

    Parameters
    ----------
    variables : dict
        The schema's variables entering the op, name → dims.
    coord_names : set of Hashable
        Which of those names are coordinates.
    over : frozenset of Hashable
        The dims being reduced and kept at size 1.

    Returns
    -------
    tuple of (dict, set)
        Every variable with its dims **unchanged**, minus the coordinates spanning
        ``over``, and the coordinate names among the survivors.

    Notes
    -----
    ``keepdims=True`` keeps each reduced dim at size 1 rather than removing it, so unlike
    ``_aggregated`` the data variables keep their dims — the caller resizes the kept dims
    to 1. Coordinates still go the same way: one spanning a reduced dim is dropped, because
    its labels have no meaningful value after the reduction (``time`` and a non-dim
    ``ref(time)`` both go from ``mean("time", keepdims=True)`` — verified against xarray
    2026.7.0). Compare ``_aggregated``, which additionally strips the dim from variables.
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
        The schema's variables after the op's own dim removal, name → dims.
    coord_names : set of Hashable
        Which of those names are coordinates. Coordinates are *not* minted onto.
    new_dims : frozenset of Hashable
        The dims the op mints.

    Returns
    -------
    dict
        The variables, each data variable now leading with the minted dims in ``str``
        order.

    Notes
    -----
    Verified against xarray 2026.7.0, and the non-obvious part: *every* data variable comes
    back carrying a minted dim, including ones that could not have had it — ``elev(lat)``
    under ``groupby("time.month")`` is ``(month, lat)``, and under ``weighted(w)`` with
    ``w(lat, member)`` it is ``(member,)``. So this is an addition to each variable, not a
    substitution of one dim for another.

    Coordinates are **not** minted onto: the existing ones keep their own dims (``region``
    stays ``(lat,)`` through ``groupby("time.month")``), and the minted coordinate a grouped
    reduce adds is a new variable spanning only itself.

    Sorted because ``new_dims`` is a set and :class:`SchemaState` is a *value*: an
    iteration-order-dependent dim tuple would make two snapshots folded from identical
    inputs unequal between processes. ``str`` is the key for the same reason
    ``pushdown_selects`` uses it — a dim name is only ``Hashable``, so it is the one total
    order available. Dim order is not otherwise meaningful here (``var_dims`` unions into a
    set, and ``Dataset.sizes`` promises no order either).
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
    """Apply a ``rename``: relabel dims and/or variable keys, preserving every shape.

    Parameters
    ----------
    variables : dict
        The schema's variables entering the op, name → dims.
    coord_names : set of Hashable
        Which of those names are coordinates.
    sizes : dict
        The dim extents entering the op, dim → size or ``None``.
    mapping : Mapping
        The ``{old: new}`` relabelling as ``rename`` was called.

    Returns
    -------
    tuple of (dict, set, dict)
        The variables, coordinate names and sizes after the relabel.

    Notes
    -----
    Rename is the third schema primitive beside :func:`_aggregated` (drop) and
    :func:`_minted` (mint), and the first to touch the **keys** of ``variables`` /
    ``coord_names`` rather than only the dim tuples -- which is why it is one primitive and
    not those two composed (``_minted`` broadcasts a new dim onto every data variable and
    ``_aggregated`` drops the coord, so ``drop(x) + mint(y)`` is not a rename). Expressed as
    a single relabel it reads the source label's shape and reassigns it, so "shape in =
    shape out" holds by construction rather than by a test keeping two copies of the shape
    equal.

    A rename can hit three targets, and a **dimension coordinate** is all three at once
    (verified against xarray 2026.7.0):

    - a **dim** (``old`` spans some variable): rewrite ``old -> new`` in every variable's
      dims tuple and move ``sizes[old] -> sizes[new]``, carrying an unknown (``None``)
      through unchanged -- never inventing an extent, which
      ``test_rewrites_survive_unknown_dim_sizes`` pins.
    - a **variable / coordinate** (``old in variables``): rekey it, and its ``coord_names``
      membership with it.
    - a **dimension coordinate**: both, since its name is a dim *and* a variable, so
      ``rename({month: time})`` yields ``time`` as a dimension coordinate.

    The rewrite is computed in one pass through a translation ``old -> new`` rather than
    applied entry by entry, so a chained or swapping mapping (``{a: b, b: c}``) cannot see a
    half-renamed intermediate. Only mapping entries whose key is an actual dim translate
    dims; a variable-only rename (an auxiliary coordinate) leaves every dim tuple untouched.
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

    ``coarsen`` divides, and how it rounds is an *option* rather than a property of the op:
    ``boundary="trim"`` discards the short final block (floor), ``"pad"`` fills it (ceil),
    and the default ``"exact"`` requires divisibility — so where it is exact the two agree,
    and where it is not, xarray raises at replay rather than producing the size computed
    here. That reading of ``boundary`` is precomputed on the node as
    :attr:`~xrexpr.ir.WindowedReduce.rounds_up` (``True`` ceil, ``False`` floor, ``None``
    unknown), so this function reasons over the node, not the raw header. A ``boundary`` the
    node does not recognise answers ``None``: a future spelling should cost precision, not
    correctness.
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
        :class:`~xrexpr.ir.Elementwise`, :class:`~xrexpr.ir.ContextOpen`, or
        :class:`~xrexpr.ir.Opaque` for anything untabulated.

    Notes
    -----
    A pure function of the call itself: every kind below is settled by the method name
    and the shape of its arguments, so nothing here reads a schema. That is deliberate
    — the one case that would otherwise need one, a bare ``mean()``, records the symbolic
    :data:`~xrexpr.ir.ALL_DIMS` instead of an eagerly expanded dim set, leaving the
    expansion to a reader whose schema is exact.

    One ``match`` over the :data:`~xrexpr.operations.OpSpec` the name is tabulated as,
    closed with ``assert_never``: the spec *is* the kind, so a variant added to the table
    without an arm here is a type error rather than a call quietly recorded as
    :class:`~xrexpr.ir.Opaque`. Each spec carries its name already typed as the
    ``Literal`` its ``Op`` counterpart demands, so no ``cast`` stands between the lookup
    and the constructor.

    - **reduce** (``mean``/``sum``/...): the dim spec — positional (``mean("lat")``),
      keyword (``mean(dim="lat")``) or tuple (``mean(("lat", "lon"))``) — collapses to
      one ``consumes`` frozenset; a **no-dim ``mean()`` consumes** :data:`~xrexpr.ir.ALL_DIMS`,
      which is what fixes the empty-dim reorder bug (``ds.mean().isel(...)``). *Which*
      positional argument holds the spec is the spec's own
      :attr:`~xrexpr.operations.ReduceSpec.dim_arg`, because ``.reduce(func, dim)`` puts
      a **function** first; and ``keepdims=True`` records :class:`~xrexpr.ir.Opaque`,
      since the "reduced" dims survive at size 1 and no ``consumes`` would be true (see
      the guard in the code).
    - **select** (``isel``/``sel``): the indexer (a positional dict and/or kwargs,
      minus option kwargs like ``drop``) becomes ``indexer``; a dim given a *scalar*
      index is dropped and so also lands in ``consumes`` (a slice/list/array keeps it).
    - **project** (``ds["tas"]`` / ``ds[["tas", "pr"]]``): the key's variable names
      become ``variables``. This is the one kind the table only *nominates* —
      ``__getitem__`` is a projection only when its key *names variables* — so it is the
      one **guarded** arm, and a mask-style key (a boolean ``DataArray``, a dict) falls
      past the guard to ``Opaque``.
    - **rechunk** (``chunk``): the *mapping* form (a positional dict and/or dim kwargs,
      minus option kwargs like ``token``) becomes ``chunks``. The uniform forms
      (``chunk()``, ``chunk(100)``, ``chunk("auto")``) name no dim, so ``chunks`` is
      empty and the spec stays in ``args``.
    - **context opener** (``groupby``/``rolling``/``weighted``/...): a
      :class:`~xrexpr.ir.ContextOpen`, which says only *a context opens here*. The one
      kind whose meaning this function deliberately does not settle — the call is half an
      operation, and the pair is ``lower.to_lower_ir``'s to read.
    - **elementwise** (``fillna``/``astype``/``round``/...): a
      :class:`~xrexpr.ir.Elementwise`, but **only when every argument is a plain value**
      (``_elementwise_safe``). ``fillna(other_da)`` and ``fillna({"tas": 0})`` carry a
      data- or per-variable-shaped argument that does not commute with a projection, so
      they fall past the guard to ``Opaque`` — the same shape-of-the-argument nomination
      the ``project`` arm makes.
    - **scan** / untabulated ops: no dims resolved (name/args/kwargs only).

    ``args``/``kwargs`` are kept verbatim for faithful replay; ``consumes``/``indexer``
    are the derived metadata the optimiser reasons about.
    """
    kw = frozendict(kwargs)

    match op_spec(name):
        case ReduceSpec(dim_arg=position):
            # ``keepdims=True`` is modelled, not refused: the named dims are kept at size
            # 1 rather than removed, which the derived ``Reduce.keepdims`` carries and
            # ``dim_effect``/``apply_schema`` read off ``consumes`` (the same named dims).
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
            # Scans put the dim first (``cumsum(dim)``, ``diff(dim)``), so position 0 --
            # ``ScanSpec`` needs no ``dim_arg`` field the way ``ReduceSpec`` does.
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
            # ``rename`` relabels dims/variables -- modelled, not ``Opaque``, so the schema
            # fold stays exact past it. ``_rename_map`` gathers the ``{old: new}`` mapping
            # from the positional dict and/or the name kwargs.
            return Rename(
                name=rename,
                args=args,
                kwargs=kw,
                mapping=frozendict(_rename_map(args, kwargs)),
            )
        case DropSpec(name=drop):
            # ``drop_vars`` removes named variables/coords -- modelled, not ``Opaque``, so
            # the schema fold stays exact past it (``_dropped_names`` splits its argument
            # exactly as xarray does).
            return Drop(
                name=drop,
                args=args,
                kwargs=kw,
                variables=_dropped_names(args, kwargs),
            )
        case ContextSpec(name=opener):
            # Decidable per-call even though what it *means* is not: a builder-returning
            # method opens a context whatever follows it. Pairing is lowering's job.
            return ContextOpen(name=opener, args=args, kwargs=kw)
        case ElementwiseSpec(name=ew) if _elementwise_safe(args, kwargs):
            # ``fillna(0)``/``astype("float32")``/... with plain-valued arguments only:
            # per-element, so it keeps every dim, size and variable. A data- or
            # per-variable-shaped argument fails the guard and falls to ``Opaque`` below.
            return Elementwise(name=ew, args=args, kwargs=kw)
        case ProjectSpec(name=getitem) if (
            variables := _projected_names(args)
        ) is not None:
            return Project(name=getitem, args=args, kwargs=kw, variables=variables)
        case SelectSpec() | ElementwiseSpec() | ProjectSpec() | None:
            # A select carrying a *vectorized* ``DataArray``/``Variable`` indexer (an
            # orthogonal one is normalised to plain values upstream and stays a ``Select``),
            # an elementwise call with an unsafe (data-/per-variable-shaped) argument, a
            # ``__getitem__`` whose key names no variable (a mask, a dict), or a method
            # with no row at all: all are replayed verbatim and never reordered. The select
            # barriers because a vectorized indexer drops its dim and mints new ones -- see
            # ``_select_has_advanced``.
            return Opaque(name=name, args=args, kwargs=kw)
        case unreachable:
            assert_never(unreachable)


def _elementwise_safe(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """Whether every argument to an elementwise call is a plain value.

    Parameters
    ----------
    args : tuple
        The call's positional arguments.
    kwargs : Mapping
        The call's keyword arguments.

    Returns
    -------
    bool
        ``True`` when no argument is data- or variable-shaped, so the call is per-element
        and commutes with any select or projection; ``False`` when one is, so the call
        must record :class:`~xrexpr.ir.Opaque` instead.

    Notes
    -----
    A **blocklist**, not an allowlist: an argument is unsafe only if it is an
    ``xr.DataArray``/``xr.Dataset`` (``fillna(other)`` fills from another array's values),
    a ``dict`` (``fillna({"tas": 0})`` is per-variable), or a ``list``/``tuple``/``ndarray``
    (array-shaped). Everything else passes — which is what lets ``astype``'s ``np.dtype``
    instances and ``type`` objects, and scalar ``clip``/``fillna`` bounds, mint an
    :class:`~xrexpr.ir.Elementwise`. An allowlist would have to enumerate every dtype
    spelling; the blocklist names only the shapes that break commutation. The shape of an
    argument deciding the kind has precedent in ``_projected_names`` (:class:`~xrexpr.ir.Project`
    vs :class:`~xrexpr.ir.Opaque`).
    """
    unsafe = (xr.DataArray, xr.Dataset, dict, list, tuple, np.ndarray)
    return not any(isinstance(v, unsafe) for v in (*args, *kwargs.values()))


def _select_has_advanced(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """Whether any of a select's indexers is an advanced (``DataArray``) indexer.

    Parameters
    ----------
    args : tuple
        The select's positional arguments.
    kwargs : Mapping
        The select's keyword arguments.

    Returns
    -------
    bool
        ``True`` when at least one indexer classifies as
        :class:`~xrexpr.indexers.Advanced`, so the whole select must record
        :class:`~xrexpr.ir.Opaque`; ``False`` when every indexer is one the schema layer
        can evolve exactly.

    Notes
    -----
    Only a **vectorized** advanced indexer reaches here as
    :class:`~xrexpr.indexers.Advanced`: :func:`_select_indexer` has already normalised the
    orthogonal ones (a fresh array named after the dim it indexes) to plain ``.values``, so
    they classify as ordinary positional indexers and their select optimises normally (see
    :func:`_orthogonal_advanced`). A vectorized indexer drops its dim and mints the array's
    dims, which no positional indexer expresses and this layer does not model, so its select
    must barrier -- and only an :class:`~xrexpr.ir.Opaque` node can withhold trust in the
    folded schema (a ``Select`` is always inside ``optimize._trusted_prefix``). The decision
    is routed through :func:`~xrexpr.indexers.classify` -- the sole constructor of the value
    taxonomy -- so "what is an advanced indexer" has one definition; the shape of an argument
    deciding the kind has precedent in :func:`_elementwise_safe` and :func:`_projected_names`.
    """
    return any(
        isinstance(classify(v), Advanced)
        for v in _select_indexer(args, kwargs).values()
    )


def _rename_map(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[Hashable, Hashable]:
    """Gather a ``rename`` call's ``{old: new}`` mapping from its positional dict and kwargs.

    Parameters
    ----------
    args : tuple
        The ``rename`` call's positional arguments — a ``{old: new}`` dict, if given
        positionally.
    kwargs : Mapping
        The call's keyword arguments — name mappings given as ``old=new``.

    Returns
    -------
    dict
        The merged ``{old: new}`` relabelling.

    Notes
    -----
    ``Dataset.rename`` is ``rename(name_dict=None, **names)``, so a call spells the mapping
    positionally (``rename({"month": "time"})``), as kwargs (``rename(month="time")``), or
    conceivably both; every keyword is a name mapping, since ``rename`` carries no option
    kwargs. Taken verbatim -- whether each ``old`` is a dim, a variable or both is decided
    by :func:`_relabelled` against the schema, not here.
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
        The ``drop_vars`` call's positional arguments — the names, if given positionally.
    kwargs : Mapping
        The call's keyword arguments — the names may be passed as ``names=`` instead.

    Returns
    -------
    tuple of Hashable
        The names being dropped.

    Notes
    -----
    Split exactly where ``Dataset.drop_vars`` splits its own argument: a **str** or any
    other **non-iterable** is one name (``drop_vars("region")``), while any other iterable
    is a sequence of them (``drop_vars(["a", "b"])``). This differs from
    :func:`_projected_names`, which follows ``__getitem__``'s rule instead — a bare tuple
    is one name there but several here — so the two are deliberately not shared.
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


def _dim_spec(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], position: int
) -> DimSet:
    """Parse the dim spec out of a reduce or scan call.

    Serves both kinds: a reduce (which *removes* the dims) and a scan (which *keeps*
    them). The parsing convention is identical — a ``dim=`` kwarg, else the positional at
    ``position``, else every current dim — because ``mean``, ``cumsum``, ``cumprod`` and
    ``diff`` all take the dim first.

    Parameters
    ----------
    args : tuple
        The call's positional arguments.
    kwargs : Mapping
        The call's keyword arguments, where a ``dim=`` spec takes precedence.
    position : int
        Where the dim spec sits among ``args`` — the reduction's
        :attr:`~xrexpr.operations.ReduceSpec.dim_arg` (0 for every reduction but
        ``.reduce``, whose first positional is a *function*), and 0 for every scan.

    Returns
    -------
    DimSet
        The named dims as a frozenset, or :data:`~xrexpr.ir.ALL_DIMS` when the call named
        none.

    Notes
    -----
    An unspecified ``dim`` is left symbolic rather than expanded here: which dims exist
    depends on where in the plan the op ends up running, and this function is not the
    place that knows. That covers ``.reduce(func)`` and a bare ``cumsum()`` as much as
    ``mean()``: a bare pass over every dim is a bare pass whatever spells it. (``diff``
    makes its dim a required positional, so its bare case can't arise.)

    ``keepdims=True`` needs no special case here: the named dims are exactly the same
    set whether they are removed or kept at size 1, so this returns them unchanged and the
    derived :attr:`~xrexpr.ir.Reduce.keepdims` tells ``dim_effect``/``apply_schema`` which
    of the two the reduce does.
    """
    if "dim" in kwargs:
        dim = kwargs["dim"]
    elif len(args) > position:
        dim = args[position]
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
        ``_SELECT_OPTION_KWARGS`` (``drop``, ``method``, ...).

    Returns
    -------
    frozendict
        The indexed dims and their indexer values, each raw except that an *orthogonal*
        advanced indexer is normalised to its ``.values`` (see :func:`_orthogonal_advanced`).

    Notes
    -----
    The verbatim call is untouched — ``to_opnode`` keeps the original ``args``/``kwargs`` for
    replay — so this normalisation only shapes the derived ``indexer`` metadata: an
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
    """Whether a ``DataArray``/``Variable`` indexer of ``dim`` is orthogonal, not vectorized.

    Parameters
    ----------
    dim : Hashable
        The dim this value indexes.
    value : Any
        The raw indexer for ``dim``.

    Returns
    -------
    bool
        ``True`` when ``value`` is an ``xr.DataArray``/``xr.Variable`` whose own dims are a
        subset of ``{dim}`` — a 0-d array (scalar-like) or a 1-d array named ``dim``.

    Notes
    -----
    Such an indexer introduces no new dim, so ``ds.isel(d=arr)`` is *identical* to
    ``ds.isel(d=arr.values)`` (verified against xarray 2026.7.0, ``isel`` and ``sel``,
    integer/boolean/label): the array's values index axis ``d`` exactly as a plain
    ``ndarray`` would. :func:`_select_indexer` therefore replaces it with ``.values`` so the
    ordinary taxonomy (:class:`~xrexpr.indexers.Positions`/``Scalar``/``Mask``/``Label``)
    classifies it — it sizes, composes and reorders like any positional select. A
    *vectorized* indexer (a fresh dim name, so ``value.dims`` is not a subset of ``{dim}``)
    is **not** normalised: it drops ``dim`` and mints the array's dims, which no positional
    indexer expresses, so it stays :class:`~xrexpr.indexers.Advanced` and its select
    barriers (``_select_has_advanced``).
    """
    return isinstance(value, xr.DataArray | xr.Variable) and set(value.dims) <= {dim}


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
        ``_CHUNK_OPTION_KWARGS`` (``token``, ``lock``, ...).

    Returns
    -------
    frozendict
        The named dims and their chunk specs; empty for the uniform forms.

    Notes
    -----
    Only the mapping form contributes. A uniform positional spec (``chunk(100)``,
    ``chunk("auto")``) names no dim, so it yields an empty mapping and is read instead by
    ``Rechunk.uniform``, which classifies ``args[0]`` through the same taxonomy.

    A uniform spec **silences the dim kwargs**, so they are dropped rather than recorded.
    That is xarray's behaviour, not a simplification: ``Dataset.chunk`` takes the
    ``dict.fromkeys(self.dims, chunks)`` branch for any non-Mapping ``chunks`` and never
    reaches ``either_dict_or_kwargs``, so ``ds.chunk("auto", lat=5)`` chunks every dim
    ``"auto"`` and applies nothing to ``lat`` — where the mapping spelling of the same
    clash, ``ds.chunk({"lat": 4}, lon=5)``, raises instead. Recording the silenced kwarg
    would let a rewrite rebuild ``args`` from it and replay a call the user did not write.

    The values stay **raw** here. Sorting them into
    :data:`~xrexpr.chunks.ChunkSpec` variants is ``Rechunk.__post_init__``'s job, so a
    hand-built node is normalised too and not only a recorded one — the same division
    ``_indexer_spec`` and ``Select.__post_init__`` keep.
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
