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
from typing import Any, Literal

import xarray as xr
from frozendict import frozendict
from typing_extensions import assert_never

from xrexpr.indexers import Indexer
from xrexpr.ir import (
    ALL_DIMS,
    AllDims,
    ContextOpen,
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
from xrexpr.operations import (
    ContextSpec,
    ProjectSpec,
    RechunkSpec,
    ReduceSpec,
    ScanSpec,
    SelectSpec,
)
from xrexpr.operations import spec as op_spec

__all__ = ["SchemaState", "apply_schema", "resolve_dims", "to_opnode"]


@dataclass(frozen=True)
class SchemaState:
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

    variables: frozendict[Hashable, tuple[Hashable, ...]] = field(
        default_factory=frozendict
    )
    coord_names: frozenset[Hashable] = frozenset()
    sizes: frozendict[Hashable, int | None] = field(default_factory=frozendict)

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
        return cls(
            variables=frozendict(variables),
            coord_names=frozenset(ds.coords),
            sizes=frozendict(ds.sizes),
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

        Restricted to :attr:`data_vars` deliberately, unchanged by the variables store:
        the callers are the projection rules, and a projection names data variables. A
        coordinate answers ``None`` here even though its dims are now known, because
        naming one in a projection is not something those rules model.
        """
        dims: set[Hashable] = set()
        data_vars = self.data_vars
        for name in names:
            if name not in data_vars:
                return None
            dims.update(data_vars[name])
        return frozenset(dims)

    @property
    def dim_names(self) -> frozenset[Hashable]:
        """The dims that exist: every dim some variable spans. Derived, never stored.

        Returns
        -------
        frozenset of Hashable
            The union of every variable's dims, which is what a symbolic
            :data:`~xrexpr.ir.ALL_DIMS` resolves to.
        """
        return frozenset(d for var_dims in self.variables.values() for d in var_dims)

    @property
    def data_vars(self) -> frozendict[Hashable, tuple[Hashable, ...]]:
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
    def coords(self) -> frozenset[Hashable]:
        """The coordinate names. Alias for :attr:`coord_names`, kept for readers.

        Returns
        -------
        frozenset of Hashable
            :attr:`coord_names`, unchanged.
        """
        return self.coord_names


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
    Each variant affects the schema differently, so this dispatches with ``match``.

    Every arm says what happened to the **variables**; the dims follow, because
    :attr:`SchemaState.dim_names` is derived. That direction is the whole point — nothing
    here removes a dim as a primary action, so no arm can leave the dims and the variables
    disagreeing, and the reconciliation tails an independently-stored ``dims`` needed are
    gone with it.

    Two variable rules do the work, and which one an op takes is *the* distinction between
    the op families (both verified against xarray 2026.7.0 — see each helper's docstring):

    - :func:`_aggregated` — data variables lose the dims, coordinates spanning them are
      **dropped**. Taken by :class:`~xrexpr.ir.Reduce`,
      :class:`~xrexpr.ir.GroupedReduce`, :class:`~xrexpr.ir.WeightedReduce`, and by a
      ``Select`` carrying ``drop=True``.
    - :func:`_indexed` — every variable loses the dims and **nothing is dropped**, so a
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
      reduce, then mints any weight dim the dataset lacked and marks every weight dim's
      extent unknown — the weights broadcast in the dims the dataset lacks and align (so
      possibly *shrink*) the ones it shares;
    - :class:`~xrexpr.ir.Scan`/:class:`~xrexpr.ir.Rechunk`/:class:`~xrexpr.ir.Opaque`
      change nothing (a rechunk changes only chunk topology).

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
            # The group labels become a coordinate of their own -- grouped-specific, and the
            # reason this arm mints a *variable* rather than only a dim. A broadcast weight
            # dim (below) need not have a coord at all.
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
                    # and alignment -- see ``WeightedReduce``. A weight dim the dataset
                    # lacks is *minted*; one it shares may be *resized* (misaligned weights
                    # inner-join and shrink it). Either way the name is known and the extent
                    # is not, which is the answer the ``int | None`` sizes exist for.
                    from_weights = frozenset(weighted.weight_dims) - named
                    present = {d for var_dims in variables.values() for d in var_dims}
                    variables = _minted(variables, coord_names, from_weights - present)
                    for dim in from_weights:
                        sizes[dim] = None
                case _:
                    assert_never(weighted.consumes)
        case WindowedReduce() as windowed:
            for dim, window in windowed.window.items():
                if dim in sizes:
                    sizes[dim] = _windowed_size(windowed, sizes[dim], window)
        case Scan() | Rechunk() | Opaque():
            pass
        case _:
            assert_never(node)

    return SchemaState(
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
    average of a coordinate's labels. Compare :func:`_indexed`, where nothing is dropped.

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
    - **scan** / untabulated ops: no dims resolved (name/args/kwargs only).

    ``args``/``kwargs`` are kept verbatim for faithful replay; ``consumes``/``indexer``
    are the derived metadata the optimiser reasons about.
    """
    kw = frozendict(kwargs)

    match op_spec(name):
        case ReduceSpec(dim_arg=position):
            if kwargs.get("keepdims", False):
                # ``keepdims=True`` *keeps* every reduced dim at size 1, so no ``Reduce``
                # this function could build would be honest: neither a named ``consumes``
                # nor ``ALL_DIMS`` describes it, and an empty one is worse than either (it
                # empties ``blocks``, so a select hops in front and the replay raises).
                # Refusing is the house posture for a call we don't model;
                # ``07-small-wins.md`` §9 (#117) specs the modelling, whose shape is
                # ``WindowedReduce``'s.
                return Opaque(name=name, args=args, kwargs=kw)
            return Reduce(
                name=name,
                args=args,
                kwargs=kw,
                consumes=_reduce_dims(args, kwargs, position),
            )
        case SelectSpec(name=select):
            return Select(
                name=select,
                args=args,
                kwargs=kw,
                indexer=_select_indexer(args, kwargs),
            )
        case ScanSpec(name=scan):
            return Scan(name=scan, args=args, kwargs=kw)
        case RechunkSpec(name=rechunk):
            return Rechunk(
                name=rechunk,
                args=args,
                kwargs=kw,
                chunks=_chunk_spec(args, kwargs),
            )
        case ContextSpec(name=opener):
            # Decidable per-call even though what it *means* is not: a builder-returning
            # method opens a context whatever follows it. Pairing is lowering's job.
            return ContextOpen(name=opener, args=args, kwargs=kw)
        case ProjectSpec(name=getitem) if (
            variables := _projected_names(args)
        ) is not None:
            return Project(name=getitem, args=args, kwargs=kw, variables=variables)
        case ProjectSpec() | None:
            # A ``__getitem__`` whose key names no variable (a mask, a dict), or a method
            # with no row at all: both are replayed verbatim and never reordered.
            return Opaque(name=name, args=args, kwargs=kw)
        case unreachable:
            assert_never(unreachable)


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


def _reduce_dims(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], position: int
) -> DimSet:
    """Parse the dims a reduction removes out of its call.

    Parameters
    ----------
    args : tuple
        The reduction's positional arguments.
    kwargs : Mapping
        The reduction's keyword arguments, where a ``dim=`` spec takes precedence.
    position : int
        Where the dim spec sits among ``args`` — the reduction's
        :attr:`~xrexpr.operations.ReduceSpec.dim_arg`, which is 0 for every reduction but
        ``.reduce``, whose first positional is a *function*.

    Returns
    -------
    DimSet
        The named dims as a frozenset, or :data:`~xrexpr.ir.ALL_DIMS` when the call named
        none.

    Notes
    -----
    An unspecified ``dim`` is left symbolic rather than expanded here: which dims exist
    depends on where in the plan the reduce ends up running, and this function is not
    the place that knows. That covers ``.reduce(func)`` as much as ``mean()``: a bare
    reduce over every dim is a bare reduce whatever spells it.

    ``keepdims=True`` is **not** handled here, because there is no answer this function
    could give: the dims survive at size 1, so neither a named ``consumes`` nor
    ``ALL_DIMS`` describes the call. :func:`to_opnode` refuses it outright instead.
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
