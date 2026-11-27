"""The expression IR: one node type per kind of operation the optimiser distinguishes.

A method chain is linear, so a plan is a **list** of :data:`Op` nodes, not a tree. Each
node carries two things:

- the verbatim call (``args``/``kwargs``), which replay uses;
- the parsed metadata the optimiser reasons about, which differs per kind. That's why
  each kind is its own dataclass rather than one flat record.

:data:`FluentOp` (what recording produces) and :data:`LoweredOp` (what the optimiser
rewrites) are two unions over the same dataclasses, so a function's signature shows
which level it works at.

Three rules to keep when editing here:

- Every ``match`` over one of these unions ends with ``assert_never``, so a new node
  kind is a type error at every site that doesn't handle it, not a silent fallthrough.
- A bare ``ds.mean()`` records :data:`ALL_DIMS`, a sentinel meaning "every dim at this
  point". It's distinct from ``None``, which means *unknown*.
- Nodes are always frozen, but only hashable if their contents are. A node holding an
  array raises on ``hash``, on purpose: hashing it would mean computing it.

Only **unary** ops are modelled. Keep that assumption stated here rather than spread
through the rules.

See ``docs/internals/ir.md``; ``test_ir.py`` checks the hashability behaviour.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from frozendict import frozendict

from xrexpr._xrexprs.ir import ALL_DIMS, AllDims
from xrexpr.chunks import ChunkSpec, classify_chunk
from xrexpr.indexers import Indexer, classify

__all__ = [
    "ALL_DIMS",
    "AllDims",
    "ContextOpen",
    "ContextOpenName",
    "DimSet",
    "Drop",
    "Elementwise",
    "FluentOp",
    "GroupedReduce",
    "LoweredOp",
    "Op",
    "Opaque",
    "Project",
    "Rechunk",
    "Reduce",
    "Rename",
    "Scan",
    "Select",
    "WeightedReduce",
    "WindowedReduce",
]


#: The dims an op removes: a concrete set, or :data:`ALL_DIMS` for a call that named
#: none. Code reading one must handle both.
DimSet = frozenset[Hashable] | AllDims


@dataclass(frozen=True)
class Reduce:
    """A reduction that removes dims (``mean``/``sum``/``std``/...).

    Attributes
    ----------
    name : str
        The method name. Reductions are an open set, so this is ``str``; ``OP_TABLE`` is
        what says which names are reductions.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    consumes : DimSet
        The dims the reduction names. A normal reduce **removes** them; with
        ``keepdims=True`` each is kept at size 1 instead (see :attr:`keepdims`).

    Notes
    -----
    ``consumes`` is stored, parsed by ``to_opnode`` from the ``dim`` argument. A bare
    ``mean()`` names no dim, so it consumes :data:`ALL_DIMS`, left symbolic rather than
    expanded against the schema at recording time (see the module docstring).

    ``args``/``kwargs`` are converted to immutable containers so a node is safe to share
    between plans, and hashable when its contents are.
    """

    name: str  # open set of tabulated reductions → str (kind-safety via OP_TABLE)
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    consumes: DimSet = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        if not isinstance(self.consumes, AllDims):
            object.__setattr__(self, "consumes", frozenset(self.consumes))

    @property
    def keepdims(self) -> bool:
        """Whether the reduction keeps its named dims at size 1 (``keepdims=True``).

        Returns
        -------
        bool
            ``True`` if the call passed ``keepdims=True``; the reduce then resizes each
            dim in :attr:`consumes` to 1 instead of removing it. Derived from
            ``kwargs``, so it can't disagree with what replay does (same approach as
            ``Project.single``).
        """
        return bool(self.kwargs.get("keepdims", False))


@dataclass(frozen=True)
class Select:
    """An ``isel``/``sel`` selection, described by its ``{dim: indexer}`` mapping.

    Attributes
    ----------
    name : {"isel", "sel"}
        Which selection this is. A closed set, so a ``Literal``; ``Select(name="mean")``
        is a type error.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    indexer : frozendict
        The ``{dim: indexer}`` mapping the optimiser reasons about.

    Notes
    -----
    Each indexer value is an :data:`~xrexpr.indexers.Indexer` variant. ``__post_init__``
    converts raw values with :func:`~xrexpr.indexers.classify`, so the values are always
    modelled variants, whether the node was recorded or built by hand.

    ``consumes`` (the dims indexed by a scalar, which get dropped) is a ``@property``
    derived from ``indexer``, not a stored field. So a merged select can't end up with a
    ``consumes`` that disagrees with its indexer.
    """

    name: Literal["isel", "sel"]  # closed set → Literal (rejects Select(name="mean"))
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    indexer: frozendict[Hashable, Indexer] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(
            self,
            "indexer",
            frozendict(
                {
                    dim: v if isinstance(v, Indexer) else classify(v)
                    for dim, v in self.indexer.items()
                }
            ),
        )

    @property
    def consumes(self) -> frozenset[Hashable]:
        """The dims this select drops: the scalar-indexed ones. Slices and lists keep theirs.

        Returns
        -------
        frozenset of Hashable
            The dropped dims, derived from :attr:`indexer` so the two can't disagree.
        """
        return frozenset(d for d, v in self.indexer.items() if v.drops_dim)


@dataclass(frozen=True)
class Scan:
    """An order-sensitive scan (``cumsum``/``cumprod``/``diff``). Keeps its dims.

    Attributes
    ----------
    name : {"cumsum", "cumprod", "diff"}
        Which scan this is. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    dims : DimSet
        The dims the scan runs along, parsed by ``to_opnode`` the same way as
        ``Reduce.consumes``: a concrete frozenset, or :data:`ALL_DIMS` for a bare
        ``cumsum()``. Called ``dims`` rather than ``consumes`` because a scan keeps
        them.

    Notes
    -----
    Different from a reduce (which removes its dims) and from an opaque op (a scan is
    known to keep its dims, but a rule still mustn't reorder across it along those
    dims). :func:`~xrexpr.optimize.dim_effect` reads :attr:`dims` to let a select or
    projection on other dims cross the scan.
    """

    name: Literal["cumsum", "cumprod", "diff"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    dims: DimSet = frozenset()  # ``frozenset | AllDims``, as ``Reduce.consumes``

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        if not isinstance(self.dims, AllDims):
            object.__setattr__(self, "dims", frozenset(self.dims))


@dataclass(frozen=True)
class Elementwise:
    """A per-element op (``fillna``/``astype``/``round``/...). Keeps every dim, size and
    variable, so it commutes with any select or projection.

    Attributes
    ----------
    name : str
        The method name. An open set, so ``str``; :data:`~xrexpr.operations.OP_TABLE`
        says which names are elementwise, as for :class:`Reduce`.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.

    Notes
    -----
    The difference from :class:`Opaque` is exactly what the optimiser needs to know: an
    elementwise op is known to keep dims, sizes and variables. So
    :func:`~xrexpr.schema.apply_schema` is exact for it, and
    :func:`~xrexpr.optimize.dim_effect` lets both pushdowns cross it. An ``Opaque``
    blocks both.

    Only recorded **when every argument is a plain value** (see
    ``schema._elementwise_safe``). ``fillna(da)`` fills from another array, and
    ``fillna({"tas": 0})`` is per-variable. Neither commutes with a projection, so both
    are recorded as :class:`Opaque`. :class:`Project` vs :class:`Opaque` is decided by
    the shape of the ``__getitem__`` key in the same way.
    """

    name: str  # open set of elementwise ops → str, kind-safety via OP_TABLE
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


@dataclass(frozen=True)
class Project:
    """A variable projection: ``ds["tas"]`` or ``ds[["tas", "pr"]]``.

    Attributes
    ----------
    name : {"__getitem__"}
        Always ``"__getitem__"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim (the key).
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    variables : tuple of Hashable
        The requested variable names, in order.

    Notes
    -----
    The one op recognised by the *shape of its key* as well as its method name.
    ``__getitem__``'s ``OP_TABLE`` row is a :class:`~xrexpr.operations.ProjectSpec`, but
    the call is only a projection when its key names variables (a boolean-mask key stays
    :class:`Opaque`). ``to_opnode`` checks that with a guard.

    ``single`` (does the call return a ``DataArray`` from a bare name, rather than a
    ``Dataset`` from a list) is derived from the verbatim key, never stored, so it can't
    disagree with replay. The list-vs-hashable split follows xarray's own
    ``Dataset.__getitem__``, so a tuple key is one name, not several.
    """

    name: Literal["__getitem__"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    variables: tuple[Hashable, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "variables", tuple(self.variables))

    @property
    def single(self) -> bool:
        """Whether this projects *one* variable to a ``DataArray`` (``ds["tas"]``).

        Returns
        -------
        bool
            ``True`` for a bare-name key, ``False`` for a list. Derived from the
            verbatim key, so it can't disagree with replay.
        """
        return bool(self.args) and not isinstance(self.args[0], list)


@dataclass(frozen=True)
class Rechunk:
    """A ``chunk`` call. Changes how data is split into blocks, never a dim, size or value.

    Attributes
    ----------
    name : {"chunk"}
        Always ``"chunk"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    chunks : frozendict
        The per-dim form, ``{dim: spec}``. See the notes.
    uniform : ChunkSpec or None
        The uniform form, one spec for every dim. See the notes. Derived from ``args``,
        never passed in.

    Notes
    -----
    A ``chunk`` call gives its spec in one of two ways, and the node has one field for
    each:

    - ``chunks``: the **per-dim form** ``{dim: spec}`` (a positional dict and/or dim
      kwargs);
    - ``uniform``: the **uniform form**, ``chunk(100)``, ``chunk("auto")``,
      ``chunk((100, 400, 500))``, which names no dim.

    At most one is set; ``chunk()`` sets neither.

    The split matters for rewrites. Only *named* dims have to be removed from the spec
    when a select drops them, and only a per-dim spec can end up empty. A uniform spec
    has no keys to remove: xarray reads it as ``dict.fromkeys(self.dims, spec)``, so it
    already means "whatever dims there are", and a select moved in front just leaves
    fewer of them.

    Both fields hold :data:`~xrexpr.chunks.ChunkSpec` variants. ``__post_init__``
    converts raw values with :func:`~xrexpr.chunks.classify_chunk`, as :class:`Select`
    does for its indexers, so values are always modelled variants. Because both forms
    use the same variants, ``chunk("auto")`` and ``chunk({dim: "auto"})`` look the same
    to a rule, which needs no second code path.

    ``uniform`` is derived (``init=False``) because it's just a reading of ``args``. If
    it could be passed in, it could disagree with ``args``, and the node would replay as
    something other than what it says. ``chunks`` *is* passed in, because extracting it
    means knowing which kwargs are options rather than dims, which is ``_chunk_spec``'s
    job.

    Whether a rechunk may be crossed isn't decided here. That's the rule's call
    (``_pushable_rechunk`` in ``optimize.py``), as it is for selects.
    """

    name: Literal["chunk"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    chunks: frozendict[Hashable, ChunkSpec] = field(default_factory=frozendict)
    uniform: ChunkSpec | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(
            self,
            "chunks",
            frozendict(
                {
                    dim: v if isinstance(v, ChunkSpec) else classify_chunk(v)
                    for dim, v in self.chunks.items()
                }
            ),
        )
        # A leading non-mapping positional is the uniform form. A leading mapping is the
        # per-dim form, already extracted into ``chunks``. This is the same
        # ``isinstance(chunks, Mapping)`` split xarray's ``Dataset.chunk`` makes.
        object.__setattr__(
            self,
            "uniform",
            (
                classify_chunk(self.args[0])
                if self.args and not isinstance(self.args[0], Mapping)
                else None
            ),
        )


@dataclass(frozen=True)
class Opaque:
    """Any op the optimiser doesn't model. Replayed verbatim, never reordered.

    Attributes
    ----------
    name : str
        The method name, verbatim. Can be anything.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


@dataclass(frozen=True)
class Drop:
    """A ``drop_vars`` call: ``ds.drop_vars("region")`` or ``ds.drop_vars(["a", "b"])``.

    Attributes
    ----------
    name : {"drop_vars"}
        Always ``"drop_vars"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    variables : tuple of Hashable
        The names being dropped, in order.

    Notes
    -----
    Modelled rather than :class:`Opaque` so the schema fold stays exact across it. The
    named variables and coordinates leave the schema, and any dim no longer spanned by
    anything is removed by :class:`~xrexpr.schema.SchemaState`. Replay is a verbatim
    ``drop_vars``; the point of modelling it is that the optimiser can see past it
    (``planning/roadmap/11-relabel-rename-drop.md``).

    It doesn't touch dims (unlike :class:`Rename`), so a select commutes with it freely.
    A *projection* may not cross it blindly, though: moved in front, it could remove the
    very name ``drop_vars`` targets and turn a valid chain into a ``KeyError``.
    """

    name: Literal["drop_vars"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    variables: tuple[Hashable, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "variables", tuple(self.variables))


@dataclass(frozen=True)
class Rename:
    """A ``rename`` call: ``ds.rename({"month": "time"})`` or ``ds.rename(month="time")``.

    Attributes
    ----------
    name : {"rename"}
        Always ``"rename"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    mapping : frozendict
        The ``{old: new}`` mapping, collected from the positional dict and/or the
        keyword arguments.

    Notes
    -----
    Modelled rather than :class:`Opaque` so the schema fold stays exact across it. A
    rename can relabel a **dim**, a **variable or coordinate**, or both at once (for a
    dimension coordinate). :func:`~xrexpr.schema._relabelled` applies it as one
    shape-preserving relabel. Replay is a verbatim ``rename``; modelling it is what lets
    the optimiser see past it (``planning/roadmap/11-relabel-rename-drop.md``).

    Unlike :class:`Drop`, a rename that touches a dim changes dim *names* mid-plan, and
    the pushdown rules don't translate names across it. A select on an untouched dim
    still commutes with it. A select on a renamed dim stays put, because it uses the new
    name, which doesn't exist before the rename.
    """

    name: Literal["rename"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    mapping: frozendict[Hashable, Hashable] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "mapping", frozendict(self.mapping))


@dataclass(frozen=True)
class ContextOpen:
    """A call that returns a builder: ``groupby``/``rolling``/``weighted``/... .

    Attributes
    ----------
    name : ContextOpenName
        Which builder the call opens. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.

    Notes
    -----
    **Fluent IR only.** xarray spells some single operations as two calls through a
    builder object, and this is the first call of the pair. It only says *a context
    opens here*. That can be decided from the call alone (``groupby(...)`` opens one
    whatever follows), so ``to_opnode`` can record it without looking ahead. What the
    pair *means* is decided by :func:`~xrexpr.lower.to_lower_ir`.

    There's one node kind for all builders, rather than one per builder. The openers all
    have the same structure (a verbatim call plus which builder it is). The per-builder
    detail belongs on the fused nodes, where there's structured data to match on. Five
    empty arms at every ``assert_never`` site would add nothing.

    It must not survive lowering, and the types enforce that: it isn't in
    :data:`LoweredOp`, so one that outlives ``to_lower_ir`` is a type error.
    """

    name: "ContextOpenName"
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


#: The builder-returning methods. This ``Literal`` is the single list of them. The
#: runtime ``ContextSpec`` rows in ``operations.OP_TABLE`` are built from it with
#: ``get_args``, so the type and the dispatch can't disagree. A test checks the names
#: against ``xr.Dataset``.
ContextOpenName = Literal[
    "groupby",
    "groupby_bins",
    "resample",
    "rolling",
    "rolling_exp",
    "coarsen",
    "weighted",
    "cumulative",
]


@dataclass(frozen=True)
class GroupedReduce:
    """A grouped aggregation, ``ds.groupby("time.month").mean()``, as *one* node.

    Attributes
    ----------
    name : {"groupby", "groupby_bins", "resample"}
        Which opener was fused. A closed set, so a ``Literal``.
    group_dim : Hashable
        The dim grouped along, which is consumed.
    new_dim : Hashable
        The dim created to index the groups.
    reduce : str
        The closing method: ``mean``/``sum``/``std``/... . An open set, so ``str``.
    args : tuple
        The opener's positional arguments, verbatim.
    kwargs : frozendict
        The opener's keyword arguments, verbatim.
    reduce_args : tuple
        The closer's positional arguments, verbatim.
    reduce_kwargs : frozendict
        The closer's keyword arguments, verbatim.
    consumes : frozenset of Hashable
        What the closing reduction removes *in addition to* ``group_dim``.

    Notes
    -----
    Flat, not nested: the semantic fields rules match on, plus the two verbatim calls
    :func:`~xrexpr.lower.emit` replays. "A context wrapping a call" is how xarray
    *spells* this, not what it is, and dropping that spelling is the point of lowering.

    The dim behaviour, checked against xarray 2026.7.0 (some of it is surprising):

    - ``group_dim`` is consumed and ``new_dim`` created. ``groupby("time.month")``
      groups along ``time`` and gives ``month``. ``groupby("lat")`` and
      ``resample(time="2D")`` recreate the dim they group over. ``groupby_bins("lat",
      2)`` gives ``lat_bins``.
    - ``consumes`` is what the closer removes **in addition**, and for a bare ``mean()``
      that's *nothing*. Unlike a Dataset-level bare reduce, a grouped one only reduces
      along the group dim: ``ds.groupby("time.month").mean()`` keeps ``lat`` and
      ``lon``.
    - Every variable ends up with ``new_dim``, even ones that never had ``group_dim``
      (``elevation(lat, lon)`` comes back as ``(month, lat, lon)``).

    ``new_dim`` isn't optional here, even though a grouped reduce in general may create
    nothing. ``ds.groupby("time.month").mean("lat")`` is a per-group *map* that keeps
    ``time`` and creates no ``month``. That case doesn't fuse (see
    :func:`~xrexpr.lower.to_lower_ir`), so every node of this type is an aggregation,
    and every aggregation creates a dim.
    """

    name: Literal["groupby", "groupby_bins", "resample"]  # closed set → Literal
    group_dim: Hashable
    new_dim: Hashable
    reduce: str  # the closing method: mean/sum/std/... (open set → str)
    args: tuple[Any, ...] = ()  # the opener's header, verbatim
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()  # the closer's header, verbatim
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    consumes: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "reduce_args", tuple(self.reduce_args))
        object.__setattr__(self, "reduce_kwargs", frozendict(self.reduce_kwargs))
        object.__setattr__(self, "consumes", frozenset(self.consumes))


@dataclass(frozen=True)
class WindowedReduce:
    """A windowed aggregation, ``ds.rolling(time=5).mean()``, as *one* node.

    Attributes
    ----------
    name : {"rolling", "coarsen"}
        Which opener was fused. A closed set, so a ``Literal``.
    reduce : str
        The closing method: ``mean``/``sum``/``max``/... . An open set, so ``str``.
    window : frozendict
        The ``{dim: window}`` mapping of windowed dims.
    args : tuple
        The opener's positional arguments, verbatim.
    kwargs : frozendict
        The opener's keyword arguments, verbatim, kept whole for replay. ``coarsen``'s
        ``boundary`` is in here too, but the part that affects sizes is stored in
        :attr:`rounds_up`.
    reduce_args : tuple
        The closer's positional arguments, verbatim.
    reduce_kwargs : frozendict
        The closer's keyword arguments, verbatim.

    Notes
    -----
    A window can't consume or create dims, so there's no ``consumes`` field. What can
    change is the *size* of the windowed dims, and only for ``coarsen``:

    - **rolling** keeps every dim at its current length: one output per input position.
      ``center`` and ``min_periods`` change values, not shape.
    - **coarsen** divides each windowed dim by its window, rounded according to
      ``boundary`` (``"trim"`` down, ``"pad"`` up, ``"exact"`` requires it to divide
      evenly). Because the size depends on that option, its meaning is stored in
      :attr:`rounds_up`, derived from :attr:`kwargs` in ``__post_init__``. That's the
      same approach as :class:`Reduce`'s ``keepdims`` and :class:`Rechunk`'s
      ``uniform``: it can't disagree with replay, and ``schema._windowed_size`` reads
      the node instead of digging ``boundary`` out of the kwargs. ``kwargs`` stays whole
      for replay and for the other options (``center``, ``min_periods``, ...), which
      nothing reasons about.

    A windowed closer takes **no dim argument**: ``DatasetRolling.mean`` is
    ``(keep_attrs=None, **kwargs)``. So ``ds.rolling(time=3).mean("lat")`` passes
    ``"lat"`` as ``keep_attrs`` and reduces nothing. Such a closer still fuses (see
    :func:`~xrexpr.lower.to_lower_ir`). ``to_opnode`` will have misparsed ``"lat"`` as
    ``closer.consumes``; that's dropped, and this node reads neither it nor
    ``reduce_args``. ``reduce_args`` is kept verbatim only so replay is faithful
    (including the truthiness of ``keep_attrs``).
    """

    name: Literal["rolling", "coarsen"]  # closed set → Literal
    reduce: str  # the closing method: mean/sum/max/... (open set → str)
    window: frozendict[Hashable, int] = field(default_factory=frozendict)
    args: tuple[Any, ...] = ()  # the opener's header, verbatim
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()  # the closer's header, verbatim
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    rounds_up: bool | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", frozendict(self.window))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "reduce_args", tuple(self.reduce_args))
        object.__setattr__(self, "reduce_kwargs", frozendict(self.reduce_kwargs))
        object.__setattr__(self, "rounds_up", self._rounds_up())

    def _rounds_up(self) -> bool | None:
        """Whether a ``coarsen`` rounds its windowed dims up (ceil) or down (floor).

        Returns
        -------
        bool | None
            ``True`` for ceil, ``False`` for floor, ``None`` if not known (or not a
            coarsen).
        """
        match self.name, self.kwargs.get("boundary", "exact"):
            case "coarsen", "pad":  # ceil
                return True
            case "coarsen", "trim" | "exact":  # floor
                return False
            case _:  # rolling (length kept), or a boundary this does not model
                return None


@dataclass(frozen=True)
class WeightedReduce:
    """A weighted aggregation, ``ds.weighted(w).mean("time")``, as *one* node.

    Attributes
    ----------
    name : {"weighted"}
        Always ``"weighted"``. A closed set, so a ``Literal``.
    reduce : str
        The closing method: ``mean``/``sum``/``std``/``var``. An open set, so ``str``.
    weight_dims : frozenset of Hashable
        The dims of the weights. Read off ``w.dims`` at fusion time (metadata only,
        nothing computed). This is the reason the node isn't just a :class:`Reduce`.
    args : tuple
        The opener's positional arguments, verbatim. Holds the weights themselves.
    kwargs : frozendict
        The opener's keyword arguments, verbatim.
    reduce_args : tuple
        The closer's positional arguments, verbatim.
    reduce_kwargs : frozendict
        The closer's keyword arguments, verbatim.
    consumes : DimSet
        The dims the closing reduction removes, exactly as for a plain reduce.

    Notes
    -----
    This node exists as much to **block** rewrites as to describe the operation. Lowered
    to a plain :class:`Reduce`, it would look like one, and ``pushdown_selects`` would
    move a select in front of a weighted mean while leaving the weights un-subset.
    That's a wrong plan, with no error. **No select ever crosses this node.** Subsetting
    ``w`` alongside would be the first rewrite here that changes array data rather than
    reordering metadata, and that needs its own design
    (``planning/roadmap/02-lowering.md`` §8.1).

    A *projection* can cross it, and that's what ``weight_dims`` is for. A projection
    removes variables rather than subsetting them, so the weights don't need rewriting.
    But it can drop a dim **coordinate**, and whether that coordinate exists decides
    whether the weights align against the dim or broadcast a new one. So the rule
    requires every weight dim to survive the projection. See ``optimize.dim_effect``.

    The dim behaviour, checked against xarray 2026.7.0, is **not** a plain ``Reduce``'s.
    ``consumes`` is a reduce's: :data:`ALL_DIMS` for a bare closer
    (``DatasetWeighted.mean`` really does take ``dim``, unlike ``DatasetRolling.mean``).
    But the weights have their own dim effect, because ``dot`` broadcasts and aligns:

    - a weight dim the dataset **lacks** is broadcast onto *every* variable. With
      ``w(lat, member)``, ``ds.weighted(w).mean("lat")`` returns ``{time, lon,
      member}``, and ``elevation(lat, lon)`` comes back as ``(lon, member)``;
    - a weight dim the dataset **has** is aligned, and misaligned weights *shrink* it.
      ``w`` on ``lat`` labelled ``[0, 1]`` against a ``lat`` of ``[0, 1, 2]``
      inner-joins to size 2.

    So after the reduce, a surviving weight dim has a known *name* but an unknown
    *size*. :class:`~xrexpr.schema.SchemaState` records its size as ``None``. A bare
    closer removes everything, weight dims included (also checked).

    Only the weight dims' **names** are stored, not their sizes. Sizes would be a guess
    about the post-alignment extent, and names are what every rule reasons about. The
    weights themselves stay in the verbatim ``args``, so this node is unhashable when
    they are (see the module docstring).
    """

    name: Literal["weighted"]  # closed set → Literal
    reduce: str  # the closing method: mean/sum/std/var (open set → str)
    weight_dims: frozenset[Hashable] = frozenset()
    args: tuple[Any, ...] = ()  # the opener's header, verbatim (holds the weights)
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()  # the closer's header, verbatim
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    consumes: DimSet = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_dims", frozenset(self.weight_dims))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "reduce_args", tuple(self.reduce_args))
        object.__setattr__(self, "reduce_kwargs", frozendict(self.reduce_kwargs))
        if not isinstance(self.consumes, AllDims):
            object.__setattr__(self, "consumes", frozenset(self.consumes))


#: The optimiser's IR node: a union over the op kinds. ``match`` binds different fields
#: per arm, and ``assert_never`` on the ``case _`` arm makes the match exhaustive
#: (adding a kind fails type-checking at every site that doesn't handle it).
Op = Reduce | Select | Scan | Elementwise | Project | Drop | Rename | Rechunk | Opaque

#: What recording produces: one node per call, as the user wrote it, including the
#: half-operations (:class:`ContextOpen`) that only mean something as a pair.
FluentOp = Op | ContextOpen

#: What the optimiser rewrites: the same node kinds, plus the fused nodes for operations
#: that take two calls to spell, and *minus* :class:`ContextOpen`, which lowering must
#: have removed. So a ``ContextOpen`` that survived ``to_lower_ir`` fails type-checking
#: everywhere downstream.
LoweredOp = Op | GroupedReduce | WindowedReduce | WeightedReduce
