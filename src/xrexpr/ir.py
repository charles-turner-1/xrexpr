"""The expression IR — a sum type over the operation *kinds* the optimiser distinguishes.

A method chain is linear, so a plan is a **list** of :data:`Op`, not a tree. Each variant
carries the verbatim call header replay re-invokes *plus* the normalised metadata the
optimiser reasons about, and that metadata differs per kind, so the variants have
genuinely different shapes rather than one flat record. :data:`FluentOp` and
:data:`LoweredOp` name the two levels over one set of dataclasses, so the level a
function works at is legible in its signature.

Three invariants to keep when editing here. Every ``match`` over one of these unions
closes with ``assert_never``, so a new variant is a type error at every unhandled site
rather than a silent fallthrough. Dim sets are symbolic where the call is — a bare
``ds.mean()`` records :data:`ALL_DIMS`, a *sentinel*, distinct from ``None``, which means
*unknown*. And a node is frozen unconditionally but hashable only when its payload is; an
array payload raises, deliberately, because hashing it would mean computing it.

Only **unary** ops are modelled. Keep that linearity assumption named here rather than
leaked into individual rules.

See ``docs/internals/ir.md``; ``test_ir.py`` pins the hashability behaviour.
"""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, final

from frozendict import frozendict

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
    "frozendict",
]


@final
@dataclass(frozen=True)
class AllDims:
    """Sentinel: *every dim present at this point*, whatever they turn out to be.

    Notes
    -----
    The dim set of a call that names none — ``ds.mean()``, ``ds.mean(dim=None)``. It
    stays unexpanded until a reader with an exact schema resolves it, which is what
    keeps a record-time guess about the dims from being frozen into the plan.

    Fieldless and frozen, so every instance compares and hashes equal; :data:`ALL_DIMS`
    is the one to use.
    """

    def __repr__(self) -> str:
        return "ALL_DIMS"


#: The singleton :class:`AllDims`.
ALL_DIMS: Final = AllDims()

#: The dims an op removes — a concrete set, or :data:`ALL_DIMS` for a call that named
#: none. Readers must handle both — see this module's docstring.
DimSet = frozenset[Hashable] | AllDims


@dataclass(frozen=True)
class Reduce:
    """A dimension-destroying reduction (``mean``/``sum``/``std``/...).

    Attributes
    ----------
    name : str
        The method name. An open set of tabulated reductions, so ``str`` — kind-safety
        comes from ``operations.OP_TABLE``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    consumes : DimSet
        The dims the reduction names. What it *does* to them depends on
        :attr:`keepdims`: an ordinary reduce **removes** them; a ``keepdims=True`` one
        **resizes** each to 1 and keeps it (see :attr:`keepdims`).

    Notes
    -----
    ``consumes`` is *stored*, parsed by ``to_opnode`` from the ``dim`` spec. A bare
    ``mean()`` names no dim and so consumes :data:`ALL_DIMS`, left symbolic rather than
    expanded against the record-time schema (see the module docstring).
    ``args``/``kwargs`` are coerced to immutable containers so the node is safe to share
    between plans, and hashable when its payload is (see the module docstring on why that
    last part is conditional).
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
            ``True`` when the call passed ``keepdims=True`` — the reduce then resizes
            each dim in :attr:`consumes` to 1 rather than removing it. Derived from the
            verbatim ``kwargs``, so it cannot disagree with what replay does (the
            ``Project.single`` precedent).
        """
        return bool(self.kwargs.get("keepdims", False))


@dataclass(frozen=True)
class Select:
    """An ``isel``/``sel`` selection, described by its ``{dim: indexer}`` mapping.

    Attributes
    ----------
    name : {"isel", "sel"}
        Which selection this is. A closed set, so a ``Literal`` — it rejects
        ``Select(name="mean")``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    indexer : frozendict
        The ``{dim: indexer}`` mapping the optimiser reasons about.

    Notes
    -----
    Each indexer value is an :data:`~xrexpr.indexers.Indexer` — the closed value sum type
    the optimiser reasons about — normalised from its raw ``isel``/``sel`` form by
    ``__post_init__`` (via :func:`~xrexpr.indexers.classify`), so a value is *always* a
    modelled variant regardless of whether the node was recorded or hand-built.

    ``consumes`` is a *derived* view of ``indexer`` (the scalar-indexed dims, which
    drop) — a ``@property``, never a stored field — so a merged select cannot disagree
    with itself the way a separately-accumulated ``consumes`` could.
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
        """Dims this select drops — the scalar-indexed ones (slices/sequences keep theirs).

        Returns
        -------
        frozenset of Hashable
            The dropped dims, derived from :attr:`indexer` so the two cannot disagree.
        """
        return frozenset(d for d, v in self.indexer.items() if v.drops_dim)


@dataclass(frozen=True)
class Scan:
    """An order-significant scan (``cumsum``/``cumprod``/``diff``) — *keeps* its dim.

    Attributes
    ----------
    name : {"cumsum", "cumprod", "diff"}
        Which scan this is. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    dims : DimSet
        The dims the scan runs along, parsed by ``to_opnode`` from the call's dim spec
        exactly as ``Reduce.consumes`` is — a concrete frozenset, or :data:`ALL_DIMS`
        for a bare ``cumsum()`` that named none. A scan *keeps* these dims (the whole
        point of the variant), so the field is ``dims`` rather than ``consumes``.

    Notes
    -----
    Distinct from a reduce (which destroys its dim) and from an opaque op (a scan is
    *known* to preserve dims, so a rule must not reorder across it). The scanned-dim
    metadata that :func:`~xrexpr.optimize.dim_effect` reads to hop a disjoint select or
    projection across the scan lives in :attr:`dims`.
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
    """A per-element op (``fillna``/``astype``/``round``/...) — keeps every dim, size and
    variable, and so commutes with any select or projection.

    Attributes
    ----------
    name : str
        The method name. An open set of tabulated elementwise ops, so ``str``; kind-safety
        comes from :data:`~xrexpr.operations.OP_TABLE` carrying the row, as for
        :class:`Reduce`.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.

    Notes
    -----
    Distinct from :class:`Opaque` in exactly the fact the optimiser needs: an elementwise
    op is *known* to preserve dims, sizes and variables, so :func:`~xrexpr.schema.apply_schema`
    is exact for it (a pass-through) and :func:`~xrexpr.optimize.dim_effect` lets both
    pushdowns cross it, where an ``Opaque`` is a full barrier.

    Minted **only when every argument is a plain value** (see
    ``schema._elementwise_safe``): ``fillna(da)`` fills from another array and
    ``fillna({"tas": 0})`` is per-variable — neither commutes with a projection the way a
    plain scalar does, so both demote to :class:`Opaque` at record time. The shape of an
    argument deciding the kind has precedent in :class:`Project` vs :class:`Opaque` on the
    ``__getitem__`` key.
    """

    name: str  # open set of elementwise ops → str, kind-safety via OP_TABLE
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


@dataclass(frozen=True)
class Project:
    """A variable projection — ``ds["tas"]`` or ``ds[["tas", "pr"]]``.

    Attributes
    ----------
    name : {"__getitem__"}
        Always ``"__getitem__"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim — the key itself.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    variables : tuple of Hashable
        The requested variable names, in order.

    Notes
    -----
    The one op recognised by the *shape of its key* as well as by a method name:
    ``__getitem__``'s ``OP_TABLE`` row is a :class:`~xrexpr.operations.ProjectSpec`, but
    the same call is a projection only when its key names variables (a boolean-mask key
    stays :class:`Opaque`), so ``to_opnode`` confirms the nomination with a guard.

    ``single`` — whether the call returns a ``DataArray`` (a bare name) rather than a
    ``Dataset`` (a list of them) — is *derived* from the verbatim key, never stored, so
    it cannot disagree with what replay will actually do. The list/hashable split mirrors
    xarray's own ``Dataset.__getitem__``, so a tuple key reads as one name rather than
    several.
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
            ``True`` for a bare-name key, ``False`` for a list of them — derived from the
            verbatim key, so it cannot disagree with what replay does.
        """
        return bool(self.args) and not isinstance(self.args[0], list)


@dataclass(frozen=True)
class Rechunk:
    """A ``chunk`` call: changes chunk topology only — never a dim, size or value.

    Attributes
    ----------
    name : {"chunk"}
        Always ``"chunk"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    chunks : frozendict
        The mapping-form ``{dim: spec}`` — see the notes.
    uniform : ChunkSpec or None
        The **uniform**-form spec, the one that applies to every dim — see the notes.
        Derived from ``args``, never passed.

    Notes
    -----
    A ``chunk`` call writes its spec in one of two places, and the node holds one field per
    place. ``chunks`` holds the **mapping form** ``{dim: spec}`` (a positional dict and/or
    dim kwargs); ``uniform`` holds the **uniform form** — ``chunk(100)``, ``chunk("auto")``,
    ``chunk((100, 400, 500))`` — whose spec names no dim. Exactly one of them is populated,
    and ``chunk()`` populates neither.

    That split is what a rewrite needs: only *named* dims have to be stripped from the spec
    when a select drops them, and only a named-dim spec can be emptied out entirely. A
    uniform spec has no key to strip and needs none — xarray reads it as
    ``dict.fromkeys(self.dims, spec)``, so it already means "whatever dims there are", and a
    select moving in front simply leaves fewer of them.

    Both fields hold :data:`~xrexpr.chunks.ChunkSpec` values — the closed value sum type the
    optimiser reasons about — normalised from their raw ``chunk`` forms by ``__post_init__``
    (via :func:`~xrexpr.chunks.classify_chunk`), so a value is *always* a modelled variant
    regardless of whether the node was recorded or hand-built. Exactly what
    :class:`Select` does for its indexers, and for the same reason. One taxonomy answering
    for both spellings is the point: ``chunk("auto")`` is the identical request to
    ``chunk({dim: "auto"})`` for every dim, so a rule that reasons about one reasons about
    the other without a second code path.

    ``uniform`` is derived rather than passed (``init=False``) because it is a *reading* of
    ``args`` rather than an independent fact, and one that disagreed with ``args`` would
    make the node replay as something other than what it claims. ``chunks`` is passed, since
    extracting it means knowing which kwargs are options rather than dims —
    ``_chunk_spec``'s job.

    Whether a given rechunk may be *crossed* is deliberately not decided here — that
    judgement lives with the rule (``_pushable_rechunk`` in ``optimize.py``), as it does
    for selects.
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
        # A leading *non*-mapping positional is the uniform form; a leading mapping is the
        # mapping form, already extracted into ``chunks``. The split xarray's own
        # ``Dataset.chunk`` makes with ``isinstance(chunks, Mapping)``.
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
    """Any op the optimiser doesn't model — replayed verbatim, never reordered.

    Attributes
    ----------
    name : str
        The method name, verbatim. Any name at all, which is the point of the variant.
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
    """A ``drop_vars`` call — ``ds.drop_vars("region")`` or ``ds.drop_vars(["a", "b"])``.

    Attributes
    ----------
    name : {"drop_vars"}
        Always ``"drop_vars"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim — the key itself.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    variables : tuple of Hashable
        The names being dropped, in order.

    Notes
    -----
    Modelled rather than :class:`Opaque` so the schema fold stays exact across it: the
    named variables and coordinates leave the schema, and any dimension left spanned by
    nothing is pruned by :class:`~xrexpr.schema.SchemaState`. Replay is a verbatim
    ``drop_vars`` — the only thing the modelling buys is that the optimiser can see past
    it, which is what ``planning/roadmap/11-relabel-rename-drop.md`` (W11) exists for.

    Dim-transparent: unlike :class:`Rename` it remaps no dim name, so a select commutes
    with it freely. A *projection* may not cross it, though — moving one earlier could
    drop the very name ``drop_vars`` targets, turning a valid chain into a ``KeyError``.
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
    """A ``rename`` call — ``ds.rename({"month": "time"})`` or ``ds.rename(month="time")``.

    Attributes
    ----------
    name : {"rename"}
        Always ``"rename"``. A closed set, so a ``Literal``.
    args : tuple
        The call's positional arguments, verbatim.
    kwargs : frozendict
        The call's keyword arguments, verbatim.
    mapping : frozendict
        The ``{old: new}`` relabelling, gathered from the positional dict and/or the name
        kwargs.

    Notes
    -----
    Modelled rather than :class:`Opaque` so the schema fold stays exact across it: a rename
    relabels a **dim**, a **variable/coordinate**, or -- for a dimension coordinate -- both
    at once, which :func:`~xrexpr.schema._relabelled` applies as a single shape-preserving
    relabel. Replay is a verbatim ``rename``; the modelling is what lets the optimiser see
    past it (``planning/roadmap/11-relabel-rename-drop.md``, W11).

    Unlike :class:`Drop` a rename that touches a dim changes dim *names* mid-plan, and the
    name-keyed pushdowns do not translate keys across it, so it is **not** dim-transparent.
    A select on an untouched dim still commutes with it; one on a renamed dim is left
    (it names the new name, which does not exist above the rename).
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
    """A builder-returning call — ``groupby``/``rolling``/``weighted``/... .

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
    **Fluent IR only.** xarray spells some single semantic operations as two calls via a
    builder object, and this is the first of the pair: it says *a context opens here*,
    which is decidable from the call alone (``groupby(...)`` opens one no matter what
    follows), so ``to_opnode`` can type it without needing the lookahead it hasn't got.
    What the pair *means* is :func:`~xrexpr.lower.to_lower_ir`'s job.

    One variant rather than one per builder: the openers carry identical structure — a
    verbatim header plus which builder it is — and per-builder differentiation belongs to
    the fused nodes, where it comes with structural data to match on. Five empty arms at
    every ``assert_never`` site would earn nothing.

    It must not survive lowering, and does not have to be trusted to: it is absent from
    :data:`LoweredOp`, so one that outlives ``to_lower_ir`` is a type error rather than a
    convention.
    """

    name: "ContextOpenName"
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


#: The builder-returning methods — **the** list of them, in the form mypy reads. The
#: runtime rows live in ``operations`` as ``ContextSpec``s in ``OP_TABLE``, but they are
#: built by ``get_args`` on this ``Literal`` rather than written out a second time, so the
#: type and the dispatch cannot disagree. A test pins the names against ``xr.Dataset``.
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
    """A grouped aggregation — ``ds.groupby("time.month").mean()`` — as *one* node.

    Attributes
    ----------
    name : {"groupby", "groupby_bins", "resample"}
        Which opener was fused. A closed set, so a ``Literal``.
    group_dim : Hashable
        The dim grouped along, and consumed.
    new_dim : Hashable
        The dim minted to index the groups.
    reduce : str
        The closing method — ``mean``/``sum``/``std``/... . An open set, so ``str``.
    args : tuple
        The opener's positional arguments, verbatim.
    kwargs : frozendict
        The opener's keyword arguments, verbatim.
    reduce_args : tuple
        The closer's positional arguments, verbatim.
    reduce_kwargs : frozendict
        The closer's keyword arguments, verbatim.
    consumes : frozenset of Hashable
        What the closing reduction removes *in addition* to ``group_dim``.

    Notes
    -----
    Flat, not nested: the semantic fields rules match on, plus the two verbatim call
    headers :func:`~xrexpr.lower.emit` replays. "A context wrapping a call" is how xarray
    *spells* this, not what it is, and discarding that spelling is what lowering is for.

    The dim algebra, verified against xarray 2026.7.0 and less obvious than it looks:

    - ``group_dim`` is consumed and ``new_dim`` minted — ``groupby("time.month")`` groups
      along ``time`` and yields ``month``; ``groupby("lat")`` and ``resample(time="2D")``
      mint the dim they group over; ``groupby_bins("lat", 2)`` yields ``lat_bins``.
    - ``consumes`` is what the closing reduction removes **in addition**, which for a bare
      ``mean()`` is *nothing*: unlike a Dataset-level bare reduce, a grouped one reduces
      only along the group dim, leaving the rest —
      ``ds.groupby("time.month").mean()`` keeps ``lat`` and ``lon``.
    - every variable ends up carrying ``new_dim``, including ones that never carried
      ``group_dim`` (``elevation(lat, lon)`` comes back as ``(month, lat, lon)``).

    ``new_dim`` is not optional here, though a grouped reduce in general may mint nothing:
    ``ds.groupby("time.month").mean("lat")`` is a per-group *map* that keeps ``time`` and
    mints no ``month``. That case deliberately does not fuse (see
    :func:`~xrexpr.lower.to_lower_ir`), so every node of this type is an aggregation and
    every aggregation mints.
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
    """A windowed aggregation — ``ds.rolling(time=5).mean()`` — as *one* node.

    Attributes
    ----------
    name : {"rolling", "coarsen"}
        Which opener was fused. A closed set, so a ``Literal``.
    reduce : str
        The closing method — ``mean``/``sum``/``max``/... . An open set, so ``str``.
    window : frozendict
        The ``{dim: window}`` mapping of windowed dims.
    args : tuple
        The opener's positional arguments, verbatim.
    kwargs : frozendict
        The opener's keyword arguments, verbatim. Kept whole because ``coarsen``'s size
        effect depends on the ``boundary`` option — see the notes.
    reduce_args : tuple
        The closer's positional arguments, verbatim.
    reduce_kwargs : frozendict
        The closer's keyword arguments, verbatim.

    Notes
    -----
    Simpler than :class:`GroupedReduce`, in the direction that matters: a window consumes
    no dim and mints none, so there is no ``consumes`` field to get wrong. What changes is
    only the *size* of the windowed dims, and only for ``coarsen``:

    - **rolling** keeps every dim at its current length — one output position per input
      position, ``center``/``min_periods`` affecting values but never shape;
    - **coarsen** divides each windowed dim by its window, rounded according to the
      ``boundary`` kwarg (``"trim"`` down, ``"pad"`` up, ``"exact"`` requiring
      divisibility). That is a size effect that depends on an *option*, which is why
      ``kwargs`` is kept whole rather than reduced to ``window``.

    A windowed closer takes **no dim argument** at all — ``DatasetRolling.mean`` is
    ``(keep_attrs=None, **kwargs)`` — so ``ds.rolling(time=3).mean("lat")`` passes
    ``"lat"`` as ``keep_attrs`` and reduces nothing. Such a closer *is* fused (see
    :func:`~xrexpr.lower.to_lower_ir`): because the parsed dim is inert, ``closer.consumes``
    — a ``to_opnode`` misparse — is dropped rather than carried, and this node reads neither
    it nor ``reduce_args``. ``reduce_args`` is kept verbatim only for faithful replay, which
    preserves the ``keep_attrs`` truthiness. If a dim spec was parsed, it was never a dim
    spec.
    """

    name: Literal["rolling", "coarsen"]  # closed set → Literal
    reduce: str  # the closing method: mean/sum/max/... (open set → str)
    window: frozendict[Hashable, int] = field(default_factory=frozendict)
    args: tuple[Any, ...] = ()  # the opener's header, verbatim
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    reduce_args: tuple[Any, ...] = ()  # the closer's header, verbatim
    reduce_kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", frozendict(self.window))
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "reduce_args", tuple(self.reduce_args))
        object.__setattr__(self, "reduce_kwargs", frozendict(self.reduce_kwargs))


@dataclass(frozen=True)
class WeightedReduce:
    """A weighted aggregation — ``ds.weighted(w).mean("time")`` — as *one* node.

    Attributes
    ----------
    name : {"weighted"}
        Always ``"weighted"``. A closed set, so a ``Literal``.
    reduce : str
        The closing method — ``mean``/``sum``/``std``/``var``. An open set, so ``str``.
    weight_dims : frozenset of Hashable
        The dims the weights carry. Read off ``w.dims`` at fusion time — metadata only,
        materialising nothing — and the reason this node is not a relabelled
        :class:`Reduce`.
    args : tuple
        The opener's positional arguments, verbatim. Holds the weights themselves.
    kwargs : frozendict
        The opener's keyword arguments, verbatim.
    reduce_args : tuple
        The closer's positional arguments, verbatim.
    reduce_kwargs : frozendict
        The closer's keyword arguments, verbatim.
    consumes : DimSet
        The dims the closing reduction removes — exactly a plain reduce's.

    Notes
    -----
    The variant exists as much for what it **blocks** as for what it describes. Lowered to
    a plain :class:`Reduce` it would be indistinguishable from one, so ``pushdown_selects``
    would match it and hop a select in front of a weighted mean with the weights left
    un-subset — a wrong plan, silently. **No select ever crosses one** (see
    ``planning/roadmap/02-lowering.md`` §8.1: subsetting ``w`` alongside would be the first
    rewrite in this package to transform an *array* rather than reorder metadata, and that
    needs its own workstream).

    A *projection* does cross one, which is the asymmetry ``weight_dims`` pays
    for: a projection discards variables rather than subsetting them, so the weights need
    no rewriting, but it can orphan a dim **coordinate** — and whether a coord is present
    decides whether the weights align against that dim or broadcast a fresh one, so the
    rule has to require every weight dim survive. See ``optimize.dim_effect``.

    The dim algebra, verified against xarray 2026.7.0 — and **not** a plain ``Reduce``'s,
    which is what ``weight_dims`` is for. ``consumes`` is exactly a reduce's, symbolic
    :data:`ALL_DIMS` for a bare closer (``DatasetWeighted.mean`` really does take ``dim``,
    unlike ``DatasetRolling.mean``). But the *weights* have a dim effect of their own,
    because ``dot`` broadcasts and aligns:

    - a weight dim the dataset **lacks** is broadcast in, onto *every* variable —
      ``w(lat, member)`` makes ``ds.weighted(w).mean("lat")`` return ``{time, lon, member}``
      and ``elevation(lat, lon)`` come back as ``(lon, member)``;
    - a weight dim the dataset **shares** is aligned, and a misaligned one *shrinks* it —
      ``w`` on ``lat`` labelled ``[0, 1]`` against a ``lat`` of ``[0, 1, 2]`` inner-joins to
      size 2.

    So a surviving weight dim is a dim whose *name* is known and whose *extent* is not:
    :class:`~xrexpr.schema.SchemaState` records its size as ``None``, which is exactly what
    unknown sizes exist for. A bare closer clears everything, weight dims included (also
    verified).

    Only the dim **names** are kept, not the weights' sizes: sizes would be a guess about
    the post-alignment extent, and the names are what every rule reasons about. The weights
    themselves stay in the verbatim payload, which is why this node inherits the
    array-payload hashability caveat in the module docstring.
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


#: The optimiser's IR node — a sum over the structural op *kinds*. ``match`` over this
#: binds different fields per arm; ``typing.assert_never`` on the ``case _`` arm makes
#: the union exhaustive (adding a variant fails type-check at every unhandled site).
Op = Reduce | Select | Scan | Elementwise | Project | Drop | Rename | Rechunk | Opaque

#: What the recorder produces — one node per call, as the fluent API spelled it —
#: including the half-operations (:class:`ContextOpen`) that only mean something paired.
FluentOp = Op | ContextOpen

#: What the optimiser rewrites — the same vocabulary, plus the nodes the fluent API cannot
#: express in a single call, and *minus* the opener, which lowering must have consumed.
#: The asymmetry is the point — a :class:`ContextOpen` that outlived ``to_lower_ir``
#: fails type-check at every site downstream of it.
LoweredOp = Op | GroupedReduce | WindowedReduce | WeightedReduce
