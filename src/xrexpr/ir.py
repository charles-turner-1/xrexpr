"""The expression IR: a sum type over the operation *kinds* the optimiser distinguishes.

A Dataset method chain is *linear* (each op has exactly one input — the previous
dataset), so the IR is a **list** of :data:`Op`, not a tree. Each variant carries the
verbatim call header (``name``/``args``/``kwargs``) that replay re-invokes, *plus* the
normalised metadata the optimiser reasons about — and that metadata differs per kind, so
the variants have genuinely different shapes rather than one flat record with mostly-empty
fields. ``match`` over :data:`Op` then binds different fields per arm, and ``assert_never``
makes the union exhaustive under mypy.

The union tracks structural **kinds**, not xarray methods: ``mean``/``std``/``sum`` are
all one :class:`Reduce`, told apart by ``name`` (the method table lives in
``operations.py``). A new *variant* is earned only by genuinely new structural data.
Kinds are usually settled by the method name, but not always: ``__getitem__`` is a
:class:`Project` when its key names variables and an :class:`Opaque` otherwise, so its
row in the table only *nominates* the kind — the shape of the *key* confirms it.

``to_opnode`` (in ``schema.py``) builds these at record time; ``lower.py`` translates
that recording into what it *means*, the optimiser (``optimize.py``) rewrites the
result, and ``lower.emit`` turns it back into the calls the ``.plan`` accessor replays.
:data:`FluentOp` and :data:`LoweredOp` name the two levels — one set of dataclasses,
two union aliases over them, so the level a function works at is in its signature.

**Dim sets are symbolic where the call is.** A bare ``ds.mean()`` names no dim: it means
"every dim there is *when this runs*", which is not knowable at record time — past an
unmodelled op the recorder's schema is a guess, so expanding it eagerly bakes in names
that may already be wrong (``rename`` is the case that bites). :data:`DimSet` therefore
admits the sentinel :data:`ALL_DIMS` alongside a concrete ``frozenset``, and the
expansion is deferred to a reader that has an exact schema. It is a *sentinel*, not
``None``: ``None`` already means **don't know** in this codebase
(``SchemaState.var_dims``), whereas ``ALL_DIMS`` means something definite. Readers
narrow it with a match arm rather than an ``assert``, so the two cases are handled where
the field is used.

**Immutability is unconditional; hashability is not.** Every variant is frozen and
coerces its containers, so a node cannot be mutated or drift from itself — and each is
hashable and comparable **when its payload is**. Some payloads aren't:
``xr.DataArray.__hash__`` is ``None``, so a node carrying an array (``weighted(w)``, a
boolean-mask ``__getitem__``, ``where(cond)``) raises ``TypeError`` on ``hash()``, and
``==`` between two such nodes holding *distinct but equal* arrays raises ``ValueError``
(the elementwise comparison has no truth value). The stronger claim is deliberately not
wanted: making an array payload hashable means hashing its *values*, which for a
dask-backed array means computing it at plan time — the one thing this package promises
never to do. (``indexers.Mask`` normalises to a tuple of booleans for a related reason,
but that payload is small and already realised, so the precedent does not transfer.)
Nothing here hashes a node, and plan-equality compares nodes that *share* the payload
object, where tuple comparison's identity check applies — see ``test_ir.py``, which pins
all three behaviours.

Only **unary** ops are modelled here. Binary/n-ary ops (``merge``/``concat``/``where``)
would add their own variants carrying plan-typed children and promote the container from
a list to a tree — an additive, orthogonal change deferred until such an op is in scope.
Keep that linearity assumption named *here*, not leaked into individual rules.
"""

from collections.abc import Hashable
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
    "FluentOp",
    "GroupedReduce",
    "LoweredOp",
    "Op",
    "Opaque",
    "Project",
    "Rechunk",
    "Reduce",
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

#: The dims an op removes: a concrete set, or :data:`ALL_DIMS` for a call that named
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
        The dims the reduction removes.

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
        """Dims this select drops: the scalar-indexed ones (slices/sequences keep theirs).

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

    Notes
    -----
    Distinct from a reduce (which destroys its dim) and from an opaque op (a scan is
    *known* to preserve dims, so a rule must not reorder across it); its scanned-dim
    metadata arrives with the first scan-aware rule.
    """

    name: Literal["cumsum", "cumprod", "diff"]  # closed set → Literal
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

    Notes
    -----
    ``chunks`` holds the **mapping-form** ``{dim: spec}`` only (a positional dict and/or
    dim kwargs). It stays empty for the uniform forms — ``chunk()``, ``chunk(100)``,
    ``chunk("auto")`` — whose spec names no dim and so lives verbatim in ``args``. That
    split is what a rewrite needs: only *named* dims have to be stripped from the spec
    when a select drops them, and only a named-dim spec can be emptied out entirely.

    Each value is a :data:`~xrexpr.chunks.ChunkSpec` — the closed value sum type the
    optimiser reasons about — normalised from its raw ``chunk`` form by ``__post_init__``
    (via :func:`~xrexpr.chunks.classify_chunk`), so a value is *always* a modelled variant
    regardless of whether the node was recorded or hand-built. Exactly what
    :class:`Select` does for its indexers, and for the same reason.

    Whether a given rechunk may be *crossed* is deliberately not decided here — that
    judgement lives with the rule (``_pushable_rechunk`` in ``optimize.py``), as it does
    for selects.
    """

    name: Literal["chunk"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    chunks: frozendict[Hashable, ChunkSpec] = field(default_factory=frozendict)

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
    ``"lat"`` as ``keep_attrs`` and reduces nothing. Only a closer that named no dim is
    fused (see :func:`~xrexpr.lower.to_lower_ir`), which is what keeps that trap out of
    this node: if a dim spec was parsed, it was never a dim spec.
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
    ``docs/roadmap/02-lowering.md`` §8.1: subsetting ``w`` alongside would be the first
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


#: The optimiser's IR node: a sum over the structural op *kinds*. ``match`` over this
#: binds different fields per arm; ``typing.assert_never`` on the ``case _`` arm makes
#: the union exhaustive (adding a variant fails type-check at every unhandled site).
Op = Reduce | Select | Scan | Project | Rechunk | Opaque

#: What the recorder produces: one node per call, as the fluent API spelled it —
#: including the half-operations (:class:`ContextOpen`) that only mean something paired.
FluentOp = Op | ContextOpen

#: What the optimiser rewrites: the same vocabulary, plus the nodes the fluent API cannot
#: express in a single call, and *minus* the opener, which lowering must have consumed.
#: The asymmetry is the point — a :class:`ContextOpen` that outlived ``to_lower_ir``
#: fails type-check at every site downstream of it.
LoweredOp = Op | GroupedReduce | WindowedReduce | WeightedReduce
