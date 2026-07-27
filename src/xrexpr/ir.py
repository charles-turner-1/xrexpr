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
:class:`Project` when its key names variables and an :class:`Opaque` otherwise, so the
table can't decide it — the shape of the *key* does.

``to_opnode`` (in ``schema.py``) builds these at record time; the optimiser
(``optimize.py``) rewrites the list and the ``.plan`` accessor replays it.

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

Only **unary** ops are modelled here. Binary/n-ary ops (``merge``/``concat``/``where``)
would add their own variants carrying plan-typed children and promote the container from
a list to a tree — an additive, orthogonal change deferred until such an op is in scope.
Keep that linearity assumption named *here*, not leaked into individual rules.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Final, Literal, final

from frozendict import frozendict

from xrexpr.indexers import Indexer, classify

__all__ = [
    "ALL_DIMS",
    "AllDims",
    "DimSet",
    "Op",
    "Opaque",
    "Project",
    "Rechunk",
    "Reduce",
    "Scan",
    "Select",
    "frozendict",
]


@final
@dataclass(frozen=True)
class AllDims:
    """Sentinel: *every dim present at this point*, whatever they turn out to be.

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

    ``consumes`` — the dims the reduction removes — is *stored*, parsed by ``to_opnode``
    from the ``dim`` spec. A bare ``mean()`` names no dim and so consumes
    :data:`ALL_DIMS`, left symbolic rather than expanded against the record-time schema
    (see the module docstring). ``args``/``kwargs`` are coerced to immutable containers
    so the node is hashable and safe to share between plans.
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
        """Dims this select drops: the scalar-indexed ones (slices/sequences keep theirs)."""
        return frozenset(d for d, v in self.indexer.items() if v.drops_dim)


@dataclass(frozen=True)
class Scan:
    """An order-significant scan (``cumsum``/``cumprod``/``diff``) — *keeps* its dim.

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

    The one op recognised by the *shape of its key* rather than by a method name:
    ``__getitem__`` isn't in ``OP_TABLE`` because the same call is a projection only
    when its key names variables (a boolean-mask key stays :class:`Opaque`).

    ``variables`` is the requested names in order. ``single`` — whether the call
    returns a ``DataArray`` (a bare name) rather than a ``Dataset`` (a list of them)
    — is *derived* from the verbatim key, never stored, so it cannot disagree with
    what replay will actually do. The list/hashable split mirrors xarray's own
    ``Dataset.__getitem__``, so a tuple key reads as one name rather than several.
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
        """Whether this projects *one* variable to a ``DataArray`` (``ds["tas"]``)."""
        return bool(self.args) and not isinstance(self.args[0], list)


@dataclass(frozen=True)
class Rechunk:
    """A ``chunk`` call: changes chunk topology only — never a dim, size or value.

    ``chunks`` holds the **mapping-form** ``{dim: spec}`` only (a positional dict and/or
    dim kwargs). It stays empty for the uniform forms — ``chunk()``, ``chunk(100)``,
    ``chunk("auto")`` — whose spec names no dim and so lives verbatim in ``args``. That
    split is what a rewrite needs: only *named* dims have to be stripped from the spec
    when a select drops them, and only a named-dim spec can be emptied out entirely.

    Whether a given rechunk may be *crossed* is deliberately not decided here — that
    judgement lives with the rule (``_pushable_rechunk`` in ``optimize.py``), as it does
    for selects.
    """

    name: Literal["chunk"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    chunks: frozendict[Hashable, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "chunks", frozendict(self.chunks))


@dataclass(frozen=True)
class Opaque:
    """Any op the optimiser doesn't model — replayed verbatim, never reordered."""

    name: str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))


#: The optimiser's IR node: a sum over the structural op *kinds*. ``match`` over this
#: binds different fields per arm; ``typing.assert_never`` on the ``case _`` arm makes
#: the union exhaustive (adding a variant fails type-check at every unhandled site).
Op = Reduce | Select | Scan | Project | Rechunk | Opaque
