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

Only **unary** ops are modelled here. Binary/n-ary ops (``merge``/``concat``/``where``)
would add their own variants carrying plan-typed children and promote the container from
a list to a tree — an additive, orthogonal change deferred until such an op is in scope.
Keep that linearity assumption named *here*, not leaked into individual rules.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from frozendict import frozendict

__all__ = ["Op", "Opaque", "Project", "Rechunk", "Reduce", "Scan", "Select", "frozendict"]


def _is_scalar_index(value: Any) -> bool:
    """Whether an indexer drops its dim (a scalar) rather than keeping it.

    A ``slice``/``list``/``tuple``/array keeps the dim (and resizes it); anything
    else is treated as a scalar that removes it. (A tuple MultiIndex *label* is the
    known exception — niche, and left for later.)
    """
    return not isinstance(value, slice | list | tuple | np.ndarray)


@dataclass(frozen=True)
class Reduce:
    """A dimension-destroying reduction (``mean``/``sum``/``std``/...).

    ``consumes`` — the dims the reduction removes — is *stored*, resolved by
    ``to_opnode`` from the ``dim`` spec against the record-time schema (a bare
    ``mean()`` consumes *every* current dim). ``args``/``kwargs`` are coerced to
    immutable containers so the node is hashable and safe to share between plans.
    """

    name: str  # open set of tabulated reductions → str (kind-safety via OP_TABLE)
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    consumes: frozenset[Hashable] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "consumes", frozenset(self.consumes))


@dataclass(frozen=True)
class Select:
    """An ``isel``/``sel`` selection, described by its ``{dim: indexer}`` mapping.

    ``consumes`` is a *derived* view of ``indexer`` (the scalar-indexed dims, which
    drop) — a ``@property``, never a stored field — so a merged select cannot disagree
    with itself the way a separately-accumulated ``consumes`` could.
    """

    name: Literal["isel", "sel"]  # closed set → Literal (rejects Select(name="mean"))
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    indexer: frozendict[Hashable, Any] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "kwargs", frozendict(self.kwargs))
        object.__setattr__(self, "indexer", frozendict(self.indexer))

    @property
    def consumes(self) -> frozenset[Hashable]:
        """Dims this select drops: the scalar-indexed ones (slices/sequences keep theirs)."""
        return frozenset(d for d, v in self.indexer.items() if _is_scalar_index(v))


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
