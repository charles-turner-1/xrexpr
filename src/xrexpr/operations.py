"""Static facts about the xarray methods the optimiser understands.

Every method the recorder can classify has one row in :data:`OP_TABLE`. Each row is an
:data:`OpSpec`, a union with one variant per kind of node ``to_opnode`` can build.

The most important distinction the table draws is between :class:`ReduceSpec`
(``mean``/``sum``/``std``), which **removes** its dim, and :class:`ScanSpec`
(``cumsum``/``cumprod``/``diff``), which **keeps** it. Treating the two as one would let
a rule reorder across a scan as though it were a reduction.

A spec's *type* is its kind; there's no ``kind: str`` field. That's what makes
``to_opnode``'s dispatch a single ``match`` closed by ``assert_never``: a variant added
here without an arm there is a type error, rather than a call silently recorded as
:class:`~xrexpr.ir.Opaque`. Each variant carries its own ``name``, typed as the
``Literal`` its node expects, and :data:`OP_TABLE` is built from those names, so a row
can't be filed under the wrong key.

:class:`ProjectSpec` is the one row the table can only *nominate*: ``__getitem__`` is a
projection when its key names variables, and an :class:`~xrexpr.ir.Opaque` otherwise.

See ``docs/internals/operations.md``.
"""

from dataclasses import dataclass
from typing import Literal, get_args

from xrexpr.ir import ContextOpenName

__all__ = [
    "CHUNK_OPTION_KWARGS",
    "CONTEXT_METHODS",
    "OP_TABLE",
    "SELECT_OPTION_KWARGS",
    "ContextSpec",
    "DropSpec",
    "ElementwiseSpec",
    "OpSpec",
    "ProjectSpec",
    "RechunkSpec",
    "ReduceSpec",
    "RenameSpec",
    "ScanSpec",
    "SelectSpec",
    "spec",
]


@dataclass(frozen=True)
class ReduceSpec:
    """A reduction that removes dims: ``mean``/``sum``/``std``/... .

    Attributes
    ----------
    name : str
        The method name. An open set, so ``str``, matching :class:`~xrexpr.ir.Reduce`.
    dim_arg : int
        Where this reduction's dim spec sits among its **positional** arguments.

    Notes
    -----
    ``dim_arg`` defaults to 0 because reductions take ``dim`` first. It's 1 for
    ``Dataset.reduce``, which is ``reduce(func, dim, ...)``; reading its first
    positional as the dim would record the *function* as the dim of ``ds.reduce(np.mean,
    "time")``. It's a field on the row rather than a separate name-keyed table, so an
    odd signature stays with the method that has it.
    """

    name: str  # open set of reductions → str, as on ``ir.Reduce``
    dim_arg: int = 0


@dataclass(frozen=True)
class ScanSpec:
    """An order-sensitive scan (``cumsum``/``cumprod``/``diff``) that *keeps* its dim.

    Attributes
    ----------
    name : {"cumsum", "cumprod", "diff"}
        Which scan this is. A closed set, so a ``Literal``.
    """

    name: Literal["cumsum", "cumprod", "diff"]


@dataclass(frozen=True)
class ElementwiseSpec:
    """A per-element op (``fillna``/``astype``/``round``/...) that keeps every dim.

    Attributes
    ----------
    name : str
        The method name. An open set, so ``str``, matching
        :class:`~xrexpr.ir.Elementwise`.

    Notes
    -----
    Like :class:`ProjectSpec`, this only *nominates*. ``fillna(0)`` is an
    :class:`~xrexpr.ir.Elementwise`, but ``fillna(other_da)`` and ``fillna({"tas": 0})``
    pass an array or per-variable argument that doesn't commute with a projection, and
    are recorded as :class:`~xrexpr.ir.Opaque`. The argument's shape decides, at record
    time, in ``to_opnode``'s guarded arm (``schema._elementwise_safe``).

    Deliberately conservative: only ops that are per-element for *every* plain-valued
    argument are listed. ``where`` (its ``cond`` is data), ``interp``/``interpolate_na``
    (dim-aware) and ``map`` (an arbitrary function) are left out. Adding an op is a
    one-line change once it's shown to be safe.
    """

    name: str  # open set of elementwise ops → str, as on ``ir.Elementwise``


@dataclass(frozen=True)
class SelectSpec:
    """An ``isel``/``sel`` selection.

    Attributes
    ----------
    name : {"isel", "sel"}
        Which selection this is. A closed set, so a ``Literal``.

    Notes
    -----
    The table says nothing about which dims a select removes, because that depends on
    the *indexer* (a scalar drops the dim, a slice keeps it). ``to_opnode`` works that
    out per call, into :attr:`~xrexpr.ir.Select.indexer`.
    """

    name: Literal["isel", "sel"]


@dataclass(frozen=True)
class RechunkSpec:
    """A ``chunk`` call. Changes how data is split into blocks, never a dim, size or value.

    Attributes
    ----------
    name : {"chunk"}
        Always ``"chunk"``. A closed set, so a ``Literal``.
    """

    name: Literal["chunk"]


@dataclass(frozen=True)
class ProjectSpec:
    """A ``__getitem__`` call, which is a projection only if its key names variables.

    Attributes
    ----------
    name : {"__getitem__"}
        Always ``"__getitem__"``. A closed set, so a ``Literal``.

    Notes
    -----
    **The one row that can only nominate.** Every other spec names the node kind its
    call *is*; this one names the kind its call *may be*. ``ds["tas"]`` is a
    :class:`~xrexpr.ir.Project`, but ``ds[mask]`` and ``ds[{"time": 0}]`` are the same
    method with the same spec and become :class:`~xrexpr.ir.Opaque`. The key's shape
    decides, at record time, in ``to_opnode``'s guarded ``match`` arm.

    It's in the table anyway, rather than special-cased after the dispatch, so every
    method the recorder can classify is classified in one place. The guard then sits in
    the same ``match`` as the arms it's an exception to.
    """

    name: Literal["__getitem__"] = "__getitem__"


@dataclass(frozen=True)
class RenameSpec:
    """A ``rename`` call: relabels dims, variables or coordinates, changing no values.

    Attributes
    ----------
    name : {"rename"}
        Always ``"rename"``. A closed set, so a ``Literal``.
    """

    name: Literal["rename"] = "rename"


@dataclass(frozen=True)
class DropSpec:
    """A ``drop_vars`` call: removes named variables or coordinates, changing no dim or value.

    Attributes
    ----------
    name : {"drop_vars"}
        Always ``"drop_vars"``. A closed set, so a ``Literal``.
    """

    name: Literal["drop_vars"] = "drop_vars"


@dataclass(frozen=True)
class ContextSpec:
    """A call that returns a builder: ``groupby``/``rolling``/``weighted``/... .

    Attributes
    ----------
    name : ContextOpenName
        Which builder the call opens. A closed set, so a ``Literal``.

    Notes
    -----
    Half of an operation. These return a ``DatasetGroupBy``/``DatasetRolling``/...
    rather than a Dataset, so ``to_opnode`` records a :class:`~xrexpr.ir.ContextOpen`,
    which only says "a context opens here". What the **pair** means is decided by
    ``lower.to_lower_ir``, which can see the other half.

    A test checks every name against ``xr.Dataset``, so a typo or a method xarray
    removes fails a test instead of silently no longer opening a context.
    """

    name: ContextOpenName


#: What :data:`OP_TABLE` records about one xarray method: which :data:`~xrexpr.ir.Op`
#: kind the call is, plus anything that kind needs to know. ``to_opnode`` matches on
#: this union and closes it with ``assert_never``. A name with no row is
#: :class:`~xrexpr.ir.Opaque`.
OpSpec = (
    ReduceSpec
    | ScanSpec
    | ElementwiseSpec
    | SelectSpec
    | RechunkSpec
    | ProjectSpec
    | RenameSpec
    | DropSpec
    | ContextSpec
)


_REDUCTIONS = (
    "count",
    "all",
    "any",
    "max",
    "min",
    "mean",
    "prod",
    "sum",
    "std",
    "var",
    "median",
)
# Built from the ``Literal`` that types them, so the openers are written once, not once
# for dispatch and once for the type checker. Every element of ``get_args`` on a
# ``Literal`` is one of its members, so this is correct by construction (mypy sees
# ``tuple[Any, ...]`` either way).
_CONTEXTS: tuple[ContextOpenName, ...] = get_args(ContextOpenName)

# Conservative on purpose (see ``ElementwiseSpec``): each of these is per-element for
# any plain-valued argument, so it keeps every dim, size and variable. Typed ``str``
# rather than a ``Literal`` because the set is expected to grow, and
# ``ir.Elementwise.name`` is ``str`` to match.
_ELEMENTWISE = ("fillna", "astype", "round", "clip", "isnull", "notnull")

_SPECS: tuple[OpSpec, ...] = (
    *(ReduceSpec(name) for name in _REDUCTIONS),
    ReduceSpec("reduce", dim_arg=1),  # ``reduce(func, dim, ...)`` — see ReduceSpec
    ScanSpec("cumsum"),
    ScanSpec("cumprod"),
    ScanSpec("diff"),
    *(ElementwiseSpec(name) for name in _ELEMENTWISE),
    SelectSpec("isel"),
    SelectSpec("sel"),
    RechunkSpec("chunk"),
    ProjectSpec(),
    RenameSpec(),
    DropSpec(),
    *(ContextSpec(name) for name in _CONTEXTS),
)

#: Every method the recorder can classify, keyed by name. Built from ``_SPECS`` rather
#: than written out as a dict, so a row can't be filed under a key that disagrees with
#: its own ``name``.
OP_TABLE: dict[str, OpSpec] = {op.name: op for op in _SPECS}

#: The builder-returning methods, as a set. Built from the table, so it can't drift from
#: what ``to_opnode`` dispatches on; the rows are built from ``ir.ContextOpenName``, so
#: it can't drift from the type either. A test checks the names against ``xr.Dataset``,
#: which is the one thing no derivation can guarantee.
CONTEXT_METHODS = frozenset(op.name for op in _SPECS if isinstance(op, ContextSpec))

#: ``isel``/``sel`` keyword arguments that are *options*, not dim indexers.
#: ``schema._select_indexer`` skips them when reading the indexer from a call, and
#: ``lower._emit_node`` copies exactly these when rebuilding the call from
#: :attr:`~xrexpr.ir.Select.indexer`. Defined once so the two can't disagree.
SELECT_OPTION_KWARGS = frozenset({"drop", "missing_dims", "method", "tolerance"})

#: ``chunk`` keyword arguments that are *options*, not per-dim chunk specs. Shared by
#: ``schema._chunk_spec`` (parsing) and ``lower._emit_node`` (rebuilding), like
#: :data:`SELECT_OPTION_KWARGS`.
CHUNK_OPTION_KWARGS = frozenset(
    {
        "name_prefix",
        "token",
        "lock",
        "inline_array",
        "chunked_array_type",
        "from_array_kwargs",
    }
)


def spec(name: str) -> OpSpec | None:
    """Look up the :data:`OpSpec` for a method name.

    Parameters
    ----------
    name : str
        The xarray method name as recorded, e.g. ``"mean"`` or ``"isel"``.

    Returns
    -------
    OpSpec or None
        The spec, or ``None`` if ``name`` isn't in :data:`OP_TABLE`. A ``None`` makes
        :func:`~xrexpr.schema.to_opnode` record :class:`~xrexpr.ir.Opaque`.
    """
    return OP_TABLE.get(name)
