"""Static metadata about the xarray operations the optimiser reasons about.

Every method the recorder can classify is one row in :data:`OP_TABLE`, and each row is an
:data:`OpSpec` — a sum type over *dispatch kinds*, one variant per :data:`~xrexpr.ir.Op`
variant ``to_opnode`` can build. The distinction the table exists to draw is
:class:`ReduceSpec` (``mean``/``sum``/``std``), which **destroys** its dim, against
:class:`ScanSpec` (``cumsum``/``cumprod``/``diff``), which **keeps** it: lumping the two
together is what lets a scan be reordered as though it were a reduction.

A spec *is* the kind rather than carrying a ``kind: str`` field, which is what makes
``to_opnode``'s dispatch a single ``match`` closed by ``assert_never`` — a variant added
here without an arm there is a type error, not a call silently recorded as
:class:`~xrexpr.ir.Opaque`. Each variant carries its own ``name``, typed as the ``Literal``
its ``Op`` counterpart demands, and :data:`OP_TABLE` is **derived** from those names, so a
row cannot be filed under a key that disagrees with it.

:class:`ProjectSpec` is the one row the table can only *nominate*: ``__getitem__`` is a
projection when its key names variables and an :class:`~xrexpr.ir.Opaque` otherwise.

See ``docs/internals/operations.md``.
"""

from dataclasses import dataclass
from typing import Literal, get_args

from xrexpr.ir import ContextOpenName

__all__ = [
    "CONTEXT_METHODS",
    "OP_TABLE",
    "ContextSpec",
    "DropSpec",
    "ElementwiseSpec",
    "OpSpec",
    "ProjectSpec",
    "RechunkSpec",
    "ReduceSpec",
    "ScanSpec",
    "SelectSpec",
    "spec",
]


@dataclass(frozen=True)
class ReduceSpec:
    """A dimension-destroying reduction — ``mean``/``sum``/``std``/... .

    Attributes
    ----------
    name : str
        The method name. An open set of reductions, so ``str``, matching
        :class:`~xrexpr.ir.Reduce`.
    dim_arg : int
        Where this reduction's dim spec sits among its **positional** arguments.

    Notes
    -----
    ``dim_arg`` defaults to 0 because reductions take ``dim`` first, and is 1 for
    ``Dataset.reduce``, which is ``reduce(func, dim, ...)`` — reading its first
    positional as a dim spec is what made ``ds.reduce(np.mean, "time")`` record a
    nonsense ``consumes`` (#96). A field rather than a lookup in a second name-keyed
    table, so a signature this odd travels with the row that has it.
    """

    name: str  # open set of reductions → str, as on ``ir.Reduce``
    dim_arg: int = 0


@dataclass(frozen=True)
class ScanSpec:
    """An order-significant scan — ``cumsum``/``cumprod``/``diff`` — which *keeps* its dim.

    Attributes
    ----------
    name : {"cumsum", "cumprod", "diff"}
        Which scan this is. A closed set, so a ``Literal``.
    """

    name: Literal["cumsum", "cumprod", "diff"]


@dataclass(frozen=True)
class ElementwiseSpec:
    """A per-element op — ``fillna``/``astype``/``round``/... — which keeps every dim.

    Attributes
    ----------
    name : str
        The method name. An open set of elementwise ops, so ``str``, matching
        :class:`~xrexpr.ir.Elementwise`.

    Notes
    -----
    The table nominates these the way :class:`ProjectSpec` nominates ``__getitem__``:
    ``fillna(0)`` is an :class:`~xrexpr.ir.Elementwise`, but ``fillna(other_da)`` and
    ``fillna({"tas": 0})`` are the same method with a data- or per-variable-shaped
    argument that does *not* commute with a projection, and record :class:`~xrexpr.ir.Opaque`.
    The shape of the argument decides, at record time, in ``to_opnode``'s guarded arm
    (``schema._elementwise_safe``).

    Deliberately conservative: only ops that are per-element for *every* plain-valued
    argument are tabulated. ``where`` (its ``cond`` is data), ``interp``/``interpolate_na``
    (dim-aware) and ``map`` (an arbitrary callable) are excluded; growing the set is a
    one-line change once a case is shown to hold.
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
    The table claims no dim effect for a select, because there is none to claim: whether
    a dim is removed depends on the *indexer* — a scalar drops it, a slice keeps it — and
    ``to_opnode`` resolves that per call, into :attr:`~xrexpr.ir.Select.indexer`.
    """

    name: Literal["isel", "sel"]


@dataclass(frozen=True)
class RechunkSpec:
    """A ``chunk`` call: changes chunk topology only — never a dim, size or value.

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
    **The one row the table can only nominate.** Every other spec names the
    :data:`~xrexpr.ir.Op` variant its call *is*; this one names the variant its call
    *may be*. ``ds["tas"]`` is a :class:`~xrexpr.ir.Project`, but ``ds[mask]`` and
    ``ds[{"time": 0}]`` are the same method with the same spec and are
    :class:`~xrexpr.ir.Opaque` — the shape of the *key* decides, at record time, in
    ``to_opnode``'s guarded match arm.

    Tabulated anyway, rather than special-cased after the dispatch, so that every method
    the recorder can classify is classified in one place: the nomination is then visible
    in the type, and the guard that confirms it sits in the ``match`` beside the arms it
    is an exception to.
    """

    name: Literal["__getitem__"] = "__getitem__"


@dataclass(frozen=True)
class DropSpec:
    """A ``drop_vars`` call: removes named variables/coordinates, changing no dim or value.

    Attributes
    ----------
    name : {"drop_vars"}
        Always ``"drop_vars"``. A closed set, so a ``Literal``.
    """

    name: Literal["drop_vars"] = "drop_vars"


@dataclass(frozen=True)
class ContextSpec:
    """A builder-returning call — ``groupby``/``rolling``/``weighted``/... .

    Attributes
    ----------
    name : ContextOpenName
        Which builder the call opens. A closed set, so a ``Literal``.

    Notes
    -----
    Half of an operation: these return a ``DatasetGroupBy``/``DatasetRolling``/... rather
    than a Dataset, so ``to_opnode`` mints a :class:`~xrexpr.ir.ContextOpen` — *a context
    opens here*, which the call alone settles — and what the **pair** means is settled by
    ``lower.to_lower_ir``, which can see the other half.

    Every name is checked against ``xr.Dataset`` by the test suite, so a typo or a method
    xarray removes fails a test rather than silently ceasing to open a context.
    """

    name: ContextOpenName


#: What :data:`OP_TABLE` records about one tabulated xarray method — which
#: :data:`~xrexpr.ir.Op` variant the call is, plus whatever that kind needs to know.
#: ``to_opnode`` matches over this union and closes it with ``assert_never``; a name with
#: no row at all is :class:`~xrexpr.ir.Opaque`.
OpSpec = (
    ReduceSpec
    | ScanSpec
    | ElementwiseSpec
    | SelectSpec
    | RechunkSpec
    | ProjectSpec
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
# Derived from the ``Literal`` that types them, so the openers are written once rather
# than once for the dispatch and once for the type checker. Sound by construction — every
# element of ``get_args`` on a ``Literal`` is one of its members — which is all mypy can
# do with it either way, ``get_args`` being ``tuple[Any, ...]``.
_CONTEXTS: tuple[ContextOpenName, ...] = get_args(ContextOpenName)

# Conservative on purpose (see ``ElementwiseSpec``): every one of these is per-element for
# any plain-valued argument, so it keeps every dim, size and variable. ``str``-typed rather
# than a ``Literal`` because the set is expected to grow, and ``ir.Elementwise.name`` is
# ``str`` to match.
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
    DropSpec(),
    *(ContextSpec(name) for name in _CONTEXTS),
)

#: Every method the recorder can classify, keyed by name. **Derived** from ``_SPECS``
#: rather than written out as a mapping, so a row cannot be filed under a key that
#: disagrees with the name it carries.
OP_TABLE: dict[str, OpSpec] = {op.name: op for op in _SPECS}

#: The builder-returning methods, as a set. Derived from the table — they are rows in it
#: like any other method, so this cannot drift from what ``to_opnode`` actually dispatches
#: on; and the rows are themselves derived from ``ir.ContextOpenName``, so it cannot drift
#: from the type either. A test pins the names against ``xr.Dataset``, the one agreement
#: no derivation can supply.
CONTEXT_METHODS = frozenset(op.name for op in _SPECS if isinstance(op, ContextSpec))


def spec(name: str) -> OpSpec | None:
    """Look up the :data:`OpSpec` for a method name.

    Parameters
    ----------
    name : str
        The xarray method name as recorded, e.g. ``"mean"`` or ``"isel"``.

    Returns
    -------
    OpSpec or None
        The tabulated spec, or ``None`` if ``name`` is not in :data:`OP_TABLE` — which
        :func:`~xrexpr.schema.to_opnode` reads as :class:`~xrexpr.ir.Opaque`.
    """
    return OP_TABLE.get(name)
