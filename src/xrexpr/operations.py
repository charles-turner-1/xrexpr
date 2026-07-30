"""Static metadata about the xarray operations the optimiser reasons about.

Each tabulated op maps to an :class:`OpSpec` recording its *kind* and, for
reductions, whether it consumes (removes) the dim it is given. The key distinction is:

- ``reduce`` ops (``mean``/``sum``/``std``/...), which **destroy** their dim, from
- ``scan`` ops (``cumsum``/``cumprod``/``diff``), which **keep** it.

``rechunk`` (``chunk``) is a third case again: it touches no dim at all, only chunk
topology, which is what lets a selection cross it.

Lumping the two together is the root of the ``cumsum`` reordering bug called out in
the report (a scan must not be treated like a reduction). This table is the single
source of truth that drives ``to_opnode``.
"""

from typing import NamedTuple


class OpSpec(NamedTuple):
    """What :data:`OP_TABLE` records about one tabulated xarray method.

    Attributes
    ----------
    kind : str
        Op family; one of ``"reduce"``/``"scan"``/``"select"``/``"rechunk"``.
        :func:`~xrexpr.schema.to_opnode` maps this to the matching
        :data:`~xrexpr.ir.Op` variant (untabulated names → ``Opaque``).
    consumes_dim : bool
        Whether the given dim is removed. True for a reduce; false for
        scan/select/rechunk — a select resolves its actual dim removal from the
        *indexer* at record time, not from here.
    """

    kind: str
    consumes_dim: bool


_REDUCTIONS = (
    "reduce",
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
_SCANS = ("cumsum", "cumprod", "diff")
_SELECTS = ("sel", "isel")
_RECHUNKS = ("chunk",)

OP_TABLE: dict[str, OpSpec] = {
    **{name: OpSpec("reduce", True) for name in _REDUCTIONS},
    **{name: OpSpec("scan", False) for name in _SCANS},
    **{name: OpSpec("select", False) for name in _SELECTS},
    **{name: OpSpec("rechunk", False) for name in _RECHUNKS},
}


#: Methods returning a *builder* (``DatasetGroupBy``, ``DatasetRolling``,
#: ``DatasetWeighted``, ...) rather than a Dataset. Deliberately not in
#: :data:`OP_TABLE`: a table entry names the ``Op`` variant a call *is*, whereas these
#: are half of an operation — ``to_opnode`` mints a :class:`~xrexpr.ir.ContextOpen` and
#: what the pair means is settled by ``lower.to_lower_ir``, which can see the other half.
#:
#: Every name is checked against ``xr.Dataset`` by the test suite, so a typo or a
#: method xarray removes fails a test rather than silently ceasing to open a context.
CONTEXT_METHODS = frozenset(
    {
        "groupby",
        "groupby_bins",
        "resample",
        "rolling",
        "rolling_exp",
        "coarsen",
        "weighted",
        "cumulative",
    }
)


def spec(name: str) -> OpSpec | None:
    """Look up the :class:`OpSpec` for a method name.

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
