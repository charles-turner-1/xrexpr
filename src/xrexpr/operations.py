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
    #: op family; one of ``"reduce"``/``"scan"``/``"select"``/``"rechunk"``.
    #: ``to_opnode`` maps this to the matching :data:`~xrexpr.ir.Op` variant
    #: (untabulated names → ``Opaque``).
    kind: str
    #: reduce: the given dim is removed. scan/select/rechunk: dim kept — selects resolve
    #: their actual dim removal from the *indexer* at record time, not from here.
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
    """Return the :class:`OpSpec` for ``name``, or ``None`` if it isn't tabulated."""
    return OP_TABLE.get(name)
