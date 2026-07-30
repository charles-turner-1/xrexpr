"""The exceptions the optimiser raises."""


class InvalidExpressionError(Exception):
    """A recorded plan that can never replay, diagnosed at optimisation time.

    Raised by :func:`~xrexpr.optimize.pushdown_selects` when a select indexes a dim a
    preceding reduce has already removed (``ds.plan.mean("lon").isel(lon=0)``). The
    chain is wrong however it is ordered, so the optimiser says so itself rather than
    emitting a reorder whose failure would be blamed on the rewrite.

    Notes
    -----
    Only the *provably* invalid case raises. Where the crossed node leaves the dim
    alive in some form — a minted dim, a windowed one — the plan is left un-reordered
    and xarray reports any error at replay, in its own words. See
    :attr:`~xrexpr.optimize.DimEffect.on_conflict`.
    """
