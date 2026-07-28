"""Render a lowered plan as text — what :meth:`~xrexpr.accessor.LazyDatasetProxy.explain` returns.

Formats the **lowered nodes** rather than the emitted calls, which is
``docs/roadmap/02-lowering.md`` §12's last item (PR 8) and what Polars' ``explain`` shows:
the plan the optimiser actually reasoned about. Three things the call-level form could not
say, each a question users were left to guess at:

- a fused node is **one** op, so ``groupby("time.month").mean()`` reads as the single
  semantic operation it is instead of two unrelated calls;
- an :class:`~xrexpr.ir.Opaque` node is *visible*, so "why did nothing move?" has an answer
  in the output rather than requiring knowledge of which methods are modelled;
- the derived facts the rules key on (a bare reduce consuming *every* dim, a grouped
  reduce minting a new one, a weighted reduce's weight dims) appear at all.

Nothing is lost in the switch, because each node is still rendered as the calls it emits,
via :func:`~xrexpr.lower.emit` itself — so the line remains a faithful answer to "what will
run", and cannot drift from the replay.

Pure text, no dataset and no accessor: :func:`format_plan` is a function of the plan, which
is what lets the whole format be tested without recording anything.
"""

from collections.abc import Hashable, Iterable

from typing_extensions import assert_never

from xrexpr.ir import (
    AllDims,
    DimSet,
    GroupedReduce,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.lower import Call, emit


def format_plan(nodes: list[LoweredOp]) -> str:
    """The plan as a numbered, multi-line listing — one line per lowered node.

    Each line is ``N. Kind  <the calls this node replays as>``, plus ``[annotations]``
    where the node knows something the calls do not say.
    """
    if not nodes:
        return "plan (0 ops)"
    body = "\n".join(f"  {i}. {_format_node(n)}" for i, n in enumerate(nodes, 1))
    return f"plan ({len(nodes)} ops):\n{body}"


def _format_node(node: LoweredOp) -> str:
    """One line for one node: its kind, the calls it emits, and what those don't say."""
    calls = ".".join(_format_call(c) for c in emit([node]))
    annotation = _annotate(node)
    line = f"{type(node).__name__}  {calls}"
    return f"{line}  [{annotation}]" if annotation else line


def _annotate(node: LoweredOp) -> str:
    """The derived facts for ``node`` that its *emitted calls* do not already state.

    That rule is the whole editorial policy of this module, and it is what keeps the
    listing worth reading: ``rolling(time=3).mean()`` already tells you the window, so
    restating :attr:`~xrexpr.ir.WindowedReduce.window` would be noise, while a bare
    ``mean()`` says nothing about consuming every dim and a ``groupby("time.month")``
    says nothing about ``month`` being a *newly minted* dim replacing ``time``.

    ``assert_never`` closes the match, so a new lowered variant fails type-check here
    until someone decides what — if anything — it has to add. "Nothing" is a fine answer,
    and the answer most kinds take; taking it silently is what this prevents.
    """
    match node:
        case Reduce(consumes=consumes):
            # A bare ``.mean()`` reads as if it did nothing to the dims.
            return _dim_set("consumes", consumes)
        case GroupedReduce(group_dim=group_dim, new_dim=new_dim):
            # The surface call names neither: ``groupby("time.month")`` spells the *group*,
            # and that the result is indexed by a fresh ``month`` is exactly the fact that
            # makes a later ``isel(time=0)`` mean something different than it looks.
            return f"{group_dim} -> {new_dim}"
        case WeightedReduce(weight_dims=weight_dims, consumes=consumes):
            # The weights are elided to ``<DataArray>`` in the call (see ``_format_arg``),
            # so their dims -- which decide whether a projection may cross this node at
            # all -- would otherwise be invisible in both halves of the line.
            weights = f"weights over {_braced(weight_dims)}" if weight_dims else ""
            parts = [weights, _dim_set("consumes", consumes)]
            return ", ".join(p for p in parts if p)
        case Opaque():
            return "not modelled -- nothing crosses it"
        case Select() | Project() | Rechunk() | Scan() | WindowedReduce():
            return ""
        case _:
            assert_never(node)


def _dim_set(label: str, dims: DimSet) -> str:
    """``label={a, b}``, or ``""`` for an empty set so the caller can drop it."""
    if isinstance(dims, AllDims):
        return f"{label}=every dim"
    return f"{label}={_braced(dims)}" if dims else ""


def _braced(dims: Iterable[Hashable]) -> str:
    """Dim names in braces, sorted by ``str`` — sets have no order, output must."""
    return "{" + ", ".join(str(d) for d in sorted(dims, key=str)) + "}"


def _format_call(call: Call) -> str:
    """One-line human-readable form of an emitted call."""
    if call.name == "__getitem__":
        return f"[{_format_arg(call.args[0])}]"
    parts = [_format_arg(a) for a in call.args]
    parts += [f"{k}={_format_arg(v)}" for k, v in call.kwargs.items()]
    return f"{call.name}({', '.join(parts)})"


def _format_arg(arg: object) -> str:
    """``repr(arg)``, unless that repr is multi-line — then just its type.

    One line per node is the whole shape of this listing, and an argument is allowed to be
    an array: ``ds.weighted(w)`` holds a ``DataArray`` whose ``repr`` is a five-line block
    of dims, values and coords, which would swallow the plan around it. Its *dims* are the
    part worth seeing and :func:`_annotate` shows them, so the payload itself elides to
    ``<DataArray>``. Taking the ``repr`` to test it computes nothing: xarray reprs a
    dask-backed variable as its graph summary, never its values — pinned by
    ``tests/test_explain.py::test_a_dask_backed_argument_elides_without_computing``.
    """
    text = repr(arg)
    return f"<{type(arg).__name__}>" if "\n" in text else text
