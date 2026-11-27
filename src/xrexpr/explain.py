"""Render a lowered plan as text, for :meth:`~xrexpr.accessor.LazyProxy.explain`.

It prints the **lowered nodes**, not the emitted calls, as Polars' ``explain`` does: it
shows the plan the optimiser actually reasoned about. That shows three things a plain
list of calls wouldn't:

- a fused node is **one** op, so ``groupby("time.month").mean()`` reads as one operation
  rather than two unrelated calls;
- an :class:`~xrexpr.ir.Opaque` node is *visible*, so "why did nothing move?" has an
  answer in the output, without needing to know which methods are modelled;
- the derived facts rules depend on are shown (a bare reduce consuming *every* dim, a
  grouped reduce creating a new dim, a weighted reduce's weight dims).

Each node is still rendered as the calls it emits, via :func:`~xrexpr.lower.emit`
itself. So each line still answers "what will run", and can't drift from replay.

Pure text, with no dataset and no accessor: :func:`format_plan` is a function of the
plan, so the whole format can be tested without recording anything.
"""

from collections.abc import Hashable, Iterable

from typing_extensions import assert_never

from xrexpr._xrexprs.ir import AllDims
from xrexpr.ir import (
    DimSet,
    Drop,
    Elementwise,
    GroupedReduce,
    LoweredOp,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Rename,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)
from xrexpr.lower import Call, emit


def format_plan(nodes: list[LoweredOp]) -> str:
    """Render the plan as a numbered listing, one line per lowered node.

    Parameters
    ----------
    nodes : list of LoweredOp
        The plan to render, after lowering and optimisation.

    Returns
    -------
    str
        The listing. Each line is ``N. Kind  <the calls this node replays as>``, plus
        ``[annotations]`` where the node knows something the calls don't show. An empty
        plan renders as ``"plan (0 ops)"``.
    """
    if not nodes:
        return "plan (0 ops)"
    body = "\n".join(f"  {i}. {_format_node(n)}" for i, n in enumerate(nodes, 1))
    return f"plan ({len(nodes)} ops):\n{body}"


def _format_node(node: LoweredOp) -> str:
    """Render one node: its kind, the calls it emits, and anything those don't show.

    Parameters
    ----------
    node : LoweredOp
        The node to render.

    Returns
    -------
    str
        The line, without its number.
    """
    calls = ".".join(_format_call(c) for c in emit([node]))
    annotation = _annotate(node)
    line = f"{type(node).__name__}  {calls}"
    return f"{line}  [{annotation}]" if annotation else line


def _annotate(node: LoweredOp) -> str:
    """Render the derived facts about ``node`` that its *emitted calls* don't already show.

    Parameters
    ----------
    node : LoweredOp
        The node to annotate.

    Returns
    -------
    str
        The annotation, or ``""`` when the calls already say everything (most kinds).

    Notes
    -----
    That's the whole rule for what gets annotated, and it keeps the listing readable.
    ``rolling(time=3).mean()`` already shows the window, so repeating
    :attr:`~xrexpr.ir.WindowedReduce.window` would be noise. But a bare ``mean()``
    doesn't say it consumes every dim, and ``groupby("time.month")`` doesn't say
    ``month`` is a *new* dim replacing ``time``.

    ``assert_never`` closes the match, so a new lowered kind fails type-checking here
    until someone decides what, if anything, it should add. "Nothing" is a fine answer,
    and most kinds give it; the point is that it's chosen, not defaulted.
    """
    match node:
        case Reduce(consumes=consumes) as reduce if reduce.keepdims:
            # ``keepdims=True`` keeps its named dims at size 1 rather than removing
            # them, so the "consumes" wording below would be misleading.
            if isinstance(consumes, AllDims):
                return "keeps every dim at size 1"
            return f"keeps {_braced(consumes)} at size 1" if consumes else ""
        case Reduce(consumes=consumes):
            # A bare ``.mean()`` reads as if it did nothing to the dims.
            return _dim_set("consumes", consumes)
        case GroupedReduce(group_dim=group_dim, new_dim=new_dim):
            # The call shows neither. ``groupby("time.month")`` names the *grouping*,
            # and the fact that the result is indexed by a new ``month`` dim is what
            # makes a later ``isel(time=0)`` mean something different from how it looks.
            return f"{group_dim} -> {new_dim}"
        case WeightedReduce(weight_dims=weight_dims, consumes=consumes):
            # The weights are shown as ``<DataArray>`` in the call (see
            # ``_format_arg``), so their dims, which decide whether a projection may
            # cross this node, would otherwise not appear anywhere on the line.
            weights = f"weights over {_braced(weight_dims)}" if weight_dims else ""
            parts = [weights, _dim_set("consumes", consumes)]
            return ", ".join(p for p in parts if p)
        case Opaque():
            return "not modelled -- nothing crosses it"
        case (
            Select()
            | Project()
            | Drop()
            | Rename()
            | Rechunk()
            | Scan()
            | Elementwise()
            | WindowedReduce()
        ):
            # ``drop_vars(["region"])`` and ``rename({"month": "time"})`` already say
            # what they do, so an annotation would repeat the call. Same for ``Project``
            # and ``Select``.
            return ""
        case _:
            assert_never(node)


def _dim_set(label: str, dims: DimSet) -> str:
    """Render a labelled dim set.

    Parameters
    ----------
    label : str
        The label to prefix, e.g. ``"consumes"``.
    dims : DimSet
        The dims to render.

    Returns
    -------
    str
        ``label={a, b}``, ``label=every dim`` for :data:`~xrexpr.ir.ALL_DIMS`, or ``""``
        for an empty set so the caller can leave it out.
    """
    if isinstance(dims, AllDims):
        return f"{label}=every dim"
    return f"{label}={_braced(dims)}" if dims else ""


def _braced(dims: Iterable[Hashable]) -> str:
    """Render dim names in braces.

    Parameters
    ----------
    dims : iterable of Hashable
        The dim names.

    Returns
    -------
    str
        The names in braces, sorted by ``str`` (sets have no order, but the output
        must).
    """
    return "{" + ", ".join(str(d) for d in sorted(dims, key=str)) + "}"


def _format_call(call: Call) -> str:
    """Render an emitted call on one line.

    Parameters
    ----------
    call : Call
        The call to render.

    Returns
    -------
    str
        ``name(args, kw=...)``, or ``[key]`` for a ``__getitem__``.
    """
    if call.name == "__getitem__":
        return f"[{_format_arg(call.args[0])}]"
    parts = [_format_arg(a) for a in call.args]
    parts += [f"{k}={_format_arg(v)}" for k, v in call.kwargs.items()]
    return f"{call.name}({', '.join(parts)})"


def _format_arg(arg: object) -> str:
    """Render one call argument, shortening anything whose ``repr`` spans several lines.

    Parameters
    ----------
    arg : object
        The argument to render.

    Returns
    -------
    str
        ``repr(arg)``, unless that repr is multi-line, in which case ``<TypeName>``.

    Notes
    -----
    The listing is one line per node, and an argument can be an array.
    ``ds.weighted(w)`` holds a ``DataArray`` whose ``repr`` is a five-line block of
    dims, values and coords, which would swamp the plan. The useful part is its dims,
    which ``_annotate`` shows, so the array itself is shortened to ``<DataArray>``.

    Taking the ``repr`` to check computes nothing: xarray shows a dask-backed variable
    as a graph summary, never its values. Checked by
    ``tests/test_explain.py::test_a_dask_backed_argument_elides_without_computing``.
    """
    text = repr(arg)
    return f"<{type(arg).__name__}>" if "\n" in text else text
