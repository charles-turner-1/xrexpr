"""The ``.plan`` accessor: a lazy recording proxy over an ``xr.Dataset`` or ``xr.DataArray``.

``ds.plan`` returns a :class:`LazyProxy` that records the chained method
calls made on it (``.mean``, ``.isel``, ``.sel``, ``__getitem__``, ...) instead
of executing them. Calling :meth:`~LazyProxy.collect` optimises the
recorded plan and replays it onto the real dataset (:meth:`~LazyProxy.explain`
returns the optimised plan as text without running it).

The recorded plan is a list of :data:`~xrexpr.ir.FluentOp` variants: each call is
normalised by :func:`~xrexpr.schema.to_opnode`, a pure function of that call, and
appended — recording holds no schema of its own. ``collect`` then runs the plan through
the pipeline in ``lower.py``: :func:`~xrexpr.lower.to_lower_ir` translates what was
written into what it means, :func:`~xrexpr.optimize.optimize` rewrites that (a fixpoint
of rules, folding the base schema forward itself), :func:`~xrexpr.lower.emit` turns the
result back into calls, and :meth:`~LazyProxy._replay` performs them against the
base dataset.

One class serves both objects, because only two things differ and neither is worth a
second implementation: a ``DataArray`` has no ``data_vars`` (so the projection rules
have nothing to fire on) and its ``__getitem__`` is *indexing*, not projection (see
:meth:`LazyProxy._getitem_is_projection`). Which behaviour applies is read off the base
object rather than stored — the same derived-property discipline
:class:`~xrexpr.schema.SchemaState` follows.
"""

from functools import wraps
from typing import Any

import xarray as xr

from xrexpr.explain import format_plan
from xrexpr.ir import ContextOpen, FluentOp, LoweredOp, Opaque, Project, frozendict
from xrexpr.lower import Call, emit, to_lower_ir
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, to_opnode


class Explanation(str):
    """The text returned by :meth:`LazyProxy.explain`.

    Notes
    -----
    A plain ``str`` whose ``repr`` is the text itself, so a bare
    ``ds.plan.xyz.explain()`` at a REPL / in Jupyter prints the formatted, multi-line
    plan instead of an escaped one-liner (``'plan (2 ops):`` followed by an escaped
    newline).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


#: Attributes that consume the plan into a non-Dataset artifact (a figure, file,
#: array or DataFrame) rather than another link in the chain. Accessing one
#: materialises via :meth:`~LazyProxy.collect` and delegates to the realised
#: object instead of recording a lazy op — otherwise the call would be silently
#: recorded and never run (``collect`` is never reached, so the rewrite never fires).
#: ``plot`` is xarray's plot *accessor*, so delegating the attribute (not calling it)
#: also fixes ``plot.line()`` / ``plot.scatter()``. The set is shared by both base types:
#: a name absent from one of them (``to_array`` on a ``DataArray``, ``to_index`` on a
#: ``Dataset``) simply raises off the realised object, as it would eagerly.
_EAGER_ATTRS = frozenset(
    {
        "plot",
        "to_netcdf",
        "to_zarr",
        "to_dataframe",
        "to_dask_dataframe",
        "to_pandas",
        "to_series",
        "to_array",
        "to_dataarray",
        "to_numpy",
        "to_dict",
        "to_stacked_array",
        # DataArray-only artifacts, same rationale
        "to_index",
        "to_masked_array",
    }
)


@xr.register_dataset_accessor("plan")  # type: ignore[no-untyped-call]
@xr.register_dataarray_accessor("plan")  # type: ignore[no-untyped-call]
class LazyProxy:
    """Record operations on an ``xr.Dataset`` or ``xr.DataArray`` and replay them on ``collect()``.

    Parameters
    ----------
    base : xarray.Dataset or xarray.DataArray
        The object the plan runs against. xarray supplies it when the ``.plan``
        accessor is first reached.
    ops : list of FluentOp, optional
        The plan recorded so far. Empty for a fresh proxy; each recorded call passes
        the extended list to a new proxy.

    Notes
    -----
    Registered as the ``.plan`` accessor on both types, so ``ds.plan`` / ``da.plan``
    yields an empty proxy over it; each recorded call returns a fresh proxy (leaving the
    original untouched) carrying the extended plan, and nothing else — recording is a
    pure append, and the schema fold belongs to the optimiser.

    Nothing downstream of recording is ``Dataset``-specific:
    :meth:`~xrexpr.schema.SchemaState.from_dataset` accepts either, lowering and the
    rewrite rules reason about dim and variable *names*, and :meth:`_replay` is a
    ``getattr`` loop. The two places the base type is legible are
    :meth:`_getitem_is_projection` and — implicitly — the projection rules, which a
    ``DataArray``'s empty ``data_vars`` makes unfireable rather than wrong.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> import xrexpr  # registers the accessor
    >>> ds = xr.Dataset({"tas": (("time", "lat"), np.zeros((4, 3)))})
    >>> result = ds.plan.mean("lat").isel(time=0).collect()
    >>> also = ds["tas"].plan.mean("lat").isel(time=0).collect()
    """

    def __init__(
        self, base: xr.Dataset | xr.DataArray, ops: list[FluentOp] | None = None
    ):
        self._base = base
        self._ops: list[FluentOp] = list(ops) if ops else []

    def _record(self, method_name: str, *args: Any, **kwargs: Any) -> "LazyProxy":
        """Append one recorded call to the plan, returning a fresh proxy.

        Parameters
        ----------
        method_name : str
            The method that was called on the proxy.
        *args, **kwargs
            That call's arguments, kept verbatim for replay.

        Returns
        -------
        LazyProxy
            A new proxy over the same base object, carrying the extended plan. The
            receiver is left untouched.
        """
        node: FluentOp
        if method_name == "__getitem__" and not self._getitem_is_projection():
            node = Opaque(name=method_name, args=args, kwargs=frozendict(kwargs))
        else:
            node = to_opnode(method_name, args, kwargs)
        return LazyProxy(self._base, self._ops + [node])

    def _getitem_is_projection(self) -> bool:
        """Report whether a ``__getitem__`` recorded here selects *variables*.

        Returns
        -------
        bool
            ``True`` only when the live object is known to be a ``Dataset``; ``False``
            sends the call to :class:`~xrexpr.ir.Opaque`, replayed verbatim and never
            reordered.

        Notes
        -----
        :func:`~xrexpr.schema.to_opnode` is a pure function of one call, so it cannot tell
        the three meanings of ``__getitem__`` apart — the *receiver* settles it, and the
        receiver is the one thing recording knows that the call alone does not:

        - on a **Dataset** it is a projection, which is what ``to_opnode`` assumes;
        - on a **builder** (immediately after a :class:`~xrexpr.ir.ContextOpen`) it selects
          a *group* — ``gb[1]`` is the matching subset, not a variable;
        - on a **DataArray** it is *indexing*: a hashable key reads a coordinate
          (``da["lat"]``) and a list key selects *positions along the first dim*
          (``da[[1, 2]]``). Both would classify as a ``Project`` on the key alone, and the
          second is the one that bites: ``merge_adjacent_projects`` is deliberately
          schema-free (it decides on the two key sets), so ``da[[1, 2]][[1]]`` would
          collapse to ``da[[1]]`` — row 1, where the eager chain indexes *within* the
          selected pair and gives row 2.

        The live object is a ``DataArray`` **either** because the base is one **or**
        because the chain has already made it one, which a bare-name projection does
        (``ds.plan["tas"][[1, 2]][[1]]`` is the same wrong-row collapse, one op later).
        Nothing else recorded can: every modelled node maps a ``Dataset`` to a
        ``Dataset``, and the two spellings are told apart by the key ``args`` hold, since
        the node's ``variables`` reads ``("tas",)`` for both ``ds["tas"]`` and
        ``ds[["tas"]]``. Once a ``DataArray``, assumed a ``DataArray`` for the rest of the
        chain: an ``Opaque`` that converts back (``da.to_dataset()``) only costs the
        projections after it their modelling, and ``Opaque`` is the answer that is always
        *correct* — this classification is only ever allowed to give up optimisation, never
        to admit a rewrite the eager chain would not permit.
        """
        if self._in_context() or isinstance(self._base, xr.DataArray):
            return False
        return not any(
            isinstance(node, Project) and not isinstance(node.args[0], list)
            for node in self._ops
        )

    def _base_schema(self) -> SchemaState:
        """Snapshot the *base* object's schema, which is what the optimiser plans against.

        Returns
        -------
        SchemaState
            The base object's logical schema.

        Notes
        -----
        The proxy keeps no schema of its own: recording is a pure append, and the single
        fold over the plan belongs to the optimiser, which rewrites from the front and so
        needs the base rather than whatever recording ended on.

        A ``DataArray``'s snapshot holds its **coords only** — it has no ``data_vars`` for
        ``from_dataset`` to read — which narrows the model in one direction and one
        direction only: fewer facts, so a rule that needs one refuses. ``var_dims``
        answers ``None`` for every name (the projection rules' no-rewrite answer), and a
        dim carried by no coordinate is not in ``dim_names``, so a ``groupby`` over it
        fails ``lower._grouper_dims``' check and lands on the opaque fallback — replayed
        verbatim, correct, merely unfused. Teaching ``SchemaState`` to hold the array
        itself would recover the fusion; it is a schema change, not an accessor one.
        """
        return SchemaState.from_dataset(self._base)

    def _is_method_callable_on_base(self, name: str) -> bool:
        """Report whether ``name`` is a callable attribute of the base object.

        Parameters
        ----------
        name : str
            The attribute being routed by :meth:`__getattr__`.

        Returns
        -------
        bool
            ``True`` for a method to record, ``False`` for a property to evaluate
            eagerly.
        """
        return callable(getattr(self._base, name, None))

    def _in_context(self) -> bool:
        """Report whether the live object is a builder rather than a Dataset or DataArray.

        Returns
        -------
        bool
            ``True`` immediately after a :class:`~xrexpr.ir.ContextOpen`, and only there.

        Notes
        -----
        One node, not a scan: a context is a *pair*, so it is open only immediately after
        its opener. ``DatasetGroupBy.__getitem__`` selects a group and Rolling/Coarsen/
        Weighted reject ``__getitem__``, so xarray offers no builder→builder call that
        would keep one open across two nodes.

        This is what retires the trailing barrier the previous design needed: the closer
        is recorded normally and lowering, which can see the pair, decides what it meant —
        so ops *after* a context are modelled again instead of every one of them being
        opaque forever.
        """
        return bool(self._ops) and isinstance(self._ops[-1], ContextOpen)

    def __repr__(self) -> str:
        ops_preview = " -> ".join(
            f"{n.name}{n.args}{dict(n.kwargs)}" for n in self._ops
        )
        return f"<LazyProxy base={type(self._base).__name__} ops=[{ops_preview}]>"

    def __getattr__(self, name: str) -> Any:
        """Route an attribute to one of three behaviours.

        Parameters
        ----------
        name : str
            The attribute being looked up.

        Returns
        -------
        Any
            A recording wrapper for a callable, or the attribute read off the
            materialised result for a terminal or a property.

        Raises
        ------
        AttributeError
            For any name starting with ``_``, so internal and dunder lookups are not
            recorded.

        Notes
        -----
        - **Terminals** (:data:`_EAGER_ATTRS`: ``.plot``, ``.to_netcdf``, ...) consume
          the plan into an artifact that is no longer an xarray object, so they force
          materialisation and are read off the realised object — even though they are
          callable.
        - **Callables** (``.mean``, ``.isel``, ...) return a wrapper that records the
          call and returns a new proxy.
        - **Non-callable attributes** (``.dims``, ``.coords``, ...) force
          materialisation and are read off the realised object.

        Inside a context — immediately after a builder-returning call (``groupby``,
        ``resample``, ``rolling``, ``coarsen``, ``weighted``, ...) — the live object is a
        builder, not a Dataset or DataArray, so the callable test above (which asks the
        *base object*) is meaningless: ``DatasetGroupBy.first`` does not exist on
        ``Dataset``. Such a
        call is therefore always recorded, and the *meaning* of the resulting pair is
        settled later by :func:`~xrexpr.lower.to_lower_ir`, which can see both halves at
        once. A pair it fuses is modelled and optimisable; a pair it does not understand
        is demoted to ``Opaque`` — replayed verbatim, correct, merely unoptimised — and in
        either case the ops that *follow* are modelled normally.

        One cost of recording unconditionally in a context: a builder *property* such as
        ``ds.plan.groupby("x").groups`` comes back as a recording wrapper rather than the
        property's value, because telling methods from attributes would mean materialising.
        It does not work today either (the property branch collects eagerly and the builder
        has no ``.compute()``).
        """
        # protect internal / dunder attribute lookups
        if name.startswith("_"):
            raise AttributeError(name)

        # terminals must be checked before the callable branch: they are callable
        # (``.plot`` is an accessor with ``__call__``) but must not be recorded.
        if name in _EAGER_ATTRS:
            return getattr(self.collect(), name)

        if self._in_context():
            # no ``@wraps``: the name need not exist on the base object at all.
            def _context_method(*args: Any, **kwargs: Any) -> "LazyProxy":
                return self._record(name, *args, **kwargs)

            return _context_method

        if self._is_method_callable_on_base(name):

            @wraps(getattr(self._base, name))
            def _method(*args: Any, **kwargs: Any) -> "LazyProxy":
                return self._record(name, *args, **kwargs)

            return _method

        # non-callable (properties): evaluate eagerly and return the attribute
        return getattr(self.collect(), name)

    def __getitem__(self, key: Any) -> "LazyProxy":
        return self._record("__getitem__", key)

    def collect(self) -> xr.Dataset | xr.DataArray:
        """Optimise the recorded plan, replay it, and materialise the result.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The result, computed. A ``DataArray`` rather than a ``Dataset`` when the chain
            selects a single variable (e.g. ``ds.plan["temperature"]``), and always one
            when the base is a ``DataArray``.

        Raises
        ------
        InvalidExpressionError
            If the plan cannot be optimised — e.g. a select on a dim a preceding reduce
            removed.

        Notes
        -----
        The Polars-flavoured terminal of the plan, and the one place the whole pipeline
        is visible: the recorded (fluent) ops are lowered into what they mean
        (:func:`~xrexpr.lower.to_lower_ir`), rewritten (:func:`~xrexpr.optimize.optimize`),
        turned back into calls (:func:`~xrexpr.lower.emit`) and replayed onto the base
        object, then materialised via xarray's own ``.compute()`` so dask-backed data is
        realised.
        """
        return self._replay(emit(self._optimized())).compute()

    def _optimized(self) -> list[LoweredOp]:
        """Lower and rewrite the recorded plan.

        Returns
        -------
        list of LoweredOp
            What :meth:`collect` will emit and replay.
        """
        schema = self._base_schema()
        # Both stages plan against the base schema; lowering needs only its dim names, to
        # tell a dim grouper from a coordinate one (see ``lower._grouper_dims``).
        return optimize(to_lower_ir(self._ops, schema.dim_names), schema)

    def compute(self) -> xr.Dataset | xr.DataArray:
        """Alias for :meth:`collect`, for xarray users who reach for ``.compute()``.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Whatever :meth:`collect` returns.
        """
        return self.collect()

    def explain(self) -> Explanation:
        """Return the optimised plan as text, without running it (à la Polars ``explain``).

        Returns
        -------
        Explanation
            The **lowered plan** :meth:`collect` would replay — i.e. *after* lowering and
            optimisation — so the rewrite (merged / pushed-down selects) is visible.

        Raises
        ------
        InvalidExpressionError
            Whenever :meth:`collect` would, since the same optimisation runs.

        Notes
        -----
        One line per lowered node, each still rendered as the calls that node emits, so
        the listing gains the optimiser's own view of the plan — fused nodes, opaque
        barriers, minted dims — without ceasing to answer "what will run". See
        :mod:`xrexpr.explain` for the format and why each annotation is there.
        """
        return Explanation(format_plan(self._optimized()))

    def _replay(self, calls: list[Call]) -> xr.Dataset | xr.DataArray:
        """Perform each emitted :class:`~xrexpr.lower.Call` against the real object.

        Parameters
        ----------
        calls : list of Call
            The calls :func:`~xrexpr.lower.emit` generated, in order.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The result of the last call — not yet computed; :meth:`collect` does that.

        Notes
        -----
        Takes calls rather than nodes, so it never needs to know what a node *means* —
        deciding that is :func:`~xrexpr.lower.emit`'s job, and this stays the short
        ``getattr`` loop it was when every node was exactly one call.
        """
        obj: xr.Dataset | xr.DataArray = self._base
        for call in calls:
            if call.name == "__getitem__":
                obj = obj[call.args[0]]
            else:
                obj = getattr(obj, call.name)(*call.args, **call.kwargs)
        return obj
