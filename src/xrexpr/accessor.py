"""The ``.plan`` accessor: a lazy proxy that records calls on a Dataset or DataArray.

``ds.plan`` returns a :class:`LazyProxy`. Calls made on it (``.mean``, ``.isel``,
``.sel``, ``__getitem__``, ...) are *recorded* instead of run.
:meth:`~LazyProxy.collect` optimises the recorded plan and replays it onto the real
dataset. :meth:`~LazyProxy.explain` returns the optimised plan as text without running
it.

Recording only appends, and the proxy keeps no schema of its own. That's a deliberate
choice. The proxy does have the base dataset, so it *could* fold a schema forward and
resolve things as it goes. But after any call it can't model, that schema is a guess,
and anything resolved against it would be wrong in ways nobody would notice. The
optimiser owns the fold, because it knows where the guessing starts.

One class handles both Datasets and DataArrays. Only two things differ, and neither is
worth a second class:

- a ``DataArray`` has no ``data_vars``, so the projection rules never fire on it;
- its ``__getitem__`` is *indexing*, not projection.

Which behaviour applies is read off the base object each time, not stored.

See ``docs/internals/pipeline.md``.
"""

from collections.abc import Hashable
from functools import wraps
from typing import Any

import xarray as xr
from frozendict import frozendict

from xrexpr.explain import format_plan
from xrexpr.ir import ContextOpen, FluentOp, LoweredOp, Opaque, Project
from xrexpr.lower import Call, emit, to_lower_ir
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, to_opnode


class Explanation(str):
    """The text returned by :meth:`LazyProxy.explain`.

    Notes
    -----
    A plain ``str`` whose ``repr`` is the text itself. So a bare
    ``ds.plan.xyz.explain()`` at a REPL or in Jupyter prints the formatted, multi-line
    plan, rather than a one-line string full of escaped newlines.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


#: Attributes that turn the plan into something that isn't a Dataset (a figure, a file,
#: an array, a DataFrame) instead of extending the chain. Accessing one runs
#: :meth:`~LazyProxy.collect` and reads the attribute off the result. Without this the
#: call would be recorded and never run, because nothing would ever call ``collect``.
#:
#: ``plot`` is xarray's plot *accessor*, so delegating the attribute (rather than
#: calling it) also covers ``plot.line()`` / ``plot.scatter()``. The set is shared by
#: both base types. A name one of them doesn't have (``to_array`` on a ``DataArray``,
#: ``to_index`` on a ``Dataset``) just raises from the realised object, as it would
#: eagerly.
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
        The object the plan runs against. xarray passes it in when ``.plan`` is first
        accessed.
    ops : list of FluentOp, optional
        The plan recorded so far. Empty for a fresh proxy; each recorded call creates a
        new proxy with the extended list.

    Notes
    -----
    Registered as the ``.plan`` accessor on both types, so ``ds.plan`` and ``da.plan``
    give an empty proxy. Each recorded call returns a *new* proxy with the extended plan
    and leaves the original unchanged. Recording does nothing else; the schema fold
    belongs to the optimiser.

    Nothing after recording is Dataset-specific:
    :meth:`~xrexpr.schema.SchemaState.from_dataset` accepts either type, lowering and
    the rules reason about dim and variable names, and ``_replay`` is a ``getattr``
    loop. The base type only matters in two places: ``_getitem_is_projection``, and
    (implicitly) the projection rules, which never fire on a ``DataArray`` because it
    has no ``data_vars``.

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
        """Append one recorded call to the plan and return a new proxy.

        Parameters
        ----------
        method_name : str
            The method that was called on the proxy.
        *args, **kwargs
            That call's arguments, kept verbatim for replay.

        Returns
        -------
        LazyProxy
            A new proxy over the same base object, with the extended plan. This proxy is
            left unchanged.
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
            ``True`` only when the object at this point is known to be a ``Dataset``.
            ``False`` records the call as :class:`~xrexpr.ir.Opaque`: replayed verbatim,
            never reordered.

        Notes
        -----
        :func:`~xrexpr.schema.to_opnode` only sees the call, so it can't tell the three
        meanings of ``__getitem__`` apart. What decides it is the object being indexed,
        and that's the one thing recording knows that the call doesn't:

        - on a **Dataset** it's a projection, which is what ``to_opnode`` assumes;
        - on a **builder** (straight after a :class:`~xrexpr.ir.ContextOpen`) it selects
          a *group*: ``gb[1]`` is the matching subset, not a variable;
        - on a **DataArray** it's *indexing*. A hashable key reads a coordinate
          (``da["lat"]``) and a list key selects *positions along the first dim*
          (``da[[1, 2]]``). Both look like a ``Project`` from the key alone, and the
          list case is the dangerous one. ``merge_adjacent_projects`` decides from the
          two keys without a schema, so ``da[[1, 2]][[1]]`` would collapse to
          ``da[[1]]``: row 1, where the eager chain indexes *within* the selected pair
          and gives row 2.

        So the answer is "projection" only while the object is *known* to be a
        ``Dataset``. That needs every node recorded so far to be modelled, and to map a
        Dataset to a Dataset. Each condition has one way to fail, and this method checks
        both:

        - a ``single`` :class:`~xrexpr.ir.Project` returns a ``DataArray``.
          ``ds.plan["tas"][[1, 2]][[1]]`` is the same wrong-row collapse as above, one
          op later. ``single`` comes from the *key*, not ``variables``, which is
          ``("tas",)`` for both ``ds["tas"]`` and ``ds[["tas"]]``.
        - an :class:`~xrexpr.ir.Opaque` is unmodelled, and that includes its *result
          type*. ``ds.plan.pipe(lambda d: d["tas"])`` and ``ds.plan.get("tas")`` both
          record one, and both return a ``DataArray``. Guessing from the method name
          which opaques return a Dataset would need a table that quietly goes stale as
          xarray grows.

        The cost is limited to ``merge_adjacent_projects``. Past an ``Opaque`` the
        folded ``data_vars`` is a guess, so ``_trusted_prefix`` already keeps
        :func:`~xrexpr.optimize.pushdown_projections` out. Two list projections after an
        opaque that really does return a Dataset (``ds.plan.transpose()[["a",
        "b"]][["a"]]``) therefore don't merge. Without knowing the object's type, that
        case can't be told apart from the unsafe one.

        Once not known to be a ``Dataset``, it stays that way for the rest of the chain.
        An ``Opaque`` that converts back (``da.to_dataset()``) just costs the
        projections after it their modelling. ``Opaque`` is always a *correct* answer:
        this check may only give up optimisation, never allow a rewrite the eager chain
        wouldn't.
        """
        if self._in_context() or isinstance(self._base, xr.DataArray):
            return False
        return not any(
            isinstance(node, Opaque) or (isinstance(node, Project) and node.single)
            for node in self._ops
        )

    def _base_schema(self) -> SchemaState[Hashable]:
        """Snapshot the *base* object's schema, which is what the optimiser plans against.

        Returns
        -------
        SchemaState
            The base object's logical schema.

        Notes
        -----
        The proxy keeps no schema of its own: recording only appends, and the optimiser
        does the one fold over the plan. It rewrites from the front, so it needs the
        base schema, not whatever recording ended on.

        A ``DataArray``'s snapshot holds its **coords only**, because it has no
        ``data_vars`` for ``from_dataset`` to read. That only ever removes facts, so
        rules that need them just don't fire:

        - ``var_dims`` returns ``None`` for every name (no projection rewrites);
        - a dim with no coordinate isn't in ``dim_names``, so a ``groupby`` over it
          fails ``lower._grouper_dims``' check and replays verbatim. Correct, just not
          fused.

        Getting that fusion back would mean teaching ``SchemaState`` to hold the array
        itself. That's a schema change, not an accessor one.
        """
        return SchemaState[Hashable].from_dataset(self._base)

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
        """Report whether the object at this point is a builder, not a Dataset or DataArray.

        Returns
        -------
        bool
            ``True`` straight after a :class:`~xrexpr.ir.ContextOpen`, and only then.

        Notes
        -----
        Only the last node matters, because a context is a *pair*: it's open only right
        after its opener. ``DatasetGroupBy.__getitem__`` selects a group, and
        Rolling/Coarsen/Weighted reject ``__getitem__``, so xarray has no call that
        keeps a builder open across two nodes.

        The closer is recorded as a normal call, and lowering (which can see the pair)
        decides what it meant. So ops *after* a context are modelled as usual.
        """
        return bool(self._ops) and isinstance(self._ops[-1], ContextOpen)

    def __repr__(self) -> str:
        ops_preview = " -> ".join(
            f"{n.name}{n.args}{dict(n.kwargs)}" for n in self._ops
        )
        return f"<LazyProxy base={type(self._base).__name__} ops=[{ops_preview}]>"

    def __getattr__(self, name: str) -> Any:
        """Decide what accessing an attribute does: record it, or run the plan.

        Parameters
        ----------
        name : str
            The attribute being looked up.

        Returns
        -------
        Any
            A recording wrapper for a method, or the attribute read off the computed
            result for a terminal or a property.

        Raises
        ------
        AttributeError
            For any name starting with ``_``, so internal and dunder lookups aren't
            recorded.

        Notes
        -----
        Three cases:

        - **Terminals** (``_EAGER_ATTRS``: ``.plot``, ``.to_netcdf``, ...) turn the plan
          into something that isn't an xarray object. They run the plan and are read off
          the result, even though they're callable.
        - **Methods** (``.mean``, ``.isel``, ...) return a wrapper that records the call
          and returns a new proxy.
        - **Properties** (``.dims``, ``.coords``, ...) run the plan and are read off the
          result.

        Inside a context (straight after ``groupby``, ``resample``, ``rolling``,
        ``coarsen``, ``weighted``, ...) the object is a builder, not a Dataset or
        DataArray. The "is it callable?" check above asks the *base object*, so it's
        meaningless here: ``DatasetGroupBy.first`` doesn't exist on ``Dataset``. So
        inside a context every attribute is recorded as a call, and
        :func:`~xrexpr.lower.to_lower_ir`, which sees both halves, decides what the pair
        means later. A pair it fuses is modelled and optimisable. A pair it doesn't
        understand becomes ``Opaque``: replayed verbatim, correct, just not optimised.
        Either way, the ops *after* the pair are modelled normally.

        One cost of always recording inside a context: a builder *property* such as
        ``ds.plan.groupby("x").groups`` comes back as a recording wrapper rather than
        the value, because telling methods from properties would mean running the plan.
        Treating it as a property wouldn't help either: that path computes the result
        eagerly, and a builder has no ``.compute()``.
        """
        # protect internal / dunder attribute lookups
        if name.startswith("_"):
            raise AttributeError(name)

        # Check terminals before the callable branch: they *are* callable (``.plot`` is
        # an accessor with ``__call__``) but must not be recorded.
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
        """Optimise the recorded plan, replay it, and compute the result.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The computed result. A ``DataArray`` when the chain selects a single
            variable (e.g. ``ds.plan["temperature"]``), and always one when the base is
            a ``DataArray``.

        Raises
        ------
        InvalidExpressionError
            If the plan can't be optimised, e.g. a select on a dim a preceding reduce
            removed.

        Notes
        -----
        The end of the plan (named after Polars), and the one place the whole pipeline
        is visible:

        1. lower the recorded ops into what they mean
        (:func:`~xrexpr.lower.to_lower_ir`); 2. rewrite them
        (:func:`~xrexpr.optimize.optimize`); 3. turn them back into calls
        (:func:`~xrexpr.lower.emit`) and replay those on the base; 4. call xarray's
        ``.compute()`` so dask-backed data is computed.
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
        # Both stages plan against the base schema. Lowering only needs its dim names,
        # to tell a dim grouper from a coordinate one (see ``lower._grouper_dims``).
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
        """Return the optimised plan as text, without running it (like Polars' ``explain``).

        Returns
        -------
        Explanation
            The **lowered plan** :meth:`collect` would replay, i.e. *after* lowering and
            optimisation, so the rewrites (merged or pushed-down selects, etc.) are
            visible.

        Raises
        ------
        InvalidExpressionError
            Whenever :meth:`collect` would, since the same optimisation runs.

        Notes
        -----
        One line per lowered node, each shown as the calls that node emits. So it shows
        the optimiser's view of the plan (fused nodes, opaque barriers, created dims)
        while still answering "what will run". See ``xrexpr.explain`` for the format.
        """
        return Explanation(format_plan(self._optimized()))

    def _replay(self, calls: list[Call]) -> xr.Dataset | xr.DataArray:
        """Run each emitted :class:`~xrexpr.lower.Call` against the real object.

        Parameters
        ----------
        calls : list of Call
            The calls :func:`~xrexpr.lower.emit` generated, in order.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The result of the last call, not yet computed (:meth:`collect` does that).

        Notes
        -----
        Takes calls, not nodes, so it never needs to know what a node *means*. That's
        :func:`~xrexpr.lower.emit`'s job, and it keeps this a short ``getattr`` loop.
        """
        obj: xr.Dataset | xr.DataArray = self._base
        for call in calls:
            if call.name == "__getitem__":
                obj = obj[call.args[0]]
            else:
                obj = getattr(obj, call.name)(*call.args, **call.kwargs)
        return obj
