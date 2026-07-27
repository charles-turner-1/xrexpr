"""The ``.plan`` accessor: a lazy recording proxy over an ``xr.Dataset``.

``ds.plan`` returns a :class:`LazyDatasetProxy` that records the chained method
calls made on it (``.mean``, ``.isel``, ``.sel``, ``__getitem__``, ...) instead
of executing them. Calling :meth:`~LazyDatasetProxy.collect` optimises the
recorded plan and replays it onto the real dataset (:meth:`~LazyDatasetProxy.explain`
returns the optimised plan as text without running it).

The recorded plan is a list of :data:`~xrexpr.ir.FluentOp` variants: each call is
normalised by :func:`~xrexpr.schema.to_opnode`, a pure function of that call, and
appended — recording holds no schema of its own. ``collect`` then runs the plan through
the pipeline in ``lower.py``: :func:`~xrexpr.lower.to_lower_ir` translates what was
written into what it means, :func:`~xrexpr.optimize.optimize` rewrites that (a fixpoint
of rules, folding the base schema forward itself), :func:`~xrexpr.lower.emit` turns the
result back into calls, and :meth:`~LazyDatasetProxy._replay` performs them against the
base dataset. See ``docs/pr-plan.md``.
"""

from functools import wraps
from typing import Any

import xarray as xr

from xrexpr.ir import FluentOp, LoweredOp, Opaque, frozendict
from xrexpr.lower import Call, emit, to_lower_ir
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, to_opnode


class Explanation(str):
    """The text returned by :meth:`LazyDatasetProxy.explain`.

    A plain ``str`` whose ``repr`` is the text itself, so a bare
    ``ds.plan.xyz.explain()`` at a REPL / in Jupyter prints the formatted,
    multi-line plan instead of an escaped one-liner (``'plan (2 ops):\\n  ...'``).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


#: Attributes that consume the plan into a non-Dataset artifact (a figure, file,
#: array or DataFrame) rather than another link in the chain. Accessing one
#: materialises via :meth:`~LazyDatasetProxy.collect` and delegates to the realised
#: object instead of recording a lazy op — otherwise the call would be silently
#: recorded and never run (``collect`` is never reached, so the rewrite never fires).
#: ``plot`` is xarray's plot *accessor*, so delegating the attribute (not calling it)
#: also fixes ``plot.line()`` / ``plot.scatter()``.
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
    }
)


#: Methods that return a non-Dataset intermediate (``DatasetGroupBy``,
#: ``DatasetRolling``, ``DatasetWeighted``, ...) whose subsequent calls mean something
#: different from the same-named Dataset ops -- ``groupby("x").mean()`` reduces *within
#: groups*, not over the dataset. Once one is recorded the proxy stops modelling: every
#: later call records as :class:`~xrexpr.ir.Opaque`, so no rewrite rule can fire on or
#: across the context and replay stays verbatim. See :meth:`LazyDatasetProxy.__getattr__`.
_CONTEXT_METHODS = frozenset(
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


@xr.register_dataset_accessor("plan")  # type: ignore[no-untyped-call]
class LazyDatasetProxy:
    """Record operations on an ``xr.Dataset`` and replay them on ``collect()``.

    Registered as the ``.plan`` accessor, so ``ds.plan`` yields an empty proxy
    over ``ds``; each recorded call returns a fresh proxy (leaving the original
    untouched) carrying the extended plan and the schema after that op.
    """

    def __init__(self, base_ds: xr.Dataset, ops: list[FluentOp] | None = None):
        self._base_ds = base_ds
        self._ops: list[FluentOp] = list(ops) if ops else []

    def _record(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> "LazyDatasetProxy":
        node = (
            Opaque(name=method_name, args=args, kwargs=frozendict(kwargs))
            if self._in_opaque_context()
            else to_opnode(method_name, args, kwargs)
        )
        return LazyDatasetProxy(self._base_ds, self._ops + [node])

    def _base_schema(self) -> SchemaState:
        """The schema of the *base* dataset, which is what the optimiser plans against.

        The proxy keeps no schema of its own: recording is a pure append, and the single
        fold over the plan belongs to the optimiser, which rewrites from the front and so
        needs the base rather than whatever recording ended on.
        """
        return SchemaState.from_dataset(self._base_ds)

    def _is_method_callable_on_dataset(self, name: str) -> bool:
        return callable(getattr(self._base_ds, name, None))

    def _in_opaque_context(self) -> bool:
        """Whether a :data:`_CONTEXT_METHODS` op has been recorded, after which the live
        object is no longer a Dataset and nothing further can be modelled.

        Derived from the recorded plan rather than stored (house discipline, cf.
        ``Select.consumes``): plans are ~10 nodes, so the scan per record is irrelevant.
        """
        return any(
            isinstance(op, Opaque) and op.name in _CONTEXT_METHODS for op in self._ops
        )

    def __repr__(self) -> str:
        ops_preview = " -> ".join(
            f"{n.name}{n.args}{dict(n.kwargs)}" for n in self._ops
        )
        return f"<LazyDatasetProxy base={type(self._base_ds).__name__} ops=[{ops_preview}]>"

    def __getattr__(self, name: str) -> Any:
        """Route an attribute to one of three behaviours.

        - **Terminals** (:data:`_EAGER_ATTRS`: ``.plot``, ``.to_netcdf``, ...) consume
          the plan into a non-Dataset artifact, so they force materialisation and are
          read off the realised object — even though they are callable.
        - **Callables** (``.mean``, ``.isel``, ...) return a wrapper that records the
          call and returns a new proxy.
        - **Non-callable attributes** (``.dims``, ``.coords``, ...) force
          materialisation and are read off the realised dataset.

        Inside an opaque context — after a :data:`_CONTEXT_METHODS` call (``groupby``,
        ``resample``, ``rolling``, ``coarsen``, ``weighted``, ...) — the live object is a
        builder, not a Dataset, so the callable test above (which asks the *base dataset*)
        is meaningless: ``DatasetGroupBy.first`` does not exist on ``Dataset``, and a
        grouped ``.mean()`` is not the Dataset reduce of the same name. Every call from
        the context op onward therefore records as ``Opaque``, which no rewrite rule can
        fire on or across, so such chains are **correct but never optimised** — replayed
        exactly as written. Modelling grouped/windowed semantics (so they optimise again)
        needs a lowering stage with lookahead over the finished plan; see
        ``docs/roadmap/02-lowering.md``.

        One cost of recording unconditionally in a context: a builder *property* such as
        ``ds.plan.groupby("x").groups`` comes back as a recording wrapper rather than the
        property's value, because telling methods from attributes would mean materialising.
        It does not work today either (the property branch collects eagerly and the builder
        has no ``.compute()``); lowering's typed context node removes the guesswork.
        """
        # protect internal / dunder attribute lookups
        if name.startswith("_"):
            raise AttributeError(name)

        # terminals must be checked before the callable branch: they are callable
        # (``.plot`` is an accessor with ``__call__``) but must not be recorded.
        if name in _EAGER_ATTRS:
            return getattr(self.collect(), name)

        if self._in_opaque_context():
            # no ``@wraps``: the name need not exist on the base dataset at all.
            def _context_method(*args: Any, **kwargs: Any) -> "LazyDatasetProxy":
                return self._record(name, *args, **kwargs)

            return _context_method

        if self._is_method_callable_on_dataset(name):

            @wraps(getattr(self._base_ds, name))
            def _method(*args: Any, **kwargs: Any) -> "LazyDatasetProxy":
                return self._record(name, *args, **kwargs)

            return _method

        # non-callable (properties): evaluate eagerly and return the attribute
        return getattr(self.collect(), name)

    def __getitem__(self, key: Any) -> "LazyDatasetProxy":
        return self._record("__getitem__", key)

    def collect(self) -> xr.Dataset | xr.DataArray:
        """Optimise the recorded plan, replay it, and materialise the result.

        The Polars-flavoured terminal of the plan, and the one place the whole pipeline
        is visible: the recorded (fluent) ops are lowered into what they mean
        (:func:`~xrexpr.lower.to_lower_ir`), rewritten (:func:`~xrexpr.optimize.optimize`),
        turned back into calls (:func:`~xrexpr.lower.emit`) and replayed onto the base
        dataset, then materialised via xarray's own ``.compute()`` so dask-backed data is
        realised. Returns a ``DataArray`` rather than a ``Dataset`` when the chain
        selects a single variable (e.g. ``ds.plan["temperature"]``).

        Raises :class:`~xrexpr.exceptions.InvalidExpressionError` if the plan cannot be
        optimised (e.g. a select on a dim a preceding reduce removed).
        """
        return self._replay(emit(self._optimized())).compute()

    def _optimized(self) -> list[LoweredOp]:
        """The lowered, rewritten plan — what :meth:`collect` will emit and replay."""
        return optimize(to_lower_ir(self._ops), self._base_schema())

    def compute(self) -> xr.Dataset | xr.DataArray:
        """Alias for :meth:`collect`, for xarray users who reach for ``.compute()``."""
        return self.collect()

    def explain(self) -> Explanation:
        """Return the optimised plan as text, without running it (à la Polars ``explain``).

        Shows the ops :meth:`collect` would actually replay — i.e. *after* lowering and
        optimisation — so the rewrite (merged / pushed-down selects) is visible. Raises
        the same :class:`~xrexpr.exceptions.InvalidExpressionError` as :meth:`collect`
        when the plan is invalid.
        """
        plan = self._optimized()
        if not plan:
            return Explanation("plan (0 ops)")
        body = "\n".join(
            f"  {i}. {self._format_node(n)}" for i, n in enumerate(plan, 1)
        )
        return Explanation(f"plan ({len(plan)} ops):\n{body}")

    @staticmethod
    def _format_node(node: LoweredOp) -> str:
        """One-line human-readable form of a lowered node for :meth:`explain`."""
        if node.name == "__getitem__":
            return f"[{node.args[0]!r}]"
        parts = [repr(a) for a in node.args]
        parts += [f"{k}={v!r}" for k, v in node.kwargs.items()]
        return f"{node.name}({', '.join(parts)})"

    def _replay(self, calls: list[Call]) -> xr.Dataset | xr.DataArray:
        """Perform each emitted :class:`~xrexpr.lower.Call` against the real dataset.

        Takes calls rather than nodes, so it never needs to know what a node *means* —
        deciding that is :func:`~xrexpr.lower.emit`'s job, and this stays the short
        ``getattr`` loop it was when every node was exactly one call.
        """
        ds: xr.Dataset | xr.DataArray = self._base_ds
        for call in calls:
            if call.name == "__getitem__":
                ds = ds[call.args[0]]
            else:
                ds = getattr(ds, call.name)(*call.args, **call.kwargs)
        return ds
