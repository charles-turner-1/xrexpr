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

from xrexpr.ir import ContextOpen, FluentOp, LoweredOp, Opaque, frozendict
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
        node: FluentOp
        if method_name == "__getitem__" and self._in_context():
            # A builder ``__getitem__`` selects a *group* (``gb[1]`` is the matching
            # subset), not a variable, so it must never classify as a ``Project``.
            node = Opaque(name=method_name, args=args, kwargs=frozendict(kwargs))
        else:
            node = to_opnode(method_name, args, kwargs)
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

    def _in_context(self) -> bool:
        """Whether the live object is a builder rather than a Dataset.

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

        Inside a context — immediately after a builder-returning call (``groupby``,
        ``resample``, ``rolling``, ``coarsen``, ``weighted``, ...) — the live object is a
        builder, not a Dataset, so the callable test above (which asks the *base dataset*)
        is meaningless: ``DatasetGroupBy.first`` does not exist on ``Dataset``. Such a
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

        Shows the calls :meth:`collect` would actually replay — i.e. *after* lowering and
        optimisation — so the rewrite (merged / pushed-down selects) is visible. Raises
        the same :class:`~xrexpr.exceptions.InvalidExpressionError` as :meth:`collect`
        when the plan is invalid.

        Formatted from the **emitted calls** rather than the lowered nodes, which keeps
        this the answer to "what will run": a fused node such as
        :class:`~xrexpr.ir.GroupedReduce` shows as the two calls it replays as. Showing
        the lowered *nodes* — the more informative artefact, and what Polars does — is a
        deliberate later change (``docs/roadmap/02-lowering.md`` PR 8), not something to
        drift into as a side effect of the first fused node.
        """
        calls = emit(self._optimized())
        if not calls:
            return Explanation("plan (0 ops)")
        body = "\n".join(
            f"  {i}. {self._format_call(c)}" for i, c in enumerate(calls, 1)
        )
        return Explanation(f"plan ({len(calls)} ops):\n{body}")

    @staticmethod
    def _format_call(call: Call) -> str:
        """One-line human-readable form of an emitted call for :meth:`explain`."""
        if call.name == "__getitem__":
            return f"[{call.args[0]!r}]"
        parts = [repr(a) for a in call.args]
        parts += [f"{k}={v!r}" for k, v in call.kwargs.items()]
        return f"{call.name}({', '.join(parts)})"

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
