"""The ``.plan`` accessor: a lazy recording proxy over an ``xr.Dataset``.

``ds.plan`` returns a :class:`LazyDatasetProxy` that records the chained method
calls made on it (``.mean``, ``.isel``, ``.sel``, ``__getitem__``, ...) instead
of executing them. Calling :meth:`~LazyDatasetProxy.collect` optimises the
recorded plan and replays it onto the real dataset (:meth:`~LazyDatasetProxy.explain`
returns the optimised plan as text without running it).

The recorded plan is a list of :class:`~xrexpr.ir.OpNode`: each call is normalised
by :func:`~xrexpr.schema.to_opnode` against a :class:`SchemaState` threaded from
``self._base_ds`` and evolved per op (no materialisation). ``collect`` runs the plan
through :func:`~xrexpr.optimize.optimize` (a fixpoint of rewrite rules), replays the
optimised ``OpNode``s onto the base dataset, and materialises the result. See
``docs/pr-plan.md``.
"""

from functools import wraps
from typing import Any

import xarray as xr

from xrexpr.ir import OpNode
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, apply_schema, to_opnode


class Explanation(str):
    """The text returned by :meth:`LazyDatasetProxy.explain`.

    A plain ``str`` whose ``repr`` is the text itself, so a bare
    ``ds.plan.xyz.explain()`` at a REPL / in Jupyter prints the formatted,
    multi-line plan instead of an escaped one-liner (``'plan (2 ops):\\n  ...'``).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return str(self)


@xr.register_dataset_accessor("plan")  # type: ignore[no-untyped-call]
class LazyDatasetProxy:
    """Record operations on an ``xr.Dataset`` and replay them on ``collect()``.

    Registered as the ``.plan`` accessor, so ``ds.plan`` yields an empty proxy
    over ``ds``; each recorded call returns a fresh proxy (leaving the original
    untouched) carrying the extended plan and the schema after that op.
    """

    def __init__(
        self,
        base_ds: xr.Dataset,
        ops: list[OpNode] | None = None,
        schema: SchemaState | None = None,
    ):
        self._base_ds = base_ds
        self._ops: list[OpNode] = list(ops) if ops else []
        # schema is threaded by ``_record``; recompute from the base only for a
        # fresh (empty) proxy such as the one xarray builds for ``ds.plan``.
        self._schema = (
            schema if schema is not None else SchemaState.from_dataset(base_ds)
        )

    def _record(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> "LazyDatasetProxy":
        node = to_opnode(self._schema, method_name, args, kwargs)
        return LazyDatasetProxy(
            self._base_ds,
            self._ops + [node],
            apply_schema(self._schema, node),
        )

    def _is_method_callable_on_dataset(self, name: str) -> bool:
        return callable(getattr(self._base_ds, name, None))

    def __repr__(self) -> str:
        ops_preview = " -> ".join(
            f"{n.name}{n.args}{dict(n.kwargs)}" for n in self._ops
        )
        return f"<LazyDatasetProxy base={type(self._base_ds).__name__} ops=[{ops_preview}]>"

    def __getattr__(self, name: str) -> Any:
        """Record callable methods; eagerly resolve non-callable attributes.

        Callables (``.mean``, ``.isel``, ...) return a wrapper that records the
        call and returns a new proxy. Non-callable attributes (``.dims``,
        ``.coords``, ...) force materialisation and are read off the realised
        dataset.
        """
        # protect internal / dunder attribute lookups
        if name.startswith("_"):
            raise AttributeError(name)

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

        The Polars-flavoured terminal of the plan: it runs the recorded ops through
        :func:`~xrexpr.optimize.optimize`, replays the optimised ``OpNode``s onto the
        base dataset, and calls xarray's own ``.compute()`` so dask-backed data is
        realised. Returns a ``DataArray`` rather than a ``Dataset`` when the chain
        selects a single variable (e.g. ``ds.plan["temperature"]``).

        Raises :class:`~xrexpr.exceptions.InvalidExpressionError` if the plan cannot be
        optimised (e.g. a select on a dim a preceding reduce removed).
        """
        return self._replay(optimize(self._ops)).compute()

    def compute(self) -> xr.Dataset | xr.DataArray:
        """Alias for :meth:`collect`, for xarray users who reach for ``.compute()``."""
        return self.collect()

    def explain(self) -> Explanation:
        """Return the optimised plan as text, without running it (à la Polars ``explain``).

        Shows the ops :meth:`collect` would actually replay — i.e. *after* optimisation
        — so the rewrite (merged / pushed-down selects) is visible. Raises the same
        :class:`~xrexpr.exceptions.InvalidExpressionError` as :meth:`collect` when the
        plan is invalid.
        """
        plan = optimize(self._ops)
        if not plan:
            return Explanation("plan (0 ops)")
        body = "\n".join(
            f"  {i}. {self._format_node(n)}" for i, n in enumerate(plan, 1)
        )
        return Explanation(f"plan ({len(plan)} ops):\n{body}")

    @staticmethod
    def _format_node(node: OpNode) -> str:
        """One-line human-readable form of an ``OpNode`` for :meth:`explain`."""
        if node.name == "__getitem__":
            return f"[{node.args[0]!r}]"
        parts = [repr(a) for a in node.args]
        parts += [f"{k}={v!r}" for k, v in node.kwargs.items()]
        return f"{node.name}({', '.join(parts)})"

    def _replay(self, nodes: list[OpNode]) -> xr.Dataset | xr.DataArray:
        """Walk the optimised ``OpNode`` plan, calling the real xarray methods."""
        ds: xr.Dataset | xr.DataArray = self._base_ds
        for node in nodes:
            if node.name == "__getitem__":
                ds = ds[node.args[0]]
            else:
                ds = getattr(ds, node.name)(*node.args, **node.kwargs)
        return ds
