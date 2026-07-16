"""The ``.plan`` accessor: a lazy recording proxy over an ``xr.Dataset``.

``ds.plan`` returns a :class:`LazyDatasetProxy` that records the chained method
calls made on it (``.mean``, ``.isel``, ``.sel``, ``__getitem__``, ...) instead
of executing them. Calling :meth:`~LazyDatasetProxy.compute` optimises the
recorded plan and replays it onto the real dataset.

As of PR 6 the recorded plan is a list of :class:`~xrexpr.ir.OpNode`: each call is
normalised by :func:`~xrexpr.schema.to_opnode` against a :class:`SchemaState` that
is threaded from ``self._base_ds`` and evolved per op (no materialisation). Replay
walks those nodes. The *optimiser* is still the demo ``_optimize_ops`` from the
``add-accessor`` branch — it is bridged to bare tuples for now and gets replaced by
an ``OpNode``-native ``optimize()`` in PR 7 (#10). See ``docs/pr-plan.md``.
"""

from functools import wraps
from typing import Any

import xarray as xr
from frozendict import frozendict

from xrexpr.ir import OpNode
from xrexpr.operations import spec as op_spec
from xrexpr.schema import SchemaState, apply_schema, to_opnode

# TEMPORARY (PR 6): the demo ``_optimize_ops`` still runs on bare
# ``(name, args, kwargs)`` tuples, so the recorded ``OpNode``s are bridged to this
# alias around it (``_legacy_ops`` / ``_legacy_to_node``). The alias, the bridge, and
# ``_optimize_ops`` are all deleted in PR 7 (#10), which optimises ``OpNode``s directly.
Op = tuple[str, tuple[Any, ...], dict[str, Any]]  # (method_name, args, kwargs)


@xr.register_dataset_accessor("plan")  # type: ignore[no-untyped-call]
class LazyDatasetProxy:
    """Record operations on an ``xr.Dataset`` and replay them on ``compute()``.

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
        return getattr(self.compute(), name)

    def __getitem__(self, key: Any) -> "LazyDatasetProxy":
        return self._record("__getitem__", key)

    def compute(self) -> xr.Dataset | xr.DataArray:
        """Optimise the recorded plan, replay it on the base dataset, return the result.

        Returns a ``DataArray`` rather than a ``Dataset`` when the chain selects a
        single variable (e.g. ``ds.plan["temperature"]``).

        TEMPORARY (PR 1): the terminal is named ``compute`` and inherited verbatim
        from the demo. It is renamed to ``.collect()`` (Polars-flavour) in PR 10
        (#13), which also makes it call xarray's own ``.compute()`` on the replayed
        result — this method only replays, it never materialises dask-backed data.
        """
        optimized = self._optimize_ops(self._legacy_ops())
        return self._replay([self._legacy_to_node(op) for op in optimized])

    def _replay(self, nodes: list[OpNode]) -> xr.Dataset | xr.DataArray:
        """Walk the optimised ``OpNode`` plan, calling the real xarray methods."""
        ds: xr.Dataset | xr.DataArray = self._base_ds
        for node in nodes:
            if node.name == "__getitem__":
                ds = ds[node.args[0]]
            else:
                ds = getattr(ds, node.name)(*node.args, **node.kwargs)
        return ds

    def _legacy_ops(self) -> list[Op]:
        """Bridge recorded ``OpNode``s to the ``(name, args, kwargs)`` tuples the demo
        optimiser consumes.

        TEMPORARY (PR 6): removed together with ``_optimize_ops`` in PR 7 (#10).
        """
        return [(n.name, n.args, dict(n.kwargs)) for n in self._ops]

    @staticmethod
    def _legacy_to_node(op: Op) -> OpNode:
        """Rebuild an ``OpNode`` from a post-optimise tuple so replay walks nodes.

        Only ``kind`` matters for replay; ``consumes``/``indexer`` are irrelevant once
        optimisation is done. TEMPORARY (PR 6): gone with the bridge in PR 7 (#10).
        """
        name, args, kwargs = op
        spec = op_spec(name)
        return OpNode(
            name=name,
            kind=spec.kind if spec is not None else "opaque",
            args=args,
            kwargs=frozendict(kwargs),
        )

    def _optimize_ops(self, ops: list[Op]) -> list[Op]:
        """Demo optimiser: merge consecutive selects and push ``isel`` past reductions.

        - Merge consecutive ``isel`` calls into one indexer dict.
        - Merge consecutive ``sel`` calls into one indexer dict.
        - Push a following ``isel`` before a ``mean``/``sum``/``prod`` when their
          dims are disjoint (safe reorder).

        TEMPORARY (PR 1): this whole method is a placeholder carried over verbatim
        from the demo. It is deleted, not refactored — lifted into ``optimize.py``
        as small rule functions run to a fixpoint (PR 7, #10), fed normalised
        ``OpNode`` metadata so the arg-poking below (``decode_*_args``, ``red_dims``
        guessing) disappears (already resolved by ``to_opnode``). Don't polish it here.
        """
        new_ops: list[Op] = []
        i = 0
        while i < len(ops):
            name, args, kwargs = ops[i]

            # Merge consecutive isel dicts: isel(dim=...) or isel(dict)
            if name == "isel":
                merged: dict[str, Any] = {}

                def decode_isel_args(
                    a: tuple[Any, ...], kw: dict[str, Any]
                ) -> dict[str, Any]:
                    if len(a) == 1 and isinstance(a[0], dict):
                        return dict(a[0], **kw)
                    return dict(kw)

                merged.update(decode_isel_args(args, kwargs))
                j = i + 1
                while j < len(ops) and ops[j][0] == "isel":
                    _, aa, kk = ops[j]
                    merged.update(decode_isel_args(aa, kk))
                    j += 1
                new_ops.append(("isel", (merged,), {}))
                i = j
                continue

            # Merge consecutive sel dicts
            if name == "sel":
                merged = {}

                def decode_sel_args(
                    a: tuple[Any, ...], kw: dict[str, Any]
                ) -> dict[str, Any]:
                    if len(a) == 1 and isinstance(a[0], dict):
                        return dict(a[0], **kw)
                    return dict(kw)

                merged.update(decode_sel_args(args, kwargs))
                j = i + 1
                while j < len(ops) and ops[j][0] == "sel":
                    _, aa, kk = ops[j]
                    merged.update(decode_sel_args(aa, kk))
                    j += 1
                new_ops.append(("sel", (merged,), {}))
                i = j
                continue

            # Push a following isel before a reduction when dims are disjoint.
            if (
                name in ("mean", "sum", "prod")
                and (i + 1) < len(ops)
                and ops[i + 1][0] == "isel"
            ):
                reduce_name, r_args, r_kwargs = ops[i]
                _, isel_args, isel_kwargs = ops[i + 1]

                # dims removed by the reduction (dim kwarg, or first positional)
                if "dim" in r_kwargs:
                    red_dims = r_kwargs["dim"]
                elif len(r_args) >= 1:
                    red_dims = r_args[0]
                else:
                    red_dims = ()
                if isinstance(red_dims, str):
                    red_dims = (red_dims,)
                else:
                    red_dims = tuple(red_dims)

                # dims touched by the isel indexer
                if len(isel_args) == 1 and isinstance(isel_args[0], dict):
                    indexer = dict(isel_args[0])
                else:
                    indexer = dict(isel_kwargs)

                if set(indexer.keys()).isdisjoint(red_dims):
                    new_ops.append(("isel", (indexer,), {}))
                    new_ops.append((reduce_name, r_args, r_kwargs))
                    i += 2
                    continue

            new_ops.append((name, args, kwargs))
            i += 1

        return new_ops
