"""The ``.plan`` accessor: a lazy recording proxy over an ``xr.Dataset``.

``ds.plan`` returns a :class:`LazyDatasetProxy` that records the chained method
calls made on it (``.mean``, ``.isel``, ``.sel``, ``__getitem__``, ...) instead
of executing them. Calling :meth:`~LazyDatasetProxy.compute` optimises the
recorded op list and replays it onto the real dataset.

This is the v1 skeleton (PR 1): the optimiser is still the demo ``_optimize_ops``
that ships on the ``add-accessor`` branch. Later PRs lift it into a schema-aware,
fixpoint optimiser (see ``docs/pr-plan.md``).
"""

from functools import wraps
from typing import Any

import xarray as xr

Op = tuple[str, tuple[Any, ...], dict[str, Any]]  # (method_name, args, kwargs)


@xr.register_dataset_accessor("plan")  # type: ignore[no-untyped-call]
class LazyDatasetProxy:
    """Record operations on an ``xr.Dataset`` and replay them on ``compute()``.

    Registered as the ``.plan`` accessor, so ``ds.plan`` yields an empty proxy
    over ``ds``; each recorded call returns a fresh proxy, leaving the original
    untouched.
    """

    def __init__(self, base_ds: xr.Dataset, ops: list[Op] | None = None):
        self._base_ds = base_ds
        self._ops = list(ops) if ops else []

    # --- internals -------------------------------------------------------
    def _record(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> "LazyDatasetProxy":
        return LazyDatasetProxy(
            self._base_ds, self._ops + [(method_name, args, kwargs)]
        )

    def _is_method_callable_on_dataset(self, name: str) -> bool:
        return callable(getattr(self._base_ds, name, None))

    def __repr__(self) -> str:
        ops_preview = " -> ".join(
            f"{name}{args}{kwargs}" for name, args, kwargs in self._ops
        )
        return f"<LazyDatasetProxy base={type(self._base_ds).__name__} ops=[{ops_preview}]>"

    # --- generic attr/method interception --------------------------------
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

    # --- compute + optimization ------------------------------------------
    def compute(self) -> xr.Dataset | xr.DataArray:
        """Optimise the recorded ops, replay them on the base dataset, return the result.

        Returns a ``DataArray`` rather than a ``Dataset`` when the chain selects a
        single variable (e.g. ``ds.plan["temperature"]``).
        """
        ops = self._optimize_ops(list(self._ops))
        ds: xr.Dataset | xr.DataArray = self._base_ds
        for method, args, kwargs in ops:
            if method == "__getitem__":
                ds = ds[args[0]]
            else:
                ds = getattr(ds, method)(*args, **kwargs)
        return ds

    def _optimize_ops(self, ops: list[Op]) -> list[Op]:
        """Demo optimiser: merge consecutive selects and push ``isel`` past reductions.

        - Merge consecutive ``isel`` calls into one indexer dict.
        - Merge consecutive ``sel`` calls into one indexer dict.
        - Push a following ``isel`` before a ``mean``/``sum``/``prod`` when their
          dims are disjoint (safe reorder).

        Intentionally conservative; superseded by a schema-aware fixpoint
        optimiser in a later PR.
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
