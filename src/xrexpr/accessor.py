# type: ignore
from functools import wraps
from typing import Any

import xarray as xr

Op = tuple[str, tuple, dict]  # (method_name, args, kwargs)


class LazyDatasetProxy:
    """
    Generic proxy that records operations on an xr.Dataset and can compute them later.
    """

    def __init__(self, base_ds: xr.Dataset, ops: list[Op] = None):
        self._base_ds = base_ds
        self._ops = list(ops) if ops else []

    # --- internals -------------------------------------------------------
    def _record(self, method_name: str, *args, **kwargs) -> "LazyDatasetProxy":
        return LazyDatasetProxy(
            self._base_ds, self._ops + [(method_name, args, kwargs)]
        )

    def _is_method_callable_on_dataset(self, name: str) -> bool:
        attr = getattr(self._base_ds, name, None)
        return callable(attr)

    # Provide a pretty repr
    def __repr__(self) -> str:
        ops_preview = " -> ".join(
            [f"{name}{args}{kwargs}" for name, args, kwargs in self._ops]
        )
        return f"<LazyDatasetProxy base={type(self._base_ds).__name__} ops=[{ops_preview}]>"

    # --- generic attr/method interception --------------------------------
    def __getattr__(self, name: str) -> Any:
        """
        If `name` is a callable/ method on Dataset, return a wrapper that records the call.
        Otherwise (non-callable attribute, like .coords, .attrs, etc.) we compute and return
        the attribute from the realized dataset.
        """
        # protect internal attrs
        if name.startswith("_"):
            raise AttributeError(name)

        # methods: return a callable that records as an op
        if self._is_method_callable_on_dataset(name):

            @wraps(getattr(self._base_ds, name))
            def _method(*args, **kwargs):
                return self._record(name, *args, **kwargs)

            return _method

        # non-callable (properties): currently evaluate eagerly and return the attribute
        # (tradeoff: simpler; could be lazily proxied in future)
        realized = self.compute()
        return getattr(realized, name)

    # array-like indexing (ds["var"], ds[...]) -> record __getitem__
    def __getitem__(self, key):
        return self._record("__getitem__", key)

    # useful to allow function-style wrapping e.g. ds.lazy_query(...) if desired
    def __call__(self, *args, **kwargs):
        # we interpret calling the proxy as a no-op but record the call for introspection
        # usually users won't call the accessor directly; this is just safety
        return self._record("__call__", *args, **kwargs)

    # --- compute + optimization -------------------------------------------
    def compute(self) -> xr.Dataset:
        """
        Apply optimizer then replay ops on base dataset and return the resulting xr.Dataset.
        """
        ops = list(self._ops)
        ops = self._optimize_ops(ops)
        ds = self._base_ds
        for method, args, kwargs in ops:
            if method == "__getitem__":
                ds = ds[args[0]]
            elif method == "__call__":
                # no-op (or could be an accessor-level call)
                continue
            else:
                fn = getattr(ds, method)
                ds = fn(*args, **kwargs)
        return ds

    def _optimize_ops(self, ops: list[Op]) -> list[Op]:
        """
        Small optimizer:
         - merge consecutive isel calls into one
         - merge consecutive sel calls into one (simple dict merge)
         - attempt to push isel before reductions (mean, sum, prod) where possible:
             * If there is an isel after a reduction and the isel picks an index for a
               dimension that the reduction reduced away, the isel must come before the reduction.
               Here we apply a heuristic: move isel before the preceding reduction whenever it
               selects dims that are not in the reduction dims (safe case).
        This is intentionally conservative and simple; it's straightforward to extend.
        """
        # helper to collect consecutive ops of same name
        new_ops: list[Op] = []
        i = 0
        while i < len(ops):
            name, args, kwargs = ops[i]

            # Merge consecutive isel dicts: isel(dim=...) or isel(dict)
            if name == "isel":
                merged = {}

                # handle positional dict arg (isel(dict)) or keyword indexers
                def decode_isel_args(a, kw):
                    if len(a) == 1 and isinstance(a[0], dict):
                        return dict(a[0], **kw)
                    return dict(kw)

                merged.update(decode_isel_args(args, kwargs))
                j = i + 1
                while j < len(ops) and ops[j][0] == "isel":
                    nn, aa, kk = ops[j]
                    merged.update(decode_isel_args(aa, kk))
                    j += 1
                new_ops.append(("isel", (merged,), {}))
                i = j
                continue

            # Merge consecutive sel dicts
            if name == "sel":
                merged = {}

                def decode_sel_args(a, kw):
                    if len(a) == 1 and isinstance(a[0], dict):
                        return dict(a[0], **kw)
                    return dict(kw)

                merged.update(decode_sel_args(args, kwargs))
                j = i + 1
                while j < len(ops) and ops[j][0] == "sel":
                    nn, aa, kk = ops[j]
                    merged.update(decode_sel_args(aa, kk))
                    j += 1
                new_ops.append(("sel", (merged,), {}))
                i = j
                continue

            # Attempt to push isel before reductions: handle simple pattern [reduce, isel] -> [isel, reduce]
            if (
                name in ("mean", "sum", "prod")
                and (i + 1) < len(ops)
                and ops[i + 1][0] == "isel"
            ):
                reduce_name, r_args, r_kwargs = ops[i]
                isel_name, isel_args, isel_kwargs = ops[i + 1]
                # determine dims reduced by reduction (heuristic: looks for dim kw or 'dim' arg)
                red_dims = None
                if "dim" in r_kwargs:
                    red_dims = r_kwargs["dim"]
                elif len(r_args) >= 1:
                    red_dims = r_args[0]
                # normalize to tuple of names
                if red_dims is None:
                    red_dims = ()
                elif isinstance(red_dims, str):
                    red_dims = (red_dims,)
                else:
                    red_dims = tuple(red_dims)

                # determine dims touched by isel (keys of indexer)
                indexer = {}
                if len(isel_args) == 1 and isinstance(isel_args[0], dict):
                    indexer = dict(isel_args[0])
                else:
                    indexer = dict(isel_kwargs)

                isel_dims = tuple(indexer.keys())

                # If the isel touches dims that are NOT removed by the reduction,
                # it's safe (and often beneficial) to move isel before reduction.
                # Conservative rule: require isel_dims subset of dims not in red_dims (always true),
                # but more useful: if isel dims intersect red_dims is empty -> safe to swap.
                if set(isel_dims).isdisjoint(red_dims):
                    # swap order
                    new_ops.append(("isel", (indexer,), {}))
                    new_ops.append((reduce_name, r_args, r_kwargs))
                    i += 2
                    continue

            # default: copy op
            new_ops.append((name, args, kwargs))
            i += 1

        return new_ops

    # convenience: let user call .compute().persist() etc.
    def persist(self):
        """Persist the computed dataset (if dask-backed)"""
        ds = self.compute()
        if hasattr(ds, "persist"):
            return ds.persist()
        return ds

    # allow materializing to a real xr.Dataset via .to_dataset() / .realize()
    def realize(self) -> xr.Dataset:
        return self.compute()


# --- xarray accessor -----------------------------------------------------
@xr.register_dataset_accessor("lazier")
class LazierAccessor:
    """
    xarray accessor that returns a LazyDatasetProxy which records Dataset operations.
    Exposed as `ds.lazier`.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj
        # Return a proxied object that behaves like the dataset lazily.
        # Users will call ds.lazy_query.method(...).other(...)
        self._proxy = LazyDatasetProxy(self._obj)

    def __getattr__(self, name: str):
        return getattr(self._proxy, name)

    def __repr__(self) -> str:
        return repr(self._proxy)

    # expose compute and realize etc directly on accessor too
    def compute(self):
        return self._proxy.compute()

    def realize(self):
        return self._proxy.realize()

    def persist(self):
        return self._proxy.persist()
