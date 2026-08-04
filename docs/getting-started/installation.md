# Installation

```bash
pip install xrexpr
```

Python 3.10 or newer. The only hard dependencies are
[xarray](https://docs.xarray.dev/en/stable/), `frozendict` and `typing_extensions`.

## Optional: dask

`xrexpr` itself is dask-free, and stays that way. Nothing in the package imports it.
You only need [dask](https://docs.dask.org/en/stable/) if your own chains call
`.chunk()`, because replaying a rechunk needs a chunk manager to replay it *with*:

```bash
pip install dask
```

Without dask, everything else works unchanged; the rewrites that reason about chunk
specs simply never come up.

## Checking it took

Importing the package is what registers the accessor, so the import is not unused even
when you never name `xrexpr` again:

```python
import xarray as xr
import xrexpr  # registers the ``.plan`` accessor on Dataset and DataArray

ds = xr.Dataset({"x": ("time", [1.0, 2.0, 3.0])})
print(ds.plan.mean("time").explain())
```

If that prints a two-line plan rather than raising `AttributeError`, you're set. The
[quickstart](quickstart.md) does the same thing on a dataset worth optimising.
