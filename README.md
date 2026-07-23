# XREXPR: Xarray Expression Rewriter

Imagine you have an xarray dataset that you want to do some analysis on. You might write something like this:

```python
%%timeit
ds.mean(dim="lat").mean(dim="lon").isel(time=0).compute()
```
`193 ms ± 49.6 ms per loop (mean ± std. dev. of 5 runs, 5 loops each)`


However, it would be a lot faster if you instead wrote:

```python
ds.isel(time=0).mean(dim="lat").mean(dim="lon").compute()
```
`925 μs ± 401 μs per loop (mean ± std. dev. of 5 runs, 5 loops each)`

In this instance, just reordering the operations makes a ~200x performance difference. We can see that these two expressions are equivalent, but unfortunately, xarray can't automatically reorder them for us (yet?).

```python
from xarray.testing import assert_equal
assert_equal(
    ds.isel(time=0).mean(dim="lat").mean(dim="lon"),
    ds.mean(dim="lat").mean(dim="lon").isel(time=0),
)

# Does not raise an AssertionError
```

That's where `xrexpr` comes in. Importing it registers a `.plan` accessor on every
`Dataset`. Chain your operations off `ds.plan` exactly as you would off `ds` — but
instead of running eagerly, each call is *recorded*. Calling `.collect()` optimises the
recorded plan (reordering and merging where it's provably safe) and replays it:

```python
import xrexpr  # registers the ``.plan`` accessor

result = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).collect()
```

(`.compute()` is a synonym for `.collect()`, if that's the terminal your fingers reach for.)

`xrexpr` pushes the `isel` in front of the reductions for you, so `.collect()` runs the
fast ordering while you keep writing the readable one. The result is exactly what the
eager chain would have produced:

```python
assert_equal(result, ds.mean(dim="lat").mean(dim="lon").isel(time=0)).compute()
```

## Seeing the rewrite

Use `.explain()` to see the optimised plan without running it:

```python
>>> print(ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).explain())
plan (3 ops):
  1. isel(time=0)
  2. mean(dim='lat')
  3. mean(dim='lon')
```

The `isel` has been hoisted to the front — that's the reorder that buys the speed-up.

Picking variables out of a dataset moves too, so the work is never done on variables you
were about to discard:

```python
>>> print(ds.plan.mean(dim="time")[["temperature"]].explain())
plan (2 ops):
  1. [['temperature']]
  2. mean(dim='time')
```

## How it optimises

`xrexpr` records each call as a normalised operation against a cheap *logical schema*
(dims, sizes and which variables carry which dims — never the array data), then rewrites
the plan to a fixpoint with a few local, result-preserving rules:

- **merge** consecutive `isel`/`sel` selections into a single indexer;
- **push** a selection left past any reduction (`mean`, `sum`, `std`, ...) whose dims it
  doesn't touch, so the reduction scans a smaller array;
- **push** a variable projection (`ds[["tas"]]`, `ds["tas"]`) left past reductions and
  selections, so only the variables you asked for flow through the plan;
- **push** a selection left past a `chunk()`, so the rechunk moves less data.

A projection only moves while the variables it keeps still carry the dimensions the
operations it crosses name. If `elevation` has no `time` dimension, then
`ds.plan.mean(dim="time")[["elevation"]]` is left exactly as written — reordering it
would leave `mean(dim="time")` with no `time` to reduce.

A selection that indexes a dimension a reduction has already removed can never run — for
example `ds.plan.mean(dim="lon").isel(lon=0)` — so `xrexpr` raises
`InvalidExpressionError` at `.collect()` (or `.explain()`) instead of failing deep inside
xarray:

```python
from xrexpr import InvalidExpressionError

try:
    ds.plan.mean(dim="lon").isel(lon=0).collect()
except InvalidExpressionError:
    ...
```

Scans (`cumsum`, `cumprod`, `diff`) are order-sensitive, so a selection on the scanned
dimension is left exactly where you put it.

### Rechunking

A `chunk()` changes no value — only chunk topology — so a selection can always move in
front of one, leaving less data to shuffle. When the selection drops the only dimension
the rechunk named, the rechunk has nothing left to do and disappears:

```python
>>> print(ds.plan.chunk({"time": 100}).isel(time=0).explain())
plan (1 ops):
  1. isel(time=0)
```

Selecting a *range* keeps the rechunk, and lands on better blocks than the eager order
does: `ds.chunk({"time": 100}).isel(time=slice(50, 250))` cuts across block boundaries
for ragged `(50, 100, 50)` chunks, where the rewritten plan rechunks the selected data
into regular `(100, 100)` ones.

One case is left alone: an *explicit block sequence* like `chunk({"time": (100, 400, 500)})`
pins blocks that must sum to the dimension's length, so nothing crosses it — if you're
spelling out block sizes, you're already planning your chunking deliberately.

___

This package is just making its way out of the proof-of-concept stage, so expect some issues. It is also unlikely to support the full range of xarray operations for some time. If it doesn't do anything for you, please open an issue!
