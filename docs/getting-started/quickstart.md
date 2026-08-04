---
file_format: mystnb
kernelspec:
  name: python3
---

# Quickstart

Every output on this page is produced by running the code when the docs are built, so
what you see is what the current version of `xrexpr` actually prints.

## A dataset to optimise

No data file to download — we'll build one, the same way the test suite does: a
`temperature` variable over `(time, lat, lon)`, alongside an `elevation` variable that
is *missing* `time`. That second variable matters later.

```{code-cell} python
import numpy as np
import xarray as xr

import xrexpr  # registers the ``.plan`` accessor on Dataset and DataArray

rng = np.random.default_rng(0)
time = np.arange("2015-01", "2025-01", dtype="datetime64[M]").astype("datetime64[ns]")

ds = xr.Dataset(
    {
        "temperature": (("time", "lat", "lon"), rng.random((time.size, 45, 90))),
        "elevation": (("lat", "lon"), rng.random((45, 90))),
    },
    coords={"time": time, "lat": np.arange(45), "lon": np.arange(90)},
)
ds
```

## The rewrite, in one line

Write the readable ordering — reduce, then select:

```{code-cell} python
plan = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0)
```

Nothing has run. `plan` is a recorded chain, and `.explain()` shows what `.collect()`
*would* run, without running it:

```{code-cell} python
plan.explain()
```

The `isel` has been hoisted to the front — that's the reorder that buys the speed-up,
because both reductions now scan one time step instead of every one in the dataset.

Each line is one operation as `xrexpr` understands it: **what kind** it is, **the calls
it will replay as**, and in brackets **what the calls don't say** — here, which
dimensions each reduction removes. Reading the annotations in full is its own page in
the user guide.

Now run it:

```{code-cell} python
result = plan.collect()
result
```

`.compute()` is a synonym for `.collect()`, if that's the terminal your fingers reach
for.

## It is the same answer

The point of the whole exercise: the optimised plan returns exactly what the eager chain
you wrote would have returned. This cell raises if that's ever untrue, which would fail
the docs build:

```{code-cell} python
from xarray.testing import assert_equal

assert_equal(result, ds.mean(dim="lat").mean(dim="lon").isel(time=0))
```

For the record, on the author's machine the eager orderings differ by roughly 200x —
`193 ms ± 49.6 ms` for reduce-then-select against `925 μs ± 401 μs` for
select-then-reduce. Those numbers are hardware-dependent and are quoted rather than
measured here; the plan text above is the part that can't drift.

## Picking variables moves too

A projection (`ds[["temperature"]]`) is pushed left as well, so work is never done on
variables you were about to discard:

```{code-cell} python
ds.plan.mean(dim="time")[["temperature"]].explain()
```

This is where `elevation` earns its place in the dataset. A projection only moves while
the variables it keeps still carry the dimensions of the operations it crosses — and
`elevation` has no `time`, so reordering *this* chain would leave `mean(dim="time")`
with no `time` to reduce:

```{code-cell} python
ds.plan.mean(dim="time")[["elevation"]].explain()
```

The plan is left exactly as written. When `xrexpr` can't prove a rewrite safe, it does
nothing.

## Builder pairs are one operation

xarray spells some single operations as *two* calls via a builder object —
`groupby(...).mean()`, `rolling(...).mean()`, `weighted(...).mean()`. `xrexpr` fuses
each pair into a single node, which is why `explain()` prints one line for it, and
selections hop in front of those too:

```{code-cell} python
ds.plan.groupby("time.month").mean().isel(lat=0).explain()
```

That's the climatology case: the grouping runs over one latitude instead of over all of
them and then discarding the rest. The `time -> month` annotation is the fact worth knowing
about a grouped reduce — the result is indexed by a *new* `month` dimension and the
original `time` is gone, so a selection on `time` after it means something quite
different from one before it, and `xrexpr` leaves those where you put them.

Same check as before, and it still holds:

```{code-cell} python
assert_equal(
    ds.plan.groupby("time.month").mean().isel(lat=0).collect(),
    ds.groupby("time.month").mean().isel(lat=0),
)
```

## What to read next

- The user guide decodes every `explain()` annotation, states exactly what `xrexpr`
  will and won't rewrite, and covers grouped/windowed/weighted chains and rechunking.
- The internals essays cover how the plan is represented, lowered and optimised.
- The API reference is generated from the source.
