---
file_format: mystnb
kernelspec:
  name: python3
---

# Reading `explain()` output

`.explain()` runs the whole pipeline except the last step: it records, lowers and
optimises your chain, then renders the result as text instead of replaying it. So what
you are reading is not a summary of your chain. It is the plan that `.collect()` would
run, verbatim.

Every plan on this page is produced by executing the code when the docs are built.

```{code-cell} python
import numpy as np
import xarray as xr

import xrexpr

rng = np.random.default_rng(0)
time = np.arange("2020-01", "2023-01", dtype="datetime64[M]").astype("datetime64[ns]")

ds = xr.Dataset(
    {
        "temperature": (("time", "lat", "lon"), rng.random((time.size, 4, 6))),
        "elevation": (("lat", "lon"), rng.random((4, 6))),
    },
    coords={"time": time, "lat": np.arange(4), "lon": np.arange(6)},
)
weights = xr.DataArray(rng.random(time.size), dims="time", coords={"time": time})
```

## The shape of a line

```{code-cell} python
ds.plan.mean(dim="lat").isel(time=0).explain()
```

The header counts the operations in the *optimised* plan, which is often fewer than the
calls you wrote. Then one line each, in the order they will run:

```text
  2. Reduce  mean(dim='lat')  [consumes={lat}]
  ^  ^       ^                ^
  |  |       |                └─ what the call doesn't say
  |  |       └─ the call that will actually be replayed
  |  └─ the kind of operation
  └─ position in the plan
```

The three parts answer three different questions. **The kind** is what the optimiser
reasons about. Rewrites are decided by kind, never by which xarray method you used.
**The call** is what will run, so a line is always a faithful answer to "what happens
next". **The bracketed annotation** is the interesting part: it states the facts the
call text leaves implicit, which are exactly the facts the rewrite rules key on.

## The kinds

```{code-cell} python
plans = {
    "Reduce": ds.plan.mean(dim="lat"),
    "Select": ds.plan.isel(time=0),
    "Project": ds.plan[["temperature"]],
    "Scan": ds.plan.cumsum("time"),
    "GroupedReduce": ds.plan.groupby("time.month").mean(),
    "WindowedReduce": ds.plan.rolling(time=3).mean(),
    "WeightedReduce": ds.plan.weighted(weights).mean("time"),
    "Rechunk": ds.plan.chunk({"time": 4}),
    "Opaque": ds.plan.fillna(0),
}
for kind, plan in plans.items():
    print(plan.explain().splitlines()[1])
```

| kind | what it is | moves? |
|---|---|---|
| `Reduce` | a reduction over named dims (`mean`, `sum`, `std`, ...) | selections and projections hop in front of it |
| `Select` | `isel`/`sel` | moves left; merges with an adjacent `Select` |
| `Project` | picking variables (`ds[["tas"]]`, `ds["tas"]`) | moves left; merges with an adjacent `Project` |
| `Scan` | order-sensitive (`cumsum`, `cumprod`, `diff`) | nothing crosses it on the scanned dim |
| `GroupedReduce` | a fused `groupby(...).<reduction>()` | selections and projections hop in front, dims permitting |
| `WindowedReduce` | a fused `rolling(...).<reduction>()` | same |
| `WeightedReduce` | a fused `weighted(...).<reduction>()` | projections only, see below |
| `Rechunk` | `chunk(...)` | selections always hop in front |
| `Opaque` | anything unmodelled | a barrier: nothing crosses it |

## The annotations, one at a time

### `[consumes={...}]`: which dims a reduction removes

```{code-cell} python
ds.plan.sum(dim=["lat", "lon"]).explain()
```

The dims listed are gone from everything downstream. That is what lets a selection be
hoisted past a reduction, but only if it touches *none* of them, since a selection on a
consumed dim could never have run after it.

### `consumes=every dim`: a bare reduction

```{code-cell} python
ds.plan.mean().explain()
```

A bare `.mean()` names no dims, so it consumes all of them, whatever they turn out to be.
This is a *symbolic* answer, resolved against the real schema during optimisation. That
is why the annotation says "every dim" rather than listing them.

### `[time -> month]`: a dimension traded for a new one

```{code-cell} python
ds.plan.groupby("time.month").mean().explain()
```

The single most important annotation to read carefully. A grouped reduce **destroys**
`time` and **mints** `month`. So `isel(month=0)` written after this operation cannot move
in front of it. Before it, `month` does not exist yet. `xrexpr` leaves such selections
exactly where you put them, and the annotation is how you see why.

Note also that this line is one operation, not two, even though you made two calls: the
pair was fused during lowering.

### `[weights over {...}, consumes={...}]`: a weighted reduce

```{code-cell} python
ds.plan.weighted(weights).mean("time").explain()
```

Two dim sets, because two arrays are involved. Weighted reduces accept **projections**
but not selections: hoisting a selection past one would mean subsetting the weights array
to match, which would be the first rewrite in the package to touch data rather than
metadata. It's deliberately left for later.

### `[not modelled -- nothing crosses it]`: an `Opaque` barrier

```{code-cell} python
ds.plan.fillna(0).isel(time=0).explain()
```

**This is the line to look for when a rewrite you expected didn't happen.** `fillna` is
not in the table of modelled operations, so `xrexpr` will not reason about what it does
to dims or variables, and therefore refuses to move anything across it. The `isel` stays
put. Without the `Opaque` line in the output you would be left guessing why.

The fix, when you need the rewrite, is to move the unmodelled call out of the middle of
the chain, or to [open an issue](https://github.com/charles-turner-1/xrexpr/issues). A
chain that *should* have been rewritten and wasn't is the interesting bug report.

### No annotation

```{code-cell} python
ds.plan.rolling(time=3).mean().explain()
```

`Scan`, `Rechunk`, `WindowedReduce` and `Select` carry no bracket: their call text already
says everything the optimiser knows. A rolling mean neither removes nor creates a
dimension (a windowed reduce over `time` returns something still indexed by `time`), so
there is no implicit dim fact to state.

## Reading a rewrite

Because `explain()` is the *optimised* plan, comparing it to what you wrote is how you
see the rewrite. Selections merging and moving:

```{code-cell} python
ds.plan.mean(dim="lat").isel(time=slice(0, 10)).isel(lon=0).explain()
```

Three calls became two operations: the two selections merged into one indexer, which then
hopped in front of the reduction. And the ordering is a real claim about what runs, which
you can check:

```{code-cell} python
from xarray.testing import assert_equal

assert_equal(
    ds.plan.mean(dim="lat").isel(time=slice(0, 10)).isel(lon=0).collect(),
    ds.mean(dim="lat").isel(time=slice(0, 10)).isel(lon=0),
)
```

That cell raises if the two ever disagree, which would fail this page's build.

Next: [what it rewrites](rewrites.md), and what it deliberately won't.
