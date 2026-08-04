---
file_format: mystnb
kernelspec:
  name: python3
---

# What it rewrites

Five rewrites, all of them local, all of them decided from dimension and variable names
alone. Every plan below is produced by running the code.

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
```

## The table

| You write | It runs | Why it's a win |
|---|---|---|
| `.mean("lat").isel(time=0)` | `.isel(time=0).mean("lat")` | the reduction scans a smaller array |
| `.isel(time=slice(0, 10)).isel(lat=0)` | one combined `isel` | one indexing pass, not two |
| `.mean("time")[["tas"]]` | `[["tas"]].mean("time")` | never reduce a variable you're about to drop |
| `.groupby("time.month").mean().isel(lat=0)` | `.isel(lat=0).groupby("time.month").mean()` | group one latitude, not all of them |
| `.chunk({"time": 100}).isel(time=0)` | `.isel(time=0)` | the rechunk had nothing left to do |

### Selections move left

Past any reduction whose dims the selection doesn't touch:

```{code-cell} python
ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).explain()
```

### Adjacent selections merge

Into a single indexer, *composing* rather than overwriting when both name the same dim:

```{code-cell} python
ds.plan.isel(time=slice(0, 10)).isel(lat=0).explain()
```

### Projections move left

So only the variables you asked for flow through the plan:

```{code-cell} python
ds.plan.mean(dim="time")[["temperature"]].explain()
```

A projection only moves while the variables it keeps still carry the dimensions of the
operations it crosses. `elevation` has no `time`, so this one stays exactly where it was
written — moving it would leave `mean(dim="time")` with no `time` to reduce:

```{code-cell} python
ds.plan.mean(dim="time")[["elevation"]].explain()
```

### Selections cross fused reduces

Grouped and windowed reduces included, whenever the dims are disjoint:

```{code-cell} python
ds.plan.groupby("time.month").mean().isel(lat=0).explain()
```

### Selections cross rechunks

A `chunk()` changes no value, only chunk topology, so a selection can always move in
front of one — and when the selection drops the only dimension the rechunk named, the
rechunk has nothing left to do and disappears entirely:

```{code-cell} python
ds.plan.chunk({"time": 100}).isel(time=0).explain()
```

Not every chunk spec may be crossed, though — an explicit block sequence pins blocks that
must sum to the dimension's length, so nothing moves past one.
[Rechunking](rechunking) covers where that line falls.

## What it deliberately won't touch

**Order-sensitive operations.** A selection never hops over `cumsum`/`cumprod`/`diff` on
the scanned dimension — the answer depends on how much of the dimension came before:

```{code-cell} python
ds.plan.cumsum("time").isel(time=0).explain()
```

**Selections on a dimension an operation created.** `isel(month=0)` after a
`groupby("time.month")` is perfectly valid — it just can't move, because `month` doesn't
exist until the grouped reduce mints it:

```{code-cell} python
ds.plan.groupby("time.month").mean().isel(month=0).explain()
```

**Anything it doesn't recognise.** An untabulated call (`fillna`, `astype`, ...) is a
barrier: it replays verbatim, and rewrites don't cross it.

```{code-cell} python
ds.plan.fillna(0).isel(time=0).explain()
```

## It catches one class of mistake early

A selection that indexes a dimension a reduction has already removed can never run. Since
the optimiser folds the schema forward anyway, it can say so at `.explain()` or
`.collect()` time, rather than letting the chain fail somewhere deep inside xarray:

```{code-cell} python
from xrexpr import InvalidExpressionError

try:
    ds.plan.mean(dim="lon").isel(lon=0).collect()
except InvalidExpressionError as err:
    print(f"InvalidExpressionError: {err}")
```

## It can also make a chain stop failing

`xrexpr` computes only what your chain actually asks for, and that occasionally means
*not* walking into an error eager evaluation walks straight into. Here is a dataset with a
string-valued variable — station names, which have no standard deviation:

```{code-cell} python
stations = xr.Dataset(
    {
        "temperature": (("time", "lat", "lon"), rng.random((time.size, 4, 6))),
        "station": (
            ("lat", "lon"),
            np.array([[f"s{i}{j}" for j in range(6)] for i in range(4)]),
        ),
    },
    coords={"time": time, "lat": np.arange(4), "lon": np.arange(6)},
)
```

Eager evaluation computes the standard deviation of `station` — purely because it happens
to be in the Dataset — and falls over doing it, *before* the projection that says outright
it isn't wanted ever gets a chance to run:

```{code-cell} python
try:
    stations.std("time")[["temperature"]]
except TypeError as err:
    print(f"TypeError: {err}")
```

The plan pushes the projection in front of the reduction, so `station` is dropped before
the reduction runs and the failure never happens:

```{code-cell} python
stations.plan.std("time")[["temperature"]].explain()
```

```{code-cell} python
stations.plan.std("time")[["temperature"]].collect()
```

`weighted` chains get the same treatment, and there the eager failure is even easier to
hit — see
[weighted reduces](grouped-windowed-weighted.md#weighted-reduces-take-projections-only).

## The invariant

This is not the optimiser playing fast and loose. Stated precisely:

:::{important}
`optimize` preserves the **values** of everything the plan asks for. It may additionally
avoid an error raised by a computation whose result the plan discards. It may never
change a value, nor introduce an error.
:::

The middle clause is exactly the licence the section above needs, and it is bounded:
`xrexpr` may skip a computation whose result you discard, and it may not do anything
else. It cannot turn a working chain into a failing one, and it cannot change a number.

Two of the rewrites above deserve a page of their own, because what moves past them is
less obvious than "do the dims overlap?":
[grouped, windowed and weighted chains](grouped-windowed-weighted) and
[rechunking](rechunking).
