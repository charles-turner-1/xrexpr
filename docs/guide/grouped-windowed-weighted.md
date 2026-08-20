---
file_format: mystnb
kernelspec:
  name: python3
---

# Grouped, windowed and weighted chains

xarray spells some single operations as **two** calls, via a builder object:

```python
ds.groupby("time.month").mean()
ds.rolling(time=5).mean()
ds.weighted(w).mean("time")
```

A recorder that sees one call at a time has a problem here. `groupby("time.month")` on its
own does not say what will happen to the data. It depends entirely on what arrives next,
and `.mean()`, `.map()` and `.construct()` are three different operations. Guessing is
exactly the sort of thing that makes an optimiser reorder something it shouldn't.

`xrexpr` resolves this by *lowering*: a pass over the finished plan, where both halves of
a pair are visible at once, which fuses each recognised pair into a single operation. That
is why `explain()` prints one line where you made two calls.

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

Two calls in, one operation out:

```{code-cell} python
ds.plan.groupby("time.month").mean().explain()
```

An opener with no closer is not an operation at all, and `xrexpr` says so rather than
inventing one:

```{code-cell} python
ds.plan.groupby("time.month").explain()
```

## The three fused kinds

| You write | `explain()` kind | The annotation |
|---|---|---|
| `groupby(...)`, `groupby_bins(...)`, `resample(...)` + a reduction | `GroupedReduce` | `[old -> new]` |
| `rolling(...)`, `coarsen(...)` + a reduction | `WindowedReduce` | none (it consumes no dim) |
| `weighted(w)` + a reduction | `WeightedReduce` | `[weights over {...}, consumes={...}]` |

Each is one node that knows which dimensions the operation really consumes and which it
creates. Everything the optimiser does with them follows from those two dim sets, so the
three behave quite differently. The annotation is how you tell which case you're in.

## Grouped reduces trade one dimension for another

A grouped reduce **destroys** the dimension it grouped over and **mints** a new one:

```{code-cell} python
ds.plan.groupby("time.month").mean().explain()
```

Selections and projections hop in front of it whenever their dims are disjoint from the
ones it touches. `lat` has nothing to do with `time` or `month`, so the selection moves.
The grouping then runs over one latitude instead of over all of them and discarding
the rest afterwards:

```{code-cell} python
ds.plan.groupby("time.month").mean().isel(lat=0).explain()
```

A selection on the *minted* dimension cannot move, because before the operation that
dimension does not exist:

```{code-cell} python
ds.plan.groupby("time.month").mean().isel(month=0).explain()
```

`groupby_bins` and `resample` lower to the same kind, and the annotation is worth reading
in each case:

```{code-cell} python
ds.plan.groupby_bins("lat", bins=2).mean().isel(lon=0).explain()
```

```{code-cell} python
ds.plan.resample(time="YE").mean().isel(lon=0).explain()
```

:::{note}
`resample` reads `[time -> time]`: it destroys `time` and mints a dimension of the *same
name*, holding one entry per resampled period rather than per original timestamp. The
consequence is easy to miss: `isel(time=0)` after a resample selects the first *year*,
not the first original timestamp, so it is a different operation from the one written
before. `xrexpr` leaves it exactly where you put it:
:::

```{code-cell} python
ds.plan.resample(time="YE").mean().isel(time=0).explain()
```

## Windowed reduces consume nothing

A rolling or coarsen reduction produces a result with the same dimensions it started
with, so there is no `consumes` set and no minted dimension. That is why the line
carries no annotation:

```{code-cell} python
ds.plan.rolling(time=5).mean().isel(lon=0).explain()
```

```{code-cell} python
ds.plan.coarsen(time=4).mean().isel(lon=0).explain()
```

That doesn't make everything free to move. A window is defined *along* its dimension, so
a selection on the windowed dimension is order-sensitive in the same way a `cumsum` is:
the value at one timestamp depends on the four before it. Such a selection stays put:

```{code-cell} python
ds.plan.rolling(time=5).mean().isel(time=0).explain()
```

Projections cross freely, since dropping a variable changes nothing about the window:

```{code-cell} python
ds.plan.rolling(time=5).mean()[["temperature"]].explain()
```

## Weighted reduces take projections only

```{code-cell} python
ds.plan.weighted(weights).mean("time")[["temperature"]].explain()
```

Two dim sets in the annotation, because two arrays are involved: the dims the weights
span, and the dims the reduction consumes.

The projection moved. A selection does not:

```{code-cell} python
ds.plan.weighted(weights).mean("time").isel(lat=0).explain()
```

This is a deliberate limit rather than an oversight. Hoisting a selection past a weighted
reduce would mean subsetting the weights array to match it, the first rewrite in the
package that would have to touch data rather than metadata, and so left for later.

The projection that *does* move is worth more than it looks, because a weighted reduce is
stricter than a plain one about the variables it is handed:

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

try:
    stations.weighted(weights).mean("time")[["temperature"]]
except TypeError as err:
    print(f"TypeError: {err}")
```

Eager evaluation weights every variable in the Dataset, `station` included, and falls over
on the strings, before the projection saying it isn't wanted ever runs. The plan hoists
that projection, so the failure never happens:

```{code-cell} python
stations.plan.weighted(weights).mean("time")[["temperature"]].collect()
```

This is the same licence the [invariant](rewrites.md#the-invariant) grants: skip a
failure from a part of the chain that cannot affect the requested values, and change
nothing else.

## Pairs it doesn't recognise

Lowering only fuses what it can describe. Anything else stays two operations, both
`Opaque`, and nothing crosses either of them:

```{code-cell} python
ds.plan.rolling_exp(time=5).mean().isel(lon=0).explain()
```

```{code-cell} python
ds.plan.cumulative("time").sum().isel(lon=0).explain()
```

`rolling_exp` is a weighting rather than a fixed window and `cumulative` is a scan in a
builder's clothes, so neither matches any of the three fused kinds. The same applies when
the *closer* is unmodelled, even under a recognised opener:

```{code-cell} python
ds.plan.rolling(time=5).construct("window").isel(lon=0).explain()
```

The result is the conservative one every time: the calls replay exactly as you wrote them.

## The results are the eager results

```{code-cell} python
from xarray.testing import assert_equal

assert_equal(
    ds.plan.groupby("time.month").mean().isel(lat=0).collect(),
    ds.groupby("time.month").mean().isel(lat=0),
)
assert_equal(
    ds.plan.weighted(weights).mean("time")[["temperature"]].collect(),
    ds.weighted(weights).mean("time")[["temperature"]],
)
print("both match")
```

Those cells run when the docs are built, so this page cannot claim a rewrite the code
doesn't actually make.

Next: [rechunking](rechunking.md), the one operation whose rewrite depends on *how* you
spelled its argument.
