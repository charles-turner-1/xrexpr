---
file_format: mystnb
kernelspec:
  name: python3
---

# Rechunking

A `chunk()` call changes no value, only how the data is divided into blocks. That makes
it the easiest operation in the package to move a selection in front of: the selection
cannot see chunking, and the rechunk then has less data to shuffle.

Or rather, it makes it the easiest *sometimes*. `chunk()` is the one operation whose
rewrite depends on **how you spelled its argument**, and this page is about where that
line falls.

The examples below chunk real arrays, so they need `dask` installed. `xrexpr` itself does
not.

```{code-cell} python
import dask  # imported for the failure message: chunking needs it, xrexpr does not
import numpy as np
import xarray as xr

import xrexpr

rng = np.random.default_rng(0)
time = np.datetime64("2020-01-01") + np.arange(1000)

ds = xr.Dataset(
    {"temperature": (("time", "lat", "lon"), rng.random((1000, 4, 6)))},
    coords={"time": time, "lat": np.arange(4), "lon": np.arange(6)},
)
```

## A rechunk moves, and can vanish

The headline case. Selecting a single timestamp leaves the rechunk with nothing to do at
all, so it disappears from the plan entirely:

```{code-cell} python
ds.plan.chunk({"time": 100}).isel(time=0).explain()
```

Selecting a *range* keeps the rechunk, but runs it on the data you asked for rather than
on all of it:

```{code-cell} python
ds.plan.chunk({"time": 100}).isel(time=slice(50, 250)).explain()
```

This ordering also lands on better blocks than the eager one does.
`ds.chunk({"time": 100}).isel(time=slice(50, 250))` cuts across block boundaries and
leaves ragged `(50, 100, 50)` chunks; the rewritten plan selects first and then chunks the
selected data into regular `(100, 100)` ones.

When a rechunk names several dims and the selection drops only one of them, the rechunk
survives with its spec rebuilt from the dims that are left:

```{code-cell} python
ds.plan.chunk({"time": 100, "lat": 2}).isel(time=0).explain()
```

## What a chunk spec is

`chunk()` takes one **spec** per named dim, and the spellings are not interchangeable.
Applied to `ds.chunk({"time": ...})` on a 1000-long, already-100-blocked dim:

| You write | What dask does | `xrexpr` calls it |
|---|---|---|
| `None` | keeps the existing 100-blocks | `NoChange` |
| `-1` | collapses to one 1000-block | `FullDim` |
| `"auto"` | re-picks by its configured byte target | `Auto` |
| `"100MB"` | re-picks to that target | `ByteSize` |
| `100` | uniform 100-blocks | `SingleSize` |
| `(100, 400, 500)` | those blocks exactly | `BlockSeq` |

So `None` and `-1` are **not** two spellings of one behaviour: on an already-chunked dim
one leaves the blocks alone and the other fuses them into a single block. Anything else
is an `OpaqueChunk`: a spelling xarray tolerates but doesn't document, or one it rejects
outright.

```{mermaid}
flowchart TB
    subgraph free["extent-free, a selection crosses"]
        nochange["<b>NoChange</b>"]
        fulldim["<b>FullDim</b>"]
        single["<b>SingleSize</b>"]
    end

    subgraph dep["extent-dependent, a barrier"]
        auto["<b>Auto</b>"]
        bytesize["<b>ByteSize</b>"]
        blockseq["<b>BlockSeq</b>"]
    end

    subgraph esc["unmodelled, a barrier"]
        opaque["<b>OpaqueChunk</b>"]
    end

    raw(["one dim's chunk spec"]) --> q1{"is it <code>None</code>?"}
    q1 -->|yes| nochange
    q1 -->|no| q2{"a string?"}
    q2 -->|"<code>'auto'</code>"| auto
    q2 -->|"anything else"| bytesize
    q2 -->|no| q3{"an integer?<br/>(a <code>bool</code> is not)"}
    q3 -->|"<code>-1</code>"| fulldim
    q3 -->|"1 or more"| single
    q3 -->|"<code>0</code>, or below <code>-1</code>"| opaque
    q3 -->|no| q4{"a non-empty sequence<br/>of integers?"}
    q4 -->|yes| blockseq
    q4 -->|no| opaque

    style free fill:#d8f3dc,stroke:#333,color:#000
    style dep fill:#ffd6d6,stroke:#333,color:#000
    style esc fill:#eeeeee,stroke:#333,color:#000
```

The order of those questions matters in one place: `-1` is tested before "is it a size?",
so it lands on `FullDim` rather than on a `SingleSize` of `-1`. Everything the tree
doesn't recognise becomes an `OpaqueChunk` rather than an error, which is what lets
`xrexpr` record a chain it can't reason about instead of refusing it.

## The line: does dask have to measure the array?

Every spec on the left of the diagram means the same thing on any array it is handed. A
block length of 100 is 100 blocks long whatever the dim's extent; `-1` is one block
whatever its length; `None` keeps whatever is already there. **Nothing a selection does
can invalidate them**, which is exactly what makes moving the selection in front safe:

```{code-cell} python
ds.plan.chunk({"time": -1}).isel(lat=0).explain()
```

```{code-cell} python
ds.plan.chunk({"time": None, "lat": 2}).isel(time=0).explain()
```

Every spec on the right is resolved by dask *measuring the array in front of it*.
`"auto"` and a byte target divide a byte budget by the array's own extent, so a selection
in front changes what they mean; an explicit block sequence must sum to the dimension's
length, so a selection in front leaves a spec that cannot replay at all. All three are
barriers, and the selection stays exactly where you wrote it:

```{code-cell} python
ds.plan.chunk({"time": "auto"}).isel(lat=0).explain()
```

```{code-cell} python
ds.plan.chunk({"time": "10MB"}).isel(lat=0).explain()
```

```{code-cell} python
ds.plan.chunk({"time": (100, 400, 500)}).isel(lat=0).explain()
```

Note that the selection in each of those is on `lat`, a dimension the rechunk never
named. It is refused anyway. If you are spelling out a byte target or a block sequence,
you are already reasoning about chunking against the data as it stands, so `xrexpr`
declines to change the data underneath you.

:::{note}
The `"auto"` case is not just a change of answer. Emptying a dimension in front of an
`"auto"` rechunk can make dask's block-size arithmetic divide by zero, so a chain that
worked eagerly would raise, and the invariant forbids introducing an error. Whether it
raises depends on dask's configured chunk size and on the array's rank, neither of which
a plan can read, so the whole spec form is a barrier rather than some narrower case of
it.
:::

A spec the taxonomy doesn't model barriers for the plainest possible reason. `xrexpr`
has said outright that it cannot reason about the value:

```{code-cell} python
ds.plan.chunk({"time": 0}).isel(lat=0).explain()
```

That one is an error waiting to happen (dask divides by it), and keeping it opaque is
what lets the failure surface at replay in xarray's own words, exactly as it would have
eagerly.

## Two things that aren't about the spec

**The uniform spelling.** `chunk(100)` applies one spec to every dim rather than naming
them, and it classifies the same way its mapping form does, so it crosses or barriers on
the same grounds:

```{code-cell} python
ds.plan.chunk(100).isel(lat=0).explain()
```

```{code-cell} python
ds.plan.chunk("auto").isel(lat=0).explain()
```

**Option keyword arguments.** `chunk()` accepts options such as `token` and
`chunked_array_type` alongside the specs. Rebuilding a rechunk around a moved selection
would have to carry those faithfully, so their presence makes the call a barrier
regardless of how good the specs are, a fact about how the call was written, not about
what any spec means:

```{code-cell} python
ds.plan.chunk({"time": 100}, token="run-42").isel(lat=0).explain()
```

## The values don't change

```{code-cell} python
from xarray.testing import assert_equal

assert_equal(
    ds.plan.chunk({"time": 100}).isel(time=slice(50, 250)).collect(),
    ds.chunk({"time": 100}).isel(time=slice(50, 250)),
)
assert_equal(
    ds.plan.chunk({"time": 100}).isel(time=0).collect(),
    ds.chunk({"time": 100}).isel(time=0),
)
print("both match")
```

Worth being clear about what those cells compare, since it is easy to expect more.
`.collect()` ends in xarray's own `.compute()`, so what comes back is materialised and
carries no chunks at all. For the second chain, the rechunk was optimised away and would
have produced none anyway. Chunking is about the shape of the *work*, not the shape of the
answer, which is why `explain()` is where you look for it.

That is the whole of the user guide. If you want the mechanism rather than the model
(how the plan is represented, what lowering guarantees, and why the rewrite loop
terminates), start with [the pipeline](../internals/pipeline.md).
