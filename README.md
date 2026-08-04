# xrexpr

[![Documentation Status](https://readthedocs.org/projects/xrexpr/badge/?version=latest)](https://xrexpr.readthedocs.io/en/latest/)

> [!WARNING]
> This is a **work in progress**, and I've had Claude (mostly Opus, some Fable) write the code for me. Because of that, it might look good (IDK),
> but it is certainly not complete or and has not been drive-tested in any meaningful sense of the word. Claims about functionality
> in this README should be considered probable at best, and aspirational at worst.
> Use at your own caution (whilst this warning is still up. I'll get rid of it once I'm confident in the codebase).
> P.S - This is not completely unread AI nonsense. I'm driving the AI pretty closely - but be warned that when you go this fast, things
> get missed and/or overlooked.

> [!NOTE]
> **This is not an xarray project.** It isn't affiliated with, endorsed by, or supported by xarray or its
> maintainers — it just happens to plug into xarray via the accessor API. It also isn't really a *package* yet,
> despite looking like one: it's closer to an LLM-assisted, unusually deep proof of concept that I'm using to
> find out whether the idea holds up.

**XREXPR: Xarray Expression Rewriter.** Write the readable chain; run the fast one.

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
`Dataset` and every `DataArray`. Chain your operations off `ds.plan` exactly as you
would off `ds` (and off `da.plan` exactly as you would off `da`) — but
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
  1. Select  isel(time=0)
  2. Reduce  mean(dim='lat')  [consumes={lat}]
  3. Reduce  mean(dim='lon')  [consumes={lon}]
```

The `isel` has been hoisted to the front — that's the reorder that buys the speed-up.

Each line is one operation as `xrexpr` understands it: **what kind** it is, **the calls it
will replay as**, and in brackets **what the calls don't say** — here, which dimensions
each reduction removes. A bare `.mean()` shows `consumes=every dim`, and anything `xrexpr`
does not model shows as `Opaque  ...  [not modelled -- nothing crosses it]`, which is where
to look when a rewrite you expected didn't happen.

Picking variables out of a dataset moves too, so the work is never done on variables you
were about to discard:

```python
>>> print(ds.plan.mean(dim="time")[["temperature"]].explain())
plan (2 ops):
  1. Project  [['temperature']]
  2. Reduce  mean(dim='time')  [consumes={time}]
```

Builder pairs like `groupby(...).mean()` are *one* operation, and selections move in front
of them as well — the climatology case, where the grouping runs over one latitude instead
of over all of them and then discarding the rest:

```python
>>> print(ds.plan.groupby("time.month").mean().isel(lat=0).explain())
plan (2 ops):
  1. Select  isel(lat=0)
  2. GroupedReduce  groupby('time.month').mean()  [time -> month]
```

`time -> month` is the fact worth knowing about a grouped reduce: the result is indexed by
a *new* `month` dimension and the original `time` is gone, so a selection on `time` after
it means something quite different from one before it — and `xrexpr` leaves those where you
put them.

## Installing

```bash
pip install xrexpr
```

The only hard dependencies are `xarray`, `frozendict` and `typing_extensions`. Python 3.10+.

## The whole idea, in three bullets

1. **Nothing runs until you ask.** `ds.plan.<...>` records calls instead of executing
   them; `.collect()` (or `.compute()`) is the only thing that touches data.
2. **Rewrites are structural, not statistical.** Between recording and replaying,
   `xrexpr` looks at *dimensions and variable names only* — never at the arrays — and
   applies rewrites that provably can't change the answer. There's no cost model and no
   guesswork.
3. **When in doubt, it does nothing.** Anything it can't prove safe is left exactly
   where you wrote it, so the worst realistic outcome is that you get the eager
   behaviour back.

## What it rewrites today

| You write | It runs | Why it's a win |
|---|---|---|
| `.mean("lat").isel(time=0)` | `.isel(time=0).mean("lat")` | the reduction scans a smaller array |
| `.isel(time=slice(0, 10)).isel(lat=0)` | one combined `isel` | one indexing pass, not two |
| `.mean("time")[["tas"]]` | `[["tas"]].mean("time")` | never reduce a variable you're about to drop |
| `.groupby("time.month").mean().isel(lat=0)` | `.isel(lat=0).groupby("time.month").mean()` | group one latitude, not all of them |
| `.chunk({"time": 100}).isel(time=0)` | `.isel(time=0)` | the rechunk had nothing left to do |

And what it deliberately *won't* touch:

- **Order-sensitive ops.** A selection never hops over `cumsum`/`cumprod`/`diff` on the
  scanned dimension.
- **Selections on a dimension an operation created.** `isel(month=0)` after a
  `groupby("time.month")` is perfectly valid — it just can't move.
- **Anything it doesn't recognise.** An untabulated call (`fillna`, `astype`, ...) is a
  barrier: it replays verbatim, and rewrites don't cross it. `explain()` labels these
  `Opaque`.

It also catches one class of mistake early. A selection that indexes a dimension a
reduction has already removed can never run, so `xrexpr` says so at `.collect()` (or
`.explain()`) rather than letting it fail somewhere deep inside xarray:

```python
>>> ds.plan.mean(dim="lon").isel(lon=0).collect()
InvalidExpressionError: isel() indexes ['lon'], which mean() has already reduced away
```

## It can also make a chain stop failing

`xrexpr` computes only what your chain actually asks for, and that occasionally means
*not* walking into an error eager evaluation walks straight into:

```python
ds  # temperature(time, lat, lon) float, and station(lat, lon) -- strings, no time

ds.std("time")[["temperature"]]                  # TypeError, raised by `station`
ds.plan.std("time")[["temperature"]].collect()   # succeeds
```

The projection says outright that `station` isn't wanted. Eager computes its standard
deviation anyway — purely because it happens to be in the Dataset — and falls over doing
it, because numpy has no standard deviation for strings. The plan drops `station` before
the reduction runs, so the failure never happens. `weighted` chains get the same
treatment, and there the eager failure is even easier to hit: a weighted reduce *refuses*
a variable lacking the reduced dim, where a plain `.mean("time")` merely wastes effort
on it.

This isn't the optimiser playing fast and loose. The invariant, stated precisely:

> `optimize` preserves the **values** of everything the plan asks for. It may additionally
> avoid an error raised by a computation whose result the plan discards. It may never
> change a value, nor introduce an error.

## Under the hood

The full documentation is at
**[xrexpr.readthedocs.io](https://xrexpr.readthedocs.io/en/latest/)**. The
[user guide](https://xrexpr.readthedocs.io/en/latest/guide/concepts.html) covers what a
plan is, [how to read `explain()` output](https://xrexpr.readthedocs.io/en/latest/guide/reading-explain.html),
[exactly which rewrites you get](https://xrexpr.readthedocs.io/en/latest/guide/rewrites.html)
and which you deliberately don't, plus the two chains that need their own page —
[grouped, windowed and weighted reduces](https://xrexpr.readthedocs.io/en/latest/guide/grouped-windowed-weighted.html)
and [rechunking](https://xrexpr.readthedocs.io/en/latest/guide/rechunking.html). Every
plan printed on those pages is produced by running the code at build time, so none of it
can drift from what the package actually does.

If you want the mechanism rather than the behaviour, the
[internals](https://xrexpr.readthedocs.io/en/latest/internals/pipeline.html) section has
the five pipeline stages and their contracts, the IR, the rule catalogue and why the
fixpoint terminates. The
[API reference](https://xrexpr.readthedocs.io/en/latest/api/index.html) is generated from
the source.

The arguments behind the design are in [`planning/`](planning/), and what comes next is in
[`planning/roadmap/`](planning/roadmap/) — start with
[`00-assessment.md`](planning/roadmap/00-assessment.md), which states where the codebase
stands and what is still missing.

## Status

Early. The core invariant — `ds.plan.<chain>.collect()` equals the eager chain — is
checked by a property-based test suite over generated datasets and generated chains, but
the set of xarray operations it understands is small, and everything outside that set
falls back to running your chain as written.

If it doesn't do anything for you, or does something surprising, please
[open an issue](https://github.com/charles-turner-1/xrexpr/issues) — the interesting bug
reports are the chains where it *should* have found a rewrite and didn't.
