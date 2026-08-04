# xrexpr

**Xarray Expression Rewriter. Write the readable chain; run the fast one.**

:::{warning}
This is a **work in progress**, and I've had Claude (mostly Opus, some Fable) write the
code for me. Because of that, it might look good (IDK), but it is certainly not complete
and has not been drive-tested in any meaningful sense of the word. Claims about
functionality in these docs should be considered probable at best, and aspirational at
worst. Use at your own caution (whilst this warning is still up — I'll get rid of it once
I'm confident in the codebase).

P.S. This is not completely unread AI nonsense. I'm driving the AI pretty closely — but
be warned that when you go this fast, things get missed and/or overlooked.
:::

:::{note}
**This is not an xarray project.** It isn't affiliated with, endorsed by, or supported by
xarray or its maintainers — it just happens to plug into xarray via the accessor API. It
also isn't really a *package* yet, despite looking like one: it's closer to an
LLM-assisted, unusually deep proof of concept that I'm using to find out whether the idea
holds up.
:::

## The problem

Imagine you have an xarray dataset you want to do some analysis on. You might write:

```python
ds.mean(dim="lat").mean(dim="lon").isel(time=0).compute()
```

However, it would be a lot faster if you instead wrote:

```python
ds.isel(time=0).mean(dim="lat").mean(dim="lon").compute()
```

Reordering the operations can make a ~200x difference on a dataset of any size — the
selection throws away all but one time step *before* the reductions have to scan it. The
two expressions are obviously equivalent, but xarray can't reorder them for you (yet?).

## The fix

Importing `xrexpr` registers a `.plan` accessor on every `Dataset` and every
`DataArray`. Chain your operations off `ds.plan` exactly as you would off `ds` — but
instead of running eagerly, each call is *recorded*. Calling `.collect()` optimises the
recorded plan and replays it:

```python
import xrexpr  # registers the ``.plan`` accessor

result = ds.plan.mean(dim="lat").mean(dim="lon").isel(time=0).collect()
```

`xrexpr` pushes the `isel` in front of the reductions for you, so `.collect()` runs the
fast ordering while you keep writing the readable one.

## The whole idea, in three bullets

1. **Nothing runs until you ask.** `ds.plan.<...>` records calls instead of executing
   them; `.collect()` (or `.compute()`) is the only thing that touches data.
2. **Rewrites are structural, not statistical.** Between recording and replaying,
   `xrexpr` looks at *dimensions and variable names only* — never at the arrays — and
   applies rewrites that provably can't change the answer. There's no cost model and no
   guesswork.
3. **When in doubt, it does nothing.** Anything it can't prove safe is left exactly where
   you wrote it, so the worst realistic outcome is that you get the eager behaviour back.

## Where to go next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download` Installation
:link: getting-started/installation
:link-type: doc

What to install, and what's optional.
:::

:::{grid-item-card} {octicon}`rocket` Quickstart
:link: getting-started/quickstart
:link-type: doc

The whole story end-to-end on a synthetic dataset — record, explain, collect. Every
output on the page is produced by running the code.
:::

:::{grid-item-card} {octicon}`book` User guide
:link: guide/concepts
:link-type: doc

What a plan is, how to read `explain()` output, and exactly which rewrites you get —
and which you deliberately don't.
:::

::::

```{toctree}
:hidden:
:caption: Getting started
:maxdepth: 2

getting-started/installation
getting-started/quickstart
```

```{toctree}
:hidden:
:caption: User guide
:maxdepth: 2

guide/concepts
guide/reading-explain
guide/rewrites
```
