# Concepts: what a plan is

Everything `xrexpr` does follows from one change: `ds.plan.mean("lat")` does not compute
a mean. It writes down that you asked for one.

## The plan

`ds.plan` returns a recording proxy. Each method you chain onto it — `.mean()`,
`.isel()`, `.sel()`, `[["temperature"]]` — is *normalised into an operation* and
appended to a list. Nothing is evaluated, nothing is looked at, and the dataset itself is
untouched. That list is the **plan**.

The plan is deliberately a flat list rather than a tree: an xarray method chain is
linear, each call taking one object and returning one, so a list is what it is. Ops are
identified by their **kind** — `Reduce`, `Select`, `Project`, `Scan`, `Rechunk`,
`Opaque` — not by which xarray method produced them. `mean` and `std` differ in the call
they will replay as, not in how the plan may be rearranged around them.

Two things end a plan:

- **`.collect()`** (or `.compute()`, its synonym) optimises the plan and replays it
  against the real dataset. This is the only step that touches data.
- **`.explain()`** optimises the plan and renders it as text. Nothing runs.

Because `explain()` stops one step short of `collect()`, it shows you exactly the plan
that would have run — see [reading `explain()` output](reading-explain).

## What happens between recording and running

```{mermaid}
flowchart LR
    A["your chain<br/><code>ds.plan.mean(...).isel(...)</code>"] --> B["<b>record</b><br/>one op per call"]
    B --> C["<b>lower</b><br/>what you wrote →<br/>what it means"]
    C --> D["<b>optimise</b><br/>reorder and merge,<br/>to a fixpoint"]
    D --> E["<b>replay</b><br/>run the calls"]
    E --> F["result"]
    style E fill:#f9c74f,stroke:#333,color:#000
```

Only the shaded box touches your data. Everything before it moves metadata around:
dimension names, variable names, sizes.

**Record.** Each call becomes one operation. This step has no memory — it looks at the
call in front of it and nothing else.

**Lower.** Some single operations are spelled as *two* calls in xarray, via a builder
object: `ds.groupby("time.month").mean()`, `ds.rolling(time=5).mean()`. Lowering runs
over the finished plan, where both halves are visible at once, and fuses each pair into
a single operation that knows which dimensions it really consumes and which it creates.
This is why `explain()` prints one line for a `groupby(...).mean()`, not two. A builder
pair it doesn't recognise is demoted to `Opaque` and replayed verbatim.

**Optimise.** A handful of local rewrite rules are applied over and over until the plan
stops changing. Each rule is a single small step — merge two adjacent selections, hop a
selection one place left — and the repetition composes them into the large rewrite: a
selection bubbles past a whole run of reductions and reaches the front.

**Replay.** The optimised operations are turned back into xarray calls and run against
your dataset, in order.

## Structural, not statistical

The optimiser never sees your data. It reasons over a *logical schema* — which
dimensions exist, how big they are, and which variables carry which dimensions — that it
folds forward through the plan as it goes.

Two consequences worth internalising:

1. **There is no cost model.** A rewrite either provably preserves the result or it
   doesn't happen. Nothing here estimates, samples or guesses, so nothing here can guess
   wrong.
2. **Ignorance is safe.** An operation `xrexpr` doesn't model becomes an `Opaque`
   barrier: it replays exactly as written and no rewrite crosses it. The worst realistic
   outcome of an unrecognised call is that you get eager behaviour back.

The precise version of the promise, which the rest of the guide leans on:

> `optimize` preserves the **values** of everything the plan asks for. It may
> additionally avoid an error raised by a computation whose result the plan discards. It
> may never change a value, nor introduce an error.

That middle clause is not a loophole — it's a feature, and
[a chain that stops failing](rewrites.md#it-can-also-make-a-chain-stop-failing) shows why.

## Where the work goes

If you want the mechanism rather than the model — how the plan is represented, what
lowering guarantees, which rules exist and why the loop terminates — that's the
internals section. This guide stays in user terms:

- [Reading `explain()` output](reading-explain) — every annotation decoded.
- [What it rewrites](rewrites) — and what it deliberately won't.
- [Grouped, windowed and weighted chains](grouped-windowed-weighted) — why two calls
  become one operation, and what moves past each kind.
- [Rechunking](rechunking) — the one operation whose rewrite turns on how you spelled
  its argument.
