# Lowering

Lowering turns what the user *wrote* into what it *means*. In practice that's one job.
xarray spells some single operations as **two** calls through a builder object, and the
recorder only sees one call at a time:

```python
ds.groupby("time.month").mean()
ds.rolling(time=5).mean()
ds.weighted(w).mean("time")
```

## Why it's a stage, not a rewrite rule

There are two separate reasons, and both have to hold.

**It can't happen during recording.** `to_opnode` handles one call at a time: given a
call, it returns one node, with no idea what comes next. That's right for almost every
method (`ds.mean("lat")` is a `Reduce`, whatever follows), but nothing that sees one call
at a time can see a pair. `to_lower_ir` runs over the *finished* plan, so it can look
ahead.

**It can't be an optimiser rule either.** The rules run repeatedly until none fires, under
a termination measure. Fusion runs **once**, and the rules depend on it having happened:
a rule that matches a fused node can't be right on a plan where that node is still two
nodes. So lowering runs before `optimize`, with its own contract:

> `to_lower_ir` **preserves semantics** (`emit(to_lower_ir(p))` replays to the same
> result as `p`) and is **idempotent** (run on an already-lowered plan, it returns it
> unchanged).

Idempotence comes for free: the output contains no `ContextOpen`, so a second pass has
nothing to match.

## The three fused kinds

`GroupedReduce`, `WindowedReduce` and `WeightedReduce`, one per family of builder. Each
knows which dimensions the operation really consumes and which it creates. That's what
lets the optimiser reason about a `groupby` at all, and why `explain()` prints one line
where you wrote two calls. What they mean for users, and what can move past each, is in
[the guide](../guide/grouped-windowed-weighted.md).

## Emit is what makes fused nodes work

A node that stands for two calls would need a special case in replay, if replay had to
know about it. `emit` avoids that: it turns each lowered node into the calls that
reproduce it. Replay stays a short `getattr` loop, and *any* node can stand for any
number of calls.

`emit` takes no dataset, so code generation can be tested on its own.

## The opaque fallback

Every `ContextOpen` has to go. A pair no fusion rule handles is **demoted to `Opaque`,
opener and closer together**:

```python
ds.plan.rolling_exp(time=5).mean()          # two Opaques
ds.plan.rolling(time=5).construct("window") # two Opaques
```

Demoting *both* halves is the important part. `to_opnode` gave the closer a provisional
type, reading it as a Dataset-level call when it never was one. If that node stayed, then
in `rolling(time=2).mean()` the `mean` would remain a `Reduce` whose bare dim spec means
"every dim", and a following select would be rejected against dimensions the rolling mean
never removed. Demoting the pair is what makes the provisional typing safe: a closer with
an unexpected shape just doesn't fuse, and the worst case is a missed optimisation, never a
wrong result.

Openers that no fused node describes always take this path: `rolling_exp` is an
exponential weighting rather than a fixed window, and `cumulative` is really a scan.

Each `_fuse_*` function can refuse for its own reasons. Refusing is always safe, so every
such condition just narrows what gets fused. None of them is a correctness risk.

## Two facts lowering needs that the calls don't contain

**The base dataset's dim names.** Whether `groupby("region")` groups along a dimension,
or along a coordinate defined on one, depends on the data, not the call. Lowering is
given `dim_names`, a `frozenset` rather than a `SchemaState`, because that's all it needs.
That keeps this module depending only on the IR, and makes the requirement obvious from
the signature.

They're the *base* dims, not the dims at each opener (which would mean folding a schema
through lowering). That shortcut can only lose fusions, never cause wrong ones:

- A dimension created mid-plan and then grouped over
  (`groupby("time.month").mean().groupby("month")`) isn't a base dim, so it refuses to
  fuse and falls back to `Opaque`.
- The reverse, a base dim that's gone by the time of the `groupby`, needs something like
  a `stack` in between. That's `Opaque`, so the plan is already past the optimiser's
  trusted prefix, and the schema there is a guess anyway.

**That contexts are pairs, not runs.** Checked against the pinned xarray: there's no call
that goes from one builder to another. `DatasetGroupBy.__getitem__` selects a *group* and
ends the context, while Rolling, Coarsen and Weighted reject `__getitem__`. So lowering
only matches adjacent pairs, and never has to scan ahead for a closer.
