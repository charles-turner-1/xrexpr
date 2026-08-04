# Lowering

Lowering translates what the user *wrote* into what it *means*. In practice that is one
job: xarray spells some single semantic operations as **two** calls through a builder
object, and the recorder sees calls one at a time.

```python
ds.groupby("time.month").mean()
ds.rolling(time=5).mean()
ds.weighted(w).mean("time")
```

## Why a stage and not a rewrite rule

Two separate arguments, and both have to hold.

**It cannot be part of recording.** `to_opnode` is a *per-call* function: handed one
recorded call, it must return one node with no knowledge of what follows. That is right
for almost every xarray method (`ds.mean("lat")` is a `Reduce` and nothing later can
change that), but no per-call function can see a pair. `to_lower_ir` runs over the
*finished* plan, so it has the lookahead the recorder structurally cannot have.

**It cannot be one of the optimiser's rules either.** Those run to a shared fixpoint
under a strictly-decreasing measure. Fusion runs **once**, and is a *precondition* of the
rules rather than one of them: a rule that matches a fused node cannot be correct on a
plan where that node is still two. So lowering is sequenced before `optimize`, with a
contract of its own:

> `to_lower_ir` is **semantics-preserving** (`emit(to_lower_ir(p))` replays to the same
> result as `p`) and **idempotent**: applied to an already-lowered plan it returns it
> unchanged.

Idempotence isn't incidental here, it's structural: the output contains no `ContextOpen`
at all, so a second pass has nothing left to match.

## The three fused kinds

`GroupedReduce`, `WindowedReduce` and `WeightedReduce`, one per builder kind. Each knows
which dimensions the operation really consumes and which it mints, which is the whole
reason the optimiser can reason about a `groupby` at all, and why `explain()` prints one
line where you made two calls. What the three mean for a user, and what moves past each,
is [the guide's version](../guide/grouped-windowed-weighted.md).

## Emit is what makes a fused node expressible

A node standing for two calls would be a special case in the replay loop, if replay had
to know about it. `emit` removes that: it maps one lowered node to the call sequence that
reproduces it, so replay stays the short `getattr` loop it has always been and *any* node
may stand for any number of calls.

`emit` takes no dataset, which makes codegen unit-testable on its own and keeps verbatim
replay a property of the pipeline rather than an obligation on each variant.

## The opaque fallback is mandatory

Every `ContextOpen` must leave, and a pair no fusion rule claims is **demoted to
`Opaque`, opener and closer together**:

```python
ds.plan.rolling_exp(time=5).mean()          # two Opaques
ds.plan.rolling(time=5).construct("window") # two Opaques
```

Demoting *both* halves is the part worth understanding. `to_opnode` typed the closer
provisionally, as a Dataset-level reading of a call that was never Dataset-level. Leave
that node in place and `rolling(time=2).mean()` keeps a `Reduce` whose bare dim spec
means "every dim", so a following selection would be rejected against dimensions the
rolling mean never removed. Demoting the pair is exactly what makes the provisional
typing safe: a closer recorded under a shape no rule expects simply fails to match, and
the failure mode is pessimisation rather than wrongness.

The same fallback is what the openers no fused node describes rely on **permanently**,
not just until their own rule lands: `rolling_exp` is a weighting rather than a fixed
window, `cumulative` is a scan in a builder's clothes.

Each `_fuse_*` helper may refuse for its own reasons, and every such condition is a
narrowing of what the package claims to understand rather than a correctness risk,
because refusing is always safe.

## Two facts it needs that the calls don't carry

**The base dataset's dim names.** Whether `groupby("region")` groups along a dimension or
along a coordinate defined on one is a fact about the data, not about the call. Lowering
is handed `dim_names`, a `frozenset`, not a `SchemaState`, because that is the whole of
what it needs: the narrower argument keeps this module's dependencies to the IR alone and
makes the requirement legible in the signature.

They are the *base* dims rather than the dims in effect at each opener, which would mean
folding a schema forward through lowering. The gap is one-sided and pessimising: a
dimension minted mid-plan and then grouped over
(`groupby("time.month").mean().groupby("month")`) refuses and falls to the fallback. The
converse case, a name that is a base dim but no longer one by the time of the `groupby`,
needs an intervening `stack` or `rename`, both of which are `Opaque`, so the plan is
already past the optimiser's trusted prefix and the tracked schema is a guess by contract
rather than by this shortcut.

**That contexts are pairs, not runs.** Verified against the pinned xarray: there is no
builder→builder middle call to skip, because `DatasetGroupBy.__getitem__` selects a
*group* and closes the context, while Rolling, Coarsen and Weighted reject `__getitem__`
outright. So lowering matches adjacent pairs only, and does not need to scan forward for
a closer.
