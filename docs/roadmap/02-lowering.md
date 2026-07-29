# W2 — A lowering stage: fuse the builder chains, then optimise

*(A design memo in the tradition of `structural-dispatch.md` — this settles a shape,
records the reasoning, and ends with a staged PR plan. It should be reviewed before any
implementation starts. It **supersedes [`09-grouped-contexts.md`](./09-grouped-contexts.md)**
and reshapes the sequencing in [`00-assessment.md`](./00-assessment.md). Origin: the
compiler-architecture discussion in [`../xrexpr_ir_discussion.md`](../xrexpr_ir_discussion.md)
and [`../xrexpr_ir_discussion_expanded.md`](../xrexpr_ir_discussion_expanded.md).)*

## 1. The problem: `to_opnode` cannot see the future

`to_opnode` (`schema.py:163`) is a *per-call* function. It is handed one recorded call and
must return one `Op`, with no knowledge of what comes next. For almost every xarray method
that is exactly right — `ds.mean("lat")` is a `Reduce` and nothing later can change that.

But xarray spells some single semantic operations as **two** calls, via builder objects:

```python
ds.groupby("time.month").mean()   # DatasetGroupBy  → Dataset
ds.rolling(time=5).mean()         # DatasetRolling  → Dataset
ds.weighted(w).mean("time")       # DatasetWeighted → Dataset
```

Each is one operation wearing two calls. A per-call function must commit to a meaning for
`groupby("time.month")` before `.mean()` exists, and it cannot — so it records `Opaque`,
and then records the `.mean()` as a *Dataset-level* `Reduce` resolved against the Dataset
schema. That is the live correctness bug documented at `accessor.py:127-133` and
dissected in [`01-grouped-barrier.md`](./01-grouped-barrier.md), whose silent failure mode
(a select swapping behind a groupby) is the reason W1 exists.

Both prior answers work *around* the constraint rather than removing it. W1 installs a
downstream barrier: once a context op is seen, everything after it records as `Opaque`
forever. W9 proposed a stateful `_ContextProxy` recorder plus a sub-plan-carrying
`Contextual` node, so that the recorder could hold the context open across two calls.

A pass that runs over the *finished* plan has the lookahead neither of them has. That is
the whole idea.

## 2. What the fluent IR already gets right

The temptation is to conclude that the recorded IR is too clever and should be demoted to
a dumb transcript of calls. It should not. **Where the fluent API is one-to-one with
semantics, the recorded IR is already correct** — `ds.mean()` genuinely *is* a `Reduce`,
`ds.isel(...)` genuinely *is* a `Select`, and `to_opnode` resolving those against a live
schema is the record-time win `schema.py`'s docstring rightly claims.

So this is not a translation into a lower-information language, and the second IR is not a
different vocabulary. It is the **same vocabulary plus the nodes the fluent API cannot
express in one call**, produced by a pass whose only real job is *fusion*. Call the two
levels the **fluent IR** (what was recorded) and the **lowered IR** (what it means).

That framing settles a lot cheaply: `to_opnode` stays where it is and keeps its job;
`ir.py` keeps one set of dataclasses rather than growing two parallel hierarchies; and the
new pass is a few dozen lines of pattern matching rather than a rewrite of the front end.

## 3. The pipeline

```
xarray calls
    ↓  record          (accessor.py)
    ↓  to_opnode       (schema.py)      — gains one arm: typed context openers (§5.5)
fluent IR   [FluentOp]  =  [Op | ContextOpen]
    ↓  to_lower_ir                      — NEW: fuse builder chains, fold the schema
lowered IR  [LoweredOp]  =  [Op | GroupedReduce | WindowedReduce | WeightedReduce]
    ↓  optimize        (optimize.py)    — rules now see fused nodes
lowered IR  [LoweredOp]
    ↓  emit                             — NEW: back to a call sequence
call list   [Call]
    ↓  _replay         (accessor.py)    — unchanged
xr.Dataset
```

Two new stages, both pure functions over lists. Everything else keeps its current shape.

## 4. Why lowering cannot be a rewrite rule

It is tempting to add fusion to `_RULES` (`optimize.py:506`) and be done. It does not fit,
for a stated reason: `optimize`'s termination argument (`optimize.py:78-86`) requires every
rule to strictly decrease `(len(plan), sum of the indices of the Select and Project
nodes)`, and all rules run to a *shared fixpoint*.

A fusion pass has different obligations. It runs **once**, not to a fixpoint — re-running
it on its own output must be a no-op, which is a different contract from "keeps shrinking
a measure". And it is a *precondition* of the rules rather than one of them: a rule that
fires on a `(GroupedReduce, Select)` adjacency cannot be correct on a plan where the
grouped reduce is still two nodes.

So lowering is a **stage**, sequenced before `optimize`, with its own contract:

> `to_lower_ir` is semantics-preserving and idempotent: `emit(to_lower_ir(p))` replays to
> the same result as `p`, and `to_lower_ir` applied to an already-lowered plan returns it
> unchanged.

Both halves are property-testable, and the second is the direct analogue of the existing
`test_optimize_is_idempotent`.

## 5. The fusion nodes

Three of them, one per builder kind, each flat: semantic fields the rules match on, plus
the verbatim payload `emit` replays. No node carries another node — the sub-plan shape W9
proposed is rejected in §5.4. (§5.6 specs a fourth, `GroupedMap`, which is **deferred** and
not part of this workstream's PR plan.)

### 5.1 `GroupedReduce`

```python
@dataclass(frozen=True)
class GroupedReduce:
    kind: Literal["groupby", "groupby_bins", "resample"]
    group_dim: Hashable                  # the dim the grouper consumes ("time")
    new_dim: Hashable | None             # the dim the aggregation mints ("month")
    reduce: str                          # the closing method: mean/sum/std/...
    consumes: frozenset[Hashable]        # dims the closing reduction also removes
    # + the verbatim call headers for emit()
```

`groupby("time.month")` → `group_dim="time"`, `new_dim="month"`. `resample(time="2D")` →
both `"time"`.

**Corrected in review (2026-07, verified against xarray 2026.7.0):** a grouped reduce
over dims that *exclude* the group dim is not an aggregation at all. With monthly `time`
and `lat`, `ds.groupby("time.month").mean("lat")` returns dims `{time: 12}` — `lat`
consumed, `time` **kept**, no `month` minted: a per-group *map*, reassembled along the
original dim. Only when the closing spec is bare, or names the group dim
(`mean()`, `mean("time")`, `mean(["time", "lat"])` — all verified to mint `month`), does
the group-aggregation shape above apply. So the fusion rule (§5.5) fuses **only the
aggregation case**; the map case falls to the opaque fallback in v1. (A later
canonicalisation could lower the map case to a plain `Reduce` — grouping partitions the
group dim, so reducing other dims per group and reassembling is the plain reduction —
but that equivalence deserves its own legality note, not a v1 assumption.)

> **The legality note, written (2026-07, verified against xarray 2026.7.0) — and the
> canonicalisation does not survive it.** Two independent failures, one of which discharges
> the premise itself:
>
> - **Order is not the hazard**, which was the suspicion worth discharging first. With
>   `time.month` over interleaved (`Jan, Feb, Jan, Feb`) and over descending time, the map
>   reassembles in the **original** order.
> - **Coverage is.** "Grouping partitions the group dim" holds only if every point lands in
>   some group. A grouper that drops points returns the group dim *shorter*, and a plain
>   reduction cannot:
>
>   ```
>   ds.groupby_bins("lat", bins=[0.5, 2]).mean("time")  ->  lat: 2   (lat=0 is in no bin)
>   ds.mean("time")                                     ->  lat: 3
>   ```
>
>   A NaN group label does the same (`region = [nan, 1.0, 1.0]` → `lat: 2`). `groupby_bins`
>   is in `GroupedReduce`'s `Literal` and in `ContextOpenName`, so this is in scope rather
>   than hypothetical.
> - **And broadcasting kills it even for a covering grouper.** A map puts `group_dim` on
>   **every** variable, including ones that never carried it — the exact mirror of the
>   aggregation case's minting, and the fact that makes the two shapes siblings rather than
>   one being a degenerate case of a plain reduce:
>
>   ```
>   ds.groupby("time.month").mean("lat")   ->  tas(time, lon)  elevation(time, lon)  flat(time, lon)
>   ds.mean("lat")                         ->  tas(time, lon)  elevation(lon)        flat(lon)
>   ```
>
>   `elevation(lat, lon)` and `flat(lon)` gain `time`. So the equivalence fails on any
>   dataset with a variable lacking the group dim, regardless of grouper.
>
> **So the map case earns its own node, and that node is a `Map`, not a `Reduce`.** Spec'd
> in §5.6; deferred, not scheduled.

### 5.2 `WindowedReduce`

```python
    kind: Literal["rolling", "coarsen"]
    window: frozendict[Hashable, int]    # {"time": 5}
    reduce: str
```

No `new_dim`: rolling **keeps** its dim at the same size (which makes its dim algebra
simpler than groupby's, not harder — W9 had this backwards); coarsen **divides** it by the
window, with the `boundary` kwarg deciding the rounding. Spell the size effect out per
kind in `apply_schema`'s arm, and note that coarsen's size depends on a kwarg, so the
`boundary="exact"`/`"trim"`/`"pad"` cases must be handled or the size marked unknown.

### 5.3 `WeightedReduce`

```python
    reduce: str                          # mean/sum/std/var/quantile/sum_of_weights/...
    consumes: frozenset[Hashable]
```

The simplest of the three, and it carries **no novel dim data at all**:
`ds.weighted(w).mean("time")` consumes `time`, identically to a plain `Reduce`. The
weights stay in the verbatim payload — they are an array, and no rule inspects them.

This makes its justification unusual, and worth stating plainly because it widens a rule
the project has been holding to. `structural-dispatch.md` §7.4 earns a variant by
"genuinely new structural data the optimiser must reason about", and by that test
`WeightedReduce` fails outright. **It is earned by what it must block.** Lowered to a
plain `Reduce` it would be *indistinguishable* from one, so `pushdown_selects`
(`optimize.py:322`) would match it and silently emit a wrong plan — a select hopped in
front of a weighted mean, with the weights left un-subset. The variant exists precisely so
that rule cannot match.

So §7.4's rule widens: **a variant is earned by data the optimiser must reason about, or
by a rewrite it must be kept away from.** The second clause is new, and `WeightedReduce`
is its first instance.

**Corrected in implementation (2026-07, PR 5, verified against xarray 2026.7.0): the claim
above that the node "carries no novel dim data at all" is false, and the §7.4 widening is
therefore not load-bearing for it.** A weighted reduce's `consumes` really is a plain
reduce's — `DatasetWeighted.mean` is `(dim=None, *, skipna, keep_attrs)`, so it takes a dim
and a bare closer means all of them — but the **weights** have a dim effect of their own,
because `dot` broadcasts and aligns:

- a weight dim the dataset **lacks is broadcast in**, onto *every* variable. With
  `w(lat, member)`, `ds.weighted(w).mean("lat")` returns `{time, lon, member}` and
  `elevation(lat, lon)` comes back as `(lon, member)`.
- a weight dim the dataset **shares is aligned**, and a misaligned one *shrinks* it: `w` on
  `lat` labelled `[0, 1]` against a `lat` of `[0, 1, 2]` inner-joins to size 2.

So the node carries `weight_dims: frozenset[Hashable]`, read off `w.dims` at fusion time
(metadata; materialising nothing), and `apply_schema` marks every surviving weight dim
**present with unknown size** — names exact, extents honestly unknown, which is precisely
the answer [`03-schema-sizes.md`](./03-schema-sizes.md) added `int | None` for, and its
first consumer beyond `GroupedReduce`. Sizes are deliberately *not* stored on the node:
they would be a guess about the post-alignment extent.

Two consequences for this memo. §7.4 as originally written already licenses the variant, so
§13 item 2's worry — that a stated design rule was being widened to admit a single node —
is moot: nothing rests on the widening. And the second clause remains worth keeping anyway,
because it is still the reason this node gets **no rules**, which is a separate claim from
why it exists.

A third fact, immaterial to the design but worth knowing before writing any rule for it: a
weighted reduce maps per variable and **raises** where a plain one doesn't —
`ds.weighted(w).mean("time")` raises `ValueError` when any variable lacks `time`, whereas
`ds.mean("time")` quietly leaves such a variable alone.

### 5.4 Why not W9's `Contextual(inner: Op)`

W9 proposed one variant carrying the context call plus a nested inner `Op`, on the grounds
that it was the IR's first sub-plan node and a deliberate step toward the eventual tree.
Rejected, for three reasons:

- **It preserves a surface fact.** "A context wrapping a call" is how *xarray spells*
  a grouped reduction, not what one *is*. Lowering exists to discard exactly that kind of
  fact; carrying it into the lowered IR is the pass declining to do its job.
- **It makes rules match through a level.** W9's own projection rule reads
  `case Contextual(inner=Reduce()) as ctx` — every rule touching a grouped node inherits a
  nested pattern and a "what if the inner isn't a Reduce" branch. Flat fields cost nothing
  and read directly.
- **The tree-compatibility argument does not hold.** A real n-ary node
  (`merge`/`concat`) carries *plans with their own dataset inputs*. `inner: Op` prefigures
  a sub-plan over the *same* input, which is a different thing; it buys no head start on
  the container promotion `ir.py:21-24` defers.

### 5.5 The fusion input, pinned: a typed opener (revised in review, 2026-07)

The memo originally left implicit what the *fluent* IR of a builder chain looks like —
and PR 3's match pattern depends on it. **Pinned: the opener gets its own fluent-level
variant, and fusion is a typed composition rule.** A context-opening call is per-call
decidable — `groupby(...)` is an opener no matter what follows — so `to_opnode` *can*
type it without violating its per-call constraint:

```python
@dataclass(frozen=True)
class ContextOpen:
    """A builder-returning call (fluent IR only; must not survive lowering)."""

    name: Literal["groupby", "groupby_bins", "resample", "rolling",
                  "rolling_exp", "coarsen", "weighted", "cumulative"]
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
```

One variant, not one per builder: the openers carry identical structure (a verbatim
header plus the kind tag), and per-builder differentiation belongs to the *fused* nodes —
five opener variants would each add an empty arm at every `assert_never` site for no
matching structural data (§7.4's own test). The two IR levels then become **two union
aliases over shared dataclasses**:

```python
FluentOp  = Op | ContextOpen
LoweredOp = Op | GroupedReduce | WindowedReduce | WeightedReduce
```

which is the payoff of typing the opener: `optimize` takes `list[LoweredOp]`, so *a
surviving `ContextOpen` is a mypy error, not a convention* — the lowering invariant is
enforced at the type level, the house exhaustiveness discipline extended across IR
levels. (`ir.py` still keeps one set of dataclasses; §2's claim stands.)

**The closer records through `to_opnode`, provisionally.** `.mean()` after a `groupby`
records as `Reduce("mean", consumes=AllDims)` — but fusion treats that node as a
*parse* (a name plus a parsed dim spec), never as Dataset semantics: the pair is fused
before `AllDims` resolution ever sees it (§6), and the fused node's schema arm supplies
the real grouped meaning. This reuses `to_opnode`'s argument parsing instead of
re-implementing `_dim_spec` inside the lowering pass. Untabulated closers
(`first`/`last`/`.map(f)`) record `Opaque` as they always did.

The fusion rules are then ordinary typed matches, with a **mandatory fallback that
closes the type**:

- `(ContextOpen("groupby" | "groupby_bins" | "resample"), Reduce)` → `GroupedReduce`,
  **only when** the reduce's dim spec is bare (`AllDims`) or names the group dim — the
  aggregation case; the within-group map case (§5.1's correction) does not fuse in v1;
- `(ContextOpen("rolling" | "coarsen"), Reduce)` → `WindowedReduce`; and
  `(ContextOpen("weighted"), Reduce)` → `WeightedReduce`, same treatment;
- **fallback:** any `ContextOpen` not consumed by a fusion rule is demoted to
  `Opaque(name, args, kwargs)` *together with its following node* — the pair replays
  verbatim, exactly W1's behaviour. This is what makes the closer's provisional typing
  safe: a closer that records under a wrong kind (or an unknown one) simply fails to
  match and the pair goes opaque — **pessimisation, never wrongness**. It also
  sequences the PRs cleanly: PR 3 types every opener but adds only the groupby-family
  rule, so rolling/coarsen/weighted pairs hit the fallback and behave exactly as under
  W1 until PRs 4–5 add their rules.

**Contexts are pairs, not runs (verified against xarray 2026.7.0).** There is no
builder→builder middle call to skip: `DatasetGroupBy.__getitem__` selects a *group*
(`gb[1]` → the matching Dataset subset — it closes the context; it has not selected a
variable since the API changed), and Rolling/Coarsen/Weighted raise `TypeError` on
`__getitem__`. So the recorder's in-context test is one line — `self._ops and
isinstance(self._ops[-1], ContextOpen)` — and fusion matches adjacent pairs only. Two
consequences: an in-context `__getitem__` must still bypass `_projected_names` (a
hashable key would classify as `Project`, but `gb[1]` is group selection, not
projection — record `Opaque`, as W1 §3 already specified); and if a future xarray
reintroduces variable selection on builders, it surfaces as an unfusable closer → the
fallback → verbatim replay, correct by construction. Hoisting such a middle projection
out of the context would be a genuine rewrite with a legality condition (the grouper
must survive the projection), not something lowering may ignore — future work, noted
here so it is a decision rather than drift.

**Grouper scope, v1: strings only.** Fuse when the grouper names a dim
(`groupby("lat")`) or a component of a dim coordinate (`groupby("time.month")`), from
which `group_dim` reads off directly. Any other grouper — a `DataArray`
(`ds.groupby(da)`), a non-dim coordinate name, a `Grouper` object — has no single
`group_dim` and falls to the fallback in v1; the `DataArray` case also trips the
unhashable-payload wart ([`07-small-wins.md`](./07-small-wins.md) §6), which staying
unfused sidesteps for `groupby` entirely.

> **The implementation does not match this paragraph (found 2026-07, unfixed).** A
> **non-dim coordinate name does fuse**, because `_grouper_dims` reads `args[0]` and
> partitions on `"."` without ever checking the result is a dim. With
> `region` a coordinate on `lat`, `ds.groupby("region").mean()` lowers to a single
> `GroupedReduce(group_dim="region", new_dim="region")`, and the tracked schema comes out
> **wrong**: `{time: 4, lat: 3, region: None}` against an actual `{region: 2, time: 4}` —
> `lat` is the dim the grouping consumed, and the fold says it survived.
>
> Replay is unaffected (`emit` reproduces the same two calls, and equality-vs-eager
> passes), so this is a *tracking* bug, not a data bug — but it is the schema every rule
> downstream reads. The property suite cannot see it: the generators draw groupers from dim
> names and `time.month`/`time.day` only. The fix is a coverage test in `_grouper_dims` —
> return `None` unless the grouper's head names a dim — plus a generator widening to draw
> non-dim coordinate groupers, which is what would have caught it. Needs its own PR.
>
> **A second consumer, since PR 6 (2026-07).** The note above was written when nothing read
> `group_dim` but the schema fold. Select pushdown now does too, so the mis-inferred
> `group_dim` reaches a *rewrite*: with `region` a coordinate on `lat`, `dim_effect`'s
> `GroupedReduce` arm returns `blocks={region}`, `lat` looks disjoint, and
> `ds.plan.groupby("region").mean().isel(lat=0)` optimises to
> `isel(lat=0)` → `groupby("region")` → `mean()`. Still not a data bug — a bare grouped
> `mean()` consumes `lat`, so that select is invalid eagerly whichever order it runs in — but
> the *error* degrades: xarray's `Dimensions {'lat'} do not exist` becomes pandas'
> `Must pass non-zero number of levels/codes`, which is precisely what the rule leaves
> intersecting dims alone in order to avoid.
>
> The unification (§11.1) widens the reach rather than narrowing it: `blocks` and
> `requires` both read `group_dim`, so projection pushdown consults the same wrong value.
> Recorded, not fixed here: the fix is still `_grouper_dims`', and this is a second reason
> for it rather than a new one.

### 5.6 `GroupedMap` — a fourth node, spec'd and deferred (2026-07)

The map case of §5.1 — a grouped closer naming dims that *exclude* the group dim — falls
to the opaque fallback today. That is correct and will stay correct; this section specifies
what modelling it would look like, so the option is a decision rather than drift. **Not
scheduled**, and the trigger is at the end.

**What the fallback costs.** An unfused pair demotes to *two* `Opaque` nodes, and the third
cost is the one that is easy to miss:

1. no rule matches it, so nothing moves across it — the intended barrier behaviour;
2. `apply_schema` models `Opaque` as dim- and variable-preserving, which a map is not
   (it removes the closer's dims), so the tracked schema is a guess from there on;
3. `optimize._trusted_prefix` returns the index of the **first** `Opaque`, so a map pair
   early in a chain disables `pushdown_projections` for *everything downstream of it*, not
   just for itself.

**Why not a plain `Reduce`** — §5.1's legality note. Two independent failures (coverage,
and broadcasting the group dim onto every variable), so the canonicalisation is dead, not
merely conditional.

**Why `GroupedMap` and not `Map`.** `DatasetGroupBy.map(func)` is a real xarray method,
and one that records `Opaque` (§5.5). A node called `Map` would read as that call's node;
it is not, and the two must not be confused. `GroupedMap` also keeps the
`<context><operation>` scheme the other three fusion nodes already use.

**Where it lives: `LoweredOp`, not `Op`.** `Op` is the *recorder's* vocabulary — one node
per fluent call — and a builder pair is two calls, so no recorder can produce this. It
belongs beside `GroupedReduce`/`WindowedReduce`/`WeightedReduce`, which is exactly the
asymmetry `ir.py`'s `LoweredOp` comment already describes.

```python
@dataclass(frozen=True)
class GroupedMap:
    name: Literal["groupby", "groupby_bins", "resample"]
    group_dim: Hashable                  # KEPT, not consumed -- the defining difference
    reduce: str                          # the closing method: mean/sum/std/...
    consumes: frozenset[Hashable]        # dims the closer names; never holds group_dim
    # + the verbatim call headers for emit()
```

No `new_dim` field at all. That is the reason this is a sibling variant rather than
`GroupedReduce` with `new_dim: Hashable | None`: an optional field would make every
consumer branch on it, which is the same dispatch a second variant gives with the
exhaustiveness check included. `consumes` is a plain `frozenset`, never `AllDims` — a bare
closer *is* the aggregation case by definition, so a map's closer always names dims.

**The dim algebra (verified against xarray 2026.7.0), and its two traps:**

- `group_dim` survives **at its original length** — `resample(time="2D").mean("lat")` on a
  4-step `time` returns `time: 4`, not the resampled `2`. Cribbing `GroupedReduce`'s size
  intuition here is the plausible bug, and see the test obligation below for why nothing
  would currently catch it.
- the closer's `consumes` dims are removed, exactly like a plain reduce;
- nothing is minted;
- **every variable comes back carrying `group_dim`**, including ones that never had it:
  `elevation(lat, lon)` and `flat(lon)` both return as `(time, lon)` under
  `groupby("time.month").mean("lat")`. This is the mirror of the aggregation case's
  minting, and in `apply_schema` it needs the existing `minted` set — which drives the
  "add to every variable" pass at the end of the fold — *without* also writing
  `dims[group_dim] = None`, since the size here is known. Those two effects are welded
  together in the `GroupedReduce` and `WeightedReduce` arms; a map needs one and not the
  other.
- unlike a weighted reduce (§5.3), a map does **not** raise when a variable lacks a named
  dim: `flat(lon)` under `.mean("lat")` comes back fine.

**Size, honestly.** `group_dim`'s length is exact for a **covering** grouper — a dim name,
a dim-coordinate component (`time.month`), or `resample`, all of which give every point a
group — and unknown (`None`) for `groupby_bins`, whose bins may exclude points. That is
decidable from the call, which keeps it inside the "read it off the opener" discipline
`_grouper_dims` already follows.

**Rules: none.** It ships as a modelled barrier, on the `WeightedReduce` precedent (§8.1):
the value is the schema arm and the retired `_trusted_prefix` truncation, both of which
arrive without any rule firing. What it *makes* reachable, for a later PR: a select on a
dim disjoint from `{group_dim} | consumes` may hop left, the same shape as every existing
pushdown — and `optimize.dim_effect` is where that decision has to be taken, since its
`assert_never` will not compile against a fourth fused variant until the variant says which
dims it `blocks` and which it `requires`. Ruleless is a choice made *there*, not an
omission — and since the unification (§11.1), it is one choice rather than three, so a
`GroupedMap` cannot pick up a conservative answer from one rule and a wrong one from
another.

> **The cited precedent has since been narrowed (W2 PR 9).** §8.1 now reads "without
> *select* rules": a **projection** hop past a `WeightedReduce` was admitted, because a
> projection discards variables rather than subsetting them and so avoids the obligations
> that keep the select hop out. That changes which rule a `GroupedMap` should expect
> first. The select hop above is still the one the dim algebra makes obvious, but the
> projection hop is the closer analogue and the cheaper one — and it comes with §8.1's
> warning attached, that the guard is about which dims the *projected* variables carry,
> not about the node's dims alone. Neither is scheduled; the trigger below is unchanged.

**Test obligation, before it lands rather than after.** The property suite would not catch
a wrong `group_dim` size. `test_tracked_schema_agrees_with_evaluation` compares dim
**names** only; `test_sizes_are_tracked_exactly_without_label_slices` never sees builder
chains; and a ruleless node changes nothing observable in the replay, so the equality
property is silent too. Reformulate the size property first — assert exactness wherever
the schema claims to know, per [`07-small-wins.md`](./07-small-wins.md) §7. The generator
side is already done: `_closer_dims` draws the map shape for both grouped kinds.

**Trigger.** Not "when someone has time". Either a real chain where a map pair sits early
enough to cost the whole plan its projection pushdown (cost 3 above, the only one that
compounds), or the first rule that would want to move across one. Until then the fallback
is correct, and correct-but-pessimised is the position this memo takes everywhere else.

## 6. The one field that must change — and W7 already specified it

For lowering to own semantic resolution it must own the **schema fold**, and today there
are two: `_record` folds `apply_schema` forward as it records (`accessor.py:96`), and
`_schemas` folds it again from the base at optimise time (`optimize.py:98`). They agree
only by doing the same thing to the same list.

Exactly one thing stops `to_opnode` being schema-free: `_reduce_dims`' bare-reduction case
(`schema.py:265-266`), which expands a dim-less `mean()` to "every dim in the schema right
now". `_select_indexer`, `_chunk_spec`, `_projected_names` and the `OP_TABLE` dispatch all
need no schema at all.

Eager expansion is also actively wrong once a builder chain is in play, because the bogus
`consumes` poisons the record-time fold for **every op after it**:

```python
ds.plan.groupby("time.month").mean().mean()
#                             |      ^ under eager expansion: resolved against a stale fold
#                             |        that still says {time, lat, lon} when the truth is
#                             |        {month, lat, lon}
#                             ^ the closer — records Reduce(consumes=AllDims), but that
#                               typing is provisional (§5.5): fusion consumes the pair
#                               before AllDims resolution ever reads it
```

Lowering cannot repair the second `mean()`, because an eagerly-expanded
`consumes={time, lat, lon}` is indistinguishable from an explicit
`mean(dim=["time", "lat", "lon"])`. The information — *this was a bare `mean()`* — is
gone.

**[`07-small-wins.md`](./07-small-wins.md) §4 already specifies the fix** — a symbolic
`consumes: frozenset[Hashable] | AllDims`, where `AllDims` is a singleton sentinel meaning
"every dim at this point, whatever they are" — and it arrived at it from an entirely
independent motivation (the `rename` trust-boundary bug, where record-time expansion
records *stale dim names* and `pushdown_selects` then swaps a select that should raise).
It also lists the touched sites and the `Scan.dims` convergence.

This memo adds a second, independent argument for the same change and **promotes it out of
small-wins**: under lowering it is not an optional sharpening but the mechanism by which
`to_opnode` becomes schema-free. Note the sentinel is better than a bare `None` here —
`None` already means *don't know* in this codebase (`var_dims`, `schema.py:72-84`), whereas
`AllDims` means something definite. Two consequences follow:

- `LazyDatasetProxy` sheds `_schema` entirely and `_record` shrinks to appending a node.
  The double fold collapses to one, owned by `to_lower_ir`.
- `AllDims` is resolved *during lowering*, against a fold that is correct because the
  builder chains have been fused by the time it matters. Post-lowering, `consumes` is
  always a concrete `frozenset`. Narrow it with a match arm
  (`case Reduce(consumes=frozenset() as c)`) rather than scattered `assert`s, so the
  invariant is enforced where it is read.

## 7. `emit`: a function, not a replay branch

`emit` maps one lowered node to the call sequence that reproduces it —
`GroupedReduce` → `groupby("time.month")` then `mean()`. Make it a pure function
`LoweredOp -> tuple[Call, ...]`, **not** a new branch inside `_replay`. Three reasons:

- `_replay` (`accessor.py:201-209`) stays the four-line `getattr` loop it is today and
  never grows an arm per variant. Per-call verbatim-ness — the property
  `structural-dispatch.md` §7.5 actually defends — is preserved exactly.
- Codegen becomes unit-testable with no dataset in sight.
- It pulls payload reconstruction **out of the rules**. Today `merge_adjacent_selects`
  rebuilds `args` from `indexer` (`optimize.py:169`) and `pushdown_selects_past_rechunks`
  rebuilds the chunk spec (`optimize.py:471-479`) — codegen concerns leaking into
  rewrites, and the reason a rule has to think about the replay channel at all. With
  `emit`, rules touch semantic fields only, which is also the clean-seam condition
  `structural-dispatch-2.md` §5 sets for any eventual port.

The round-trip is a property test: for any recorded plan, `emit(to_lower_ir(p))` replays
equal to eager, and for a plan containing no builder chain the emitted calls are exactly
`p`'s call headers, node-for-node (`emit` returns `[Call]`, not `[Op]`, so the equality
is between call headers, not nodes).

## 8. The rewrites this unlocks

**Select pushdown past a fused reduce.** Fires on `(GroupedReduce | WindowedReduce,
Select)`. The select hops left when its dims are disjoint from all of `{group_dim}`,
`{new_dim}` (or `window`'s keys) and `consumes`. `ds.plan.groupby("time.month").mean()
.isel(lat=0)` runs the grouping over one latitude instead of all of them — the canonical
climatology win, and the pattern this project exists for. Intersecting → **leave, never
raise**: selecting on `month` after the aggregation is perfectly valid and merely
immovable, so this follows the scan discipline, not the reduce one.

**Projection pushdown.** A new arm in `pushdown_projections`' `match crossed`
(`optimize.py:413-419`): `needed = consumes | {group_dim}` for a grouped node, checked
against the entering schema like every other arm.

Both read flat semantic fields only.

### 8.1 Why `WeightedReduce` ships without rules

The dim algebra would permit the hop; the **weights** are what don't.
`ds.weighted(w).mean("time").isel(lat=0)` is equivalent to selecting first only if `w` is
subset alongside — `w.isel(lat=0)` when `w` carries `lat`, untouched when it doesn't. That
would be the **first rewrite in this package that transforms an array rather than
reordering metadata**, and it carries obligations no existing rule has: reading `w.dims`
at optimise time (the optimiser currently touches no xarray object at all), deciding what
a dask-backed `w` means for a pass that runs before any compute, and handling a `w` whose
dims only partly overlap the select's.

Each is tractable; none is free. So model the node now and give the rule its own
workstream. Modelling alone already pays: `consumes` becomes correct (today a bare
`.mean()` after `weighted` carries the same wrong dims as after `groupby`), `apply_schema`
gets an exact arm, and — because lowering only needs to opaque the *pair* — ops after a
weighted reduce are modelled again instead of barriered.

> **Amended (2026-07, W2 PR 9): "without rules" now means "without *select* rules".**
> Everything above is about the **select** hop and stands unchanged. A **projection** hop
> is a different proposition, and it is now admitted: a projection *discards* variables
> rather than subsetting them, so `w` is never rewritten and none of the obligations in the
> paragraph above arise. The trigger was
> [`07-small-wins.md`](./07-small-wins.md) §8 retiring the second argument for exclusion —
> that hopping would mask the error a weighted reduce raises for a variable lacking a named
> dim — which §8 reclassifies as the *better* answer, and of which weighted is the sharpest
> instance (it raises where a plain reduce merely wastes effort).
>
> The hop is conditional, and the condition came from probing xarray rather than from the
> dim algebra. A projection drops a dim **coordinate** no surviving variable uses, and the
> weights' treatment of a dim depends on whether that coord exists — inner-join against it
> if so, broadcast a fresh one if not. So the rule requires `consumes | weight_dims` of the
> projected variables; without the `weight_dims` term two chains diverge in *values and
> coords*, not merely in errors. This is the second thing `weight_dims` (§5.3) pays for, and
> the first rule anywhere in the package whose guard is about a **coordinate's existence**
> rather than a dim's. A bare closer needs no such term: it clears every dim, minted weight
> dims included, so nothing survives for a coord to be missing from.

## 9. What this retires from W1

[`01-grouped-barrier.md`](./01-grouped-barrier.md) should still land first, unchanged: it
is ~50 LOC against wrong reorders of valid chains, and lowering is a multi-PR refactor.
But lowering retires half of it.

W1's `_CONTEXT_METHODS` table is **reused** — lowering needs the same list of builder
methods, and the recorder still needs it to route attribute lookups for objects that are
no longer Datasets (`DatasetGroupBy.first` does not exist on `Dataset`).

W1's **trailing barrier is retired**. With lookahead, an unfusable pair needs only *itself*
made opaque, not everything downstream. So `ds.plan.groupby("lat").first().mean("time")`
gets its `mean` modelled correctly — a chain W1 barriers permanently, because `first`
closes the context and returns a Dataset just as `mean` does.

## 10. What doesn't transfer (the honesty beat)

- **The lowering stage has no standalone user-visible benefit.** PRs 1–2 below change no
  behaviour whatsoever. They touch `accessor.py`, `schema.py`, `optimize.py` and a good
  fraction of the ~2,300-line test suite (`test_to_opnode.py` becomes a lowering test;
  `test_properties.py::_build_plan` threads the schema exactly as `_record` does today, so
  it changes too). The cost is paid up front and collected in PRs 3–8. That is also
  precisely why lowering must land *before* any new node kind — otherwise
  `GroupedReduce` gets built twice.
- **Unknown group counts still need `int | None` sizes.**
  `groupby("time.month").mean()` mints a dim whose size comes from coordinate *values*.
  That sub-design is salvaged intact from W9 §4 into
  [`03-schema-sizes.md`](./03-schema-sizes.md) and gates `GroupedReduce`.
- **The weighted pushdown rule is deferred** (§8.1), as is anything else needing a
  data-touching rewrite. *Still true of the **select** rule, which is the one that needs
  the rewrite. The **projection** rule was admitted in W2 PR 9 — see §8.1's amendment —
  precisely because it needs no rewrite at all.*
- **The tree is still not here.** Nothing above touches binary ops
  (`merge`/`concat`/`ds1 + ds2`); the container is still a list and the deferral at
  `ir.py:21-24` still stands. This memo deliberately declines to spend that budget.
  Named specifically because it is the case this project's audience will hit first:
  **grouped arithmetic** — `ds.groupby("time.month") - clim`, the anomaly idiom — is a
  *binary* op on the builder itself. The proxy cannot even record it today (it defines
  no arithmetic dunders), no flat fusion node can express it, and it lands with the
  container promotion, not with lowering. (It is also a second, independent argument
  for §5.4's rejection: `Contextual(inner: Op)` could not have expressed it either.)
- **An unhashable-payload wart becomes prominent.** `xr.DataArray.__hash__ is None`, so
  `Opaque(name="weighted", args=(w,))` is **already** unhashable today — a latent conflict
  with `ir.py`'s "hashable and safe to share between plans" claim and with
  `test_ir.py`'s frozen/hashable assertions, which evidently never exercise an array-valued
  arg. Pre-existing and not caused by lowering, but a modelled node routinely carrying
  weights makes it worth closing; filed as a small win. `indexers.Mask` (`indexers.py:135`)
  is the precedent for how the payload can be made hash-safe.
  *Closed in PR 5 ([`07-small-wins.md`](./07-small-wins.md) §6): the **claim was narrowed**,
  not the payload changed — hashing a `DataArray` means hashing its values, i.e. computing a
  dask array at plan time. `Mask` turns out not to be the precedent it looks like, its
  payload being small and already realised. Also worse than recorded here: `==` between two
  nodes holding distinct-but-equal arrays **raises** rather than returning `False`.*
- **`.reduce()` is still mis-tabulated.** `"reduce"` is in `_REDUCTIONS`
  (`operations.py:31`), but its first positional argument is a *function*, which
  `_reduce_dims` reads as a dim spec. The property generators exclude it
  (`test_properties.py:28-30`) rather than fixing it. Lowering doesn't change this; noting
  it because §6 rewrites the same function.

## 11. Named and deferred

Three ideas from the same discussion that lowering makes reachable but that should not be
smuggled into this workstream. Recorded here so they are decisions rather than drift.

### 11.1 Dims as a type system

The discussion this memo came from floats treating dimensions almost as a type system —
every op declaring which dims it **consumes**, **preserves** and **reorders** — with rules
reasoning from those properties instead of from variant pairs. The pull is real and
visible today: `pushdown_selects` and `pushdown_selects_past_rechunks` are *the same rule*
modulo the dim effect of the node being crossed, and each new fused node adds another
near-duplicate.

Deferred, but worth recording how it would fit, because at a glance it looks like the
protocol-style composition `structural-dispatch.md` §7.3 rejected. It isn't. The variants
stay fat and exhaustively matched; the effect is a **derived** view —
`dim_effect(op) -> DimEffect` computed by one `match` closed with `assert_never`, with the
rules consuming its output. That is the house's own "derive, don't store" discipline
(`Select.consumes`, `Project.single`) applied one level up, at the plan rather than the
node. One dispatch site instead of N.

The trigger for doing it: the third rule that would otherwise be written twice. Until
then it is a note.

> **The trigger is now met** (2026-07, W2 PR 6). `pushdown_selects_past_fused_reduces` is
> the third select-pushdown rule, joining `pushdown_selects` (crosses a `Reduce`) and
> `pushdown_selects_past_rechunks` (crosses a `Rechunk`). All three are the same rule
> modulo two things: *which dims the crossed node involves*, and *what to do when they
> intersect* — raise for a plain reduce, leave for a rechunk or a fused one. That second
> axis is worth noting, because it means the unification needs a `DimEffect` **plus** an
> intersection policy, not the dim sets alone; the sketch above only accounts for the
> first.
>
> Not taken in PR 6 — its scope was the rule — and taken immediately after PR 7, once the
> fourth instance (the projection arm) made the shape unambiguous.
>
> **Done (2026-07).** `optimize.DimEffect` + `dim_effect`, one `match` closed with
> `assert_never`, consumed by both pushdown rules; `pushdown_selects` and
> `pushdown_selects_past_fused_reduces` collapse into one rule, and `_fused_dims` and
> `pushdown_projections`' four-arm `match crossed` both disappear into it. Three partial
> dispatches with three separate silent fallbacks became one exhaustive dispatch, which is
> the win the sketch above claimed.
>
> Two corrections to that sketch, both found by writing it:
>
> - **It needs two dim sets, not one, and the reason is structural rather than
>   incidental.** The rules approach a node from opposite sides. A *select* moving left
>   comes from *after* the node, so it must avoid the minted dim as well (it is entitled to
>   name `month`). A *projection* moving left lands *before* it, where the minted dim does
>   not exist, so requiring it would block every hop. Hence `blocks` and `requires` — and
>   `GroupedReduce` is where they actually differ. Landing PR 7 first is what made this
>   visible; unifying earlier would have produced a single "dims involved" set and then
>   had to break it apart.
> - **Plus the intersection policy**, as noted above — kept as an explicit
>   `on_conflict` field rather than derived. It is tempting to compute it (*"is the blocked
>   dim actually gone?"*), but that would silently change behaviour: a select on a
>   consumed `group_dim` would start raising `InvalidExpressionError` where it currently
>   replays and lets xarray report it, which W2 PR 6 chose deliberately and pinned.
>
> Behaviour-preserving, and checked as such rather than asserted: the whole suite passes
> unmodified, and the merged optimiser was diffed against the pre-refactor one over 8000
> generated plans — same output, and the same plans raising.
>
> `pushdown_selects_past_rechunks` stays outside, which is worth recording as a decision.
> It looked like a third instance from a distance but is a different rule: it has no
> disjointness test at all (a rechunk changes no dim, so a select *always* commutes) and
> instead rewrites the spec it crosses. Folding it in would mean inventing a `DimEffect`
> shape for "always crossable, but adjust me on the way", which no other node wants.

### 11.2 Legality is not profitability

Every rule in `_RULES` today conflates two questions: *may* these ops commute, and
*should* they. `pushdown_selects` swaps whenever the dim sets are disjoint, on the
assumption that selecting earlier is always at least as cheap. That assumption is sound
for the current vocabulary and should stay — but it is an assumption, not a theorem, and
lowering makes it more visible: `WindowedReduce` is the first node where a select pushed
onto the *windowed* dim would change *which elements a window sees at the boundary*
rather than merely how much data flows. (Resolved in review, 2026-07: this concern does
not touch PR 6's rule, which moves only selects on dims **disjoint** from the window
dims — windows run along their own dims independently at each position of the others,
`center`/`min_periods` included, so disjoint selects cannot change window membership.
The boundary question becomes live only if someone later proposes moving an
*intersecting* select, which the rule as specified leaves.)

The position, unchanged from the discussion: **rules decide legality; a cost model would
only choose among rewrites that are already legal.** With a vocabulary this small, good
heuristics capture most of the available benefit, and an explicit cost model earns its
place only when there are competing legal rewrites to choose between (or when the target
becomes the dask graph rather than the xarray call sequence). Not now. But when writing
each new rule, say which question it is answering — the docstrings that do this well
already (`pushdown_selects_past_rechunks` on why a rechunk always commutes but is still
worth moving) are the model.

### 11.3 Fuzzing to *discover* rules, not to prove them

`test_properties.py` already generates plans and asserts optimised-equals-eager. The same
machinery pointed the other way is a rule-discovery tool: generate a plan, enumerate its
nearby permutations, execute each, and report which pairs agree on real data. That
surfaces *candidate* algebraic laws — it proves nothing, since agreement on sampled data
is not commutativity — but it is a cheap way to find rewrites worth reasoning about, and
anything it turns up gets validated the existing way, as a property. Worth doing once the
lowered vocabulary is settled and there are more adjacencies than a person will enumerate
by hand.

## 12. Staged PR plan (after this memo is reviewed)

1. **`AllDims` for bare reduces** — [`07-small-wins.md`](./07-small-wins.md) §4, promoted
   here. `to_opnode` becomes schema-free; `LazyDatasetProxy` sheds `_schema`.
   Behaviour-identical, and the existing suite is the proof.
2. **`to_lower_ir` + `emit` + pipeline rewiring, with no new node kinds** — an identity
   lowering. The entire existing suite must stay green unmodified apart from the relocated
   `to_opnode` tests. This is the keystone and the least glamorous PR in the roadmap.
3. **`GroupedReduce`** + the `ContextOpen` opener, the `FluentOp`/`LoweredOp` aliases,
   the groupby-family fusion rule and the mandatory fallback (§5.5 — every opener is
   typed here, but rolling/coarsen/weighted pairs hit the fallback until PRs 4–5) +
   `apply_schema` arm. No rules yet; behaviour
   still matches W1's (verbatim replay), proven by W1's regression suite staying green plus
   new goldens on the lowered shape. This is where `assert_never` forces every dispatch
   site to be visited.
4. **`WindowedReduce`**, same treatment.
5. **`WeightedReduce`**, same treatment, plus a test asserting no rule fires on it.
6. **Select pushdown** past `GroupedReduce`/`WindowedReduce` + goldens + equality-vs-eager
   + property widening.
7. **The projection-pushdown arm** + the same test treatment.
8. **`explain()` switches to the lowered plan** — it is the more informative artefact and
   matches what Polars shows. Updates the `explain()` string assertions in
   `test_accessor.py`.

Deliberately outside this sequence: the weighted pushdown rule (§8.1) and the dim-effect
unification (§11), each of which needs its own decision first.

> **Both since taken** (2026-07), in the order the decisions actually became answerable:
> the dim-effect unification (§11.1) once PR 7's projection arm made the shape unambiguous,
> then **PR 9 — projection pushdown past `WeightedReduce`** (§8.1's amendment above), which
> the unification reduced to one `dim_effect` arm. The *select* half of the weighted
> pushdown rule remains outside the sequence, undecided and still needing a data-touching
> rewrite. PR 8 (`explain()` on the lowered plan) is the remaining item.

Each PR ships green under:

```
pixi run python -m pytest tests -q
pixi run mypy
pixi run python -m ruff check src tests
```

## 13. Review notes: where this is most likely wrong

This memo overturns settled decisions and widens a stated design rule, so it should be
read adversarially. Three claims carry the weight; if any fails, a good deal of the memo
goes with it. Listed with what would actually change the answer, so a reviewer isn't left
guessing what counts as a refutation.

**1. §5.4's rejection of `Contextual(inner: Op)`.** W9 settled on a sub-plan-carrying node
and justified it as a deliberate first step toward the eventual tree (`ir.py:21-24`). This
memo reverses that, and the load-bearing claim is narrow: **a real n-ary node
(`merge`/`concat`) carries plans with their own *dataset inputs*, whereas `inner: Op` is a
sub-plan over the *same* input, so it buys no head start on the container promotion.** If
that is wrong — if there is a plausible tree design in which `inner: Op` really is the
first increment — then W9's shape is the better one and §5's three flat nodes should
become one nested variant. The other two arguments in §5.4 (surface-fact preservation,
rules matching through a level) are real but would not on their own justify reversing a
settled decision.

**2. The §7.4 widening for `WeightedReduce`.** `structural-dispatch.md` §7.4 earns a
variant by "genuinely new structural data the optimiser must reason about".
`WeightedReduce` fails that test outright — its dim effect is a plain `Reduce`'s — so §5.3
widens the rule to admit variants earned by *what they must block*. Changing a stated
design rule to admit a single node is exactly the move that should draw fire. The
alternative is to keep §7.4 as written and express the blocking some other way (a flag on
`Reduce`, or a rule-side name check against a `_WEIGHTED` set), accepting a less pleasant
mechanism to avoid loosening a rule that has served well. I think the widening is right
and the second clause is genuinely principled rather than a carve-out, but it is a
one-instance generalisation and should be treated as such.

*Resolved in implementation (2026-07, PR 5):* moot. The premise — that `WeightedReduce`
carries no novel structural data — is simply false, and was corrected in §5.3 above: the
weights broadcast and align dims, so the node carries `weight_dims` and §7.4 as written
licenses it. Nothing rests on the widening. The rest of this item stands as a record of the
reasoning, and its conclusions about the *rejected* alternatives are unaffected.

*Review note (2026-07):* there is a reading that keeps §7.4 intact rather than widening
it — treat the discriminant itself as structural data. "This reduce carries a hidden
data dependency" is a fact the optimiser must reason about; its reasoning output happens
to be *don't fire*. On that reading §5.3 is subsumption, not widening. Either framing
licenses the node; what stays rejected either way is the flag-on-`Reduce` alternative
(the every-field-on-every-node shape the sum type exists to prevent) and the rule-side
name check (a shadow table waiting to drift).

**3. §10's cost admission.** PRs 1–2 change **no behaviour at all** and churn a large slice
of a 2,300-line test suite. Every benefit in this memo arrives in PR 3 or later. If a
reviewer concludes that cost is not worth paying, the memo falls entirely — and that is
much better heard now than after PR 1 lands. Two framings worth weighing against each
other: the refactor is justified only if the fused nodes are actually wanted (in which case
building them on the current recorder means building them twice), or W1's barrier is
simply left in place permanently and grouped chains stay correct-but-unoptimised forever.
The second is a legitimate answer, and it is cheap. This memo argues for the first because
the climatology pushdown in §8 is the canonical case the project exists for, but that is a
product judgement, not a technical one.

*Review note (2026-07), one fact that tilts the judgement further than the memo
originally credited:* W1's barrier is not "grouped chains stay unoptimised" — it is
"**everything downstream of the first builder call** stays unoptimised", because the
barrier is trailing and permanent. In climate workflows `groupby`/`resample` sit at the
head of most chains, so the leave-it-forever alternative forfeits optimisation for
essentially the whole plan, for exactly the project's core audience.

Lower-confidence items, flagged rather than defended: `coarsen`'s size effect depends on a
`boundary` kwarg and may want to be marked unknown rather than modelled (§5.2), and
`explain()` switching to the lowered plan is a UX call made on thin evidence (PR 8). A
third item originally listed here — whether a pushed select changes `WindowedReduce`'s
*boundary semantics* — was resolved in review (2026-07): it cannot, for the
disjoint-dims-only rule PR 6 actually specifies (§11.2).

---

The one-line version, in the words the original discussion ended on: **don't optimise the
syntax the user happened to write — translate it into a small semantic language, then
optimise that.** The fluent IR is what was written; the lowered IR is that small semantic
language; `emit` puts the syntax back on the way out. Every prior attempt at the grouped-op
problem failed because it tried to teach the *recorder* to see two calls at once. A pass
that runs afterwards simply can.
