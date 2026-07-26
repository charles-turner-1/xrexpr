# W8 — A lowering stage: fuse the builder chains, then optimise

*(A design memo in the tradition of `structural-dispatch.md` — this settles a shape,
records the reasoning, and ends with a staged PR plan. It should be reviewed before any
implementation starts. It **supersedes [`05-grouped-contexts.md`](./05-grouped-contexts.md)**
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
forever. W5 proposed a stateful `_ContextProxy` recorder plus a sub-plan-carrying
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
    ↓  to_opnode       (schema.py)      — unchanged
fluent IR   [Op]
    ↓  to_lower_ir                      — NEW: fuse builder chains, fold the schema
lowered IR  [LoweredOp]
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

## 5. The three fusion nodes

One node per builder kind, each flat: semantic fields the rules match on, plus the
verbatim payload `emit` replays. No node carries another node — the sub-plan shape W5
proposed is rejected in §5.4.

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
both `"time"`. A grouped `mean("lat")` also puts `lat` in `consumes`.

### 5.2 `WindowedReduce`

```python
    kind: Literal["rolling", "coarsen"]
    window: frozendict[Hashable, int]    # {"time": 5}
    reduce: str
```

No `new_dim`: rolling **keeps** its dim at the same size (which makes its dim algebra
simpler than groupby's, not harder — W5 had this backwards); coarsen **divides** it by the
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

### 5.4 Why not W5's `Contextual(inner: Op)`

W5 proposed one variant carrying the context call plus a nested inner `Op`, on the grounds
that it was the IR's first sub-plan node and a deliberate step toward the eventual tree.
Rejected, for three reasons:

- **It preserves a surface fact.** "A context wrapping a call" is how *xarray spells*
  a grouped reduction, not what one *is*. Lowering exists to discard exactly that kind of
  fact; carrying it into the lowered IR is the pass declining to do its job.
- **It makes rules match through a level.** W5's own projection rule reads
  `case Contextual(inner=Reduce()) as ctx` — every rule touching a grouped node inherits a
  nested pattern and a "what if the inner isn't a Reduce" branch. Flat fields cost nothing
  and read directly.
- **The tree-compatibility argument does not hold.** A real n-ary node
  (`merge`/`concat`) carries *plans with their own dataset inputs*. `inner: Op` prefigures
  a sub-plan over the *same* input, which is a different thing; it buys no head start on
  the container promotion `ir.py:21-24` defers.

## 6. The one field that must change — and W6 already specified it

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
#                             |      ^ resolved against a schema with no dims left
#                             ^ records consumes={time, lat, lon}; apply_schema pops all three
```

Lowering cannot repair the second `mean()`, because by then `consumes=frozenset()` is
indistinguishable from an explicit `mean(dim=[])`. The information is gone.

**[`06-small-wins.md`](./06-small-wins.md) §4 already specifies the fix** — a symbolic
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
equal to eager, and for a plan containing no builder chain it is `p` node-for-node.

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

## 9. What this retires from W1

[`01-grouped-barrier.md`](./01-grouped-barrier.md) should still land first, unchanged: it
is ~50 LOC against a silent wrongness, and lowering is a multi-PR refactor. But lowering
retires half of it.

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
  That sub-design is salvaged intact from W5 §4 into
  [`09-schema-sizes.md`](./09-schema-sizes.md) and gates `GroupedReduce`.
- **The weighted pushdown rule is deferred** (§8.1), as is anything else needing a
  data-touching rewrite.
- **The tree is still not here.** Nothing above touches binary ops
  (`merge`/`concat`/`ds1 + ds2`); the container is still a list and the deferral at
  `ir.py:21-24` still stands. This memo deliberately declines to spend that budget.
- **An unhashable-payload wart becomes prominent.** `xr.DataArray.__hash__ is None`, so
  `Opaque(name="weighted", args=(w,))` is **already** unhashable today — a latent conflict
  with `ir.py`'s "hashable and safe to share between plans" claim and with
  `test_ir.py`'s frozen/hashable assertions, which evidently never exercise an array-valued
  arg. Pre-existing and not caused by lowering, but a modelled node routinely carrying
  weights makes it worth closing; filed as a small win. `indexers.Mask` (`indexers.py:135`)
  is the precedent for how the payload can be made hash-safe.
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

### 11.2 Legality is not profitability

Every rule in `_RULES` today conflates two questions: *may* these ops commute, and
*should* they. `pushdown_selects` swaps whenever the dim sets are disjoint, on the
assumption that selecting earlier is always at least as cheap. That assumption is sound
for the current vocabulary and should stay — but it is an assumption, not a theorem, and
lowering makes it more visible: `WindowedReduce` is the first node where pushing a select
in front changes *which elements a window sees at the boundary* rather than merely how
much data flows, so its rule needs the legality question answered on its own terms.

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

1. **`AllDims` for bare reduces** — [`06-small-wins.md`](./06-small-wins.md) §4, promoted
   here. `to_opnode` becomes schema-free; `LazyDatasetProxy` sheds `_schema`.
   Behaviour-identical, and the existing suite is the proof.
2. **`to_lower_ir` + `emit` + pipeline rewiring, with no new node kinds** — an identity
   lowering. The entire existing suite must stay green unmodified apart from the relocated
   `to_opnode` tests. This is the keystone and the least glamorous PR in the roadmap.
3. **`GroupedReduce`** + its fusion pattern + `apply_schema` arm. No rules yet; behaviour
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

**1. §5.4's rejection of `Contextual(inner: Op)`.** W5 settled on a sub-plan-carrying node
and justified it as a deliberate first step toward the eventual tree (`ir.py:21-24`). This
memo reverses that, and the load-bearing claim is narrow: **a real n-ary node
(`merge`/`concat`) carries plans with their own *dataset inputs*, whereas `inner: Op` is a
sub-plan over the *same* input, so it buys no head start on the container promotion.** If
that is wrong — if there is a plausible tree design in which `inner: Op` really is the
first increment — then W5's shape is the better one and §5's three flat nodes should
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

Lower-confidence items, flagged rather than defended: `coarsen`'s size effect depends on a
`boundary` kwarg and may want to be marked unknown rather than modelled (§5.2); `explain()`
switching to the lowered plan is a UX call made on thin evidence (PR 8); and §11.2's
observation that `WindowedReduce` is the first node where a pushed select changes *boundary
semantics* rather than merely data volume deserves checking against xarray's actual rolling
behaviour before the rule in PR 6 is written.

---

The one-line version, in the words the original discussion ended on: **don't optimise the
syntax the user happened to write — translate it into a small semantic language, then
optimise that.** The fluent IR is what was written; the lowered IR is that small semantic
language; `emit` puts the syntax back on the way out. Every prior attempt at the grouped-op
problem failed because it tried to teach the *recorder* to see two calls at once. A pass
that runs afterwards simply can.
