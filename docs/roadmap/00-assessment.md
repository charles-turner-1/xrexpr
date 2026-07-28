# Roadmap: finish the structural encoding, then gate the Rust port on it

*(2026-07. Follows [`structural-dispatch.md`](../structural-dispatch.md),
[`structural-dispatch-2.md`](../structural-dispatch-2.md) and
[`indexer-follow-ups.md`](../indexer-follow-ups.md); supersedes none of them — this is
the plan for what comes after them.)*

## Where the codebase stands

The structural programme both memos argued for has landed on `main`:

- **The op sum type** — `Op = Reduce | Select | Scan | Project | Rechunk | Opaque`
  (`ir.py:198`), fat variants, exhaustive `match` + `assert_never`, derived-property
  discipline throughout (`Select.consumes`, `Project.single`).
- **The indexer value sum type** — `Indexer = Scalar | ForwardSlice | GeneralSlice |
  Positions | Mask | Label` (`indexers.py:185`), `classify` as sole constructor, the
  composition policy closed with `assert_never` (#66–#71, all merged).
- **Four rewrite rules** under a fixpoint with a stated termination measure
  (`optimize.py:78-86`), a schema layer with a named trust boundary
  (`optimize.py:108-118`), and a property-based suite (`test_properties.py`).

The package is ~2,300 lines of strict-typed Python. The IR's node layer and (half of)
its value layer are as principled as the memos asked for.

## What remains, honestly

Three gaps, in decreasing order of urgency:

1. **A live correctness gap: grouped ops.** `groupby`/`resample`/`rolling`/`coarsen`/
   `weighted` record as `Opaque`, but the *next* call (`.mean()`) records as a
   Dataset-level `Reduce` (`accessor.py:127-133`), so `pushdown_selects` reorders
   valid chains — the swap is silent, though replay then fails loudly (verified
   2026-07: builder objects have no `isel`, so the misordered replay raises
   `AttributeError` rather than returning wrong data; see
   [`01-grouped-barrier.md`](./01-grouped-barrier.md)). Documented as "do not chain",
   but valid chains crash with baffling errors rather than being refused.
2. **The chunk-spec half of the value sum type** (`structural-dispatch-2.md` §3) is
   still `Any`: `Rechunk.chunks: frozendict[Hashable, Any]` (`ir.py:174`) and the
   `isinstance` ladder in `_pushable_rechunk` (`optimize.py:486`) — the last one in the
   package.
3. **Opaque-barrier losses.** Every untabulated op is a full barrier. One `fillna` or
   `astype` in a chain stops every pushdown reaching the front. The memo's own trigger
   for reintroducing `Elementwise` ("a rule dispatches on it", memo §7.2/§5) is now met
   by an obvious rule: selects and projections commute with elementwise ops.

## Decisions taken (design review, 2026-07)

1. **Grouped ops: barrier now, model next.** A small correctness PR makes everything
   after a context-returning op record as `Opaque` (no rule can fire wrongly); the full
   structural modelling of grouped contexts follows as its own workstream.
2. **The tree/DAG is in the vision.** The question both memos deferred as "a
   product-vision call" is answered: yes, eventually — but see the revision below; the
   grouped nodes are no longer the vehicle for getting there.
3. **Rust is gated on structure.** There is still no speed case (the optimiser runs
   once per `collect()` on ~10-node plans), and the container shape is about to change.
   The port would transliterate a shape we're about to break. So: finish the chunk
   taxonomy and grouped contexts first, then run a **time-boxed PyO3 spike** — done for
   correctness sharpening and to keep the door open, adopted only if it really is a
   transliteration.

### Revision (2026-07, after the IR-architecture discussion)

Decisions 1 and 3 stand. **Decision 2's implementation is reversed**, and the roadmap
re-folded, by [`02-lowering.md`](./02-lowering.md).

The root cause of the grouped-op problem is narrower than W9 assumed: `to_opnode`
(`schema.py:163`) is a *per-call* function, and xarray spells some single semantic
operations as two calls via builder objects (`DatasetGroupBy`, `DatasetRolling`,
`DatasetWeighted`). No per-call function can model those. W1 works around it with a
downstream barrier; W9 worked around it with a stateful recorder.

A **lowering stage** between recording and optimisation has lookahead over the finished
plan and removes the constraint instead. The pipeline gains two pure stages:

```
xarray calls → to_opnode → fluent IR → to_lower_ir → lowered IR → optimize → emit → replay
```

The fluent IR is kept as-is — where the fluent API is one-to-one with semantics
(`ds.mean()` really is a `Reduce`), the recorded IR is already right. Lowering's only job
is fusing the builder chains into flat semantic nodes (`GroupedReduce`, `WindowedReduce`,
`WeightedReduce`) and owning the single schema fold. W9's `Contextual(inner: Op)` sub-plan
shape is rejected; the tree stays deferred where `ir.py:21-24` left it, rather than being
approached obliquely through grouped ops.

## The workstreams

The numbering is the reading order: workstreams are numbered by phase (below), so the
files read linearly — the keystone right after the safety fix, the superseded memo last
as an appendix.

| # | spec | what | size |
|---|---|---|---|
| W1 | [`01-grouped-barrier.md`](./01-grouped-barrier.md) | opaque-context barrier for accessor-returning ops (correctness) | ~50 LOC, 1 PR |
| W2 | [`02-lowering.md`](./02-lowering.md) | the lowering stage + the three fused builder nodes (design memo) | memo + 8 PRs |
| W3 | [`03-schema-sizes.md`](./03-schema-sizes.md) | `SchemaState` sizes → `int \| None` (salvaged from W9 §4) | 1 PR |
| W4 | [`04-chunk-taxonomy.md`](./04-chunk-taxonomy.md) | the chunk-spec value sum type — closes doc 2 §3 | 1–2 PRs |
| W5 | [`05-elementwise.md`](./05-elementwise.md) | reintroduce `Elementwise` + selects/projections cross it | 2–3 PRs |
| W6 | [`06-scan-dims.md`](./06-scan-dims.md) | `Scan` gains its dims + scan-aware select pushdown | 1 PR |
| W7 | [`07-small-wins.md`](./07-small-wins.md) | independent small rules & cleanups, pick up between workstreams | ~1 PR each |
| W8 | [`08-rust-gate.md`](./08-rust-gate.md) | the Rust gate conditions and the PyO3 spike spec | timeboxed spike |
| ~~W9~~ | [`09-grouped-contexts.md`](./09-grouped-contexts.md) | **superseded by W2**; kept for the record, §4 moved to W3 | — |

## Sequencing

Shown as phases rather than a flat order, because the dependency structure is the
honest thing to show: only phase 1 blocks anything.

- **Phase 0 — correctness now.** W1. Unchanged and still first: ~50 LOC against wrong
  reorders of valid chains, while W2 is a multi-PR refactor. W2 later reuses its
  `_CONTEXT_METHODS` table and retires its trailing barrier.
- **Phase 1 — the keystone.** W2 PRs 1–2 (`AllDims`, then `to_lower_ir`/`emit` and the
  pipeline rewiring). Behaviour-identical, no user-visible payoff, and blocks all of
  phase 2 — build a fused node before this lands and you build it twice.
- **Phase 2 — what lowering unlocks.** W3, then W2 PRs 3–8: `GroupedReduce`,
  `WindowedReduce`, `WeightedReduce`, their pushdown rules, and `explain()` moving to the
  lowered plan.
- **Phase 3 — independent, any order.** W4, W5, W6, W7. Each sharpens the structure the
  optimiser reasons over and none depends on lowering; they slot in opportunistically,
  including alongside phase 1.
- **Phase 4 — gated.** The weighted pushdown rule and data-touching rewrites generally
  (W2 §8.1); the dim-effect unification (W2 §11); the Rust spike (W8).

## The Rust position, restated

`structural-dispatch-2.md` §7 named three triggers for the port: a second consumer of
the IR, a contributor to own the Rust, or the value-and-schema layers modelled to the
point where the port is a transliteration. The third is one workstream (W4) from being
met **for today's IR** — but the IR is about to gain a level. Porting the optimiser
before lowering lands means porting against a plan shape that is not the one the rules
will finally see.

The gate is therefore unchanged in spirit and re-pointed in fact: it now opens after
**W4 and W2 phase 1**, not after W9. Lowering also *improves* the port surface it gates
on — with `emit` owning payload reconstruction (W2 §7), rules touch semantic fields only,
which is exactly the clean seam `structural-dispatch-2.md` §5 sets as the precondition.
The structural work is the payoff either way: each workstream above improves the
optimisations we can apply whether or not any Rust is ever written.

## Checkpoint — where this stands (2026-07-28)

A status entry, not a revision: nothing below changes a decision above. Written at the
point where **W2's PR sequence is complete** and the untouched work is all in phase 3.

### Landed on `main`

| WS | What | PRs |
|---|---|---|
| **W1** | opaque-context barrier | #75 |
| **W2 PRs 1–5** | `AllDims`; the `to_lower_ir`/`emit` keystone; `GroupedReduce` + `ContextOpen`; `WindowedReduce`; `WeightedReduce` | #76, #77, #79, #80, #83 |
| **W3** | `SchemaState` sizes → `int \| None` | #78, #82 |

### In review, as one stacked chain

| PR | What | Item |
|---|---|---|
| #84 | property suite generates builder chains | W7 §7 (carries §6 too) |
| #85 | selects hop past grouped/windowed reduces | W2 PR 6 |
| #86 | projections hop past grouped/windowed reduces | W2 PR 7 |
| #87 | the `dim_effect` unification | W2 §11.1 |
| #88 | projections hop past weighted reduces | W2 §8.1 → "PR 9" |
| #89 | `explain()` on the lowered plan | W2 PR 8 (last of §12) |

### Not started

- **W4** — `Rechunk.chunks` is still `frozendict[Hashable, Any]`, and `_pushable_rechunk`'s
  `isinstance` ladder — §"What remains" item 2 calls it *the last one in the package* — is
  still there.
- **W5** — no `Elementwise` in `src/` at all.
- **W6** — `Scan` still has no dims; its docstring still says the metadata "arrives with the
  first scan-aware rule".
- **W8** — gated on *W4 + W2 phase 1*. Phase 1 landed, so **W4 is the only thing holding the
  gate shut.**

### W7, section by section

Paid: §4 (`AllDims`, promoted to W2 PR 1), §6 (hashability, narrowed not fixed), §7
(property widening), §8 (written 2026-07, settled, implemented by #88).

Open: §1 (merge adjacent `Project`s), §2 (merge adjacent `Rechunk`s — its own note defers it
behind W4), §3 (`pushdown_projections` across `Scan`/`Rechunk` — the `Scan` arm needs W6, the
`Rechunk` arm needs the empirical check the section asks for), §5 (`.plan` for `DataArray`).

### Against this document's own framing

Of the three gaps in §"What remains, honestly": **gap 1 is closed** — grouped ops were
barriered by W1, then modelled properly by W2, and the trailing barrier is retired. **Gaps 2
and 3 are untouched**; they are exactly W4 and W5.

By phase: **0, 1 and 2 complete; 3 is where all the untouched work sits.** W2's memo has one
item left and it was never in the sequence — the weighted *select* rule (§8.1).

One deviation from the sequencing worth recording, since it was not planned: **three of the
four items parked in phase 4 (gated) were taken ahead of phase 3**, because each became
answerable in the course of W2 rather than needing its own decision later —
the dim-effect unification (§11.1, #87) once PR 7 made its shape unambiguous, and the
weighted **projection** rule (§8.1, #88) once `07-small-wins.md` §8 retired the argument
that had excluded it. The weighted **select** rule is the one phase-4 item still genuinely
deferred, and the reason is unchanged: it needs a rewrite that transforms an array rather
than reordering metadata. The Rust spike remains untouched and gated on W4.
