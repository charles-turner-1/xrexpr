# Roadmap: finish the structural encoding, then gate the Rust port on it

*(2026-07. Follows [`structural-dispatch.md`](../structural-dispatch.md),
[`structural-dispatch-2.md`](../structural-dispatch-2.md) and
[`indexer-follow-ups.md`](../indexer-follow-ups.md); supersedes none of them — this is
the plan for what comes after them.)*

## Status at a glance

- [x] W1 — grouped-op barrier
- [x] W2 — lowering stage + fused GroupedReduce/WindowedReduce/WeightedReduce
- [x] W3 — SchemaState sizes → int | None
- [x] W4 — chunk-spec taxonomy (closes the W8 Rust gate)
- [x] W5 — Elementwise + its cross rules
- [x] W6 — Scan gains its dims
- [~] W7 — small wins (§1, §2, §3, §5, §9, §10 done — the rule strand complete; §9/#117 modelled `keepdims`)
- [ ] W8 — PyO3 spike (now ungated; unscheduled)
- [x] W10 — documentation site

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

| ✓ | # | spec | what | size |
|---|---|---|---|---|
| [x] | W1 | [`01-grouped-barrier.md`](./01-grouped-barrier.md) | opaque-context barrier for accessor-returning ops (correctness) | ~50 LOC, 1 PR |
| [x] | W2 | [`02-lowering.md`](./02-lowering.md) | the lowering stage + the three fused builder nodes (design memo) | memo + 8 PRs |
| [x] | W3 | [`03-schema-sizes.md`](./03-schema-sizes.md) | `SchemaState` sizes → `int \| None` (salvaged from W9 §4) | 1 PR |
| [x] | W4 | [`04-chunk-taxonomy.md`](./04-chunk-taxonomy.md) | the chunk-spec value sum type — closes doc 2 §3 | 1–2 PRs |
| [ ] | W5 | [`05-elementwise.md`](./05-elementwise.md) | reintroduce `Elementwise` + selects/projections cross it | 2–3 PRs |
| [ ] | W6 | [`06-scan-dims.md`](./06-scan-dims.md) | `Scan` gains its dims + scan-aware select pushdown | 1 PR |
| [~] | W7 | [`07-small-wins.md`](./07-small-wins.md) | independent small rules & cleanups, pick up between workstreams | ~1 PR each |
| [ ] | W8 | [`08-rust-gate.md`](./08-rust-gate.md) | the Rust gate conditions and the PyO3 spike spec | timeboxed spike |
| — | ~~W9~~ | [`09-grouped-contexts.md`](./09-grouped-contexts.md) | **superseded by W2**; kept for the record, §4 moved to W3 | — |
| [x] | W10 | [`10-documentation.md`](./10-documentation.md) | the docs site (Sphinx + MyST, RTD-hosted, executed examples), essay migration, docstring slimming | 8 PRs |

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

Paid: §1 (merge adjacent `Project`s, #102), §4 (`AllDims`, promoted to W2 PR 1), §5
(`.plan` for `DataArray`, #105 — and the wrong-row `__getitem__` collapse it turned out to
have been reachable on a `Dataset` chain all along), §6 (hashability, narrowed not fixed),
§7 (property widening), §8 (written 2026-07, settled, implemented by #88).

Open: §2 (merge adjacent `Rechunk`s — its own note defers it
behind W4), §3 (`pushdown_projections` across `Scan`/`Rechunk` — the `Scan` arm needs W6, the
`Rechunk` arm needs the empirical check the section asks for).

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

### Everything left, in one list

The three subsections above split the remainder by *why* it is outstanding, which is the
useful cut for a reviewer and the wrong one for picking up the next job. So, flat — once
#84–#89 land, this is the whole of it:

| Item | State | Note |
|---|---|---|
| **W4** — chunk taxonomy | landed (#99) | `xrexpr.chunks`; `_pushable_rechunk` is exhaustive. The W8 Rust gate's last condition is met, so **W8 is now ungated** — still unscheduled, and still a spike rather than a commitment. |
| **W5** — `Elementwise` | not started | Nothing in `src/` at all; §"What remains" gap 3. |
| **W6** — `Scan` dims | not started | Also unblocks W7 §3's `Scan` arm. |
| **W7 §1** — merge adjacent `Project`s | landed (#102) | `optimize.merge_adjacent_projects`; see §1's closing note for the `single` barrier and the schema-free posture. |
| **W7 §2** — merge adjacent `Rechunk`s | unblocked | Was parked behind W4, which has landed: the values it would merge are `ChunkSpec`s now. |
| **W7 §3** — projections across `Scan`/`Rechunk` | part-blocked | `Scan` arm needs W6; `Rechunk` arm needs the empirical check §3 asks for. |
| **W7 §5** — `.plan` for `DataArray` | landed (#105) | `accessor.LazyProxy`, registered on both types; `__getitem__` is classified by the receiver, which also fixed a latent wrong-row collapse on `Dataset` chains. See §5's closing note. |
| **W2 §8.1** — the weighted **select** rule | deferred, undecided | The only item left of the lowering memo, and it was never in §12's sequence. Unchanged reason: it needs a rewrite that *transforms an array* (subsetting the weights alongside the select) rather than one that reorders metadata — the first in the package to do so. |
| **W8** — the Rust spike | ungated, unscheduled | Its last condition was W4, which has landed. Still a timeboxed spike to be decided on, not a commitment. |

One item not in any workstream, found while writing #89: **the README never mentions
`groupby`** — no builder-chain section at all, though fused nodes are W2's headline result.
#89 adds a single grouped `explain()` example because the format change needed one; a real
"grouped and windowed operations" section is still owed, and is a docs job rather than part
of any workstream above.

> **Promoted to a workstream (2026-08-04).** The debt above was paid into the README by
> #92/#93 because the README was the only rendered artefact there was. That posture is now
> superseded: [`10-documentation.md`](./10-documentation.md) (W10) gives the material a
> real home — a MyST/Sphinx site on Read the Docs — migrates the module-docstring essays
> into it, and then slims both the docstrings and the README's `<details>` blocks down to
> pointers.

## Checkpoint — handover (2026-07-29)

Written at the point where the previous checkpoint's whole "in review" chain (#84–#89)
has merged, plus three items outside it: the README rewrite (#92, #93 — which pays the
"grouped and windowed operations" debt recorded above, `README.md:240-264`), the numbagg
CI split (#95), and the path-dependence comment cleanup (issue #94 — `src/` and `tests/`
no longer narrate landed sprints; the standing rule is that **a comment may cite the
roadmap iff the work it points at is still open**). Phases 0–2 are complete; everything
below is phase 3/4 or a bug. This section is the entry point for whoever picks the work
up — the specs are implementation-grade, and each had its anchors refreshed against
`main` on this date.

### One structural fact the open specs predate

The `dim_effect` unification (`02` §11.1, #87) landed after W5/W6 were written, and it
changes their optimiser side from "write a new rule" to "**answer one dispatch arm**":
`optimize.dim_effect` (`optimize.py:204`) is a single `match` closed with `assert_never`,
and both pushdown rules are generic over the node they cross. Each spec now carries a
dated note (2026-07-29) saying exactly what its arm is. Nothing else in them changed.

### The work, in recommended order

| # | Item | Spec / issue | Why this position |
|---|---|---|---|
| 1 | ~~**#90** — `_grouper_dims` fuses non-dim coordinate groupers~~ **done 2026-07-30.** `to_lower_ir` now takes the base dim names and refuses a grouper whose head is not among them; the same bug turned out to live in the `resample` and dotted-`groupby_bins` branches too. The generator widening that caught it also surfaced **#109** (see item 6). | issue #90, `02-lowering.md` §5.5's note — rewritten to what was wrong and what stays open | Was first because everything later trusts the schema. |
| 2 | ~~**W4 — chunk taxonomy**~~ **done 2026-07-31** (issue #99). `chunks.py` carries the seven variants and `classify_chunk`; `Rechunk.chunks` is `frozendict[Hashable, ChunkSpec]`; `_pushable_rechunk`'s value test is a `match` closed with `assert_never`. The two questions the spec said to measure were measured: `None` ≠ `-1` (so `NoChange` and `FullDim` both stay), and a tuple of block lengths round-trips. **This closes the W8 gate's last condition** and unblocks W7 §2. | issue #99, [`04-chunk-taxonomy.md`](./04-chunk-taxonomy.md) | Was the highest-leverage item left. |
| 3 | **W6 — `Scan` dims** | issue #100, [`06-scan-dims.md`](./06-scan-dims.md) (see its 2026-07-29 note) | One PR; also unblocks W7 §3's `Scan` arm — take that in the same `dim_effect` answer if `requires=dims` is chosen. |
| 4 | **W5 — `Elementwise`** | issue #101, [`05-elementwise.md`](./05-elementwise.md) (see its 2026-07-29 note) | The biggest coverage win; 2–3 PRs. |
| 5 | ~~**W7 §1** merge adjacent `Project`s (#102)~~ **done 2026-07-30** (`merge_adjacent_projects`; the failing-subset case left a *different* rewrite on the table, filed as #115); **§2** merge adjacent `Rechunk`s (#103, after W4); **§3** projections across `Scan`/`Rechunk` (#104 — the `Scan` half may already be paid by item 3; the `Rechunk` half needs §3's empirical check); ~~**§5** `.plan` for `DataArray` (#105)~~ **done 2026-07-31** (`LazyProxy` on both types; the `__getitem__` classification fixed a wrong-row collapse that predated it, and §7's list-form narrowing is now retirable) | [`07-small-wins.md`](./07-small-wins.md) | Independent; slot between the numbered items. |
| 6 | **The rest of the "schema lies" family.** **#60** — a `DataArray` indexer falls to `classify`'s `_scalar` catch-all (`indexers.py`, last line) and mis-evolves the schema; the issue body specifies scope and the conservative fallback (record such selects `Opaque`). **#109** — `SchemaState.coords` is a bare set of names carrying no dims, so a coordinate xarray orphans by consuming the dim it is defined on is tracked as surviving; needs `coords` to become a mapping, mirroring `data_vars`. | issues #60, #109 | Both are the family item 1 belonged to. Less reachable than #90 was: #60 needs a bare reduce downstream, and nothing in `optimize` reads `coords` at all, so #109 is inert until the first rule does. |
| ~~7~~ | ~~**`.reduce` mis-tabulation**~~ — **done 2026-07-30** (issue #96). **Parsed properly**, not untabulated: which positional holds the dim spec is now the `schema._DIM_ARG_POSITION` table, so `.reduce` records an honest `Reduce` and its chains optimise. The generator draws it positionally (3.1% of chains). Closing it surfaced a **live** bug in the same branch — `keepdims=True` keeps its dims at size 1, so the recorded `consumes` made the optimiser *reject a valid chain*; now `Opaque`, with the modelling spec'd as `07-small-wins.md` §9 (issue #117). | issue #96, `07-small-wins.md` §7's closing note | Landed. |
| ~~9~~ | ~~**#121 — an emptying select crosses an auto-sizing rechunk**~~ **done 2026-08-01.** Fixed on the *spec*, not the pair: `Auto`, `ByteSize` and `BlockSeq` are the specs dask resolves by measuring the array it is handed, so all three barrier and `_pushable_rechunk` stays one `match`. The narrower guard was built first and rejected — measurement showed the hazard depends on `array.chunk-size` and the rank, so no per-dim discriminant separates safe from unsafe (`chunks.py`'s `Auto` carries the table). `Rechunk` now classifies its **uniform** form too, which retired the two `isinstance(args[0], ...)` reach-ins. The generator exclusion is lifted and the strict xfail is a passing test. | issue #121, [`04-chunk-taxonomy.md`](./04-chunk-taxonomy.md) | Was the only case in the package where the rewrite was worse than eager rather than merely different (contrast `07-small-wins.md` §8). |
| ~~8~~ | ~~**NumPy-style docstrings**~~ — **done** (issue #97). Every callable in `src/xrexpr/` now carries `Parameters`/`Returns`/`Raises` sections with its correctness argument under `Notes`, and the style is enforced rather than aspirational: ruff `D` with `convention = "numpy"`, plus the `numpydoc-validation` pre-commit hook for the content ruff can't check. | issue #97 | Landed. |
| 10 | **W10 — the documentation site** (added 2026-08-04). Sphinx + myst-nb on Read the Docs; the module-docstring essays migrate to `internals/` pages; the final PR slims the docstrings and the README. PRs 1–7 are additive and interleave freely with the code items above; PR 8 edits the same module docstrings W5/W6 will touch, so it lands after whichever of those is in flight. | issue to file, [`10-documentation.md`](./10-documentation.md) | Last, as item 10: docs block nothing, and the destructive PR wants the essay churn behind it. |

Deliberately *not* on the list: the weighted **select** rule (`02` §8.1 — needs the
package's first data-touching rewrite, and its own design decision; issue #107, now a
sub-issue of #158, whose coord/index-aware second pass is where that decision is framed —
see §8.1's 2026-08-10 note), `GroupedMap` (`02` §5.6 — modelled spec, waiting on its
stated trigger; issue #91), and
the W8 Rust spike (gated on item 2; issue #106, spec in
[`08-rust-gate.md`](./08-rust-gate.md)). All three are tracked so the gate is visible in
the tracker rather than only here; none is scheduled.

Every row above carries its issue number, and #99–#107 were filed in this table's order.
Item numbers are kept stable as rows are struck through, so that ordering stays readable;
#109 arrived later and sits with the family it belongs to rather than at the end, and #121
later still — as item 9, since it belongs to no family and the numbering is a reading order
rather than a priority. Anything
added from here should be filed and linked in the same pass — the checkpoint is only useful
while it and the tracker agree.

### Ground rules for the implementer

- Every PR ships green under `pixi run python -m pytest tests -q`, `pixi run mypy`
  (strict), and `pixi run python -m ruff check src tests`.
- Behaviour-preserving PRs prove it by the existing suite passing unmodified; each new
  node kind must answer `dim_effect` and `apply_schema` explicitly (the `assert_never`s
  make skipping this a type error, not a review comment).
- Widen `test_properties.py` in (or immediately after) the PR that lands a feature —
  the schedule is `07-small-wins.md` §7's table.
- Don't reintroduce sprint narration into comments; cite a roadmap file only for work
  that is still open.
- Write **new or edited docstrings in NumPy style** (`Parameters`/`Returns`/`Raises`,
  correctness arguments under `Notes`). This is enforced, not a convention: ruff's `D`
  rules run with `convention = "numpy"`, and the `numpydoc-validation` pre-commit hook
  checks the sections against the signature — every parameter documented, in order, and
  a `Returns` wherever a value comes back. Both run in CI's pre-commit job. The check
  list and the two checks deliberately left off (`GL01`, and the
  extended-summary family) are in `[tool.numpydoc_validation]` in `pyproject.toml`.
  Module docstrings keep their design-narrative prose; the sectioned format is for
  callables.
- **Tests carry a docstring too.** Every test function gets a one-line NumPy-style
  summary stating the scenario it pins — declaratively ("A select hops in front of a
  disjoint rolling window..."), not as an instruction, since the sentence describes what
  the test *is*, not an action to perform. Promote any load-bearing reasoning that would
  otherwise sit in a leading `#` comment into a `Notes` section instead; leave trailing,
  line-level comments where they are. `ruff`'s `D` rules run on `tests/**` too (only
  `D401` is exempted there, for the declarative-summary point above).

## Checkpoint — status refresh (2026-08-04)

A status entry, not a revision: no decision above changes. This reconciles the doc with
`main` and adds the checkboxes in §"Status at a glance".

Landed since the 2026-07-29 handover:

- **W4** — chunk taxonomy (#122/#123) and the extent-dependent-spec barrier (#121/#129);
  this closed the **W8 Rust gate's last condition** — the gate is now fully open.
- **W10** — the documentation site, in full: Sphinx skeleton + executed quickstart, CI/RTD
  build, user guide, internals essays, generated API reference, docstring/README slimming
  (#134–#144, PRs 1–9).
- Planning docs moved to `planning/` (#138) — hence this file's new path.

Everything still open: **W5** (Elementwise, #101), **W6** (Scan dims, #100), **W7 §2/§3**
(#103/#104), **W8** (the Rust spike, #106 — ungated, unscheduled), and the "schema lies"
family (#60, #109, #115, #117). Deferred by design: #91 (GroupedMap), #107 (the
weighted-select rule).

## Checkpoint — status refresh (2026-08-05)

A status entry, not a revision. Reconciles the doc with `main` after four more items
landed; the previous refresh's "everything still open" line above is now stale in three
places and this supersedes it.

Landed since the 2026-08-04 refresh:

- **W6** — `Scan` gains its dims, `diff` becomes its own variant, scan-aware select
  pushdown (#100), together with **W7 §3** — `pushdown_projections` across `Scan` and
  `Rechunk` (#104), both in #146. The `Scan` arm §3 was part-blocked on is paid; the
  `Rechunk` arm rode along on the same `dim_effect` answer.
- **#109** — `SchemaState.coords` became a dims-carrying mapping, so coordinate lifetimes
  are trackable; the schema no longer keeps three views of one fact.
- **#149** — the pre-2026.4.0 `cumsum`/`cumprod` scan-coord drop is now reproduced
  bug-for-bug in `apply_schema`, gated on the xarray version (`SCAN_DROPS_SCANNED_COORDS`
  in `schema.py`), so the tracked schema is *exact* on every supported version and the
  property test checks coords unconditionally (#150). Fixing it also closed a latent,
  version-independent `GroupedReduce` bug: a dimension-name grouper on a *bare* dimension
  minted a coordinate the real xarray does not.

**W5 (#101) is now the head of the queue** — with W6 done it is the last unstarted feature
workstream and, per the handover order, the next item. Still open: **W5** (Elementwise,
#101), **W7 §2** (merge adjacent `Rechunk`s, #103), and the smaller "schema lies" items
**#60** (DataArray indexers misclassified as scalar — a live *hides-an-invalid-chain* bug,
not a wrong value), **#115** (coord-only projection — now unblocked, since its stated
prerequisite #109 has landed), and **#117** (model `keepdims=True`). Deferred by design:
#91 (GroupedMap), #106 (Rust spike, ungated/unscheduled), #107 (weighted-select rule).

## Checkpoint — W5 landed (2026-08-05)

A status entry, not a revision, written the same day as the refresh above and superseding
its "head of the queue" line: **W5 is done.** With it, every feature workstream (W1–W6,
W10) has landed; what remains is small wins, the "schema lies" bugs, and the gated Rust
spike.

### W5 — `Elementwise`, landed as a 4-PR stack (#101)

| PR | What |
|---|---|
| #152 | the `Elementwise` variant + `ElementwiseSpec`, classified at record time behind the `_elementwise_safe` guard; behaviour-neutral (a barrier in `dim_effect` still) |
| #153 | the single `dim_effect` arm (`blocks`/`requires` both empty) that lets selects and projections cross it; no new rule, nothing in `_RULES` |
| #154 | the property generator draws `astype`/`fillna`/`clip`/`round` |
| #156 | scattered NaNs in the generated data + soak `max_examples` 100→250 |

Two facts the spec (`05-elementwise.md`) predated, resolved in the build:

- **`OpSpec` is a sum type now**, so the table row is a new `ElementwiseSpec` dataclass and
  `to_opnode` gains a `case ElementwiseSpec(...)` arm — not the spec's stale
  `OpSpec("elementwise", False)` / `kind == "elementwise"` string. House style: sum types
  over kind strings.
- **The lowering pipeline added two exhaustive sites** the spec never named —
  `lower._emit_node` and `explain._annotate` — which the `assert_never`s flagged the moment
  `Elementwise` joined the `Op` union. Both are mechanical (single-call replay, no
  annotation).

The optimiser side really was one arm: both pushdowns are generic over `dim_effect`, and
`_trusted_prefix` needed no change because `apply_schema` is exact for the node.

### Two things found in review, worth recording

- **A reduce never reorders across an elementwise, and that is now pinned.** A `mean` does
  not commute with a `fillna` (fill-then-average ≠ average-then-fill), and it must never
  cross one — which it cannot, since **no rule moves a `Reduce`** (it is only ever the
  *crossed* node in `dim_effect`). #153 adds structural goldens for the invariant plus a
  NaN-bearing equality test, since the base fixtures are NaN-free and a value-level check
  needs a NaN to have teeth.
- **A latent upstream dask bug, surfaced by the larger soak — not ours.** Strided-slicing
  an uneven block-tuple chunking (`chunk(lat=(1, 2))` then `isel(lat=slice(0, 2, 2))`)
  leaves a degenerate `(1, 0)` chunk, and a *full reduction* over it — the `(a == b).all()`
  inside `xarray.testing.assert_equal`'s `array_equiv` — raises `AxisError` from dask's
  `_concatenate2`. Reproducible in pure xarray+dask, and the values agree; it is the
  comparison, not any rewrite, that fails. The suite's three replayed-vs-eager checks now
  go through `_assert_replays_equal`, which compares *materialised* values (`.compute()`
  moves the `.all()` off dask and onto numpy, where it is well-defined). A root-cause memo
  is written; a dask issue (primary) and an xarray cross-link are still **to be filed** —
  the one follow-up this checkpoint leaves open that has no tracker number yet.

### Everything still open

**W7 §2** (merge adjacent `Rechunk`s, #103); the "schema lies" items **#60** (DataArray
indexers misclassified as scalar) and **#117** (model `keepdims=True`); and **W8** (the
Rust spike, #106 — ungated, unscheduled). Deferred by design: #91 (GroupedMap), #107
(weighted-select rule). No feature workstream remains — the roadmap's structural
programme is complete.

## Checkpoint — #115 landed (2026-08-10)

A status entry, not a revision. **#115 is done** —
`optimize.eliminate_projection_before_coord` (`07-small-wins.md` §10, the follow-up §1
filed): a projection whose only consumer names a coordinate that survives it is dropped,
so `ds[["temperature"]]["lat"]` no longer materialises `temperature`. Schema-based and
confined to the trusted prefix; declines wherever the coordinate does not survive `p1`
(never turning an eager error into a value).

Widening the soak for it surfaced one new item, filed in the same pass: **#162** — a
projection naming a *coordinate* falls into `apply_schema`'s `Project` arm's "decline"
leg, so the tracked schema over-reports it (claims a data var survives that evaluation
drops). Inert today (no rule reads the post-projection schema of a coord projection, and
the new rule reads only the schema *entering* `p1`), but a schema-lie of the #60/#109
family — and the reason the coord-projection soak generator is kept out of
`test_tracked_schema_agrees_with_evaluation`.

Still open: **W7 §2** (#103), the "schema lies" items **#60**, **#117** and **#162**, and
**W8** (#106). Deferred by design: #91, #107.

## Checkpoint — W7 §2 landed (2026-08-11)

A status entry, not a revision. **W7 §2 is done** — `optimize.merge_adjacent_rechunks`
(with the `_mergeable_rechunk` predicate; #103): an adjacent pair of mapping-form,
`_pushable_rechunk` `chunk` calls fuses into one `chunk({...})` node, later-wins per dim.
Syntactic (no schema read), registered after the other merge rules, and it shrinks the
plan — so the termination measure holds on its first component, the same footing as
`merge_adjacent_projects` — while handing `pushdown_selects_past_rechunks` a single better
input than two adjacent nodes.

Two facts the spec (`07-small-wins.md` §2) predated:

- **The spec's naive `{**r1.chunks, **r2.chunks}` union is wrong for a *later* `NoChange`.**
  `chunk({dim: None})` means "leave this dim as it is", so
  `chunk({"time": 100}).chunk({"time": None})` keeps `time` at 100-blocks where a plain
  union would drop it to the base's single block — measured against dask, not reasoned. A
  later `NoChange` is therefore non-overriding (`setdefault`); every concrete later spec
  wins outright.
- **The random soak does not reach it, so it gets a dedicated generator.** Two adjacent
  *mergeable* chunks are too rare to fire the rule over 250 examples (instrumented: 0), so
  — the `select_runs`/`coord_projection_plans` precedent — `rechunk_runs` asks for the run
  by name, with an anti-vacuity `len(optimised) == 1`. And it asserts the merged *chunking*
  through a new `_assert_chunking_equal` (replay-without-compute, mirroring
  `test_accessor`'s `_replayed`), **not** `_assert_replays_equal`: a rechunk preserves
  values whatever blocks it lands on, so the materialised-value check passes vacuously and
  only `.chunks` catches a dropped spec or a repeated dim resolved the wrong way — confirmed
  to fail against a naive-union merge.

Still open: the "schema lies" items **#60**, **#117** and **#162**, and **W8** (#106).
Deferred by design: #91, #107. **With §2 landed, W7's small-wins rule strand is
complete** — what remains across the whole roadmap is the schema-lies bugs and the gated,
unscheduled Rust spike.

## Checkpoint — W7 §9 / #117 landed (2026-08-11)

A status entry, not a revision. **#117 is done** — `keepdims=True` is *modelled* rather
than refused (`07-small-wins.md` §9). It was recorded `Opaque` since #96 (the conservative
fix for the live half); now it records an ordinary `Reduce` and a **derived `Reduce.keepdims`
property** (the `Project.single` precedent) tells the three dispatch sites the named dims
are *resized to 1*, not removed. So a `mean(keepdims=True)` no longer barriers the chain: a
disjoint select hops it, and a select on a kept dim is *left* (an `immovable` `dim_effect`
mirroring `WindowedReduce`, not the plain reduce's `invalid`) rather than raised.

Two facts the spec (`07-small-wins.md` §9) predated:

- **The coordinate half was underspecified, and it's a drop, not a keep.** Measured against
  xarray 2026.7.0: `mean("time", keepdims=True)` keeps `time` at size 1 in every data
  variable but **drops every coordinate spanning it** — the `time` dim-coord *and* a non-dim
  `ref(time)` both go (a reduced dim has no meaningful label). So `apply_schema` is a hybrid
  of `_aggregated` (drop spanning coords) and a resize (keep data-var dims, size → 1), and
  reuses neither — it gets a dedicated `_keepdims_reduced` helper alongside the other two.
- **The lowering fusion refusal had to land in the same PR, and did.** A modelled `keepdims`
  closer now *passes* the `isinstance(closer, Reduce)` guard, so each `_fuse_*` gained an
  `or closer.keepdims` refusal — otherwise `groupby(...).mean(keepdims=True)` would fuse into
  a `GroupedReduce` whose semantics are wrong. `Opaque` gave this for free before; the guard
  restores it, and the pair demotes to a verbatim `Opaque` pair exactly as it used to.

The property generator's plain reduce draw now carries `keepdims`, so the size-exactness
property checks the kept-at-1 dims exactly; the builder-closer refusal is pinned directly in
`test_lower.py` (all three families) rather than via the soak, where it would only generate
`Opaque`-demoted chains the tracked-schema property assumes away.

Still open: the "schema lies" items **#162** and, tracked as an *optimisation* now rather
than a bug, **#166** (#60's vectorized-modelling half — #60 itself closed, correctness having
shipped in #159); and **W8** (#106). Deferred by design: #91, #107.
