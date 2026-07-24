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
   Dataset-level `Reduce` (`accessor.py:127-133`), so `pushdown_selects` can silently
   reorder a select behind a groupby. Documented as "do not chain", but it fails
   silently rather than loudly.
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
   product-vision call" is answered: yes, eventually. Grouped contexts are therefore
   designed as the *first sub-plan-carrying variant*, deliberately compatible with the
   later container promotion.
3. **Rust is gated on structure.** There is still no speed case (the optimiser runs
   once per `collect()` on ~10-node plans), and the container shape is about to change.
   The port would transliterate a shape we're about to break. So: finish the chunk
   taxonomy and grouped contexts first, then run a **time-boxed PyO3 spike** — done for
   correctness sharpening and to keep the door open, adopted only if it really is a
   transliteration.

## The workstreams

| # | spec | what | size |
|---|---|---|---|
| W1 | [`01-grouped-barrier.md`](./01-grouped-barrier.md) | opaque-context barrier for accessor-returning ops (correctness) | ~50 LOC, 1 PR |
| W2 | [`02-chunk-taxonomy.md`](./02-chunk-taxonomy.md) | the chunk-spec value sum type — closes doc 2 §3 | 1–2 PRs |
| W3 | [`03-elementwise.md`](./03-elementwise.md) | reintroduce `Elementwise` + selects/projections cross it | 2–3 PRs |
| W4 | [`04-scan-dims.md`](./04-scan-dims.md) | `Scan` gains its dims + scan-aware select pushdown | 1 PR |
| W5 | [`05-grouped-contexts.md`](./05-grouped-contexts.md) | grouped/windowed contexts as the first sub-plan variant (design memo) | memo + staged PRs |
| W6 | [`06-small-wins.md`](./06-small-wins.md) | independent small rules & cleanups, pick up between workstreams | ~1 PR each |
| W7 | [`07-rust-gate.md`](./07-rust-gate.md) | the Rust gate conditions and the PyO3 spike spec | timeboxed spike |

**Sequencing:** W1 first (correctness). W2, W3, W4 are independent of each other and can
land in any order — each sharpens the structure the optimiser reasons over. W5 is the
big one and reshapes the IR; it starts with its memo, which should be reviewed before
implementation. W7 opens only after W2 and W5 (or an explicit decision to shelve W5).
W6 items slot in opportunistically.

## The Rust position, restated

`structural-dispatch-2.md` §7 named three triggers for the port: a second consumer of
the IR, a contributor to own the Rust, or the value-and-schema layers modelled to the
point where the port is a transliteration. The third is one workstream (W2) from being
met **for today's IR** — but W5 changes what the IR *is*. Porting the linear-list
optimiser before the first sub-plan variant lands means porting twice. Hence the gate:
structure first, spike after, adoption criteria written down in advance
([`07-rust-gate.md`](./07-rust-gate.md)). The structural work is the payoff either way —
each workstream above improves the optimisations we can apply whether or not any Rust
is ever written.
