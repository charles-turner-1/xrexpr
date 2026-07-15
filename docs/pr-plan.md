# PR plan: source-rewriting → expression IR

Incremental delivery of [`docs/improvement-report.md`](./improvement-report.md).
Each PR ships green and adds ~100 LOC or fewer (deletions don't count). Builds on
the `add-accessor` branch, whose `LazyDatasetProxy` already exists but is **not yet
registered** as an accessor.

**Critical path:** 1 → (2, 3) → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11.
PRs 2 and 3 are independent and can run in parallel.

## Phase 1 — Foundation (no behavior change)

- [ ] **PR 1 · Land & register the `.plan` accessor** — `~20 LOC`
  Merge/rebase `add-accessor`; add `@xr.register_dataset_accessor("plan")` so `ds.plan`
  returns the proxy. Keep `_optimize_ops` verbatim.
  *Tests:* accessor-equality — `ds.plan.<chain>.collect()` == eager `ds.<chain>` for the
  README pipelines. (Terminal stays `.compute()` until PR 10.)

- [ ] **PR 2 · `ir.py`: the `OpNode` type** — `~80 LOC`
  `@dataclass(frozen=True) OpNode` + a `frozendict` helper (or add the dep). No callers yet.
  *Tests:* construction + immutability (mutating `kwargs`/`indexer` raises).

- [ ] **PR 3 · `operations.py`: metadata table** — `~60 LOC`
  Replace the two bare sets with `op → (kind, consumes_dim)`: `reduce` vs `scan` vs `select`.
  Derive back-compat `AGGREGATIONS`/`SELECTIONS` from it.
  *Tests:* table lookups; parametrized kind assertions.

## Phase 2 — Schema-aware recording

- [ ] **PR 4 · `SchemaState` + `apply_schema`** — `~70 LOC`
  Cheap logical schema (dims/sizes/coords) threaded from `self._base_ds`, updated per op —
  no materialization.
  *Tests:* schema evolves correctly through mean/isel/sel on a sample ds.

- [ ] **PR 5 · `to_opnode(schema, name, args, kwargs)`** — `~90 LOC`
  Record-time normalization: positional/keyword/tuple dims → one `frozenset`;
  **no-dim `mean()` → all current dims**; `sel` labels resolved vs coords.
  *Tests:* golden `OpNode`s per dim-spelling; the `mean()`→all-dims case.

- [ ] **PR 6 · Wire `_record` to build `OpNode`s** — `~60 LOC changed`
  `_record` uses `SchemaState` + `to_opnode`; replay walks `OpNode`s. Behavior unchanged
  (old optimizer still runs on the node list).
  *Tests:* existing accessor-equality suite stays green.

## Phase 3 — The real optimizer

- [ ] **PR 7 · `optimize.py` + fixpoint scaffold + merge rule** — `~80 LOC`
  `optimize(nodes)` fixpoint loop; port merge-adjacent-selects onto `OpNode`s. Retire
  `_optimize_ops`.
  *Tests:* golden op-list for merges.

- [ ] **PR 8 · `pushdown_selects` (generalized, single hop)** — `~80 LOC`
  Select hops left past *any* `reduce` with disjoint `consumes`. **Fixes the mean-only
  limitation (#1).**
  *Tests:* golden op-list proving `sum("lat").isel(time=0)` now reorders.

- [ ] **PR 9 · Fixpoint proof + validity trichotomy** — `~50 LOC`
  Loop reaches the front (`mean("lat").mean("lon").isel(time=0)` → selection-first).
  Trichotomy: disjoint → swap, consumed-dim → `InvalidExpressionError`, scan-dim → leave.
  **Fixes #3 + the `cumsum` conflation + the `mean()` empty-dim bug.**
  *Tests:* the two demo regressions; `cumsum("time").isel(time=5)` left untouched, not raised.

## Phase 4 — Cutover & polish

- [ ] **PR 10 · `.collect()` + `.explain()`** — `~40 LOC`
  Rename terminal `compute()` → `.collect()`; add `explain()` pretty-printer.

- [ ] **PR 11 · Delete the source-rewriting path** — `net negative`
  Remove `cst.py`, `decorators.py`, `rewrite_expr`/`peek_rewritten_expr` exports and their
  tests; update README to the `.plan` API.
