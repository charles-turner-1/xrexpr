# Improving `xrexpr`: from source-rewriting to an expression IR

## Context

`xrexpr` rewrites chained xarray expressions to run faster. Its motivating example:

```python
ds.mean("lat").mean("lon").isel(time=0)     # slow: reduces all timesteps, keeps 1
ds.isel(time=0).mean("lat").mean("lon")     # ~200x faster: reduces 1 timestep
```

These are equal, but xarray won't reorder them. `xrexpr` does — today by reading the
function's **source text** (`inspect.getsource` + `libcst`), pattern-matching a
`.mean(...).isel/sel(...)` pair, swapping it, and `exec`-ing the new source.

This is a classic database/query-optimizer move called **selection (predicate) pushdown**:
do the cheap data-shrinking operations before the expensive ones. The idea is sound and
worth generalizing. The current *mechanism* — rewriting syntax — is what limits it.

The goal of this report: explain **why the mechanism should change**, what to change it
to, and how to build it incrementally while replacing the current function-rewriting API.
It is written to be read top-to-bottom: **Part A** is the design argument (why), **Part B**
is the concrete implementation plan (how).

---

# Part A — Design memo

## A1. What the current implementation actually does

- `decorators.py`: `getsource` → `textwrap.dedent` → `libcst.parse_module` → visit with
  `SelectionPushdown` → `compile` + `exec` in `func.__globals__`.
- `cst.py`: `SelectionPushdown.leave_Call` matches exactly one shape —
  a call whose func is `<X>.mean(...).<sel|isel>(...)` — and swaps the two calls,
  recursively re-visiting to let a selection "bubble" up a chain.
- `_check_valid_ordering`: extracts dim names by string-munging kwargs
  (`key.value.value.strip("'").strip('"')`), and raises `InvalidExpressionError` if the
  `mean` dims and the selection dims overlap.

## A2. Concrete limitations (the case for change)

**1. It only reorders `mean`.** Despite the 14-entry `AGGREGATIONS` set, `cst.py`
hard-codes `attr=cst.Name(value="mean")` on the inner node. `AGGREGATIONS` is imported and
bound to `_` — unused. So `ds.sum("lat").isel(time=0)` is **not** optimized. The README
promises generality the code doesn't deliver.

**2. It reasons about text, not computation.** Because it parses source, it only sees
syntax shapes that happen to match the CST pattern. Any expression built through ordinary
Python control flow or intermediate values is either invisible to the optimizer or requires
more source-rewriting machinery. That is the wrong abstraction layer for this project.

**3. The correctness check is fragile and conflates two different things.**
`_check_valid_ordering` assumes every dim is a simple `dim="name"` **keyword** with a
string literal. It breaks on `dim=("lat","lon")` tuples, positional dims, variable dims,
and `sel(time=slice(...), method="nearest")`. More importantly it treats **"cannot
reorder"** as **"invalid expression"** — true for `mean` (the dim is gone, so a later
`isel` on it *is* invalid), but false in general. `cumsum` is in `AGGREGATIONS` and keeps
its dimension: `ds.cumsum("time").isel(time=5)` is perfectly valid and simply
*not reorderable* — the current model would (once `cumsum` were handled) wrongly raise.
The semantics of "which ops consume which dims, and which commute" are encoded implicitly
and incompletely.

**4. `exec` in `func.__globals__`.** Recompiling model-emitted source and executing it in
the caller's globals is a foot-gun (surprising scoping, a code-injection surface if source
is ever attacker-influenced) and makes caching/debugging awkward.

## A3. Why "rewrite the dask graph" is the right instinct at the wrong altitude

Moving off source text onto a **computation graph** is exactly right — semantics, not
syntax. But the *low-level dask task graph* is the hard place to do it:

- **The information you need is already gone.** By the time xarray lowers `.mean("lat")` to
  dask it's a tree of blockwise `chunk`/`combine`/`aggregate` tasks over opaque keys.
  Recognizing "this `getitem` commutes with this reduction tree" at the raw
  `{key: (func, *args)}` level means reverse-engineering the algebra xarray just discarded.
- **It only helps the lazy case.** The README's 200x is an **eager numpy** effect (compute
  full mean, throw it away). There's no dask graph there to rewrite. Dask's own optimizer
  already does culling/fusion/some slice pushdown, but **not** high-level algebraic
  reduction/selection reordering — so the lazy case is under-served too, but the fix isn't
  at the task level.
- **The proven place is a high-level expression IR.** Every serious array/relational
  optimizer — Ibis, Spark Catalyst, DuckDB, and dask's own `dask-expr` — captures a
  **high-level expression tree**, rewrites *that* (predicate/projection pushdown,
  reordering), then lowers to a physical graph. That's the altitude where "these two ops
  commute" is cheap to see.

### A3.1. "But `dask-expr` is core dask now — doesn't that already do this?"

It doesn't cover xarray, and the reasons are exactly what pin down where `xrexpr` belongs:

- **`dask-expr` is the DataFrame backend, not the array backend.** It merged into core
  dask and is the default (and only) `dask.dataframe` implementation since 2024.3.0, but
  it is a **tabular** query planner (pushes filters/column projections into `read_parquet`,
  etc.). **xarray rides on `dask.array`, a different collection** — so "dask-expr is core
  now" gives xarray workflows no algebraic reordering.
- **The array-side expression work is opt-in and immature.** There is a `dask.array`
  expression reimplementation, but it is gated behind the `array.query-planning` config
  flag, which **defaults to off** (unlike the DataFrame path). It is not the default
  backend.
- **Even when it lands, it is axis-level, not dim-level.** A `dask.array` optimizer sees
  `getitem on axis 0` vs. a `reduction tree on axis 1` over chunks. It has **no concept of
  xarray named dimensions**, and **no concept of coordinates** — so it cannot represent
  `.sel(time=...)` (label-based selection) at all. The naming that makes "`isel(time=0)`
  commutes with `mean(dim='lat')`" *obvious* lives in xarray, above dask.
- **The eager/numpy path is entirely outside dask.** The README's headline 200x is
  numpy-backed — no dask graph exists, so no dask optimizer (core or not) ever runs.

**Takeaway — layering, not rivalry.** `dask-expr` is the *proof the architecture works*,
and the correct division of labour:
- **`xrexpr` = the xarray-semantic optimizer** (named dims + coordinates), above dask.
- **dask (`dask-expr` / array query-planning) = the physical optimizer** below it (chunks,
  blocks, fusion).
- They **compose**, exactly like Ibis (logical plan) → engine (physical plan). When
  `array.query-planning` matures, `xrexpr`'s IR should **lower onto it**, not duplicate it.

**Recommendation:** keep the user's instinct (operate on a graph), but capture the graph at
the **xarray-API level as an explicit IR**, optimize there, then replay onto the real data.
This serves *both* targets: eager numpy gets the reorder win directly, and the dask case
lowers to an already-better graph (and can still be handed to dask's physical optimizer
afterward).

## A4. Target architecture: record → optimize → replay

```
ds.opt.mean(...).isel(...)  ──record──►  [OpNode]  ──optimize──►  [OpNode]'  ──replay──►  result
        (accessor + proxy)   (flat list)   (rules,   (reordered            (.compute())
                                            fixpoint)  op list)
```

There are three pieces. The frontend is an **accessor**; the IR is a **flat op list**; the
optimizer is a **fixpoint rewrite** over that list. A working skeleton of this already
exists on the `add-accessor` branch (`ds.lazier` → `LazyDatasetProxy`); this section
describes what it is and the three deltas that turn it from demo into the plan.

**Record (frontend): an xarray accessor.** Register an accessor
(`@xr.register_dataset_accessor("opt")`) that returns a lazy proxy. Every method
(`.mean`, `.isel`, `.sel`, `__getitem__`, …) is intercepted via `__getattr__`, appended to
an op list, and returns a new proxy; a terminal `.compute()` optimizes and replays. This is
**much cleaner than tracing user code**: no `inspect.getsource`, no `exec`, no
`func.__globals__`. The operations are captured because the user calls them on the proxy.
It is explicit and opt-in (`ds.opt.…`), which reads naturally and sidesteps the entire
class of limitations #2 and #4 at once.
- *Inherent limit (document, don't fight):* control flow that branches on **data values**
  (`if ds.opt.max() > 0: ...`) forces materialization at that point — the recorded chain
  simply ends there and a new one begins. That's fine and expected.

**Record-time metadata resolution — the key win the proxy unlocks.** Because the proxy
starts with the **real dataset** (`self._base_ds`), it can maintain a cheap logical schema
as operations are recorded: current dims, sizes, and coordinate names, updated after each
`OpNode` without materialising data. Each op is normalised against that current schema,
not blindly against the original dataset — something the CST approach could never do. This
is where the demo's `_optimize_ops` (which re-pokes raw `args`/`kwargs` to guess dims)
should move to:
- `mean()` with no `dim` → `consumes = {all current dims}` (not `()`). The demo's empty-dim
  heuristic mis-handles this: it would swap `ds.mean().isel(time=0)` (which is *invalid* —
  `time` is gone after `mean()`) into a valid-but-**different** result. Resolving against
  the schema catches it.
- positional (`mean("lat")`) vs keyword (`mean(dim="lat")`) vs tuple (`mean(("lat","lon"))`)
  all normalise to the same `frozenset` — no more per-call arg archaeology inside the rules.
- `sel(...)` label indexers resolve against coordinates, which only exist because we're at
  the xarray level (see A3.1).

**Optimize (middle): a flat `OpNode` list + rewrite rules to a fixpoint.** A Dataset method
chain is **linear** (each op has exactly one input — the previous dataset), so the IR is a
**list**, not a tree. (A tree is only needed once an op has >1 input — binary arithmetic
`ds.mean("x") - ds.mean("y")`, `concat`, `where(expr)`; that's a later extension, noted in
the future-work block.) Each `OpNode` carries `kind` (`reduce` | `scan` | `select` |
`elementwise` | `opaque`), `consumes` (dims removed), and `indexer` (for selects). Rules
run to a **fixpoint**:
- **Merge adjacent selects:** consecutive `isel`/`sel` collapse into one indexer dict (the
  demo already does this).
- **Selection pushdown (generalized), to a fixpoint:** each select hops left past *any*
  reduce with disjoint dims. The demo does a single forward pass, so it only bubbles a
  select past **one** reduction — `mean("lat").mean("lon").isel(time=0)` optimises to
  `mean("lat").isel(time=0).mean("lon")` and stops, never reaching the front. Looping to a
  fixpoint gets the full `isel(time=0).mean("lat").mean("lon")`. Applies to **all**
  `AGGREGATIONS`, fixing limitation #1.
- **A real validity trichotomy** (replacing the string-munging predicate), from `OpNode`
  metadata:

  | select dims vs. the op it meets | action |
  | --- | --- |
  | disjoint from a `reduce`'s `consumes` | **swap** (safe) |
  | dim ∈ a `reduce`'s `consumes` | **invalid** — dim destroyed; raise `InvalidExpressionError` |
  | dim shared with a `scan` (`cumsum`/`cumprod`/`diff`) | **leave** — valid but *not* reorderable |

  This fixes limitation #3 and the latent `cumsum` bug (the demo, and the old CST, conflate
  "not reorderable" with "invalid").

**Replay (backend): materialise onto real xarray.** Walk the optimised op list and call the
*actual* xarray methods in the new order (the demo's `compute()` loop is already exactly
this). No `getsource`, no `exec`. Public surface is the accessor plus:
- `ds.opt.…….compute()` / `.realize()` → the optimised result.
- `ds.opt.…….explain()` → pretty-print the optimised op list for debugging. Deterministic,
  unlike timing.

## A5. What this buys, mapped to the limitations

| Limitation | Fixed by |
| --- | --- |
| Only `mean` | Generalized pushdown over `AGGREGATIONS` + `OpNode` metadata (A4 optimize) |
| Source-shape/source-availability failures | Accessor records real calls — no source parsing at all (A4 record) |
| Fragile / wrong validity check | Record-time metadata resolution + validity trichotomy (A4) |
| `exec` in caller globals | Accessor `.compute()` replays real method calls; no recompilation |
| Perf claims unverifiable in tests | Golden op-list + accessor equality tests (Part B) |

---

# Part B — Implementation plan

Build on the **`add-accessor` branch** as the skeleton (accessor + proxy + replay loop
already work). The plan lifts its optimizer out of the proxy, gives it real metadata, and
makes it a fixpoint. The source-rewriting path (`cst.py` + `decorators.py` `getsource`/
`exec`) is **deleted**, not extended. The current `rewrite_expr`/`peek_rewritten_expr`
API is retired with it.

## B1. Modules (under `src/xrexpr/`)

- `accessor.py` (from `add-accessor`) — the `@xr.register_dataset_accessor` proxy. Keep its
  `__getattr__`/`__getitem__` recording and its `compute()`/`realize()` replay loop.
  **Change:** `_record` normalises each call into an `OpNode` (below) using a `SchemaState`
  initialised from `self._base_ds`, then updates that schema for the next op; `compute()`
  calls `optimize()` then replays. Add `explain()`.
- `ir.py` — a single flat record type (no tree):
  ```python
  @dataclass(frozen=True)
  class OpNode:
      name: str                    # "mean", "isel", "sel", "cumsum", "__getitem__", ...
      kind: str                    # "reduce" | "scan" | "select" | "elementwise" | "opaque"
      args: tuple
      kwargs: frozendict[str, Any]
      consumes: frozenset[str]     # dims removed — resolved vs the current logical schema
      indexer: frozendict[str, Any] # for selects: {dim: indexer}
  ```
  Use `frozendict` (or a small local equivalent) for top-level `kwargs`/`indexer` so
  optimizer rules cannot mutate recorded call metadata in place. A helper
  `to_opnode(schema, name, args, kwargs) -> OpNode` does the record-time resolution
  (handles positional/keyword/tuple dims and no-dim `mean()` → all current dims). A paired
  `apply_schema(schema, node) -> SchemaState` updates dims/sizes/coords for subsequent ops.
- `optimize.py` — `optimize(nodes: list[OpNode]) -> list[OpNode]`: the **fixpoint** loop
  ```python
  def optimize(nodes):
      changed = True
      while changed:
          nodes, c1 = merge_adjacent_selects(nodes)
          nodes, c2 = pushdown_selects(nodes)   # each select hops left past disjoint reduces
          changed = c1 or c2
      return nodes
  ```
  Rules are plain functions in a list, so new ones (e.g. a cost-driven reorder) just append.
  `pushdown_selects` implements the A4 validity trichotomy.
- Enrich `operations.py`: replace the bare `AGGREGATIONS`/`SELECTIONS` sets with a metadata
  table (op name → `kind`, whether it consumes its dim), so `reduce` (`mean`/`sum`/`std`/…)
  and `scan` (`cumsum`/`cumprod`/`diff`) are distinguished. This table drives `to_opnode`.
- Delete `cst.py` and `decorators.py`; remove the function-rewriting exports from the
  package surface.

## B2. Public API

- **Primary and only execution API:** the accessor —
  `ds.opt.mean("lat").mean("lon").isel(time=0).compute()` (rename `lazier` → a keeper
  name, e.g. `opt`). `.explain()` prints the optimised op list.
- **Removed:** `rewrite_expr(func)` and `peek_rewritten_expr(func)`. The project should not
  carry a compatibility shim for source rewriting; examples and tests should move to the
  accessor API.

## B3. Testing — the highest-value robustness upgrade

The current tests mostly `assert_equal(original, rewritten)`, and several pass *trivially*
without proving a rewrite occurred (e.g. `test_multiple_operations_pushdown` would pass even
if nothing changed). Replace/augment:

- **Golden op-list tests.** Assert the *optimised `[OpNode]`* for fixed inputs —
  deterministic, unlike timing. Replaces the brittle source-string comparison.
- **Regression cases from the demo:** (1) `mean("lat").mean("lon").isel(time=0)` optimises
  fully to selection-first (proves the fixpoint, not single-pass); (2) `ds.mean().isel(time=0)`
  raises `InvalidExpressionError` rather than silently swapping (proves record-time
  resolution).
- **Accessor equality tests.** For the current README-style pipelines, assert
  `ds.opt.<chain>.compute()` equals the direct eager `ds.<chain>`.
- **Move timing out of correctness.** Keep the perf claims in an optional `pytest-benchmark`
  / `asv` suite, not in `assert`-based tests.

## B4. Dependencies / packaging

- Add `frozendict` if using the PyPI package rather than a local immutable mapping helper.
- No change to `requires-python` or build backend needed.

## B5. Suggested sequencing (each step ships green)

1. Merge/rebase `add-accessor`. Introduce `OpNode`, `SchemaState`, `to_opnode`, and
   `apply_schema`; have `_record` build immutable `OpNode`s and lift `_optimize_ops` verbatim
   into `optimize.py`. Behaviour unchanged; just restructured. Tests stay green.
2. Make `optimize` a **fixpoint**; add the metadata table and the `reduce`/`scan`/validity
   trichotomy; generalise pushdown to all `AGGREGATIONS`. Fixes mean-only, single-pass, the
   `mean()` empty-dim bug, and the `cumsum` conflation.
3. Add golden op-list tests + the two demo regression cases; delete `cst.py`, `decorators.py`,
   and the `getsource`/`exec` path.
4. Rename the accessor to its keeper name; add `explain()`; update README/tests to remove
   `rewrite_expr` and `peek_rewritten_expr`.
5. Stop there for v1: accessor + linear IR + schema-aware select/reduce pushdown + golden
   tests. The next set of ideas belongs after that foundation is working.

<details>
<summary>After the linear IR is solid</summary>

- **Property-based tests:** add Hypothesis-generated small datasets and legal accessor
  chains; assert `ds.opt.<chain>.compute()` equals direct eager xarray and, once a cost
  metric exists, that optimization is monotone.
- **Cost model + dask extra:** annotate `OpNode`s with output-size estimates (dim sizes /
  chunk sizes), add `dask` as an optional extra, and for dask inputs hand the replayed graph
  to dask's physical optimizer afterward (A3.1 layering).
- **Multi-input ops → DAG IR:** binary arithmetic (`ds.opt.mean("x") - ds.opt.mean("y")`),
  `concat`, and `where(expr)` have more than one input. Until modelled, treat these as
  `opaque` barriers and do not reorder across them.
- **Weighted reductions:** `ds.weighted(w).mean(...)` requires slicing weights alongside
  data. Until modelled, treat `weighted` as a barrier.
- **Order-dependent ops:** `cumsum`, `cumprod`, `diff`, and `.rolling` are `kind="scan"`:
  do not commute with a slice on their own dim; leave in place, never raise.
- **`sel` specifics:** label indexers, label slices, `method="nearest"`, and tolerance
  commute with reductions on other dims, but replay must preserve kwargs exactly.
- **Non-linear reductions:** `median`, `std`, and `var` commute with selection on other dims,
  but never with selection on the reduced dim.
- **Value-dependent control flow:** materialise at the branch; the recorded chain ends there
  and a fresh one starts after it.

</details>

---

## Verification

- `pixi run pytest` stays green after each B5 step.
- Manual end-to-end smoke: run the README pipelines via `ds.opt.….compute()`; confirm equal
  results and that `.explain()` shows selections pushed to the front.
- Perf claims: an optional `pytest-benchmark` run comparing original vs optimized on a
  representative dataset (kept out of correctness tests).
