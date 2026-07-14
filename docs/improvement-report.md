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
to, and how to build it incrementally without breaking the current API. It is written to
be read top-to-bottom: **Part A** is the design argument (why), **Part B** is the concrete
implementation plan (how).

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

**2. It reasons about text, not computation.** Because it parses source, it breaks on
everything that isn't a single flat method chain in an importable function:
- locally-defined functions → the repo's own `test_local_func_pushdown...` is `xfail`
  ("CST fails on locally defined functions, indentation error");
- lambdas, REPL/Jupyter-defined functions, `functools.partial`, decorated functions,
  C-defined callables (no source);
- intermediate variables (`a = ds.mean("lat"); return a.isel(time=0)`), loops, and
  comprehensions that build the chain — invisible to a single-Call pattern.

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

## A4. Target architecture: capture → optimize → materialize

```
func(ds)  ──trace──►  Expr IR  ──optimize──►  Expr IR'  ──materialize──►  result
          (proxy)     (tree)     (rules)                  (replay calls)
```

**Capture (frontend): a tracing proxy.** Call `func` with a stand-in object `X` that
records operations instead of executing them. Each xarray method (`.mean`, `.isel`,
`.sel`, `__getitem__`, arithmetic) is intercepted, appended to an expression tree, and
returns a new proxy. This observes the **actual runtime sequence of calls**, so lambdas,
intermediate variables, loops, and helper functions all "just work" — directly killing
limitation #2 and the `xfail` test. This is the same trick JAX/Ibis/dask-expr use.
- *Inherent limit (document, don't fight):* control flow that branches on **data values**
  (`if ds.max() > 0: ...`) can't be traced symbolically. Fine — that's rare in these
  pipelines and is the same limitation every tracer has. Fall back to running `func`
  unchanged in that case.

**Optimize (middle): a typed IR + rewrite rules to a fixpoint.** Each node carries its op,
its params, and — crucially — **metadata**: which dims it *consumes* (reduces away), which
it *preserves*, and whether it commutes with a selection on a shared dim. Rules:
- **Selection pushdown (generalized):** `Reduce(dim=D) ∘ Sel(I)` ⇄ `Sel(I) ∘ Reduce(D)`
  when `dims(I) ∩ D = ∅`. Applies to **all** reductions and pure elementwise ops — finally
  using `AGGREGATIONS`, fixing limitation #1.
- **A real validity predicate**, replacing the string-munging: a function of
  `(op_metadata, selection)` that distinguishes *invalid* (selecting a dim a reduction
  destroyed) from *valid-but-not-reorderable* (`cumsum` on its own dim) — fixing
  limitation #3.
- **Optional cost model:** annotate each node with an output-size estimate (dim sizes /
  chunk sizes) and push selections down greedily to minimize elements touched (eager) or
  task count/size (dask). Gives a principled objective instead of a single hard-coded swap.

**Materialize (backend): replay onto real xarray.** Walk the optimized tree and call the
*actual* xarray methods in the new order. No `getsource`, no `exec`, no `func.__globals__`
— removing limitation #4. Public surface stays identical:
- `rewrite_expr(func)` → wrapper that traces once (cached), optimizes, replays on each call.
- `peek_rewritten_expr(func)` / `explain(func)` → pretty-print the optimized IR (and,
  optionally, emit equivalent source for readability).

## A5. What this buys, mapped to the limitations

| Limitation | Fixed by |
| --- | --- |
| Only `mean` | Generalized pushdown rule over `AGGREGATIONS` + op metadata (A4 optimize) |
| Breaks on non-trivial source (lambdas, locals, vars, loops) | Tracing proxy instead of `getsource` (A4 capture) — un-`xfail`s the local-func test for free |
| Fragile / wrong validity check | Explicit per-op dim metadata + validity predicate |
| `exec` in caller globals | Replay real method calls; no recompilation |
| Perf claims unverifiable in tests | Property-based + golden-IR tests (Part B) |

---

# Part B — Implementation plan

Incremental; the current API (`rewrite_expr`, `peek_rewritten_expr`) is preserved at every
step. Build the IR alongside the CST code, then cut over.

## B1. New modules (under `src/xrexpr/`)

- `ir.py` — the expression node types. A small algebraic data model:
  - `Source` (the input dataset placeholder), `Reduce(child, op, dims, kwargs)`,
    `Select(child, kind={"isel","sel"}, indexers, kwargs)`, `Elementwise(child, op, ...)`,
    and a generic `Method(child, name, args, kwargs)` fallback for un-modelled calls.
  - Each node exposes `consumes_dims()`, `preserves_dims()`, and `commutes_with_select_on(dim)`.
- `trace.py` — the tracing proxy `_Recorder` (a stand-in Dataset). `__getattr__` /
  `__getitem__` build IR nodes and return new recorders. `trace(func) -> Expr`.
- `optimize.py` — `optimize(expr) -> expr`: apply rewrite rules to a fixpoint. Start with
  `pushdown_selection`; structure it so new rules are just functions registered in a list.
  Include the validity predicate (`InvalidExpressionError` vs. leave-in-place).
- `materialize.py` — `materialize(expr, ds) -> xr.Dataset`: fold the tree into real
  xarray calls. Optionally `to_source(expr) -> str` for `peek`/`explain`.
- Keep `operations.py` but enrich it: replace the bare `AGGREGATIONS`/`SELECTIONS` sets
  with a small metadata table (op name → consumes-dim? preserves-dim? commutes-with-slice-on-own-dim?),
  so `cumsum`/`cumprod` (order-dependent) are distinguished from `mean`/`sum` (order-free).

## B2. Rewire the public API

- Reimplement `decorators.rewrite_expr` as: `trace` → `optimize` → return a closure that
  calls `materialize(optimized, ds)`. Cache the traced+optimized expr on the wrapper.
- Reimplement `peek_rewritten_expr` on top of `to_source(optimize(trace(func)))`, keeping
  its current string-returning contract (so `test_peek_rewritten_expr` still passes, though
  it should migrate to asserting on the IR — see B3).
- Leave `cst.py` in place initially as a fallback source-emitter; deprecate once the IR
  `to_source` reaches parity, then delete.

## B3. Testing — the highest-value robustness upgrade

The current tests mostly `assert_equal(original, rewritten)`, and several pass *trivially*
without proving a rewrite occurred (e.g. `test_multiple_operations_pushdown` would pass even
if nothing changed). Replace/augment:

- **Property-based (Hypothesis) — the headline addition.** Generate random small datasets +
  random *legal* op chains; assert for all: (a) `materialize(optimize(e), ds)` equals
  `materialize(e, ds)` (correctness), and (b) the cost metric never increases (optimizer is
  monotone). This is what catches the `cumsum`-on-own-dim and weighted/edge cases before
  users do.
- **Golden IR tests.** Assert the *optimized tree* for fixed inputs — deterministic, unlike
  timing. Replaces brittle source-string comparison.
- **Un-`xfail`** `test_local_func_pushdown_optimization_readme` — it passes for free under
  tracing. Add lambda / intermediate-variable / loop cases that the CST path cannot handle.
- **Move timing out of correctness.** Keep the perf claims in an optional `pytest-benchmark`
  / `asv` suite, not in `assert`-based tests.

## B4. Dependencies / packaging

- Add `hypothesis` (dev) and, for the dask target, `dask` (optional extra, e.g.
  `xrexpr[dask]`). Note `dask` is currently **not** a dependency at all.
- No change to `requires-python` or build backend needed.

## B5. Suggested sequencing (each step ships green)

1. `ir.py` + `trace.py` + `materialize.py`; make `rewrite_expr` route through them with a
   single pushdown rule ported from `cst.py`. Keep CST as fallback. Tests stay green.
2. Generalize pushdown to all `AGGREGATIONS`; add op metadata table; add the
   valid-vs-not-reorderable distinction (fixes the latent `cumsum` bug).
3. Add Hypothesis + golden-IR tests; un-`xfail` the local-func test; add lambda/var/loop
   tests.
4. Reimplement `peek_rewritten_expr` via `to_source`; deprecate then remove `cst.py`.
5. (Optional) cost model + dask extra: annotate sizes, and for dask inputs hand the lowered
   graph to dask's optimizer after replay.

## B6. Known hard cases to handle explicitly (so generality is real)

- **Weighted reductions** (`ds.weighted(w).mean(...)`): pushing a slice must slice the
  weights too — model `weighted` as a node so the rule knows. Until modelled, treat as a
  barrier (don't reorder across it).
- **Order-dependent ops** (`cumsum`, `cumprod`, `diff`, rolling/`.rolling`): do **not**
  commute with a positional slice on their own dim; mark non-reorderable, never raise.
- **`sel` specifics**: label indexers, `slice` labels, `method="nearest"`, tolerance —
  commute w.r.t. reductions on *other* dims, but keep their kwargs intact on replay.
- **Non-linear reductions** (`median`, `std`, `var`): still commute with selection on
  *other* dims (they only need the reduced dim's values), but never push a slice on the
  reduced dim through them.
- **Value-dependent control flow**: not traceable — fall back to running `func` unchanged.

---

## Verification

- `pixi run pytest` stays green after each B5 step (the env currently lacks `dask`; add via
  the optional extra before exercising the dask path).
- New Hypothesis suite: `pixi run pytest tests/test_properties.py` — random pipelines prove
  `optimized == original` and cost-monotonicity.
- Manual end-to-end smoke: trace/optimize/replay the three README pipelines plus a
  lambda and an intermediate-variable version; confirm equal results and that `explain`
  shows selections pushed to the front.
- Perf claims: an optional `pytest-benchmark` run comparing original vs optimized on a
  representative dataset (kept out of correctness tests).
