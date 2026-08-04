# W6 — `Scan` grows its dims; scan-aware select pushdown

**Goal:** deliver the metadata `Scan`'s own docstring promises ("its scanned-dim
metadata arrives with the first scan-aware rule", `ir.py:179-180`) and the rule that
is that trigger: a select on dims *disjoint* from the scanned dims hops past the scan.
`cumsum("time").isel(lat=0)` currently doesn't reorder at all — the scan is a full
barrier — even though `isel(lat=0)` is trivially safe to move in front.

**Size:** 1 PR.

> **Unaffected in substance by the lowering re-fold** (2026-07,
> [`02-lowering.md`](./02-lowering.md)). Scans are single-call ops needing no fusion; this
> is lowered-IR node work and independent of lowering in either order. One convergence to
> watch: `02` PR 1 promotes [`07-small-wins.md`](./07-small-wins.md) §4's `AllDims`
> sentinel for bare reduces, and §4 already asks that `Scan.dims` adopt the same type. If
> `02` PR 1 lands first, resolve `dims` to `AllDims` for a bare `cumsum()` here rather
> than expanding it against the record-time schema — which is also the shared `_dim_spec`
> helper this memo introduces, so it is one change in one place.

> **Updated for the landed pipeline (2026-07-29; anchors refreshed against `main`).**
> `02` PR 1 *did* land first: `to_opnode` is schema-free and a bare reduce records
> `ALL_DIMS`, so `Scan.dims` is typed `DimSet` (`frozenset | AllDims`) exactly as
> `Reduce.consumes` is, and a bare `cumsum()` records `ALL_DIMS` — the optimiser's
> `_resolve_dims` already resolves the sentinel against the entering schema.
>
> The `dim_effect` unification (`02` §11.1) also landed, and it **replaces this memo's
> `pushdown_selects_past_scans` rule with one dispatch arm**: `Scan` currently takes the
> conservative `_OPAQUE_EFFECT` (`optimize.py:260-261`), and the whole select rule below
> is `case Scan(dims=dims): return DimEffect(blocks=dims, requires=..., on_conflict="immovable")`
> — `pushdown_selects` (`optimize.py:533`) then swaps disjoint selects and leaves
> intersecting ones, which is exactly the trichotomy stance specified below; nothing is
> registered in `_RULES`. One decision the arm forces that the old shape didn't:
> `requires` must be answered in the same breath. `requires=dims` also enables the
> projection hop across scans — [`07-small-wins.md`](./07-small-wins.md) §3's `Scan` arm,
> spec'd there as safe on the same rationale as `Reduce`'s — while `requires=None`
> keeps this PR select-only and leaves §3(a) for later. Either is defensible; decide and
> say so in the arm's comment.

## Design

### `ir.py` — the field

`Scan` (`ir.py:175-189`) gains a stored dim set, mirroring `Reduce.consumes`'s shape
(stored, record-time-resolved — but named `dims`, because a scan *keeps* its dims,
which is the whole point of the variant):

```python
dims: DimSet = frozenset()  # ``frozenset | AllDims``, exactly as ``Reduce.consumes``
```

with the `frozenset` coercion added to `__post_init__` (copy `Reduce`,
`ir.py:128-133`). Update the class docstring: the "arrives with the first scan-aware
rule" sentence has now paid out.

### `schema.py` — resolution

`to_opnode`'s scan branch (`schema.py:412`) resolves the dim spec with the *same*
helper reduces use — `_reduce_dims` (`schema.py:452`) already implements exactly the
needed convention (kwarg `dim`, else first positional, else every current dim; all
spellings → one frozenset). Since it now serves both kinds, rename it `_dim_spec` (or
similar) and update the reduce call site — the docstring's "reductions take `dim`
first" note generalises: `cumsum(dim=...)`, `cumprod(dim=...)` and `diff(dim)` all put
the dim first too.

Bare-`cumsum()` semantics: xarray applies it over every dim, so "no spec → all
current dims" is correct for scans exactly as for reduces. (Verify `diff` — its `dim`
is required positional, so the bare case can't arise for it.)

**`apply_schema`'s `diff` size effect** — do this in the same PR since the dims are
now carried: `diff(dim, n=1)` shrinks `dim` by `n` (`label` kwarg doesn't change the
size, only which end). Today the `Scan` arm is a blanket `pass` (`schema.py:230`),
which is exact for `cumsum`/`cumprod` but wrong for `diff`. Give `Scan` its own arm:

```python
case Scan(name="diff", dims=dims) as scan:
    n = scan.kwargs.get("n", 1)
    for dim in dims:
        if dim in dims_map:
            dims_map[dim] = max(dims_map[dim] - n, 0)
case Scan() | Rechunk() | Opaque():
    pass
```

Latent today (nothing consults sizes downstream of a scan), but the schema should not
carry a known lie once the information exists to correct it.

### `optimize.py` — the rule

`pushdown_selects_past_scans`, the same single-hop shape as
`pushdown_selects_past_rechunks` (`optimize.py:698`):

- fires on a `(Scan, Select)` adjacency;
- **disjoint** — `frozenset(select.indexer).isdisjoint(scan.dims)` — swap: selecting
  on other dims commutes with a per-dim scan (the scan acts along its own dims,
  independently at each position of the others);
- **intersecting** — *leave, never raise*: `cumsum("time").isel(time=5)` is valid and
  order-significant, the exact case the trichotomy discipline
  (`structural-dispatch.md` §4, and `pushdown_selects`'s docstring, `optimize.py:533`) exists to keep distinct from
  the invalid `(reduce, select)` overlap.

Register in `_RULES` (`optimize.py:773`). Termination: moves a `Select` strictly left,
never lengthens the plan — the measure (`optimize.py:112-119`) still strictly
decreases; say so in the docstring. *(Superseded by the 2026-07-29 note above: no new
rule — one `dim_effect` arm; the termination argument is inherited from
`pushdown_selects` unchanged.)*

**Out of scope, note as future work in the rule docstring:** a *prefix* forward-slice
(`start ∈ {None, 0}`, `step ∈ {None, 1}`) on the scanned dim also commutes with
`cumsum`/`cumprod` (prefix sums only look backward) but **not** with `diff` — a
worthwhile later refinement now that `ForwardSlice` makes "is a prefix" a cheap
structural test, but it needs its own careful goldens, so keep it out of this PR.

## Tests

- **Golden:** `[Scan(cumsum, dims={time}), Select(isel, {lat: 0})]` optimises to the
  swapped order. Composition golden: `cumsum("time") → mean("lat") → isel(lon=0)`
  optimises to `isel(lon=0) → cumsum("time") → mean("lat")` — the select hops past
  the reduce (existing rule) and then past the scan (new rule), proving the fixpoint
  composes them.
- **Golden (leave):** `cumsum("time").isel(time=5)` — plan unchanged, no raise (this
  case exists in `test_optimize.py` already as a no-reorder assertion; extend rather
  than duplicate).
- **Equality vs eager** for both, via `.plan...collect()`.
- **Schema:** `apply_schema` through `diff("time")` shrinks `time` by 1 (and by `n`
  for `diff("time", n=2)`).
- **Property widening:** add scans to the generated pool in `test_properties.py`
  (currently reduce-only, `test_properties.py:62-64`), with the select-dim
  disjointness left to the optimiser rather than the generator — the
  equality-vs-eager property is exactly what proves the leave-don't-raise leg.

## Acceptance criteria

- `cumsum("time").isel(lat=0)` reorders; `cumsum("time").isel(time=5)` doesn't and
  doesn't raise; both collect equal to eager.
- `_dim_spec` (née `_reduce_dims`) has one definition serving both kinds.
- Suite, `pixi run mypy`, `pixi run python -m ruff check src tests` clean.
