# W4 — `Scan` grows its dims; scan-aware select pushdown

**Goal:** deliver the metadata `Scan`'s own docstring promises ("its scanned-dim
metadata arrives with the first scan-aware rule", `ir.py:112-113`) and the rule that
is that trigger: a select on dims *disjoint* from the scanned dims hops past the scan.
`cumsum("time").isel(lat=0)` currently doesn't reorder at all — the scan is a full
barrier — even though `isel(lat=0)` is trivially safe to move in front.

**Size:** 1 PR.

## Design

### `ir.py` — the field

`Scan` (`ir.py:107-122`) gains a stored dim set, mirroring `Reduce.consumes`'s shape
(stored, record-time-resolved — but named `dims`, because a scan *keeps* its dims,
which is the whole point of the variant):

```python
dims: frozenset[Hashable] = frozenset()
```

with the `frozenset` coercion added to `__post_init__` (copy `Reduce`,
`ir.py:62-65`). Update the class docstring: the "arrives with the first scan-aware
rule" sentence has now paid out.

### `schema.py` — resolution

`to_opnode`'s scan branch (`schema.py:215-220`) resolves the dim spec with the *same*
helper reduces use — `_reduce_dims` (`schema.py:255`) already implements exactly the
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
size, only which end). Today the `Scan` arm is a blanket `pass` (`schema.py:131`),
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
`pushdown_selects_past_rechunks` (`optimize.py:431`):

- fires on a `(Scan, Select)` adjacency;
- **disjoint** — `frozenset(select.indexer).isdisjoint(scan.dims)` — swap: selecting
  on other dims commutes with a per-dim scan (the scan acts along its own dims,
  independently at each position of the others);
- **intersecting** — *leave, never raise*: `cumsum("time").isel(time=5)` is valid and
  order-significant, the exact case the trichotomy discipline
  (`structural-dispatch.md` §4, `optimize.py:339-343`) exists to keep distinct from
  the invalid `(reduce, select)` overlap.

Register in `_RULES` (`optimize.py:506`). Termination: moves a `Select` strictly left,
never lengthens the plan — the measure (`optimize.py:78-86`) still strictly
decreases; say so in the docstring.

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
  (currently reduce-only, `test_properties.py:44-46`), with the select-dim
  disjointness left to the optimiser rather than the generator — the
  equality-vs-eager property is exactly what proves the leave-don't-raise leg.

## Acceptance criteria

- `cumsum("time").isel(lat=0)` reorders; `cumsum("time").isel(time=5)` doesn't and
  doesn't raise; both collect equal to eager.
- `_dim_spec` (née `_reduce_dims`) has one definition serving both kinds.
- Suite, `pixi run mypy`, `pixi run python -m ruff check src tests` clean.
