# W6 — Small independent wins

Each item is self-contained, sized at roughly one small PR, and can be picked up
between the numbered workstreams. Ordered by value. Each spec here is deliberately
short; the implementer should follow the conventions of the neighbouring rules/tests
it names.

## 1. Merge adjacent `Project`s

`ds[["tas", "pr"]][["tas"]]` records two `Project` nodes; one suffices. New rule (or
a case folded into a future merge pass): an adjacent `(Project p1, Project p2)` pair
collapses to `p2` alone **iff `set(p2.variables) <= set(p1.variables)`**. The subset
guard is load-bearing: `p2` may legally name a *coordinate* (`ds[["tas"]]["lat"]`
works eagerly because projection keeps coords), and dropping `p1` would then change
which variables exist — when the subset test fails, *leave*, never raise (the chain
may still be valid eagerly). Shrinks the plan → termination measure
(`optimize.py:78-86`) fine. Goldens plus equality-vs-eager for the coord-name edge.

## 2. Merge adjacent `Rechunk`s

`chunk({"time": 100}).chunk({"lat": 50})` → one `chunk({"time": 100, "lat": 50})`:
dask applies later specs per dim on top of earlier ones, so the merged mapping is
later-wins (`{**r1.chunks, **r2.chunks}`). Fire only when **both** nodes pass
`_pushable_rechunk` (`optimize.py:486`) *and* both are pure mapping-form (empty
positional `args`) — a uniform positional spec (`chunk(100)`) rechunks every dim and
does not compose by dict union. Rebuild `args` from the merged mapping the way
`pushdown_selects_past_rechunks` does (`optimize.py:471-479`). Do after W2 so the
values are `ChunkSpec`s.

## 3. `pushdown_projections` across `Scan` and `Rechunk`

Two new arms in the `match crossed` (`optimize.py:413-419`), after W4 gives `Scan`
its dims:

- `case Scan(dims=dims): needed = dims` — a scan is per-variable, so projecting first
  is safe *provided* the projected variables still carry the scanned dims (else the
  swapped `cumsum(dim)` could raise where the eager order didn't) — the same
  rationale as the `Reduce` arm.
- `case Rechunk(chunks=chunks): needed = frozenset(chunks)` — **verify first**: the
  risk is a chunk spec naming a dim that projection orphans. Since projection keeps
  coords and coords keep their dims alive, the dim may well survive regardless —
  check xarray's actual behaviour for `ds[["tas"]].chunk({"lon": 50})` where only a
  dropped variable carried `lon`, and pick the conservative condition accordingly.
  Write the finding into the rule docstring.

Goldens plus equality-vs-eager per arm.

## 4. Symbolic `AllCurrentDims` for bare reduces

**The bug it fixes:** `_reduce_dims` resolves a bare `mean()` to "every dim in the
schema right now" (`schema.py:255-267`), but past the first `Opaque` the schema is a
guess (`optimize.py:108-118`) — `apply_schema` models `Opaque` as dim-preserving,
which `rename`/`stack`/`squeeze` are not. So
`ds.plan.rename(time="t").mean().isel(t=0)` records `consumes={time, lat, lon}`
(stale names), `t` tests disjoint, and `pushdown_selects` swaps — the optimised plan
*silently succeeds* where the eager chain *raises* (mean() consumed `t`; the isel is
invalid). Same family as the empty-dim reorder bug the schema resolution originally
fixed, one trust-boundary further out.

**The fix:** make the dim set symbolic — `consumes: frozenset[Hashable] | AllDims`
where `AllDims` is a singleton sentinel type meaning "every dim at this point,
whatever they are". Semantically exact with no schema read at record time. Touched
sites: `_reduce_dims` (returns the sentinel for the bare case), `pushdown_selects`
(any select intersects `AllDims` → the raise leg, which is always right: after an
all-dims reduce nothing is left to select), `pushdown_projections` (`needed = AllDims`
resolves against the entering folded schema, inside the trusted prefix where it *is*
exact), `apply_schema`'s `Reduce` arm (`AllDims` → clear all dims). W4's `Scan.dims`
should adopt the same type. This is design-note grade — model it on the derived-
property discipline and write a short paragraph in `ir.py`'s docstring. Do after W4
so both kinds convert together.

## 5. Register `.plan` for `DataArray`

Only `Dataset` has the accessor (`accessor.py:66`). Add
`@xr.register_dataarray_accessor("plan")` — the proxy is already almost generic
(`SchemaState.from_dataset` accepts a `DataArray`, `schema.py:59-70`; replay and
record don't care). Differences to handle: no `data_vars` (projection rules simply
never fire — `var_dims` returns `None` for everything, which the rules already treat
as no-rewrite), and `__getitem__` on a `DataArray` is *indexing*, not projection —
`_projected_names` (`schema.py:233`) must not classify it as `Project`, so the
DataArray proxy should record `__getitem__` as `Opaque` (or the accessor gets a flag
the record path consults). Accessor-equality tests over the README-style chains on a
`DataArray`.

## 6. Property-suite widening schedule

`test_properties.py`'s generators are deliberately narrowed
(`test_properties.py:10-28`); each workstream retires a narrowing — widen in the same
PR that lands the feature, or immediately after:

| after | widen |
|---|---|
| W2 | add `chunk` calls (mapping and uniform forms) to generated chains |
| W3 | add elementwise ops with scalar args |
| W4 | add `cumsum`/`cumprod`/`diff` |
| W5 | add `groupby(...).reduce` chains; assert rewrites survive unknown sizes |

The invariant asserted is always the same and is the project's crown jewel:
`ds.plan.<chain>.collect()` equals the eager chain, for generated datasets and
chains, plus idempotence of `optimize`.
