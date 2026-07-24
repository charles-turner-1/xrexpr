# W1 — Opaque-context barrier for accessor-returning ops

**Goal:** make `ds.plan.groupby(...).mean().isel(...)` (and every chain through
`resample`/`rolling`/`coarsen`/`weighted`/...) *correct* — replayed verbatim, never
reordered — without yet modelling grouped semantics. This is the safety fix that
precedes the structural work in [`05-grouped-contexts.md`](./05-grouped-contexts.md).

**Size:** ~50 LOC in `src/xrexpr/accessor.py` plus tests. One PR.

## The bug being fixed

`LazyDatasetProxy.__getattr__` (`accessor.py:116`) treats every callable on the *base
dataset* as a Dataset→Dataset op. A context-returning method like `groupby` is
untabulated, so it records as `Opaque` — fine so far. But the **next** call is checked
against the base dataset again (`_is_method_callable_on_dataset`, `accessor.py:107`),
so `.mean()` after a groupby records as a Dataset-level `Reduce` with `consumes`
resolved against the Dataset schema (`accessor.py:127-133` documents this as a known
limitation).

Concretely, `ds.plan.groupby("time.month").mean()` records as
`[Opaque(groupby), Reduce(mean, consumes={time,lat,lon})]` — the bare `mean()`
resolves against the *Dataset* schema, not the group. Both failure modes of
`pushdown_selects` (`optimize.py:322`) then fire on valid chains:

- `.isel(lat=0)` appended: `lat` intersects that bogus `consumes`, so a perfectly
  valid eager chain **raises** `InvalidExpressionError`;
- `.isel(month=0)` appended: `month` is disjoint from it, so the select **silently
  swaps** behind the reduce, replaying `ds.groupby(...).isel(month=0).mean()` — wrong
  order, wrong (or erroring) result.

The second mode is silent; that is what makes it urgent.

## Design

### 1. The context table

Add to `accessor.py` (module level, near `_EAGER_ATTRS` at `accessor.py:48`):

```python
#: Methods that return a non-Dataset intermediate (GroupBy/Rolling/...) whose
#: subsequent calls mean something different from Dataset ops. Once one is recorded,
#: the proxy stops modelling: every later call records as Opaque, so no rewrite rule
#: can fire on or across the context — replay stays verbatim and correct.
_CONTEXT_METHODS = frozenset(
    {
        "groupby",
        "groupby_bins",
        "resample",
        "rolling",
        "rolling_exp",
        "coarsen",
        "weighted",
        "cumulative",
    }
)
```

Keep only names that exist on `xr.Dataset` in the pinned xarray version (verify each
with `hasattr(xr.Dataset, name)` in a test; drop any that don't exist rather than
guarding at runtime).

### 2. Detecting "in context" — derive, don't store

House discipline is to derive rather than store (cf. `Select.consumes`). Add a helper:

```python
def _in_opaque_context(self) -> bool:
    """Whether an accessor-returning op has been recorded, after which the live
    object is no longer a Dataset and nothing further can be modelled."""
    return any(
        isinstance(op, Opaque) and op.name in _CONTEXT_METHODS for op in self._ops
    )
```

(`Opaque` needs importing from `xrexpr.ir` — the accessor currently imports only `Op`.)
Plans are ~10 nodes, so the O(n) scan per record is irrelevant. No new constructor
parameter, no state to thread.

### 3. Behaviour changes when in context

All in `__getattr__` (`accessor.py:116`) and `_record` (`accessor.py:89`):

- **`_record`:** when `self._in_opaque_context()`, bypass `to_opnode` and construct
  `Opaque(name=method_name, args=args, kwargs=frozendict(kwargs))` directly. Keep the
  `apply_schema` threading unchanged — `apply_schema` on an `Opaque` is a no-op
  (`schema.py:131`), and the schema past the first context op is untrusted anyway
  (`_trusted_prefix`, `optimize.py:108`).
- **`__getattr__`:** when in context, skip the `_is_method_callable_on_dataset` check
  and **always** return a recording wrapper (the live object is a GroupBy/Rolling
  whose methods — e.g. `DatasetGroupBy.first` — need not exist on `Dataset`; today
  those fall into the non-callable branch and break). Do **not** apply
  `@wraps(getattr(self._base_ds, name))` (`accessor.py:146`) in this branch — the
  attribute may not exist on the base dataset; return a plain closure.
- **`_EAGER_ATTRS` terminals keep working:** leave the terminal branch
  (`accessor.py:141-142`) checked *before* the context branch, exactly as it is
  checked before the callable branch today. Replay is verbatim, so
  `ds.plan.groupby(x).mean().to_dataframe()` collects correctly.
- **`__getitem__`** (`accessor.py:155`): goes through `_record`, so it picks up the
  Opaque demotion automatically — a `__getitem__` inside a context must *not* classify
  as `Project`.

### 4. Why this is sufficient for correctness

Every rewrite rule fires only on specific non-`Opaque` variant adjacencies
(`_RULES`, `optimize.py:506`): `(Reduce, Select)`, same-name `Select` runs,
`(Reduce | Select, Project)`, `(Rechunk, Select)`. After this change, every node from
the context op onward is `Opaque`, so no rule can fire on, across, or after the
context. `_trusted_prefix` already stops at the first `Opaque`, so variable-level
reasoning was safe before and stays safe. Replay (`accessor.py:201-209`) is a verbatim
passthrough and replays the chain exactly as written — GroupBy methods included, since
`getattr(ds, node.name)(...)` is called on whatever the previous op returned.

State the residual limitation in the docstring: chains through a context are now
**correct but never optimised**. Update the known-limitation paragraph at
`accessor.py:127-133` to say exactly that (and point at
`docs/roadmap/05-grouped-contexts.md` for the modelling that lifts it).

## Tests (`tests/test_accessor.py`)

1. **The regression.** `ds.plan.groupby("time.month").mean().isel(month=0).collect()`
   equals eager `ds.groupby("time.month").mean().isel(month=0)`
   (`xarray.testing.assert_equal`). Before the fix, the recorded `Reduce`/`Select`
   adjacency swaps and replay breaks — assert it no longer does.
2. **No reordering across a context.** `explain()` on the same chain shows the ops in
   the order written (3 ops, groupby first). Also assert every recorded node from the
   context op on is `Opaque` (inspect `proxy._ops`).
3. **Context-only methods record.** `ds.plan.groupby("lat").first().collect()` equals
   eager — `first` is not a `Dataset` method, so this exercises the
   skip-callable-check branch.
4. **`resample` parity.** `ds.plan.resample(time="2D").mean().collect()` equals eager
   (needs a datetime `time` coord in the fixture).
5. **Terminal after context.** `ds.plan.groupby("time.month").mean().to_dataframe()`
   equals the eager equivalent.
6. **Non-context plans untouched.** The existing suite passes unmodified.

## Acceptance criteria

- All tests above green; existing suite green; `pixi run mypy` (strict) and
  `pixi run python -m ruff check src tests` clean.
- No changes outside `accessor.py` + tests.
- The `accessor.py:127-133` docstring paragraph rewritten to describe barrier
  semantics ("correct, unoptimised") instead of "do not chain".

## Verification commands

```
pixi run python -m pytest tests -q
pixi run mypy
pixi run python -m ruff check src tests
```
