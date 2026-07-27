# W3 — `SchemaState` sizes become `int | None`

*(Salvaged intact from [`09-grouped-contexts.md`](./09-grouped-contexts.md) §4, which is
otherwise superseded by [`02-lowering.md`](./02-lowering.md). The sub-design survives the
change of surrounding architecture unaltered — it was always about the schema layer, not
about how grouped ops get recorded. It **gates `GroupedReduce`** (`02` PR 3).)*

**Goal:** give `SchemaState` a way to say *I don't know this dim's size*, so that a fused
grouped reduce can mint a dim whose extent isn't statically evident without the schema
carrying a lie.

**Size:** 1 PR, self-contained, no behaviour change for plans whose sizes are all known.

## 1. Why it's needed

`groupby("time.month").mean()` removes `time` and mints `month` with **size = the number
of distinct groups** — a fact knowable only from coordinate *values*, not from
`ds.sizes`. `SchemaState.dims` is `frozendict[Hashable, int]` (`schema.py:43`) and has no
way to express the gap.

## 2. Position: dim-coordinate values are fair game

Reading a *dimension* coordinate materialises nothing dask-shaped — xarray holds them as
loaded pandas indexes, because they back label lookup. `SchemaState.from_dataset` already
reads `.sizes`/`.coords`/`.dims` (`schema.py:58-70`); reading `ds.indexes[dim]` is the
same class of metadata access, and the no-materialisation promise in `schema.py`'s module
docstring survives it.

So, resolved during lowering (`02` §3), where the schema fold lives:

- `groupby("time.month")` → `new_dim="month"`, size
  `len(np.unique(ds.indexes["time"].month))`.
- `resample(time="2D")` → size from the resampled index.
- A grouper over a **non-dim (possibly lazy) coordinate**, or a `groupby_bins` whose bin
  count isn't statically evident → **size unknown**.

`WindowedReduce` needs the same escape hatch: `coarsen`'s output size depends on the
`boundary` kwarg (`02` §5.2), and the cases that aren't statically resolvable should mark
the size unknown rather than guess.

## 3. The change

```python
dims: frozendict[Hashable, int | None] = field(default_factory=frozendict)
```

with `None` meaning *don't know*. The discipline is already written down and proven
elsewhere in this codebase: `var_dims`' `None`-means-*don't-know* contract
(`schema.py:72-84`) is the template, including its sharp warning that callers must treat
it as "no rewrite", never as "no dims" — here, never as "size zero".

## 4. The consumers to audit

Every site that reads a size must propagate `None` conservatively rather than crash or
silently coerce:

- **`apply_schema`'s `Select` arm** (`schema.py:126-128`) — `dims[dim] = index.size(dims[dim])`
  is the one place a size is *computed*. An unknown input size yields an unknown output
  size; `Indexer.size` should not be called with `None` at all.
- **`Indexer.size(current: int)`** (`indexers.py`) — keep its signature `int`-only and
  guard at the call site, rather than pushing `| None` through six variant
  implementations. `Positions`/`Mask` don't consult `current` and could in principle stay
  exact, but the uniform rule is simpler to defend and nothing today depends on the
  extra precision.
- **`dim_names`** (`schema.py:86-89`) — unaffected; a dim with an unknown size is still a
  dim that exists.

No optimiser rule consults sizes today — every rule reasons about dim *names* — which is
what makes this PR self-contained and why the headline rewrites in `02` §8 survive unknown
sizes untouched. That deserves an explicit property test rather than being left as a
remark.

## 5. Tests

1. **Known sizes are unchanged.** The existing suite passes unmodified; that is the
   primary evidence.
2. **`None` propagates through a select.** A schema with an unknown dim size, put through
   `apply_schema` with a slice select, keeps the dim and keeps its size `None`.
3. **`None` never becomes `0`.** Explicit assertion, since that coercion is the failure
   mode the `var_dims` docstring warns about.
4. **Rewrites survive unknown sizes.** A property test: plans whose schema carries an
   unknown dim size still optimise to the same result as eager evaluation. This is the
   test that lets `02`'s select/projection rules be trusted post-`GroupedReduce`.
5. The existing `xfail(strict=True)` on
   `test_sel_label_slice_size_is_tracked_correctly` (`test_properties.py:342`) should be
   revisited in this PR — an integer-labelled `sel` slice currently under-reports its
   size, and "unknown" is now an available, honest answer for it. Decide deliberately
   whether to convert it; record the decision either way.

## Acceptance criteria

- All tests above green; existing suite green; `pixi run mypy` (strict) and
  `pixi run python -m ruff check src tests` clean.
- No optimiser rule gains a size dependency in this PR.

## Verification commands

```
pixi run python -m pytest tests -q
pixi run mypy
pixi run python -m ruff check src tests
```
