# W7 — Small independent wins

Each item is self-contained, sized at roughly one small PR, and can be picked up
between the numbered workstreams. Ordered by value. Each spec here is deliberately
short; the implementer should follow the conventions of the neighbouring rules/tests
it names.

## 1. Merge adjacent `Project`s

> **Closed (2026-07-30)** as `optimize.merge_adjacent_projects` (#102). The spec below is
> what was implemented, plus two facts it did not carry:
>
> - **A `single` `p1` is a second, independent barrier**, and the subset test does not
>   imply it. After a bare-name projection the object is a `DataArray`, on which
>   `__getitem__` is *indexing*: `ds["temperature"]["temperature"]` raises `KeyError`
>   where the collapsed `ds["temperature"]` returns data. Turning an error into a value is
>   the one thing §8's contract forbids outright, so the pair is left.
> - **The rule is deliberately schema-free**, deciding on the two key sets alone. That is
>   what lets it fire past an `Opaque` and admit a `p1` that names a coordinate
>   (`ds[["lat", "temperature"]][["lat"]]` composes correctly). The price is one skipped
>   error: a name in `p1` that the dataset lacks makes eager raise `KeyError` where the
>   merged plan succeeds — §8's middle clause, the same licence `pushdown_projections`
>   runs on, since no value moves and the error comes from work the chain discards.
>
> Verified against xarray 2026.7.0 rather than assumed: `ds[["temperature",
> "elevation"]][["temperature"]]` is `identical` to `ds[["temperature"]]` (a projection's
> coord pruning composes, because `needed_dims(p2) ⊆ needed_dims(p1)`), and so is the
> list-then-bare-name form. No generator change was needed — `_calls` already draws
> list-form projections, so `p2 ⊆ p1` holds by construction on a run of two; measured, the
> rule fires on **5.7%** of generated chains.
>
> **Follow-up, not folded in:** the failing-subset case leaves real work on the table.
> `ds[["temperature"]]["lat"]` materialises `temperature` and throws it away to return a
> coordinate. Eliminating `p1` there is a *different* rewrite — it must prove the coord's
> values survive `p1` untouched, which is a schema question inside `_trusted_prefix`,
> not a key-set one. Filed as #115 (likely wanting #109 first, since it must ask which
> dims a coordinate carries).

`ds[["tas", "pr"]][["tas"]]` records two `Project` nodes; one suffices. New rule (or
a case folded into a future merge pass): an adjacent `(Project p1, Project p2)` pair
collapses to `p2` alone **iff `set(p2.variables) <= set(p1.variables)`**. The subset
guard is load-bearing: `p2` may legally name a *coordinate* (`ds[["tas"]]["lat"]`
works eagerly because projection keeps coords), and dropping `p1` would then change
which variables exist — when the subset test fails, *leave*, never raise (the chain
may still be valid eagerly). Shrinks the plan → termination measure
(`optimize.py:112-119`) fine. Goldens plus equality-vs-eager for the coord-name edge.

## 2. Merge adjacent `Rechunk`s

> **Closed (2026-08-11)** as `optimize.merge_adjacent_rechunks`, gated by a
> `_mergeable_rechunk` predicate (#103). The spec below is what was implemented, plus two
> facts it did not carry:
>
> - **The later-wins union `{**r1.chunks, **r2.chunks}` is subtly wrong for a *later*
>   `NoChange`.** `chunk({dim: None})` asks nothing of the dim ("leave it as it is"), so
>   `chunk({"time": 100}).chunk({"time": None})` keeps `time` at 100-blocks where the plain
>   union would drop it to :class:`~xrexpr.chunks.NoChange` and thus to the base's single
>   block. Measured against dask, not reasoned. A later `NoChange` is therefore
>   non-overriding (`setdefault` onto the earlier mapping); every concrete later spec still
>   wins. The three specs that reach the merge — `SingleSize`/`FullDim`/`NoChange`, all that
>   `_pushable_rechunk` admits — are exactly the ones this rule has to get right.
> - **"Pure mapping-form (empty positional `args`)" is really `uniform is None` *and*
>   non-empty `chunks`.** `_mergeable_rechunk` reads the derived `Rechunk.uniform` rather
>   than inspecting `args`, so `chunk()` (empty, all-dims default — neither form) is excluded
>   alongside the uniform positional specs the spec names. Reusing `_pushable_rechunk` as the
>   second condition folds in the option-kwarg and extent-dependent barriers for free, so the
>   rule is two predicate calls rather than a restated ladder.
>
> **Coverage note the spec did not anticipate.** Two adjacent *mergeable* chunks are too
> rare for the random soak to fire the rule — instrumented at **0** over 250 examples — so,
> the `select_runs`/`coord_projection_plans` precedent, a dedicated `rechunk_runs` generator
> asks for the run by name with an anti-vacuity `len(optimised) == 1`. Its equality half
> uses a new `_assert_chunking_equal` (replay-without-compute, mirroring `test_accessor`'s
> `_replayed`) rather than `_assert_replays_equal`: rechunking preserves *values* whatever
> topology it lands on, so the materialised-value check cannot tell a correct merge from a
> broken one — only `.chunks` can, and it was confirmed to fail against a naive-union merge.
> Goldens in `test_optimize.py` (disjoint union, later-wins, `NoChange`-both-directions,
> uniform/extent-dependent/option-kwarg barriers, run-of-three, idempotence) plus
> equality-vs-eager in `test_accessor.py`.

`chunk({"time": 100}).chunk({"lat": 50})` → one `chunk({"time": 100, "lat": 50})`:
dask applies later specs per dim on top of earlier ones, so the merged mapping is
later-wins (`{**r1.chunks, **r2.chunks}`). Fire only when **both** nodes pass
`_pushable_rechunk` (`optimize.py:753`) *and* both are pure mapping-form (empty
positional `args`) — a uniform positional spec (`chunk(100)`) rechunks every dim and
does not compose by dict union. Rebuild `args` from the merged mapping the way
`pushdown_selects_past_rechunks` does (`optimize.py:735-748`). Do after W4 so the
values are `ChunkSpec`s.

## 3. `pushdown_projections` across `Scan` and `Rechunk`

Two new `requires` answers in `dim_effect` (`optimize.py:204`) — the `match crossed`
this section predates is unified there — after W6 gives `Scan` its dims:

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

> **Promoted out of small-wins** (2026-07) — this is now **PR 1 of
> [`02-lowering.md`](./02-lowering.md)** (see its §6). Lowering gives the same change a
> second, independent motivation: deferring the bare-reduce expansion is the *mechanism*
> by which `to_opnode` becomes schema-free, which is what lets `to_lower_ir` own the
> single schema fold. The spec below is unchanged and is what should be implemented; only
> its priority and its "do after W6" sequencing note have moved. Keep the sentinel
> spelling rather than a bare `None` — `None` already means *don't know* in this codebase
> (`var_dims`, `schema.py:72-84`), whereas this means something definite.

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
exact), `apply_schema`'s `Reduce` arm (`AllDims` → clear all dims). W6's `Scan.dims`
should adopt the same type. This is design-note grade — model it on the derived-
property discipline and write a short paragraph in `ir.py`'s docstring. Do after W6
so both kinds convert together.

## 5. Register `.plan` for `DataArray`

> **Closed (2026-07-31)** as `accessor.LazyProxy`, registered on both types (#105). The
> spec below is what was implemented — the second option, "the accessor gets a flag the
> record path consults" (`LazyProxy._getitem_is_projection`), since `to_opnode` stays a
> pure function of one call and the *receiver* is what settles what `__getitem__` means.
> One class serves both bases: only two things differ, and the base type is read rather
> than stored. Four facts the section did not carry:
>
> - **The `__getitem__` trap is not hypothetical, and it was already reachable.** The
>   dangerous key is not `da["lat"]` (a coordinate read, which replays fine either way) but
>   the *positional* one: `da[[1, 2]]` selects positions along the first dim, so as a pair
>   of `Project`s `da[[1, 2]][[1]]` passes `merge_adjacent_projects`' subset test and
>   collapses to `da[[1]]` — row 1, where eager indexes *within* the pair and gives row 2.
>   Measured on `main` before the fix: the optimised plan returns the wrong row.
> - **And it did not need the new accessor to happen.** A `Dataset` chain *becomes* a
>   `DataArray` at a bare-name projection, so `ds.plan["tas"][[1, 2]][[1]]` is the same
>   wrong-row collapse one op later — a latent wrong-answer bug on the existing accessor,
>   found by this work and fixed with it. §1's barrier on a single `p1` stops the *first*
>   pair only. So the flag reads the recorded plan, not just the base: a bare-name
>   projection (told from `ds[["tas"]]` by the *key*, since both record
>   `variables=("tas",)`) puts the chain in DataArray-land for good. Giving up the
>   classification only ever costs modelling — `Opaque` replays verbatim — so the
>   conservative answer is always available.
> - **A `DataArray`'s schema is coords-only, so a coordless one models no dims at all**
>   (`from_dataset` has no `data_vars` to read). The narrowing is in the safe direction —
>   `var_dims` answers `None`, the projection rules' no-rewrite answer, and a `groupby`
>   over an unmodelled dim fails `lower._grouper_dims`' check and lands on the opaque
>   fallback — but it costs the fusion. Teaching `SchemaState` to hold the array itself
>   (under `.name`, or a sentinel when unnamed) would recover it; that is a schema change,
>   not an accessor one, and is left.
> - **§7's list-form narrowing is now retirable.** `_calls` draws list-form projections
>   only, justified there partly by this section; a bare-name draw would put the generated
>   chains into DataArray-land and exercise the classification above. Left as follow-up
>   work rather than folded in, since it widens the property suite rather than this
>   surface.

Only `Dataset` has the accessor (`accessor.py:70`). Add
`@xr.register_dataarray_accessor("plan")` — the proxy is already almost generic
(`SchemaState.from_dataset` accepts a `DataArray`, `schema.py:89`; replay and
record don't care). Differences to handle: no `data_vars` (projection rules simply
never fire — `var_dims` returns `None` for everything, which the rules already treat
as no-rewrite), and `__getitem__` on a `DataArray` is *indexing*, not projection —
`_projected_names` (`schema.py:430`) must not classify it as `Project`, so the
DataArray proxy should record `__getitem__` as `Opaque` (or the accessor gets a flag
the record path consults). Accessor-equality tests over the README-style chains on a
`DataArray`.

## 6. Nodes carrying arrays are not hashable

> **Closed (2026-07) in [`02-lowering.md`](./02-lowering.md) PR 5**, which is the trigger
> named below. The decision this section asks for, written down: **the claim was
> narrowed**, not the payload made hash-safe. Hashing a `DataArray` means hashing its
> *values*, which for a dask-backed weights array means computing it at plan time — the
> one thing the package promises never to do — so the `Mask` precedent does not transfer
> (that payload is small and already realised). `ir.py`'s module docstring now states the
> conditional claim and `test_ir.py` pins all three behaviours.
>
> One fact this section understated, found while closing it: `==` is affected as well as
> `hash()`, and worse. Two nodes holding *distinct but equal* arrays raise
> `ValueError: truth value of an array … is ambiguous` rather than returning `False`. It
> stays latent for the same reason: a plan holds one payload *object*, and tuple
> comparison short-circuits on identity, so the suite's plan-equality assertions (lowering
> idempotence, the optimiser fixpoint) are sound as written.

`ir.py`'s variants all promise to be "hashable and safe to share between plans", and
`test_ir.py` asserts it — but `xr.DataArray.__hash__ is None`, so
`Opaque(name="weighted", args=(w,))` raises `TypeError` on `hash()` today. Any op whose
payload holds a `DataArray` is affected: `weighted(w)`, a boolean-mask `__getitem__`, a
`where(cond)`. Latent, because nothing in the package currently hashes a node — but the
invariant is written down and untrue, and the hashability tests evidently never exercise
an array-valued arg.

Pick one and write down which: **narrow the claim** in `ir.py`'s docstrings to
"hashable when the payload is" and add a test pinning the actual behaviour, or **make the
payload hash-safe**, for which `indexers.Mask` (`indexers.py:135`) is the precedent — it
stores booleans as a tuple precisely so the enclosing `Select` stays comparable, since an
ndarray would make `==` return an array. Worth closing before
[`02-lowering.md`](./02-lowering.md) PR 5, which introduces a *modelled* node
(`WeightedReduce`) that routinely carries weights.

## 7. Property-suite widening schedule

`test_properties.py`'s generators are deliberately narrowed
(`test_properties.py:10-28`); each workstream retires a narrowing — widen in the same
PR that lands the feature, or immediately after:

| after | widen |
|---|---|
| W4 | add `chunk` calls (mapping and uniform forms) to generated chains |
| W5 | add elementwise ops with scalar args |
| W6 | add `cumsum`/`cumprod`/`diff` |
| W2 phase 1 | assert `emit(to_lower_ir(p))` replays equal to eager, and that `to_lower_ir` is idempotent, over the existing generated chains |
| W3 | assert rewrites survive unknown dim sizes |
| W2 PRs 3–5 | ~~add `groupby`/`resample`/`rolling`/`coarsen`/`weighted` builder chains~~ — **paid in the PR after W2 PR 5** (see below) |
| W2 PR 7 | ~~multi-variable datasets and `__getitem__` projections~~ — **paid in W2 PR 7** (see below) |

> **Paid (2026-07) in W2 PR 7, having first been measured.** Instrumenting
> `pushdown_projections` over 5000 generated chains fired it **0 times**: `_calls` never
> drew `__getitem__`, and `datasets()` built a single variable, so even a projection would
> have been vacuous — the rule's whole question is whether the *projected subset* still
> carries the dims the crossed op names, which needs a variable that lacks one. The gap had
> been open since `pushdown_projections` landed (#32); PR 7's fused arms only inherited it.
>
> `datasets()` now builds a second variable over a proper subset of the dims (the
> `elevation(lat, lon)` shape the hand-written fixtures use) and `_calls` draws list-form
> projections. The rule fires on **10.7%** of generated chains. List form only: a bare name
> yields a `DataArray`, whose downstream algebra is a different thing and whose accessor
> does not exist yet (§5).
>
> It paid for itself immediately — see §8, and the corrected `Project` arm in
> `apply_schema` (a projection *orphans* dims, which that arm used to miss, and a
> hand-written test had encoded the wrong behaviour).

> **Paid (2026-07), one PR late and in one go.** Deferred when W2 PR 5 landed, because PRs
> 3, 4 and 5 had each left it and there was no later fused node to re-pledge it against;
> done immediately after, covering all three fused nodes at once rather than as three
> partial widenings. `builder_plans` now generates the pairs and feeds them to the contract
> properties (optimised equals eager, lowering idempotent, the emit round-trip, tracked
> names, rewrites under unknown sizes), with `_builder_pair` drawing opener and closer as a
> unit — which is what the one-call-at-a-time loop could not do — and a per-kind
> anti-vacuity test so a kind that stopped being generated fails rather than goes quiet.
>
> Three things it turned up, none of them a bug in the package:
>
> - **The size-exactness property cannot be widened**, and shouldn't be: a fused node is
>   entitled to answer "unknown", which is what W3 built `int | None` for. It keeps the
>   plain generator.
>
>   > **Too broad, on review.** True of `GroupedReduce` (mints `None`) and
>   > `WeightedReduce` (`None` for surviving weight dims), but **not** of
>   > `WindowedReduce`, which computes its sizes *exactly* via `_windowed_size`. So
>   > rolling/coarsen chains are size-exact today and excluded from the only property that
>   > would prove it. The narrowing is on the wrong axis: the property should assert
>   > exactness **wherever the schema claims to know** — compare the entries that are not
>   > `None` — rather than restricting the generator by chain kind. That covers the
>   > windowed nodes for free and needs no revisiting when a node later becomes exact. Not
>   > done here: it is its own change, and it carries its own risk of discovering that some
>   > node is less exact than believed.
> - **The tracked-schema property silently assumed no unmodelled op.** True by accident
>   while the generator produced none; an unfusable builder pair is an `Opaque`, past which
>   `apply_schema` is documented to be a guess (`optimize._trusted_prefix`). The assumption
>   is now stated as an `assume` rather than relied upon.
> - **Two generator gaps around non-float data**, reachable from the plain chains alone
>   (`all`/`any` yield bool, `count` yields int) and only surfaced by the longer chains
>   builders produce. Both are xarray/numpy limitations, now filtered in one place with the
>   mapping written down: `median` of a zero-size non-float array raises where a float one
>   answers NaN, and the *windowed* builders' kernels are float-only on bool.
>
> **A fourth, found reviewing the above (2026-07) and fixed in the follow-up.** The
> per-group **map** case — a grouped closer naming dims that *exclude* the group dim, the
> one shape lowering deliberately refuses to fuse (`02-lowering.md` §5.1) — was reachable
> through `groupby` alone. `_closer_dims` drew resample's closer dims from the group dim
> only, so every generated resample pair aggregated, and the `resample` arm of the
> anti-vacuity test's `len(lowered) == 1 or kind in {...}` escape was dead code. Confirmed
> by the `event()` shares before the fix: groupby ~12% `opaque pair`, resample **0%**.
>
> It matters because resample reaches the fusion decision by its *own* route —
> `_grouper_dims` filters `_RESAMPLE_OPTION_KWARGS` out of the kwargs rather than reading
> `args[0]` as `groupby` does — so a change that made it wrongly infer a group dim would
> fuse a map into an aggregation and return the wrong shape, with nothing drawing the case.
> `_closer_dims` now offers every dim to both grouped kinds; both report a nonzero
> `opaque pair` share.
>
> **Forward note.** If the map case is ever modelled (as `GroupedMap` — spec'd in
> [`02-lowering.md`](./02-lowering.md) §5.6, deferred), the size property above must be
> reformulated *before* that lands, not after. The tracked-schema property compares dim
> **names** only (`set(schema.dims) == set(result.sizes)`), and the size property never
> sees builder chains — so a map arm that cribbed `GroupedReduce`'s size intuition and
> gave the group dim the group count instead of its original length would be caught by
> nothing: same names, same replayed data, and a v1 node firing no rules changes nothing
> observable. Only the wrong size would sit in the schema, feeding every consumer
> downstream.

> **The `.reduce` narrowing is paid (2026-07-30, #96).** It read: `.reduce` is excluded
> because its first positional argument is a *function*, which `_reduce_dims` misreads as
> a dim spec, while `operations.py` tabulates it as a reduction like any other — a real
> latent bug, not just a generator limitation.
>
> **Decision: parse it properly**, not untabulate it. `.reduce` genuinely *is* a
> reduction, and making it `Opaque` would have bought safety with a full barrier for an op
> that has an honest dim spec sitting in `args[1]`. Which positional holds the spec is now
> a table, `schema._DIM_ARG_POSITION`, so the next such signature is an entry rather than
> a special case. (`quantile` — which takes `q` first, and which `lower.py`'s weighted
> note already flags — stays untabulated; it is not a reduction we model.) The generator
> draws `.reduce(np.mean, dims)` **positionally**, the shape the bug lived in: 3.1% of
> chains.
>
> **A live bug found next door, fixed in the same PR.** `keepdims=True` *keeps* every
> reduced dim at size 1, but `_reduce_dims` recorded `consumes` regardless, so
> `dim_effect` classified a following select as indexing a reduced-away dim and the
> optimiser **raised `InvalidExpressionError` on a chain that works eagerly**
> (`ds.plan.mean("time", keepdims=True).isel(time=0)`). It affects every tabulated
> reduction, not just `.reduce`. Now recorded `Opaque`.
>
> Modelling it rather than refusing it is spec'd in §9 below (#117) — the shape is
> `WindowedReduce`'s, and `consumes=frozenset()` alone is **unsound**, which was checked
> rather than reasoned: it empties `blocks`, the select hops to the front, and the replay
> raises where eager returned data.

The invariant asserted is always the same and is the project's crown jewel:
`ds.plan.<chain>.collect()` equals the eager chain, for generated datasets and
chains, plus idempotence of `optimize`.

## 8. Projection pushdown eliminates discarded work — errors included

Found by the property widening in §7 (2026-07, during W2 PR 7). Raised as a possible
divergence, **settled as a deliberate and desirable one**, and specified here so it is a
property rather than an accident.

`pushdown_projections` moves a projection earlier, so variables the plan throws away are
never computed. Usually that only saves time. Occasionally it also skips an *error*:

```python
ds = xr.Dataset({"temperature": (("time", "lat"), ...),    # float, has time
                 "station":     ("lat", ["alpha", ...])})  # str, no time

ds.std("time")[["temperature"]]                  # TypeError, raised by `station`
ds.plan.std("time")[["temperature"]].collect()   # succeeds
```

**This is the better answer, not a bug.** The projection says outright that `station` is
not wanted; eager computes its standard deviation anyway, purely because it happens to be
in the Dataset, and falls over doing it. The rewrite computes exactly what was asked for.
So the optimiser turns a footgun into an answer — a small win, and the kind a plan-then-
execute design is *for*.

What makes it safe rather than merely convenient is that the values can't move. The rule
only fires once the guard has established that the projected variables carry the dims the
crossed op names, so the surviving variables are reduced identically either way; the sole
observable difference is whether a discarded variable raised on the way. Verified against
evaluation, and pinned by `test_projection_pushdown_skips_an_error_from_a_discarded_variable`.

### The contract, sharpened

The crown-jewel invariant is usually stated as *optimised equals eager*. Strictly it is:

> `optimize` preserves the **values** of everything the plan asks for. It may additionally
> avoid an error raised by a computation whose result the plan discards. It may never
> change a value, nor introduce an error.

Only `pushdown_projections` exercises the middle clause today. A new rule may rely on the
first and third; it may not rely on failures being preserved.

The third clause has been violated exactly once, and it is worth recording because the
clause is what made it a bug rather than a preference. `pushdown_selects_past_rechunks`
hoisted a select in front of `chunk({dim: "auto"})`, handing dask a zero-size array to
divide by: the optimised plan **raised where the eager chain succeeded** — no discarded
computation, no footgun turned into an answer, just a new error. Issue #121, fixed
2026-08-01 by making the extent-dependent chunk specs a barrier
([`04-chunk-taxonomy.md`](./04-chunk-taxonomy.md)). The property
`test_no_rule_lands_a_select_before_an_extent_sized_rechunk` now asserts it of the whole
optimised plan rather than of the one rule, since the clause binds every rule.

### What triggers it

- **A string variable.** The example above: numpy will not take a standard deviation of
  `<U4` data, so a Dataset `std("time")` raises `TypeError: the resolved dtypes are not
  compatible with add.reduce` while reducing a variable the projection discards. Plain
  xarray/numpy semantics, owned upstream and stable, which is why the hand-written test
  uses this trigger.

- **`std`/`var` over an empty axis set — real for us, but the culprit is numbagg.**

  **Corrected (2026-07).** This bullet read "arguably an upstream bug — `std(dim=[])`
  should be a no-op — present in both supported xarray versions". Wrong three times, and
  worth restating because it changes how much weight §8 can put on this trigger.

  Every step up to the failure is *correct*. A Dataset reduce is applied per variable, and
  a variable carrying none of the named dims is reduced over the **empty axis set** — which
  is exactly how such a variable survives a `ds.mean("time")` untouched.
  `namedarray.core.reduce` then computes the surviving dims, which for an empty axis set is
  all of them (`('lat',)`), and finds a 0-d array where a 1-d one was promised
  (`dimensions ('lat',) must have the same length as … ndim=0`).

  That 0-d array comes from **numbagg**, which reads `axis=()` ("reduce over no axes") as
  `axis=None` ("reduce over all of them"):

  ```python
  np.nanstd(a, axis=())         # [0. 0. 0.]   correct, shape preserved
  numbagg.nanstd(a, axis=())    # 1.0          a scalar
  ```

  So xarray's error message is xarray *catching* numbagg's mistake, and
  `xr.set_options(use_numbagg=False)` makes it go away. The right answer is also **zeros,
  not a no-op**: reducing over nothing is the identity for `sum`/`mean`, but zero for
  `std`/`var`.

  That is why only these two break, and the reason inverts the old reading.
  `duck_array_ops._create_nan_agg_method` short-circuits `axis == ()` for every reduce that
  *is* the identity there (`invariant_0d=True`: `max min sum median prod mean cumsum
  cumprod`), returning before dispatch. `std`/`var` are excluded **correctly** — the
  shortcut would be wrong for them — so they are the only two that reach numbagg, and they
  reach it *because* xarray classified them right. numbagg is equally wrong for
  `nanmean`/`nansum`/`nanmedian`; the fast path merely masks it.

  **Consequence: this trigger is environment-conditional, not version-conditional.** It is
  present in any xarray with numbagg installed and absent in any without. We have numbagg
  because `rolling_exp` needs it (`pyproject.toml`), so it is real for our CI — but a
  numbagg release could retire it, which is why neither §8's example nor its test depends
  on it any more.
- **`weighted`** is the same shape, and the sharpest instance of it: a weighted reduce
  *refuses* a variable lacking a named dim where a plain one merely skips it, so
  `ds.weighted(w).mean("time")[["temperature"]]` raises eagerly over a discarded variable
  lacking `time` and succeeds once the projection runs first.

  **Decided (2026-07, W2 PR 9): admitted.** The error-masking argument that had excluded
  `WeightedReduce` from `pushdown_projections` is exactly what this section retires, and
  the other argument — subsetting the weights — belongs to the *select* rule
  ([`02-lowering.md`](./02-lowering.md) §8.1), not this one: a projection discards
  variables rather than subsetting them, so the weights are untouched.

  It is **not** unconditional, and the condition was found by probing xarray rather than by
  reasoning. A projection drops a dim *coordinate* that no surviving variable uses, and
  what the weights do with a dim turns on whether its coord is there — inner-join against
  it if so, broadcast a fresh one if not. So with weights on `lat` and a projection to a
  variable spanning only `time`, the two orders disagree in **values and coords**, not just
  in errors (an uncoordinated `lat` loses its coord entirely; a *disjoint* `lat` diverges
  outright). The rule therefore requires `consumes | weight_dims` of the projected
  variables, which is what `WeightedReduce.weight_dims` was added for in W2 PR 5.

  A bare closer (`ALL_DIMS`) needs no such term and could not carry one: it clears every
  dim including minted weight dims, so nothing survives for a coord to be missing from,
  and a weight dim the dataset lacks could never be a subset of the projected variables'
  dims.

  Verified against the contract above rather than against the guard's own logic: 512
  weighted chains (4 variable layouts × 8 weights × 4 closers × every projection) driven
  end to end through `ds.plan`, of which 278 hopped. **No value changed, no coordinate
  changed, and no error was introduced.** 285 came back identical to eager; 191 raised both
  ways; the 36 that skipped an error each returned *exactly* the projection-first answer,
  and every one of those errors was traceable to a discarded variable — including a
  `MergeError` where disjoint weights gave the kept variable `lat` size 0 and the discarded
  one size 2, so combining them conflicted.

  The property suite covers the hop but not this last clause: its generator constrains the
  weighted closer so no variable raises, because `optimised == eager` cannot express "eager
  raises and that is fine" — the same reason it filters `EMPTY_AXIS_UNSAFE_REDUCES`. The
  hop still fires on ~21% of generated weighted-plus-projection plans, all asserting
  equality with eager.

### A note on the property suite

`test_properties.py` asserts `optimised == eager`, which cannot express "eager raises and
that is fine". Its generator therefore filters the `std`/`var` trigger
(`EMPTY_AXIS_UNSAFE_REDUCES`) — excluding chains whose *eager reference* is broken, not
hiding anything about the optimiser. The behaviour itself is pinned by the hand-written
test named above. Should the property ever be widened to cover it, the assertion has to
become the sharpened contract, not the equality.

## 9. Model `keepdims=True` instead of refusing it

Found while closing §7's `.reduce` narrowing (2026-07-30, #96), where the *live* half of
the bug was fixed by recording `keepdims=True` as `Opaque`. This section is the modelling
that refusal defers; filed as **#117**.

A `keepdims=True` reduce consumes nothing and mints nothing — it can only **resize**, and
only the dims it names, each to 1. That is exactly `WindowedReduce`'s dim effect, and the
table is short:

| | `Reduce` today | `keepdims` modelled |
|---|---|---|
| `blocks` | `consumes` | the named dims — *already right* |
| `requires` | `consumes` | the named dims — *already right* |
| `on_conflict` | `"invalid"` | `"immovable"`: the dim survives, so the chain is valid and merely un-reorderable |
| `apply_schema` | removes the dims | keeps them, sizes → 1 (exactly known, not `None`) |

**`consumes=frozenset()` alone is unsound**, and this was checked rather than reasoned.
It empties `blocks`, so `pushdown_selects` hoists a select on the named dim to the front;
a scalar `isel` *drops* that dim, and the replayed reduce then raises
`ValueError: Dimension(s) 'time' do not exist` where the eager chain returned data. The
`DimEffect` docstring already states the rule this violates: `blocks` must cover
everything the node consumed, minted **or resized**.

So the work is two `match` arms rather than a new field — *provided* `consumes` is allowed
to mean "the dims this reduce names" rather than "removes", which is the kind of untrue
written claim §7's own bug was. Do it with a **derived** `keepdims` property on `Reduce`
(the `Project.single` precedent) and a `consumes` docstring that states both modes.

Two things not to miss:

- **`lower.py` needs a fusion refusal.** The three `_fuse_*` arms read `closer.consumes`,
  so a `groupby(...).mean(keepdims=True)` closer would fuse into a `GroupedReduce` whose
  semantics are wrong. `Opaque` gets this right for free today — an opaque closer never
  fuses — so the guard must land *with* the modelling, not after it.
- **The property generator should draw it**, per §7's schedule: `keepdims` is a kwarg on
  every tabulated reduction, and the size-exactness property is where a wrong arm would
  show up (the dims survive at 1, which the tracked schema must say exactly).

## 10. A projection feeding a coordinate need not compute its input

> **Closed (2026-08-10)** as `optimize.eliminate_projection_before_coord` (#115), the
> follow-up §1 filed and deliberately scoped out of the merge rule.

`ds[["temperature"]]["lat"]` materialises `temperature` only to throw it away and return
the `lat` coordinate; the optimal plan reads `lat` from the base and never touches the
data variable. `merge_adjacent_projects` (§1) cannot do this — its subset test
`set(p2) ⊆ set(p1)` fails, and it must, because that rule is *syntactic* and `ds["lat"]`
is a **different expression** unless the schema proves `lat` passes through `p1`
untouched. Proving it is a schema question, so the rule lives in `pushdown_projections`'
family: confined to `_trusted_prefix`, reading the folded schema.

For an adjacent `(Project p1, Project p2)` pair, drop `p1` iff, in the schema entering
`p1`: `p1` is not `single` (the `DataArray`-indexing barrier §1 carries); every name `p1`
selects is a tracked data variable (so its spanned dims are known, not guessed); and
**every name `p2` selects is a coordinate whose dims are a subset of the dims `p1`'s
variables span** — the exact survival test `apply_schema`'s `Project` arm applies
(`schema.py`). A projection never touches a coordinate's *values*, so a surviving
coordinate reads identically from `p1`'s result and the base.

The survival test is load-bearing, not an optimisation: it fires only when the coordinate
is present in `p1`'s output — exactly when the eager chain succeeds. Where the coordinate
does not survive (`ds[["elevation"]]["time"]`, `elevation` lacking `time`) the eager chain
raises and the rule **declines**, never turning that error into a value (§8's third
clause). Adjacency-only by design: a select or reduce between the two projections (the
divergence case #115 names) means they are not adjacent, so the rule structurally cannot
fire — which is why adjacency is the safe condition. The rule shrinks the plan, so the
termination measure holds on its first component, the same footing as §1. Goldens in
`test_optimize.py` (collapse, dropped-coord left, past-`Opaque` left, `single`-`p1` left)
plus equality-vs-eager in `test_accessor.py` (collapse over a dim and an auxiliary coord,
and the select-between divergence declined).

The soak fires on no random chain — `_calls` draws only data-variable projections — so a
dedicated `coord_projection_plans` generator plus an anti-vacuity + value test
(`test_a_coord_projection_drops_its_input_and_replays_equal`) asks for the pair by name.
It is kept separate rather than folded into `_calls` for two reasons: a coordinate
projection must be *terminal* (it yields a coord-only dataset with no legal downstream op),
and its chains cannot feed `test_tracked_schema_agrees_with_evaluation`, because
`apply_schema`'s `Project` arm over-reports a coordinate projection.

> **Follow-up, not folded in:** that over-report is **#162**. A projection naming a
> coordinate falls into the `Project` arm's "decline" leg even though the coord is known,
> so the tracked schema claims a data var survives that evaluation drops. Inert for this
> rule (it reads only the schema *entering* `p1`), but a schema-lie of the #60/#109 family
> — and modelling it exactly is what would let the soak generator join `any_plans()`
> without an `assume`.
