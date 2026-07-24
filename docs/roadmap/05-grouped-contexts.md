# W5 — Grouped and windowed contexts: the first sub-plan variant

*(A design memo in the tradition of `structural-dispatch.md` — this settles a shape,
records the reasoning, and ends with a staged PR plan. It should be reviewed before
any implementation starts. It builds on the W1 barrier
([`01-grouped-barrier.md`](./01-grouped-barrier.md)) being merged, which makes grouped
chains correct-but-unoptimised; this workstream is what lifts "unoptimised".)*

## 1. The problem, and why it earns a variant

`ds.plan.groupby("time.month").mean().isel(lat=0)` is the canonical climatology
pattern: group, aggregate, then look at a region. The profitable rewrite is obvious —
`isel(lat=0)` first, so the groupby machinery runs over one latitude instead of all of
them — and it is exactly the class of win this project exists for. After W1, the chain
is *correct* but the optimiser refuses to touch it: everything from `groupby` on is
`Opaque`.

Modelling it meets §7.4's bar for a new variant with room to spare: a grouped
aggregation carries **genuinely new structural data** — the grouper (which dim it
consumes, which new dim it mints) *and* an inner operation applied within the grouped
context. That second part is the novelty: it makes this the IR's first
**sub-plan-carrying node**, and per the 2026-07 decision (tree/DAG is in the long-term
vision), its shape should be chosen as the deliberate first step toward the tree, not
an ad-hoc special case.

## 2. The shape: `Contextual` with a single inner op

The key observation that keeps this tractable: **xarray's own API closes the context
after one aggregating call.** `ds.groupby(x)` returns a `DatasetGroupBy`; the next
method (`.mean()`, `.sum()`, `.map(f)`, ...) returns a Dataset. The same holds for
`resample`, `rolling`, `coarsen` and `weighted`. There is no user-visible chain *of
Dataset ops* living inside the context — the "sub-plan" is, in practice, one op.

So the variant carries one inner `Op`, not a `Plan`:

```python
@dataclass(frozen=True)
class Contextual:
    """A context-returning call (groupby/resample/...) plus the single op applied
    within it — replayed as ``getattr(ds, name)(*args, **kwargs)`` then the inner op.

    The IR's first non-flat node: ``inner`` is itself an :data:`Op`. Deliberately a
    single op rather than a sub-plan — xarray's API closes a grouped context after
    one aggregating call — but typed so the later promotion (``inner: Op`` →
    ``body: tuple[Op, ...]`` → plan-typed children) is additive.
    """

    name: Literal["groupby", "resample"]          # start closed; widen deliberately
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
    inner: Op = ...                                # the op applied to the GroupBy
    group_dim: Hashable = ...                      # the dim the grouper consumes ("time")
    new_dim: Hashable | None = ...                 # the dim the aggregation mints ("month")
```

Design notes:

- **Scope: `groupby` (+ `groupby_bins`) and `resample` first.** They share the
  "consume one dim, mint another" structure. `rolling`/`coarsen`/`weighted` are
  *windowed/weighted*, not grouped — different dim algebra (rolling keeps its dim,
  coarsen resizes it) — and stay behind the W1 barrier until this shape has proven
  itself. Widening `name`'s `Literal` later is the cheap part.
- **`group_dim` / `new_dim` are the semantic fields** the rules match on, resolved at
  record time (§4). For `groupby("time.month")`: `group_dim="time"`,
  `new_dim="month"`. For `resample(time="2D")`: both are `"time"`.
- **`inner` is a full `Op`**, usually a `Reduce` minted by the *grouped* record path
  (§3) — so the existing `Reduce` machinery (name, verbatim args/kwargs, `consumes`)
  is reused rather than duplicated. An inner the recorder can't model (`.map(f)`)
  records as `Opaque` inside the `Contextual`, which simply makes the node
  unrewritable — same posture as everywhere else.
- **Replay stops being a pure one-call-per-node passthrough** — the honest cost.
  `_replay` (`accessor.py:201-209`) gains one branch: a `Contextual` replays as the
  context call then the inner call, *each* still verbatim
  (`getattr(getattr(ds, name)(*args, **kwargs), inner.name)(*inner.args,
  **inner.kwargs)`). Per-call verbatim-ness — the property §7.5 actually defends — is
  preserved; only "one node, one call" is given up, and only for this variant.
  `_format_node` (`accessor.py:192`) needs a matching branch
  (`groupby('time.month').mean()` as one line).

### Why not the alternatives

- **A flat `GroupedReduce` variant** (grouper fields + reduce fields mashed into one
  record) — re-creates exactly the every-field-on-every-node shape the sum type
  removed, and duplicates `Reduce`'s header. Rejected.
- **A full sub-plan (`body: tuple[Op, ...]`) now** — models chains xarray's API cannot
  produce, and every rule would have to reason about a list-in-a-list for no present
  payoff. The single-`Op` form is forward-compatible (promoting a field from `Op` to
  `tuple[Op, ...]` is additive, the same argument §7.6 made for the tree). Rejected
  for now, on §7.4's "variants are earned by data the optimiser must reason about"
  grounds — no rule today reasons about a grouped *chain*.

## 3. The record path

The W1 barrier is *replaced for the modelled names* and kept for the rest:

- `LazyDatasetProxy.__getattr__` on `groupby`/`groupby_bins`/`resample` returns a
  **`_ContextProxy`** — a tiny recorder holding the parent proxy plus the context
  call's name/args/kwargs. It is not a `LazyDatasetProxy`; it deliberately supports
  only method calls.
- The next method call on `_ContextProxy` builds the inner `Op` via `to_opnode`
  against the *pre-context* schema (a grouped `mean("lat")` still consumes `lat`; a
  bare grouped `mean()` reduces the group's dims — see §4 for what that means), wraps
  both into a `Contextual`, appends it via the parent's `_record` machinery, and
  returns a normal `LazyDatasetProxy`. The chain is back in Dataset-land, exactly
  mirroring xarray.
- Unmodelled context names (`rolling`, `coarsen`, `weighted`, `rolling_exp`,
  `cumulative`) keep the W1 barrier behaviour unchanged.
- Attribute access / `__getitem__` / terminals on `_ContextProxy`: not supported in
  v1 — raise `AttributeError` with a message pointing at eager xarray. (The W1
  barrier's catch-all is *lost* for the modelled names, so this must be a loud error,
  never silence.)

## 4. Schema: what `apply_schema` may know post-group

The hard question. `groupby("time.month").mean()` removes `time` and mints `month`
with **size = number of distinct groups** — knowable only from coordinate *values*.

Position: **dim-coordinate values are fair game at record time.** xarray holds
dimension coordinates as loaded pandas indexes (they back label lookup); reading them
materialises nothing dask-shaped. `SchemaState.from_dataset` (`schema.py:59-70`)
already reads `.sizes`/`.coords`; reading `ds.indexes[dim]` is the same class of
metadata access. So:

- `groupby("time.month")` → `new_dim="month"`, size =
  `len(np.unique(ds.indexes["time"].month))` — resolved at record time, threaded into
  `apply_schema`'s new `Contextual` arm.
- A grouper over a **non-dim (possibly lazy) coordinate**, or a `groupby_bins` whose
  bin count isn't statically evident: fall back to **size unknown**. `SchemaState`
  has no "unknown size" today — extend `dims`' value type to `int | None` with `None`
  meaning *don't know*, and audit the three consumers of sizes
  (`Select`-arm resizing at `schema.py:126-128`, `Indexer.size`) to propagate `None`
  conservatively. This is the memo's largest open sub-design; it gets its own PR and
  its own tests, and `var_dims`' documented `None`-discipline (`schema.py:73-78`) is
  the template for how the ambiguity must be handled.
- `apply_schema`'s `Contextual` arm otherwise: drop `group_dim`, add `new_dim`,
  apply the *inner* op's effect to the remaining dims (a grouped `mean("lat")` also
  removes `lat`), keep `data_vars` (grouped reductions are per-variable).

## 5. The rewrites it unlocks

**The headline — select pushdown past a grouped reduce.** Fires on a
`(Contextual, Select)` adjacency; the select hops left when its dims are disjoint from
*all three* of: `{group_dim}`, `{new_dim}`, and `inner.consumes` (for a `Reduce`
inner; any other inner blocks the hop). `ds.plan.groupby("time.month").mean()
.isel(lat=0)` → `isel(lat=0)` first. Intersecting → *leave, never raise* (selecting
on `month` after the aggregation is valid and simply not moveable — the scan
discipline, not the reduce one). Same single-hop shape and termination argument as
every existing pushdown rule.

**Second — projection pushdown past a grouped reduce.** Extend `pushdown_projections`
(`optimize.py:413-419`): `case Contextual(inner=Reduce()) as ctx:` with
`needed = ctx.inner.consumes | {ctx.group_dim}` — projected variables must carry the
grouped dim and the reduced dims, checked against the entering schema like every other
arm. `ds.plan.groupby("time.month").mean()[["tas"]]` → project first.

Both rules read only the semantic fields (`group_dim`/`new_dim`/`inner.consumes`),
keeping the eventual FFI seam clean (`structural-dispatch-2.md` §5).

## 6. What doesn't transfer (the honesty beat)

- **`rolling`/`coarsen`/`weighted` are not covered** by this shape's dim algebra and
  stay barriered. Each needs its own memo section when its turn comes (rolling keeps
  its dim like a scan; coarsen divides sizes; weighted composes with a reduce).
- **The tree is still not here.** `Contextual.inner` is a child *op*, not a child
  *plan with its own inputs* — binary ops (`merge`/`concat`/`ds1 + ds2`) remain
  exactly as deferred as §7.6 left them. This memo spends the "first non-flat node"
  budget on the case with a proven optimisation behind it.
- **Unknown group counts** make downstream size reasoning vanish (§4). Every rule
  must already tolerate `None` sizes after that change; the select/projection rules
  above depend only on dim *names*, so the headline rewrites survive unknown sizes —
  worth a property test asserting exactly that.

## 7. Staged PR plan (after this memo is reviewed)

1. **Memo review** — settle §2's field set, §4's `int | None` sizes, and the v1 scope
   (`groupby`/`groupby_bins`/`resample`).
2. **`SchemaState` sizes → `int | None`** — self-contained, no behaviour change for
   all-known plans; tests for the propagation discipline.
3. **`Contextual` variant + record path** for `groupby`/`resample`, replacing the W1
   barrier for those names only. No rules yet: behaviour identical to W1's
   (verbatim replay), proven by the W1 regression suite staying green plus new
   goldens on the recorded shape. `apply_schema` arm included (this is where
   `assert_never` forces every dispatch site to be visited).
4. **The select pushdown rule** + goldens + equality-vs-eager + property widening.
5. **The projection pushdown arm** + same test treatment.
6. **Assessment note:** what it would take to widen to `rolling`/`coarsen`/
   `weighted`, written as an addendum to this memo once 3–5 are in.

Each PR ships green under `pixi run python -m pytest tests -q`, `pixi run mypy`,
`pixi run python -m ruff check src tests`.
