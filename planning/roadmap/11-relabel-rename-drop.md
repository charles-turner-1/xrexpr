# W11 — A `relabel` schema primitive, and `Rename`/`Drop` nodes

*(A design memo in the tradition of [`09-grouped-contexts.md`](./09-grouped-contexts.md):
it settles a shape, records the reasoning, and ends with a staged PR plan. Review before any
implementation starts. It builds on W2's lowering stage and the fused
`GroupedReduce`/`WindowedReduce`/`WeightedReduce` nodes, and on W7 §9/#117 having modelled
`keepdims=True` for plain reduces.)*

> **v1 scope**: the `_relabelled` schema primitive plus `Rename` and `Drop` nodes for the
> common `.rename()` / `.drop_vars()` ops. The `groupby(...).mean(keepdims=True)` synthesis is
> **deferred** to a follow-up (design recorded in §6). `Rename`'s `dim_effect` starts
> **conservative**; name-translating pushdown is flagged as a large future unlock, not v1.

## 1. The problem, and why it earns new nodes

Three ordinary xarray operations are, today, **`Opaque` trust boundaries**:

- `.rename(...)`
- `.drop_vars(...)` (and coordinate-dropping kin)
- the retain-the-group-dim behaviour of `groupby(...).mean(keepdims=True)`

`apply_schema` models an `Opaque` node as *variable-preserving* (the `pass` arm, `schema.py`),
a deliberate guess the code itself calls "not true in general (`rename`/`drop_vars`/`assign`)".
Because the guess can be wrong, the optimiser treats the first `Opaque` as a **trust boundary**
(`optimize._trusted_prefix`) and goes dark on everything downstream of it. So every `.rename()`
/ `.drop_vars()` a user writes doesn't just mis-predict its own output — it **blacks out
optimisation for the rest of the pipeline**. Modelling them re-opens the trusted prefix across
a common class of ops, which is the win this workstream exists for.

### The provoking case

`groupby("time.month").mean(keepdims=True)` is genuinely meaningful — unlike the rolling
`keepdims` no-op (an upstream xarray issue: `rolling().mean()` silently swallows `keepdims`
via `**kwargs`). Verified against xarray 2026.7.x:

| call | result | in xrexpr today |
| --- | --- | --- |
| `groupby(month).mean()` | `(month:2, lat:3)`, `month` dim-coord | modelled → `GroupedReduce` |
| `groupby(month).mean(keepdims=True)` | `(time:2, lat:3)`, **no coord** | refused → `Opaque` |

`keepdims=True` **retains the group dim** (`time`, resized to the group count) and **drops
its coordinate**, instead of minting `month`. That is the *opposite* of what `GroupedReduce`
models (remove the group dim, mint `new_dim`), which is why `_fuse_grouped`'s `closer.keepdims`
guard correctly refuses to fuse it. Refusing is safe — compute still replays real xarray — but
it plants an `Opaque` boundary.

### Why not just a `Rename` node

The keepdims result is **not** reachable by a faithful `.rename()`. Renaming carries the
coordinate along; `keepdims` sheds it. Verified:

| operation | result | == `keepdims`? |
| --- | --- | --- |
| `rename({month: time})` | `time` **dim-coord** (coord followed the rename) | ✗ |
| `rename_dims({month: time})` | `time` dim + `month` **non-dim coord** (coord demoted, kept) | ✗ |
| `rename_dims + drop_vars("month")` | `time` dim, no coord | ✓ |
| `rename + drop_vars("time")` | `time` dim, no coord | ✓ |

Renaming a coordinate never *drops* it — it carries it under the new name or demotes it to a
non-dimension coordinate. So the synthesis is **relabel-the-dim + drop-the-coord**, two pieces
— which is exactly why the right abstraction is a small primitive vocabulary, not a monolithic
`Rename` node.

## 2. The core idea — a shape-preserving `relabel` schema effect

The schema effect layer already thinks in **drop** (`_aggregated`) and **mint** (`_minted`)
primitives. What is missing is **relabel**.

Conceptually, renaming label `x` of shape `A` ≡ *drop `x`, create `y` of the same shape `A`*.
At the **schema** level this is exact, because the schema tracks structure, not values — there
is nothing to reconstruct.

**But it must be ONE primitive, not two composed:**

1. `_minted` is an *addition* that broadcasts a new dim onto **every data variable** and skips
   coords — its own docstring spells this out: "an addition to each variable, not a
   substitution of one dim for another". `_aggregated` *drops* the coord. Composing
   `_aggregated({month})` then `_minted({time})` gives the wrong schema, not a rename.
2. House style is **by-construction invariants over drift tests**. Expressing rename as
   `drop(x, A)` + `mint(y, A)` makes shape `A` an input stated *twice* that a test must keep
   equal. A single relabel primitive that *reads* the source label's shape and reassigns it
   keeps "shape-A-in = shape-A-out" structural — unbreakable.

The **"dims are derived" invariant** (`SchemaState.__post_init__`: a dim exists only because
some variable spans it; `sizes` is pruned to spanned dims) makes the dim case fall out for
free: relabelling a *dim* is just "rewrite `old → new` in each variable's dims-tuple, move
`sizes[old] → sizes[new]`" — no separate dim registry to keep in sync.

### The effect helper

A `_relabelled` helper joining `_aggregated` / `_minted` / `_keepdims_reduced`, handling the
three targets a rename can hit:

- **dim**: substitute `old → new` in every `variables` dims-tuple where present; move the
  `sizes` entry (carrying `None` — unknown — through unchanged). Dim existence follows from
  `variables`.
- **variable / coord**: rekey `variables` (and `coord_names` if it was a coord).
- xarray's `.rename()` on a dim-coord does both at once; the helper composes the two.

Unlike `_minted` / `_aggregated` / `_keepdims_reduced`, which touch only the dim *tuples* and
membership, `_relabelled` touches the **keys** of `variables`/`coord_names` — a genuinely new
kind of edit, which is the concrete reason it is a new primitive rather than a reuse.

## 3. What gets built

Today `.rename()` / `.drop_vars()` carry no `OP_TABLE` row, so `op_spec(name)` returns `None`
and `to_opnode` hits its fall-through arm → `Opaque`. Making them modelled means threading a
new node type through the same handful of closed `match`es every existing node passes through.
The checklist, from the existing shapes (`Reduce`, `GroupedReduce`):

- **spec + registry**: a new `OpSpec` variant (or reuse) + `_SPECS`/`OP_TABLE` rows in
  `operations.py` so `op_spec("rename")`/`op_spec("drop_vars")` resolve.
- **IR node**: a frozen dataclass with the common header (`name` + verbatim `args`/`kwargs`,
  coerced in `__post_init__`) plus semantic metadata — a `{old: new}` mapping for `Rename`, a
  `variables` tuple for `Drop` — and membership in the `Op`/`FluentOp`/`LoweredOp` unions.
- **record**: a `to_opnode` arm producing the node from the verbatim call.
- **schema effect**: an `apply_schema` arm calling the new `_relabelled` / the existing
  `_aggregated` drop. This is the arm that moves the node out of the `Opaque` no-op bucket.
- **dim effect**: a `dim_effect` arm returning a `DimEffect`, so the dim-name pushdowns know
  what commutes past it. See the rename caveat in §4.
- **emit/replay**: `_emit_node` emits **one** verbatim `Call`; `_replay` re-invokes
  `getattr(obj, name)(*args, **kwargs)` unchanged. Replay is byte-identical to the `Opaque`
  path — the only thing that changes is that the schema is now modelled and the optimiser can
  see across it.
- **explain**: an `_annotate` arm stating the derived effect the surface call obscures —
  `f"{old} -> {new}"` for `Rename` (mirroring the `GroupedReduce` arm), a `_dim_set`-style
  `"drops {…}"` for `Drop`.

### The `assert_never` contract makes this safe by construction

Every dispatch `match` over the IR union (`to_opnode`, `apply_schema`, `dim_effect`,
`_emit_node`, `_annotate`) closes with `case _: assert_never(node)`. Adding a variant to the
`Op`/`FluentOp`/`LoweredOp` unions therefore **fails `mypy` at every site until each is
handled** — the compiler enforces "answer every question at once" rather than a drift test
catching a missed arm later. This is the house-style spine of the change: the node cannot
half-exist, and the checklist above *is* the complete set of arms.

### v1 deliverables

1. **`_relabelled` schema effect** (`schema.py`, joining `_aggregated`/`_minted`) + a `Rename`
   node recorded for user `.rename()`. Conservative `dim_effect` (§4). Replay a verbatim
   `.rename()`.
2. **`Drop` node** for `.drop_vars()` (and coord-dropping kin), reusing the `_aggregated`
   drop. Dim-transparent (no dim remap), so no `dim_effect` subtlety. Replay a verbatim
   `.drop_vars()`.

## 4. Caveat: `Rename` remaps dim names, so it is not dim-transparent

Unlike `Elementwise` (transparent — every dim survives), a `Rename` that touches a **dim**
changes dim *names* mid-pipeline. The dim-name-keyed pushdowns (`merge_adjacent_selects`,
select/project commuting via `dim_effect`) reason about names and do **not** translate them
across a node. So a downstream `sel(time=…)` cannot naively hop above a `rename(month→time)` —
above it the dim is still `month`.

**v1 decision: conservative `dim_effect`.** The renamed dim names act as a barrier for
name-keyed pushdowns (treat them like the opaque effect for those dims only); untouched dims
stay transparent. Simple, correct, and it still delivers the primary win — modelling rename
re-opens `_trusted_prefix` so schema folding and `pushdown_projections` see across it. `Drop`
of a coord has no dim remap, so it is fully transparent with no such caveat.

**Flagged future unlock (not v1): name-translating cross.** Teaching the pushdown rules to
rewrite indexer keys across a rename (so a `sel(time=…)` becomes `sel(month=…)` as it hops
above `rename(month→time)`) would let selects/projections commute *through* the renamed dim —
turning rename from a soft barrier into a fully transparent relabel for the optimiser. It
carries real surface area and edge cases (multi-key renames, chained renames, conflicts), so
it earns its own design pass once v1 lands and we can measure whether real pipelines want it.

## 5. Separation of concerns to hold onto

- **Schema effect** (drop / mint / relabel) = how the schema *models* the transform.
- **Replay** = real `.rename()` / `.drop_vars()` on the concrete object. You cannot
  drop-and-mint *values* at runtime; the primitive story is schema-only.

## 6. Deferred follow-up: the groupby-`keepdims` synthesis

Not in v1 — recorded here so the design is not rediscovered. Once `_relabelled` + `Drop`
exist, teach `_fuse_grouped`, instead of refusing to `Opaque`, to emit the chain

```
GroupedReduce(consume time, mint month)  →  relabel dim month → time  →  drop coord month
```

for the bare-consumes keepdims case only; the `mean("lat", keepdims=True)` map variant
(`(time:6, lat:1)`) is a different transform and stays refused.

**Settled design (chosen over the alternative):** the synthesis lives in **lowering, emitting
a three-node chain** — *not* a new `keepdims` variant on `GroupedReduce`. This keeps
`GroupedReduce` honest to its single "consume group, mint new" meaning. The cost to absorb:
`_fuse_grouped` would emit a *multi-node chain* from one recorded pair, whereas every `_fuse_*`
sibling today is 1-pair → 1-node, so `to_lower_ir`/`_emit_node` (which assume single/double-call
nodes) need to accommodate a fusion that expands to three lowered nodes. Confirm that shape is
clean before starting.

## 7. Non-goals (out of scope entirely)

- The `mean("lat", keepdims=True)` groupby *map* variant — a different transform, stays
  refused to `Opaque`.
- The rolling/coarsen/weighted `keepdims` guards — rolling is a genuine no-op, coarsen/weighted
  raise `TypeError`; leave those guards as-is.
- Any change to replay semantics.

## 8. Staged PR plan (after this memo is reviewed)

1. **Memo review** — settle §2's `_relabelled` shape, §3's node field sets, and §4's
   conservative-`dim_effect` decision.
2. **`_relabelled` + `Drop`** — the dim-transparent half first: the `_relabelled` helper, the
   `Drop` node (reusing `_aggregated`), full `to_opnode`/`apply_schema`/`dim_effect`/
   `_emit_node`/`_annotate` arms, `.drop_vars()` recorded. `assert_never` forces every dispatch
   site to be visited. Round-trip goldens + property-suite widening.
3. **`Rename`** — the node, its `_relabelled`-backed `apply_schema` arm, and the **conservative**
   `dim_effect` barrier. Tests: round-trip equality, "projection now pushes across a rename"
   (trusted prefix stays open), and "a `sel` on the *renamed* dim does not hop above the
   `Rename`" (the v1 barrier holds).
4. **Follow-up (separate memo/PR):** the groupby-`keepdims` three-node synthesis (§6), and —
   independently — the name-translating pushdown unlock (§4), each with its own justification.

Each PR ships green under `pixi run test`, `pixi run mypy` (the `assert_never` sites are the
real gate — mypy failing is the checklist telling you which arm is missing), and
`pixi run ruff`.

## 9. Verification (what the property suite already pins)

The invariants are already pinned; the work is to bring the new nodes under them.

- **Schema exactness** — `test_tracked_schema_agrees_with_evaluation` (`tests/test_properties.py`).
  Hypothesis over `any_plans()`; `assume`s no `Opaque` remains, then compares the folded schema
  to a real evaluation for **exact** dim names, **exact** coords (not subset), and each
  variable's exact dims. Register `Rename`/`Drop` in the `any_plans()` generators so these plans
  now clear the no-`Opaque` `assume` and must still agree.
  `test_every_builder_kind_is_generated_and_replays_equal_to_eager` likewise needs the new kinds
  generated.
- **Size-blind rewrites** — `test_rewrites_survive_unknown_dim_sizes`: blanking every size must
  change no rewrite decision. `_relabelled` must move `sizes[old] → sizes[new]` carrying `None`
  through unchanged, never inventing an extent.
- **Targeted round-trips** — `.compute()` identical to eager xarray for `ds.plan.rename({...})`
  and `ds.plan.drop_vars(...)`.
