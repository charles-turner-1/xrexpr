# The optimiser

`optimize` rewrites a lowered plan into an equivalent, cheaper one. It's small on
purpose: a handful of local rules and a loop.

## Local rules plus a fixpoint

Each rule maps a plan to an equivalent plan, and returns `None` when it has nothing to
do. The loop keeps applying the whole set until every rule returns `None`:

```{mermaid}
flowchart TB
    IN(["lowered plan"]) --> LOOP
    LOOP["for each rule in <code>_RULES</code>,<br/>apply once"] --> Q{"did any rule<br/>change the plan?"}
    Q -->|yes| LOOP
    Q -->|no| OUT(["lowered plan,<br/>equivalent and cheaper"])

    LOOP -.->|"the pushdown rules<br/>ask the same question"| DE["<b>dim_effect</b><br/>one <code>match</code>,<br/>closed by <code>assert_never</code>"]

    style DE fill:#f9c74f,stroke:#333,color:#000
```

This is how small rewrites add up to big ones without any rule looking at the whole
chain. A select moving one step left is one rule firing. A select reaching the front of a
plan past five reductions is the same rule firing five times. No code knows about the
second thing.

The loop spots the fixpoint from the rules' `None` return, rather than comparing whole
plans on each pass.

## The rules

| Rule | What it does |
|---|---|
| `merge_adjacent_selects` | Merge a run of consecutive `isel`s (or `sel`s) into one. Two indexers on the same dim are *composed*, not overwritten. |
| `merge_adjacent_projects` | Drop the first of two adjacent projections when the second asks for a subset, so `ds[["tas", "pr"]][["tas"]]` doesn't build a dataset holding `pr` only to throw it away. |
| `merge_adjacent_rechunks` | Merge two adjacent per-dim `chunk` calls into one. |
| `eliminate_projection_before_coord` | Drop a projection when the next one only reads coordinates that survive it: `ds[["tas"]]["lat"]` becomes `ds["lat"]`. |
| `push_projection_past_drop` | Move a projection in front of a preceding `drop_vars`, removing the drop entirely when the projection makes it redundant. |
| `pushdown_selects` | Move a select left past any node whose dims allow it: a reduce, a scan, an elementwise op, a `drop_vars`, a `rename` (on dims it leaves alone), or a grouped or windowed reduce. |
| `pushdown_projections` | Move a projection left past a reduce, select, scan, elementwise op or fused reduce (weighted included). |
| `pushdown_selects_past_rechunks` | Move a select in front of a `chunk()` so the rechunk handles less data. |

`pushdown_projections` can also stop a chain from failing, not just save work: discarded
variables can no longer raise for missing a dimension nobody asked them about. See
[the chain that stops failing](../guide/rewrites.md#it-can-also-make-a-chain-stop-failing).

## Why the loop terminates

Every rule strictly decreases this measure (compared lexicographically):

> `(len(plan), sum of the indices of the Select and Project nodes)`

The merges and eliminations shorten the plan. The pushdowns keep the length the same but
move a select or projection strictly left. Neither part of the measure can grow.

As a rule for anyone adding a rule: **no rule may move a node right, and no rule may
lengthen a plan.** A rule that needs to do either needs a new termination argument.

## One place for dim facts

The pushdown rules ask different questions of the node they want to cross, and they all
read the answer from `dim_effect`: a single `match`, closed with `assert_never`.

```python
@dataclass(frozen=True)
class DimEffect:
    blocks: DimSet | None      # dims a select may not hop over
    requires: DimSet | None    # dims the node needs from its input
    on_conflict: Literal["immovable", "invalid"]
```

There are two dim fields because the rules approach a node from opposite sides:

- **`blocks`** is read when moving a *select* left, from after the node. The select must
  avoid every dim the node consumed, created or resized. That includes a grouped
  reduce's new dim, because a select after the reduce may name it.
- **`requires`** is read when moving a *projection* to before the node. It's what the
  node needs from its **input**, so it excludes any dim the node creates, which doesn't
  exist yet at that point.

`on_conflict` says what it means when a select names a blocked dim:

- `"invalid"`: the node *removed* those dims, so the chain can never run. The optimiser
  raises `InvalidExpressionError`. That's why `.mean("lon").isel(lon=0)` is reported as a
  mistake.
- `"immovable"`: the dims still exist in some form, so the chain is probably fine and
  just can't be reordered. It's left alone. That's why
  `groupby("time.month").mean().isel(month=0)` stays where it was written.

`None` in either field means *don't know*, and is always read as **no rewrite**, never as
"no dims". Nothing crosses a node that answers `None`.

### Why it's one place

**A new node kind has to answer every question at once.** The rules don't match on node
kinds themselves; they ask `dim_effect`. So there's no rule with a silent "anything else,
leave it" fallback that a new kind could fall into. Because the `match` is closed with
`assert_never`, mypy forces a decision for the new kind right here.

The conservative answer, `None` for both fields, is still there, and most kinds the
optimiser doesn't reason about use it. But a kind only gets it by someone choosing it.

`pushdown_selects_past_rechunks` doesn't use `dim_effect`. A rechunk changes no dimension,
so a select *always* commutes with one as far as the result goes. The question is whether
the chunk spec still means the same thing after the select, and that's
[a question about the spec's value](values.md).

## The schema fold and how far it can be trusted

Dim-level rules read nearly everything they need straight off the nodes. A
*variable*-level rule can't: whether a projection may cross an op depends on which dims
the kept variables have **at that point in the plan**. So `optimize` takes the **base**
schema, and `_schemas` folds it forward to give the schema each node sees. That's the
plan's one fold, and it's also where a symbolic `ALL_DIMS` gets resolved.

It has to be the base schema, not the one at the end of recording, because rewriting
changes which node sees what.

`apply_schema` assumes an `Opaque` keeps every variable. That isn't true in general
(`assign`, `stack`, ...), so the folded schemas are only exact up to the first opaque
node. `_trusted_prefix` marks that point, and every rule that reads `data_vars` stays
before it. `Rename` and `Drop` are modelled exactly, so the trusted prefix continues past
them. Dim-level rules don't need the limit, because an `Opaque` blocks them outright. So an
`Opaque` costs you rewrites, never correctness.
