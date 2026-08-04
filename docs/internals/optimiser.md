# The optimiser

`optimize` rewrites a lowered plan into an equivalent, cheaper one. It is deliberately
small: five local rules and a loop.

## Local rules plus a fixpoint

Each rule maps a plan to an equivalent plan and returns `None` when it changes nothing.
The driver reapplies the whole set until every rule declines:

```{mermaid}
flowchart TB
    IN(["lowered plan"]) --> LOOP
    LOOP["for each rule in <code>_RULES</code>,<br/>apply once"] --> Q{"did any rule<br/>change the plan?"}
    Q -->|yes| LOOP
    Q -->|no| OUT(["lowered plan,<br/>equivalent and cheaper"])

    LOOP -.->|"both pushdown rules<br/>ask the same question"| DE["<b>dim_effect</b><br/>one <code>match</code>,<br/>closed by <code>assert_never</code>"]

    style DE fill:#f9c74f,stroke:#333,color:#000
```

Local rules plus a fixpoint is what lets a small rewrite compose into a large one without
any rule reasoning about the whole chain. A select moving one hop left is a single rule
firing; a select reaching the front of a plan past a run of five reductions is that rule
firing five times. Nothing in the codebase knows about the second thing.

Fixpoint detection comes from the rules' own `None` signal rather than from comparing
whole plans each pass.

## The rule catalogue

| Rule | What it does |
|---|---|
| `merge_adjacent_selects` | fold a run of consecutive `isel`/`isel` (or `sel`/`sel`) into one indexer, *composing* rather than overwriting when both name the same dim |
| `merge_adjacent_projects` | drop the first of an adjacent pair of projections when the second asks for a subset, so `ds[["tas", "pr"]][["tas"]]` stops building an intermediate holding a variable it discards one node later |
| `pushdown_selects` | hop a select left past any node whose dims permit it: a plain reduce, or a grouped or windowed one, which is what lowering's fused nodes exist to make expressible |
| `pushdown_projections` | hop a variable projection left past a preceding reduce, select or fused reduce, weighted included |
| `pushdown_selects_past_rechunks` | hop a select left past a `chunk()` so the rechunk moves less data |

`pushdown_projections` also disarms a footgun rather than only saving work: the discarded
variables can no longer raise for lacking a dimension they were never asked about. That
is [the chain that stops failing](../guide/rewrites.md#it-can-also-make-a-chain-stop-failing).

## Why the loop terminates

Every rule strictly decreases a lexicographic measure:

> `(len(plan), sum of the indices of the Select and Project nodes)`

Merging and dropping a spent rechunk shrink the plan; the three pushdown rules leave the
length alone but move a select or projection strictly left. Neither component can grow.

Stated as an obligation on future rules, that reads: **no rule may push a node right, and
no rule may lengthen a plan.** A new rule that wants to do either needs a new measure, not
an exception.

The three pushdown rules fire on disjoint adjacencies: `(*, select)`, `(*, project)` and
`(rechunk, select)`. So they cannot undo one another, which is the other half of why the
loop settles.

## One dispatch site for dim algebra

The two pushdown rules ask different questions of the node they cross, and both read the
answer from `dim_effect`, a single `match` closed with `assert_never`:

```python
@dataclass(frozen=True)
class DimEffect:
    blocks: DimSet | None      # dims a select may not hop over
    requires: DimSet | None    # dims the node needs of its input
    on_conflict: Literal["immovable", "invalid"]
```

Two fields rather than one dim set, because the rules approach a node from **opposite
sides** and that turned out to be load-bearing:

- **`blocks`** is read by a rule moving a select left, i.e. from *after* the node. It must
  avoid everything the node consumed, minted or resized, a grouped reduce's minted
  dimension included, since a select is entitled to name it.
- **`requires`** is read by a rule moving a projection left, i.e. to *before* the node. It
  is what the node needs of its **input**, which excludes a minted dimension outright:
  that dimension does not exist yet where the projection would land.

`on_conflict` distinguishes the two things a blocked hop can mean. `"invalid"`: the node
*removed* those dims, so the chain can never replay and the optimiser is entitled to say
so. `"immovable"`: the dims survive in some form, so the chain is very likely valid and
merely un-reorderable; leave it alone. Only the first raises `InvalidExpressionError`,
which is why `.mean("lon").isel(lon=0)` is reported as a mistake while
`groupby("time.month").mean().isel(month=0)` is silently left where it was written.

`None` on either field means *don't know*, and must be read as **no rewrite**, never as
"no dims". Nothing crosses a node that answers `None`.

### Why it is one site

The point is that **a new node kind must answer every question at once**. Before
`dim_effect` existed the same facts were spread across three partial matches, each with
its own silent "anything else is left alone" fallback, so a new kind would quietly
collect the conservative answer from all three without anyone deciding it should.

The conservative answer is still available, and is what most kinds take: `None` on both
fields. What is no longer available is taking it *by accident*.

`pushdown_selects_past_rechunks` deliberately stays outside `dim_effect`: it has no
disjointness test at all, because a rechunk changes no dimension and a select therefore
*always* commutes with one. What it does instead is rewrite the spec it crosses, and
which specs may be crossed is [a value-level question](values.md).

## The schema fold and how far it can be trusted

Dim-level rules read almost everything they need off the nodes themselves. A
*variable*-level rule cannot: whether a projection may cross an op depends on which dims
the projected variables carry **at that point in the plan**. So `optimize` takes the
**base** schema (not the one left at the end of recording) and `_schemas` folds it
forward to give the schema each node sees. That is the plan's single fold, and also where
a symbolic `ALL_DIMS` is resolved.

The base, because rewriting changes which node sees what. Folding from the end of
recording would mean folding from a schema describing a plan that no longer exists.

`apply_schema` models an `Opaque` as variable-preserving, which is not true of `rename` or
`drop_vars`. So the folded schemas are exact only as far as the first opaque node.
`_trusted_prefix` marks that boundary, and every rule that consults `data_vars` stays
inside it. The dim-level rules are unaffected, which is why an `Opaque` costs you rewrites
without costing you correctness.
