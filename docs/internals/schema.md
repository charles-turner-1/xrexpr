# The logical schema

`SchemaState` is an immutable snapshot of a dataset's *logical* shape at one point in a
plan: which variables exist, which dims each one spans, which names are coordinates, and
how big the dims are.

It holds **no array data**. The `.plan` proxy starts from a real dataset, so this much can
be tracked through a plan without reading a single value. That's how a rule knows the
shape each node sees, and what a symbolic `ALL_DIMS` is resolved against.

```python
@dataclass(frozen=True)
class SchemaState:
    variables: frozendict[Hashable, tuple[Hashable, ...]]
    coord_names: frozenset[Hashable]
    sizes: frozendict[Hashable, int | None]
```

All three fields are converted to immutable containers on construction, so a snapshot is
hashable and safe to pass around.

## Variables are stored; dims are derived

`variables` maps **every** name, coordinate or not, to the dims it spans. Which dims
exist isn't stored anywhere: a dim exists exactly when some variable spans it.

That's how `xr.Dataset` works too (it stores `_variables` and `_coord_names`, and derives
`.dims`), and it matters. Storing dims separately would store the same fact twice. A
schema could then say `sizes={"time": 4}` while a variable spans `("time", "lat")`: a
phantom `lat` that would be reported to any rule that asked. Deriving the dims makes that
state impossible to build, not just unlikely.

`data_vars` is derived the same way, as `variables` minus `coord_names`. That's what makes
variable-level rules possible: whether a projection may move left past an op depends on
whether the kept variables still have the dims that op names.

Coordinates are stored as **variables with dims**, not bare names. Otherwise
`apply_schema` couldn't track their lifetimes. An aggregating op drops a coordinate over
the dimension it aggregates, while an indexing op keeps it as a 0-d coordinate. You can't
express that if a coordinate has no dims to lose.

`coord_names` is a *role*, not a different kind of variable, as in `xr.Dataset`:
`reset_coords` moves a name between the two sets without changing the variable.

## `sizes` says *how big*, never *which*

There's deliberately no `dims` attribute, because that name could mean either "which dims
exist" or "how big they are". `dim_names` answers the first, `sizes` the second.

A size of `None` means **don't know**: the dim exists, but its length can't be known
statically. Callers must treat it as "no rewrite", never as size zero. Under-reporting a
size is the dangerous direction, because a rewrite could act on it.

In fact **no optimiser rule reads a size at all.** Every rule reasons about dimension
*names*. A property test checks this by blanking every size in the schema and requiring
the same output. The rechunk pushdown looks like it should need sizes, but it doesn't: it
decides from [the chunk spec alone](values.md#extent-dependence-is-the-discriminant).

## The optimiser owns the fold

`to_opnode` handles one call at a time, so nothing is resolved at recording time against
a schema that would only be a guess about where the op ends up running. The fold lives in
`optimize._schemas` instead: one fold, in the stage that also knows how far it can be
trusted.

The module has three main pieces:

- **`SchemaState`**: the immutable snapshot.
- **`apply_schema`**: the next snapshot after one node is applied.
- **`to_opnode`**: turns a raw recorded call into its node.

Where the trust boundary falls, and why an `Opaque` moves it, is on
[the optimiser page](optimiser.md#the-schema-fold-and-how-far-it-can-be-trusted).
