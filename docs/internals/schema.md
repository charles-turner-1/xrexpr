# The logical schema

`SchemaState` is an immutable snapshot of a dataset's *logical* shape at one point in a
plan: which variables exist, which dims each spans, which names are coordinates, and how
big the dims are.

It holds **no array data**, and that is the whole point. The `.plan` proxy starts from a
real dataset, so this much can be folded forward through a plan without touching a single
value — which is what lets a rule know the shape each node sees, and what a symbolic
`ALL_DIMS` is finally resolved against.

```python
@dataclass(frozen=True)
class SchemaState:
    variables: frozendict[Hashable, tuple[Hashable, ...]]
    coord_names: frozenset[Hashable]
    sizes: frozendict[Hashable, int | None]
```

All three fields are coerced to immutable containers on construction, so a snapshot is
hashable and safe to thread through a plan.

## Variables are the store; dims are derived

`variables` maps **every** name — coordinate or not — to the dims it spans. Dim existence
is not stored anywhere: a dim exists exactly when some variable spans it.

That mirrors `xr.Dataset`, which keeps `_variables` plus `_coord_names` and derives
`.dims`, and it is load-bearing rather than cosmetic. Storing dims separately would mean
storing one fact twice, and a schema could then say `sizes={"time": 4}` while a variable
spanned `("time", "lat")` — a phantom `lat`, dutifully reported to any rule that asked.
Deriving makes that state *unconstructible* rather than merely unlikely, which is the same
trade the `assert_never` discipline makes elsewhere in the package.

`data_vars` is derived the same way — `variables` minus `coord_names`. It is what makes
variable-level reasoning possible at all: whether a projection may hop left past an op
depends on whether the projected subset still carries the dims that op names.

Coordinates are modelled as **variables with dims**, not as bare names. Without that,
`apply_schema` could not state their lifetimes: an aggregating op drops a coordinate over
the dimension it aggregates while an indexing op keeps it, demoted to 0-d — a distinction
that is inexpressible if a coordinate has no dims to lose.

`coord_names` is a *role*, not a type, for the same reason `xr.Dataset` treats it as one:
`reset_coords` moves a name between the two sets without touching the variable.

## `sizes` answers *how big*, never *which*

There is deliberately no `dims` attribute. That name meant both things before the
variables store existed, and this class is now free of the ambiguity — so it is not
reintroduced under a familiar spelling.

A size of `None` means **don't know**: the dim exists, but its extent is not statically
evident. Callers must read that as "no rewrite", never as *size zero*. Under-reporting a
size is the unsafe direction, because it is the one a rewrite could act on.

In fact **no optimiser rule reads a size at all.** Every rule reasons about dimension
*names*, and a property test pins that by blanking every size in the schema and demanding
the same output. The rechunk pushdown is the rule that most looks like it should need an
extent — it doesn't; it decides from
[the chunk spec alone](values.md#extent-dependence-is-the-discriminant).

## The optimiser owns the fold

`to_opnode` is a pure function of one call, so nothing is resolved at record time against
a schema that would only be a guess about where the op will run. The fold lives in
`optimize._schemas` instead — one fold, in the stage that also knows how far it can be
trusted.

Three pieces make up the module:

- **`SchemaState`** — the immutable snapshot.
- **`apply_schema`** — the next snapshot after one `Op` node is applied.
- **`to_opnode`** — normalise a raw recorded call into that `Op` variant.

Where the boundary of trust falls, and why `Opaque` moves it, is on
[the optimiser page](optimiser.md#the-schema-fold-and-how-far-it-can-be-trusted).
