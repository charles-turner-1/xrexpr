# The IR

The intermediate representation is a union over the *kinds* of operation the optimiser
tells apart. Five design decisions shape it, and each is a choice to keep it simpler than
it could be.

## A list, not a tree

An xarray method chain is *linear*: every operation has exactly one input, the previous
dataset. So a plan is a `list[Op]`, and rules are written over adjacent pairs.

Only **unary** operations are modelled. Binary or n-ary ones (`merge`, `concat`,
`where`) would need child plans, which would turn the list into a tree. That's a separate
change, put off until such an operation is in scope. The assumption is stated in one
place (the `ir.py` module docstring) rather than baked into every rule, so making that
change later only touches the container.

## Kinds, not methods

The union tracks kinds of operation, not xarray methods. `mean`, `std` and `sum` are all
a `Reduce`, told apart by a `name` field that replay uses:

```python
Op = Reduce | Select | Scan | Elementwise | Project | Drop | Rename | Rechunk | Opaque
```

A rule asking "may this select move left past that node?" needs to know that the node
removes `{lat}`, not that it was spelled `std`. A new kind is only added when there's new
structural data to carry, which is why there are nine of these rather than one per
method.

Usually the method name decides the kind, but not always. `__getitem__` is a `Project`
when its key names variables, and an `Opaque` otherwise. See
[the operation table](operations.md#nominate-then-confirm).

## Each kind has its own fields

Every node carries the verbatim call (`name`, `args`, `kwargs`) that replay uses, **plus**
the parsed metadata the optimiser reasons about. That metadata differs per kind:

- a `Reduce` has a `consumes` dim set;
- a `Select` has an `indexer` mapping;
- a `Rechunk` has a mapping of dims to chunk specs;
- a `GroupedReduce` has both a consumed dim and a newly created one.

So each kind is its own dataclass, rather than one record with mostly empty fields.
`match` over `Op` binds different fields per arm, and `assert_never` in the `case _` arm
makes mypy check the match is exhaustive. Adding a kind fails type-checking at every site
that doesn't handle it, so someone has to make a decision about it everywhere, instead of
it quietly falling through to a default.

## A bare reduce records a symbol, not a dim list

A bare `ds.mean()` names no dimension. It means "every dim there is *when this runs*",
and that can't be known at recording time.

```python
DimSet = frozenset[Hashable] | AllDims
ALL_DIMS: Final = AllDims()
```

Expanding it at recording time would fix it to the dims the recorder happened to see,
and after an unmodelled operation those could already be wrong. So expansion is put off
until something with an exact schema reads it, and until then the field holds a sentinel.

`ALL_DIMS` is a *sentinel*, not `None`, and that matters. `None` already means **don't
know** elsewhere (`SchemaState.var_dims`), whereas `ALL_DIMS` means something definite.
Code reading the field handles it with a `match` arm rather than an `assert`, so both
cases are dealt with where the field is used.

This is what shows up as `consumes=every dim` in
[`explain()` output](../guide/reading-explain.md).

## Always immutable, only sometimes hashable

Every node is frozen and converts its containers to immutable ones, so a node can't be
changed after it's made. It's hashable and comparable **when its contents are**, and some
contents aren't. `xr.DataArray.__hash__` is `None`, so a node holding an array
(`weighted(w)`, a boolean-mask `__getitem__`) raises `TypeError` on `hash()`. Comparing
two such nodes holding different-but-equal arrays raises `ValueError`, because an
elementwise comparison has no single truth value.

Making those nodes hashable isn't wanted. Hashing an array means hashing its *values*,
and for a dask-backed array that means computing it while planning, which is the one
thing this package promises never to do. Nothing in the pipeline hashes a node, and plan
comparisons compare nodes that *share* the same array object, where Python's identity
check makes the comparison work.

:::{note}
`indexers.Mask` is converted to a tuple of booleans for a related reason, but a mask is
small and already in memory, so the same reasoning doesn't apply to arrays.
:::

## Two levels, one set of node types

```python
Op         = Reduce | Select | Scan | Elementwise | Project | Drop | Rename | Rechunk | Opaque
FluentOp   = Op | ContextOpen
LoweredOp  = Op | GroupedReduce | WindowedReduce | WeightedReduce
```

One set of dataclasses, and two union types over them. The level a function works at
shows up in its signature rather than a comment.

```{mermaid}
flowchart LR
    subgraph fluent["<b>FluentOp</b>, one node per call"]
        direction TB
        fShared["Reduce · Select · Scan · Elementwise<br/>Project · Drop · Rename · Rechunk"]
        fOpaque["Opaque"]
        fOpen["<b>ContextOpen</b><br/><i>fluent only</i>"]
    end

    subgraph lowered["<b>LoweredOp</b>, what the optimiser rewrites"]
        direction TB
        lShared["Reduce · Select · Scan · Elementwise<br/>Project · Drop · Rename · Rechunk"]
        lOpaque["Opaque"]
        lFused["<b>GroupedReduce</b><br/><b>WindowedReduce</b><br/><b>WeightedReduce</b><br/><i>lowered only</i>"]
    end

    fShared -->|"unchanged"| lShared
    fOpaque -->|"unchanged"| lOpaque
    fOpen -->|"opener + closer,<br/>a pair a rule fuses"| lFused
    fOpen -->|"a pair nothing fuses"| lOpaque

    style fOpen fill:#ffd6d6,stroke:#333,color:#000
    style lFused fill:#d8f3dc,stroke:#333,color:#000
```

The two levels differ on purpose:

- `ContextOpen` is *fluent only*. It says "a context opens here", which can be decided
  from the call alone, but what the pair **means** is decided by lowering.
- The three fused kinds are *lowered only*. Each describes something the fluent API
  can't express in a single call.

Because `LoweredOp` leaves out `ContextOpen`, one that survived `to_lower_ir` would fail
type-checking everywhere downstream. Lowering's main obligation is enforced by the types,
not by a runtime check someone has to remember to write.

There's one `ContextOpen` kind rather than one per builder. The openers all have the same
structure (a verbatim call plus which builder it is), and the per-builder detail belongs
on the fused nodes, where there's structured data to match on. Five empty arms at every
`assert_never` site would add nothing.
