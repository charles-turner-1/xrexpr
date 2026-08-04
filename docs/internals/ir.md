# The IR

The intermediate representation is a sum type over the operation **kinds** the optimiser
distinguishes. Five decisions shape it, and each one is a refusal of something more
general.

## A list, not a tree

An xarray method chain is *linear*: every operation has exactly one input, the previous
dataset. So the plan is a `list[Op]`, and rules are written over adjacent pairs.

Only **unary** operations are modelled. Binary and n-ary ones (`merge`, `concat`,
`where`) would carry plan-typed children and promote the container from a list to a tree.
That is an additive, orthogonal change, deferred until such an operation is in scope —
and the linearity assumption is named in one place rather than leaked into every rule, so
the change stays a change to the container rather than an archaeology exercise.

## Kinds, not methods

The union tracks structural kinds, not xarray methods. `mean`, `std` and `sum` are all
one `Reduce`, told apart by a `name` field that replay re-invokes:

```python
Op = Reduce | Select | Scan | Project | Rechunk | Opaque
```

A rule that asks "may this selection move left past that node?" wants to know that the
node destroys `{lat}` — not that it was spelled `std`. A new *variant* is earned only by
genuinely new structural data, which is why there are six of these rather than one per
method.

The kind is usually settled by the method name, but not always — `__getitem__` is a
`Project` when its key names variables and an `Opaque` otherwise. See
[the operation table](operations.md#nominate-then-confirm).

## Fat variants, not one flat record

Each variant carries the verbatim call header (`name`, `args`, `kwargs`) that replay
re-invokes, **plus** the normalised metadata the optimiser reasons about. That metadata
differs per kind: a `Reduce` has a `consumes` dim set, a `Rechunk` has a mapping of dims
to chunk specs, a `GroupedReduce` has both a consumed dim and a minted one.

So the variants have genuinely different shapes, rather than one record with mostly-empty
fields. `match` over `Op` then binds different fields per arm, and `assert_never` on the
`case _` arm makes the union exhaustive under mypy — adding a variant fails type-checking
at every site that hasn't handled it, which is how a new kind gets a decision made about
it everywhere instead of silently falling through a default.

## Dim sets are symbolic where the call is

A bare `ds.mean()` names no dimension. It means "every dim there is *when this runs*",
and that is not knowable at record time.

```python
DimSet = frozenset[Hashable] | AllDims
ALL_DIMS: Final = AllDims()
```

Expanding it eagerly would bake in the names the recorder happened to see — and past an
unmodelled operation those names may already be wrong. So the expansion is deferred to a
reader holding an exact schema, and until then the field holds a sentinel.

`ALL_DIMS` is a *sentinel*, not `None`, and the distinction is load-bearing: `None`
already means **don't know** elsewhere in the codebase (`SchemaState.var_dims`), whereas
`ALL_DIMS` means something perfectly definite. Readers narrow it with a match arm rather
than an `assert`, so both cases are handled at the point the field is used.

This is the annotation that surfaces as `consumes=every dim` when
[reading `explain()` output](../guide/reading-explain).

## Immutability is unconditional; hashability is not

Every variant is frozen and coerces its containers, so a node cannot be mutated or drift
from itself. Each is hashable and comparable **when its payload is** — and some payloads
aren't. `xr.DataArray.__hash__` is `None`, so a node carrying an array (`weighted(w)`, a
boolean-mask `__getitem__`, `where(cond)`) raises `TypeError` on `hash()`, and `==`
between two such nodes holding distinct-but-equal arrays raises `ValueError`, the
elementwise comparison having no truth value.

The stronger claim is deliberately not wanted. Making an array payload hashable means
hashing its *values*, which for a dask-backed array means computing it at plan time — the
one thing this package promises never to do. Nothing in the pipeline hashes a node, and
plan-equality compares nodes that *share* the payload object, where tuple comparison's
identity check applies.

:::{note}
`indexers.Mask` normalises to a tuple of booleans for a related reason, but that payload
is small and already realised, so the precedent doesn't transfer to arrays.
:::

## Two levels, one vocabulary

```python
Op         = Reduce | Select | Scan | Project | Rechunk | Opaque
FluentOp   = Op | ContextOpen
LoweredOp  = Op | GroupedReduce | WindowedReduce | WeightedReduce
```

One set of dataclasses, two union aliases over them — so the level a function works at is
visible in its signature rather than in a comment.

```{mermaid}
flowchart LR
    subgraph fluent["<b>FluentOp</b> — one node per call"]
        direction TB
        fShared["Reduce · Select · Scan<br/>Project · Rechunk"]
        fOpaque["Opaque"]
        fOpen["<b>ContextOpen</b><br/><i>fluent only</i>"]
    end

    subgraph lowered["<b>LoweredOp</b> — what the optimiser rewrites"]
        direction TB
        lShared["Reduce · Select · Scan<br/>Project · Rechunk"]
        lOpaque["Opaque"]
        lFused["<b>GroupedReduce</b><br/><b>WindowedReduce</b><br/><b>WeightedReduce</b><br/><i>lowered only</i>"]
    end

    fShared -->|"unchanged"| lShared
    fOpaque -->|"unchanged"| lOpaque
    fOpen -->|"opener + closer,<br/>pair a rule claims"| lFused
    fOpen -->|"pair nothing claims"| lOpaque

    style fOpen fill:#ffd6d6,stroke:#333,color:#000
    style lFused fill:#d8f3dc,stroke:#333,color:#000
```

The asymmetry is the point. `ContextOpen` is *fluent only*: it says "a context opens
here", which is decidable from the call alone, but what the pair **means** is lowering's
job. The three fused kinds are *lowered only*: the fluent API cannot express either of
them in a single call.

Because `LoweredOp` omits the opener, a `ContextOpen` that outlived `to_lower_ir` fails
type-checking at every site downstream of it. The stage's central obligation is
discharged by the type rather than by a runtime check nobody remembers to write.

One `ContextOpen` variant rather than one per builder: the openers carry identical
structure — a verbatim header plus which builder it is — and per-builder differentiation
belongs to the fused nodes, where it arrives with structural data to match on. Five empty
arms at every `assert_never` site would earn nothing.
