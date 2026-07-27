# Conversation Notes: xrexpr, IR Design, and Compiler Concepts

## Project overview

The project records chained Xarray operations into a linear intermediate
representation (IR), optimises that representation, and then replays it
back into Xarray. Rather than acting as a compiler for a programming
language, it is effectively a compiler **middle-end** or **query
optimiser** for Xarray expressions.

------------------------------------------------------------------------

## Current architecture

Current pipeline:

``` text
Xarray API calls
    ↓
Record operations into a linear list of Op nodes
    ↓
Optimise the list
    ↓
Replay back into Xarray
```

The IR is already semantic rather than simply recording Python syntax.
Variants such as `Reduce`, `Select`, `Project`, `Scan`, `Rechunk` and
`Opaque` represent structural operation kinds rather than individual
Xarray methods.

------------------------------------------------------------------------

## Multiple IRs

A useful compiler pattern is to have multiple intermediate
representations.

Suggested pipeline:

``` text
Xarray fluent API
    ↓
Recording IR
    ↓
Canonicalisation / Lowering
    ↓
Semantic IR
    ↓
Optimisation
    ↓
Replay / Code generation
```

The key insight is that each IR is designed to make the *next* stage
simpler.

------------------------------------------------------------------------

## Lowering and canonicalisation

Lowering means translating one IR into another, usually one that is
simpler or more regular.

Example:

``` python
ds.plan.rolling(time=5).mean()
```

could become a single semantic node such as

``` text
RollingReduce(
    reduction="mean",
    window={"time": 5}
)
```

instead of two separate operations (`rolling` then `mean`).

Optimisation rules then reason about `RollingReduce` rather than the
fluent API.

------------------------------------------------------------------------

## Builder objects

Xarray has intermediate objects such as:

-   `DatasetRolling`
-   `DatasetGroupBy`
-   `DatasetWeighted`

These are not formally called "builders" by Xarray, but they behave like
builder objects: configuration is accumulated until a terminal operation
(such as `mean()` or `sum()`) is called.

------------------------------------------------------------------------

## Replay

Replay does **not** require "raising" back into the original IR.

Instead, code generation simply knows how to emit the canonical
operation:

``` text
RollingReduce
    ↓
ds.rolling(...).mean()
```

The canonical IR is the source language for replay.

------------------------------------------------------------------------

## Legality vs cost

These are separate concepts.

Legality determines whether a rewrite is correct.

Cost determines whether it is worth performing.

Initially, heuristics are perfectly reasonable. Later these may evolve
into an explicit cost model.

------------------------------------------------------------------------

## Dimensions as semantic information

One promising direction is treating dimensions almost like a type
system.

Operations carry information such as:

-   dimensions consumed
-   dimensions preserved
-   dimensions reordered

Rewrite rules can then reason from these semantic properties rather than
by hard-coding relationships between every pair of operations.

------------------------------------------------------------------------

## Relation to compiler architecture

Classic compiler pipeline:

1.  Front end
    -   scanning/tokenisation
    -   parsing
    -   semantic analysis
2.  Middle end
    -   optimisation
    -   rewriting
    -   analysis
3.  Back end
    -   code generation

For xrexpr:

-   Xarray effectively provides the front end.
-   xrexpr is almost entirely the middle end.
-   Replay acts as a lightweight back end.

------------------------------------------------------------------------

## MLIR and LLVM

LLVM IR is a low-level IR close to machine execution.

MLIR is a framework for building multiple higher-level IRs and
progressively lowering between them.

The project is much more analogous to MLIR than LLVM.

------------------------------------------------------------------------

## Related fields worth studying

-   Compiler middle-end design
-   Intermediate representations (IRs)
-   Term rewriting systems
-   Equality saturation (e-graphs)
-   Database query optimisation
-   Relational algebra
-   Apache Calcite
-   MLIR

The project sits at the intersection of compiler optimisation and
database query optimisation rather than traditional compiler
construction.
