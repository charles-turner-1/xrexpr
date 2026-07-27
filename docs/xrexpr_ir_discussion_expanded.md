# xrexpr / Compiler Design Conversation (Expanded Notes)

> **Context:** This is an expanded narrative reconstruction of the
> discussion rather than a terse summary. It keeps much of the
> conversational flow and "thinking aloud" that led to the design ideas.

------------------------------------------------------------------------

## Starting point

The project is an optimiser for Xarray expressions.

Rather than parsing a programming language, it records a chain of Xarray
method calls into an intermediate representation (IR), performs
rewrites, and then replays the optimised sequence back into Xarray.

This naturally led to the question:

> "Am I basically just building the middle of a compiler?"

The answer was essentially **yes**.

Unlike a traditional compiler:

-   Xarray itself acts as the front-end API.
-   There is no lexer.
-   There is no parser.
-   There is no machine-code backend.

Instead the project lives almost entirely in the optimisation stage.

------------------------------------------------------------------------

## Thinking in terms of compiler phases

A traditional compiler looks something like

``` text
Source language
    ↓
Scanning / tokenising
    ↓
Parsing
    ↓
Semantic analysis
    ↓
Intermediate Representation(s)
    ↓
Optimisation
    ↓
Code generation
    ↓
Machine code
```

For xrexpr the analogous pipeline is

``` text
Xarray method calls
    ↓
Record operations
    ↓
Intermediate Representation
    ↓
Rewrite / optimise
    ↓
Replay into Xarray
```

So the project is fundamentally a compiler middle-end.

------------------------------------------------------------------------

## Rule-based optimisation

The existing optimiser works by recognising legal rewrites.

Examples discussed were moving

-   selections
-   reductions
-   projections

into positions where they reduce the amount of work performed.

At the moment this is heuristic driven rather than cost driven.

That is a perfectly reasonable place to start.

The important distinction is

-   legality
-   profitability

Legality answers

> "Can these operations commute?"

Profitability answers

> "Should they commute?"

Those are independent concepts.

------------------------------------------------------------------------

## Cost models

The discussion then moved to whether a cost model should replace rewrite
rules.

The conclusion was **no**.

Rewrite rules determine correctness.

Cost models choose between multiple correct rewrites.

Initially a collection of good heuristics is likely to achieve most of
the available benefit because the Xarray operation vocabulary is
relatively small.

Only later---especially if optimising Dask graphs or searching larger
rewrite spaces---does an explicit cost model become especially valuable.

------------------------------------------------------------------------

## Dimensions as semantic information

One recurring idea was treating dimensions almost like a static type
system.

Rather than reasoning about specific operations such as

-   mean
-   std
-   sum

the optimiser should reason about facts like

-   consumes dimensions
-   preserves dimensions
-   changes ordering
-   changes chunk topology

That allows rewrite rules to become generic.

Instead of

> "isel can commute with mean"

the rule becomes

> "isel may commute with any reduction that does not consume the indexed
> dimension."

------------------------------------------------------------------------

## Discovering rewrite rules

One idea explored was fuzzing.

Generate many random plans.

Enumerate nearby permutations.

Execute them.

Detect equivalent results.

This is an interesting way to **discover candidate rewrite rules**, but
not a proof system.

Property-based testing then becomes an excellent way to validate
proposed algebraic laws.

------------------------------------------------------------------------

## Looking at the actual IR

After reading the Op definitions it became clear that the current
representation is already fairly semantic.

The variants

-   Reduce
-   Select
-   Scan
-   Project
-   Rechunk
-   Opaque

represent structural operation kinds rather than individual Xarray
methods.

That is a strong design decision.

The observation was that the project is already performing some lowering
during recording.

------------------------------------------------------------------------

## The rolling().mean() problem

The main design question was how to represent

``` python
ds.plan.rolling(time=5).mean()
```

Recording this literally gives

``` text
Rolling
Mean
```

but those two objects together actually represent one semantic
operation.

The proposed solution was to think in terms of builder objects.

Xarray exposes objects such as

-   DatasetRolling
-   DatasetGroupBy
-   DatasetWeighted

These accumulate configuration before a terminal operation.

Rather than placing an incomplete builder into the optimiser's IR, the
builder should exist only while recording.

Once the terminal operation appears, it is lowered into something like

``` text
RollingReduce(
    reduction="mean",
    window={"time": 5}
)
```

The optimiser then never has to understand fluent API mechanics.

------------------------------------------------------------------------

## Multiple intermediate representations

A key insight from the discussion was that having multiple IRs is not a
smell.

Quite the opposite.

Each transformation simplifies the job of the following stage.

Conceptually

``` text
Recording IR
      ↓
Canonical IR
      ↓
Optimised IR
```

Each representation exists because it makes the next pass simpler.

This is exactly the philosophy behind MLIR.

------------------------------------------------------------------------

## Replay

One concern was whether the optimiser would need to "raise" the
canonical IR back into the original form.

Probably not.

Instead the replay/code-generation stage simply knows how to emit

``` text
RollingReduce
```

as

``` python
ds.rolling(...).mean()
```

No intermediate builder nodes need to be reconstructed.

------------------------------------------------------------------------

## MLIR versus LLVM

LLVM is often thought of as *the* compiler IR.

In reality LLVM IR is only one relatively low-level IR.

MLIR generalises the idea.

A compiler may lower through many increasingly concrete IRs before
finally reaching LLVM.

The project resembles MLIR's philosophy much more than LLVM's.

It is perfectly acceptable---even desirable---to have several IRs if
each one is better suited to a particular optimisation stage.

------------------------------------------------------------------------

## Database optimisation

The conversation also noted that many ideas come from databases rather
than traditional compilers.

Relevant concepts include

-   predicate pushdown
-   projection pushdown
-   operator fusion
-   common sub-expression elimination
-   rewrite rules
-   logical plans
-   physical plans
-   cost-based optimisation

Polars is heavily inspired by database query planners, and xrexpr
naturally occupies a similar design space.

------------------------------------------------------------------------

## Overall conclusion

The strongest architectural insight from the discussion was probably
this:

> Don't optimise the syntax the user happened to write.
>
> First translate it into a small semantic language.
>
> Then optimise *that*.

Once viewed that way, much of the design becomes clearer.

The optimiser is no longer trying to understand the Xarray fluent API.

Instead it operates on a compact algebra of operations, leaving replay
to reconstruct whichever API calls are required.
