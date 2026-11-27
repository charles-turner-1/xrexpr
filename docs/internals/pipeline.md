# The pipeline

There are five stages between the method you chain and the array you get back. Four of
them are pure functions over metadata. Only the last one touches data.

```{mermaid}
flowchart LR
    A["xarray calls"] --> B["<b>to_opnode</b><br/><code>schema.py</code>"]
    B --> C[["fluent IR<br/><code>list[FluentOp]</code>"]]
    C --> D["<b>to_lower_ir</b><br/><code>lower.py</code>"]
    D --> E[["lowered IR<br/><code>list[LoweredOp]</code>"]]
    E --> F["<b>optimize</b><br/><code>optimize.py</code>"]
    F --> G[["lowered IR<br/><code>list[LoweredOp]</code>"]]
    G --> H["<b>emit</b><br/><code>lower.py</code>"]
    H --> I[["<code>list[Call]</code>"]]
    I --> J["<b>_replay</b><br/><code>accessor.py</code>"]
    J --> K["<code>xr.Dataset</code>"]
    style J fill:#f9c74f,stroke:#333,color:#000
```

The shaded box is the only stage that sees an array. Everything before it works with
dimension names, variable names and sizes. That's why the rewrites can be proved correct
rather than just tested.

`optimize` takes a lowered plan and returns a lowered plan. Lowering and emission are the
translation steps on either side, so the optimiser only ever deals with one vocabulary.
See [lowering](lowering.md) for why lowering is its own stage rather than another rewrite
rule.

## The stages and their contracts

### Record: `accessor.LazyProxy`, `schema.to_opnode`

`ds.plan` returns a recording proxy. Each chained call is turned into a node by
`to_opnode` and appended to a list.

> `to_opnode` is a **pure function of one call**. Recording keeps no schema of its own.

That's deliberate. The proxy *does* have the base dataset, so it could fold a schema
forward while recording and resolve things as it goes. It mustn't, because after any
call it can't model, that schema is a guess. An unmodelled `assign` or `stack` might
change the very names the recorder thinks it knows, and anything resolved against them
would be silently wrong. So nothing is resolved at record time. The optimiser does the
fold, because it knows where the guessing starts.

One class handles both `Dataset` and `DataArray`. Only two things differ: a `DataArray`
has no `data_vars`, so the projection rules never fire on it, and its `__getitem__`
means *indexing* rather than projection. Which applies is read off the base object, not
stored.

### Lower: `lower.to_lower_ir`

Turns what was written into what it means, by fusing the builder pairs xarray spells as
two calls (`groupby(...).mean()` and friends). Two guarantees:

> `to_lower_ir` **preserves semantics** (`emit(to_lower_ir(p))` replays to the same
> result as `p`) and is **idempotent** (run on an already-lowered plan, it returns it
> unchanged).

Idempotence means the stage can be run without checking whether it already has been.
Lowering also removes every `ContextOpen`, and the types enforce that: `LoweredOp`
doesn't include `ContextOpen`, so one that survived would fail type-checking everywhere
downstream.

### Optimise: `optimize.optimize`

Applies local rewrite rules until none fires, and owns the schema fold. Its contract is
the package's main promise:

> `optimize` preserves the **values** of everything the plan asks for. It may also avoid
> an error raised while computing or validating part of the chain that can't affect those
> values. It may never change a value, or introduce an error.

The fold lives here rather than in recording for the reason above, and because the
optimiser is the stage that knows how far the fold can be trusted. The fold assumes an
`Opaque` keeps every variable, which isn't true in general (`assign`, `stack`, ...). So
the folded schemas are only exact up to the first opaque node. `_trusted_prefix` marks
that point, and rules that read `data_vars` stay before it. Dim-level rules mostly read
what they need straight off the nodes; it's the *variable*-level rules that need the fold.

### Emit: `lower.emit`

The reverse of lowering: each lowered node becomes the calls that reproduce it, so a node
that stands for two calls emits two. It's a pure function of the plan, with no dataset
involved. That keeps replay a short `getattr` loop with no special cases per node kind.

The calls are rebuilt from each node's semantic fields, not copied from what was
recorded. So when a rule changes a field (say, merges two selects' indexers), the replayed
call changes with it.

### Replay: `accessor.LazyProxy._replay`

Runs the emitted calls against the base object, in order, then calls xarray's own
`.compute()`. This is the only stage that touches data. It knows nothing about node
kinds, rules or schemas; it just makes calls.

## Where each stage's design is written down

- [The IR](ir.md): why a list, why kinds rather than methods, and what the two levels are
  for.
- [The operation table](operations.md): how a method name becomes a node kind.
- [Lowering](lowering.md): why fusion is a stage, and what happens to a pair nothing
  fuses.
- [The optimiser](optimiser.md): the rules, the fixpoint loop, and where the schema stops
  being trustworthy.
- [The logical schema](schema.md): what's tracked, and why dims are derived rather than
  stored.
- [The value types](values.md): indexers and chunk specs as closed sets of variants.

The design discussions behind all this are in the
[`planning/`](https://github.com/charles-turner-1/xrexpr/tree/main/planning) directory,
indexed by its README. They're uneven: some are settled reasoning, some are transcripts
of arguments, and some are out of date. Where a memo and the code disagree, the code is
right, and these pages follow the code.
