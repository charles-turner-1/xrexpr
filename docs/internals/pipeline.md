# The pipeline

Five stages between the method you chained and the array you get back. Four of them are
pure functions over metadata; one touches data.

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

The shaded box is the only stage that sees an array. Everything to its left moves
dimension names, variable names and sizes around, which is what makes the rewrite
guarantee provable rather than empirical.

Note the shape of the middle: `optimize` takes a lowered plan and returns a lowered plan.
Lowering and emission are the translation boundaries either side of it, so the optimiser
works in exactly one vocabulary. See [lowering](lowering.md) for why that boundary is a
stage rather than one more rewrite rule.

## The stages and their contracts

### Record: `accessor.LazyProxy`, `schema.to_opnode`

`ds.plan` returns a recording proxy. Each chained call is normalised by `to_opnode` and
appended to a list. The contract is narrower than it looks:

> `to_opnode` is a **pure function of one call**. Recording holds no schema of its own.

That is a deliberate refusal. The proxy *does* have the base dataset in hand, so it could
fold a schema forward as it records and resolve things eagerly, but it must not, because
past an unmodelled call that schema is a guess. A `rename` the recorder cannot model
makes every dim name it thinks it knows potentially wrong, so anything resolved against
the recorder's schema would be resolved against a fiction. Nothing is resolved at record
time; the fold belongs to the optimiser, which knows where the guessing starts.

One class serves both `Dataset` and `DataArray`. Only two things differ: a `DataArray`
has no `data_vars`, so the projection rules have nothing to fire on, and its
`__getitem__` is *indexing* rather than projection. Which applies is read off the
base object rather than stored.

### Lower: `lower.to_lower_ir`

Translates what was written into what it means, fusing the builder pairs xarray spells as
two calls. Two guarantees:

> `to_lower_ir` is **semantics-preserving** (`emit(to_lower_ir(p))` replays to the same
> result as `p`) and **idempotent**: applied to an already-lowered plan it returns it
> unchanged.

Idempotence is what lets the stage be run without first asking whether it has been run.
The stage also *consumes* every `ContextOpen`, and that is enforced by its return type
rather than trusted: `LoweredOp` doesn't include the opener, so a surviving one fails
type-checking at every site downstream.

### Optimise: `optimize.optimize`

Applies local rewrite rules to a fixpoint, and owns the schema fold. Its contract is the
package's headline invariant:

> `optimize` preserves the **values** of everything the plan asks for. It may additionally
> avoid an error raised by a computation whose result the plan discards. It may never
> change a value, nor introduce an error.

The fold lives here, not in recording, for the reason above, and the optimiser is also
the only stage that knows how far it can be trusted. The fold models an `Opaque` as
variable-preserving, which is not true of `rename` or `drop_vars`, so the folded schemas
are exact only as far as the first opaque node. `_trusted_prefix` marks that boundary,
and the rules that consult `data_vars` stay inside it. Dim-level rules read almost
everything they need off the nodes themselves; it is *variable*-level reasoning that
needs the fold at all.

### Emit: `lower.emit`

The inverse direction of lowering: one lowered node becomes the call sequence that
reproduces it, so a node standing for two calls emits two. A pure function of the plan
with no dataset in sight, which is what keeps the replay loop a short `getattr` loop
rather than something with an arm per node kind. Verbatim-ness ends up a property of the
whole pipeline instead of something each variant has to be trusted with individually.

### Replay: `accessor.LazyProxy._replay`

Performs the emitted calls against the base object, in order, then materialises through
xarray's own `.compute()`. This is the only stage that touches data, and it knows nothing
about kinds, rules or schemas. It invokes calls.

## Where each stage's design is written down

- [The IR](ir.md): why a list, why kinds rather than methods, and what the two levels are for.
- [The operation table](operations.md): how a method name becomes a kind.
- [Lowering](lowering.md): why fusion is a stage, and what happens to a pair nothing claims.
- [The optimiser](optimiser.md): the rule catalogue, the fixpoint, and where the schema
  stops being trustworthy.
- [The logical schema](schema.md): what is tracked, and why dims are derived rather than stored.
- [The value taxonomies](values.md): indexers and chunk specs as closed sum types.

Further back than that, the arguments themselves are on record. The
[`planning/`](https://github.com/charles-turner-1/xrexpr/tree/main/planning) directory
holds the design memos and the roadmap, indexed by its own README. Expect them to be
uneven: some are settled rationale, some are transcripts of arguments, and some are
superseded by the code they were arguing about. Where a memo and the code disagree, the
code wins, and these pages follow the code.
