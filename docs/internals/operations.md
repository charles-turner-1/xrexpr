# The operation table

`OP_TABLE` is the single source of truth for what the recorder can classify. Every method
it can see is one row, and every row is an `OpSpec`.

## A sum type, not a record with a `kind` string

`OpSpec` is a **sum over dispatch kinds** — one variant per `Op` variant `to_opnode` can
build:

```python
OpSpec = ReduceSpec | ScanSpec | SelectSpec | RechunkSpec | ProjectSpec | ContextSpec
```

The spec *is* the kind, rather than carrying a `kind: str` field. That is what makes
`to_opnode`'s dispatch a single `match` closed by `assert_never`: a variant added here
without a corresponding arm there is a type error, not a call silently recorded as
`Opaque`. It is the same move `ir.Op` and `indexers.Indexer` make, and the reasoning is
written up at length in the design memos — a `kind` string is a sum type whose variants
live only in the designer's head.

Being a real variant also gives kind-specific metadata somewhere to live.
`ReduceSpec.dim_arg` is a field on the spec, rather than a second table keyed by method
name all over again.

## The distinctions the table exists to make

**`ReduceSpec` destroys its dim; `ScanSpec` keeps it.** `mean`/`sum`/`std` collapse the
dimension they name; `cumsum`/`cumprod`/`diff` return a result with the same dimensions
they started with, and the value at each position depends on what came before it along
that axis. Lumping the two together is precisely the `cumsum` reordering bug: a selection
that may safely hop a reduction may not hop a scan on the scanned dimension.

**`RechunkSpec` touches no dim at all** — only chunk topology, which is what lets a
selection cross it. Which selections may cross which chunk specs is a separate question,
and a subtler one: see [rechunking](../guide/rechunking.md).

**`SelectSpec` resolves its dim effect per call**, from the indexer. `isel(time=0)` drops
`time`; `isel(time=slice(0, 10))` keeps it. The spec cannot say which, because the answer
is in the argument rather than in the method.

**`ContextSpec` is half an operation** — a builder-returning call whose meaning
[lowering](lowering.md) settles once it can see the closer.

**`ProjectSpec` is the one row the table can only nominate.** Next section.

## Nominate, then confirm

Every other spec names the `Op` variant its call *is*. `ProjectSpec` names the variant
its call *may be*:

```python
ds[["temperature"]]   # Project — the key names variables
ds[mask]              # Opaque  — the key is a boolean array
ds[{"time": 0}]       # Opaque  — the key is an indexer mapping
```

Same method, same row, different kinds. The shape of the *key* decides, at record time,
in `to_opnode`'s guarded match arm.

It is tabulated anyway, rather than special-cased after the dispatch, so that every
method the recorder can classify is classified in one place. The nomination is then
visible in the type, and the guard that confirms it sits in the `match` beside the arms
it is an exception to — where a reader looking at the dispatch will actually find it.

## Derived, so it cannot drift

Two things in this module are computed rather than written:

```python
OP_TABLE = {op.name: op for op in _SPECS}
CONTEXT_METHODS = frozenset(op.name for op in _SPECS if isinstance(op, ContextSpec))
```

`OP_TABLE` is derived from the specs rather than written as a mapping literal, so a row
cannot be filed under a key that disagrees with the name it carries. Each variant's
`name` is typed as the `Literal` its `Op` counterpart demands, so a lookup's name arrives
at the `Op` constructor already correctly typed, with no `cast` in between.

The builder-opener names go one step further: they are derived by `get_args` from the
`ContextOpenName` `Literal` that types them, so the runtime rows and the type cannot
disagree either. What no derivation can supply is agreement with xarray itself — that
those method names exist and still return builders — so a test pins them against
`xr.Dataset`.

## What isn't in the table

A method name absent from `OP_TABLE` becomes an `Opaque` node: it replays verbatim, and
no rewrite crosses it. That is the conservative default the whole design leans on, and
the reason an unrecognised call costs you the rewrite rather than the correctness.
