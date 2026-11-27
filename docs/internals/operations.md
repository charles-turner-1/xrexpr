# The operation table

`OP_TABLE` is the single list of methods the recorder can classify. Each method is one
row, and each row is an `OpSpec`.

## Each kind is a type, not a `kind` string

`OpSpec` is a union with **one variant per kind of node** `to_opnode` can build:

```python
OpSpec = (
    ReduceSpec | ScanSpec | ElementwiseSpec | SelectSpec | RechunkSpec
    | ProjectSpec | RenameSpec | DropSpec | ContextSpec
)
```

The spec's type *is* its kind; there's no `kind: str` field. That's what lets
`to_opnode` dispatch with one `match` closed by `assert_never`. A variant added here
without an arm there is a type error, rather than a call quietly recorded as `Opaque`.
`ir.Op` and `indexers.Indexer` work the same way. A `kind` string is really a union whose
variants only exist in someone's head.

Being a real type also gives kind-specific data somewhere to live. `ReduceSpec.dim_arg`
(where the dim sits among the positional arguments) is a field on the spec, not a second
table keyed by method name.

## What the table distinguishes

**`ReduceSpec` removes its dim; `ScanSpec` keeps it.** `mean`/`sum`/`std` collapse the
dimension they name. `cumsum`/`cumprod`/`diff` keep the same dimensions, and each value
depends on what came before it along that axis. Treating them the same would let a select
on the scanned dimension hop a scan, which changes the answer. A select can safely hop a
reduction over a *different* dim, but not a scan over the *same* one.

**`ElementwiseSpec` keeps everything.** `fillna`/`astype`/`round`/`clip`/... work element
by element, so any select or projection commutes with them. But only when every argument
is a plain value: `fillna(other_da)` or `fillna({"tas": 0})` becomes `Opaque`.

**`RechunkSpec` changes no dim at all**, only how the data is split into blocks, which is
why a select can cross it. Which selects may cross which chunk specs is a separate and
trickier question: see [rechunking](../guide/rechunking.md).

**`SelectSpec` works out its dim effect per call**, from the indexer. `isel(time=0)`
drops `time`; `isel(time=slice(0, 10))` keeps it. The spec can't say which, because the
answer is in the argument, not the method.

**`RenameSpec` and `DropSpec` are modelled** so the schema stays exact past them. A
rename relabels dims or variables; a `drop_vars` removes variables or coordinates.

**`ContextSpec` is half an operation**: a call that returns a builder, whose meaning
[lowering](lowering.md) decides once it can see the closer.

**`ProjectSpec` can only nominate.** See the next section.

## Nominate, then confirm

Every other spec names the node kind its call *is*. `ProjectSpec` names the kind its call
*may be*:

```python
ds[["temperature"]]   # Project: the key names variables
ds[mask]              # Opaque:  the key is a boolean array
ds[{"time": 0}]       # Opaque:  the key is an indexer mapping
```

Same method, same row, different kinds. The key's shape decides, at recording time, in a
guarded arm of `to_opnode`'s `match`. (`ElementwiseSpec` nominates in the same way, based
on its arguments.)

It's still in the table, rather than special-cased after dispatch, so every method the
recorder can classify is classified in one place. The guard that confirms it sits in the
same `match` as the other arms, where someone reading the dispatch will find it.

## Built from the specs, so it can't drift

Two things in this module are computed rather than written out:

```python
OP_TABLE = {op.name: op for op in _SPECS}
CONTEXT_METHODS = frozenset(op.name for op in _SPECS if isinstance(op, ContextSpec))
```

`OP_TABLE` is built from the specs rather than written as a dict literal, so a row can't
be filed under a key that disagrees with its own name. Each variant's `name` is typed as
the `Literal` its node expects, so a looked-up name reaches the node constructor already
correctly typed, with no `cast`.

The builder method names go one step further. They're built with `get_args` from the
`ContextOpenName` `Literal` that types them, so the runtime rows and the type can't
disagree either. What no derivation can guarantee is that xarray agrees (that the methods
exist and still return builders), so a test checks them against `xr.Dataset`.

## What isn't in the table

A method missing from `OP_TABLE` becomes an `Opaque` node. It replays verbatim, and no
rewrite crosses it. That's the safe default the whole design relies on: an unrecognised
call costs you the rewrite, not correctness.
