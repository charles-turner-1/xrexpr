# The value taxonomies

Two of the package's operations carry an argument that is a sum type in disguise:

- `Select.indexer` maps each dim to one **indexer** value — a position, a slice, a
  sequence, an array, a coordinate label.
- `Rechunk.chunks` maps each dim to one **chunk spec** — a size, a symbolic request, an
  explicit block sequence.

Held as `Any`, either one forces every call site in `ir.py`, `optimize.py` and
`schema.py` to re-derive the shapes by hand. `indexers.py` and `chunks.py` make each
taxonomy a *type* instead. They were written months apart and ended up with the same four
design moves, which is the interesting part.

## Move 1 — `classify` is the sole constructor

One function per taxonomy decides which variant a raw value is, and every raw value goes
through it. There is no second place where a value is sorted, and no call site guesses.

Both are **total**: anything the taxonomy cannot model gets the escape-hatch variant
rather than an error. That matters because classification happens at record time, on
whatever the user handed the accessor, and refusing there would turn a chain xarray would
have accepted into a chain `xrexpr` rejects.

## Move 2 — what follows becomes a method, not a re-decision

Each variant answers the questions its consumers ask:

| | indexers | chunk specs |
|---|---|---|
| does it drop the dim? | `drops_dim` | — |
| how long is the dim afterwards? | `size()` | — |
| what does replay get handed? | `to_raw()` | `to_raw()` |

`to_raw` is the one both need, and it exists because a rewrite may *rebuild* a node's
`args` from its normalised field — a select that merged with another, a rechunk that lost
a dim to a hoisted select. Replay must then be handed the exact xarray-facing value, and
the variant is what knows it.

## Move 3 — policy stays with the optimiser

Neither module models what the optimiser should *do*.

Whether two indexers **compose** is not on `Indexer`; whether a chunk spec may be
**crossed** is not on `ChunkSpec`. Both are policies the optimiser chooses to prove, not
intrinsic facts about a value — so both stay a `match` in `optimize.py`, closed with
`assert_never` so that a new variant fails type-checking until someone decides which side
of the line it falls on.

What the value modules guarantee is the *discriminant* that policy matches on. No variant
carries a flag saying what should therefore happen to it.

## Move 4 — one escape hatch per layer

`Label` for indexers, `OpaqueChunk` for chunk specs, and `Opaque` for op kinds one level
up. The same idea at three depths: a variant meaning "this is real, it will replay
verbatim, and nothing may be inferred from it."

That is what lets `classify` be total, and it is why an unrecognised value costs a rewrite
rather than correctness.

## The indexer taxonomy

```{mermaid}
flowchart TB
    raw(["one dim's indexer"]) --> q1{"a slice?"}
    q1 -->|yes| q2{"all bounds<br/>integers?"}
    q2 -->|no| label
    q2 -->|yes| q3{"forward,<br/>non-negative?"}
    q3 -->|yes| fwd
    q3 -->|no| gen
    q1 -->|no| q4{"an array or<br/>a sequence?"}
    q4 -->|"0-d array"| scalar
    q4 -->|"all booleans, 1-d"| mask
    q4 -->|"all integers"| positions
    q4 -->|"anything else"| label
    q4 -->|no| scalar

    subgraph drops["<b>drops_dim = True</b>"]
        scalar["<b>Scalar</b><br/><code>isel(time=0)</code>"]
    end

    subgraph keeps["<b>drops_dim = False</b>"]
        fwd["<b>ForwardSlice</b><br/><code>slice(0, 10)</code>"]
        gen["<b>GeneralSlice</b><br/><code>slice(None, None, -1)</code>"]
        positions["<b>Positions</b><br/><code>[0, 2, 4]</code>"]
        mask["<b>Mask</b><br/><code>[True, False, ...]</code>"]
        label["<b>Label</b><br/><code>sel(time='2020-01')</code>"]
    end

    style drops fill:#ffd6d6,stroke:#333,color:#000
    style keeps fill:#d8f3dc,stroke:#333,color:#000
```

Exactly one variant drops its dimension, which is the fact `Select.consumes` is derived
from. Order matters twice in that tree: a boolean sequence is a `Mask` and not
`Positions`, even though `bool` subclasses `int`, so the all-boolean test runs first; and
a 0-d array is a `Scalar` rather than a one-element enumeration, so the rank check runs
before the dtype dispatch.

**`ForwardSlice` earns its own variant** so that the "forward, non-negative bounds"
carve-out the composer needs is a *constructor invariant* — a `ForwardSlice` cannot be
built with a negative bound — rather than a guard re-run at every call site. `GeneralSlice`
takes every other integer slice, and a slice with a non-integer bound is a label slice,
which is not positional at all.

**`Label` is the escape hatch.** A `sel` coordinate label — a string, a timestamp, a tuple
key, a label slice, a label sequence — is genuinely open and cannot be reasoned about
positionally.

`size()` has a boundary worth noting: sizing a dim whose current length is unknown, or a
`sel` label slice, is not this layer's call. Those answer "unknown" one level up, in the
guard that wraps every call to it.

## The chunk taxonomy

Seven variants, and the same tree shape — but grouped by a discriminant that has no
counterpart on the indexer side.

```{mermaid}
flowchart TB
    raw(["one dim's chunk spec"]) --> q1{"is it <code>None</code>?"}
    q1 -->|yes| nochange
    q1 -->|no| q2{"a string?"}
    q2 -->|"<code>'auto'</code>"| auto
    q2 -->|"anything else"| bytesize
    q2 -->|no| q3{"an integer?<br/>(a <code>bool</code> is not)"}
    q3 -->|"<code>-1</code>"| fulldim
    q3 -->|"1 or more"| single
    q3 -->|"<code>0</code>, or below <code>-1</code>"| opaque
    q3 -->|no| q4{"a non-empty sequence<br/>of integers?"}
    q4 -->|yes| blockseq
    q4 -->|no| opaque

    subgraph free["extent-free"]
        nochange["<b>NoChange</b>"]
        fulldim["<b>FullDim</b>"]
        single["<b>SingleSize</b>"]
    end

    subgraph dep["extent-dependent"]
        auto["<b>Auto</b>"]
        bytesize["<b>ByteSize</b>"]
        blockseq["<b>BlockSeq</b>"]
    end

    subgraph esc["escape hatch"]
        opaque["<b>OpaqueChunk</b>"]
    end

    style free fill:#d8f3dc,stroke:#333,color:#000
    style dep fill:#ffd6d6,stroke:#333,color:#000
    style esc fill:#eeeeee,stroke:#333,color:#000
```

The variants are one per *behaviour*, not one per spelling, and the split was checked
against the pinned xarray and dask rather than reasoned about. The case that settles it:
on an already-chunked dim, `None` leaves the existing blocks alone and `-1` collapses them
into one. So `NoChange` and `FullDim` both exist.

### Extent-dependence is the discriminant

`Auto`, `ByteSize` and `BlockSeq` are the specs dask resolves by **measuring the array it
is handed**. What they mean therefore changes when a select changes the data; a size,
`-1` and `None` mean the same thing on any array.

That, and nothing else, is what the rechunk pushdown matches on. Note what it does *not*
do: it reads no extent, no size and no schema, so the barrier follows from the spec alone
and the "no rewrite reads a size" property holds unqualified.

The three extent-dependent variants differ only in **how** they would fail, which is why
they share an arm despite looking unalike:

| variant | what a select in front would cause |
|---|---|
| `BlockSeq` | cannot replay at all — the blocks no longer sum to the dim |
| `ByteSize` | replays to a different answer — the byte target divides a different extent |
| `Auto` | either of the above, or a `ZeroDivisionError` if the select empties a dim |

The `Auto` case is the one that forced the whole variant to be a barrier rather than some
narrower condition. Whether it raises depends on dask's configured chunk size and on the
array's rank — nothing a plan can read — so no narrower discriminant exists.

`OpaqueChunk` is reached by a spelling xarray tolerates but does not document (a whole
float; `True`, which asks for blocks of 1) and by one it rejects outright (`0`, `-2`, a
set). Neither is modelled: the first is a spelling xarray's own API does not offer, and
the second is an error that stays the caller's to see. Recording it opaquely leaves the
rechunk a barrier and lets the failure surface at replay in xarray's words, exactly as it
would have eagerly.

A byte-target string is never opaque even when it is nonsense — it is a `ByteSize`, and
whether it parses is dask's judgement at replay, for the same reason.

## Where the user-facing version lives

[Rechunking](../guide/rechunking) walks the chunk taxonomy with an executed plan per
variant.
