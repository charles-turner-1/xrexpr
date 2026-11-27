# The value types

Two operations carry an argument that is really one of a fixed set of shapes:

- `Select.indexer` maps each dim to one **indexer**: a position, a slice, a sequence, an
  array, or a coordinate label.
- `Rechunk.chunks` maps each dim to one **chunk spec**: a size, a symbolic request like
  `"auto"`, or an explicit list of block sizes.

Stored as `Any`, either would force every user in `ir.py`, `optimize.py` and `schema.py`
to work out the shape by hand. `indexers.py` and `chunks.py` turn each into a proper type
instead. They were written months apart and ended up following the same four rules.

## Rule 1: one function builds every value

One function per type decides which variant a raw value is (`classify` and
`classify_chunk`), and every raw value goes through it. Nothing else sorts values, and no
caller guesses.

Both accept anything. A value the type can't model gets the fallback variant rather than
an error. That matters because classification happens at recording time, on whatever the
user passed. Raising there would reject a chain xarray would have accepted.

## Rule 2: facts about a value are methods

Each variant answers the questions its users ask:

| | indexers | chunk specs |
|---|---|---|
| does it drop the dim? | `drops_dim` | n/a |
| how long is the dim afterwards? | `size()` | n/a |
| what does replay pass to xarray? | `to_raw()` | `to_raw()` |

Both need `to_raw`, because replay rebuilds each call from the node's parsed fields. A
select merged with another, or a rechunk that lost a dim to a select moved in front of
it, has to be turned back into exactly the value xarray expects, and the variant knows
what that is.

## Rule 3: decisions stay in the optimiser

Neither module decides what the optimiser should *do*.

Whether two indexers **compose** isn't a method on `Indexer`, and whether a chunk spec
may be **crossed** isn't a method on `ChunkSpec`. Both are choices the optimiser makes
about what it's willing to prove, not facts about a value. So both are a `match` in
`optimize.py`, closed with `assert_never`, so a new variant fails type-checking until
someone decides which side it belongs on.

The value modules just provide the variants those decisions are based on. No variant has a
flag saying what should happen to it.

## Rule 4: one fallback per level

`Label` for indexers, `OpaqueChunk` for chunk specs, and `Opaque` for whole operations.
The same idea at three levels: "this is real, it will replay verbatim, and nothing may be
inferred from it."

That's what lets the classify functions accept anything, and why an unrecognised value
costs a rewrite rather than correctness.

## Indexers

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

Only one variant drops its dimension, and `Select.consumes` is derived from that. The
order of the checks matters in two places:

- A boolean sequence is a `Mask`, not `Positions`, even though `bool` is a subclass of
  `int`. So the all-boolean check runs first.
- A 0-d array is a `Scalar`, not a one-element list, so the rank check runs before the
  dtype checks.

**`ForwardSlice` is its own variant** so that "forward, non-negative bounds", which the
composer relies on, is guaranteed by the constructor (you can't build a `ForwardSlice`
with a negative bound) rather than re-checked everywhere. `GeneralSlice` takes every
other integer slice. A slice with a non-integer bound is a label slice, which isn't
positional at all.

**`Label` is the fallback.** A `sel` coordinate label (a string, a timestamp, a tuple key,
a label slice, a list of labels) can't be reasoned about positionally.

`size()` doesn't handle unknowns. Sizing a dim of unknown length, or a `sel` label slice,
is answered as "unknown" one level up, by the check that wraps every call to `size()`.

## Chunk specs

Seven variants, in a similar tree, but grouped by a property indexers don't have.

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

There's one variant per *behaviour*, not per spelling, and the split was checked against
the pinned xarray and dask rather than reasoned about. The deciding case: on an
already-chunked dim, `None` leaves the existing blocks alone, while `-1` merges them into
one. So `NoChange` and `FullDim` are separate.

### Extent-dependence is the discriminant

`Auto`, `ByteSize` and `BlockSeq` are the specs dask resolves by **measuring the array
it's given**. So their meaning changes if a select changes the data first. A fixed size,
`-1` and `None` mean the same thing on any array.

That's the only thing the rechunk pushdown checks. It reads no size and no schema, so the
decision comes from the spec alone, and "no rule reads a size" holds without exceptions.

The three extent-dependent variants share a rule even though they look different. They
only differ in **how** they'd fail if a select moved in front:

| variant | what a select in front would cause |
|---|---|
| `BlockSeq` | can't replay at all: the blocks no longer add up to the dim |
| `ByteSize` | replays to a different answer: the byte target divides a different size |
| `Auto` | either of the above, or a `ZeroDivisionError` if the select empties a dim |

`Auto` is why the whole variant blocks selects rather than some narrower condition.
Whether it raises depends on dask's configured chunk size and on the number of dims, and
a plan can't see either. (The one exception: on xarray 2026.09.0 and later, a select that
provably empties the dim may cross, because xarray handles that case before dask sees it.)

`OpaqueChunk` catches two kinds of value: spellings xarray tolerates but doesn't document
(a whole float, or `True`, which asks for blocks of 1), and ones it rejects (`0`, `-2`, a
set). Neither is modelled. Recording them opaquely makes the rechunk a barrier, and any
error comes from xarray at replay, in its own words, as it would have eagerly.

A byte-target string is never opaque, even when it's nonsense. It's a `ByteSize`, and
whether it parses is left to dask at replay, for the same reason.

## The user-facing version

[Rechunking](../guide/rechunking.md) walks through the chunk spec variants with a real
plan for each.
