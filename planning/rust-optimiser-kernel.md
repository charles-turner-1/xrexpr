# A Rust optimiser kernel via interning

## Context

The Rust spike so far ports the IR **data structures** — `Reduce` became a `#[pyclass]`,
and we hand-wrote `__hash__`/`__eq__` that delegate to Python's `hash`/`==` for the
verbatim `args`/`kwargs`. That work is correct, but it reproduces, by hand and across an
FFI boundary, exactly what `@dataclass(frozen=True)` already gave for free — and, more to
the point, it does not pay off even once the optimiser itself moves to Rust. A Rust
optimiser built on pyclasses would *still* thread `Py<PyAny>` through every reasoned-about
field: holding the GIL throughout, delegating every dim hash/eq to Python (`PyResult`
everywhere), unable to build a `HashSet` of dims or `match` over a clean `enum` — so it
would never reach the exhaustive integer algebra that is the whole point. The pyclass is
dominated either way. Fed straight to the Rust optimiser, it drags Python objects into the
algebra; lifted to a native form first, it was a pointless intermediate the Python
dataclass could have supplied. And the `Py<PyAny>` threading only worsens as more node
kinds land (dims are arbitrary `Hashable`, not `str`; see
`test_reduce_non_string_hashable_dim`).

The goal that actually justifies Rust here is **not** speed (the optimiser runs once, over a
handful of ops; array compute dominates by orders of magnitude). It is an **exhaustive,
compiler-checked algebra over the operations** — the thing `dim_effect` (`optimize.py`) and
`08-rust-gate.md` reach for with `assert_never`. In Rust, a non-exhaustive `match` is a
build failure, so "a new variant must answer every question" becomes a guarantee, not a
convention.

This memo sketches the design that gets there: **interning**. Rust becomes a *function*
over integers, not a *type* Python manipulates.

## The pipeline

```
record (Python)  →  lift (Python)  →  optimise (RUST)  →  lower (Python)  →  replay (Python)
   list[Op]          CorePlan          CorePlan            list[Op]
 (dataclasses)     (ints + tags)      (native enums)     (dataclasses)
```

The Python dataclass IR stays as the public recording form. Rust only ever sees `CorePlan`.

## The idea: intern dim names to integers

Dim names are arbitrary hashables, so `HashSet<DimName>` can't exist natively in Rust.
Interning sidesteps this: give each distinct dim name an integer ticket, keep a two-way
table, and reason over tickets. The Python object is touched exactly twice — intern (in)
and resolve (out). **Build the table in Python**, where hashing arbitrary objects is free,
so Rust never touches a Python dim object at all.

> ▸ **Exercise — the interner.** Write `intern(dim) -> int` / `resolve(id) -> dim` backed by
> a `dict` + `list`. Decide: is the interner per-`optimise` call, or persistent across a
> session? What keys the forward table — value equality or identity — and why does the
> answer have to be value equality? (Checkpoint: `intern(("a","b")) == intern(("a","b"))`.)

## Three buckets for everything a node carries

The optimiser never reads raw `args`/`kwargs` values (grep `optimize.py`: the only raw
`.kwargs` reads are over *keys*). It reads **normalised sum types** — `Select.indexer`
(`dim → Indexer`), `Rechunk.chunks` (`dim → ChunkSpec`), `mapping`, `consumes`. Everything a
node holds sorts into three buckets, and each has a home:

| bucket | examples | Rust home |
|---|---|---|
| numeric/structural (algebra computes on it) | `ForwardSlice{start,stop,step}`, `Positions`, `Mask` | native fields (`Option<i64>`, `Vec<i64>`, `Vec<bool>`) |
| identity/keys (dim names) | indexer/chunks keys, `Advanced.dims`, `consumes` | `DimId` / `HashSet<DimId>` |
| opaque (carried, not interpreted) | `Scalar.value`, `Label.value`, `GeneralSlice.value`, args/kwargs | side-table handle (`u32`), or interned-for-equality |

The truly uninterpretable case is *already* demoted to `Opaque`/`OpaqueChunk` at record
time (`classify`, `indexers.py`) — a barrier the optimiser won't reason across, so it needs
neither modelling nor interning. The modelled/opaque line the code already draws is the
line this design needs; you inherit it.

> ▸ **Exercise — the `Indexer` enum in Rust.** Port the seven `Indexer` variants
> (`indexers.py`) to a Rust `enum`, assigning each field to a bucket. Which variants become
> fully native? Which need a `DimId`? Which need a side-table handle? (Checkpoint: `Label`
> and `Scalar` should carry *no* interpreted Python — only a handle.)

## The boundary format

`CorePlan` crosses into Rust as **plain Python data** (lists, ints, small tagged
tuples/dataclasses) that pyo3 extracts via `FromPyObject` into native structs — **not**
pyclasses. No `__hash__`/`__eq__`/`borrow` on the Rust side; `#[derive(Hash, PartialEq, Eq)]`
just works because every field is an integer or a native sum type.

> ▸ **Decide — the wire format.** How is one op represented on the Python side before it
> crosses? A tagged tuple `(kind, ...)`? A small plain dataclass per kind? A flat columnar
> form? Pick the one whose `FromPyObject` is most mechanical, and note the trade you made.

## Sequencing — boundary first, algebra once

1. **`lift` + `lower`, pure Python.** Build `CorePlan` and the two conversions. No optimiser,
   no Rust yet.
2. **Round-trip test the boundary, alone.** Assert `lower(lift(p)) == p` over the suite and
   via hypothesis. This validates the scary part with zero optimiser and zero Rust.
3. **Write the optimiser once, in Rust**, over `CorePlan`.
4. **Differential-test against the oracle you already have:** assert
   `lower(rust_optimise(lift(p))) == optimize(p)` (`optimize.py`). Do **not** rewrite the
   optimiser in Python first — the existing one *is* the reference. Step 2's identity
   guarantee is what makes step 4 a fair test of the algebra alone.

Start smaller than step 1's full scope: get `lift`/`lower` + round-trip green for just
`Reduce` and `Select` before committing to the whole port.

## The exhaustiveness payoff (the point)

Once ops are a Rust `enum`, `dim_effect` is a `match` where a missing arm is a compile
error. Port `dim_effect` (`optimize.py:232`) first as the smoke test: it touches the most
node kinds and is pure metadata, so it proves the enum earns its keep before you port a
rewrite rule. `DimSet` becomes `AllDims | Concrete(HashSet<DimId>)` — a total algebra whose
`union`/`intersect`/`subset`/`is_disjoint` you define once, each an exhaustive two-arm match.

> ▸ **Exercise — `dim_effect` in Rust.** Port it. Where the Python arm returns the
> conservative `None`/`None`, keep that, but let the compiler force you to write the arm.
> Which Python runtime `assert_never` sites vanish entirely?

## Open decisions to settle yourself

- How is `ALL_DIMS` represented once dims are integers? (It is *not* a `DimId`. Is it a
  variant of `DimSet`, or a reserved sentinel id? What did the pyclass do, and why does that
  inform this?)
- `keepdims` is derived from kwargs today (`Reduce.new`). Does it stay a derived-at-lift
  boolean on `CorePlan`, or get recomputed? Where is the single source of truth?
- Equality on `Label`/`Scalar` values (composing two label-selects): intern them too, or
  compare opaque handles via Python `==` at the few sites that need it? What does each cost?
- Which ops cross first, and which stay Python-only barriers for now (`GroupedReduce`,
  `WeightedReduce`, `Rechunk`)? A partial port is fine if unported kinds lower to `Opaque`.
- Where does normalisation (`classify`, `_dim_spec`) live — it stays in Python at record
  time, so lift consumes *already-parsed* nodes. Confirm nothing in lift re-parses raw args.

## What to stop doing

Retire the `Reduce` pyclass hash/eq work in `rust/src/ir.rs`. Under this design the boundary
is data, not objects, so there is no pyclass to give value semantics to. Keep the memo of
*why* it was hard (the `Py<PyAny>` hash/eq delegation) — it is the argument for interning.
