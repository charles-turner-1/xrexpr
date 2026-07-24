# W7 — The Rust gate, and the PyO3 spike behind it

**Position (2026-07, superseding nothing — this makes `structural-dispatch-2.md` §7's
"name its trigger" concrete):** the port is **gated, then spiked, then decided**. Not
committed, not dismissed.

## Why gated rather than started

The honest ledger from doc 2 §6, updated:

- **Still no speed case.** `optimize()` runs once per `collect()` over ~10-node plans,
  before any array work. Rust cannot buy a measurable win here. The port's only
  arguments remain correctness sharpening (compiler-checked exhaustiveness, `Option`
  discipline on `var_dims`-style maybes) and keeping the door open for a second
  consumer of the IR.
- **The value layer is one workstream from closed** —
  [`02-chunk-taxonomy.md`](./02-chunk-taxonomy.md) removes the last `Any` from the
  reasoning surface, which is the precondition doc 2 §5 set for a clean FFI seam
  (semantic fields cross as native Rust data; `args`/`kwargs` cross as opaque
  handles).
- **But the container shape is about to change.**
  [`05-grouped-contexts.md`](./05-grouped-contexts.md) introduces the first
  sub-plan-carrying variant, and the tree/DAG is now confirmed long-term vision.
  Porting the linear-list optimiser first means porting twice.

## Gate conditions (all must hold before the spike starts)

1. W2 merged (chunk taxonomy — value layer closed).
2. W5 landed through its PR 4 (the `Contextual` variant and its first rule in the
   code), **or** an explicit decision to shelve W5 — either way the container shape
   is settled, not pending.
3. Someone is willing to own `cargo`/`maturin` in CI and locally. (Doc 2's
   "contributor who will own the Rust" trigger, demoted from trigger to
   prerequisite.)

## The spike (time-boxed: one week of effort, hard stop)

**Scope — port `optimize` and nothing else.** The seam is the one doc 2 §5 drew:
record and replay stay Python by necessity.

- A `rust/` crate (`xrexpr-opt`), PyO3 + maturin, not wired into the default install.
- **Types:** `enum Op` (fat variants mirroring `ir.py`, incl. `Contextual` if W5
  landed), `enum Indexer` (mirroring `indexers.py:185`), `enum ChunkSpec` (mirroring
  W2), `SchemaState`. Dim names: support `String` keys and **fall back to the Python
  optimiser for any plan with non-string Hashables** — dims are near-universally
  `str`, and the fallback keeps the port honest instead of forcing `PyObject` keys
  into `BTreeMap`.
- **Opaque handles:** `args`/`kwargs` and the open values (`Scalar`/`Label` payloads,
  `Opaque` nodes wholesale) cross as `Py<PyAny>`, stored and reordered but never
  inspected — the optimiser reads only the semantic fields, which is already true in
  Python (`structural-dispatch-2.md` §5).
- **Behaviour:** all rules in `_RULES` (`optimize.py:506`), the fixpoint, the
  termination measure, `apply_schema`, `_trusted_prefix`, and `InvalidExpressionError`
  raised across the boundary with the same message.

**Validation — the differential property test.** The spike's deliverable is not "it
compiles", it is: a Hypothesis suite generating plans (reusing
`test_properties.py`'s strategies) asserting `optimize_rs(plan, schema) ==
optimize_py(plan, schema)` node-for-node, including error parity on invalid plans.
Wired to run only when the extension is built (`pytest.importorskip`).

**Non-goals:** porting record/replay/`SchemaState.from_dataset`; any performance
claim or benchmark; publishing wheels; changing the Python package's default
behaviour.

## The decision, written down before the spike biases it

**Adopt** (behind `XREXPR_RUST=1`, pure-Python remaining the default and the
permanent fallback — the package must keep installing without a compiler) only if
*all* of:

- the port was a genuine transliteration — no semantic forks were needed, every
  divergence found by the differential suite was a spike bug, not a design gap;
- the dev loop (`maturin develop` + the existing pixi tasks) is judged acceptable to
  the people who actually maintain this;
- CI can build and test the extension (abi3 wheels or source-only) without doubling
  pipeline time.

**Otherwise:** write the findings into this file as an addendum — specifically *which*
seam assumption failed — delete or archive the crate, and stay Python. That outcome
is a success, not a failure: the structural workstreams (W1–W6) were each justified
on their own optimisation and correctness merits, and they are the durable payoff.
The one-line version, echoing both memos: **reform the structure and the port's seam
draws itself; the spike exists to test the seam, not to smuggle in a commitment.**
