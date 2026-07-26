# W8 — The Rust gate, and the PyO3 spike behind it

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
  [`04-chunk-taxonomy.md`](./04-chunk-taxonomy.md) removes the last `Any` from the
  reasoning surface, which is the precondition doc 2 §5 set for a clean FFI seam
  (semantic fields cross as native Rust data; `args`/`kwargs` cross as opaque
  handles).
- **But the IR is about to gain a level.** ~~[`09-grouped-contexts.md`](./09-grouped-contexts.md)
  introduces the first sub-plan-carrying variant~~ — superseded (2026-07). The container
  stays a flat list, but [`02-lowering.md`](./02-lowering.md) splits the IR into a fluent
  level and a lowered level, and it is the **lowered** one the rules see. Porting the
  optimiser first means porting against a plan shape that isn't the final one.

  Lowering also *improves* the surface being gated on, in two ways worth naming since both
  are preconditions doc 2 §5 set: `emit` takes payload reconstruction out of the rules
  (`02` §7), so the optimiser touches semantic fields **only** — today
  `merge_adjacent_selects` still rebuilds `args` (`optimize.py:169`). And `02` PR 1
  replaces the eagerly-expanded bare-reduce dim set with an `AllDims` sentinel, which is a
  cleaner Rust enum than a set that means two different things.

## Gate conditions (all must hold before the spike starts)

1. W4 merged (chunk taxonomy — value layer closed).
2. **W2 phase 1 merged** (`02` PRs 1–2: `AllDims`, `to_lower_ir`, `emit`, the pipeline
   rewiring), **or** an explicit decision to shelve W2 — either way the shape the rules
   consume is settled, not pending. Note this is a *weaker* gate than the superseded W9
   one: the fused nodes (`02` PRs 3–5) need not have landed, since adding a variant to a
   flat enum is additive on both sides of the FFI.
3. Someone is willing to own `cargo`/`maturin` in CI and locally. (Doc 2's
   "contributor who will own the Rust" trigger, demoted from trigger to
   prerequisite.)

## The spike (time-boxed: one week of effort, hard stop)

**Scope — port `optimize` and nothing else.** The seam is the one doc 2 §5 drew:
record and replay stay Python by necessity.

- A `rust/` crate (`xrexpr-opt`), PyO3 + maturin, not wired into the default install.
- **Types:** `enum Op` (fat variants mirroring `ir.py`, incl. whichever of
  `GroupedReduce`/`WindowedReduce`/`WeightedReduce` have landed), `enum Indexer`
  (mirroring `indexers.py:185`), `enum ChunkSpec` (mirroring
  W4), `SchemaState`. Dim names: support `String` keys and **fall back to the Python
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
is a success, not a failure: the structural workstreams (W1–W7, W2–W3) were each
justified on their own optimisation and correctness merits, and they are the durable
payoff.
The one-line version, echoing both memos: **reform the structure and the port's seam
draws itself; the spike exists to test the seam, not to smuggle in a commitment.**
