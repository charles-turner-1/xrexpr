# W8 — The Rust port (decision: commit)

**Position (2026-08-28, reverses the gate).** The port is no longer *gated, spiked, then
decided*. It is **committed.** xrexpr becomes a native Rust/Python package: the optimiser
core moves to a PyO3 crate, maturin becomes the build backend, and the pure-Python
optimiser is kept only as long as it is useful as a correctness oracle. The gate-and-spike
framing that filled this file until today is preserved at the bottom, marked superseded.

## What changed, and the honest reason

Not a new speed case. The ledger from `structural-dispatch-2.md` §6 is **unchanged**:
`optimize` (`optimize.py:85`) runs once per `collect()` over ~10-node plans, before any
array work, and Rust buys no measurable win there. Anyone reading this later should not go
looking for the benchmark that flipped the decision — there isn't one.

The reversal is a maintainer decision, made on three facts:

1. **The gate conditions are all met.** W4 (chunk taxonomy) and W2 phase 1 (`AllDims`,
   `to_lower_ir`, `emit`, the pipeline rewiring) have both landed — the shape the rules
   consume is settled, `ChunkSpec` (`chunks.py:275`) removed the last `Any`, and the whole
   structural programme (W1–W7, W10, W11) is on `main`. The gate would open on its own
   terms today.
2. **Prerequisite 3 — a contributor to own `cargo`/`maturin`, in CI and locally — is now
   satisfied.** The maintainer is taking that on directly, and building a proper native
   package with Python bindings is an *explicit goal of the work*, not a cost reluctantly
   paid. That reframes the port: it no longer has to earn its place against "no speed case"
   alone. The standing arguments (compiler-checked exhaustiveness, `Option` discipline over
   the `var_dims`-style maybes, keeping the door open for a second consumer of the IR) plus
   the learning goal are judged sufficient to commit.
3. **The costs are understood and accepted** (recorded in full below): a permanent wheel
   matrix, a Rust compiler in the dev and CI loop, the loss of the "installs anywhere with
   no compiler" property, and the coupling of the whole package's fate to Rust.

So the plan is no longer "gate → time-boxed spike → adopt only if a transliteration." It is
**commit → build out → keep the differential test as the one hard gate on faithfulness.**

## What "commit" means concretely (this reverses the fallback clause)

The superseded framing promised pure-Python "remaining the default and the permanent
fallback — the package must keep installing without a compiler," behind `XREXPR_RUST=1`.
**That clause is withdrawn.** In its place:

- **maturin is the build backend.** xrexpr is a native package; there is no pure-Python
  wheel and no `XREXPR_RUST` opt-in. The Rust extension is mandatory.
- **Distribution is prebuilt wheels.** `osx-arm64`, `linux-64`, `win-64` (the platforms
  already in `[tool.pixi.workspace]`), built in CI via `maturin-action`/cibuildwheel, abi3
  to collapse the CPython-version axis to one wheel per platform. An sdist is published and
  compiles for anything unshipped. This is the CI rebuild the maintainer has signed up for.
- **The dev loop is `maturin develop` + the existing pixi tasks.** The editable install in
  every pixi env now compiles the crate, including the matrix arms that have nothing to do
  with Rust — that is the accepted cost, not an oversight.

## The seam does not move (technical, not a preference)

Going "all in" does **not** mean all of `src/` becomes Rust. The seam
`structural-dispatch-2.md` §5 drew is a fact about what touches live xarray, not a hedge:

- **record stays Python.** `to_opnode` (`schema.py:923`) and `SchemaState.from_dataset`
  inspect live `xr.Dataset`/`DataArray` objects; `to_lower_ir` (`lower.py:95`) and `emit`
  (`lower.py:536`) bracket them.
- **replay stays Python.** `_replay` (`accessor.py:432`) is a `getattr(obj, call.name)(...)`
  loop over xarray methods.
- **`optimize` and the IR it reasons over move to Rust.** The rules, the fixpoint, the
  termination measure (`optimize.py`), `dim_effect` (`optimize.py:233`), `apply_schema`
  (`schema.py:330`), `_trusted_prefix` (`optimize.py:156`), and the `_RULES` table
  (`optimize.py:1465`).

The Python package becomes a thin **recording/replay shell over a Rust optimiser core**.
That is the same seam the spike scoped — the only change is that it is now permanent and
load-bearing rather than an experiment with a Python safety net behind it.

## The port surface (types and behaviour)

Unchanged from the spike scope, anchors refreshed against `main`:

- **Types.** `enum Op` (fat variants mirroring `ir.py:787`, plus the lowered
  `GroupedReduce`/`WindowedReduce`/`WeightedReduce` from `ir.py:797`), `enum Indexer`
  (`indexers.py:460`, including `Advanced`), `enum ChunkSpec` (`chunks.py:275`, all seven
  variants), and `SchemaState` (`schema.py:76`).
- **Opaque handles.** `args`/`kwargs` and the open payloads (`Scalar`/`Label` values,
  `Opaque` nodes wholesale) cross as `Py<PyAny>` — stored and reordered, never inspected —
  because the optimiser already reads only the semantic fields (`structural-dispatch-2.md`
  §5).
- **Behaviour.** Every rule in `_RULES`, the fixpoint, the termination measure,
  `apply_schema`, `_trusted_prefix`, and `InvalidExpressionError` (`exceptions.py:4`) raised
  across the FFI boundary **with the same message**.

## The build-out (replaces "the spike")

Phased, so each phase leaves the tree green and the package importable:

1. **Infra.** maturin backend, the `rust/` crate, the pixi `rust` feature/task, the CI
   wheel-and-test matrix, and the versioneer resolution (below). Deliverable: a stub
   `#[pyfunction]` builds and imports as part of `xrexpr` on all three platforms, in CI. This
   is the "learn to build a native package" phase, and it is worth landing on its own before
   any optimiser logic rides on it.
2. **IR types.** The four enums above as PyO3 classes, constructible from and comparable
   with their Python counterparts. No behaviour yet.
3. **`optimize`.** The rules, fixpoint, `dim_effect`, `apply_schema`, `_trusted_prefix`,
   error parity.
4. **Differential validation** (the one hard gate — see below).
5. **Flip the default and decide the Python optimiser's fate** (open item below).

## The one non-negotiable: the differential property test

The deliverable is not "it compiles." It is a Hypothesis suite — reusing
`test_properties.py`'s plan strategies — asserting `optimize_rs(plan, schema) ==
optimize_py(plan, schema)` **node-for-node, including error parity on invalid plans.** This
survives the reversal intact and is *more* important now than under the spike: with no
Python fallback shipped, the differential test is the only thing standing between a port bug
and a wrong answer in production. It runs in CI against the built extension. The pure-Python
`optimize` is therefore kept alive **as the oracle** at least through phase 4.

## Open migration items (decide as you reach them)

The reversal surfaces three decisions the spike could dodge because it shipped nothing:

- **Versioning.** maturin owns the wheel metadata, so `[tool.versioneer]` no longer feeds
  `[project].version`. maturin has no setuptools-scm-style git-tag scheme built in. Options:
  make `rust/Cargo.toml`'s `package.version` the single source (manual bumps), or derive the
  version from the git tag in CI and stamp both. Leaning: single-source in `Cargo.toml`,
  retire versioneer — but confirm what `_version.py` consumers (`src/xrexpr/_version.py`,
  the coverage/mypy excludes) need first.
- **Non-string dim keys.** The spike's escape hatch was "fall back to the Python optimiser
  for any plan with non-string `Hashable` dims." **That fallback no longer exists.** Either
  the Rust side carries `PyObject` keys for the general case, or the recorder conservatively
  records such plans in a way the Rust optimiser can pass through untouched. Dims are
  near-universally `str`, so the conservative-record route is cheapest; decide before phase 3.
- **The Python optimiser's fate after parity.** All-in argues for deleting it once the
  differential suite is green on the matrix. Against: it is a cheap, readable oracle that
  keeps the port honest on every future rule. Leaning: keep it in `tests/` as a reference
  oracle rather than in `src/`, so it stops being shipped but stays the differential anchor.

## Accepted costs (recorded, per house practice)

Named so a future reader sees they were chosen, not missed:

- A permanent **wheel matrix** to build and publish on every release; unshipped platforms
  compile from sdist.
- A **Rust compiler in the dev loop** — every editable pixi install, every CI matrix arm.
- Loss of the **compiler-free install** and the pure-Python wheel; the package's build,
  distribution, and versioning are now coupled to Rust.
- **CI is rebuilt**, knowingly. The pure-Python-only arms change meaning; the pre-commit and
  type/lint jobs stay Python, but the test job now depends on a built extension.

---

## Superseded (2026-07): the gate-and-spike framing, kept for the record

The original position was *gated, then spiked, then decided — not committed, not
dismissed.* It rested on: no speed case; the value layer one workstream (W4) from closed;
and the IR "about to gain a level" via lowering (W2), so porting first would port against a
non-final shape. The gate conditions were (1) W4 merged, (2) W2 phase 1 merged or shelved,
(3) a contributor to own `cargo`/`maturin`. All three are now met — the first two by the
structural programme landing, the third by the maintainer taking it on — which is *why* the
gate is being opened rather than overridden.

The spike was scoped to "port `optimize` and nothing else," time-boxed to one week, with the
adopt/abandon decision "written down before the spike biases it": adopt behind
`XREXPR_RUST=1` only if the port was a genuine transliteration, the dev loop was acceptable,
and CI could build the extension without doubling pipeline time; otherwise archive the crate
and stay Python. **That conditional adoption is what today's decision replaces with an
unconditional one** — the transliteration test now lives on as the differential property
suite (a faithfulness gate on the *code*), not as a go/no-go on the *port*. The structural
workstreams remain justified on their own optimisation and correctness merits regardless, as
that framing insisted; that has not changed and is not in question.
