# Design discussion: how §7 of the structural-dispatch memo was reached

Companion to [`structural-dispatch.md`](./structural-dispatch.md). That memo states the
*conclusions* (most recently the committed IR shape in **§7**); this file records the
*path* — the questions asked, the turns that proved wrong, and why the decision landed
where it did. Kept separate from the memo so the memo stays a clean design statement while
the reasoning remains recoverable.

**TL;DR of the outcome:** commit the unary sum type **now**, as **fat variants**
(`Op = Reduce | Select | Scan | Opaque`, each self-contained). Plain composition rejected.
Replay stays a verbatim passthrough. Binary ops (tree/DAG) deferred as an additive,
orthogonal change. The memo's original "not now" is superseded.

---

## 1. Starting point

The memo argued *for* the sum type but recommended **deferring** it: one real
kind-dispatch site (`pushdown_selects`), so the day's-work-and-wide-diff wasn't yet paid
for. Triggers were listed (§5); until one hit, the flat `OpNode` was "the right size."

## 2. The question that opened it

> "Replacing `OpNode` with `Select | Reduce` looks like a no-brainer — much clearer to
> read. Any reason not to?"

Clearer at the dispatch site — yes. But "no-brainer" turned out to overstate it.

## 3. Why it isn't a clean win: not every consumer dispatches

`OpNode` has **three** consumers, and only one wants variants:

| consumer | wants | why |
|---|---|---|
| `pushdown_selects` | **variants** | binds different fields per kind — the one pass the sum type helps |
| `_replay` / `_format_node` / `__repr__` | **the flat shape** | call every node identically: `getattr(ds, name)(*args, **kwargs)` — no dispatch |
| `apply_schema` | **per-node field access** | reads `.consumes` + `.indexer` on every node |

So the sum type is clearer *at the dispatch site* and more *ceremonious* at the uniform
sites. Net line count can even rise. (This became §5.1.)

## 4. The replay lifecycle (we had to pin this down)

"Replay" was load-bearing but undefined, so we nailed it (now in the memo's Context):

1. **Record** — each `.plan` call is normalised to an `OpNode` and appended to a list; nothing runs.
2. **Optimise** — `optimize()` rewrites the list (merge / pushdown); the *only* stage that reads `kind`/`consumes`/`indexer`.
3. **Replay** — `getattr(ds, node.name)(*node.args, **node.kwargs)` down the list; actually calls xarray.

**Key property:** replay is a **uniform passthrough** — it never looks at
`kind`/`consumes`/`indexer`. Whatever shape the IR takes, every node must still hand replay
a `name` + `args`/`kwargs`.

## 5. Reconstruction is a trap (the `skipna` detour)

Question raised: *why would we ever drop kwargs?* Answer: we wouldn't — but a tempting
design ("store only semantic fields, rebuild the call at replay") silently would. The
semantic fields are a **lossy summary**, not the whole call:

- `mean("lat", skipna=False)` → `consumes={'lat'}`, but `skipna` lives only in `kwargs`.
- `std("time", ddof=1)` → `ddof` only in `kwargs`.
- `isel(time=0, drop=True)` → `drop` is stripped into `kwargs` (`schema.py:110`).
- `opaque` ops have **no** semantic fields to rebuild from at all.

So rebuilding from `consumes`/`indexer` would drop `skipna`/`ddof`/`drop` — a wrong-answer
bug. **Decision: keep `args`/`kwargs` verbatim on every variant** (semantic fields layer
*on top of* the call payload, never instead of it). This is §7.5.

## 6. The shape bake-off: plain composition vs hybrid vs fat variants

Three ways to model "replay header on every variant + semantic fields per variant":

- **Plain composition** — `OpNode(name, args, kwargs, op: Reduce | Select | …)`.
  **Footgun 1 (fatal):** `name` sits *outside* the union, so
  `OpNode(name="mean", op=Select(...))` is constructible — the "name matches kind"
  invariant is back, by convention not construction.
- **Fat variants** — each variant carries `name`+`args`+`kwargs`+its semantics; single-level
  `match node`. `name` can be `Literal`-typed per variant.
- **Hybrid** — composition's header, but `name` pushed into the payload so it can be
  `Literal`-typed.

**Footgun 2 (a wash across all shapes):** `merge_adjacent_selects` builds a merged select's
`args=(dict(indexer),)` *from* the merged `indexer` — so `args` and `indexer` must agree.
This survives in every shape; it lives centralised in the merge rule (the honest place for
it). The sum type kills the *`consumes`↔`indexer`* desync (via the `@property`), not this
one.

**Correction made mid-discussion:** composition's "uniform replay" edge was oversold — fat
variants *also* get match-free replay (mypy allows attribute access on a union when all
members share the field). So replay ergonomics are ~a wash; composition's real edges are
only no-field-duplication and (initially mis-stated) Rust idiom.

## 7. The Rust "unrepresentable" clarification (the pivotal insight)

Claim tested: *"Python has this footgun, but a Rust port would make it unrepresentable."*
**Only half true, and it flipped the decision:**

- The name↔kind desync is a **shape** fault, not a Python weakness. A literal port of plain
  composition — `struct OpNode { name: String, op: OpKind }` — **inherits it**: Rust will
  happily build `OpNode { name: "mean", op: Select {..} }`.
- What makes it unrepresentable, in *either* language, is putting `name` **inside** the
  variant (`Literal["isel","sel"]` in Python; a `SelectName` enum in Rust).
- So: **decide the shape now — the port strengthens the check, it can't retrofit the
  structure.** This ruled out plain composition, and walked back an earlier over-claim that
  "Rust favours composition" (Rust enums with repeated per-variant fields are perfectly
  idiomatic).

## 8. Where the win actually is (an uncomfortable finding)

Rewriting the two rules under the sum type:

- **`pushdown_selects`** (the only kind-dispatch site) — barely improves; the
  `kind`/`_SELECTS` guard becomes a `case` pattern, set algebra unchanged. Roughly a wash.
- **`merge_adjacent_selects`** — genuinely improves: the second accumulator (`consumes |= …`,
  the latent-bug one) **vanishes**, because `consumes` becomes a derived `@property` on `Select`.

**The catch:** that biggest win comes from *`consumes`-as-property*, **not** from
`match`/dispatch. You could get it without the full sum type. Honest framing kept in the
memo.

## 9. "Design for growth" reframed it — kinds, not methods

The steer: the project *will* grow; get the shape right early; happy to write a long enum.
Two clarifications resulted:

- This retires the "too small, not now" objection — designing the shape early is a
  legitimate goal. (Superseded §5–§6.)
- **Variants track structural *kinds*, not xarray *methods*.** `mean`/`std`/`sum` are all
  one `Reduce`, told apart by `name`. New *methods* → rows in `OP_TABLE`; a new *variant* is
  earned only by genuinely new structural data (`groupby`→grouper, `rolling`→size). A
  method-level enum would be the anti-pattern. So "willing to write a long enum" is a red
  flag, not an asset — the enum stays a dozen-ish arms.

## 10. The container question: list vs tree (and why it's deferrable)

Growth surfaces a *bigger* shape question than fat-vs-hybrid: the IR is a **linear
`list[OpNode]`**, but "anything in xarray" includes **binary ops** (`merge`, `concat`,
`where(cond, other)`, `ds1 + ds2`) with a second input. Three tiers:

1. **unary, constant params** — a list holds perfectly (nearly all of xarray).
2. **binary, eager operand** — a list still holds it as an opaque node carrying the operand.
3. **binary over a *lazy* operand you optimise across** (or a shared lazy prefix) — this,
   and only this, forces a **tree/DAG** (the Polars-`LazyFrame` shape).

**The unblock:** Tier 3 is **orthogonal** to the unary variant shape. It *adds* a
`Join`/`Concat` variant with plan-typed children and promotes the container; it does **not**
reopen `Reduce`/`Select`/`Scan`. So we can commit the unary shape now and add Tier 3 later
as an additive change. Whether Tier 3 is in scope is a *feature* decision (is xrexpr
Polars-for-xarray?), not a code one, and it isn't on the table yet.

## 11. The decision (→ §7) and what we accepted eyes-open

**Committed:**
- Unary sum type **now**, as **fat variants**: `Op = Reduce | Select | Scan | Opaque`, each
  self-contained, single-level `match node`.
- `Literal` names where closed (`Select`, `Scan`); `str` where open (`Reduce`, `Opaque`).
- `consumes` a `@property` on `Select`; `Elementwise` dropped (phantom kind).
- Replay stays verbatim.
- Binary ops / tree deferred as additive + orthogonal.

**Accepted, eyes-open:**
- The `args`↔`indexer` consistency in the merge rule (footgun 2) survives — centralised there.
- The headline concrete win is the merge accumulator, which comes from the `@property`, not
  the dispatch. Pushdown barely benefits.
- Discipline while the tree is deferred: **keep the linearity assumption named in one place**
  (`ir.py`'s docstring already does) rather than leaking it into every rule.

## 12. Not decided / next

- Whether Tier 3 (lazy joins → tree/DAG) is ever in scope — a product-vision call.
- The small independent wins from §6 still stand (tighten `Rule` to `Plan | None`; the
  trichotomy docstring; writing down the merge invariant).
- Implementation itself (the memo PR is docs-only): touches `ir.py`, `schema.py`
  `to_opnode`, `operations.py`, the accessor replay, and every golden-`OpNode` test.
