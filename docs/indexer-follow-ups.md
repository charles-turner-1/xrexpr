# Indexer value sum type: follow-ups after PR #68

Working notes, not a design doc. Three items left over from verifying PR #68 against
[`structural-dispatch-2.md`](./structural-dispatch-2.md) §3. Each is independent; the branching
differs per item and is **not** what it first looks like, so that is called out for each.

Status (2026-07-24): **all merged** — #66, #67, #68, #70 and #71 are on `main`. Items 1
and 3 are done as described below. **Item 2 (the chunk-spec taxonomy) is the only one
still open**; it is now specified in full as
[`roadmap/02-chunk-taxonomy.md`](./roadmap/02-chunk-taxonomy.md) — implement from that
spec, not from the sketch below. Original status table kept for the record:

| PR | branch | base | item |
|---|---|---|---|
| #68 | `indexers-value-sum-type-3` | `main` | the optimizer wiring + property coverage — merged |
| #70 | `indexers-mask-value-type` | #68 | **item 3 below — done**, merged |
| #71 | `indexers-compose-exhaustive` | #68 | **item 1 below — done**, merged |

---

## 1. `_compose_indexer` has no `assert_never` — DONE (#71)

Resolved by spelling out the four refusing shapes and closing the match with `assert_never`,
scoped to `_compose_indexer` only; `_index_sequence` and `_compose_slice` keep their wildcards
as argued below. Verified by adding a temporary 7th variant, which fails mypy at exactly that
site. Original note follows.

**Branch off #68, not main.** This is the one that cannot go off main: on main
`_compose_indexer` is still `(outer: Any, inner: Any) -> Any` returning `_UNCOMPOSABLE` — the
old `isinstance` ladder. The `match` over `Indexer` that this note is about only exists on #68.
So either stack on #68, or wait for it to merge.

§3.1 promises the composition policy "becomes a visible, exhaustive set of arms instead of a
fall-through `return _UNCOMPOSABLE` you have to trust covers the rest", via "one `match` with
`assert_never`". #68 delivers the `match` and retires the sentinel, but keeps a catch-all:

- `optimize.py:236` — `_compose_indexer`
- `optimize.py:258` — `_index_sequence`
- `optimize.py:297` — `_compose_slice`

Functionally safe (a fall-through refuses to merge, which is the conservative direction), but it
gives up the property §2.1/§3.1 actually sell: adding a seventh `Indexer` variant would silently
become uncomposable rather than failing mypy. `ir.py` and `schema.py` both use `assert_never`
already, so the discipline is applied at the node layer and not the value layer.

**Suggested scope — `_compose_indexer` only.** That is the *policy* site, where "which outer
shapes compose" is the decision worth making exhaustive and where the memo's argument really
lands. Write out `Scalar | GeneralSlice | Mask | Label → None` explicitly, then `assert_never`.

Leave the two inner helpers with their catch-alls. Their matches are over `inner` and the
catch-all covers genuinely unrelated shapes; enumerating them buys the same compile-time check
at a real cost in readability, and `_index_sequence`'s arms would have to repeat the same
`-> None` four times. Worth deciding deliberately rather than applying `assert_never` everywhere
by reflex — note that this is a *deviation* from a literal reading of §3.1, so say so in the PR.

---

## 2. The chunk-spec half of §3 is not done — STILL OPEN

**Branch off main.** Verified clean: #68 does not touch `_pushable_rechunk` or `Rechunk.chunks`
at all, so this will not conflict. Neither do #70 or #71.

§3 scopes the value sum type to "the *values* inside a select's `indexer` **and a rechunk's
`chunks`**", and lists `_pushable_rechunk` as the fourth `isinstance` ladder alongside the three
that #66–#68 retire. Only the indexer half shipped:

- `ir.py:174` — `chunks: frozendict[Hashable, Any]`, still `Any`
- `optimize.py:470` — `_pushable_rechunk` still does `isinstance(node.args[0], list | tuple)` to
  tell an explicit block sequence (a barrier) from a single size

§6's ledger row ("the indexer/chunk **value** sum type (§3) — still `Any`") is therefore only
half-cleared, and §7's "do first" recommendation is only half-done.

This is a **real piece of work, not a cleanup** — a second taxonomy, its own `classify`, and
`_pushable_rechunk` rewritten against it. It deserves its own PR and probably its own think about
what the variants are (single size / explicit block sequence / `"auto"` / `None` / dict form?).
The prior note in project memory already flagged it as "separate, lower-value taxonomy", so it is
worth confirming it is wanted before building it.

---

## 3. `Mask` breaks equality on the node that contains it — DONE (#70)

Resolved by storing `tuple[bool, ...]`. **One claim below turned out to be wrong**: the note
said this would make `Mask` "the one variant whose `to_raw()` isn't the identity", and framed
that trade as the substance of the PR. It isn't — `Positions.to_raw()` already returns
`list(self.values)` rather than its field, so `Mask` rebuilding a list is *consistency*, not an
exception. The trade largely evaporated on inspection. Original note follows, uncorrected.

**Branch off #68.** `Mask` itself is unchanged since #66, but the fix touches `indexers.py`
(which #68 has already modified at `e94ade8`) and would let a test-level workaround be removed
that only exists on #68 — see below.

This is the same footgun class as the `Scalar(np.array(0))` one fixed in `e94ade8`, but it bites
harder. `Mask` stores its values verbatim, so:

```python
Mask(np.array([True, False, True])) == Mask(np.array([True, False, True]))
# -> array([True, True, True])     an *array*, not a bool

Select(name="isel", indexer=frozendict({"x": mask})) == <the same Select>
# -> ValueError: The truth value of an array with more than one element is ambiguous
```

So **node equality raises** when a mask is in the indexer, and `hash()` fails too (for both the
ndarray form and the list form — `Mask([True, False])` holds a `list`).

**Latent today, not live.** Nothing in `src/` hashes or set-compares an `Op`, and a masked plan
records, optimises and collects correctly — verified. But `frozen=True` advertises hashability,
and plan *equality* is used in tests: `test_properties.py::test_optimize_is_idempotent` does
`assert optimize(once, schema) == once`. It only stays green because those generators never
produce a mask. Two places currently work around it rather than fix it:

- `tests/test_indexers.py:302` — `assume(not isinstance(indexer, Mask))` in the round-trip
  property, commented "ndarray-backed, so `==` is ill-defined"
- the same reason `test_mask_to_raw_returns_the_mask` used to assert identity (`is`) rather than
  equality

**The design question to settle first**, before writing any code: normalising to
`tuple[bool, ...]` makes `Mask` hashable and equality-sane, matching what `Positions` and
`ForwardSlice` already do — but `to_raw()` then has to return something xarray accepts, and a
*tuple* is not it (`da.isel(x=(0, 2))` raises `TypeError`; a tuple of bools was not tested but
should not be assumed to work). So `to_raw()` would need to rebuild a `list` or `np.array`, which
means `to_raw()` stops being the identity it is for every other variant. That trade is the whole
substance of this PR.

Once fixed, drop the `assume` at `test_indexers.py:302` and let the round-trip property cover
masks like every other variant — that is the visible payoff.
