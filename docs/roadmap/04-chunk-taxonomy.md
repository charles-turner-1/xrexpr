# W4 — The chunk-spec value sum type

*(Anchors refreshed 2026-07-29 against `main`; the spec itself is unchanged.)*

**Landed 2026-07-31 (#99), as specified, with the three open questions answered by
measurement rather than choice:**

- **`None` is not `-1`.** On an already-chunked dim, `None` keeps the existing blocks and
  `-1` collapses them into one, so both `NoChange` and `FullDim` exist. The evidence is in
  `chunks.py`'s module docstring and pinned by
  `test_chunks.test_no_change_and_full_dim_are_different_behaviours`.
- **`BlockSeq.to_raw()` returns a tuple**, which xarray accepts — unlike the sequence
  indexers, which must be lists.
- **The catch-all is `OpaqueChunk(value)`**, a barrier, taking the role `Label` plays for
  indexers. It is what makes `classify_chunk` total, which `Rechunk.__post_init__` needs.
  Two kinds of value reach it: ones xarray rejects outright (`0`, `-2`), where the error
  stays xarray's to word at replay, and ones dask tolerates but xarray's API does not offer
  (a whole float, `True`). The second is a **deliberate, conservative behaviour change** —
  such a rechunk used to push and is now a barrier, costing an optimisation and never
  correctness.

**Extended 2026-08-01 (#121), with a fourth measured answer the spec did not anticipate:**

- **The discriminant the policy turns on is *extent-dependence*.** `Auto`, `ByteSize` and
  `BlockSeq` are the specs dask resolves by measuring the array it is handed, so what they
  mean changes when a select changes the data. All three barrier; `SingleSize`, `FullDim`
  and `NoChange` cross. The spec above had `"auto"` crossing on the reasoning that it
  "re-picks against whatever survives" — true, and precisely the problem.

  The narrower rule (refuse only an *emptying* select) was built first, as PR #126, and
  rejected on measurement: dask pins every dim below `limit ** (1 / ndim)` and *recurses*,
  so an emptied dim that was itself `"auto"` re-enters non-`"auto"` and poisons
  `largest_block` anyway — `2-D {lat: auto, lon: auto}` with `lat` emptied raises at a 1 kB
  `array.chunk-size` and not at 128 MiB. Whether a plan crashes therefore depends on
  configuration and rank, neither of which a plan can read, so **no per-dim variant can
  carry the distinction** — which is what rules out a `PerDimAuto`/`PartialAuto` shape. The
  table is in `chunks.py`'s `Auto`.

  Cost, measured over 2000 generated chains: of 179 `(Rechunk, Select)` adjacencies, 46
  auto-sizing, the pre-fix rule pushed 105 and this one pushes 78. Fourteen of those hops
  were legitimate; thirteen were the #121 hazard.

- **Both spellings go through the one taxonomy.** `Rechunk` gained `uniform: ChunkSpec |
  None`, classified from `args[0]` in `__post_init__`, because xarray reads any non-Mapping
  spec as `dict.fromkeys(self.dims, spec)` — so `chunk("auto")` *is* the mapping form, just
  written without the keys. That retired both `isinstance(args[0], ...)` reach-ins in
  `optimize.py`: `chunk((100, 400, 500))` now barriers as a `BlockSeq` and `chunk("auto")`
  as an `Auto`, on the same arms their mapping spellings take. The note below on pushability
  being policy rather than a property still holds — no variant carries a flag; the match
  reads the variants and decides.

  It also fixed a misparse: `_chunk_spec` recorded the dim kwargs of `chunk("auto", lat=5)`,
  which xarray silently ignores.

**Goal:** finish `structural-dispatch-2.md` §3. The indexer half of the value sum type
shipped in #66–#71; the chunk half is still `Any`: `Rechunk.chunks:
frozendict[Hashable, Any]` (`ir.py:241`) and the `isinstance` ladder in
`_pushable_rechunk` (`optimize.py:753`) — the last one in the package. Model it exactly
the way `indexers.py` modelled indexer values.

**Size:** 1–2 PRs (taxonomy module + tests first; wiring second — the same split as
the indexer series #66/#67/#68).

## The taxonomy

New module `src/xrexpr/chunks.py`, mirroring `indexers.py`'s structure (module
docstring stating the "sum type in disguise" argument, frozen dataclasses, a sole
`classify_chunk` constructor, `to_raw()` on every variant, a closing union alias):

| variant | raw form | meaning |
|---|---|---|
| `SingleSize(size: int)` | `chunk({dim: 100})` | uniform block size |
| `Auto` | `chunk({dim: "auto"})` | let dask pick |
| `ByteSize(value: str)` | `chunk({dim: "100MB"})` | auto with a byte target |
| `FullDim` | `chunk({dim: -1})` | one block spanning the dim |
| `NoChange` | `chunk({dim: None})` | leave this dim's chunking as-is |
| `BlockSeq(sizes: tuple[int, ...])` | `chunk({dim: (100, 400, 500)})` | explicit block sizes — must sum to the dim length |

```python
ChunkSpec = SingleSize | Auto | ByteSize | FullDim | NoChange | BlockSeq
```

Notes for the implementer:

- **Verify the `None` and `-1` semantics against the pinned xarray/dask before
  coding** (xarray `Dataset.chunk` docs; dask "chunks" docs). If `None` turns out to
  mean the same as `-1` in this codepath, collapse `NoChange` into `FullDim` and say
  so in the module docstring. Do not guess — one variant per *behaviour*, not per
  spelling.
- **Normalise stored values** (the `Mask` lesson, `indexers.py:136`): `BlockSeq`
  stores `tuple[int, ...]` (from list/tuple/ndarray input, `int()`-coercing numpy
  ints), `SingleSize` stores plain `int`. Every variant must keep the enclosing
  `Rechunk` hashable and equality-sane — add the same equality/hash tests
  `test_indexers.py` has.
- **`to_raw()`** returns what xarray accepts: `BlockSeq` returns a `tuple` (xarray
  accepts a tuple of block sizes here — verify with a live call in the tests; if it
  wants a list, return a list, as `Positions.to_raw` does), `Auto` returns `"auto"`,
  `FullDim` returns `-1`, `NoChange` returns `None`, `ByteSize`/`SingleSize` return
  their value.
- **`classify_chunk(value) -> ChunkSpec` is total.** Anything unrecognised (a float, a
  frozenset, ...) goes to a conservative catch-all. Prefer **no open escape variant**:
  if a value doesn't classify, raise `TypeError` at record time *only if* xarray would
  itself reject it — otherwise add the missing variant. If genuinely open values
  exist, add an `OpaqueSpec(value: Any)` variant treated as a barrier and note the
  hashability caveat. Decide from what xarray actually accepts; write the decision in
  the module docstring.
- **Pushability is policy, not a property.** Do *not* put an `is_barrier` flag on the
  variants. The `Rechunk` docstring already states the rule owns that judgement
  (`ir.py:224-240`), matching `indexers.py`'s "composition is policy" stance
  (`indexers.py:19-22`). The taxonomy provides the discriminant; `optimize.py` keeps
  the decision.

## Wiring (second PR)

- **`ir.py`** — `Rechunk.chunks: frozendict[Hashable, ChunkSpec]`; `__post_init__`
  classifies raw values exactly as `Select.__post_init__` classifies indexers
  (`ir.py:154`): `v if isinstance(v, ChunkSpec) else classify_chunk(v)` — note
  `ChunkSpec` is a union, so the `isinstance` needs the variant tuple or
  `get_args(ChunkSpec)`; copy whatever `Select` does for `Indexer`.
- **`schema.py`** — `_chunk_spec` (`schema.py:490`) keeps building the raw mapping;
  classification happens in `Rechunk.__post_init__` so hand-built nodes are normalised
  too (the `Select` precedent, stated in its docstring `ir.py:136-153`).
- **`optimize.py`** —
  - `_pushable_rechunk` (`optimize.py:753`) rewrites its per-value check as a `match`
    over `ChunkSpec` closed with `assert_never`: `BlockSeq` → barrier (`False`),
    every other variant → pushable. This is the policy site, so it gets the
    exhaustiveness treatment `_compose_indexer` got in #71 (`optimize.py:370`):
    a seventh variant must fail mypy here until someone decides which side of the
    line it falls on. Keep the two *args-shape* checks (`optimize.py:768-772` — the
    option-kwargs test and the positional-sequence test) as they are: they are about
    the call header, not the value taxonomy. Say so in the docstring, the same
    deliberate-deviation note `indexer-follow-ups.md` §1 made for the inner helpers.
  - The rebuild in `pushdown_selects_past_rechunks` (`optimize.py:735-748`) becomes
    `args=({dim: spec.to_raw() for dim, spec in kept.items()},)`, mirroring the merged
    select rebuild (`optimize.py:315`).

## Tests

- `tests/test_chunks.py`, mirroring `test_indexers.py`: classification goldens for
  every raw form, normalisation (numpy ints, list vs tuple), equality/hash of each
  variant and of a `Rechunk` containing each, `to_raw` round-trips **verified against
  a live `ds.chunk()` call** (needs the dask test dependency, already present for the
  rechunk-rule tests).
- Existing rechunk-pushdown goldens in `test_optimize.py` stay green unchanged — the
  wiring PR is behaviour-preserving.
- **Exhaustiveness proof** (the #71 technique, recorded in `indexer-follow-ups.md`
  §1): temporarily add a seventh `ChunkSpec` variant and confirm `pixi run mypy` fails
  at exactly `_pushable_rechunk`'s match; remove it. Note the check in the PR
  description.

## Acceptance criteria

- No `Any`-typed value remains in the optimiser's reasoning surface: `Rechunk.chunks`
  values are `ChunkSpec`; `_pushable_rechunk` contains no `isinstance` on spec values.
- Behaviour identical before/after wiring (golden tests unchanged).
- `pixi run python -m pytest tests -q`, `pixi run mypy`,
  `pixi run python -m ruff check src tests` all clean.
- `docs/indexer-follow-ups.md` §2 marked DONE with a pointer here.
