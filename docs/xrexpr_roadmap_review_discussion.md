# Roadmap Review Conversation (Claude Fable 5, 2026-07-26)

> **Context:** A reconstruction of the review session that produced commit `c1dc3cd`
> ("docs: review the lowering roadmap, pin the fusion input, renumber to reading
> order"). The lowering re-fold (then `08-lowering.md`, now `02-lowering.md`) had been
> drafted in discussions with ChatGPT and Claude Opus (recorded in
> `xrexpr_ir_discussion{,_expanded}.md`); this session was an adversarial review of it
> by Claude Fable 5 running in Claude Code, followed by the fixes the review produced.
> The sibling discussion docs are terse summaries; this one keeps the actual exchange,
> including the empirical checks, because several of them overturned claims.

---

## Round 1 — the review request

**User:** read everything on this branch that isn't on `main`, then the diff of the
last commit, and give an opinion on the lowering re-fold — noting it came out of
discussions with ChatGPT and Opus.

**Reviewer's method:** read all ten roadmap docs plus the two discussion docs, then
re-verified every load-bearing claim against the source
(`schema.py`/`accessor.py`/`optimize.py`/`ir.py`/`operations.py`) and against the
pinned xarray (2026.7.0) by running code, rather than taking the memos' word.

### What verified clean

- `DatasetWeighted.mean("time")` returns a Dataset with `time` gone — 05's (old
  numbering) "composes with a reduce" was indeed wrong; the reversal in the lowering
  memo is correct.
- Rolling keeps its dim (6→6); coarsen divides it (6→2). "Rolling is simpler than
  groupby" holds.
- `xr.DataArray.__hash__ is None`; `hash(w)` raises `TypeError` — the `ir.py`
  hashability invariant really is untrue today.
- All eight `_CONTEXT_METHODS` names exist on `Dataset`; `DatasetGroupBy.first`
  exists while `Dataset.first` does not.
- Every checked code reference (`to_opnode` at `schema.py:163`, `_reduce_dims`'
  bare case, the double schema fold, the mis-tabulated `.reduce`) was accurate.
- "No optimiser rule consults sizes today" (the schema-sizes memo's load-bearing
  claim) — confirmed by grep.

### The verdict and the pushbacks

Overall verdict: the lowering re-fold is right — the diagnosis (`to_opnode` is
per-call; builders are two-call spellings of one op) is exact, the flat nodes beat
`Contextual(inner: Op)`, and section 13's refutation-conditions practice is worth
keeping. Five pushbacks:

1. **W1's "silent" urgency claim is empirically wrong.** Tested:
   `ds.plan.groupby("time.month").mean().isel(month=0).collect()` reorders (visible
   in `explain()`) but replay raises
   `AttributeError: 'DatasetGroupBy' object has no attribute 'isel'` — loud, not
   silent. No builder object exposes `isel`/`sel`, and projection pushdown is fenced
   by `_trusted_prefix`, so a silently-wrong replay appears unreachable today. The
   genuinely silent member of the bug family is the stale-rename case
   (small-wins §4): `ds.plan.rename(time="t").mean().isel(t=0)` — eager raises
   `ValueError`, optimised **silently returns an empty Dataset** (reproduced). That
   one is fixed by `AllDims` (lowering PR 1), not by W1. W1 stays first on cost.
2. **The fusion input pattern was never pinned** — what nodes does the recorder hand
   `to_lower_ir` for a builder chain? (Expanded in round 2.)
3. **`group_dim: Hashable` assumes string groupers** — `ds.groupby(da)` has no
   single group dim and trips the unhashability wart; scope v1 to string groupers.
4. **Grouped arithmetic is invisible** — `ds.groupby("time.month") - clim`, the
   anomaly idiom, is a binary op on the builder, unrecordable by the proxy,
   inexpressible by any flat node; it deserved a named sentence in the tree deferral
   (and is a second argument against `Contextual`).
5. **The §11.2 boundary-semantics flag can be discharged** — a select on dims
   disjoint from the window dims (the only case the rule moves) cannot change window
   membership along the windowed dim.

Plus positions on section 13's three explicit questions: agree with the `Contextual`
rejection; the §7.4 "widening" is better read as *subsumption* (the discriminant
"hidden data dependency" is itself structural data whose reasoning output is *don't
fire*); and the refactor is worth paying because W1's barrier is trailing and
permanent — it forfeits optimisation for everything downstream of the first builder
call, and groupby/resample head most climate chains.

One nit later withdrawn: the "~2,300-line test suite" figure is correct (tests
2,323 lines, source 2,276 — the reviewer's first count had lumped them together).

---

## Round 2 — explaining pushback 2; first round of edits

**User:** happy with everything except wanted pushback 2 (fusion input never pinned)
explained in detail; otherwise "update the plan to your satisfaction."

**The explanation:** `to_lower_ir`'s job is a pattern match, and the memo specified
its *output* (`GroupedReduce` etc.) but never its *input*. Two passages implied
different shapes: §6's poisoning example showed the closing `.mean()` recording as a
`Reduce` (i.e. through `to_opnode`), while §9's "the recorder still needs
`_CONTEXT_METHODS`" implied the W1 demotion (closer records `Opaque`). PR 3's matcher
is `(Opaque, Reduce)` in one world and `(Opaque, Opaque)` in the other — different
code, helpers, tests. Under shape (a) the fluent IR contains a pseudo-`Reduce` (a
GroupBy method wearing a Dataset costume) and the kind assignment hangs on
name-coincidence with `OP_TABLE`, guarded by nothing; closers are heterogeneous
either way (`mean` tabulated, `first` not).

**Edits made in this round** (the reviewer initially pinned shape (b), closer stays
`Opaque` — later superseded, see round 3): corrected the silence claims in 00/01,
re-annotated §6, discharged §11.2, added the grouped-arithmetic paragraph to §10,
added the subsumption note to §13.2 and the undersold-cost note to §13.3, fixed the
`emit` round-trip type nit, and scoped v1 groupers to strings.

---

## Round 3 — the user's counter-proposal: a typed opener

**User:** can't this be resolved more simply? Ops are directed, so give the groupby a
*type* — `groupby` becomes a `GroupBy` node, and `[GroupBy, Reduce("mean", ...)]` is
a composition fusion rule at the second IR. And stuff in the middle (a `__getitem__`)
can be ignored by lowering and the pair still fused.

**Outcome: adopted, with two corrections — both established by running xarray.**

*Adopted:* opening a context is per-call decidable (`groupby(...)` is an opener no
matter what follows), so `to_opnode` can mint a typed `ContextOpen` node without
violating its per-call constraint — it was only the *closer* it could never type, and
it no longer needs to. One opener variant, not five (openers share structure; the
differentiation belongs to the fused nodes). The payoff is type-level: the IR levels
become two union aliases over shared dataclasses —

    FluentOp  = Op | ContextOpen
    LoweredOp = Op | GroupedReduce | WindowedReduce | WeightedReduce

— so with `optimize` taking `list[LoweredOp]`, a surviving opener is a **mypy
error**, not a convention. The closer records through `to_opnode` *provisionally*
(bare grouped `mean()` → `Reduce(consumes=AllDims)`); fusion treats it as a parse,
never as Dataset semantics, and consumes the pair before `AllDims` resolution reads
it. A **mandatory fallback** demotes any unconsumed `ContextOpen` plus its follower
to `Opaque` verbatim (exactly W1's shape), so a mistyped/unknown closer degrades to
*unfused, never wrong* — which also lets PR 3 type every opener while only adding the
groupby rule (rolling/weighted pairs keep W1 behaviour until PRs 4–5).

*Correction 1 — the "middle" doesn't exist (but "ignored" would have been wrong):*
on xarray 2026.7.0, `DatasetGroupBy.__getitem__` selects a **group**, not a variable
— `gb["tas"]` raises `KeyError`; `gb[1]` returns the January Dataset, *closing* the
context — and Rolling/Coarsen/Weighted raise `TypeError` on `__getitem__`. So builder
contexts are always adjacent pairs and the in-context test is one `isinstance` on
`_ops[-1]`. Had a middle variable-selection existed, lowering could not have ignored
it (it changes `data_vars` and the result type); it could only *hoist* it as a
`Project` under a legality condition — recorded as future work. Trap surfaced:
an in-context `__getitem__` has a hashable key and would classify as `Project`;
it must record `Opaque` (it is group selection).

*Correction 2 — §5.1 had a real semantic error:*
`ds.groupby("time.month").mean("lat")` returns dims `{time: 12}` — `lat` consumed,
`time` kept, **no `month` minted**: a per-group *map* reassembled along the original
dim, not an aggregation. Only a bare spec or one naming the group dim mints the new
dim (verified for `mean()`, `mean("time")`, `mean(["time","lat"])`). The fusion rule
gained a guard: fuse the aggregation case only; the map case falls to the fallback in
v1, with a note that it could later canonicalise to a plain `Reduce`.

Edits: §5.5 rewritten around `ContextOpen`; §5.1 corrected; the §3 pipeline diagram
now shows the two aliases; §6's annotation and the PR 3 blurb updated; 01's revision
note rewritten.

---

## Round 4 — "are you personally happy with the plan now (yes or no)?"

**Yes.** Remaining opens (coarsen's `boundary` size effect, the `explain()` UX call,
the deferred weighted pushdown) are flagged as open rather than hidden; nothing
verified is misstated; the riskiest seam fails safe via the fallback. Recommendation:
start W1.

---

## Round 5 — renumbering to reading order

**User:** humans read linearly; the lowering phase needs to go first(ish) but was
file 8 — check the ordering.

The numbering was a historical accident (00–07 in the first commit; the keystone
arrived later as 08), so a linear read hit four workstreams whose revision notes all
pointed forward at a memo not yet read, plus the superseded 05 mid-sequence — the
dead `Contextual` design *before* the memo that killed it. Since everything was
unmerged on one branch, renumbering was at its cheapest.

Files and W-labels moved together (a half-renumber would be worse than none),
applied as one simultaneous pass over filenames, `W<n>` tokens and backticked bare
refs — simultaneous because the mapping is a permutation:

    02-chunk-taxonomy   -> 04        06-small-wins    -> 07
    03-elementwise      -> 05        07-rust-gate     -> 08
    04-scan-dims        -> 06        08-lowering      -> 02
    05-grouped-contexts -> 09        09-schema-sizes  -> 03

Result: 01 = phase 0, 02 = the phase-1 keystone, 03 = phase 2, 04–07 = phase 3
independents, 08 = gated, 09 = superseded appendix. 00 now states the convention.
The one external referrer (`indexer-follow-ups.md`) was updated; renames went
through `git mv` so history follows them.

---

## Round 6 — commit and push

Everything committed as `c1dc3cd` on `docs-roadmap` and pushed, with a long message
documenting each verified correction, the `ContextOpen` design and its rationale, the
smaller resolutions, the withdrawn nit, and the renumber mapping. The untracked
`ocean-3d-temp-1-monthly-mean-*.nc` and `scratch/` were deliberately left out.

---

## Appendix: the empirical results that did the work

All on xarray 2026.7.0, monthly `time` (12) × `lat` (2) test Datasets:

| check | result |
|---|---|
| `plan.groupby(...).mean().isel(month=0).collect()` | reorders, then `AttributeError: 'DatasetGroupBy' object has no attribute 'isel'` |
| `plan.rename(time="t").mean().isel(t=0)` | eager raises `ValueError`; optimised **silently** returns `Dataset {}` |
| `weighted(w).mean("time")` | `Dataset`, `time` gone (closes after one call) |
| `rolling(time=3).mean()` / `coarsen(time=3).mean()` | `{time: 6}` kept / `{time: 2}` divided |
| `groupby(...).mean()` / `.mean("time")` | `{month: 12, lat: 2}` — aggregation |
| `groupby(...).mean("lat")` | `{time: 12}` — per-group **map**; no `month` |
| `groupby(...).mean(["time","lat"])` | `{month: 12}` — aggregation |
| `gb["tas"]` | `KeyError` — `__getitem__` selects **groups** now |
| `gb[1]` | the matching Dataset subset (context closes) |
| `rolling/coarsen/weighted.__getitem__` | `TypeError` |
| `xr.DataArray.__hash__` | `None`; `hash(da)` raises |
| `Dataset.first` / `DatasetGroupBy.first` | absent / present |

The session's one-line lesson, to sit beside the discussion docs' "translate into a
small semantic language, then optimise that": **verify the semantic language against
the library before optimising it** — three of the claims that shaped the design
(silence, `__getitem__`, the grouped map case) were wrong in ways only running the
code revealed.
