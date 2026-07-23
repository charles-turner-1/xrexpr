# Structural dispatch, part 2: the second sum type, the schema layer, and what a Rust port actually costs

## Where part 1 left us

[`structural-dispatch.md`](./structural-dispatch.md) (the "memo") argued for a sum-type IR
and, in **§7**, committed to one: fat variants, `Op = Reduce | Select | Scan | Opaque`, each
self-contained. [`structural-dispatch-discussion.md`](./structural-dispatch-discussion.md)
records how that decision was reached. This memo is the sequel, and it starts from a
different footing: **§7 is no longer a proposal, it is the code.** The IR shipped, grew two
variants, and is now load-bearing across four rewrite rules. So the question the first memo
kept at arm's length — "a Rust port is not a stated goal … the port is a tiebreaker, not the
argument" (§4) — is the one actually on the table now, and it deserves to be looked at
squarely rather than as a rhetorical device.

The finding, up front: **the op-kind sum type was the first of two.** There is a second,
still-implicit sum type sitting one level down — the *values* inside a select's `indexer`
and a rechunk's `chunks`, still typed `Any` and dispatched by scattered `isinstance` ladders
— and it, not the op enum, is what stands between us and a port that is mechanical rather
than a rewrite. Two things the first memo never modelled (the schema subsystem, and the fact
that replay *must* stay Python) decide where the port's seam can fall. This memo maps all
three and stops there. It is a survey, not a plan: no §7-style "do it now."

**TL;DR.** §7 shipped and grew exactly as §7.4 predicted (`Project`, `Rechunk` — kinds, not
methods). Its cost premise ("one dispatch site", §5) is obsolete: there are four-plus, and
the sum type has paid for itself. The next structural-dispatch job is the **indexer/chunk
*value*** sum type (§3 below) — doc 1 §1's "stringly-typed sum type" critique, one level
down. A port must also carry the **schema subsystem** (§4) and accept that replay stays
**Python** (§5), which fixes the port as a Rust *core* with Python *edges*, not a rewrite of
the whole thing. The value sum type (§3) is the precondition that lets that seam fall
cleanly — and it is worth doing in Python first, port or no port.

---

## 1. What shipped, and how it grew

The committed enum was four variants. The one running today is six
(`ir.py:184`):

```python
Op = Reduce | Select | Scan | Project | Rechunk | Opaque
```

`Project` and `Rechunk` are not scope creep — they are §7.4 ("variants are kinds, not
methods") playing out on schedule. Each was earned by *genuinely new structural data the
optimiser must reason about*, which is the only thing §7.4 said should mint a variant:

- **`Project`** (`ir.py:111-139`) carries `variables` — the ordered names a `ds[["tas"]]`
  keeps — because the projection-pushdown rule has to know *which* variables flow on. No
  existing field could hold that.
- **`Rechunk`** (`ir.py:142-165`) carries `chunks` — the mapping-form `{dim: spec}` — because
  the select-past-rechunk rule has to strip dropped dims from the spec, and only *named* dims
  can be stripped.

Neither is a new *method* dressed as a variant: `mean`/`std`/`sum` are still one `Reduce`
told apart by `name`, exactly as promised. The enum grew by two categories over two features,
which is the slow-growth §7.4 argued the shape would have.

The derived-property discipline generalised, too. §2's insight — that `consumes` on a
`Select` should be a `@property` off `indexer`, never a stored field, so a merged node cannot
contradict itself — is now a *pattern*, not a one-off. `Select.consumes` (`ir.py:87-90`) was
joined by `Project.single` (`ir.py:136-139`, "one variable → `DataArray`", read off the key)
and by `Rechunk`'s mapping-form/verbatim-`args` split (`ir.py:142-165`). Every one is the
same move: store the source of truth, derive the rest, make disagreement unrepresentable.

One dispatch subtlety the first memo never anticipated is worth flagging, because it
complicates the tidy "the name settles the kind" story the `OP_TABLE` tells. **`Project` is
the one kind the table cannot decide** (`ir.py:14-16`, `schema.py:254-278`): `__getitem__` is
a projection when its key *names variables* and an `Opaque` when it is a boolean mask or a
dict. Same method name, two kinds, disambiguated by the *shape of the key* at record time.
That is a small foreshadowing of §3 — the shape of a *value*, not a method name, driving a
structural decision — and the table's clean `name → kind` story already has one exception
because of it.

## 2. The premise part 1 planned around is gone

The first memo's entire cost case was built on a single sentence: "there is exactly one
`kind` dispatch site in the entire package" (§5), reinforced by the §5.1 table showing two of
three `OpNode` consumers wanting the *flat* shape, not variants. On that count, the sum type
was ceremony that might never pay off, and the honest recommendation was "not now."

That count is now wrong by a factor of four-plus. Kind dispatch on `Op` variants lives in:

- `pushdown_selects` — `match nodes[i], nodes[i+1]` on `(Reduce, Select)` (`optimize.py:344`);
- `pushdown_projections` — `match crossed` on `Reduce`/`Select` (`optimize.py:407`);
- `pushdown_selects_past_rechunks` — `match` on `(Rechunk, Select)` (`optimize.py:454`);
- `apply_schema` — a full `match node` with `assert_never` over every variant
  (`schema.py:118-135`);
- `merge_adjacent_selects` — `isinstance`-based, via the `_mergeable_select` `TypeGuard`
  (`optimize.py:170-178`).

The §2.1 payoff — exhaustiveness under mypy — is not hypothetical any more either: it is
`assert_never(node)` on the final arm of `apply_schema` (`schema.py:135`), and it is the
reason adding `Project` and `Rechunk` forced a compile-time visit to every rule instead of a
silently-dead branch. The §5 triggers ("a third rule that dispatches on kind"; "a variant
that carries genuinely new data") both fired. **The sum type has paid for itself**, and part
1's "not now" is retired by events, not just by the §7 review that pre-empted it.

That is the bridge to the rest of this memo. The shape is settled and proven in use; the live
question is no longer "was the enum worth it" but "how far does this shape carry toward the
port the first memo kept declining to plan."

## 3. The second, still-implicit sum type: indexer and chunk *values*

Here is the core finding. Part 1 §1 opened by calling `OpNode.kind: str` a *stringly-typed
sum type* — "the variants exist in the designer's head and in a `frozenset` literal, not in
the type." We fixed that for op *kinds*. We did **not** fix it for the *values the ops carry*,
and the same critique applies verbatim, one level down.

A `Select.indexer` is a `frozendict[Hashable, Any]`. That `Any` is a sum type in disguise. A
single indexer value is one of:

- a **scalar** (`isel(time=0)`) — drops the dim;
- a **forward slice** (`isel(time=slice(0, 5))`) — keeps and resizes it;
- a **general slice** with negative or non-integer bounds — keeps it, but needs the dim
  length to reason about;
- a **position sequence** (`isel(time=[0, 2, 4])`) — a concrete enumeration;
- a **boolean mask** (an array of `bool`);
- a **coordinate label / label-slice** (`sel(time="2020")`) — genuinely open, `Any`.

Nothing in the type says so. The variants live, as §1 put it, "in the designer's head and in
scattered `isinstance` checks." And the code is thick with those checks, each re-deriving the
same taxonomy:

- `_is_scalar_index` (`ir.py:37-44`) — "is this value a `slice | list | tuple | ndarray`, or
  else a scalar?" — is the scalar-vs-keeps-dim discriminant, written as a negated
  `isinstance` tuple.
- `_compose_indexer` / `_compose_slice` / `_index_sequence` (`optimize.py:220-313`) are an
  `isinstance` ladder that *is* a match over that taxonomy: array→list, sequence-of-ints,
  slice, else give up. `_is_forward_slice` (`optimize.py:300-308`) carves the "forward,
  non-negative" slice sub-variant out by hand.
- `_indexer_size` (`schema.py:148-170`) re-walks the *same* taxonomy independently — slice,
  boolean array, integer array, boolean sequence, integer sequence — to size a kept dim.
- `_pushable_rechunk` (`optimize.py:480-497`) does the chunk-spec version: "is this spec a
  `list | tuple`?" distinguishes an explicit block sequence (a barrier) from a single size.

Three functions, in three files, each re-deciding "what kind of indexer value is this?" by
`isinstance`. That is precisely the smell the op-kind sum type removed at the node level, and
it is still here at the value level.

### 3.1 What modelling it buys

Make the taxonomy a type:

```python
Indexer = Scalar | ForwardSlice | GeneralSlice | Positions | Mask | Label
```

and the payoff mirrors §2.1 exactly:

- `_compose_indexer` collapses from an `isinstance` ladder to one `match` with `assert_never`,
  and the "we handle forward slices and integer sequences, everything else is uncomposable"
  policy becomes a visible, exhaustive set of arms instead of a fall-through `return
  _UNCOMPOSABLE` (`optimize.py:242`) you have to trust covers the rest.
- The `_UNCOMPOSABLE` sentinel (`optimize.py:217`) — "distinct from `None`, which is itself a
  legitimate slice bound" — is a hand-rolled `Option`. Under the value sum type it becomes a
  returned variant (or `Indexer | None`), and the sentinel-vs-`None` footgun the comment calls
  out disappears.
- `_is_forward_slice`'s carve-out stops being a guard re-run at every call site and becomes a
  *constructor invariant*: a `ForwardSlice` is non-negative because it cannot be built
  otherwise, the same way `Select(name="mean")` is now unconstructable (§2).
- `_is_scalar_index` and `_indexer_size` stop re-deriving the taxonomy and start pattern-
  matching a settled one, so the "which values drop a dim" rule lives in one place instead of
  being re-answered by negation in `ir.py` and by enumeration in `schema.py`.

### 3.2 What doesn't transfer (the §4 honesty beat)

Two limits, stated plainly, because part 1's credibility came from doing this and this memo
should not overclaim.

**The arithmetic is arithmetic.** `_compose_slice` composing `slice(100, 1000)` then
`slice(10, 20)` into `slice(110, 120)` is bound arithmetic; the sum type buys the *dispatch to
that arithmetic*, not the arithmetic itself — exactly as §4 said `match` buys the kind
dispatch but never the dim algebra. Splitting the composition into one arm per value-pair
would duplicate the arithmetic, not clarify it. The win is the discriminant, not the body.

**`sel` labels stay open.** A coordinate label is `Any` by nature — a string, a timestamp, a
tuple MultiIndex key — and no closed enum can hold it. So `Label` is the value sum type's
`Opaque`: an irreducible escape hatch, present for the same reason the op enum keeps `Opaque`.
This is why `_compose_into` refuses `sel` composition outright ("`sel` composition needs
coordinate values; positions only", `optimize.py:200`) — the label variant is the one the
optimiser can't reason about structurally, and the model should say so rather than pretend
otherwise.

Net: the value sum type is a real §2-grade win at the two composition/sizing sites and a wash
(or mild loss) at the label boundary — the same balanced trade §5.1 drew for the op enum, one
level down.

## 4. The schema subsystem part 1 never mentioned

Part 1's §3 Rust-mapping table is a tidy five rows: enum, `Vec`, `BTreeSet`, `BTreeMap`,
`match`. It maps the *IR*. It does not map the thing the IR now depends on to reason about
variables — because that thing did not exist when the memo was written. `optimize` today takes
a `SchemaState` and threads it (`optimize.py:57`, `Rule` is now `Callable[[Plan, SchemaState],
Plan | None]`, `optimize.py:54`). The schema is a whole second data structure and a subtle
invariant, and both must port *with* the IR or the variable-level rules can't come along:

- **`SchemaState`** (`schema.py:30-90`) — `dims` (name→size), `coords`, and `data_vars`
  (name→its dims). A second record type to port, with its own `apply_schema` transition
  function (`schema.py:93-145`) that is itself a full `match` over every `Op` variant.
- **The trust boundary.** `apply_schema` models `Opaque` as variable-preserving — false for
  `rename`/`drop_vars`/`assign` — so the folded schema is exact only up to the first opaque op.
  `_trusted_prefix` (`optimize.py:99-109`) marks that frontier, and `pushdown_projections`
  confines itself inside it (`optimize.py:398`). This is a *correctness* invariant, not a data
  shape: a naive transliteration that ports `SchemaState` but drops the prefix discipline
  would produce a plausible-looking optimiser that silently mis-reorders projections across a
  `rename`. It is exactly the kind of non-local, by-convention invariant §2 warned about,
  living in the schema layer instead of the node layer.
- **`var_dims`' three answers.** `var_dims` returns `frozenset | None`, and the `None` is
  load-bearing: "*don't know* — a name that isn't a tracked data variable — callers must treat
  as 'no rewrite', never as 'no dims'" (`schema.py:73-85`). That is an `Option` whose `None`
  arm has a mandatory interpretation, and Rust's type system would *sharpen* it (you cannot
  forget to handle `None`) rather than merely translate it — a place the port genuinely
  improves on the Python, like §2.1's exhaustiveness.

The takeaway for §6's ledger: the port surface is materially larger than part 1's §3 table
implied. The IR is the easy half; the schema layer is a second structure plus an invariant,
and it is where a mechanical-looking port is most likely to quietly lose correctness.

## 5. The boundary that decides everything: replay is Python

Part 1 pinned down the record → optimise → replay lifecycle and named its key property:
**replay is a uniform passthrough** — `getattr(ds, node.name)(*node.args, **node.kwargs)`,
which never inspects the semantic fields (`accessor.py:160-168`). The memo used that property
to settle the §7.5 "keep `args`/`kwargs` verbatim" question. It has a second consequence the
memo never drew, and it is the architectural crux of any port:

**Replay calls xarray, and xarray is Python.** So "rewrite the IR in Rust" cannot mean a
standalone Rust program. It means drawing a *seam*:

| stage | who owns it | why |
|---|---|---|
| **record** (`to_opnode`, `schema.py:189`) | Python | needs live xarray introspection (`SchemaState.from_dataset`, `sizes`/`coords`/`dims`) |
| **optimise** (`optimize`, the rules) | **Rust** | pure, `match`-heavy, `Any`-free once §3 lands — the part that benefits most |
| **replay** (`_replay`, `accessor.py:160`) | Python | *is* a sequence of xarray method calls — nothing to port |

The optimiser is the natural — and only worthwhile — thing to move. It is where the `match`
lives, where exhaustiveness pays, and where §3's value sum type turns `Any` into native Rust
types. Record and replay are Python by necessity, not by choice.

The one hard part of that seam is the `Any`-typed `args`/`kwargs`. Every node carries them
verbatim for replay (§7.5), and they are arbitrary Python objects — dicts, `DataArray` masks,
coordinate labels. They cannot become Rust values. **But they don't need to**, and this is the
payoff of §3 landing first: the optimiser *only ever reads the semantic fields* —
`consumes`, `indexer`, `variables`, `chunks` — and never `args`/`kwargs` for anything but
rebuilding a merged node's `args` from its `indexer` (`optimize.py:157-163`). So the seam can
carry `args`/`kwargs` across the FFI boundary as **opaque handles** (`PyObject`) the Rust
optimiser stores and reorders but never inspects, while the semantic fields — once §3 makes
them a closed sum type instead of `frozendict[Hashable, Any]` — cross as **native Rust data**
the optimiser matches on. `sel` labels (§3.2) are the exception that proves the rule: they are
the one semantic value that stays `PyObject`, which is exactly why `sel` composition is already
refused (`optimize.py:200`).

So §3 is not just "the next cleanup." It is the precondition that decides whether the
Rust/Python seam falls on a clean line (semantic data native, replay payload opaque) or a
ragged one (semantic data still `Any`, dragging Python objects into the optimiser's hot path).

## 6. Port-readiness ledger

Part 1 §5.1's three-column honesty, redrawn for the port with everything above:

| maps 1:1 (the easy half) | newly harder than §3's table | genuine blockers / open |
|---|---|---|
| `Op` enum → Rust `enum Op` (§1) | the indexer/chunk **value** sum type (§3) — still `Any`, three `isinstance` ladders | binary-op **tree/DAG** — still deferred (§7.6, discussion §10), still orthogonal, still a *feature* call |
| `frozenset`→`BTreeSet`, `frozendict`→`BTreeMap` | the **schema subsystem** (§4) — a 2nd structure + the trust-boundary invariant | `sel` **label** openness (§3.2) — irreducibly `PyObject` |
| `Plan | None`→`Option`, `_UNCOMPOSABLE`→variant | `Any` **`args`/`kwargs`** at the FFI seam (§5) — opaque handles, only viable once §3 lands | whether a **PyO3 core is worth it at all** for a package this size |
| `match` + `assert_never` → exhaustive `match` (compiler-checked, §2.1) | `var_dims`' `None`-means-*don't-know* (§4) — sharpened, but must be ported deliberately | — |

Two rows deserve the §5-style "is it even worth it" honesty part 1 applied to the sum type
itself. **The tree/DAG question is untouched and still orthogonal**: nothing here reopens it,
and it remains a product-vision call (is xrexpr Polars-for-xarray?), not a code one. And **the
PyO3-core question is real**: the optimiser is a few hundred lines of pure Python that runs
once per `.collect()`, before any array touches memory. Its cost is already negligible against
the xarray/dask work replay does. So the port's argument cannot be *speed* — it would have to
be *correctness* (Rust's exhaustiveness and `Option` sharpening, §2.1/§4) or a *second
consumer* of the IR that wants it in a systems language. That is the same tiebreaker-not-
argument framing §4 held for the port from the start, and nothing above overturns it.

## 7. Where this leaves the port

Exploratory, so this ends where part 1's §5–§6 ended before the §7 review forced a decision:
with a recommendation for the *next step* and an explicit refusal to plan the port itself.

**Do first, independent of Rust: model the indexer/chunk-value sum type in Python (§3).** It
stands on its own merits — it removes the last `Any` from the IR's reasoning surface, collapses
three re-derived `isinstance` taxonomies (`ir.py`, `optimize.py`, `schema.py`) into one checked
`match`, and turns the `_UNCOMPOSABLE` sentinel and the forward-slice carve-out into
type-level guarantees. It is doc 1 §2 done one level down, with the same payoff and the same
honest limit (§3.2). Whether or not a port ever happens, this is the change that makes the IR's
value layer as principled as its node layer already is. And *if* the port happens, it is the
seam (§5) that makes it mechanical rather than a rewrite — so it is worth doing before, not
during.

**Do not commit the port.** The op enum was worth committing (§7) because it was better Python,
better typed, and a 1:1 Rust enum all at once — three arguments, one change. The port has one
argument (correctness sharpening), a negligible speed case (§6), a materially larger surface
than part 1's table implied (§4), and a genuine feature fork still open beneath it (the tree,
§6). Name its trigger the way §5 named the sum type's triggers — a **second consumer of the
IR** that wants a systems language, a **contributor** who will own the Rust, or the
**value-and-schema layers modelled** (§3, §4) to the point where the port really is a
transliteration — and until one lands, leave it a decision, not a plan.

The one-line version, echoing part 1: **the op enum was the first sum type; the indexer values
are the second, and modelling them — not writing Rust — is the next structural move.** Reform
the value type and the port's seam draws itself; reach for Rust first and you inherit the `Any`
you never modelled.
