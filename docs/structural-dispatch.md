# Structural dispatch in the optimiser: `match`, sum types, and a Rust port

## Context

The optimiser's rules are pattern matching in all but name. `pushdown_selects` asks
"is this adjacency a `(reduce, select)`?"; `merge_adjacent_selects` asks "is this a run
of same-name selects?". Both are written as attribute tests and `if` guards. The
suggestion on the table: rewrite them as `match`/`case`, which reads more like the Rust
this may eventually be ported to.

This memo argues that the instinct is right and the target is wrong. **The thing worth
having is the sum type, not the `match` syntax.** `match` over today's IR is cosmetic;
`match` over a properly-modelled IR is load-bearing, and only the latter ports to Rust
as anything other than a transliteration.

It also tries to be honest about what *doesn't* transfer, and about the fact that
today's optimiser is too small to pay for any of this yet.

### The plan lifecycle: record → optimise → replay

Three stages, and the memo leans on the third, so pin it down up front. A chain like
`ds.plan.mean("lat").isel(time=0).collect()` runs:

1. **Record.** Each call on the `.plan` proxy is *not executed* — it is normalised by
   `to_opnode` into an `OpNode` and appended to a list (`accessor.py:_record`). The list
   *is* the plan.
2. **Optimise.** `optimize()` rewrites that list into an equivalent, cheaper list of
   `OpNode`s (the merge / pushdown rules) — this is the only stage that inspects a node's
   `kind`/`consumes`/`indexer`.
3. **Replay.** `_replay` (`accessor.py:145`) walks the optimised list and *actually calls
   the real xarray methods* on the base dataset, one node at a time:

   ```python
   for node in nodes:                                     # essentially the whole of replay
       if node.name == "__getitem__":
           ds = ds[node.args[0]]
       else:
           ds = getattr(ds, node.name)(*node.args, **node.kwargs)
   ```

**Replay is a uniform passthrough.** It never looks at `kind`, `consumes`, or `indexer` —
it re-invokes the recorded xarray call verbatim from `name`/`args`/`kwargs`. That single
property is what the §2 and §5.1 caveats turn on: whatever shape the IR takes, every node
must still hand replay a `name` and the `args`/`kwargs` to call it with.

---

## 1. Why `match` over today's IR is cosmetic

`OpNode` is one flat frozen dataclass with a `kind: str` field constrained at runtime
to `KINDS`. That is a **stringly-typed sum type**: the variants exist in the designer's
head and in a `frozenset` literal, not in the type. Four different xarray ops all record
as the *same* six-field record, differing only in which fields happen to be filled:

```python
ds.mean("lat")     # OpNode(name='mean',   kind='reduce',   consumes={'lat'},         indexer={})
ds.isel(time=0)    # OpNode(name='isel',   kind='select',   consumes={'time'},        indexer={'time': 0})
ds.cumsum("time")  # OpNode(name='cumsum', kind='scan',     consumes=set(),           indexer={})
ds["temp"]         # OpNode(name='__getitem__', kind='opaque', consumes=set(),        indexer={})
```

(These are the actual values `to_opnode` produces; `args`/`kwargs` elided.)

Python's `match` destructures *shape*. Since every node is the same shape, there is no
shape to destructure — a class pattern like `OpNode(kind="reduce")` is an attribute
equality test wearing a costume. Compare, on `_mergeable_select`:

```python
# today — one line
return node.name in _SELECTS and all(k in node.indexer for k in node.kwargs)

# as match — five
match node:
    case OpNode(name="isel" | "sel", indexer=indexer, kwargs=kwargs):
        return all(k in indexer for k in kwargs)
    case _:
        return False
```

The `name in _SELECTS` half becomes a pattern. The `all(...)` half — a relation
*between two fields of the same node* — cannot, because patterns match fields
independently and can't compare them to each other. It stays as ordinary code inside
the arm. Net result: more lines, same logic, one extra indent.

`pushdown_selects` is the best case and still roughly a wash: pair destructuring and
`"isel" | "sel"` genuinely read better than `select_node.name not in _SELECTS`, and the
`continue` guard disappears, but the body gains an indent level and the same number of
lines. Worth doing only alongside §2, not on its own.

### 1.1 One concrete footgun

`_SELECTS` cannot be reused inside a pattern:

```python
case OpNode(name=_SELECTS):   # BUG: matches every node
```

A bare name in a pattern position is a **capture** pattern, not a value test. This
binds `_SELECTS` to `node.name` and matches unconditionally — so the merge rule would
start folding `mean` into `isel`. Only dotted names (`mod._SELECTS`) are value patterns.
The workaround is to inline the literals and lose the shared constant, or to make the
variants real (§2), which removes the need for the constant entirely.

Rust has the same lowercase-binding rule, but `const`s in patterns match by value and
the compiler lints unreachable arms. Python has neither. This footgun is silent here.

---

## 2. The actual proposal: model the variants

The reason `match` reads well in Rust is that it destructures an `enum` whose variants
carry *different data*. Our variants genuinely do carry different data, and today we
paper over it by giving every node every field. Per `to_opnode` (`schema.py:140-159`):

| variant | populated fields | always empty |
|---|---|---|
| `reduce` | `consumes` (from the dims arg) | `indexer` |
| `select` | `indexer`, `kwargs`, `consumes` (**derived** from `indexer`) | — |
| `scan` | — (name only) | `consumes`, `indexer` |
| `elementwise` / `opaque` | — | `consumes`, `indexer` |

Concretely, `ds.mean("lat").isel(time=0)` records today as two nodes of identical shape,
each carrying two fields it never uses:

```python
# today — flat record, empty fields marked ✗
OpNode(name='mean', kind='reduce', consumes={'lat'},  indexer={})          # ✗ indexer
OpNode(name='isel', kind='select', consumes={'time'}, indexer={'time': 0}) #   both used
```

Under §2's variants the same chain carries only what each op actually has:

```python
# proposed — each variant only has its own fields
Reduce(name='mean', consumes={'lat'})
Select(name='isel', indexer={'time': 0})   # consumes={'time'} is now a @property, not stored
```

Two distinct problems show up here, and only the first is the obvious one:

**Empty fields.** `indexer` on a `mean` node, and `consumes` on a `cumsum`, are not
merely unused — they are *unrepresentable states* that every rule must remember not to
read. Nothing stops a future rule from consulting `node.indexer` on a reduce and getting
a plausible-looking empty dict instead of a type error.

**A derived field stored as state.** `consumes` on a select is not independent data —
`schema.py:150` computes it as exactly the scalar-indexed subset of `indexer`
(`frozenset(d for d, v in indexer.items() if _is_scalar_index(v))`). Concretely,
`ds.isel(time=0, lat=slice(0, 5))` records as:

```python
OpNode(name='isel', kind='select',
       indexer={'time': 0, 'lat': slice(0, 5)},   # the two indexed dims
       consumes={'time'})                          # ← just "which of those are scalar"
```

`consumes` here carries no information `indexer` doesn't already have: `time` is in it
because `time=0` is a scalar, `lat` is out because `slice(0, 5)` keeps its dim. Given the
`indexer`, you could recompute `consumes` at any time — the flat record instead
**denormalises** it into a stored field. So "a select's `consumes` agrees with its
`indexer`" is an invariant held by convention rather than by construction, and every rule
that builds a select is quietly responsible for keeping the two in step.

`merge_adjacent_selects` is already carrying that responsibility: it runs two
accumulators down a run, `indexer.update(...)` beside `consumes |= ...`. Those two can
disagree — `indexer.update` keeps the **last** value for a dim while `consumes` **unions**
across all of them. Take `ds.isel(time=0).isel(time=slice(0, 5))`, which records as:

```python
OpNode(name='isel', kind='select', consumes={'time'}, indexer={'time': 0})
OpNode(name='isel', kind='select', consumes=set(),    indexer={'time': slice(0, 5)})  # slice keeps the dim
```

Merging folds `indexer` (last-wins → `{time: slice(0, 5)}`) and unions `consumes`
(`{time} | ∅ → {time}`) independently, producing a node that contradicts itself — its
stored `consumes` says `time` was dropped, its own `indexer` says `time` survives:

```
merged: indexer={'time': slice(0, 5)}  consumes={'time'}
derived from indexer:                  ∅          (slice(0,5) is not a scalar index)
agree: False
```

This is **latent, not a live bug.** That ordering means indexing a dim a previous scalar
select already dropped, which xarray rejects before we could ever record it — so in
practice the invariant holds. But it holds by a *non-local* argument about what xarray
permits upstream, not by anything visible at the merge site. Under §2's `Select` the
question disappears: the merged node holds only `indexer={'time': slice(0, 5)}` and the
`consumes` property reads `∅` off it — one source of truth, and the node cannot describe
itself incorrectly.

```python
@dataclass(frozen=True)
class Reduce:
    name: str
    consumes: frozenset[Hashable]

@dataclass(frozen=True)
class Select:
    name: Literal["isel", "sel"]
    indexer: frozendict[Hashable, Any]
    kwargs: frozendict[str, Any] = frozendict()

    @property
    def consumes(self) -> frozenset[Hashable]:
        return frozenset(d for d, v in self.indexer.items() if _is_scalar_index(v))

@dataclass(frozen=True)
class Scan:
    name: str
    dim: Hashable

Op = Reduce | Select | Scan | Elementwise | Opaque
```

Note that `consumes` stays meaningful on *both* `Reduce` and `Select` — the two variants
share the concept, they just source it differently. That's a structural-typing detail
worth getting right before a port: it's a shared *accessor*, not a shared *field*.

**Caveat: the sketch above is semantic-only — it drops the fields replay needs.** The
accessor replays a plan by calling every node uniformly:
`getattr(ds, node.name)(*node.args, **node.kwargs)` (`accessor.py:145`), and
`_format_node`/`__repr__` read the same three fields. So `name`/`args`/`kwargs` have to
live on *every* variant, not just the semantic ones shown — `Reduce(name, consumes)` as
written **can't replay** `mean("lat")` (it has no `args=('lat',)`), and `__getitem__`
replays through `node.args[0]`. There are two ways out, and choosing between them is the
real first design step — ahead of the enum itself:

- **Keep `args`/`kwargs` on every variant.** Simplest, and replay is untouched — but it
  reintroduces the "every node carries the same replay fields" shape the split was meant
  to escape. The variants then differ only in their *semantic* fields (`consumes` /
  `indexer`), which is still a real gain, just a smaller one than the sketch implies.
- **Reconstruct the call from the semantic fields** (`consumes`/`indexer`) at replay time.
  Drops `args`/`kwargs` for a genuinely minimal variant, but is a much larger change and
  gives up the "keep `args`/`kwargs` verbatim for faithful replay" guarantee `to_opnode`
  documents (`schema.py:133`) — replay stops being a passthrough and starts re-deriving
  xarray calls.

**This is now decided (§7.5): keep `args`/`kwargs` verbatim on every variant.**
Reconstruction is a trap — the semantic fields are a *lossy* summary that omit orthogonal
kwargs (`skipna`/`ddof` on a reduce, `drop`/`method` on a select), and `opaque` ops have
no semantic fields to rebuild from at all. So the committed sketch (§7.2) carries the
replay header on every variant and layers the semantic fields on top.

Now `match` earns its keep, because the arms bind *different fields* — here the
`pushdown_selects` adjacency, binding `consumes` off the reduce and `indexer` off the
select in one line:

```python
# today (optimize.py:131-132):
#   reduce_node, select_node = nodes[i], nodes[i + 1]
#   if reduce_node.kind != "reduce" or select_node.name not in _SELECTS: continue
#   select_dims = frozenset(select_node.indexer)
#   ...
# proposed:
match nodes[i], nodes[i + 1]:
    case Reduce(consumes=consumes) as red, Select(indexer=indexer) as sel:
        # e.g. ds.mean("lat").isel(time=0): consumes={'lat'}, indexer={'time': 0}
        ...   # the disjoint/overlap trichotomy (§4) lives in the arm body
```

`Select` has no `consumes` to misread; `Reduce` has no `indexer`. The `_SELECTS` constant
disappears — "is a select" becomes `isinstance`, checked by the type checker rather than
by a string compare. `KINDS` and the `__post_init__` validation disappear too: an
unknown kind becomes unconstructable rather than a `ValueError` at runtime.

### 2.1 The payoff Python can actually collect: exhaustiveness

This is the part that matters, and it's easy to miss. Rust's `match` is valuable less
for its syntax than because **it won't compile if you forget a variant**. Add a variant,
and the compiler hands you the list of every site to update.

Python's `match` has no exhaustiveness checking at runtime — but this project runs mypy
with `strict = true` and `warn_unreachable = true`, and mypy *does* narrow a tagged
union and *does* check exhaustiveness via `assert_never`:

```python
case _:
    assert_never(node)   # mypy errors here if a variant is unhandled
```

With `Op` as a union of distinct classes, that gives us the Rust property in Python:
adding a `Window` variant fails type-check at every rule that doesn't handle it. With
today's `kind: str`, mypy cannot help — `"reduce"` is a valid `str` and a silently dead
branch.

(`assert_never` is `typing.assert_never` on 3.11+; on our 3.10 floor it needs
`typing_extensions`, which is a new dependency — small, but not free.)

---

## 3. The Rust mapping

The point of §2 is that it makes the port mechanical rather than a redesign:

| Python (proposed) | Rust |
|---|---|
| `Op = Reduce \| Select \| Scan \| ...` | `enum Op { Reduce {..}, Select {..}, Scan {..}, .. }` |
| `list[OpNode]` | `Vec<Op>` |
| `frozenset[Hashable]` | `BTreeSet<Dim>` |
| `frozendict[Hashable, Any]` | `BTreeMap<Dim, Indexer>` |
| `match` + `assert_never` | `match` (exhaustive by construction) |
| `Literal["isel", "sel"]` | its own two-variant enum |

Today's `OpNode` maps to a Rust struct with a `kind: String` and four
always-maybe-empty fields — which is the thing a Rust reviewer would immediately ask
you to rewrite as an enum. Porting §1 (match over a flat record) means porting the
problem.

Two things the mapping table understates, both settled in §7:

- **The union tracks structural *kinds*, not xarray methods.** `mean`/`std`/`sum` are all
  one `Reduce`; the enum stays small and slow-growing (a handful of *categories*), while
  the *methods* live in `OP_TABLE` (`operations.py`). Modelling one variant per xarray
  method would be the anti-pattern — an enum of near-identical arms.
- **`name` must live *inside* each variant, not on a shared header.** A struct with `name`
  outside the enum (`struct OpNode { name: String, op: OpKind }`) lets `name` disagree with
  the variant (`name: "mean"` beside `op: Select`) — and Rust inherits that footgun just as
  Python does; the port doesn't fix it. Putting `name` in the variant (a `SelectName` enum
  in Rust, `Literal["isel","sel"]` in Python) is what makes the mismatch unrepresentable.

### 3.1 A rule-signature finding that falls out of this

`Rule = Callable[[Plan], Plan]` conflates three outcomes that Rust would force apart:

- **changed** → new plan
- **unchanged** → the same plan back
- **invalid** → `pushdown_selects` raises `InvalidExpressionError`

`optimize` currently recovers "changed" by comparing whole plans with `==` each pass,
which is how it detects the fixpoint. In Rust this would naturally be:

```rust
fn rule(plan: &[Op]) -> Result<Option<Vec<Op>>, InvalidExpression>
```

`Option` for changed/unchanged, `Result` for the invalid leg. Making the Python
signature explicit about "changed" (returning `Plan | None`) would drop the per-pass
full-list equality compare *and* pre-shape the port. This is worth doing independently
of the sum type, and is a much smaller change.

---

## 4. What doesn't transfer

Set algebra is not pattern matching, **in either language**. The trichotomy's three legs
are:

- disjoint dims → swap
- intersecting dims → raise
- not a `(reduce, select)` adjacency at all → leave

Only the third is structural. The first two are `isdisjoint` / `&` on the dim sets, and
they land in a guard in Rust exactly as they do in Python:

```rust
(Op::Reduce { consumes, .. }, Op::Select { indexer, .. })
    if indexer.keys().all(|d| !consumes.contains(d)) => { /* swap */ }
```

So `match` buys the **kind dispatch**; it never buys the **dim algebra**. Splitting the
trichotomy into three `case` arms would duplicate the pattern and recompute the
`frozenset` per guard — worse than the current single `if`. The trichotomy should stay
one arm plus an `if`, and the docstring should stop implying it's three cases of the
same kind: two of the three legs are set relations, one is a shape.

Also worth stating plainly: a Rust port is not a stated goal of this project, and
nothing here should be justified by the port alone. The sum type is defensible on its
own terms (§2.1) — the port is a tiebreaker, not the argument.

---

## 5. Cost, and when this is worth it

The honest case against doing this now: **there is exactly one `kind` dispatch site in
the entire package** (`optimize.py:132`). One dispatch site does not need a dispatch
mechanism. The refactor touches `ir.py`, `schema.py`'s `to_opnode`, `operations.py`'s
metadata table, the accessor's replay, and every test with a golden `OpNode` — call it
a day's work and a wide diff, to make one `if` prettier.

It becomes worth it at any of these triggers, whichever comes first:

1. **A third rule that dispatches on kind.** Two is a coincidence; three is a pattern,
   and the first `elementwise` or `scan`-aware rule makes the flat record start hurting.
2. **A variant that carries genuinely new data** — e.g. `groupby` (carries a grouper),
   `window`/`rolling` (carries a size). These cannot be squeezed into
   `consumes`/`indexer` without inventing more empty-field conventions, and that's the
   point where the flat record stops being merely untidy and starts being wrong.
3. **The Rust port becoming real**, since the enum is where the port starts anyway.

Until then the flat record is the right size for the problem.

### 5.1 Not every pass dispatches — the uniform ones are a standing cost

The "one dispatch site" count above understates the case, because it's easy to picture
*all* of the optimiser wanting variants. It doesn't. `OpNode` has three consumers, and
only one of them dispatches on kind:

| consumer | wants | why |
|---|---|---|
| `pushdown_selects` (`optimize.py:132`) | **variants** | binds different fields per kind — the one pass the sum type helps |
| `_replay` / `_format_node` / `__repr__` (`accessor.py`) | **the flat shape** | treat every node identically: `getattr(ds, name)(*args, **kwargs)` — no dispatch at all |
| `apply_schema` (`schema.py:70,75`) | **per-node field access** | reads `.consumes` *and* `.indexer` on every node; under the split, `Scan`/`Opaque` have neither, so it needs a shared `Protocol` with empty defaults or its own `match` |

So the real trade is one cleaner `match` at the dispatch site against *added* ceremony at
the two uniform sites — plus every golden-`OpNode` test moving. Net line count can rise,
not fall. "Strictly clearer" overstates it: the sum type is clearer **at the dispatch
site** and more ceremonious **at the uniform sites**. That balance still tips toward the
sum type once a §5 trigger lands and there's more than one dispatch site to pay for it —
but it is a balance, not a free win, and the replay caveat in §2 is the first thing to
resolve when the time comes.

---

## 6. Recommendation

> **Superseded (2026-07) — see §7.** The structural "not now" below was written on the
> premise that the optimiser stays small. A design review took the opposite premise —
> *design for growth* — which flips trigger 3's calculus and resolves the open questions
> (§7). The three small wins under "Now" are still worth doing; the "Not now" / "at the
> first trigger" framing is now **do it, as a fat-variant sum type**. Kept below for the
> reasoning that led there.

- **Now:** nothing structural. Three small, independent things worth doing on their own
  merits: tighten `Rule` to `Plan | None` (§3.1), which drops a redundant full-plan
  compare per fixpoint pass and moves the design toward the port; fix the docstring's
  "trichotomy" framing (§4); and write down the latent `consumes`/`indexer` invariant in
  `merge_adjacent_selects` (§2) — it is currently load-bearing, unstated, and depends on
  xarray rejecting the bad ordering upstream.
- **Not now:** `match` over the current flat `OpNode` (§1). It's a syntax change wearing
  a design change's clothes, it duplicates or breaks `_SELECTS`, and it ports the
  problem rather than the solution.
- **At the first trigger in §5:** land the sum type (§2) and let `match` + `assert_never`
  fall out of it. That is the change that is simultaneously better Python, better typed,
  and a 1:1 Rust enum — but settle the replay question first (§2's caveat: variants still
  need `name`/`args`/`kwargs`, or replay must reconstruct calls), and expect it to add
  ceremony at the uniform passes even as it cleans up the dispatch site (§5.1).

The summary in one line: **`match` is the payoff, not the reform.** Reform the type and
the `match` writes itself; write the `match` first and you've bought the syntax without
the safety.

---

## 7. Decision: commit the unary sum type (fat variants)

*(2026-07 — supersedes the "not now" of §5–§6.)*

### 7.1 What, and why now

The trigger calculus in §5 assumed the optimiser stays small. The working premise is now
the opposite — **the op surface is expected to grow** — so the shape is worth fixing
*before* the growth, not after. Decision: **land the unary sum type now, as fat variants**,
and let `match` + `assert_never` (§2.1) fall out of it. "Unary" is load-bearing — this
covers the one-input ops; §7.6 defers the rest.

### 7.2 The shape

Each variant is **self-contained**: it carries the replay header (`name`/`args`/`kwargs`,
§7.5) *and* its own semantic fields, so dispatch is a single-level `match node`.

```python
@dataclass(frozen=True)
class Reduce:
    name: str                                   # tabulated reduction; open set → str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = frozendict()
    consumes: frozenset[Hashable] = frozenset()

@dataclass(frozen=True)
class Select:
    name: Literal["isel", "sel"]                # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = frozendict()
    indexer: frozendict[Hashable, Any] = frozendict()

    @property
    def consumes(self) -> frozenset[Hashable]:  # derived, never stored (§2)
        return frozenset(d for d, v in self.indexer.items() if _is_scalar_index(v))

@dataclass(frozen=True)
class Scan:
    name: Literal["cumsum", "cumprod", "diff"]  # closed set → Literal
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = frozendict()

@dataclass(frozen=True)
class Opaque:
    name: str
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = frozendict()

Op = Reduce | Select | Scan | Opaque
```

- **`Literal` where the name set is closed** (`Select`, `Scan`) — mypy then rejects
  `Select(name="mean")`, making a name↔kind mismatch unrepresentable (§7.3). `Reduce` and
  `Opaque` names are open, so they stay `str`; a `Reduce`'s kind-safety comes instead from
  `to_opnode` only ever building one from a tabulated reduction.
- **`consumes` is a `@property` on `Select`, never a field** — one accumulator, so a merged
  node cannot describe itself incorrectly (§2). It stays a stored field on `Reduce`, which
  sources it differently.
- **`Elementwise` is dropped.** `ir.KINDS` lists it but `OP_TABLE` tabulates no elementwise
  op, so `to_opnode` never emits one — a phantom. Reintroduce it only when a rule
  dispatches on it (§7.4).
- **`Scan` stays distinct from `Opaque`** even though neither carries a semantic field
  today: a scan is *known* to keep its dim (order-significant), and tabulating it
  (`operations.py`) is what guarantees a `cumsum` is never misread as a reduce. Its
  semantic field (a scanned `dim`) arrives with the first scan-aware rule.

### 7.3 Why not plain composition, or hybrid

- **Plain composition** — `OpNode(name, args, kwargs, op: Reduce | Select | …)`, semantics
  in `op`. Rejected: `name` on the shared header can contradict the payload (`name="mean"`
  beside `op=Select(...)`), reintroducing the by-convention invariant the sum type exists
  to kill — and it's a *shape* fault, not a Python one, so a literal Rust port
  (`struct { name, op: enum }`) inherits it. Unrepresentability needs `name` *inside* the
  variant, which composition structurally can't do (its header `name` must stay `str` for
  opaque).
- **Hybrid** — composition's header, but `name` pushed into the payload so it can be
  `Literal`-typed. Sound, and it removes fat variants' field repetition — but it splits a
  node's call across `node.op.name` + `node.args`, and the repetition it saves is mild
  because variants are *kinds*, not methods (§7.4): there are only ever a dozen-ish. Keeping
  each op's call data in one place wins.

### 7.4 Variants are kinds, not methods

The enum grows with **structural categories**, not xarray's method count. `mean`/`std`/
`sum`/`median` are all `Reduce`, told apart by `name`. A new *method* is a row in
`OP_TABLE` (`operations.py`, `name → kind`); a new *variant* is earned only by genuinely
new structural data the optimiser must reason about — `groupby` (a grouper), `rolling`/
`window` (a size). One variant per method would be the anti-pattern: an enum of
structurally identical arms no `match` benefits from.

### 7.5 Replay stays a verbatim passthrough

The §2 caveat's open choice resolves to **keep `args`/`kwargs` on every variant**;
reconstruction-from-semantics is a trap:

- **Lossy.** `consumes`/`indexer` are the optimiser's *summary*, not the whole call.
  `mean("lat", skipna=False)` records `consumes={'lat'}` with `skipna` living only in
  `kwargs`; rebuilding `mean(dim={'lat'})` silently drops it. Same for `ddof` on `std`, and
  `drop`/`method` on a select (stripped into `kwargs` at `schema.py:110`).
- **Impossible for `opaque`.** An untabulated op has no semantic fields to rebuild from.

So replay stays the uniform passthrough from the lifecycle definition (Context):
`getattr(ds, node.name)(*node.args, **node.kwargs)`, unchanged. Semantic fields layer *on
top of* the call payload, never *instead of* it.

### 7.6 Deferred, because it's additive: binary ops → tree/DAG

Only **unary** ops (one input — the previous dataset) are in scope now. Binary/n-ary ops —
`merge`, `concat`, `where(cond, other)`, `ds1 + ds2` — are deferred on the finding that
they're **orthogonal** to the variant shape: they *add* a `Join`/`Concat` variant carrying
plan-typed children and promote the container `list → tree/DAG`, but they don't reopen
`Reduce`/`Select`/`Scan`. Three tiers:

1. **unary, constant params** — a list holds perfectly (nearly all of xarray).
2. **binary with an eager operand** — a list still holds it, as an opaque node carrying the
   already-materialised operand.
3. **binary over a *lazy* operand you optimise across** (or a shared lazy prefix) — this,
   and only this, forces a tree/DAG.

Tier 3 is a genuine feature commitment (the Polars-`LazyFrame` shape), not a code detail,
and isn't on the table yet. The one discipline it asks meanwhile: **keep the linearity
assumption named in one place** (`ir.py`'s module docstring already states it) rather than
leaking "my input is the previous element" into every rule — so the eventual promotion
stays the additive change it should be.
