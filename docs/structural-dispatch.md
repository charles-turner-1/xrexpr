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
(`frozenset(d for d, v in indexer.items() if _is_scalar_index(v))`). The flat record
denormalises it into a field, so "a select's `consumes` agrees with its `indexer`" is an
invariant held by convention rather than by construction, and every rule that builds a
select is quietly responsible for it.

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

---

## 6. Recommendation

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
  and a 1:1 Rust enum.

The summary in one line: **`match` is the payoff, not the reform.** Reform the type and
the `match` writes itself; write the `match` first and you've bought the syntax without
the safety.
