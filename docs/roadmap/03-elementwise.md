# W3 — Reintroduce `Elementwise`, with the rules that earn it

**Goal:** the single biggest optimisation-coverage win available. Today every
untabulated op is an `Opaque` barrier, so one `fillna(0)` or `astype("float32")` in a
chain stops every pushdown from reaching the front. Elementwise ops commute with
selects and projections and preserve the schema exactly — modelling them removes the
barrier and extends the trusted prefix in one move.

`structural-dispatch.md` §7.2 dropped `Elementwise` as a phantom kind and said to
reintroduce it "only when a rule dispatches on it" (§5 trigger 1). The rules below are
that trigger.

**Size:** 2–3 PRs — (1) table + variant + record-time classification (no behaviour
change), (2) the two rule changes + goldens, (3) property-suite widening.

## PR 1 — the variant and its guard

### `operations.py`

Add a conservative elementwise set to the tables (`operations.py:30-46`):

```python
_ELEMENTWISE = ("fillna", "astype", "round", "clip", "isnull", "notnull")
```

with `OpSpec("elementwise", False)` rows in `OP_TABLE`. Start conservative; growing the
tuple later is a one-line change. Deliberately excluded for now: `where` (its `cond`
is data), `interp`/`interpolate_na` (dim-aware), `map` (arbitrary callable).

### `ir.py`

```python
@dataclass(frozen=True)
class Elementwise:
    """A per-element op (``fillna``/``astype``/...) — keeps every dim, size and
    variable; commutes with any select or projection. Only minted when its arguments
    are plain values (see ``_elementwise_safe``); data-shaped arguments demote to
    ``Opaque``."""

    name: str  # open set of tabulated elementwise ops → str (kind-safety via OP_TABLE)
    args: tuple[Any, ...] = ()
    kwargs: frozendict[str, Any] = field(default_factory=frozendict)
```

(with the standard `__post_init__` coercions — copy `Scan`'s, `ir.py:120-122`). Add it
to the `Op` union (`ir.py:198`). Adding a variant fails mypy at `apply_schema`'s
`assert_never` (`schema.py:134`) — that is the exhaustiveness discipline working;
follow the errors.

### `schema.py`

- `apply_schema`: add `Elementwise()` to the pass-through arm
  (`schema.py:131`, `case Scan() | Rechunk() | Opaque():`). Unlike `Opaque`, an
  `Elementwise` genuinely *is* dim- and variable-preserving, so this arm is exact for
  it — which is precisely why `_trusted_prefix` (`optimize.py:108`) extends past it
  **automatically** (it stops only at `Opaque`). No change to `_trusted_prefix`
  needed; state this in the PR description.
- `to_opnode` (`schema.py:163`): a `kind == "elementwise"` branch that mints
  `Elementwise` **only when the arguments are safe**, else falls through to `Opaque`:

```python
def _elementwise_safe(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """Whether every argument is a plain value (scalar/str/dtype/None) rather than
    data- or variable-shaped. ``fillna(da)`` fills from another array's values and
    ``fillna({"tas": 0})`` is per-variable — neither commutes with a projection the
    way a plain scalar does, so both demote to Opaque."""
    return all(
        not isinstance(v, (xr.DataArray, xr.Dataset, dict, list, tuple, np.ndarray))
        for v in (*args, *kwargs.values())
    )
```

The shape-of-the-value deciding the kind has precedent: `Project` vs `Opaque` on the
`__getitem__` key (`schema.py:233`, `ir.py:14-16`). Note `clip(min=..., max=...)`
with scalar bounds passes; with `DataArray` bounds it demotes — correct both ways.
(`astype`'s dtype argument: `np.dtype` instances and `type`s must pass the guard —
they do under the blocklist form above; that is why it is a blocklist, not an
allowlist. Say so in the docstring.)

## PR 2 — the rules

### `pushdown_selects_past_elementwise` (new, in `optimize.py`)

The same single-hop shape as `pushdown_selects_past_rechunks` (`optimize.py:431`):

```python
for i in range(len(nodes) - 1):
    match nodes[i], nodes[i + 1]:
        case (Elementwise() as ew, Select() as select):
            swapped = list(nodes)
            swapped[i], swapped[i + 1] = select, ew
            return swapped
return None
```

Always safe, no dim condition: an elementwise op is applied per element independent of
position, so `fillna(0).isel(time=0) == isel(time=0).fillna(0)` for every indexer
shape — and the guard already excluded the argument shapes for which that argument
fails. Never raises. Docstring should carry the correctness argument and the guard
cross-reference.

Register in `_RULES` (`optimize.py:506`) after `pushdown_selects`.

### `pushdown_projections` extension

One arm in the `match crossed` (`optimize.py:413-419`):

```python
case Elementwise():
    needed = frozenset()
```

A projection needs no dims to cross an elementwise op (per-variable, uniform
arguments — the dict-valued `fillna` that would break this is already `Opaque`). The
existing `available is None` guard (`optimize.py:421-422`) still conservatively blocks
unknown variables; with `needed` empty the subset test passes whenever `var_dims`
knows the names, which is the right behaviour.

### Termination

The measure (`optimize.py:78-86`) is `(len(plan), sum of Select and Project indices)`.
Both rules move a `Select`/`Project` strictly left and never lengthen the plan, and no
rule moves an `Elementwise` at all, so the measure still strictly decreases — quote
this in the new rule's docstring, as the existing rules do.

## Tests

- **Golden compositions** (`test_optimize.py`): `mean("lat") → fillna(0) → isel(time=0)`
  optimises to `isel(time=0) → mean("lat") → fillna(0)` — the select hops past
  `fillna` (new rule) and then past `mean` (existing rule), so the golden proves the
  fixpoint composes the two. (The elementwise node itself never moves; only the select
  passes it.)
- **Projection golden:** `mean("time") → astype("float32") → [["tas"]]` ends with the
  projection in front.
- **Guard goldens:** `fillna(some_dataarray)` and `fillna({"tas": 0})` record as
  `Opaque` and nothing reorders around them.
- **Equality vs eager** (`test_accessor.py`): the chains above through
  `.plan...collect()` equal their eager spellings, including with a scalar select
  (`isel(time=0)`) and a mask select crossing `fillna`.
- **Property widening** (PR 3, `test_properties.py`): add the elementwise names to the
  generated call pool (`_calls`) with scalar arguments; the
  optimised-equals-eager property then covers arbitrary interleavings. Check the
  module docstring's narrowings list (`test_properties.py:10-28`) and update it.

## Acceptance criteria

- A chain containing only tabulated ops plus elementwise calls optimises identically
  to the same chain with the elementwise calls removed, modulo the elementwise nodes
  themselves (selects/projections reach the front).
- `Elementwise` with unsafe args is unobservable: such calls are `Opaque` end-to-end.
- Suite, `pixi run mypy`, `pixi run python -m ruff check src tests` clean.
