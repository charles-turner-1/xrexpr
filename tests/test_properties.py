"""Property-based tests for rewrite correctness and schema tracking (issue #61).

The hand-written suites pin the optimiser against examples somebody thought of. These
generate the awkward combinations nobody wrote down: small datasets, short chains of
selects and reductions, and indexers that interact badly.

The generators are deliberately narrow (see ``_calls``), because a chain that is
*invalid* is not interesting here — ``InvalidExpressionError`` is a real failure signal
in this module, never expected noise. Two narrowings are worth naming:

- **No dim is indexed by two selects anywhere in the chain.** On ``main`` a folded run
  uses ``dict.update``, which overrides rather than composes and silently returns wrong
  data. Known bug, fix in flight (PR #56).

  The restriction has to be chain-wide rather than run-local, which is worth stating
  because it is not obvious: the two selects need never be adjacent in the chain the user
  wrote. ``pushdown_selects`` hops a select left past a reduce, so
  ``isel(time=[1]).all(dim=['lat']).isel(time=0)`` becomes a run and folds to
  ``isel(time=0)`` — the wrong timestep, silently. Once #56 lands, drop the
  ``selected_dims`` bookkeeping and these properties become the general check on the
  composition arithmetic. Verified against that branch: widened, the suite fails on
  ``main`` and passes with #56 applied.
- **No ``chunk``/rechunk ops.** Selection/rechunk commutation is property 2 of #61; it
  needs the ``Rechunk`` op and the dask test dependency that arrive with PR #58.

``reduce`` is also excluded: it takes a *function* first, which ``_reduce_dims`` misreads
as a dim spec, so a generated ``.reduce(...)`` would fail for reasons unrelated to the
rewrites under test.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from xarray.testing import assert_equal

import xrexpr  # noqa: F401 -- registers the ``.plan`` accessor
from xrexpr.operations import OP_TABLE
from xrexpr.optimize import optimize
from xrexpr.schema import SchemaState, apply_schema, to_opnode

# Reductions worth generating: every tabulated reduce except ``reduce`` itself.
REDUCE_NAMES = tuple(
    sorted(n for n, s in OP_TABLE.items() if s.kind == "reduce" and n != "reduce")
)

#: Reductions with no identity element, which numpy refuses to apply to an empty axis
#: ("zero-size array to reduction operation fmax which has no identity"). An empty
#: selection is a case worth generating, so these are skipped when one is in play — the
#: chain would raise in eager xarray too, making it uninteresting here.
NO_IDENTITY_REDUCES = frozenset({"max", "min"})

DIM_NAMES = ("time", "lat", "lon")

# xarray op timing is jittery on small arrays, and generation applies ops eagerly, so a
# per-example deadline just produces flakes. Function-scoped fixtures are not used here
# (the dataset is generated), so that health check is irrelevant rather than suppressed.
SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])

# Empty selections are generated on purpose, and reducing an empty (or single-element)
# axis makes numpy warn about degenerate statistics. That is expected here rather than a
# signal, and left unfiltered it would train the reader to skim past warnings. Scoped to
# the exact messages so an unrelated RuntimeWarning still surfaces.
pytestmark = [
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning"),
]


class Call(dict):
    """One recorded call: a method name plus its keyword arguments.

    A ``dict`` subclass so Hypothesis shrinks and prints it readably — a failing example
    reports ``isel(time=0)`` rather than a nest of tuples.
    """

    def __init__(self, name: str, **kwargs: object) -> None:
        super().__init__(kwargs)
        self.name = name

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"{self.name}({args})"


def _apply(obj, calls):
    """Replay ``calls`` against ``obj`` — a real Dataset, or a ``.plan`` proxy."""
    for call in calls:
        obj = getattr(obj, call.name)(**call)
    return obj


def _build_plan(ds, calls):
    """Normalise ``calls`` into a plan the way the recorder would, threading the schema.

    Returns the plan and the *final* schema, so a caller can compare tracked metadata
    against what evaluation actually produced.
    """
    schema = SchemaState.from_dataset(ds)
    plan = []
    for call in calls:
        node = to_opnode(schema, call.name, (), dict(call))
        plan.append(node)
        schema = apply_schema(schema, node)
    return plan, schema


@st.composite
def datasets(draw):
    """A tiny dataset with 2-3 dims, monotonic integer coords and readable values."""
    ndim = draw(st.integers(min_value=2, max_value=3))
    dims = DIM_NAMES[:ndim]
    sizes = [draw(st.integers(min_value=1, max_value=5)) for _ in dims]
    values = np.arange(int(np.prod(sizes)), dtype=float).reshape(sizes)
    return xr.Dataset(
        {"temperature": (dims, values)},
        coords={d: np.arange(s) for d, s in zip(dims, sizes)},
    )


@st.composite
def indexers(draw, obj, dim, name):
    """One indexer for ``dim`` of ``obj``, valid for ``isel`` or ``sel`` respectively.

    The two cannot share a generator: ``isel`` addresses *positions* ``0..size-1``, while
    ``sel`` addresses *coordinate labels*, which stop being ``0..size-1`` as soon as an
    earlier op subsets the dim. Drawing sel indexers from the live coordinate values is
    what keeps a generated chain replayable.

    Negative bounds and reversed slices are excluded on purpose: they count from the end,
    so composing them needs the dim length that the optimiser deliberately does not
    carry. They are the uncomposable cases, and belong with the rule that handles them
    rather than here.
    """
    size = obj.sizes[dim]
    if name == "isel":
        values = list(range(size))
        bounds = st.integers(min_value=0, max_value=size)
    else:
        values = [v.item() for v in obj[dim].values]
        # a label slice is inclusive of both ends, and needs labels that exist
        bounds = st.sampled_from(values) if values else st.none()

    strategies = [
        st.builds(slice, bounds, bounds, st.one_of(st.none(), st.integers(1, 2))),
        st.just([]),  # empty selection
    ]
    if values:
        strategies += [
            st.sampled_from(values),  # scalar: drops the dim
            st.lists(
                st.sampled_from(values), min_size=1, max_size=min(3, size), unique=True
            ).map(sorted),
        ]
    return draw(st.one_of(strategies))


@st.composite
def _calls(draw, ds, max_ops=4):
    """A chain of ops that is legal against ``ds`` by construction.

    Legality is guaranteed by *evaluating as we generate*: each drawn call is applied to
    a running dataset, so the next call sees the real post-op dims and sizes rather than
    a reimplementation of them. The datasets are tiny, so this is cheap and exact.
    """
    calls = []
    current = ds
    selected_dims: set[str] = set()  # every dim any select has already indexed

    for _ in range(draw(st.integers(min_value=0, max_value=max_ops))):
        if not current.sizes:
            break  # everything has been reduced away; nothing legal is left

        kind = draw(st.sampled_from(["isel", "sel", "reduce"]))

        if kind == "reduce":
            dims = draw(
                st.lists(
                    st.sampled_from(sorted(current.sizes)),
                    min_size=1,
                    max_size=len(current.sizes),
                    unique=True,
                ).map(sorted)
            )
            names = REDUCE_NAMES
            if any(current.sizes[d] == 0 for d in dims):
                names = tuple(n for n in names if n not in NO_IDENTITY_REDUCES)
            call = Call(draw(st.sampled_from(names)), dim=dims)
        else:
            # No dim is indexed twice anywhere in the chain — not merely twice in a row.
            # Adjacency is not a property of the chain the user wrote: ``pushdown_selects``
            # hops a select left past a reduce, so two selects with a reduction between
            # them become a run that ``merge_adjacent_selects`` then folds. Restricting
            # only within an existing run leaves that path wide open (see module docstring).
            available = sorted(set(current.sizes) - selected_dims)
            if not available:
                break
            dim = draw(st.sampled_from(available))
            call = Call(kind, **{dim: draw(indexers(current, dim, kind))})
            selected_dims.add(dim)

        current = _apply(current, [call])
        calls.append(call)

    return calls


@st.composite
def plans(draw):
    """A dataset paired with a legal chain of calls against it."""
    ds = draw(datasets())
    return ds, draw(_calls(ds))


@st.composite
def select_runs(draw):
    """A dataset paired with a run of >=2 adjacent same-name selects on distinct dims."""
    ds = draw(datasets())
    name = draw(st.sampled_from(["isel", "sel"]))
    dims = draw(
        st.lists(
            st.sampled_from(sorted(ds.sizes)),
            min_size=2,
            max_size=len(ds.sizes),
            unique=True,
        )
    )
    calls = []
    current = ds
    for dim in dims:
        call = Call(name, **{dim: draw(indexers(current, dim, name))})
        current = _apply(current, [call])
        calls.append(call)
    return ds, calls


@SETTINGS
@given(plans())
def test_optimised_plan_matches_eager_evaluation(case):
    """The headline property: optimising must not change the answer.

    This exercises the whole optimiser without encoding any individual rule, which is
    what makes it worth generating rather than enumerating.
    """
    ds, calls = case
    assert_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@SETTINGS
@given(plans())
def test_optimize_is_idempotent(case):
    """A second pass changes nothing — the fixpoint really is a fixed point.

    Reaching this assertion at all is half the property: ``optimize`` loops until no rule
    fires, so a rule pair that undid each other's work would hang here rather than fail.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    once = optimize(plan)
    assert optimize(once) == once


@SETTINGS
@given(select_runs())
def test_adjacent_selects_collapse_without_changing_meaning(case):
    """A run of same-name selects folds to one node that means the same thing.

    Asserting the fold *fires* keeps this from passing vacuously: an over-cautious merge
    rule that never merged would satisfy the equality half on its own.
    """
    ds, calls = case
    plan, _ = _build_plan(ds, calls)
    optimised = optimize(plan)

    assert len(optimised) == 1, "a run of selects on distinct dims should fold to one"
    assert_equal(_apply(ds.plan, calls).collect(), _apply(ds, calls))


@SETTINGS
@given(plans())
def test_tracked_schema_agrees_with_evaluation(case):
    """Tracked dim *names* and coords must describe what evaluation actually produces.

    Sizes are asserted separately (below), because one case gets them wrong today. What
    holds universally is:

    - **dims**: the same dim names survive. This is what the rewrites actually reason
      about — ``consumes`` and the pushdown conflict check are name-based.
    - **coords**: tracked coords are a *subset* of the result's. A scalar select drops
      the dim but xarray keeps a scalar coordinate behind, which the schema does not
      model.

    Dim *order* is deliberately not asserted: ``Dataset.sizes`` does not promise the
    insertion order the schema threads, so comparing as sets is the honest check.
    """
    ds, calls = case
    _, schema = _build_plan(ds, calls)
    result = _apply(ds, calls)

    assert set(schema.dims) == set(result.sizes)
    assert set(schema.coords) <= set(result.coords)


@SETTINGS
@given(plans())
def test_sizes_are_tracked_exactly_without_label_slices(case):
    """Away from label slices, tracked sizes are exact — not merely conservative.

    Every indexer except a ``sel`` slice is sized from the indexer alone: positions,
    lengths of sequences, boolean counts. This pins that the imprecision below is
    confined to the label-slice case rather than lurking generally.
    """
    ds, calls = case
    assume(not any(_has_label_slice(call) for call in calls))

    _, schema = _build_plan(ds, calls)
    assert dict(schema.dims) == dict(_apply(ds, calls).sizes)


def _has_label_slice(call):
    """Whether ``call`` is a ``sel`` carrying a slice indexer."""
    return call.name == "sel" and any(isinstance(v, slice) for v in call.values())


@pytest.mark.xfail(
    strict=True, reason="integer-labelled sel slices are sized as if positional"
)
def test_sel_label_slice_size_is_tracked_correctly():
    """A ``sel`` label slice over integer coords is sized with *positional* semantics.

    Found by ``test_tracked_schema_agrees_with_evaluation``, which shrank it to a
    one-op chain. ``_indexer_size`` takes the positional branch whenever a slice's bounds
    are all ints, but ``sel`` bounds are *labels* and the slice is inclusive of both ends.
    Its "label slice — needs coords to size" fallback only fires for non-integer bounds
    (datetimes, strings), so integer coords slip straight past it.

    Here labels ``[10, 20, 30, 40]`` and ``sel(lat=slice(20, 30))`` really yield 2
    elements, but the schema records 0 — under-reporting, which is the *unsafe*
    direction. ``_indexer_size``'s docstring argues imprecision is always safe because it
    is conservative; that argument holds only for over-reporting.

    Latent rather than user-visible today: no rewrite consults tracked sizes yet. It
    would stop being latent the moment a size-driven rule (a cost model, an empty-result
    shortcut) lands. Fixing it needs the select's ``name`` at the ``_indexer_size`` call
    site in ``apply_schema``, which is a source change beyond this test-only PR.
    """
    ds = xr.Dataset({"t": ("lat", np.arange(4.0))}, coords={"lat": [10, 20, 30, 40]})
    call = Call("sel", lat=slice(20, 30))

    _, schema = _build_plan(ds, [call])
    assert dict(schema.dims) == dict(_apply(ds, [call]).sizes)
