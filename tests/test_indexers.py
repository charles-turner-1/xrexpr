"""Tests for the indexer value type (:mod:`xrexpr.indexers`).

:func:`classify` is the only way to build an indexer from a raw ``isel``/``sel`` value,
so the classification is tested here, in two deliberate styles.

**Examples, at the top.** Which raw shape becomes which variant is a decision, not
something you can derive, so it's written out. Generating it would mean reimplementing
``classify`` in the strategy to know the expected answer, which tests nothing. The other
examples are the cases the properties can't reach: the ``sel`` label path, the rule that
a scalar has no size, and the ``ForwardSlice`` constructor check.

**Properties, at the bottom.** The claims that *can* be derived are checked against
xarray itself, not hand-computed answers:

- ``size`` predicts the real length after ``isel``;
- ``drops_dim`` predicts whether the dim survives;
- ``to_raw`` replays like the value it came from;
- a ``ForwardSlice`` selects a prefix that doesn't depend on the dim's length;
- composing two indexers replays the same as applying them one after the other (the
  same-dim case the optimiser properties in ``test_properties.py`` deliberately leave
  out).

Together these cover every positional size and round-trip branch, so those aren't
repeated as examples.
"""

from typing import get_args

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from xarray.testing import assert_equal

from xrexpr.indexers import (
    Advanced,
    ForwardSlice,
    GeneralSlice,
    Indexer,
    Label,
    Mask,
    Positions,
    Scalar,
    _is_int,
    classify,
)
from xrexpr.ir import Select
from xrexpr.optimize import _compose_indexer

# --- classify: each raw shape lands in exactly one variant --------------------------------


def test_int_scalar_classifies_as_scalar():
    """An integer position is a ``Scalar`` — the one variant that drops its dim."""
    assert classify(0) == Scalar(0)


def test_label_scalar_classifies_as_scalar():
    """A coordinate label is a ``Scalar`` too: it drops its dim just as a position does."""
    assert classify("2020") == Scalar("2020")


def test_forward_slice_classifies_as_forward_slice():
    """A slice with non-negative bounds is a ``ForwardSlice`` — the composable variant."""
    assert classify(slice(0, 5)) == ForwardSlice(0, 5, None)


def test_stepped_forward_slice_is_forward():
    """A positive step keeps a slice forward, so it stays composable."""
    assert classify(slice(0, 10, 2)) == ForwardSlice(0, 10, 2)


def test_negative_bound_slice_classifies_as_general_slice():
    """A negative bound counts from an end the composer cannot locate, so it is general."""
    assert classify(slice(-3, None)) == GeneralSlice(slice(-3, None))


def test_reversed_step_slice_classifies_as_general_slice():
    """A reversed step is sizable but uncomposable, so it is a ``GeneralSlice``."""
    assert classify(slice(None, None, -1)) == GeneralSlice(slice(None, None, -1))


def test_label_slice_classifies_as_label():
    """A slice with non-integer bounds names labels, not positions, so it is a ``Label``."""
    assert classify(slice("a", "z")) == Label(slice("a", "z"))


def test_integer_list_classifies_as_positions():
    """A list of integers is a ``Positions`` enumeration."""
    assert classify([0, 2, 4]) == Positions((0, 2, 4))


def test_integer_array_classifies_as_positions():
    """An integer ndarray classifies exactly as the list spelling does."""
    assert classify(np.array([0, 1])) == Positions((0, 1))


def test_boolean_list_classifies_as_mask():
    """A boolean list is a ``Mask``, not ``Positions`` — the all-bool test runs first."""
    assert isinstance(classify([True, False, True]), Mask)


def test_boolean_array_classifies_as_mask():
    """A boolean ndarray classifies as a ``Mask`` like the list spelling."""
    assert isinstance(classify(np.array([True, False])), Mask)


def test_label_list_classifies_as_label():
    """A list of labels is irreducibly open, so it lands in ``Label``."""
    assert classify(["2020", "2021"]) == Label(["2020", "2021"])


# numpy positions aren't unusual: ``argmin``, ``np.where`` and ``arr.values[i]`` all
# return ``np.int64``. An ``isinstance(x, int)`` test would treat them as labels, which
# gets the dim's size wrong *and* quietly makes the indexer uncomposable.


def test_zero_dim_integer_array_classifies_as_scalar():
    """A 0-d integer array is a ``Scalar``, not a one-element list.

    Notes
    -----
    It indexes exactly like the bare int (xarray drops the dim), so the rank check has
    to run before the dtype checks.
    """
    assert classify(np.array(0)) == Scalar(0)


@pytest.mark.parametrize("raw", [np.int64(0), np.int32(0), np.array(0)])
def test_integer_scalars_normalise_to_plain_int(raw):
    """Every integer spelling becomes a plain ``int`` inside the ``Scalar``.

    Notes
    -----
    The same conversion ``Positions`` and ``ForwardSlice`` do. Two things depend on it:
    an ``np.int64`` selection must compare equal to the ``int`` spelling (plan equality,
    expected plans in tests), and a ``Scalar`` holding an *array* is unhashable (see the
    next test).
    """
    assert classify(raw) == Scalar(0)
    assert type(classify(raw).value) is int


def test_classified_scalars_are_hashable():
    """All three integer spellings hash the same, so the enclosing node stays hashable.

    Notes
    -----
    ``Select`` stores indexers in a frozendict, so an unhashable value quietly makes the
    whole node unhashable. A 0-d array stored as is would do exactly that.
    """
    assert {classify(np.array(0)), classify(np.int64(0)), classify(0)} == {Scalar(0)}


def test_numpy_int_slice_bounds_classify_as_forward_slice():
    """Numpy-typed slice bounds still classify as forward, and narrow to plain ``int``."""
    assert classify(slice(np.int64(0), np.int64(3))) == ForwardSlice(0, 3, None)


def test_numpy_int_list_classifies_as_positions():
    """A list of numpy integers is a ``Positions``, narrowed to plain ``int``."""
    assert classify([np.int64(0), np.int64(2)]) == Positions((0, 2))


def test_numpy_bool_list_still_classifies_as_mask():
    """Accepting numpy integer types doesn't turn boolean masks into positions.

    Notes
    -----
    The flip side of the tests above: ``np.bool_`` is *not* ``numbers.Integral``, and
    Python ``bool`` is explicitly excluded.
    """
    assert isinstance(classify([np.True_, np.False_]), Mask)


# --- Mask: normalised so the node containing it stays comparable --------------------------


def test_mask_normalises_every_spelling_to_one_value():
    """Array, list and numpy-bool-list spellings of one mask give the same value.

    Notes
    -----
    They're the same selection, so they must be the same value. Otherwise whether two
    plans are equal would depend on how the user wrote the mask.
    """
    expected = Mask((True, False, True))
    assert classify(np.array([True, False, True])) == expected
    assert classify([True, False, True]) == expected
    assert classify([np.True_, np.False_, np.True_]) == expected


def test_mask_bearing_node_compares_and_hashes():
    """Two selects built from the same mask compare equal and collapse in a set.

    Notes
    -----
    This is why masks are converted at all. Stored as an array, ``Mask(arr) ==
    Mask(arr)`` returns an *array*, so comparing the ``Select`` nodes would raise
    ``ValueError: truth value
    ... ambiguous`` instead of answering. And plans really are compared:
    ``test_properties.py``'s idempotence property asserts ``optimize(once) == once``.
    """
    mask = np.array([True, False, True])
    nodes = [
        Select(name="isel", indexer=frozendict({"x": classify(mask)})) for _ in range(2)
    ]
    assert nodes[0] == nodes[1]
    assert len({*nodes}) == 1


def test_multidimensional_bool_array_is_not_a_mask():
    """A higher-rank boolean array becomes a ``Label``, not a flattened ``Mask``.

    Notes
    -----
    xarray rejects it as a single-dim indexer, so no valid plan contains one. ``Label``
    is the variant for values we can't reason about, and xarray still raises its own
    error at replay.
    """
    assert isinstance(classify(np.array([[True, False], [True, False]])), Label)


def test_dataarray_indexer_classifies_as_advanced():
    """Any ``DataArray`` indexer is ``Advanced``: ``classify`` doesn't know the dim.

    Notes
    -----
    ``classify`` only sees the value, not the dim it indexes, so it can't tell an
    orthogonal indexer (which the schema layer converts to plain values) from a
    vectorized one. Both end up here. The schema layer makes the orthogonal/vectorized
    split, where the dim is known.
    """
    idx = classify(xr.DataArray([0, 1], dims="time"))
    assert isinstance(idx, Advanced)
    assert idx.dims == ("time",)


def test_vectorized_dataarray_indexer_classifies_as_advanced():
    """A fresh-named ``DataArray`` indexer is ``Advanced``, carrying its new dim."""
    idx = classify(xr.DataArray([0, 1], dims="points"))
    assert isinstance(idx, Advanced)
    assert idx.dims == ("points",)


def test_variable_indexer_classifies_as_advanced():
    """An ``xr.Variable`` indexer is advanced indexing too, not a dim-dropping scalar."""
    assert isinstance(classify(xr.Variable("time", [0, 1])), Advanced)


def test_advanced_indexer_is_hashable():
    """``Advanced`` is hashable where the raw ``DataArray`` isn't, so ``Select`` stays hashable.

    Notes
    -----
    Only the array's dims are stored, not the array, which a ``Scalar`` holding the raw
    ``DataArray`` couldn't offer.
    """
    idx = classify(xr.DataArray([0, 1], dims="time"))
    assert hash(idx) == hash(Advanced(dims=("time",)))


def test_advanced_indexer_is_uncomposable():
    """Two advanced indexers do not compose — its select is a barrier, never composed."""
    idx = classify(xr.DataArray([0, 1], dims="time"))
    assert _compose_indexer(idx, idx) is None


# --- drops_dim: only a scalar removes its dim ---------------------------------------------


def test_exactly_one_variant_drops_its_dim():
    """``Scalar`` is the only variant that drops its dim, and every variant declares ``drops_dim``.

    Notes
    -----
    Checked by walking the ``Indexer`` union rather than variant by variant. Listing
    variants by hand only checks the ones that already exist, so a *new* variant without
    ``drops_dim`` would slip through. Walking the union makes the union itself the
    checklist.
    """
    dropping = [v for v in get_args(Indexer) if getattr(v, "drops_dim", None) is True]
    keeping = [v for v in get_args(Indexer) if getattr(v, "drops_dim", None) is False]

    assert dropping == [Scalar]
    assert len(dropping) + len(keeping) == len(get_args(Indexer))


# --- size: the cases the properties below cannot reach ------------------------------------
#
# Every *positional* size branch (forward and general slices, positions, boolean masks)
# is checked against xarray by ``test_size_predicts_the_real_isel_length``, over far
# more inputs than could be written out here, so those examples aren't repeated. What's
# left is what the properties can't generate: the ``sel`` label path (they only draw
# ``isel`` values) and the scalar rule (a scalar has no size to compare).


def test_label_slice_keeps_current_size():
    """A label slice can't be sized here, so ``size`` leaves the dim's length unchanged.

    Notes
    -----
    Its length depends on coordinate values. ``schema._selected_size`` returns ``None``
    rather than using this fallback (see that function).
    """
    assert classify(slice("a", "z")).size(4) == 4


def test_label_list_sizes_by_length():
    """A label *sequence* sizes exactly, by its own length."""
    assert classify(["2020", "2021"]).size(4) == 2


def test_scalar_size_is_undefined():
    """Asking a ``Scalar`` for a size raises: it drops its dim, so there is none."""
    with pytest.raises(AssertionError):
        Scalar(0).size(4)


# --- to_raw: round-trips to a replayable value --------------------------------------------
#
# Positional values are covered against xarray by
# ``test_to_raw_replays_like_the_value_it_came _from``. Only the label shapes are pinned
# by hand, since the properties draw ``isel`` values.


@pytest.mark.parametrize("raw", ["2020", slice("a", "z"), ["2020", "2021"]])
def test_label_to_raw_round_trips_by_value(raw):
    """Every label shape survives a ``classify``/``to_raw`` round trip unchanged."""
    assert classify(raw).to_raw() == raw


# --- constructor invariant: a ForwardSlice cannot be built backwards ----------------------


def test_forward_slice_rejects_negative_bound():
    """Constructing a ``ForwardSlice`` with a negative bound raises, keeping the invariant."""
    with pytest.raises(ValueError):
        ForwardSlice(-1, 5)


def test_forward_slice_rejects_reversed_step():
    """Constructing a ``ForwardSlice`` with a reversed step raises, keeping the invariant."""
    with pytest.raises(ValueError):
        ForwardSlice(0, 5, -1)


# --- property-based: the model must agree with xarray -------------------------------------
#
# Small arrays applied eagerly make a per-example deadline flaky, and the data is
# generated rather than coming from function-scoped fixtures, so both checks are turned
# off, as in test_properties.py.
_SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _da(n: int) -> xr.DataArray:
    """A length-``n`` array on one dim ``x`` — the ground truth to measure against."""
    return xr.DataArray(np.arange(n), dims="x", coords={"x": np.arange(n)})


def _positions(lo, hi):
    """Integers in ``[lo, hi]``, sometimes numpy-typed rather than plain ``int``.

    Including numpy types matters. ``argmin``, ``np.where`` and ``arr.values[i]`` all
    return ``np.int64``, and a classifier using ``isinstance(x, int)`` would treat those
    as labels, getting the dim's size wrong and quietly making the indexer uncomposable.
    Drawing both spellings puts that kind of bug in reach of every property below, not
    just the hand-written examples.
    """
    return st.integers(lo, hi).flatmap(
        lambda i: st.sampled_from([i, np.int64(i), np.int32(i)])
    )


def _forward_slice(draw, n):
    """Draw a forward slice over ``[0, n]``, with plain-or-numpy bounds.

    Parameters
    ----------
    draw : callable
        Hypothesis' draw function.
    n : int
        The dim length the bounds stay within.

    Returns
    -------
    slice
        A slice whose bounds resolve without knowing the dim length.
    """
    bound = st.one_of(st.none(), _positions(0, n))
    return slice(draw(bound), draw(bound), draw(st.one_of(st.none(), _positions(1, 3))))


@st.composite
def _isel_value(draw, n):
    """A raw ``isel`` value valid against a dim of size ``n``, across every variant shape."""
    kind = draw(st.sampled_from(["scalar", "forward", "general", "positions", "mask"]))
    if kind == "scalar":
        # a 0-d array indexes like the bare int, so it belongs in the scalar arm
        return draw(
            st.one_of(_positions(-n, n - 1), st.integers(-n, n - 1).map(np.array))
        )
    if kind == "forward":
        return _forward_slice(draw, n)
    if (
        kind == "general"
    ):  # a negative bound or reversed step — keeps the dim, uncomposable
        return draw(
            st.sampled_from([slice(-n, None), slice(None, -1), slice(None, None, -1)])
        )
    if kind == "positions":
        return draw(st.lists(_positions(0, n - 1), max_size=n))
    return np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))


@st.composite
def _concrete_outer(draw, n):
    """An ``isel`` value whose selected positions are knowable without the dim length.

    ``Positions`` and ``Mask`` both enumerate their selection outright — one as indices,
    the other as flags — so neither needs to know how long the dim is. That is the exact
    property the composer requires, which is why these two must *always* compose.
    """
    if draw(st.booleans()):
        return draw(st.lists(_positions(0, n - 1), min_size=1, max_size=n))
    return np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))


@st.composite
def _composable_outer(draw, n):
    """An ``isel`` value the composer handles as *outer*: a forward slice, or a concrete one."""
    if draw(st.booleans()):
        return _forward_slice(draw, n)
    return draw(_concrete_outer(n))


@st.composite
def _sized_value(draw):
    """Draw a dim size and one raw ``isel`` value valid against it.

    Returns
    -------
    tuple of (int, object)
        The dim length, and a value of any variant shape that indexes it legally.
    """
    n = draw(st.integers(1, 8))
    return n, draw(_isel_value(n))


@st.composite
def _same_dim_pair(draw):
    """A dim size and two ``isel`` values, the first composable-shaped and dim-keeping."""
    n = draw(st.integers(1, 8))
    outer = draw(_composable_outer(n))
    m = int(_da(n).isel(x=outer).sizes["x"])
    assume(m > 0)  # nothing legal to index into an emptied dim
    return n, outer, draw(_isel_value(m))


@st.composite
def _concrete_pair(draw):
    """Like :func:`_same_dim_pair`, but ``outer`` is one of the two fully concrete shapes."""
    n = draw(st.integers(1, 8))
    outer = draw(_concrete_outer(n))
    m = int(_da(n).isel(x=outer).sizes["x"])
    assume(m > 0)
    return n, outer, draw(_isel_value(m))


@_SETTINGS
@given(_sized_value())
def test_size_predicts_the_real_isel_length(case):
    """``size(n)`` equals the dim length xarray actually produces (or the dim drops)."""
    n, value = case
    result = _da(n).isel(x=value)
    indexer = classify(value)
    if indexer.drops_dim:
        assert "x" not in result.dims
    else:
        assert indexer.size(n) == result.sizes["x"]


@_SETTINGS
@given(_sized_value())
def test_drops_dim_matches_whether_isel_removes_the_dim(case):
    """``drops_dim`` predicts whether the dim actually survives a real ``isel``."""
    n, value = case
    result = _da(n).isel(x=value)
    assert classify(value).drops_dim == ("x" not in result.dims)


@_SETTINGS
@given(_sized_value())
def test_classify_to_raw_round_trips_to_the_same_variant(case):
    """``classify`` is stable across a ``to_raw`` round trip — every variant, masks included."""
    _, value = case
    indexer = classify(value)
    assert classify(indexer.to_raw()) == indexer


@_SETTINGS
@given(_sized_value())
def test_to_raw_replays_like_the_value_it_came_from(case):
    """``to_raw()`` selects the same data as the raw value ``classify`` was given.

    The round trip above checks that the *variant* survives; this checks the data. It's
    what ``optimize.py`` relies on whenever a rewrite rebuilds a node's call from its
    ``indexer``. If ``to_raw`` drifted from its input, every optimised plan would replay
    something slightly different from what the user wrote, and nothing else would
    notice.
    """
    n, value = case
    da = _da(n)
    assert_equal(da.isel(x=classify(value).to_raw()), da.isel(x=value))


@_SETTINGS
@given(st.integers(1, 8), st.data())
def test_forward_slice_selects_a_prefix_independent_of_length(n, data):
    """A ``ForwardSlice`` resolved at length ``n`` is a prefix of the same slice resolved at a longer length.

    This is *why* the variant exists. The composer does arithmetic on a ForwardSlice
    without knowing the dim's length, which is only safe if making the dim longer can
    never change the positions already selected. A negative bound breaks that (it counts
    from the end, so every position moves), and ``_is_forward`` exists to exclude it. So
    this property fails if that check is ever loosened too far.
    """
    indexer = classify(_forward_slice(data.draw, n))
    assume(isinstance(indexer, ForwardSlice))

    at_n = list(range(*indexer.to_raw().indices(n)))
    at_longer = list(range(*indexer.to_raw().indices(n + data.draw(st.integers(1, 5)))))
    assert at_longer[: len(at_n)] == at_n


def _respell(raw, wrap):
    """The same ``isel`` value with every integer rewritten via ``wrap`` (``int`` or ``np.int64``)."""
    if isinstance(raw, slice):
        bounds = (raw.start, raw.stop, raw.step)
        return slice(*(None if b is None else wrap(b) for b in bounds))
    if isinstance(raw, list):
        return [wrap(i) for i in raw]
    if isinstance(raw, np.ndarray) and raw.ndim == 0:
        return wrap(raw.item())
    if _is_int(raw):
        return wrap(raw)
    return raw  # a mask: no integers to respell


@_SETTINGS
@given(_same_dim_pair())
def test_composition_does_not_depend_on_integer_spelling(case):
    """Whether two indexers compose must not depend on how their integers were spelled.

    The replay property below only sees compositions that *happen*: it uses ``assume``
    to discard a ``None``, so an indexer that should have composed but didn't is
    skipped, not failed. A ``np.int64`` scalar that refused to compose (say, because the
    composer checked ``isinstance(i, int)`` instead of asking the value) would go
    unnoticed there. Comparing the two spellings needs no way to know *when* composition
    should succeed, and still catches that kind of bug.
    """
    _, outer, inner = case
    as_int = _compose_indexer(
        classify(_respell(outer, int)), classify(_respell(inner, int))
    )
    as_numpy = _compose_indexer(
        classify(_respell(outer, np.int64)), classify(_respell(inner, np.int64))
    )
    assert as_int == as_numpy


@_SETTINGS
@given(_same_dim_pair())
def test_composed_indexer_replays_like_sequential_isel(case):
    """When two indexers compose, the single result replays identically to applying both.

    This is the arithmetic the optimiser's merge rule relies on, over the same-dim case
    the ``test_properties.py`` suite excludes by construction — so it is the property
    that actually guards :func:`_compose_indexer` end to end.
    """
    n, outer, inner = case
    composed = _compose_indexer(classify(outer), classify(inner))
    assume(composed is not None)
    da = _da(n)
    assert_equal(da.isel(x=composed.to_raw()), da.isel(x=outer).isel(x=inner))


@_SETTINGS
@given(_concrete_pair())
def test_a_concrete_outer_always_composes(case):
    """Here a refusal to compose is a *failure*, not something to ``assume`` away.

    The property above only checks compositions that happen: its ``assume`` discards
    every refusal. So an indexer the optimiser *could* merge but doesn't is invisible to
    it. The plan is still correct, just bigger, and nothing reports it. That gap has
    hidden real bugs before (a numpy-scalar bug, and several missed compositions).

    This closes it where refusing is never justified. ``Positions`` and ``Mask`` both
    list their selection explicitly, so ``outer``'s positions are known without the
    dim's length, which is the one thing the composer doesn't have. If the chain replays
    at all, there's nothing left to prove and no good reason to decline, whatever
    ``inner`` is.
    """
    n, outer, inner = case
    composed = _compose_indexer(classify(outer), classify(inner))

    assert composed is not None, (
        f"refused {classify(outer)} then {classify(inner)}, but a concrete outer leaves "
        "nothing to prove — this is a missed merge, not an uncomposable pair"
    )
    da = _da(n)
    assert_equal(da.isel(x=composed.to_raw()), da.isel(x=outer).isel(x=inner))
