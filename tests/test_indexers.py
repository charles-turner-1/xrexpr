"""Tests for the indexer *value* sum type (:mod:`xrexpr.indexers`).

:func:`classify` is the sole constructor from raw ``isel``/``sel`` values, so the taxonomy is
pinned here, split deliberately between two styles.

**Examples, at the top.** The taxonomy itself — which raw shape becomes which variant — is a
decision, not a derivable fact, so it is written out. Generating it would mean reimplementing
``classify`` in the strategy in order to know the expected answer, which asserts nothing. The
examples that remain beyond the taxonomy table are the ones the properties structurally cannot
reach: the ``sel`` label path, the scalar-has-no-size contract, and the ``ForwardSlice``
constructor invariant.

**Properties, at the bottom.** The claims that *are* derivable are checked against xarray
itself rather than a hand-computed answer: ``size`` predicts the real post-``isel`` length,
``drops_dim`` predicts whether the dim survives, ``to_raw`` replays like the value it came
from, a ``ForwardSlice`` selects a length-independent prefix, and composing two indexers
replays identically to applying them in sequence (the same-dim case the optimiser property
suite in ``test_properties.py`` deliberately excludes). Between them these cover every
positional size and round-trip branch, so those examples are not duplicated above.
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
    assert classify(0) == Scalar(0)


def test_label_scalar_classifies_as_scalar():
    assert classify("2020") == Scalar("2020")


def test_forward_slice_classifies_as_forward_slice():
    assert classify(slice(0, 5)) == ForwardSlice(0, 5, None)


def test_stepped_forward_slice_is_forward():
    assert classify(slice(0, 10, 2)) == ForwardSlice(0, 10, 2)


def test_negative_bound_slice_classifies_as_general_slice():
    assert classify(slice(-3, None)) == GeneralSlice(slice(-3, None))


def test_reversed_step_slice_classifies_as_general_slice():
    assert classify(slice(None, None, -1)) == GeneralSlice(slice(None, None, -1))


def test_label_slice_classifies_as_label():
    assert classify(slice("a", "z")) == Label(slice("a", "z"))


def test_integer_list_classifies_as_positions():
    assert classify([0, 2, 4]) == Positions((0, 2, 4))


def test_integer_array_classifies_as_positions():
    assert classify(np.array([0, 1])) == Positions((0, 1))


def test_boolean_list_classifies_as_mask():
    assert isinstance(classify([True, False, True]), Mask)


def test_boolean_array_classifies_as_mask():
    assert isinstance(classify(np.array([True, False])), Mask)


def test_label_list_classifies_as_label():
    assert classify(["2020", "2021"]) == Label(["2020", "2021"])


# Numpy-typed positions are not an exotic case: ``argmin``, ``np.where`` and ``arr.values[i]``
# all hand back ``np.int64``, so an ``isinstance(x, int)`` test would misfile them as labels --
# mis-sizing the dim *and* silently making the indexer uncomposable.


def test_zero_dim_integer_array_classifies_as_scalar():
    # indexes exactly like the bare int -- xarray drops the dim -- so it must not be read as a
    # one-element enumeration
    assert classify(np.array(0)) == Scalar(0)


@pytest.mark.parametrize("raw", [np.int64(0), np.int32(0), np.array(0)])
def test_integer_scalars_normalise_to_plain_int(raw):
    # Same normalisation Positions and ForwardSlice apply. Two things depend on it: an
    # np.int64 selection must compare equal to the int spelling (plan equality, golden
    # assertions), and a Scalar holding an *array* is unhashable -- see the test below.
    assert classify(raw) == Scalar(0)
    assert type(classify(raw).value) is int


def test_classified_scalars_are_hashable():
    # Select stores indexers in a frozendict, so an unhashable value silently makes the whole
    # node unhashable. A 0-d array would do exactly that if it were stored verbatim.
    assert {classify(np.array(0)), classify(np.int64(0)), classify(0)} == {Scalar(0)}


def test_numpy_int_slice_bounds_classify_as_forward_slice():
    assert classify(slice(np.int64(0), np.int64(3))) == ForwardSlice(0, 3, None)


def test_numpy_int_list_classifies_as_positions():
    assert classify([np.int64(0), np.int64(2)]) == Positions((0, 2))


def test_numpy_bool_list_still_classifies_as_mask():
    # the counterweight to the three above: ``np.bool_`` is *not* ``numbers.Integral``, and
    # Python ``bool`` is explicitly excluded, so widening to integers must not swallow masks
    assert isinstance(classify([np.True_, np.False_]), Mask)


# --- Mask: normalised so the node containing it stays comparable --------------------------


def test_mask_normalises_every_spelling_to_one_value():
    # array, list and numpy-bool list are the same selection, so they must be the same value --
    # otherwise a plan's identity depends on how the user happened to write the mask
    expected = Mask((True, False, True))
    assert classify(np.array([True, False, True])) == expected
    assert classify([True, False, True]) == expected
    assert classify([np.True_, np.False_, np.True_]) == expected


def test_mask_bearing_node_compares_and_hashes():
    # The reason this variant is normalised at all. Held verbatim, ``Mask(arr) == Mask(arr)``
    # returns an *array*, so the enclosing Select raised ``ValueError: truth value ...
    # ambiguous`` rather than answering -- and plan equality is real: test_properties.py's
    # idempotence property asserts ``optimize(once) == once``.
    mask = np.array([True, False, True])
    nodes = [
        Select(name="isel", indexer=frozendict({"x": classify(mask)})) for _ in range(2)
    ]
    assert nodes[0] == nodes[1]
    assert len({*nodes}) == 1


def test_multidimensional_bool_array_is_not_a_mask():
    # xarray rejects a higher-rank boolean array as a single-dim indexer, so no valid plan
    # contains one -- it is Label (the can't-reason-about-it variant), not a flattened Mask
    assert isinstance(classify(np.array([[True, False], [True, False]])), Label)


# --- drops_dim: only a scalar removes its dim ---------------------------------------------


def test_exactly_one_variant_drops_its_dim():
    # Asserted by reflection over the union rather than variant by variant: enumerating the
    # variants by hand only ever checks the ones that already exist, so a *new* variant added
    # without a ``drops_dim`` would slip through. Walking ``Indexer`` makes the union itself
    # the checklist -- the test fails the moment a variant is added and left undeclared.
    dropping = [v for v in get_args(Indexer) if getattr(v, "drops_dim", None) is True]
    keeping = [v for v in get_args(Indexer) if getattr(v, "drops_dim", None) is False]

    assert dropping == [Scalar]
    assert len(dropping) + len(keeping) == len(get_args(Indexer))


# --- size: the cases the properties below cannot reach ------------------------------------
#
# Every *positional* size branch (forward and general slices, positions, boolean masks) is
# checked against xarray itself by ``test_size_predicts_the_real_isel_length``, over far more
# inputs than could be written out here — so those examples are not repeated. What is left is
# what the properties structurally cannot generate: the ``sel`` label path (they draw ``isel``
# values only) and the scalar contract (a scalar has no size to compare against).


def test_label_slice_keeps_current_size():
    assert classify(slice("a", "z")).size(4) == 4


def test_label_list_sizes_by_length():
    assert classify(["2020", "2021"]).size(4) == 2


def test_scalar_size_is_undefined():
    with pytest.raises(AssertionError):
        Scalar(0).size(4)


# --- to_raw: round-trips to a replayable value --------------------------------------------
#
# Positional values are covered against xarray by ``test_to_raw_replays_like_the_value_it_came
# _from``. Only the label shapes are pinned by hand, since the properties draw ``isel`` values.


@pytest.mark.parametrize("raw", ["2020", slice("a", "z"), ["2020", "2021"]])
def test_label_to_raw_round_trips_by_value(raw):
    assert classify(raw).to_raw() == raw


# --- constructor invariant: a ForwardSlice cannot be built backwards ----------------------


def test_forward_slice_rejects_negative_bound():
    with pytest.raises(ValueError):
        ForwardSlice(-1, 5)


def test_forward_slice_rejects_reversed_step():
    with pytest.raises(ValueError):
        ForwardSlice(0, 5, -1)


# --- property-based: the model must agree with xarray -------------------------------------
#
# Small arrays and eager application make a per-example deadline flaky, and the fixtures are
# generated rather than function-scoped, so both are turned off the way test_properties.py
# does.
_SETTINGS = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _da(n: int) -> xr.DataArray:
    """A length-``n`` array on one dim ``x`` — the ground truth to measure against."""
    return xr.DataArray(np.arange(n), dims="x", coords={"x": np.arange(n)})


def _positions(lo, hi):
    """Integers in ``[lo, hi]``, sometimes numpy-typed rather than plain ``int``.

    Widening past plain ``int`` is not incidental coverage. ``argmin``, ``np.where`` and
    ``arr.values[i]`` all hand back ``np.int64``, and an ``isinstance(x, int)`` classifier
    misfiles those as labels — mis-sizing the dim and quietly making the indexer
    uncomposable. Drawing both spellings is what puts that class of bug in reach of every
    property below rather than only of the hand-written examples above.
    """
    return st.integers(lo, hi).flatmap(
        lambda i: st.sampled_from([i, np.int64(i), np.int32(i)])
    )


def _forward_slice(draw, n):
    """A forward slice over ``[0, n]``, with plain-or-numpy bounds."""
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

    ``Positions`` and ``Mask`` both enumerate their selection outright — one as indices, the
    other as flags — so neither needs to know how long the dim is. That is the exact property
    the composer requires, which is why these two must *always* compose.
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
    """``to_raw()`` selects the same data as the raw value ``classify`` was handed.

    The round trip above is about the *variant* surviving; this is about the data. It is the
    claim ``optimize.py`` leans on whenever a rewrite rebuilds a node's ``args`` from its
    ``indexer`` — if ``to_raw`` drifted from its input, every optimised plan would replay
    something subtly different from what the user wrote, and nothing else would notice.
    """
    n, value = case
    da = _da(n)
    assert_equal(da.isel(x=classify(value).to_raw()), da.isel(x=value))


@_SETTINGS
@given(st.integers(1, 8), st.data())
def test_forward_slice_selects_a_prefix_independent_of_length(n, data):
    """A ``ForwardSlice`` resolved at length ``n`` is a prefix of itself resolved longer.

    This is *why* the variant exists: the composer reasons about a ForwardSlice arithmetically
    without carrying the dim length, which is only sound if growing the dim can never change
    the positions already chosen. A negative bound breaks it (it counts from the end, so every
    position moves), which is exactly what ``_is_forward`` exists to exclude — so this is the
    property that fails if that predicate is ever loosened too far.
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

    The replay property below can only see compositions that *happen*: it ``assume``s away a
    ``None``, so an indexer that should have composed but silently didn't is filtered out
    rather than failed. That blind spot is real — it hid a live bug, where a ``np.int64``
    scalar refused to compose because the composer tested ``isinstance(i, int)`` directly
    instead of asking the value. Comparing the two spellings needs no oracle for *when*
    composition ought to succeed, and still catches exactly that class of regression.
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

    This is the arithmetic the optimiser's merge rule relies on, over the same-dim case the
    ``test_properties.py`` suite excludes by construction — so it is the property that
    actually guards :func:`_compose_indexer` end to end.
    """
    n, outer, inner = case
    composed = _compose_indexer(classify(outer), classify(inner))
    assume(composed is not None)
    da = _da(n)
    assert_equal(da.isel(x=composed.to_raw()), da.isel(x=outer).isel(x=inner))


@_SETTINGS
@given(_concrete_pair())
def test_a_concrete_outer_always_composes(case):
    """A refusal is a *failure* here, not something to ``assume`` away.

    The property above can only check compositions that happen: its ``assume`` discards
    every refusal, so an indexer the optimiser *could* merge but doesn't is invisible to it —
    the plan stays correct, just larger, and nothing says so. That blind spot has now cost
    twice: it hid a numpy-scalar bug, and it hid the three missed compositions this commit
    adds.

    This closes it for the case where refusal is never justified. ``Positions`` and ``Mask``
    both enumerate their selection outright, so ``outer``'s positions are known without the
    dim length — the one thing the composer lacks. Given a chain that replays at all, there
    is therefore nothing left to prove and no honest reason to decline, whatever ``inner``
    turns out to be.
    """
    n, outer, inner = case
    composed = _compose_indexer(classify(outer), classify(inner))

    assert composed is not None, (
        f"refused {classify(outer)} then {classify(inner)}, but a concrete outer leaves "
        "nothing to prove — this is a missed merge, not an uncomposable pair"
    )
    da = _da(n)
    assert_equal(da.isel(x=composed.to_raw()), da.isel(x=outer).isel(x=inner))
