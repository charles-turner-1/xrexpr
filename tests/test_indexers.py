"""Tests for the indexer *value* sum type (:mod:`xrexpr.indexers`).

:func:`classify` is the sole constructor from raw ``isel``/``sel`` values, so the taxonomy
is pinned here: each raw shape lands in exactly one variant, ``drops_dim`` matches the old
scalar/keeps-dim split, ``size`` matches the old ``_indexer_size`` branches, and ``to_raw``
round-trips back to a value replay can hand to xarray.

The example-based cases at the top fix the taxonomy by hand; the property-based section at
the bottom checks the two claims that matter against xarray *itself* rather than against a
hand-computed answer — ``size`` predicts the real post-``isel`` length, and composing two
indexers replays identically to applying them in sequence (the same-dim case the optimiser
property suite in ``test_properties.py`` deliberately excludes).
"""

from typing import get_args

import numpy as np
import pytest
import xarray as xr
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
    classify,
)
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
    assert classify(np.array(0)) == Scalar(np.array(0))


def test_numpy_int_slice_bounds_classify_as_forward_slice():
    assert classify(slice(np.int64(0), np.int64(3))) == ForwardSlice(0, 3, None)


def test_numpy_int_list_classifies_as_positions():
    assert classify([np.int64(0), np.int64(2)]) == Positions((0, 2))


def test_numpy_bool_list_still_classifies_as_mask():
    # the counterweight to the three above: ``np.bool_`` is *not* ``numbers.Integral``, and
    # Python ``bool`` is explicitly excluded, so widening to integers must not swallow masks
    assert isinstance(classify([np.True_, np.False_]), Mask)


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


# --- size: reproduces the old _indexer_size branches --------------------------------------


def test_forward_slice_sizes_via_indices():
    assert classify(slice(0, 2)).size(4) == 2


def test_general_slice_sizes_against_length():
    assert classify(slice(-3, None)).size(4) == 3


def test_positions_sizes_by_length():
    assert classify([0, 2, 4]).size(5) == 3


def test_integer_array_sizes_by_length():
    assert classify(np.array([0, 1])).size(5) == 2


def test_boolean_array_sizes_by_true_count():
    assert classify(np.array([True, False, True, True])).size(4) == 3


def test_boolean_list_sizes_by_true_count():
    assert classify([True, False, True, True]).size(4) == 3


def test_label_slice_keeps_current_size():
    assert classify(slice("a", "z")).size(4) == 4


def test_label_list_sizes_by_length():
    assert classify(["2020", "2021"]).size(4) == 2


def test_scalar_size_is_undefined():
    with pytest.raises(AssertionError):
        Scalar(0).size(4)


# --- to_raw: round-trips to a replayable value --------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [0, "2020", slice(0, 5), slice(-3, None), slice("a", "z"), [0, 2, 4]],
)
def test_to_raw_round_trips_by_value(raw):
    assert classify(raw).to_raw() == raw


def test_mask_to_raw_returns_the_mask():
    mask = np.array([True, False])
    assert classify(mask).to_raw() is mask


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


@st.composite
def _isel_value(draw, n):
    """A raw ``isel`` value valid against a dim of size ``n``, across every variant shape."""
    kind = draw(st.sampled_from(["scalar", "forward", "general", "positions", "mask"]))
    if kind == "scalar":
        return draw(st.integers(-n, n - 1))
    if kind == "forward":
        bound = st.one_of(st.none(), st.integers(0, n))
        return slice(
            draw(bound), draw(bound), draw(st.one_of(st.none(), st.integers(1, 3)))
        )
    if (
        kind == "general"
    ):  # a negative bound or reversed step — keeps the dim, uncomposable
        return draw(
            st.sampled_from([slice(-n, None), slice(None, -1), slice(None, None, -1)])
        )
    if kind == "positions":
        return draw(st.lists(st.integers(0, n - 1), max_size=n))
    return np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)))


@st.composite
def _composable_outer(draw, n):
    """An ``isel`` value the composer actually handles as *outer*: a forward slice or positions."""
    if draw(st.booleans()):
        bound = st.one_of(st.none(), st.integers(0, n))
        return slice(
            draw(bound), draw(bound), draw(st.one_of(st.none(), st.integers(1, 3)))
        )
    return draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=n))


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
    """``classify`` is stable across a ``to_raw`` round trip (masks compare by identity, skip)."""
    _, value = case
    indexer = classify(value)
    assume(not isinstance(indexer, Mask))  # ndarray-backed, so ``==`` is ill-defined
    assert classify(indexer.to_raw()) == indexer


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
