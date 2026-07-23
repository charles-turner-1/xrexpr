"""Tests for the indexer *value* sum type (:mod:`xrexpr.indexers`).

:func:`classify` is the sole constructor from raw ``isel``/``sel`` values, so the taxonomy
is pinned here: each raw shape lands in exactly one variant, ``drops_dim`` matches the old
scalar/keeps-dim split, ``size`` matches the old ``_indexer_size`` branches, and ``to_raw``
round-trips back to a value replay can hand to xarray. Composition is *not* tested here — it
is an ``optimize.py`` policy over these variants, not an intrinsic fact of a value.
"""

from typing import get_args

import numpy as np
import pytest

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
