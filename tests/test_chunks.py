"""Tests for the chunk *value* sum type (:mod:`xrexpr.chunks`).

:func:`classify_chunk` is the sole constructor from raw ``chunk`` specs, so the taxonomy is
pinned here, split the same way ``test_indexers.py`` splits its own.

**Examples, at the top.** Which raw spelling becomes which variant is a decision, not a
derivable fact, so it is written out; generating it would mean reimplementing
``classify_chunk`` in the strategy in order to know the expected answer. The constructor
invariants (``SingleSize`` below 1, an empty ``BlockSeq``) and the hashability of a
``Rechunk`` carrying each variant live here too.

**Live checks against xarray, at the bottom.** The claims that are *about xarray* are
asked of xarray rather than asserted from memory: that every variant's ``to_raw`` replays,
and — the one that decides the shape of the taxonomy — that ``None`` and ``-1`` really are
two behaviours rather than two spellings of one. That pair is why :class:`NoChange` and
:class:`FullDim` both exist, so it is pinned rather than left in a module docstring.
"""

import importlib.util
from typing import get_args

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.chunks import (
    Auto,
    BlockSeq,
    ByteSize,
    ChunkSpec,
    FullDim,
    NoChange,
    OpaqueChunk,
    SingleSize,
    _is_int,
    classify_chunk,
)
from xrexpr.ir import Rechunk

#: xrexpr itself never needs dask, but replaying a ``chunk()`` call does -- without a chunk
#: manager xarray raises rather than chunking, so the live checks below are skipped.
requires_dask = pytest.mark.skipif(
    importlib.util.find_spec("dask") is None, reason="dask is not installed"
)


# --- classify_chunk: each raw spelling lands in exactly one variant ------------------------


def test_int_classifies_as_single_size():
    """A positive integer is a ``SingleSize`` — uniform blocks of that length."""
    assert classify_chunk(100) == SingleSize(100)


def test_minus_one_classifies_as_full_dim():
    """``-1`` is a ``FullDim``, tested before the size case so it never becomes a length."""
    assert classify_chunk(-1) == FullDim()


def test_none_classifies_as_no_change():
    """``None`` is a ``NoChange``: it names the dim without asking anything of it."""
    assert classify_chunk(None) == NoChange()


def test_auto_classifies_as_auto():
    """``"auto"`` is the one string that means "dask picks", so it is its own variant."""
    assert classify_chunk("auto") == Auto()


@pytest.mark.parametrize("raw", ["100MB", "10MiB", "1GB"])
def test_byte_target_strings_classify_as_byte_size(raw):
    """Any other string is a byte target, carried verbatim for dask to parse at replay."""
    assert classify_chunk(raw) == ByteSize(raw)


def test_tuple_of_lengths_classifies_as_block_seq():
    """A tuple of integers is an explicit ``BlockSeq``."""
    assert classify_chunk((100, 400, 500)) == BlockSeq((100, 400, 500))


def test_list_of_lengths_classifies_as_block_seq():
    """The list spelling of a block sequence classifies exactly as the tuple one does."""
    assert classify_chunk([100, 400, 500]) == BlockSeq((100, 400, 500))


@pytest.mark.parametrize("raw", [0, -2, True, 100.0, {100}, {"a": 1}])
def test_unmodelled_specs_classify_as_opaque(raw):
    """Anything outside the taxonomy is an ``OpaqueChunk``, which keeps classification total.

    Notes
    -----
    Two kinds arrive here and neither is modelled. ``0`` and ``-2`` are specs xarray
    rejects, so the plan cannot replay either way and the error stays xarray's to word;
    ``True`` and ``100.0`` are ones dask happens to accept but xarray's API does not offer,
    where declining to model costs an optimisation and never correctness.
    """
    assert classify_chunk(raw) == OpaqueChunk(raw)


def test_mixed_sequence_classifies_as_opaque():
    """A sequence that is not all integers is not a block sequence, so it is opaque."""
    assert classify_chunk((100, "auto")) == OpaqueChunk((100, "auto"))


def test_empty_sequence_classifies_as_opaque():
    """An empty sequence is opaque rather than an empty ``BlockSeq``, which dask rejects."""
    assert classify_chunk(()) == OpaqueChunk(())


# Numpy-typed block lengths are not exotic: a dim length read off ``ds.sizes`` arithmetic
# arrives as ``np.int64``, and an ``isinstance(x, int)`` test would file it as opaque --
# silently turning a perfectly ordinary rechunk into a barrier.


@pytest.mark.parametrize("raw", [np.int64(100), np.int32(100)])
def test_numpy_sizes_normalise_to_plain_int(raw):
    """Every integer spelling narrows to a plain ``int`` inside the ``SingleSize``."""
    assert classify_chunk(raw) == SingleSize(100)
    assert type(classify_chunk(raw).size) is int


def test_numpy_block_lengths_normalise_to_plain_ints():
    """A block sequence of numpy integers stores plain ``int``s, so it compares equal."""
    spec = classify_chunk((np.int64(100), np.int64(900)))
    assert spec == BlockSeq((100, 900))
    assert all(type(size) is int for size in spec.sizes)


def test_numpy_minus_one_is_still_a_full_dim():
    """The ``-1`` sentinel is recognised in its numpy spelling too."""
    assert classify_chunk(np.int64(-1)) == FullDim()


def test_bool_is_not_an_integer_size():
    """``True`` is an ``int`` subclass but means a mistake, not blocks of 1."""
    assert not _is_int(True)


# --- constructor invariants ---------------------------------------------------------------


@pytest.mark.parametrize("size", [0, -1, -100])
def test_single_size_rejects_a_non_positive_length(size):
    """``SingleSize`` cannot be built with a length dask would refuse to divide by.

    Notes
    -----
    An invariant rather than a downstream guard, the ``ForwardSlice`` precedent: it is what
    lets every other site treat a ``SingleSize`` as a usable block length without re-testing
    it. ``-1`` is excluded here *because* it means something else — see ``FullDim``.
    """
    with pytest.raises(ValueError, match="must be >= 1"):
        SingleSize(size)


def test_block_seq_rejects_an_empty_sequence():
    """``BlockSeq`` cannot be empty: dask rejects an empty tuple of blocks outright."""
    with pytest.raises(ValueError, match="at least one block"):
        BlockSeq(())


# --- equality and hashing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [100, -1, None, "auto", "100MB", (100, 900)],
    ids=["single", "full", "none", "auto", "bytes", "blocks"],
)
def test_a_rechunk_bearing_each_variant_compares_and_hashes(raw):
    """A ``Rechunk`` holding any modelled spec is still equal to its twin, and hashable.

    Notes
    -----
    The ``Mask`` lesson from ``test_indexers.py``, and the reason ``BlockSeq`` stores a
    tuple: plan equality is a real operation (the idempotence property in
    ``test_properties.py`` asserts it), so a variant that made the enclosing node compare
    or hash wrongly would be a landmine rather than a curiosity.
    """
    spec = classify_chunk(raw)
    left = Rechunk(name="chunk", chunks=frozendict({"time": spec}))
    right = Rechunk(name="chunk", chunks=frozendict({"time": classify_chunk(raw)}))
    assert left == right
    assert hash(left) == hash(right)


def test_variants_of_different_kinds_are_never_equal():
    """``FullDim`` and ``NoChange`` are distinct values, not two names for one.

    Notes
    -----
    Both are payload-free, so this is the assertion that they are two *types*. What makes
    the distinction earn its keep is behavioural, and pinned against xarray below.
    """
    assert FullDim() != NoChange()
    assert Auto() != ByteSize("auto")


# --- against xarray -----------------------------------------------------------------------


@requires_dask
def test_no_change_and_full_dim_are_different_behaviours():
    """``None`` keeps an already-chunked dim's blocks where ``-1`` collapses them into one.

    Notes
    -----
    The fact the taxonomy's shape rests on. If these agreed, the two variants would be one
    behaviour spelled twice and the module would model them as a single variant; they do
    not, so it does not. Asked of xarray rather than asserted from the docs, since it is
    the pinned dask's answer that matters.
    """
    ds = xr.Dataset({"tas": (("time",), np.zeros(1000))}).chunk({"time": 100})
    assert ds.chunk({"time": NoChange().to_raw()}).chunks["time"] == (100,) * 10
    assert ds.chunk({"time": FullDim().to_raw()}).chunks["time"] == (1000,)


@requires_dask
@pytest.mark.parametrize(
    "raw",
    [100, -1, None, "auto", "100MB", (100, 900), [100, 900]],
    ids=["single", "full", "none", "auto", "bytes", "tuple", "list"],
)
def test_to_raw_chunks_the_dim_exactly_as_the_original_spelling_did(raw):
    """Every variant's ``to_raw`` reproduces the chunking its raw spec asked for.

    Notes
    -----
    ``to_raw`` is what a rewrite hands back to replay when it rebuilds a rechunk's
    ``args``, so "xarray accepts it" is not enough — it has to land on the same blocks.
    That covers the two spellings a variant *normalises*: a list of block lengths comes
    back as a tuple, which xarray chunks identically.
    """
    ds = xr.Dataset({"tas": (("time",), np.zeros(1000))})
    assert (
        ds.chunk({"time": classify_chunk(raw).to_raw()}).chunks
        == ds.chunk({"time": raw}).chunks
    )


@requires_dask
def test_classify_to_raw_round_trips_back_to_the_same_variant():
    """Classifying a variant's own ``to_raw`` returns that variant, for every variant.

    Notes
    -----
    Idempotence of the pair, which is what lets a rewritten plan be re-recorded or
    re-optimised without drifting. ``OpaqueChunk`` is included: an unmodelled spec must
    stay unmodelled rather than sliding into a variant on a second pass.
    """
    for spec in (
        SingleSize(100),
        FullDim(),
        NoChange(),
        Auto(),
        ByteSize("100MB"),
        BlockSeq((100, 900)),
        OpaqueChunk(0),
    ):
        assert classify_chunk(spec.to_raw()) == spec


def test_every_variant_is_reachable_from_a_raw_spec():
    """No variant exists that ``classify_chunk`` never mints — an unreachable arm is dead.

    Notes
    -----
    The counterpart of the ``assert_never`` that closes the optimiser's match: mypy pins
    that every variant is *handled*, and nothing but this pins that every variant is
    *produced*. Both directions matter, since a variant no raw spec reaches would be a
    barrier rule nobody could trigger.
    """
    reached = {
        type(classify_chunk(raw))
        for raw in (100, -1, None, "auto", "100MB", (100, 900), 0)
    }
    assert reached == set(get_args(ChunkSpec))
