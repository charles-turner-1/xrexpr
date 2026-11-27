"""Tests for the chunk spec value type (:mod:`xrexpr.chunks`).

:func:`classify_chunk` is the only way to build a chunk spec from a raw ``chunk``
argument, so the classification is tested here, split the same way as in
``test_indexers.py``.

**Examples, at the top.** Which raw spelling becomes which variant is a decision, not
something you can derive, so it's written out. Generating it would mean reimplementing
``classify_chunk`` in the strategy to know the expected answer. The constructor checks
(``SingleSize`` below 1, an empty ``BlockSeq``) and the hashability of a ``Rechunk``
holding each variant are tested here too.

**Live checks against xarray, at the bottom.** Claims that are *about xarray* are
checked against xarray rather than from memory: that every variant's ``to_raw`` replays,
and that ``None`` and ``-1`` really behave differently rather than being two spellings
of one thing. That difference is why :class:`NoChange` and :class:`FullDim` are separate
variants, so it's tested rather than just stated in a docstring.
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

#: xrexpr itself never needs dask, but replaying a ``chunk()`` call does -- without a
#: chunk manager xarray raises rather than chunking, so the live checks below are
#: skipped.
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


@pytest.mark.parametrize("raw", ["100MB", "10MiB", "1GB", "banana", ""])
def test_byte_target_strings_classify_as_byte_size(raw):
    """Any other string is a byte target, kept verbatim for dask to parse at replay.

    Notes
    -----
    Including nonsense strings, which is what "verbatim" really means here. Checking the
    string would mean importing dask to call ``parse_bytes``, and dask is optional. Nor
    can the rule be guessed: ``""`` parses as one byte and ``"100"`` as a hundred, while
    ``"auto "`` with a trailing space raises. A pattern of our own would drift from
    dask's. Not checking leaves dask to raise ``Could not interpret 'banana' as a byte
    unit`` at replay, just as an eager ``chunk`` would.
    """
    assert classify_chunk(raw) == ByteSize(raw)


def test_tuple_of_lengths_classifies_as_block_seq():
    """A tuple of integers is an explicit ``BlockSeq``."""
    assert classify_chunk((100, 400, 500)) == BlockSeq((100, 400, 500))


def test_list_of_lengths_classifies_as_block_seq():
    """The list spelling of a block sequence classifies exactly as the tuple one does."""
    assert classify_chunk([100, 400, 500]) == BlockSeq((100, 400, 500))


@pytest.mark.parametrize("raw", [0, -2, True, 100.0, {100}, {"a": 1}])
def test_unmodelled_specs_classify_as_opaque(raw):
    """Anything outside the modelled variants is an ``OpaqueChunk``, so every input gets an answer.

    Notes
    -----
    Two kinds of value end up here, and neither is modelled. ``0`` and ``-2`` are specs
    xarray rejects, so the plan can't replay either way and the error comes from xarray.
    ``True`` and ``100.0`` are ones dask happens to accept but xarray's API doesn't
    offer; not modelling them costs an optimisation, never correctness.
    """
    assert classify_chunk(raw) == OpaqueChunk(raw)


def test_mixed_sequence_classifies_as_opaque():
    """A sequence that is not all integers is not a block sequence, so it is opaque."""
    assert classify_chunk((100, "auto")) == OpaqueChunk((100, "auto"))


def test_empty_sequence_classifies_as_opaque():
    """An empty sequence is opaque rather than an empty ``BlockSeq``, which dask rejects."""
    assert classify_chunk(()) == OpaqueChunk(())


# numpy block lengths aren't unusual: arithmetic on ``ds.sizes`` gives ``np.int64``, and
# an ``isinstance(x, int)`` test would classify it as opaque, quietly turning an
# ordinary rechunk into a barrier.


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
    """``SingleSize`` can't be built with a length dask would refuse.

    Notes
    -----
    Checked in the constructor rather than downstream, like ``ForwardSlice``. That's
    what lets everything else treat a ``SingleSize`` as a valid block length without
    re-checking. ``-1`` is excluded here *because* it means something else (see
    ``FullDim``).
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
    """A ``Rechunk`` holding any modelled spec still equals an identical one, and is hashable.

    Notes
    -----
    Same lesson as ``Mask`` in ``test_indexers.py``, and the reason ``BlockSeq`` stores
    a tuple. Plans really are compared (the idempotence property in
    ``test_properties.py`` does it), so a variant that broke equality or hashing of the
    enclosing node would be a real bug.
    """
    spec = classify_chunk(raw)
    left = Rechunk(name="chunk", chunks=frozendict({"time": spec}))
    right = Rechunk(name="chunk", chunks=frozendict({"time": classify_chunk(raw)}))
    assert left == right
    assert hash(left) == hash(right)


def test_variants_of_different_kinds_are_never_equal():
    """``FullDim`` and ``NoChange`` are different values, not two names for one.

    Notes
    -----
    Neither has a payload, so this checks they're two separate *types*. The reason they
    need to be is a difference in behaviour, tested against xarray below.
    """
    assert FullDim() != NoChange()
    assert Auto() != ByteSize("auto")


# --- against xarray -----------------------------------------------------------------------


@requires_dask
def test_no_change_and_full_dim_are_different_behaviours():
    """``None`` keeps an already-chunked dim's blocks, where ``-1`` merges them into one.

    Notes
    -----
    This is why there are two variants. If they behaved the same, they'd be one
    behaviour spelled two ways, and the module would model them as one variant. They
    don't, so it doesn't. Checked against xarray rather than the docs, because what
    matters is what the pinned dask actually does.
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
    ``to_raw`` is what a rewrite hands back to replay when it rebuilds a rechunk's call,
    so "xarray accepts it" isn't enough: it has to produce the same blocks. That
    includes the cases a variant *normalises*: a list of block lengths comes back as a
    tuple, which xarray chunks the same way.
    """
    ds = xr.Dataset({"tas": (("time",), np.zeros(1000))})
    assert (
        ds.chunk({"time": classify_chunk(raw).to_raw()}).chunks
        == ds.chunk({"time": raw}).chunks
    )


@requires_dask
def test_classify_to_raw_round_trips_back_to_the_same_variant():
    """Classifying a variant's own ``to_raw`` gives back that variant, for every variant.

    Notes
    -----
    The pair is idempotent, so a rewritten plan can be recorded or optimised again
    without drifting. ``OpaqueChunk`` is included: an unmodelled spec must stay
    unmodelled, not turn into a modelled variant on a second pass.
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
    """Every variant can be produced by ``classify_chunk``; an unreachable variant is dead code.

    Notes
    -----
    The other half of the ``assert_never`` in the optimiser's ``match``. mypy checks
    that every variant is *handled*; only this checks that every variant is *produced*.
    Both matter: a variant no raw spec reaches would be a barrier rule nobody could
    trigger.
    """
    reached = {
        type(classify_chunk(raw))
        for raw in (100, -1, None, "auto", "100MB", (100, 900), 0)
    }
    assert reached == set(get_args(ChunkSpec))
