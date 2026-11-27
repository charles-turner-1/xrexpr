"""The chunk spec *value* type: a closed set of shapes one dim's chunk spec can take.

``Rechunk.chunks`` maps each named dim to one spec. A raw chunk spec is one of a few
shapes (a size, a symbolic request like ``"auto"``, an explicit list of blocks), so this
module makes them explicit types. :func:`classify_chunk` is the only way to build one
and accepts any value, and :meth:`to_raw` turns a variant back into what xarray expects.

There's one variant per *behaviour*, not per spelling. That's why ``None`` (keep the
existing blocks) and ``-1`` (merge them into one block) are separate variants.

Whether a select may move past a spec isn't decided here. That's the optimiser's policy
(``optimize._pushable_rechunk``). What this module provides is the property that policy
uses: **extent-dependence**. :class:`Auto`, :class:`ByteSize` and :class:`BlockSeq` are
the specs dask sizes by measuring the array it's given. Each variant's notes say whether
it's extent-dependent; none of them says what the optimiser should do about it.

See ``docs/internals/values.md`` for the classification tree, and
``docs/guide/rechunking.md`` for the raw-value-to-variant table (measured against the
pinned xarray/dask).
"""

import numbers
from dataclasses import dataclass
from typing import Any

from xrexpr.payload import Payload

__all__ = [
    "Auto",
    "BlockSeq",
    "ByteSize",
    "ChunkSpec",
    "FullDim",
    "NoChange",
    "OpaqueChunk",
    "SingleSize",
    "classify_chunk",
]


@dataclass(frozen=True)
class SingleSize:
    """A fixed block size (``chunk({"time": 100})``): every block that long except the last.

    Attributes
    ----------
    size : int
        The requested block length, at least 1.

    Raises
    ------
    ValueError
        On construction with a size below 1.

    Notes
    -----
    The lower bound is checked by the constructor rather than downstream, as
    :class:`~xrexpr.indexers.ForwardSlice` does. ``0`` raises ``ZeroDivisionError``
    inside dask, and any negative other than ``-1`` raises too, so no valid plan has
    one. :func:`classify_chunk` sends those to :class:`OpaqueChunk`, and the constructor
    check stops anyone building one by hand.
    """

    size: int

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"SingleSize must be >= 1, got {self.size}")

    def to_raw(self) -> int:
        """Return the xarray-facing value: the block length itself.

        Returns
        -------
        int
            :attr:`size`, unchanged.
        """
        return self.size


@dataclass(frozen=True)
class Auto:
    """A request for dask to choose block sizes (``chunk({"time": "auto"})``).

    Notes
    -----
    No payload: dask picks the sizes against its configured byte target. Because it
    picks them from whatever array it receives, it's *extent-dependent* (like
    :class:`BlockSeq`): dask divides the byte budget by the array's own size, so the
    spec's meaning changes when the data does.

    Measured against xarray 2026.7.0 / dask 2026.7.1, it's worse than a different
    answer. ``auto_chunks`` computes ``largest_block = prod(cs if Number else max(cs)
    for cs in chunks if cs != "auto")``, so an empty dim makes that a
    ``ZeroDivisionError``. An emptied dim that *was* ``"auto"`` hits it too: dask pins
    every dim below ``limit ** (1 / ndim)`` and *recurses*, and on the second pass that
    dim is no longer ``"auto"``:

    ==================================== ========= =========================
    ``chunk`` then empty a dim, hoisted  at 1 kB   at 128 MiB
    ==================================== ========= =========================
    2-D ``{lat: auto, lon: auto}``, lat  raises    fine
    3-D ``{lon: auto}``, lon             fine      fine
    3-D ``{lon: auto}``, lat             raises    raises
    ==================================== ========= =========================

    Whether it raises depends on the configured ``array.chunk-size`` and on the number
    of dims, neither of which a plan can see. So there's no narrower test, and the whole
    variant blocks selects. The one exception is a select that provably empties the dim,
    on xarray 2026.09.0 and later (``optimize._empty_autochunk_is_pushable``).
    """

    def to_raw(self) -> str:
        """Return the xarray-facing value: the literal ``"auto"``.

        Returns
        -------
        str
            ``"auto"``.
        """
        return "auto"


@dataclass(frozen=True)
class ByteSize:
    """A byte target for dask to size blocks against (``chunk({"time": "100MB"})``).

    Attributes
    ----------
    value : str
        The target as written, e.g. ``"100MB"`` or ``"10MiB"``.

    Notes
    -----
    :class:`Auto` with an explicit target instead of dask's configured one. It's only a
    separate variant because it has a value to carry back to replay, and it's
    extent-dependent for the same reason (it divides that target by the array's size).
    Whether the string parses is left to dask at replay. An unparseable one raises
    ``ValueError: Could not interpret ... as a byte unit`` there, which is a better
    error than anything this layer could produce.
    """

    value: str

    def to_raw(self) -> str:
        """Return the xarray-facing value: the target string.

        Returns
        -------
        str
            :attr:`value`, unchanged.
        """
        return self.value


@dataclass(frozen=True)
class FullDim:
    """One block spanning the whole dim (``chunk({"time": -1})``).

    Notes
    -----
    Behaves differently from :class:`NoChange`, not just spelled differently: on a dim
    that's already chunked, ``-1`` merges the blocks into one and ``None`` leaves them
    alone. Checked against the pinned xarray/dask (see the module docstring).
    """

    def to_raw(self) -> int:
        """Return the xarray-facing value: ``-1``.

        Returns
        -------
        int
            ``-1``.
        """
        return -1


@dataclass(frozen=True)
class NoChange:
    """Leave this dim's chunking as it is (``chunk({"time": None})``).

    Notes
    -----
    The identity spec for one dim. It names the dim but asks nothing of it, which is why
    it survives a select unchanged. See :class:`FullDim` for why the two are separate.
    """

    def to_raw(self) -> None:
        """Return the xarray-facing value: ``None``.

        Returns
        -------
        None
            ``None``, the spelling xarray reads as "leave this dim alone".
        """
        return None


@dataclass(frozen=True)
class BlockSeq:
    """Explicit block lengths (``chunk({"time": (100, 400, 500)})``). Must sum to the dim.

    Attributes
    ----------
    sizes : tuple of int
        The block lengths, in order. A ``tuple`` rather than a list so the enclosing
        node stays hashable and comparable, like :class:`~xrexpr.indexers.Positions`.

    Raises
    ------
    ValueError
        On construction with no blocks: dask rejects an empty tuple.

    Notes
    -----
    Extent-dependent, like :class:`Auto` and :class:`ByteSize`, and the most obvious
    case: a select moved in front would shrink the dim, and the blocks would no longer
    add up. The three fail differently. This one can't replay at all; the other two
    replay to a different answer or a ``ZeroDivisionError``. Treating all three as
    barriers is the optimiser's decision, not this class's (see the module docstring).
    """

    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sizes:
            raise ValueError("BlockSeq needs at least one block")

    def to_raw(self) -> tuple[int, ...]:
        """Return the value xarray expects: the block lengths as a ``tuple``.

        Returns
        -------
        tuple of int
            :attr:`sizes`, unchanged. xarray accepts a tuple of block lengths here,
            unlike the list indexers in ``xrexpr.indexers``, which must be lists.
        """
        return self.sizes


@dataclass(frozen=True)
class OpaqueChunk:
    """Any chunk spec the taxonomy doesn't model. Replayed verbatim, never crossed.

    Attributes
    ----------
    value : Payload[Any]
        The spec, exactly as passed to ``chunk``. A :data:`~xrexpr.payload.Payload`: the
        Rust reasoner never reads it.

    Notes
    -----
    The value-level equivalent of :class:`~xrexpr.ir.Opaque`. Two kinds of value end up
    here: spellings xarray tolerates but doesn't document (a whole float, ``True``), and
    ones it rejects (``0``, ``-2``, a set). Keeping the value verbatim makes replay
    faithful for the first kind, and keeps xarray's own error for the second. Like
    :class:`~xrexpr.indexers.Label`, it can hold an unhashable value, which makes the
    enclosing node unhashable too (see ``ir.py``'s module docstring).
    """

    value: Payload[Any]

    def to_raw(self) -> Any:
        """Return the xarray-facing value: the spec itself.

        Returns
        -------
        Any
            :attr:`value`, unchanged.
        """
        return self.value


#: One dim's chunk spec, as a closed union. The optimiser matches on it;
#: :func:`classify_chunk` is the only way to build one from a raw value.
ChunkSpec = SingleSize | Auto | ByteSize | FullDim | NoChange | BlockSeq | OpaqueChunk


def _is_int(x: Any) -> bool:
    """Report whether ``x`` is an integer *block length*, numpy or Python.

    Parameters
    ----------
    x : Any
        A candidate chunk size or block length.

    Returns
    -------
    bool
        ``True`` for an integer, ``False`` for anything else, booleans included.

    Notes
    -----
    The same rule ``indexers._is_int`` uses for positions, for the same two reasons.
    ``numbers.Integral`` covers every numpy width, so an ``np.int64`` block length
    classifies like the ``int`` spelling. And ``bool`` is excluded even though it
    subclasses ``int``, because ``chunk({"time": True})`` (blocks of 1) is a mistake,
    and mistakes go to :class:`OpaqueChunk`. It's written out here rather than imported
    so neither module depends on the other.
    """
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def classify_chunk(value: Any) -> ChunkSpec:
    """Sort one dim's raw ``chunk`` spec into its :data:`ChunkSpec` variant.

    Parameters
    ----------
    value : Any
        The dim's chunk spec, exactly as passed to ``chunk``: a size, ``-1``, ``None``,
        ``"auto"``, a byte-target string, or a sequence of block lengths.

    Returns
    -------
    ChunkSpec
        The matching variant. Anything not modelled becomes an :class:`OpaqueChunk`, so
        every input gets an answer.

    Notes
    -----
    The one place chunk specs are classified. Order matters once: ``-1`` is a
    :class:`FullDim`, not a :class:`SingleSize`, so it's checked before sizes. Every
    other non-positive integer falls through to :class:`OpaqueChunk` rather than failing
    :class:`SingleSize`'s check.

    Integer checks use ``_is_int``, so numpy lengths classify the same as Python ones. A
    block sequence is converted to ``tuple[int, ...]`` for the hashability reason
    :class:`~xrexpr.indexers.Mask` explains.
    """
    if value is None:
        return NoChange()
    if isinstance(value, str):
        return Auto() if value == "auto" else ByteSize(value)
    if _is_int(value):
        if value == -1:
            return FullDim()
        return SingleSize(int(value)) if value >= 1 else OpaqueChunk(value)
    if isinstance(value, (list | tuple)):
        if value and all(_is_int(size) for size in value):
            return BlockSeq(tuple(int(size) for size in value))
        return OpaqueChunk(value)
    return OpaqueChunk(value)
