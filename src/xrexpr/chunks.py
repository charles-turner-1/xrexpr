"""The *chunk* value sum type: a closed taxonomy over what one dim's chunk spec is.

``Rechunk.chunks`` maps each named dim to one chunk *spec*. Like an indexer value, a raw
chunk spec is a sum type in disguise — a size, a symbolic request, or an explicit block
sequence — and holding it as ``Any`` forces every call site to re-derive the shapes by
hand. This module makes the taxonomy a *type*, so the classification lives in one place
(:func:`classify_chunk`) and what follows from it becomes a method rather than a
re-decision:

- :meth:`to_raw` — the exact xarray-facing value to hand back to replay when a rewrite
  rebuilds a rechunk's ``args`` from its ``chunks``.

Whether a given spec may be *crossed* by a select is deliberately **not** modelled here:
pushability is a policy the optimiser chooses (``optimize._pushable_rechunk``), not an
intrinsic fact of a value — the same stance :mod:`xrexpr.indexers` takes on composition.
What this module guarantees is the discriminant that policy matches on.

The variants are one per *behaviour*, not one per spelling, and the split was checked
against the pinned xarray/dask rather than reasoned about. Applying each to
``ds.chunk({"time": 100})`` on a 1000-long dim:

============ ================================ ================================
raw          result                           variant
============ ================================ ================================
``None``     keeps the existing 100-blocks    :class:`NoChange`
``-1``       collapses to one 1000-block      :class:`FullDim`
``"auto"``   dask re-picks by its byte target :class:`Auto`
``"100MB"``  dask re-picks to that target     :class:`ByteSize`
``100``      uniform 100-blocks               :class:`SingleSize`
``(100, 9)`` those blocks exactly             :class:`BlockSeq`
============ ================================ ================================

So ``None`` and ``-1`` are **not** two spellings of one behaviour, which is why both
:class:`NoChange` and :class:`FullDim` exist.

:class:`OpaqueChunk` is this layer's escape hatch, the role :class:`~xrexpr.indexers.Label`
plays for indexers and :class:`~xrexpr.ir.Opaque` plays for op kinds. It exists because
:func:`classify_chunk` must be **total** — ``Rechunk.__post_init__`` classifies hand-built
nodes too, and the accessor records whatever it is handed. Some of what lands there xarray
accepts (a whole float coerces; ``True`` means blocks of 1) and most of it xarray rejects
(``0`` divides by zero, ``-2`` and an unparseable byte string raise). Neither case is
modelled: the first is a spelling xarray's own API does not offer, the second is an error
that stays the caller's to see — recording it opaquely leaves the rechunk a barrier and
lets the failure surface at replay, in xarray's words, exactly as it would have eagerly.
"""

import numbers
from dataclasses import dataclass
from typing import Any

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
    """A uniform block size (``chunk({"time": 100})``) — every block that long but the last.

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
    The lower bound is a constructor invariant rather than a guard re-run downstream, the
    same move :class:`~xrexpr.indexers.ForwardSlice` makes: ``0`` raises
    ``ZeroDivisionError`` inside dask and a negative other than ``-1`` raises, so no valid
    plan holds one. :func:`classify_chunk` sends those to :class:`OpaqueChunk` instead, and
    the invariant is what makes that unconstructible by hand as well.
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
    """A request for dask to pick block sizes (``chunk({"time": "auto"})``).

    Notes
    -----
    Carries no payload: the sizes are dask's to choose, against its configured byte
    target. That it re-picks against whatever data survives is precisely why a select may
    hop in front of one.
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
    :class:`Auto` with an explicit target rather than dask's configured one, so it is a
    separate variant only because it has a payload to carry back to replay. Whether the
    string parses is dask's judgement, made at replay: an unparseable one raises
    ``ValueError: Could not interpret ... as a byte unit`` there, which is a better error
    than anything this layer could invent.
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
    Distinct from :class:`NoChange` in behaviour, not just spelling: on an
    already-chunked dim ``-1`` collapses the existing blocks into one and ``None`` leaves
    them alone. Verified against the pinned xarray/dask — see the module docstring.
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
    The identity spec for one dim. It names the dim without asking anything of it, which
    is why it survives a select unchanged. See :class:`FullDim` on why the two are
    separate variants.
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
    """Explicit block lengths (``chunk({"time": (100, 400, 500)})``) — must sum to the dim.

    Attributes
    ----------
    sizes : tuple of int
        The block lengths, in order. A ``tuple`` rather than a list so the enclosing node
        stays hashable and comparable, as :class:`~xrexpr.indexers.Positions` is.

    Raises
    ------
    ValueError
        On construction with no blocks: dask rejects an empty tuple outright.

    Notes
    -----
    The one spec whose validity depends on the dim's *length*, which is what makes it a
    barrier: a select moving in front would shrink the dim and leave blocks that no longer
    add up. That judgement is the rule's, not this class's — see the module docstring.
    """

    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sizes:
            raise ValueError("BlockSeq needs at least one block")

    def to_raw(self) -> tuple[int, ...]:
        """Return the xarray-facing value: the block lengths as a ``tuple``.

        Returns
        -------
        tuple of int
            :attr:`sizes`, unchanged — xarray accepts a tuple of block lengths here,
            unlike the sequence indexers of :mod:`xrexpr.indexers`, which must be lists.
        """
        return self.sizes


@dataclass(frozen=True)
class OpaqueChunk:
    """Any chunk spec the taxonomy doesn't model — replayed verbatim, never crossed.

    Attributes
    ----------
    value : Any
        The spec, exactly as it was passed to ``chunk``.

    Notes
    -----
    The value layer's counterpart of :class:`~xrexpr.ir.Opaque`, reached by a spelling
    xarray tolerates but does not document (a whole float, ``True``) and by one it rejects
    (``0``, ``-2``, a set). Holding it verbatim keeps replay faithful in the first case and
    keeps the error xarray's own in the second. Like :class:`~xrexpr.indexers.Label`, it
    can hold an unhashable payload, which makes the enclosing node unhashable too — the
    conditional hashability ``ir.py``'s module docstring already describes.
    """

    value: Any

    def to_raw(self) -> Any:
        """Return the xarray-facing value: the spec itself.

        Returns
        -------
        Any
            :attr:`value`, unchanged.
        """
        return self.value


#: One dim's chunk spec, as a closed sum. ``match`` over this binds the shape the optimiser
#: reasons about; :func:`classify_chunk` is the sole constructor from raw values.
ChunkSpec = SingleSize | Auto | ByteSize | FullDim | NoChange | BlockSeq | OpaqueChunk


def _is_int(x: Any) -> bool:
    """Report whether ``x`` is an integer *block length*, numpy-typed or not.

    Parameters
    ----------
    x : Any
        A candidate chunk size or block length.

    Returns
    -------
    bool
        ``True`` for an integer, ``False`` for anything else — booleans included.

    Notes
    -----
    The rule :func:`xrexpr.indexers._is_int` applies to positions, for the same two
    reasons: ``numbers.Integral`` covers every numpy width, so an ``np.int64`` block length
    classifies like the ``int`` spelling of the same request; and ``bool`` is excluded
    despite subclassing ``int``, because ``chunk({"time": True})`` asking for blocks of 1
    is a mistake rather than a request, and :class:`OpaqueChunk` is where mistakes go.
    Spelled out here rather than imported so neither module owns the other's rule.
    """
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def classify_chunk(value: Any) -> ChunkSpec:
    """Sort one dim's raw ``chunk`` spec into its :data:`ChunkSpec` variant.

    Parameters
    ----------
    value : Any
        The dim's chunk spec, exactly as it was passed to ``chunk``: a size, ``-1``,
        ``None``, ``"auto"``, a byte-target string, or a sequence of block lengths.

    Returns
    -------
    ChunkSpec
        The matching variant. Anything the taxonomy does not model becomes an
        :class:`OpaqueChunk`, so the function is total.

    Notes
    -----
    The single place the chunk taxonomy is decided. Order matters once: ``-1`` is a
    :class:`FullDim` rather than a :class:`SingleSize`, so the sentinel is tested before
    the size is, and every remaining non-positive integer falls through to
    :class:`OpaqueChunk` rather than violating :class:`SingleSize`'s invariant.

    Integer-ness is decided by :func:`_is_int` throughout, so numpy-typed lengths classify
    the same way their Python equivalents do — and a block sequence normalises to
    ``tuple[int, ...]`` for the hashability reason :class:`~xrexpr.indexers.Mask`
    documents.
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
