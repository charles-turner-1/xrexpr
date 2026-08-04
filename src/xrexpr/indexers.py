"""The *value* sum type — a closed taxonomy over what one ``isel``/``sel`` indexer is.

``Select.indexer`` maps each dim to one indexer value, and a raw one is a sum type in
disguise: exactly one of six shapes. This module makes the taxonomy a *type*, so
:func:`classify` is the sole constructor and is **total**, and the facts that follow
become methods rather than re-decisions at every call site in
``ir.py``/``optimize.py``/``schema.py``: :attr:`drops_dim`, :meth:`size`, :meth:`to_raw`.

Whether two indexers *compose* is deliberately **not** modelled here: composition is a
policy the optimiser chooses to prove, not an intrinsic fact of a value, so it stays a
``match`` in ``optimize.py``. What this module guarantees is the discriminant that policy
matches on. :class:`ForwardSlice` earns its own variant so the "forward, non-negative
bounds" carve-out the composer needs is a **constructor invariant** rather than a guard
re-run at every call site, and :class:`Label` is this layer's escape hatch, for ``sel``
labels that cannot be reasoned about positionally.

:meth:`size` has one boundary: a dim of unknown length, or a ``sel`` label slice, answers
"unknown" one level up at ``schema._selected_size``, which guards every call into it.

See ``docs/internals/values.md``.
"""

import numbers
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "ForwardSlice",
    "GeneralSlice",
    "Indexer",
    "Label",
    "Mask",
    "Positions",
    "Scalar",
    "classify",
]


@dataclass(frozen=True)
class Scalar:
    """A single position or label (``isel(time=0)``, ``sel(time="2020")``) — *drops* the dim.

    Attributes
    ----------
    value : Any
        The position or label, verbatim.
    drops_dim : bool
        Always ``True``: a scalar index removes the dim it addresses.
    """

    value: Any
    drops_dim: ClassVar[bool] = True

    @property
    def position(self) -> int | None:
        """The integer position this scalar selects, or ``None`` if it is a coordinate label.

        Returns
        -------
        int or None
            The position, or ``None`` for a label — which the composer cannot reason
            about positionally.

        Notes
        -----
        The one question the composer asks of a scalar — ``isel(time=3)`` composes
        arithmetically, ``sel(time="2020")`` cannot — so it lives here rather than being
        re-decided by ``isinstance`` at each composition site, exactly as
        :attr:`drops_dim` answers the scalar test for ``ir.py``.
        """
        return self.value if _is_int(self.value) else None

    def size(self, current: int) -> int:
        """Never returns: a scalar has no size, because it keeps no dim.

        Parameters
        ----------
        current : int
            The dim's current length. Unused.

        Returns
        -------
        int
            Never returned.

        Raises
        ------
        AssertionError
            Always. Callers must consult :attr:`drops_dim` first; ``apply_schema``
            does, and drops the dim rather than sizing it.
        """
        raise AssertionError("a scalar indexer drops its dim; its size is undefined")

    def to_raw(self) -> Any:
        """Return the xarray-facing value: the position or label itself.

        Returns
        -------
        Any
            :attr:`value`, unchanged.
        """
        return self.value


@dataclass(frozen=True)
class ForwardSlice:
    """A forward, non-negative integer slice (``isel(time=slice(0, 5))``) — composable.

    Attributes
    ----------
    start, stop, step : int or None
        The slice bounds. ``step`` is at least 1 and the bounds are non-negative.
    drops_dim : bool
        Always ``False``: a slice keeps its dim, at a new size.

    Raises
    ------
    ValueError
        On construction with a negative bound or a step below 1.

    Notes
    -----
    The non-negative/forward property is an invariant: :func:`classify` mints this variant
    only when the bounds need no dim length to resolve, and ``__post_init__`` rejects any
    construction that would violate it. That is exactly what lets the composer reason about
    it arithmetically without knowing how long the dim is.
    """

    start: int | None = None
    stop: int | None = None
    step: int | None = None
    drops_dim: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.step is not None and self.step < 1:
            raise ValueError(f"ForwardSlice step must be >= 1, got {self.step}")
        for bound in (self.start, self.stop):
            if bound is not None and bound < 0:
                raise ValueError(f"ForwardSlice bounds must be >= 0, got {bound}")

    def size(self, current: int) -> int:
        """Return the length of the sliced dim, resolved against its current length.

        Parameters
        ----------
        current : int
            The dim's length before this indexer is applied.

        Returns
        -------
        int
            The number of positions the slice selects.
        """
        return len(range(*self.to_raw().indices(current)))

    def to_raw(self) -> slice:
        """Return the xarray-facing value: the bounds as a ``slice``.

        Returns
        -------
        slice
            ``slice(start, stop, step)``.
        """
        return slice(self.start, self.stop, self.step)


@dataclass(frozen=True)
class GeneralSlice:
    """An integer slice with a negative bound or reversed step (``isel(time=slice(-3, None))``).

    Attributes
    ----------
    value : slice
        The slice, verbatim.
    drops_dim : bool
        Always ``False``: a slice keeps its dim, at a new size.

    Notes
    -----
    Sizable — ``slice.indices`` resolves it against a known dim length — but *not* composable,
    since a negative bound counts from the end, which the composer doesn't carry.
    """

    value: slice
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        """Return the length of the sliced dim, resolved against its current length.

        Parameters
        ----------
        current : int
            The dim's length before this indexer is applied.

        Returns
        -------
        int
            The number of positions the slice selects.
        """
        return len(range(*self.value.indices(current)))

    def to_raw(self) -> slice:
        """Return the xarray-facing value: the slice itself.

        Returns
        -------
        slice
            :attr:`value`, unchanged.
        """
        return self.value


@dataclass(frozen=True)
class Positions:
    """A concrete enumeration of integer positions (``isel(time=[0, 2, 4])``) — composable.

    Attributes
    ----------
    values : tuple of int
        The selected positions, in order. A ``tuple`` rather than a list so the enclosing
        node stays hashable and comparable.
    drops_dim : bool
        Always ``False``: an enumeration keeps its dim, at a new size.
    """

    values: tuple[int, ...]
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        """Return the length of the indexed dim, which is exact here.

        Parameters
        ----------
        current : int
            The dim's length before this indexer is applied. Unused — the enumeration
            already says how many positions it selects.

        Returns
        -------
        int
            ``len(values)``.
        """
        return len(self.values)

    def to_raw(self) -> list[int]:
        """Return the xarray-facing value: the positions as a ``list``.

        Returns
        -------
        list of int
            :attr:`values` as a list, since xarray rejects a tuple indexer outright.
        """
        return list(self.values)


@dataclass(frozen=True)
class Mask:
    """A boolean mask (``isel(time=[True, False, ...])`` or a bool array) — sizes by True count.

    Attributes
    ----------
    values : tuple of bool
        One flag per position of the dim, in order.
    drops_dim : bool
        Always ``False``: a mask keeps its dim, at a new size.

    Notes
    -----
    Stored as a ``tuple`` for the same reason :class:`Positions` is, and it matters more here.
    Held verbatim, an ndarray-backed mask makes the *node containing it* compare and hash
    wrongly: ``Mask(arr) == Mask(arr)`` returns an array rather than a bool, so
    ``Select(...) == Select(...)`` raises ``ValueError: truth value ... is ambiguous`` instead of
    answering. Plan equality is a real operation (the idempotence property in
    ``test_properties.py`` asserts it), so a variant that breaks it is a landmine, not a
    curiosity.
    """

    values: tuple[bool, ...]
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        """Return the length of the masked dim: the number of ``True`` flags.

        Parameters
        ----------
        current : int
            The dim's length before this indexer is applied. Unused — xarray requires a
            mask to be exactly as long as its dim, so the flags already say the answer.

        Returns
        -------
        int
            The count of kept positions.
        """
        return sum(self.values)

    def to_raw(self) -> list[bool]:
        """Return the xarray-facing value: the flags as a ``list``.

        Returns
        -------
        list of bool
            :attr:`values` as a list, not the stored tuple: xarray rejects a tuple
            indexer outright (``Could not convert tuple of form (dims, data[, ...])``).
            Same as :class:`Positions`.
        """
        return list(self.values)


@dataclass(frozen=True)
class Label:
    """A coordinate label, label slice, or label sequence (``sel(time="2020")``) — irreducibly open.

    Attributes
    ----------
    value : Any
        The label, label slice or label sequence, verbatim.
    drops_dim : bool
        Always ``False``: a label *scalar* is a :class:`Scalar`, so what reaches this
        variant keeps its dim.

    Notes
    -----
    The value-layer counterpart of :class:`~xrexpr.ir.Opaque`: it keeps its dim but the
    optimiser can't reason about it positionally (no length, no composition). A label
    *sequence* still sizes exactly, by its own length.

    A label *slice* cannot be sized here at all — its extent is a fact about coordinate
    values — and ``apply_schema`` does not ask: it answers ``None`` (unknown) for any
    slice under a ``sel``, this variant included. :meth:`size`'s keep-the-current-size
    fallback exists only for a caller reaching past the schema layer.
    """

    value: Any
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        """Return the length of the labelled dim, exactly where the labels are enumerated.

        Parameters
        ----------
        current : int
            The dim's length before this indexer is applied.

        Returns
        -------
        int
            The sequence's own length for an array or list/tuple of labels; ``current``
            for a label slice, whose extent is a fact about coordinate values. See the
            class notes on why no caller should reach that fallback.
        """
        if isinstance(self.value, np.ndarray):
            return int(self.value.size)
        if isinstance(self.value, (list | tuple)):
            return len(self.value)
        return current  # a label slice needs coord values to size — leave it unchanged

    def to_raw(self) -> Any:
        """Return the xarray-facing value: the label payload itself.

        Returns
        -------
        Any
            :attr:`value`, unchanged.
        """
        return self.value


#: One dim's indexer, as a closed sum. ``match`` over this binds the shape the optimiser and
#: schema layers reason about; :func:`classify` is the sole constructor from raw values.
Indexer = Scalar | ForwardSlice | GeneralSlice | Positions | Mask | Label


def _is_int(x: Any) -> bool:
    """Report whether ``x`` is an integer *position*, numpy-typed or not.

    Parameters
    ----------
    x : Any
        A candidate index, slice bound or sequence element.

    Returns
    -------
    bool
        ``True`` for an integer position, ``False`` for anything else — booleans
        included.

    Notes
    -----
    ``isinstance(x, int)`` is too narrow: numpy integers fall out of ``argmin``, ``np.where``
    and ``arr.values[i]`` routinely, and a ``np.int64`` bound misfiled as a label makes its
    slice both mis-sized and uncomposable. ``numbers.Integral`` covers every numpy width;
    ``bool`` is excluded because it is an ``int`` subclass but means a *mask*, not a position.
    (``np.bool_`` is not ``Integral``, so it needs no exclusion.)
    """
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _scalar(value: Any) -> Scalar:
    """Build a :class:`Scalar`, narrowing an integer position to plain ``int``.

    Parameters
    ----------
    value : Any
        The raw scalar index — a position or a coordinate label.

    Returns
    -------
    Scalar
        The variant, with an integer position narrowed to ``int`` and a label kept as is.

    Notes
    -----
    The same normalisation :class:`Positions` and :class:`ForwardSlice` already apply, for the
    same two reasons: an ``np.int64`` position must compare equal to the ``int`` spelling of
    the same selection (golden plan assertions, and plan equality in general), and a variant
    holding a numpy *array* is unhashable — which would silently make the enclosing ``Select``
    unhashable too. Labels pass through untouched: they are open by nature (§3.2).
    """
    return Scalar(int(value) if _is_int(value) else value)


def _is_forward(s: slice) -> bool:
    """Report whether ``s`` steps forward from non-negative integer bounds.

    Parameters
    ----------
    s : slice
        The slice to test.

    Returns
    -------
    bool
        ``True`` when every bound resolves without knowing the dim's length — the
        invariant :class:`ForwardSlice` is built on.
    """
    if s.step is not None and (not _is_int(s.step) or s.step < 1):
        return False
    return all(b is None or (_is_int(b) and b >= 0) for b in (s.start, s.stop))


def _classify_slice(s: slice) -> ForwardSlice | GeneralSlice | Label:
    """Sort a slice indexer into its variant.

    Parameters
    ----------
    s : slice
        The raw slice, from ``isel`` or ``sel``.

    Returns
    -------
    ForwardSlice or GeneralSlice or Label
        :class:`ForwardSlice` when the bounds are forward and non-negative,
        :class:`GeneralSlice` for another integer slice, and :class:`Label` when a bound
        is not an integer at all — a label slice, which is not positional.
    """
    bounds = (s.start, s.stop, s.step)
    if not all(b is None or _is_int(b) for b in bounds):
        return Label(s)  # a label slice (e.g. sel) — not positional
    if _is_forward(s):
        # normalise to plain ``int`` so the variant's declared field types stay honest and a
        # numpy-bounded slice compares equal to the same slice written by hand
        return ForwardSlice(*(None if b is None else int(b) for b in bounds))
    return GeneralSlice(s)


def classify(value: Any) -> Indexer:
    """Sort a raw ``isel``/``sel`` indexer value into its :data:`Indexer` variant.

    Parameters
    ----------
    value : Any
        One dim's indexer, exactly as it was passed to ``isel``/``sel``: a position, a
        label, a slice, a sequence, or an array.

    Returns
    -------
    Indexer
        The matching variant. Anything the taxonomy cannot reason about positionally
        becomes a :class:`Label`, the value layer's escape hatch.

    Notes
    -----
    The single place the value taxonomy is decided. Order matters twice: a boolean sequence is a
    :class:`Mask`, not :class:`Positions`, even though ``bool`` is an ``int`` subclass, so the
    all-boolean test runs before the all-integer one; and a 0-d array is a :class:`Scalar`
    rather than an enumeration, so the rank check runs before the dtype dispatch.

    Integer-ness is decided by ``_is_int`` throughout, so numpy-typed positions and bounds
    classify the same way their Python equivalents do.
    """
    if isinstance(value, slice):
        return _classify_slice(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:  # a 0-d array indexes like the bare value: drops the dim
            return _scalar(
                value.item() if np.issubdtype(value.dtype, np.integer) else value
            )
        if value.dtype == bool:
            # Only a 1-d mask is a mask. xarray rejects a higher-rank boolean array as a
            # single-dim indexer outright ("Unlabeled multi-dimensional array cannot be used
            # for indexing"), so there is no valid plan containing one — it goes to ``Label``,
            # the variant for values the optimiser cannot reason about, and the error still
            # surfaces from xarray at replay in its own words.
            if value.ndim != 1:
                return Label(value)
            return Mask(tuple(bool(x) for x in value.tolist()))
        if np.issubdtype(value.dtype, np.integer):
            return Positions(tuple(int(x) for x in value.tolist()))
        return Label(value)
    if isinstance(value, (list | tuple)):
        if value and all(isinstance(x, (bool | np.bool_)) for x in value):
            return Mask(tuple(bool(x) for x in value))
        if all(_is_int(x) for x in value):  # pure-bool already handled above
            return Positions(tuple(int(x) for x in value))
        return Label(value)  # a label sequence, or a mixed one
    return _scalar(value)  # anything else drops its dim
