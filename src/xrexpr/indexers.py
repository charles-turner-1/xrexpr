"""The indexer value type: the closed set of shapes an ``isel``/``sel`` indexer takes.

``Select.indexer`` maps each dim to one indexer, and a raw indexer is always one of a
handful of shapes. This module makes those shapes explicit types:

- :func:`classify` is the only way to build one, and it accepts any value.
- The facts that follow from a shape are methods (:attr:`drops_dim`, :meth:`size`,
  :meth:`to_raw`), so ``ir.py``/``optimize.py``/``schema.py`` don't each re-decide them.

Whether two indexers *compose* isn't decided here. That's a policy the optimiser chooses
(which compositions it's willing to prove), not a fact about a value, so it lives in a
``match`` in ``optimize.py``. This module just provides the variants that ``match``
works on. Two of them exist specifically for it:

- :class:`ForwardSlice` exists so "forward, non-negative bounds", which the composer
  needs, is guaranteed by the constructor rather than re-checked at every use.
- :class:`Label` is the fallback for ``sel`` labels that can't be reasoned about
  positionally.

:meth:`size` doesn't handle unknowns. A dim of unknown length, or a ``sel`` label slice,
is answered as "unknown" one level up in ``schema._selected_size``, which guards every
call into it.

See ``docs/internals/values.md``.
"""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import xarray as xr

from xrexpr.payload import Payload

__all__ = [
    "Advanced",
    "ForwardSlice",
    "GeneralSlice",
    "Indexer",
    "Label",
    "Mask",
    "Positions",
    "Scalar",
    "classify",
]


class GenericIndex(ABC):
    @abstractmethod
    def size(self, current: int) -> int: ...

    @abstractmethod
    def to_raw(self) -> Any: ...


@dataclass(frozen=True)
class Scalar(GenericIndex):
    """A single position or label (``isel(time=0)``, ``sel(time="2020")``). *Drops* the dim.

    Attributes
    ----------
    value : Any
        The position or label, verbatim. Half semantic, half payload: an integer position
        is something the Rust reasoner may compose (``isel(time=3)`` after a slice), a label
        is a :data:`~xrexpr.payload.Payload` it can only carry. Which one is decided by
        :attr:`position`, so the field is not annotated either way.
    drops_dim : bool
        Always ``True``: a scalar index removes the dim it indexes.
    """

    value: Any
    drops_dim: ClassVar[bool] = True

    @property
    def position(self) -> int | None:
        """The integer position this scalar selects, or ``None`` if it's a coordinate label.

        Returns
        -------
        int or None
            The position, or ``None`` for a label, which the composer can't reason about
            positionally.

        Notes
        -----
        The one thing the composer asks of a scalar: ``isel(time=3)`` composes
        arithmetically, ``sel(time="2020")`` can't. Having it here means composition
        sites don't each check ``isinstance``, in the same way :attr:`drops_dim` answers
        "does this drop the dim?" for ``ir.py``.
        """
        return self.value if _is_int(self.value) else None

    def size(self, current: int) -> int:
        """Never returns: a scalar keeps no dim, so it has no size.

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
            Always. Callers must check :attr:`drops_dim` first. ``apply_schema`` does,
            and drops the dim instead of sizing it.
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
class ForwardSlice(GenericIndex):
    """A forward, non-negative integer slice (``isel(time=slice(0, 5))``). Composable.

    Attributes
    ----------
    start, stop, step : int or None
        The slice bounds. ``step`` is at least 1, and the bounds are non-negative.
    drops_dim : bool
        Always ``False``: a slice keeps its dim, at a new size.

    Raises
    ------
    ValueError
        On construction with a negative bound or a step below 1.

    Notes
    -----
    Forward and non-negative is guaranteed: :func:`classify` only creates this variant
    when the bounds need no dim length to resolve, and ``__post_init__`` rejects
    anything else. That's what lets the composer do arithmetic on it without knowing the
    dim's length.
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
class GeneralSlice(GenericIndex):
    """An integer slice with a negative bound or reversed step (``slice(-3, None)``).

    Attributes
    ----------
    value : slice
        The slice, verbatim.
    drops_dim : bool
        Always ``False``: a slice keeps its dim, at a new size.

    Notes
    -----
    Can be sized (``slice.indices`` resolves it against a known length), but not
    composed: a negative bound counts from the end, and the composer doesn't know where
    the end is.
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
class Positions(GenericIndex):
    """A concrete enumeration of integer positions (``isel(time=[0, 2, 4])``) — composable.

    Attributes
    ----------
    values : tuple of int
        The selected positions, in order. A ``tuple`` rather than a list so the
        enclosing node stays hashable and comparable.
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
class Mask(GenericIndex):
    """A boolean mask (``isel(time=[True, False, ...])`` or a bool array). Sized by its ``True`` count.

    Attributes
    ----------
    values : tuple of bool
        One flag per position of the dim, in order.
    drops_dim : bool
        Always ``False``: a mask keeps its dim, at a new size.

    Notes
    -----
    Stored as a ``tuple``, like :class:`Positions`, and it matters more here. Stored as
    an ndarray, a mask makes the *node containing it* compare and hash wrongly:
    ``Mask(arr) == Mask(arr)`` returns an array rather than a bool, so comparing two
    ``Select`` nodes raises ``ValueError: truth value ... is ambiguous``. Plans do get
    compared (the idempotence property in ``test_properties.py`` does it), so this would
    be a real bug.
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
class Label(GenericIndex):
    """A coordinate label, label slice, or label sequence (``sel(time="2020")``). Not positional.

    Attributes
    ----------
    value : Payload[Any]
        The label, label slice or label sequence, verbatim. A
        :data:`~xrexpr.payload.Payload`: the Rust reasoner sees only :attr:`known_size`.
    known_size : int or None
        The number of labels for an array, list or tuple of labels; ``None`` otherwise
        (a single label slice, whose length depends on coordinate values). Derived from
        :attr:`value`, never passed in.
    drops_dim : bool
        Always ``False``. A scalar label is a :class:`Scalar`, so anything that reaches
        this variant keeps its dim.

    Notes
    -----
    The value-level equivalent of :class:`~xrexpr.ir.Opaque`: it keeps its dim, but the
    optimiser can't reason about it positionally (no length, no composition). A label
    *sequence* can still be sized exactly, by its own length.

    A label *slice* can't be sized here at all, since its extent depends on coordinate
    values. ``apply_schema`` doesn't ask: it returns ``None`` (unknown) for any slice
    under a ``sel``, including this variant. :meth:`size`'s keep-the-current-size
    fallback is only for callers that go around the schema layer.

    ``known_size`` is a stored field (``init=False``) rather than a property so it survives
    the Rust boundary: there :attr:`value` is replaced by a
    :class:`~xrexpr.payload.PayloadRef`, and a property would then compute ``None``. Same
    pattern as ``Rechunk.uniform``: derived in ``__post_init__``, so it can't disagree with
    :attr:`value`.
    """

    value: Payload[Any]
    known_size: int | None = field(init=False, default=None)
    drops_dim: ClassVar[bool] = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "known_size", _label_count(self.value))

    def size(self, current: int) -> int:
        """Return the labelled dim's length, exactly where the labels are listed.

        Parameters
        ----------
        current : int
            The dim's length before this indexer is applied.

        Returns
        -------
        int
            The sequence's own length for an array, list or tuple of labels. ``current``
            for a label slice, whose extent depends on coordinate values. See the class
            notes on why no caller should reach that fallback.
        """
        ks = self.known_size
        return (
            ks if ks is not None else current
        )  # a label slice needs coord values to size — leave it unchanged

    def to_raw(self) -> Any:
        """Return the xarray-facing value: the label payload itself.

        Returns
        -------
        Any
            :attr:`value`, unchanged.
        """
        return self.value


@dataclass(frozen=True)
class Advanced(GenericIndex):
    """An ``xr.DataArray``/``xr.Variable`` indexer: xarray's *advanced* indexing.

    Attributes
    ----------
    dims : tuple of Hashable
        The indexer array's own dims. The array itself isn't stored, so the variant (and
        a ``Select`` holding it) stays hashable, which a raw ``DataArray`` isn't.
    drops_dim : bool
        Always ``False``, but that's a placeholder, not the truth. Whether the dim is
        dropped depends on *which dim is being indexed*, which this value doesn't know.
        **Orthogonal** indexing (``dims`` contains that dim) keeps it, resized.
        **Vectorized** indexing (``dims`` is a new name) drops it and adds the array's
        dims. Deciding needs the dim key, which lives one level up, so it isn't modelled
        here.

    Notes
    -----
    In practice this only ever represents a **vectorized** indexer. ``classify`` doesn't
    know the dim, so it creates an :class:`Advanced` for *any*
    ``DataArray``/``Variable``. But before a select's indexers get here,
    ``schema._select_indexer`` converts the *orthogonal* ones (a 0-d array, or a 1-d
    array named after the dim it indexes) to plain ``.values``, since they index the
    same way a bare ``ndarray`` would (see ``schema._orthogonal_advanced``). What's left
    as :class:`Advanced` drops its dim and adds the array's dims, which no positional
    indexer can express.

    Any select holding one is recorded as :class:`~xrexpr.ir.Opaque` by
    ``schema.to_opnode`` (``schema._select_has_advanced``). An indexer can't end the
    trusted prefix (only an op can), and neither ``drops_dim`` nor a fixed :meth:`size`
    can describe an effect that adds dims. So this variant never appears on a live plan.
    It exists to *represent* the value and *trigger* the demotion. Without it, the
    ``_scalar`` catch-all in :func:`classify` would treat the array as a dim-dropping
    :class:`Scalar`, which is wrong and unhashable. Modelling vectorized indexing as a
    live ``Select`` is #60.
    """

    dims: tuple[Hashable, ...]
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        """Never returns: an advanced indexer's size depends on the indexed dim.

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
            Always. A select holding this variant is :class:`~xrexpr.ir.Opaque`, so
            ``apply_schema`` never sizes it.
        """
        raise AssertionError(
            "an advanced indexer is never emitted; its select is Opaque"
        )

    def to_raw(self) -> Any:
        """Never returns: the array isn't stored, and the op replays as ``Opaque``.

        Returns
        -------
        Any
            Never returned.

        Raises
        ------
        AssertionError
            Always. The verbatim ``DataArray`` lives in the :class:`~xrexpr.ir.Opaque`
            node's ``args``/``kwargs``, which is what replays. This variant is never on
            a live plan.
        """
        raise AssertionError(
            "an advanced indexer is never emitted; its select is Opaque"
        )


#: One dim's indexer, as a closed union. The optimiser and schema layers ``match`` on
#: it; :func:`classify` is the only way to build one from a raw value.
Indexer = Scalar | ForwardSlice | GeneralSlice | Positions | Mask | Label | Advanced


def _label_count(value: Any) -> int | None:
    """Count the labels in a ``Label``'s value, if they're listed.

    Parameters
    ----------
    value : Any
        The label, label slice or label sequence.

    Returns
    -------
    int or None
        The length of an array, list or tuple of labels; ``None`` otherwise.
    """
    if isinstance(value, np.ndarray):
        return int(value.size)
    if isinstance(value, (list | tuple)):
        return len(value)
    return None


def _is_int(x: Any) -> bool:
    """Report whether ``x`` is an integer *position*, numpy or Python.

    Parameters
    ----------
    x : Any
        A candidate index, slice bound or sequence element.

    Returns
    -------
    bool
        ``True`` for an integer position, ``False`` for anything else, booleans
        included.

    Notes
    -----
    ``isinstance(x, int)`` is too narrow. numpy integers come out of ``argmin``,
    ``np.where`` and ``arr.values[i]`` all the time, and an ``np.int64`` bound mistaken
    for a label makes its slice both wrongly sized and uncomposable.
    ``numbers.Integral`` covers every numpy width. ``bool`` is excluded because,
    although it's an ``int`` subclass, it means a *mask*, not a position. (``np.bool_``
    isn't ``Integral``, so it needs no exclusion.)
    """
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _scalar(value: Any) -> Scalar:
    """Build a :class:`Scalar`, converting an integer position to plain ``int``.

    Parameters
    ----------
    value : Any
        The raw scalar index: a position or a coordinate label.

    Returns
    -------
    Scalar
        The variant. An integer position becomes ``int``; a label is kept as is.

    Notes
    -----
    Same normalisation :class:`Positions` and :class:`ForwardSlice` apply, for two
    reasons:

    - an ``np.int64`` position must compare equal to the ``int`` spelling of the same
      selection (expected-plan assertions, and plan equality in general);
    - a variant holding a numpy *array* is unhashable, which would silently make the
      enclosing ``Select`` unhashable too.

    Labels pass through unchanged.
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
        ``True`` if every bound resolves without knowing the dim's length. This is the
        guarantee :class:`ForwardSlice` is built on.
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
        :class:`ForwardSlice` for forward, non-negative bounds; :class:`GeneralSlice`
        for any other integer slice; :class:`Label` if a bound isn't an integer at all
        (a label slice, which isn't positional).
    """
    bounds = (s.start, s.stop, s.step)
    if not all(b is None or _is_int(b) for b in bounds):
        return Label(s)  # a label slice (e.g. sel) — not positional
    if _is_forward(s):
        # Convert to plain ``int`` so the field types match their annotations, and a
        # numpy-bounded slice compares equal to the same slice written by hand.
        return ForwardSlice(*(None if b is None else int(b) for b in bounds))
    return GeneralSlice(s)


def classify(value: Any) -> Indexer:
    """Sort a raw ``isel``/``sel`` indexer into its :data:`Indexer` variant.

    Parameters
    ----------
    value : Any
        One dim's indexer, exactly as passed to ``isel``/``sel``: a position, a label, a
        slice, a sequence, or an array.

    Returns
    -------
    Indexer
        The matching variant. Anything that can't be reasoned about positionally becomes
        a :class:`Label`.

    Notes
    -----
    The one place indexer values are classified. Order matters in two places:

    - a boolean sequence is a :class:`Mask`, not :class:`Positions`, even though
      ``bool`` is an ``int`` subclass, so the all-boolean check runs before the
      all-integer one;
    - a 0-d array is a :class:`Scalar`, not a list, so the rank check runs before the
      dtype checks.

    Integer checks use ``_is_int`` throughout, so numpy positions and bounds classify
    the same as their Python equivalents.
    """
    if isinstance(value, slice):
        return _classify_slice(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:  # a 0-d array indexes like the bare value: drops the dim
            return _scalar(
                value.item() if np.issubdtype(value.dtype, np.integer) else value
            )
        if value.dtype == bool:
            # Only a 1-d boolean array is a mask. xarray rejects a higher-rank boolean
            # array as a single-dim indexer ("Unlabeled multi-dimensional array cannot
            # be used for indexing"), so no valid plan contains one. It goes to
            # ``Label`` (the variant for values the optimiser can't reason about), and
            # xarray raises its own error at replay.
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
    if isinstance(value, xr.DataArray | xr.Variable):
        # A DataArray/Variable indexer is xarray's advanced indexing, whose dim effect
        # depends on the dim being indexed (see ``Advanced``). Represent it explicitly.
        # Falling through to the ``_scalar`` catch-all would treat it as a dim-dropping
        # ``Scalar``, which is wrong and unhashable.
        return Advanced(dims=tuple(value.dims))
    return _scalar(value)  # anything else drops its dim
