"""The *value* sum type: a closed taxonomy over what a single ``isel``/``sel`` indexer is.

``Select.indexer`` maps each dim to one indexer *value*, and that value was historically
typed ``Any`` — a sum type in disguise. A value is exactly one of six shapes, and three
separate call sites used to re-derive that taxonomy by hand (a negated ``isinstance`` in
``ir.py``, an ``isinstance`` ladder in ``optimize.py``, and an independent walk in
``schema.py``). This module makes the taxonomy a *type*, so the classification lives in one
place (:func:`classify`) and the facts that follow from it become methods rather than
re-decisions:

- :attr:`drops_dim` — does this indexer remove its dim (a scalar) or keep it? (``ir.py``'s
  ``Select.consumes``.)
- :meth:`size` — the new length of a *kept* dim under this indexer. (``schema.py``'s old
  ``_indexer_size``.)
- :meth:`to_raw` — the exact xarray-facing value to hand back to replay when a rewrite
  rebuilds a node's ``args`` from its ``indexer``.

Whether two indexers *compose* (the ``optimize.py`` concern) is deliberately **not** modelled
here: composition is a policy the optimiser chooses to prove, not an intrinsic fact of a value,
so it stays a ``match`` in ``optimize.py``. What this module guarantees is the discriminant it
matches on.

``ForwardSlice`` earns its own variant so the "forward, non-negative bounds" carve-out that
the composer needs is a *constructor invariant* — a ``ForwardSlice`` cannot be built with a
negative bound — rather than a guard re-run at every call site. ``Label`` is the value layer's
escape hatch: a ``sel`` coordinate label (a string, timestamp, tuple key, label slice, or
label sequence) is genuinely open and cannot be reasoned about positionally — the same role
``Opaque`` plays for op kinds.
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
    """A single position or label (``isel(time=0)``, ``sel(time="2020")``) — *drops* the dim."""

    value: Any
    drops_dim: ClassVar[bool] = True

    def size(self, current: int) -> int:
        raise AssertionError("a scalar indexer drops its dim; its size is undefined")

    def to_raw(self) -> Any:
        return self.value


@dataclass(frozen=True)
class ForwardSlice:
    """A forward, non-negative integer slice (``isel(time=slice(0, 5))``) — composable.

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
        return len(range(*self.to_raw().indices(current)))

    def to_raw(self) -> slice:
        return slice(self.start, self.stop, self.step)


@dataclass(frozen=True)
class GeneralSlice:
    """An integer slice with a negative bound or reversed step (``isel(time=slice(-3, None))``).

    Sizable — ``slice.indices`` resolves it against a known dim length — but *not* composable,
    since a negative bound counts from the end, which the composer doesn't carry.
    """

    value: slice
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        return len(range(*self.value.indices(current)))

    def to_raw(self) -> slice:
        return self.value


@dataclass(frozen=True)
class Positions:
    """A concrete enumeration of integer positions (``isel(time=[0, 2, 4])``) — composable."""

    values: tuple[int, ...]
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        return len(self.values)

    def to_raw(self) -> list[int]:
        return list(self.values)


@dataclass(frozen=True)
class Mask:
    """A boolean mask (``isel(time=[True, False, ...])`` or a bool array) — sizes by True count."""

    values: Any
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        if isinstance(self.values, np.ndarray):
            return int(self.values.sum())
        return int(sum(bool(x) for x in self.values))

    def to_raw(self) -> Any:
        return self.values


@dataclass(frozen=True)
class Label:
    """A coordinate label, label slice, or label sequence (``sel(time="2020")``) — irreducibly open.

    The value-layer counterpart of :class:`~xrexpr.ir.Opaque`: it keeps its dim but the
    optimiser can't reason about it positionally (no length, no composition), so a label slice
    conservatively keeps the current size while a label sequence sizes by its length.
    """

    value: Any
    drops_dim: ClassVar[bool] = False

    def size(self, current: int) -> int:
        if isinstance(self.value, np.ndarray):
            return int(self.value.size)
        if isinstance(self.value, (list | tuple)):
            return len(self.value)
        return current  # a label slice needs coord values to size — leave it unchanged

    def to_raw(self) -> Any:
        return self.value


#: One dim's indexer, as a closed sum. ``match`` over this binds the shape the optimiser and
#: schema layers reason about; :func:`classify` is the sole constructor from raw values.
Indexer = Scalar | ForwardSlice | GeneralSlice | Positions | Mask | Label


def _is_int(x: Any) -> bool:
    """Whether ``x`` is an integer *position*, numpy-typed or not.

    ``isinstance(x, int)`` is too narrow: numpy integers fall out of ``argmin``, ``np.where``
    and ``arr.values[i]`` routinely, and a ``np.int64`` bound misfiled as a label makes its
    slice both mis-sized and uncomposable. ``numbers.Integral`` covers every numpy width;
    ``bool`` is excluded because it is an ``int`` subclass but means a *mask*, not a position.
    (``np.bool_`` is not ``Integral``, so it needs no exclusion.)
    """
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_forward(s: slice) -> bool:
    """Whether ``s`` steps forward from non-negative integer bounds (no dim length needed)."""
    if s.step is not None and (not _is_int(s.step) or s.step < 1):
        return False
    return all(b is None or (_is_int(b) and b >= 0) for b in (s.start, s.stop))


def _classify_slice(s: slice) -> ForwardSlice | GeneralSlice | Label:
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

    The single place the value taxonomy is decided. Order matters twice: a boolean sequence is a
    :class:`Mask`, not :class:`Positions`, even though ``bool`` is an ``int`` subclass, so the
    all-boolean test runs before the all-integer one; and a 0-d array is a :class:`Scalar`
    rather than an enumeration, so the rank check runs before the dtype dispatch.

    Integer-ness is decided by :func:`_is_int` throughout, so numpy-typed positions and bounds
    classify the same way their Python equivalents do.
    """
    if isinstance(value, slice):
        return _classify_slice(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return Scalar(
                value
            )  # a 0-d array indexes like the bare value: drops the dim
        if value.dtype == bool:
            return Mask(value)
        if np.issubdtype(value.dtype, np.integer):
            return Positions(tuple(int(x) for x in value.tolist()))
        return Label(value)
    if isinstance(value, (list | tuple)):
        if value and all(isinstance(x, (bool | np.bool_)) for x in value):
            return Mask(value)
        if all(_is_int(x) for x in value):  # pure-bool already handled above
            return Positions(tuple(int(x) for x in value))
        return Label(value)  # a label sequence, or a mixed one
    return Scalar(value)  # anything else drops its dim
