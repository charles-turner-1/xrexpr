"""The interner module provides a simple interning mechanism for hashable objects.

We use this to intern names, arguments, and keyword arguments in the expression tree
into ints.

Doing so means we are passing around ints instead of arbitrary hashable objects,
which is 1. more efficient and 2. allows us to pass the IR into a Rust backend
without having to worry about the specifics of how arbitrary Python objects
implement hashing and equality.
"""

from collections.abc import Hashable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, final

T = TypeVar("T", bound=Hashable)


@final
@dataclass(frozen=True)
class InternedVal:
    """A name relabeled to an interner handle — a *distinct type* from a literal ``int``.

    Interning turns dim and variable *names* into handles, but a plan is also full of
    literal ``int`` values that are **not** names — a scalar index position, a chunk block
    size, a slice bound. Wrapping a handle keeps the two apart: an :class:`InternedVal` is
    always a name to look up, a bare ``int`` is always a value to read. That distinction is
    what lets interning stay *selective* (relabel names, leave values alone) and lets a Rust
    reader extract each into the right kind of struct field.
    """

    handle: int


class Interner(Generic[T]):
    """Interner to intern a particular type of object.

    The interner takes eg. a dimensions name and returns a unique handle for that name.
    The handle is guaranteed to be the same for the same name, and different for different names.
    We do this basically by adding all new items to a dictionary.

    The generic type here basically does nothing, but can be handle to let the
    type checker know that we are interning a particular type of object, and not just any hashable object.

    Singleton so that we have one interner everywhere. This is unlikely to ever
    get so long we have a problem, and stops us threading interners through everywhere.
    """

    _instance: ClassVar["Interner[Any] | None"] = None

    def __new__(cls) -> "Interner[T]":
        """Ensure that only one instance of the interner exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the interner with empty forward and reverse mappings.

        Guarded so that re-constructing the singleton does not wipe its tables:
        ``__new__`` sets ``_instance`` *before* ``__init__`` runs, so we key the guard on
        whether *this instance* has been initialized, not on ``_instance``.
        """
        if hasattr(self, "initialised"):
            return None
        self.initialised = True
        self.forward: dict[T, int] = {}
        self.reverse: list[T] = []

    def __call__(self, item: T) -> int:
        """Return a unique handle for the given item, interning it if necessary."""
        if item not in self.forward:
            self.forward[item] = len(self.reverse)
            self.reverse.append(item)
        return self.forward[item]

    def __getitem__(self, handle: int) -> T:
        """Return the item corresponding to the given handle."""
        return self.reverse[handle]

    def __len__(self) -> int:
        """Return the number of items interned."""
        return len(self.reverse)

    def __contains__(self, item: T) -> bool:
        """Return whether the given item is interned."""
        return item in self.forward

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the interned items."""
        return iter(self.reverse)

    def _clear(self) -> None:
        """Clear the interner. This is mainly for testing purposes."""
        self.forward = {}
        self.reverse = []
        Interner._instance = None
