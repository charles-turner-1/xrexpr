"""The interner module provides a simple interning mechanism for hashable objects.

We use this to intern names, arguments, and keyword arguments in the expression tree
into ints.

Doing so means we are passing around ints instead of arbitrary hashable objects,
which is 1. more efficient and 2. allows us to pass the IR into a Rust backend
without having to worry about the specifics of how arbitrary Python objects
implement hashing and equality.
"""

from collections.abc import Hashable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)

NameType = Hashable
ArgType = Hashable
KwargType = Hashable
DimType = Hashable


class Interner(Generic[T]):
    """Interner to intern a particular type of object.

    The interner takes eg. a dimensions name and returns a unique handle for that name.
    The handle is guaranteed to be the same for the same name, and different for different names.
    We do this basically by adding all new items to a dictionary
    """

    def __init__(self) -> None:
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


name_interner = Interner[NameType]()
arg_interner = Interner[ArgType]()
kwarg_interner = Interner[KwargType]()
dim_interner = Interner[DimType]()
