"""Taking an interned plan across the Rust boundary and back, without losing anything.

Rust can reason about an interned record's semantic fields, but it can't hold arbitrary
Python objects. So before a plan crosses, every :data:`~xrexpr.payload.Payload` field is
swapped for a :class:`~xrexpr.payload.PayloadRef` into a :class:`PayloadTable`, and after
it comes back the refs are swapped back. Rust carries the refs but can't look inside
them (on the Rust side the ref's field is private).

What counts as a payload:

- every field marked ``Payload`` (a record's ``args``/``kwargs`` and
  ``reduce_args``/``reduce_kwargs``, ``Label.value``, ``OpaqueChunk.value``);
- a ``Scalar``'s value when it's a *label*. A ``Scalar`` is half and half: an integer
  position is semantic (the optimiser composes it), a label is a payload. Rust's
  ``Scalar`` is ``Position(i64) | Label(PayloadRef)`` to match.

The round trip is::

    table = PayloadTable()
    crossed = [to_rust(op, table) for op in interned_ops]
    ...  # Rust extracts, reasons, and hands back ``(class name, fields)`` views
    rebuilt = [from_rust(view, table) for view in views]

and ``rebuilt == interned_ops`` when Rust changed nothing. A table belongs to one
crossing: make a new one each time, and drop it once the plan is back.
"""

import copy
import dataclasses
from collections.abc import Hashable, Mapping
from functools import cache
from typing import Any, get_args, get_type_hints

from frozendict import frozendict

from xrexpr import chunks, indexers
from xrexpr._xrexprs.ir import AllDims
from xrexpr.intern import ir as interned
from xrexpr.intern.interner import InternedVal
from xrexpr.payload import PayloadRef, _Payload

__all__ = ["PayloadTable", "from_rust", "to_rust"]


class PayloadTable:
    """The payloads of one crossing, each reachable by its :class:`PayloadRef`.

    Notes
    -----
    A plain list, indexed by ``PayloadRef.index``. Payloads are deduplicated by *identity*
    (``id``), not equality, because many of them can't be compared or hashed (arrays,
    dicts). The table holds a reference to every payload, so an ``id`` can't be reused
    while the table is alive.
    """

    def __init__(self) -> None:
        self._items: list[Any] = []
        self._by_id: dict[int, int] = {}

    def put(self, value: Any) -> PayloadRef:
        """Store ``value`` (or find it, if it's already stored) and return its ref.

        Parameters
        ----------
        value : Any
            The payload.

        Returns
        -------
        PayloadRef
            The handle to pass to Rust in its place.
        """
        index = self._by_id.get(id(value))
        if index is None:
            index = len(self._items)
            self._items.append(value)
            self._by_id[id(value)] = index
        return PayloadRef(index)

    def __getitem__(self, ref: PayloadRef) -> Any:
        """Return the payload ``ref`` stands for.

        Parameters
        ----------
        ref : PayloadRef
            A ref this table handed out.

        Returns
        -------
        Any
            The original object (the same object, not a copy).
        """
        return self._items[ref.index]

    def __len__(self) -> int:
        """Return how many distinct payloads are stored.

        Returns
        -------
        int
            The number of payloads.
        """
        return len(self._items)


@cache
def _payload_fields(cls: type) -> frozenset[str]:
    """Return the names of ``cls``'s fields marked ``Payload``.

    Parameters
    ----------
    cls : type
        A dataclass.

    Returns
    -------
    frozenset of str
        The payload field names.
    """
    hints = get_type_hints(cls, include_extras=True)
    return frozenset(
        f.name
        for f in dataclasses.fields(cls)
        if _Payload in getattr(hints[f.name], "__metadata__", ())
    )


#: Every class a Rust view can name, by class name: the interned records and the value
#: types inside them.
_CLASSES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        *get_args(interned.InternedFluentOp),
        *get_args(interned.InternedLoweredOp),
        *get_args(indexers.Indexer),
        *get_args(chunks.ChunkSpec),
    )
}


def to_rust(value: Any, table: PayloadTable) -> Any:
    """Return ``value`` with every payload inside it swapped for a :class:`PayloadRef`.

    Parameters
    ----------
    value : Any
        An interned record, or any value inside one.
    table : PayloadTable
        This crossing's table. The payloads are stored in it.

    Returns
    -------
    Any
        A copy of ``value`` that Rust can extract. ``value`` itself is unchanged.

    Notes
    -----
    Copies are made with ``copy.copy`` and ``object.__setattr__``, which skips
    ``__post_init__``. That matters for ``Label``: rebuilding it would recompute
    ``known_size`` from the ref (giving ``None``), and ``known_size`` is what Rust reads.
    """
    if isinstance(value, InternedVal | PayloadRef | AllDims):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payloads = _payload_fields(type(value))
        changes = {}
        for f in dataclasses.fields(value):
            if not f.init:
                continue
            current = getattr(value, f.name)
            if f.name in payloads:
                changes[f.name] = table.put(current)
            else:
                changes[f.name] = to_rust(current, table)
        if isinstance(value, indexers.Scalar) and value.position is None:
            changes["value"] = table.put(value.value)
        crossed = copy.copy(value)
        for name, new in changes.items():
            object.__setattr__(crossed, name, new)
        return crossed
    if isinstance(value, Mapping):
        return frozendict({k: to_rust(v, table) for k, v in value.items()})
    if isinstance(value, tuple):
        return tuple(to_rust(v, table) for v in value)
    if isinstance(value, frozenset):
        return frozenset(to_rust(v, table) for v in value)
    return value


def from_rust(view: Any, table: PayloadTable) -> Any:
    """Rebuild a value from what Rust hands back, swapping refs for their payloads.

    Parameters
    ----------
    view : Any
        A record or value as Rust converts it to Python: ``(class name, {field: value})``
        for a record or value type, a list for a sequence, a dict for a mapping, a set for
        a set, a :class:`PayloadRef` for a payload.
    table : PayloadTable
        The table the plan was crossed with.

    Returns
    -------
    Any
        The rebuilt value, with containers frozen the way the Python types expect: lists
        become tuples, dicts become frozendicts, sets become frozensets.

    Raises
    ------
    ValueError
        If a view names an unknown class, or its fields don't match that class's
        constructor fields. Rust may include derived fields (``init=False``), which are
        ignored and recomputed by the constructor.
    """
    if isinstance(view, PayloadRef):
        return table[view]
    if isinstance(view, InternedVal | AllDims):
        return view
    if isinstance(view, tuple):
        name, fields = view
        cls = _CLASSES.get(name)
        if cls is None:
            raise ValueError(f"Rust returned an unknown class {name!r}")
        init = {f.name for f in dataclasses.fields(cls) if f.init}
        derived = {f.name for f in dataclasses.fields(cls) if not f.init}
        if not init <= fields.keys() <= init | derived:
            raise ValueError(
                f"Rust's {name} has fields {sorted(fields)}, expected {sorted(init)}"
            )
        return cls(**{k: from_rust(fields[k], table) for k in init})
    if isinstance(view, dict):
        return frozendict(
            {
                _hashable(from_rust(k, table)): from_rust(v, table)
                for k, v in view.items()
            }
        )
    if isinstance(view, list):
        return tuple(from_rust(v, table) for v in view)
    if isinstance(view, set | frozenset):
        return frozenset(from_rust(v, table) for v in view)
    return view


def _hashable(key: Any) -> Hashable:
    """Return ``key``, typed as a mapping key.

    Parameters
    ----------
    key : Any
        A rebuilt mapping key. Rust only uses handles and strings as keys.

    Returns
    -------
    Hashable
        The same object.
    """
    assert isinstance(key, Hashable)
    return key
