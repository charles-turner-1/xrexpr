"""The ``Payload`` marker: a field the Rust reasoner carries but must never read or change.

A plan crossing to Rust holds two kinds of data. *Semantic* fields — the dims a reduce
consumes, a select's indexer, a rechunk's chunks — are what the reasoner reasons about and
may rewrite. *Payload* fields are everything else Python needs to replay the plan: the
option kwargs of a call (``skipna``, ``drop``), the labels of a ``sel``, a chunk spec the
taxonomy doesn't model. The reasoner has no business inside them, and replay is only
faithful if they come back exactly as they went in.

That is safe because a replayed call's header is *derived* from the semantic fields
(``lower._emit_node``); only the payloads are copied verbatim. So a rule may rewrite a
semantic field and the call follows, while a payload just has to be carried through.

Marking a field ``Payload[T]`` states that contract. It lives in its own module, rather than
beside the interned records, because the value types in :mod:`xrexpr.indexers` and
:mod:`xrexpr.chunks` carry payloads too and :mod:`xrexpr.intern.ir` already imports them.
"""

from typing import Annotated, TypeVar

__all__ = ["Payload"]

T = TypeVar("T")


class _Payload:
    """The ``Annotated`` metadata tag that marks a field as a payload."""


#: A field the Rust reasoner carries but never reads or changes. See the module docstring.
Payload = Annotated[T, _Payload]
