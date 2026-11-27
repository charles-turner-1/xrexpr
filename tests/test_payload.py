"""Which fields are ``Payload``: the ones the Rust reasoner carries but never reads.

The marker is the contract the Rust boundary is built against -- every field it lists
crosses as an opaque handle, every other field as data a rule may reason about. These pin
the list, so a field moving between the two kinds is a deliberate change.
"""

import dataclasses
from typing import get_args, get_type_hints

import pytest

from xrexpr import chunks, indexers
from xrexpr.intern import ir as interned
from xrexpr.payload import _Payload

#: The replay header: a call's verbatim ``args``/``kwargs``, and a fused node's second call.
HEADER = {"args", "kwargs", "reduce_args", "reduce_kwargs"}


def _payload_fields(cls: type) -> set[str]:
    hints = get_type_hints(cls, include_extras=True)
    return {
        f.name
        for f in dataclasses.fields(cls)
        if _Payload in getattr(hints[f.name], "__metadata__", ())
    }


@pytest.mark.parametrize(
    "record",
    get_args(interned.InternedFluentOp) + get_args(interned.InternedLoweredOp),
    ids=lambda cls: cls.__name__,
)
def test_an_interned_record_marks_exactly_its_replay_header(record):
    header = {f.name for f in dataclasses.fields(record)} & HEADER
    assert header >= {"args", "kwargs"}
    assert _payload_fields(record) == header


@pytest.mark.parametrize(
    "value_type",
    get_args(indexers.Indexer) + get_args(chunks.ChunkSpec),
    ids=lambda cls: cls.__name__,
)
def test_only_labels_and_opaque_chunks_carry_a_payload_value(value_type):
    """A ``Scalar``'s value is a payload only when it is a label, so it is not marked."""
    expected = (
        {"value"} if value_type in (indexers.Label, chunks.OpaqueChunk) else set()
    )
    assert _payload_fields(value_type) == expected
