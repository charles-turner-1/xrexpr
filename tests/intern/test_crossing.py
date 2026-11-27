"""The Python half of the Rust boundary: swapping payloads out and back in.

These don't need Rust. ``to_rust`` is checked on its output directly, and ``from_rust`` on
hand-built views in the ``(class name, fields)`` shape Rust hands back. The full round
trip through Rust is tested in ``test_rust_boundary.py`` and ``test_properties.py``.
"""

import numpy as np
import pytest
from frozendict import frozendict

from xrexpr.chunks import OpaqueChunk
from xrexpr.indexers import Label, Positions, Scalar
from xrexpr.intern import InternedReduce, InternedRechunk, InternedSelect, InternedVal
from xrexpr.intern.crossing import PayloadTable, from_rust, to_rust
from xrexpr.ir import ALL_DIMS
from xrexpr.payload import PayloadRef


def test_to_rust_swaps_every_payload_for_a_ref():
    """``args``/``kwargs`` become refs, and the table holds the originals."""
    args = (np.arange(3),)
    op = InternedReduce(name="mean", consumes=frozenset({InternedVal(0)}), args=args)
    table = PayloadTable()
    crossed = to_rust(op, table)
    assert isinstance(crossed.args, PayloadRef)
    assert isinstance(crossed.kwargs, PayloadRef)
    assert table[crossed.args] is args
    assert crossed.consumes == op.consumes  # semantic fields are left alone


def test_to_rust_leaves_the_original_unchanged():
    """``to_rust`` copies. The record that was passed in still holds its payloads."""
    args = ({"time": [1, 3]},)
    op = InternedSelect(
        name="isel", indexer=frozendict({InternedVal(0): Positions((1, 3))}), args=args
    )
    to_rust(op, PayloadTable())
    assert op.args is args


def test_a_label_scalar_is_a_payload_and_a_position_is_not():
    """A ``Scalar``'s value is swapped only when it's a label."""
    table = PayloadTable()
    label = to_rust(Scalar("2020"), table)
    position = to_rust(Scalar(3), table)
    assert isinstance(label.value, PayloadRef)
    assert table[label.value] == "2020"
    assert position.value == 3


def test_a_label_keeps_its_size_after_its_value_is_swapped():
    """``known_size`` is stored, so it's still there when ``value`` is a ref."""
    crossed = to_rust(Label(np.array(["a", "b", "c"])), PayloadTable())
    assert isinstance(crossed.value, PayloadRef)
    assert crossed.known_size == 3


def test_an_opaque_chunk_value_is_a_payload():
    """An unmodelled chunk spec is carried, never read."""
    spec = {1, 2}
    op = InternedRechunk(
        name="chunk", chunks=frozendict({InternedVal(0): OpaqueChunk(spec)})
    )
    table = PayloadTable()
    crossed = to_rust(op, table)
    assert table[crossed.chunks[InternedVal(0)].value] is spec


def test_from_rust_rebuilds_a_record_and_returns_its_payloads():
    """A view rebuilds into the record: lists become tuples, refs become the originals."""
    args = ({"time": [1, 3]},)
    table = PayloadTable()
    view = (
        "InternedSelect",
        {
            "name": "isel",
            "indexer": {InternedVal(0): ("Positions", {"values": [1, 3]})},
            "args": table.put(args),
            "kwargs": table.put(frozendict()),
        },
    )
    rebuilt = from_rust(view, table)
    assert rebuilt == InternedSelect(
        name="isel", indexer=frozendict({InternedVal(0): Positions((1, 3))}), args=args
    )
    assert rebuilt.args is args


def test_from_rust_ignores_derived_fields_and_recomputes_them():
    """Rust sends ``known_size`` back too. The constructor recomputes it."""
    labels = np.array(["a", "b"])
    table = PayloadTable()
    view = ("Label", {"value": table.put(labels), "known_size": 2})
    rebuilt = from_rust(view, table)
    assert rebuilt.value is labels
    assert rebuilt.known_size == 2


def test_from_rust_passes_all_dims_through():
    """``ALL_DIMS`` comes back as the singleton."""
    assert from_rust(ALL_DIMS, PayloadTable()) is ALL_DIMS


def test_a_table_stores_each_payload_once():
    """The table deduplicates by identity: the same object twice is one entry."""
    shared = [1, 2]
    table = PayloadTable()
    assert table.put(shared) == table.put(shared)
    assert table.put([1, 2]) != table.put(shared)  # equal but distinct: a new entry
    assert len(table) == 2


def test_from_rust_rejects_a_record_with_missing_fields():
    """A view missing one of the class's constructor fields is an error, not a default."""
    with pytest.raises(ValueError, match="expected"):
        from_rust(("InternedElementwise", {"name": "fillna"}), PayloadTable())


def test_from_rust_rejects_an_unknown_class():
    """A class name that isn't a record or value type is an error."""
    with pytest.raises(ValueError, match="unknown class"):
        from_rust(("NotARecord", {}), PayloadTable())
