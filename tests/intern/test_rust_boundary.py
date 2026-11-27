"""The Rust boundary, on hand-built records and the cases a random plan may not reach.

``test_properties.test_a_plan_crosses_the_rust_boundary_and_back_unchanged`` checks the
full round trip on every generated plan. These pin the details: what Rust hands back
before it's rebuilt, that payloads come back as the *same* objects, and the errors for
things that mustn't cross.
"""

from types import MappingProxyType

import numpy as np
import pytest
from frozendict import frozendict

from xrexpr._xrexprs import roundtrip_fluent_plan, roundtrip_lowered_plan
from xrexpr.chunks import OpaqueChunk, SingleSize
from xrexpr.indexers import GeneralSlice, Label, Positions, Scalar
from xrexpr.intern import (
    InternedContextOpen,
    InternedElementwise,
    InternedProject,
    InternedRechunk,
    InternedReduce,
    InternedRename,
    InternedSelect,
    InternedVal,
)
from xrexpr.intern.crossing import PayloadTable, from_rust, to_rust
from xrexpr.ir import ALL_DIMS
from xrexpr.payload import PayloadRef


def _roundtrip(op, lowered=True):
    """Send one interned op across and rebuild it.

    Parameters
    ----------
    op : InternedFluentOp or InternedLoweredOp
        The op to send.
    lowered : bool, default True
        Whether to send it as part of a lowered plan (``False``: a fluent plan).

    Returns
    -------
    tuple
        ``(rebuilt op, Rust's raw view of it, the payload table)``.
    """
    table = PayloadTable()
    roundtrip = roundtrip_lowered_plan if lowered else roundtrip_fluent_plan
    [view] = roundtrip([to_rust(op, table)])
    return from_rust(view, table), view, table


def test_rust_hands_back_class_name_and_every_field():
    """Rust returns ``(class name, fields)`` with every field, payloads as refs."""
    op = InternedSelect(
        name="isel",
        indexer=frozendict({InternedVal(0): Positions((1, 3))}),
        args=({"time": [1, 3]},),
    )
    rebuilt, view, _ = _roundtrip(op)
    name, fields = view
    assert name == "InternedSelect"
    assert fields["indexer"] == {InternedVal(0): ("Positions", {"values": [1, 3]})}
    assert isinstance(fields["args"], PayloadRef)
    assert isinstance(fields["kwargs"], PayloadRef)
    assert rebuilt == op


def test_payloads_come_back_as_the_same_objects():
    """A payload isn't copied or rebuilt: the very object that was sent comes back."""
    args = (np.arange(3),)
    op = InternedReduce(name="mean", consumes=frozenset({InternedVal(0)}), args=args)
    rebuilt, _, _ = _roundtrip(op)
    assert rebuilt.args is args


def test_a_label_scalar_crosses_as_a_payload():
    """``sel(time="2020")``: the label goes as a ref, and comes back as the same string."""
    label = "2020"
    op = InternedSelect(name="sel", indexer=frozendict({InternedVal(0): Scalar(label)}))
    rebuilt, view, _ = _roundtrip(op)
    [(_, scalar_view)] = view[1]["indexer"].items()
    assert scalar_view[0] == "Scalar"
    assert isinstance(scalar_view[1]["value"], PayloadRef)
    assert rebuilt.indexer[InternedVal(0)].value is label


def test_a_position_scalar_crosses_as_data():
    """``isel(time=3)``: an integer position is semantic, so Rust gets the int itself."""
    op = InternedSelect(name="isel", indexer=frozendict({InternedVal(0): Scalar(3)}))
    rebuilt, view, _ = _roundtrip(op)
    assert view[1]["indexer"][InternedVal(0)] == ("Scalar", {"value": 3})
    assert rebuilt == op


def test_a_label_keeps_its_size_while_its_labels_are_a_payload():
    """Rust sees a ``Label``'s ``known_size`` but holds only a ref to the labels."""
    labels = np.array(["a", "b"])
    op = InternedSelect(name="sel", indexer=frozendict({InternedVal(0): Label(labels)}))
    rebuilt, view, _ = _roundtrip(op)
    name, fields = view[1]["indexer"][InternedVal(0)]
    assert name == "Label"
    assert fields["known_size"] == 2
    assert isinstance(fields["value"], PayloadRef)
    assert rebuilt.indexer[InternedVal(0)].value is labels


def test_a_general_slice_comes_back_as_a_slice():
    """``GeneralSlice.value`` is a Python ``slice``, and Rust converts it back to one."""
    op = InternedSelect(
        name="isel",
        indexer=frozendict({InternedVal(0): GeneralSlice(slice(-3, None, -1))}),
    )
    rebuilt, _, _ = _roundtrip(op)
    assert rebuilt == op


def test_an_opaque_chunk_keeps_its_value_as_a_payload():
    """An unmodelled chunk spec crosses as a ref and comes back as the same object."""
    spec = {1, 2}  # a set: something nothing models, and unhashable
    op = InternedRechunk(
        name="chunk", chunks=frozendict({InternedVal(0): OpaqueChunk(spec)})
    )
    rebuilt, _, _ = _roundtrip(op)
    assert rebuilt.chunks[InternedVal(0)].value is spec


def test_all_dims_comes_back_as_the_sentinel():
    """``ALL_DIMS`` goes back as the singleton itself."""
    rebuilt, view, _ = _roundtrip(InternedReduce(name="mean", consumes=ALL_DIMS))
    assert view[1]["consumes"] is ALL_DIMS
    assert rebuilt.consumes is ALL_DIMS


def test_a_uniform_rechunk_carries_its_spec():
    """``chunk(10)`` keeps its uniform spec across the boundary."""
    op = InternedRechunk(name="chunk", uniform=SingleSize(10))
    rebuilt, view, _ = _roundtrip(op)
    assert view[1]["uniform"] == ("SingleSize", {"size": 10})
    assert rebuilt == op


def test_a_record_is_read_as_its_own_variant_not_a_look_alike():
    """Every record has a ``name``, and that's all ``InternedElementwise`` reads.

    So "the first variant whose fields can be read" would take this ``Project`` for an
    ``Elementwise``. Dispatch is by class name instead.
    """
    op = InternedProject(name="__getitem__", variables=(InternedVal(0),), single=True)
    rebuilt, view, _ = _roundtrip(op)
    assert view[0] == "InternedProject"
    assert rebuilt == op


def test_each_plan_kind_rejects_the_other_kinds_ops():
    """A ``ContextOpen`` is fluent-only; a lowered plan never has one."""
    opener = InternedContextOpen(name="groupby")
    rebuilt, _, _ = _roundtrip(opener, lowered=False)
    assert rebuilt == opener
    with pytest.raises(
        TypeError, match="cannot convert InternedContextOpen to LoweredOp"
    ):
        _roundtrip(opener)


def test_a_mapping_need_not_be_a_dict():
    """Mapping fields are read through ``items()``, not as a ``dict``.

    ``frozendict``'s C build (conda-forge's Python 3.10 one) isn't a ``dict`` subclass,
    and pyo3's own ``HashMap`` extraction requires one. A ``MappingProxyType`` is never a
    ``dict``, so this checks it whichever ``frozendict`` is installed.
    """
    op = InternedRename(
        name="rename", mapping=MappingProxyType({InternedVal(0): InternedVal(1)})
    )
    rebuilt, _, _ = _roundtrip(op)
    assert rebuilt.mapping == {InternedVal(0): InternedVal(1)}


def test_a_payload_that_was_not_swapped_out_is_rejected():
    """Sending a record without ``to_rust`` fails, rather than Rust misreading a payload."""
    op = InternedElementwise(name="fillna", args=(0,))
    with pytest.raises(
        TypeError, match="failed to extract field InternedElementwise.args"
    ):
        roundtrip_lowered_plan([op])
