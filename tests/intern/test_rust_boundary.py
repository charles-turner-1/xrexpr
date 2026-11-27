"""The Rust reader of the interned IR, pinned on cases the plan generator may not reach.

``test_properties.test_rust_reads_every_interned_op`` checks Rust's view of every generated
plan against ``_rust_view``. These pin the exact shape of that view on hand-built records,
and the cases a random plan is unlikely to produce.
"""

from types import MappingProxyType

import numpy as np
import pytest
from frozendict import frozendict

from xrexpr._xrexprs import view_fluent_plan, view_lowered_plan
from xrexpr.chunks import SingleSize
from xrexpr.indexers import Label, Positions, Scalar
from xrexpr.intern import (
    InternedContextOpen,
    InternedProject,
    InternedRechunk,
    InternedRename,
    InternedReduce,
    InternedSelect,
    InternedVal,
)
from xrexpr.ir import ALL_DIMS


def test_view_is_class_name_and_reasoning_fields():
    """A record comes back as ``(class name, fields)``, handles as ints, the header dropped."""
    op = InternedSelect(
        name="isel",
        indexer=frozendict({InternedVal(0): Positions((1, 3))}),
        args=("not", "extracted"),
    )
    assert view_lowered_plan([op]) == [
        (
            "InternedSelect",
            {"name": "isel", "indexer": {0: ("Positions", {"values": [1, 3]})}},
        )
    ]


def test_a_label_scalar_is_read_as_an_unknown_position():
    """``sel(time="2020")``: the label is not an integer, so Rust sees ``position=None``."""
    op = InternedSelect(
        name="sel", indexer=frozendict({InternedVal(0): Scalar("2020")})
    )
    assert view_lowered_plan([op]) == [
        (
            "InternedSelect",
            {"name": "sel", "indexer": {0: ("Scalar", {"position": None})}},
        )
    ]


def test_a_label_keeps_only_its_known_size():
    labels = InternedSelect(
        name="sel", indexer=frozendict({InternedVal(0): Label(np.array(["a", "b"]))})
    )
    [(_, fields)] = view_lowered_plan([labels])
    assert fields["indexer"] == {0: ("Label", {"size": 2})}


def test_all_dims_comes_back_as_the_sentinel():
    [(_, fields)] = view_lowered_plan([InternedReduce(name="mean", consumes=ALL_DIMS)])
    assert fields["consumes"] is ALL_DIMS


def test_a_uniform_rechunk_carries_its_spec():
    op = InternedRechunk(name="chunk", uniform=SingleSize(10))
    [(_, fields)] = view_lowered_plan([op])
    assert fields == {"name": "chunk", "chunks": {}, "uniform": ("SingleSize", 10)}


def test_a_record_is_read_as_its_own_variant_not_a_look_alike():
    """Every record has a ``name``, and that is all ``InternedElementwise`` reads.

    So "the first variant whose fields are readable" would take this ``Project`` for an
    ``Elementwise``. Dispatch is by class name instead.
    """
    op = InternedProject(name="__getitem__", variables=(InternedVal(0),), single=True)
    assert view_lowered_plan([op]) == [
        ("InternedProject", {"name": "__getitem__", "variables": [0], "single": True})
    ]


def test_each_plan_kind_rejects_the_other_kinds_ops():
    """A ``ContextOpen`` is fluent-only; a lowered plan never carries one."""
    opener = InternedContextOpen(name="groupby")
    assert view_fluent_plan([opener]) == [("InternedContextOpen", {"name": "groupby"})]
    with pytest.raises(
        TypeError, match="cannot convert InternedContextOpen to LoweredOp"
    ):
        view_lowered_plan([opener])


def test_a_mapping_need_not_be_a_dict():
    """Mapping fields are read through ``items()``, not as a ``dict``.

    ``frozendict``'s C build (conda-forge's Python 3.10 one) is not a ``dict`` subclass,
    which pyo3's own ``HashMap`` extraction requires. A ``MappingProxyType`` is never a
    ``dict``, so this pins the behaviour whichever ``frozendict`` is installed.
    """
    op = InternedRename(
        name="rename", mapping=MappingProxyType({InternedVal(0): InternedVal(1)})
    )
    assert view_lowered_plan([op]) == [
        ("InternedRename", {"name": "rename", "mapping": {0: 1}})
    ]
