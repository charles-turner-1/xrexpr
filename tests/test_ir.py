"""Tests for the expression IR (PR 2): the ``OpNode`` record type.

``kwargs``/``indexer`` are backed by the third-party ``frozendict``, so we don't
re-test that library's internals — only that ``OpNode`` coerces to it, validates
its ``kind``, and stays frozen/hashable.
"""

import dataclasses

import pytest
from frozendict import frozendict as _pkg_frozendict

from xrexpr.ir import KINDS, OpNode, frozendict


def test_ir_reexports_third_party_frozendict():
    assert frozendict is _pkg_frozendict


def test_opnode_minimal_defaults():
    node = OpNode(name="mean", kind="reduce")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.consumes == frozenset()
    assert node.indexer == frozendict()


def test_opnode_coerces_containers():
    node = OpNode(
        name="isel",
        kind="select",
        args=["temperature"],
        kwargs={"drop": True},
        consumes=["lat"],
        indexer={"time": 0},
    )
    assert node.args == ("temperature",)
    assert isinstance(node.kwargs, frozendict)
    assert isinstance(node.indexer, frozendict)
    assert node.consumes == frozenset({"lat"})


def test_opnode_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown op kind"):
        OpNode(name="mean", kind="nonsense")


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_opnode_accepts_every_known_kind(kind):
    assert OpNode(name="op", kind=kind).kind == kind


def test_opnode_is_frozen():
    node = OpNode(name="mean", kind="reduce")
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.name = "sum"


def test_opnode_metadata_cannot_be_mutated_in_place():
    node = OpNode(
        name="isel", kind="select", kwargs={"drop": True}, indexer={"time": 0}
    )
    with pytest.raises(TypeError):
        node.kwargs["drop"] = False
    with pytest.raises(TypeError):
        node.indexer["time"] = 1


def test_opnode_hashable_and_value_equal():
    a = OpNode(name="mean", kind="reduce", args=("lat",), consumes=["lat"])
    b = OpNode(name="mean", kind="reduce", args=("lat",), consumes=["lat"])
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}
