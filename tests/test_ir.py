"""Tests for the expression IR (PR 2): ``frozendict`` and ``OpNode``.

The type has no callers yet, so these cover it directly: the Mapping interface,
immutability, hashing, and OpNode's construction/coercion/validation.
"""

import dataclasses

import pytest

from xrexpr.ir import KINDS, OpNode, frozendict


# --- frozendict --------------------------------------------------------------


def test_frozendict_mapping_interface():
    fd = frozendict(a=1, b=2)
    assert fd["a"] == 1
    assert len(fd) == 2
    assert set(fd) == {"a", "b"}
    assert "a" in fd and "z" not in fd
    assert dict(fd.items()) == {"a": 1, "b": 2}
    assert fd.get("z", 9) == 9


def test_frozendict_constructs_from_mapping_and_pairs():
    assert frozendict({"a": 1}) == frozendict([("a", 1)]) == frozendict(a=1)


def test_frozendict_equals_plain_dict():
    assert frozendict(a=1) == {"a": 1}
    assert frozendict(a=1) != {"a": 2}


def test_frozendict_is_immutable():
    fd = frozendict(a=1)
    with pytest.raises(TypeError):
        fd["a"] = 2
    with pytest.raises(TypeError):
        del fd["a"]


def test_frozendict_hashable_and_cached():
    fd = frozendict(a=1, b=2)
    first = hash(fd)
    assert first == hash(fd)  # exercises the cached branch
    assert first == hash(frozendict(b=2, a=1))  # order-independent
    assert {fd, frozendict(a=1, b=2)} == {fd}  # usable as a set member


def test_frozendict_repr_roundtrips_contents():
    assert repr(frozendict(a=1)) == "frozendict({'a': 1})"


# --- OpNode ------------------------------------------------------------------


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
    node = OpNode(name="isel", kind="select", kwargs={"drop": True}, indexer={"time": 0})
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
