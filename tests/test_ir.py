"""Tests for the expression IR: the ``Op`` sum type (``Reduce``/``Select``/``Scan``/``Opaque``).

``kwargs``/``indexer`` are backed by the third-party ``frozendict``, so we don't
re-test that library's internals — only that each variant coerces to it, stays
frozen/hashable, and that ``Select.consumes`` is *derived* from ``indexer`` (never a
stored field that could drift from it).
"""

import dataclasses

import pytest
from frozendict import frozendict as _pkg_frozendict

from xrexpr.ir import Opaque, Rechunk, Reduce, Scan, Select, frozendict


def test_ir_reexports_third_party_frozendict():
    assert frozendict is _pkg_frozendict


def test_reduce_minimal_defaults():
    node = Reduce(name="mean")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.consumes == frozenset()


def test_select_minimal_defaults():
    node = Select(name="isel")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.indexer == frozendict()
    assert node.consumes == frozenset()


def test_scan_and_opaque_minimal_defaults():
    assert Scan(name="cumsum").args == ()
    assert Opaque(name="where").kwargs == frozendict()


def test_rechunk_minimal_defaults():
    node = Rechunk(name="chunk")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.chunks == frozendict()


def test_rechunk_coerces_containers():
    node = Rechunk(name="chunk", args=[{"time": 100}], chunks={"time": 100})
    assert node.args == ({"time": 100},)
    assert isinstance(node.chunks, frozendict)


def test_reduce_coerces_containers():
    node = Reduce(name="mean", args=["lat"], kwargs={"skipna": True}, consumes=["lat"])
    assert node.args == ("lat",)
    assert isinstance(node.kwargs, frozendict)
    assert node.consumes == frozenset({"lat"})


def test_select_coerces_containers():
    node = Select(name="isel", kwargs={"drop": True}, indexer={"time": 0})
    assert isinstance(node.kwargs, frozendict)
    assert isinstance(node.indexer, frozendict)


def test_select_consumes_is_derived_from_indexer():
    # scalar-indexed dims drop (land in consumes); slice/sequence dims are kept.
    node = Select(name="isel", indexer={"time": 0, "lat": slice(0, 2), "lon": [0, 1]})
    assert node.consumes == frozenset({"time"})


def test_select_consumes_cannot_drift_from_indexer():
    # The desync the flat record risked: a merged select whose indexer re-indexes a
    # dim with a slice reads consumes=∅ off that indexer — it cannot claim the dim was
    # dropped, because consumes has no independent storage.
    node = Select(name="isel", indexer={"time": slice(0, 5)})
    assert node.consumes == frozenset()


def test_select_name_is_literal_typed():
    # Select(name="mean") is a *type* error (name↔kind unrepresentable) — enforced by
    # mypy, not at runtime, so there is nothing to assert here beyond the valid names.
    assert Select(name="isel").name == "isel"
    assert Select(name="sel").name == "sel"


def test_variants_are_frozen():
    node = Reduce(name="mean")
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.name = "sum"


def test_metadata_cannot_be_mutated_in_place():
    node = Select(name="isel", kwargs={"drop": True}, indexer={"time": 0})
    with pytest.raises(TypeError):
        node.kwargs["drop"] = False
    with pytest.raises(TypeError):
        node.indexer["time"] = 1


def test_variants_hashable_and_value_equal():
    a = Reduce(name="mean", args=("lat",), consumes=["lat"])
    b = Reduce(name="mean", args=("lat",), consumes=["lat"])
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_distinct_variants_are_unequal():
    # a Select and an Opaque with the same name are different nodes
    assert Opaque(name="isel") != Select(name="isel")
