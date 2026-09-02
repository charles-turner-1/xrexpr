"""Interning is *selective*: names become handles, values and structure stay put.

The round-trip is pinned by ``test_properties.test_roundtrip_intern_ops_unchanged`` over
generated plans; this file pins the *shape* of an interned node — which atoms are relabeled
to :class:`~xrexpr.intern.interner.InternedVal` and which are left exactly as they are — since that
is the contract the Rust reader extracts against.
"""

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.chunks import SingleSize
from xrexpr.indexers import ForwardSlice, Positions, Scalar
from xrexpr.intern import (
    InternedProject,
    InternedReduce,
    InternedSelect,
    InternedVal,
    Interner,
    deintern,
    intern,
)
from xrexpr.ir import ALL_DIMS, Project, Rechunk, Reduce, Select


@pytest.fixture(autouse=True)
def _clear_interner():
    """The interner is a process-wide singleton; reset it before each test."""
    Interner()._clear()
    yield


def test_names_relabel_to_interned_val():
    r = intern(Reduce(name="sum", consumes=frozenset({"time", "lat"})))
    assert isinstance(r, InternedReduce)
    assert all(isinstance(h, InternedVal) for h in r.consumes)

    p = intern(
        Project(name="__getitem__", args=(["tas", "pr"],), variables=("tas", "pr"))
    )
    assert isinstance(p, InternedProject)
    assert all(isinstance(h, InternedVal) for h in p.variables)


def test_all_dims_sentinel_is_preserved():
    r = intern(Reduce(name="mean", consumes=ALL_DIMS))
    assert r.consumes is ALL_DIMS


def test_indexer_keys_relabel_but_variant_and_values_stay():
    s = intern(
        Select(name="isel", indexer={"time": 0, "lat": slice(0, 5), "lon": [0, 2, 4]})
    )
    assert isinstance(s, InternedSelect)
    assert all(isinstance(k, InternedVal) for k in s.indexer)  # keys relabeled
    values = list(s.indexer.values())
    # the variant wrappers and their literal values are untouched
    assert Scalar(0) in values  # position stays the bare int 0
    assert any(isinstance(v, ForwardSlice) for v in values)
    assert Positions((0, 2, 4)) in values  # positions stay bare ints


def test_chunk_keys_relabel_but_spec_stays():
    rc = intern(
        Rechunk(name="chunk", args=({"time": 100},), chunks=frozendict({"time": 100}))
    )
    assert all(isinstance(k, InternedVal) for k in rc.chunks)
    assert list(rc.chunks.values()) == [SingleSize(100)]  # spec kept structural


def test_derived_flags_are_materialized():
    reduce = Reduce(
        name="sum", kwargs=frozendict(keepdims=True), consumes=frozenset({"time"})
    )
    assert intern(reduce).keepdims is True

    single = Project(name="__getitem__", args=("tas",), variables=("tas",))
    listy = Project(name="__getitem__", args=(["tas"],), variables=("tas",))
    assert intern(single).single is True
    assert intern(listy).single is False

    rechunk = Rechunk(name="chunk", args=("auto",))
    assert intern(rechunk).uniform == rechunk.uniform


def test_replay_header_is_verbatim():
    w = xr.DataArray(np.ones(3), dims="lat")
    op = Reduce(
        name="mean",
        args=("time",),
        kwargs=frozendict(skipna=False),
        consumes=frozenset({"time"}),
    )
    interned = intern(op)
    assert interned.args == ("time",)  # replay header untouched
    assert interned.kwargs == frozendict(skipna=False)
    # an array payload rides along verbatim (same object)
    opaque = intern(Reduce(name="mean", args=(w,)))
    assert opaque.args[0] is w


def test_deintern_ignores_materialized_flags_and_rebuilds_from_header():
    op = Reduce(
        name="sum", kwargs=frozendict(keepdims=True), consumes=frozenset({"time"})
    )
    assert deintern(intern(op)) == op
