"""Tests for the expression IR: the ``Op`` sum type (``Reduce``/``Select``/``Scan``/``Project``/``Opaque``).

``kwargs``/``indexer`` are backed by the third-party ``frozendict``, so we don't
re-test that library's internals — only that each variant coerces to it, stays
frozen/hashable, and that the derived properties really are derived: ``Select.consumes``
from ``indexer``, and ``Project.single`` from the verbatim key (never stored fields that
could drift from what replay does).

Hashability is the one claim that is *conditional*, so it is pinned in both directions: a
node whose payload holds a ``DataArray`` is unhashable and uncomparable, while nodes
sharing that payload object still compare equal. See ``ir.py``'s module docstring for why
the weaker claim is the one wanted.
"""

import dataclasses

import pytest
import xarray as xr
from frozendict import frozendict as _pkg_frozendict

from xrexpr.chunks import (
    Auto,
    BlockSeq,
    ByteSize,
    FullDim,
    NoChange,
    OpaqueChunk,
    SingleSize,
)
from xrexpr.ir import (
    ALL_DIMS,
    AllDims,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
    WindowedReduce,
    frozendict,
)


def test_ir_reexports_third_party_frozendict():
    """``ir.frozendict`` is the third-party class, not a local reimplementation."""
    assert frozendict is _pkg_frozendict


def test_reduce_minimal_defaults():
    """A ``Reduce`` built from a name alone has empty args, kwargs and consumes."""
    node = Reduce(name="mean")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.consumes == frozenset()


def test_all_dims_instances_are_interchangeable():
    """A fresh ``AllDims()`` equals, hashes and reprs like the ``ALL_DIMS`` singleton.

    Notes
    -----
    Fieldless and frozen, so the sentinel is a value rather than an identity: a node built
    with a fresh ``AllDims()`` equals and hashes like one built with ``ALL_DIMS``.
    """
    assert AllDims() == ALL_DIMS
    assert hash(AllDims()) == hash(ALL_DIMS)
    assert Reduce(name="mean", consumes=AllDims()) == Reduce(
        name="mean", consumes=ALL_DIMS
    )
    assert repr(ALL_DIMS) == "ALL_DIMS"


def test_all_dims_is_not_coerced_and_stays_hashable():
    """A node built with ``ALL_DIMS`` keeps the sentinel itself, and stays hashable.

    Notes
    -----
    ``__post_init__`` coerces a dim *set* to frozenset; the sentinel must pass through
    untouched, and the node must stay shareable between plans like every other.
    """
    node = Reduce(name="mean", consumes=ALL_DIMS)
    assert node.consumes is ALL_DIMS
    assert hash(node)


def test_all_dims_is_distinct_from_the_empty_dim_set():
    """``ALL_DIMS`` is not ``frozenset()``, and the two make unequal nodes.

    Notes
    -----
    The distinction the whole change rests on: "every dim, whatever they are" is not "no
    dims". Conflating them is what let a select hop in front of a bare reduce.
    """
    assert ALL_DIMS != frozenset()
    assert Reduce(name="mean", consumes=ALL_DIMS) != Reduce(name="mean")


def test_select_minimal_defaults():
    """A ``Select`` built from a name alone has an empty indexer, so it consumes nothing."""
    node = Select(name="isel")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.indexer == frozendict()
    assert node.consumes == frozenset()


def test_scan_and_opaque_minimal_defaults():
    """``Scan`` and ``Opaque`` built from a name alone carry empty args and kwargs."""
    assert Scan(name="cumsum").args == ()
    assert Opaque(name="where").kwargs == frozendict()


def test_rechunk_minimal_defaults():
    """A ``Rechunk`` built from a name alone has an empty chunk spec and no uniform one."""
    node = Rechunk(name="chunk")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.chunks == frozendict()
    assert node.uniform is None


def test_rechunk_coerces_containers():
    """A ``Rechunk`` coerces a list of args to a tuple and its chunk spec to a frozendict."""
    node = Rechunk(name="chunk", args=[{"time": 100}], chunks={"time": 100})
    assert node.args == ({"time": 100},)
    assert isinstance(node.chunks, frozendict)


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), None),
        ((100,), SingleSize(100)),
        ((-1,), FullDim()),
        (("auto",), Auto()),
        (("10MB",), ByteSize("10MB")),
        (((100, 400, 500),), BlockSeq((100, 400, 500))),
        ((100.0,), OpaqueChunk(100.0)),
        (({"time": 100},), None),
    ],
    ids=[
        "chunk()",
        "chunk(100)",
        "chunk(-1)",
        'chunk("auto")',
        'chunk("10MB")',
        "chunk((100,400,500))",
        "chunk(100.0)",
        'chunk({"time": 100})',
    ],
)
def test_rechunk_classifies_its_uniform_spec(args, expected):
    """The uniform form is classified through the same taxonomy as the mapping form.

    Notes
    -----
    A ``chunk`` call writes its spec in one of two places and ``Rechunk`` holds one field
    per place, so a rule that reasons about ``chunk({"time": "auto"})`` reasons about
    ``chunk("auto")`` without a second code path. The last case is the discriminant: a
    leading *mapping* is the mapping form, already extracted into ``chunks``, so it leaves
    ``uniform`` empty — the split xarray's own ``Dataset.chunk`` makes with
    ``isinstance(chunks, Mapping)``.
    """
    assert Rechunk(name="chunk", args=args).uniform == expected


def test_rechunk_uniform_is_derived_from_args_not_passed():
    """``uniform`` cannot be set independently of ``args``, so the two can never disagree.

    Notes
    -----
    Unlike ``chunks``, which needs :func:`~xrexpr.schema._chunk_spec` to know which kwargs
    are dims rather than options, ``uniform`` is a plain *reading* of ``args[0]``. One that
    disagreed with ``args`` would make the node replay as something other than what it
    claims, so it is ``init=False`` and a hand-built node is normalised exactly as a
    recorded one is.
    """
    with pytest.raises(TypeError):
        Rechunk(name="chunk", args=("auto",), uniform=NoChange())  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("name", "kwargs", "expected"),
    [
        ("coarsen", {"boundary": "pad"}, True),
        ("coarsen", {"boundary": "trim"}, False),
        ("coarsen", {"boundary": "exact"}, False),
        ("coarsen", {}, False),  # ``exact`` is xarray's default
        ("coarsen", {"boundary": "wrap"}, None),  # a boundary the node does not model
        ("rolling", {}, None),  # rolling keeps every length; rounding is irrelevant
        ("rolling", {"boundary": "pad"}, None),  # ... even if a boundary rides along
    ],
    ids=[
        "coarsen-pad",
        "coarsen-trim",
        "coarsen-exact",
        "coarsen-default",
        "coarsen-unknown",
        "rolling",
        "rolling-with-boundary",
    ],
)
def test_windowed_reduce_classifies_its_rounding(name, kwargs, expected):
    """``rounds_up`` reads ``coarsen``'s ``boundary`` into the size layer's rounding rule.

    Notes
    -----
    ``pad`` ceils, ``trim``/``exact`` (and the default) floor, and a boundary the node does
    not model answers ``None`` — precision, not correctness, is what an unrecognised spelling
    costs (:func:`~xrexpr.schema._windowed_size`). ``rolling`` keeps every dim's length, so it
    never consults this and reads ``None`` whatever the header carries.
    """
    assert WindowedReduce(name=name, reduce="mean", kwargs=kwargs).rounds_up == expected


def test_windowed_reduce_rounds_up_is_derived_from_kwargs_not_passed():
    """``rounds_up`` cannot be set independently of ``kwargs``, so the two can never disagree.

    Notes
    -----
    The ``Rechunk.uniform`` precedent for a fused node: ``boundary`` is replayed verbatim from
    ``kwargs``, and its size-relevant reading is ``init=False`` so a hand-built node is
    normalised exactly as a fused one is, rather than able to claim a rounding its call does
    not perform.
    """
    with pytest.raises(TypeError):
        WindowedReduce(  # type: ignore[call-arg]
            name="coarsen", reduce="mean", kwargs={"boundary": "pad"}, rounds_up=False
        )


def test_reduce_coerces_containers():
    """A ``Reduce`` coerces args to a tuple, kwargs to a frozendict, consumes to a frozenset."""
    node = Reduce(name="mean", args=["lat"], kwargs={"skipna": True}, consumes=["lat"])
    assert node.args == ("lat",)
    assert isinstance(node.kwargs, frozendict)
    assert node.consumes == frozenset({"lat"})


def test_select_coerces_containers():
    """A ``Select`` coerces both its kwargs and its indexer to frozendicts."""
    node = Select(name="isel", kwargs={"drop": True}, indexer={"time": 0})
    assert isinstance(node.kwargs, frozendict)
    assert isinstance(node.indexer, frozendict)


def test_select_consumes_is_derived_from_indexer():
    """A mixed indexer reports only its scalar-indexed dim as consumed.

    Notes
    -----
    Scalar-indexed dims drop (and so land in ``consumes``); slice and sequence dims are
    kept.
    """
    node = Select(name="isel", indexer={"time": 0, "lat": slice(0, 2), "lon": [0, 1]})
    assert node.consumes == frozenset({"time"})


def test_select_consumes_cannot_drift_from_indexer():
    """A select re-indexing a dim with a slice reports it as kept, not dropped.

    Notes
    -----
    The desync the flat record risked: a merged select whose indexer re-indexes a dim with
    a slice reads ``consumes=∅`` off that indexer — it cannot claim the dim was dropped,
    because ``consumes`` has no independent storage.
    """
    node = Select(name="isel", indexer={"time": slice(0, 5)})
    assert node.consumes == frozenset()


def test_select_name_is_literal_typed():
    """Both valid select names construct; the invalid ones are a type error, not a raise.

    Notes
    -----
    ``Select(name="mean")`` is a *type* error (name↔kind unrepresentable) — enforced by
    mypy, not at runtime, so there is nothing to assert here beyond the valid names.
    """
    assert Select(name="isel").name == "isel"
    assert Select(name="sel").name == "sel"


def test_variants_are_frozen():
    """Assigning to a node's field raises, so a node cannot drift from itself."""
    node = Reduce(name="mean")
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.name = "sum"


def test_metadata_cannot_be_mutated_in_place():
    """Freezing reaches into the containers: neither kwargs nor indexer accepts an item set."""
    node = Select(name="isel", kwargs={"drop": True}, indexer={"time": 0})
    with pytest.raises(TypeError):
        node.kwargs["drop"] = False
    with pytest.raises(TypeError):
        node.indexer["time"] = 1


def test_variants_hashable_and_value_equal():
    """Two nodes built from equal payloads compare equal, hash equal, and dedupe in a set."""
    a = Reduce(name="mean", args=("lat",), consumes=["lat"])
    b = Reduce(name="mean", args=("lat",), consumes=["lat"])
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_a_node_carrying_an_array_is_not_hashable():
    """Hashing a node whose payload holds a ``DataArray`` raises ``TypeError``.

    Notes
    -----
    ``ir.py``'s hashability promise is deliberately *conditional* — hashable when the
    payload is. ``xr.DataArray.__hash__`` is ``None``, so any node whose payload holds one
    raises, and this pins that condition. Making the promise unconditional would mean
    hashing the array's *values* — computing a dask array at plan time, which is the one
    thing the package promises never to do.
    """
    weights = xr.DataArray([1.0, 2.0, 3.0], dims="lat")
    with pytest.raises(TypeError, match="unhashable type"):
        hash(Opaque(name="weighted", args=(weights,)))


def test_two_nodes_holding_distinct_but_equal_arrays_cannot_be_compared():
    """Comparing two nodes holding *distinct but equal* arrays raises rather than answering.

    Notes
    -----
    Sharper than the hashability gap and easy to miss: ``==`` does not merely return
    ``False``, it *raises* — comparing the payloads elementwise yields an array, which has
    no truth value.
    """
    weights = xr.DataArray([1.0, 2.0, 3.0], dims="lat")
    a = Opaque(name="weighted", args=(weights,))
    b = Opaque(name="weighted", args=(weights.copy(),))
    with pytest.raises(ValueError, match="truth value of an array"):
        a == b  # noqa: B015 -- the comparison itself is what's under test


def test_nodes_sharing_an_array_payload_still_compare_equal():
    """Two nodes sharing the same array *object* compare equal, so plan equality holds.

    Notes
    -----
    Why the gap above is latent rather than live: a plan holds one payload *object*, and
    tuple comparison short-circuits on identity, so every plan-equality assertion in the
    suite (lowering idempotence, the optimiser fixpoint) is sound as written.
    """
    weights = xr.DataArray([1.0, 2.0, 3.0], dims="lat")
    assert Opaque(name="weighted", args=(weights,)) == Opaque(
        name="weighted", args=(weights,)
    )


def test_distinct_variants_are_unequal():
    """A ``Select`` and an ``Opaque`` sharing a name are different nodes, not equal ones."""
    assert Opaque(name="isel") != Select(name="isel")


def test_project_minimal_defaults():
    """A ``Project`` built from a name alone names no variables."""
    node = Project(name="__getitem__")
    assert node.args == ()
    assert node.kwargs == frozendict()
    assert node.variables == ()


def test_project_coerces_variables_to_a_tuple():
    """A ``Project`` given a list of variables stores them as a tuple, preserving order."""
    node = Project(name="__getitem__", args=(["tas", "pr"],), variables=["tas", "pr"])
    assert node.variables == ("tas", "pr")


def test_project_is_frozen_and_hashable():
    """A ``Project`` rejects assignment to ``variables`` and hashes by value."""
    node = Project(name="__getitem__", args=("tas",), variables=("tas",))
    with pytest.raises(dataclasses.FrozenInstanceError):
        node.variables = ("pr",)
    assert hash(node) == hash(
        Project(name="__getitem__", args=("tas",), variables=("tas",))
    )


def test_project_single_is_derived_from_the_key():
    """A bare key reports ``single``; the same name in a list does not.

    Notes
    -----
    A bare (hashable) key selects one variable → ``DataArray``; a list → ``Dataset``.
    Derived from ``args``, so it can never disagree with what replay does.
    """
    assert Project(name="__getitem__", args=("tas",), variables=("tas",)).single
    assert not Project(name="__getitem__", args=(["tas"],), variables=("tas",)).single


def test_project_tuple_key_is_a_single_name():
    """A tuple key reports ``single``: xarray reads a hashable key as *one* name."""
    assert Project(
        name="__getitem__", args=(("a", "b"),), variables=(("a", "b"),)
    ).single
