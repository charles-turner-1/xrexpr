"""Tests for the lowering stage: ``to_lower_ir`` and ``emit``.

The stage is currently an identity — no fused node kinds exist yet — so these pin the
*contract* rather than any translation: lowering is semantics-preserving and idempotent,
and ``emit`` reproduces the calls the recorder saw, spelling included. Written now, while
the answers are trivially known, so the first fusion rule lands against assertions that
were not written to accommodate it.

``emit`` is a pure function of the plan, so everything here runs without a dataset; the
end-to-end equality lives in ``test_accessor.py`` and ``test_properties.py``.
"""

import pytest
from frozendict import frozendict

from xrexpr.ir import Opaque, Project, Rechunk, Reduce, Scan, Select
from xrexpr.lower import Call, emit, to_lower_ir
from xrexpr.schema import to_opnode


@pytest.fixture
def plan():
    """A plan covering every ``Op`` variant, built the way the recorder builds one."""
    return [
        to_opnode("chunk", ({"time": 100},), {}),
        to_opnode("__getitem__", (["temperature"],), {}),
        to_opnode("isel", (), {"time": 0, "drop": True}),
        to_opnode("cumsum", ("lat",), {}),
        to_opnode("mean", ("lat",), {}),
        to_opnode("where", ("cond",), {}),
    ]


def test_the_fixture_plan_really_covers_every_variant(plan):
    # keeps the round-trip tests below from passing vacuously on a partial vocabulary
    assert {type(node) for node in plan} == {
        Rechunk,
        Project,
        Select,
        Scan,
        Reduce,
        Opaque,
    }


def test_lowering_is_the_identity_today(plan):
    assert to_lower_ir(plan) == plan


def test_lowering_is_idempotent(plan):
    once = to_lower_ir(plan)
    assert to_lower_ir(once) == once


def test_lowering_does_not_mutate_its_input(plan):
    before = list(plan)
    to_lower_ir(plan).append(to_opnode("mean", (), {}))
    assert plan == before


def test_emit_reproduces_every_call_verbatim(plan):
    # one node, one call, header untouched -- an unmodified plan replays what was written
    assert emit(to_lower_ir(plan)) == [
        Call(name=node.name, args=node.args, kwargs=node.kwargs) for node in plan
    ]


def test_emit_keeps_the_recorded_spelling(plan):
    # ``isel(time=0, drop=True)`` was recorded as kwargs and stays kwargs -- emit must not
    # canonicalise it to the positional-dict form the merge rule happens to build.
    select = emit(to_lower_ir(plan))[2]
    assert select == Call(name="isel", kwargs=frozendict({"time": 0, "drop": True}))


def test_emit_of_an_empty_plan_is_empty():
    assert emit(to_lower_ir([])) == []


def test_call_coerces_to_immutable_containers_and_hashes():
    call = Call(name="mean", args=["lat"], kwargs={"skipna": True})
    assert call.args == ("lat",)
    assert call.kwargs == frozendict({"skipna": True})
    assert hash(call)


def test_calls_compare_by_value():
    assert Call(name="mean", args=("lat",)) == Call(name="mean", args=("lat",))
    assert Call(name="mean", args=("lat",)) != Call(name="mean", args=("lon",))
