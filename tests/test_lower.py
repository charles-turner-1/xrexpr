"""Tests for the lowering stage: ``to_lower_ir`` and ``emit``.

Two things are pinned here. The *contract* — lowering is semantics-preserving and
idempotent, and ``emit`` reproduces the calls the recorder saw, spelling included — and
the *fusion policy*: which builder pairs v1 claims to understand, and that everything else
takes the mandatory opaque fallback rather than being modelled on a guess.

Refusing to fuse is always safe, so the negative cases matter as much as the positive
ones: each pins a narrowing of what is claimed, not a bug.

``emit`` is a pure function of the plan, so everything here runs without a dataset; the
end-to-end equality lives in ``test_accessor.py`` and ``test_properties.py``.
"""

from typing import get_args

import pytest
from frozendict import frozendict

from xrexpr.ir import (
    ContextOpen,
    ContextOpenName,
    GroupedReduce,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
)
from xrexpr.lower import Call, emit, to_lower_ir
from xrexpr.operations import CONTEXT_METHODS
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


def test_context_open_names_match_the_runtime_table():
    # The Literal in ``ir`` and the frozenset in ``operations`` are the same list written
    # twice -- one for the type checker, one for ``to_opnode``'s dispatch. Pin them
    # together so neither can drift.
    assert set(get_args(ContextOpenName)) == set(CONTEXT_METHODS)


def test_every_context_method_records_as_an_opener():
    assert all(
        isinstance(to_opnode(name, (), {}), ContextOpen)
        for name in sorted(CONTEXT_METHODS)
    )


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


# --- fusing the groupby family -------------------------------------------------------


def _lower(*calls):
    """Lower the plan the recorder would build from ``(name, args, kwargs)`` triples."""
    return to_lower_ir([to_opnode(*call) for call in calls])


@pytest.mark.parametrize(
    ("grouper", "group_dim", "new_dim"),
    [
        # verified against xarray 2026.7.0 -- see GroupedReduce's docstring
        (("groupby", ("time.month",), {}), "time", "month"),
        (("groupby", ("lat",), {}), "lat", "lat"),
        (("groupby_bins", ("lat", 2), {}), "lat", "lat_bins"),
        (("resample", (), {"time": "2D"}), "time", "time"),
    ],
)
def test_grouper_dims_are_read_off_the_opener(grouper, group_dim, new_dim):
    (node,) = _lower(grouper, ("mean", (), {}))
    assert isinstance(node, GroupedReduce)
    assert (node.group_dim, node.new_dim) == (group_dim, new_dim)
    assert node.reduce == "mean"


def test_bare_closer_consumes_nothing_extra():
    # The correction that is easy to get backwards: a grouped bare ``mean()`` reduces
    # only along the group dim, so unlike a Dataset-level bare reduce it leaves the other
    # dims alone. ``consumes`` is what it removes *in addition*, which here is nothing.
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", (), {}))
    assert node.consumes == frozenset()


def test_closer_naming_the_group_dim_fuses_with_no_extra():
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", ("time",), {}))
    assert node.consumes == frozenset()


def test_closer_naming_more_than_the_group_dim_keeps_the_rest_as_consumes():
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", (["time", "lat"],), {}))
    assert node.consumes == frozenset({"lat"})


def test_within_group_map_does_not_fuse():
    # ``ds.groupby("time.month").mean("lat")`` is a per-group *map*: it keeps ``time``
    # and mints no ``month``, so the aggregation node would misdescribe it. Refusing
    # leaves a verbatim pair, which is correct if unoptimised.
    lowered = _lower(("groupby", ("time.month",), {}), ("mean", ("lat",), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]
    assert [n.name for n in lowered] == ["groupby", "mean"]


@pytest.mark.parametrize(
    "opener",
    [
        ("rolling", (), {"time": 3}),
        ("coarsen", (), {"time": 2}),
        ("weighted", ("w",), {}),
        ("cumulative", ("time",), {}),
        ("rolling_exp", (), {"time": 3}),
    ],
)
def test_openers_without_a_fusion_rule_demote_the_whole_pair(opener):
    # Until their own PRs land these behave exactly as the accessor's barrier made them:
    # both halves opaque, so no rule can fire on or across the pair.
    lowered = _lower(opener, ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_demoting_the_closer_is_not_optional():
    # The failure this guards: leaving the closer as the ``Reduce`` ``to_opnode``
    # provisionally built keeps a *Dataset-level* reading of a call that was never
    # Dataset-level -- a bare grouped ``mean()`` would carry ALL_DIMS and a following
    # select would be rejected against dims it never removed.
    lowered = _lower(("rolling", (), {"time": 3}), ("mean", (), {}))
    assert not any(isinstance(n, Reduce) for n in lowered)


def test_non_string_grouper_does_not_fuse():
    # A DataArray/Grouper argument has no single statically-known group dim.
    lowered = _lower(("groupby", (object(),), {}), ("mean", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_untabulated_closer_does_not_fuse():
    # ``first`` records Opaque, not Reduce, so there is no dim spec to reason about.
    lowered = _lower(("groupby", ("lat",), {}), ("first", (), {}))
    assert [type(n) for n in lowered] == [Opaque, Opaque]


def test_unclosed_context_demotes_to_a_single_opaque():
    lowered = _lower(("groupby", ("lat",), {}))
    assert [type(n) for n in lowered] == [Opaque]


def test_no_context_open_ever_survives_lowering():
    # The invariant the LoweredOp alias enforces at type-check time, asserted at runtime
    # too across every shape above -- fused, refused, and unclosed.
    plans = [
        _lower(("groupby", ("time.month",), {}), ("mean", (), {})),
        _lower(("groupby", ("time.month",), {}), ("mean", ("lat",), {})),
        _lower(("rolling", (), {"time": 3}), ("mean", (), {})),
        _lower(("groupby", ("lat",), {})),
    ]
    assert not any(isinstance(n, ContextOpen) for plan in plans for n in plan)


def test_ops_after_a_context_are_modelled_again():
    # What retires the trailing barrier: only the pair goes opaque, so the ``mean`` that
    # follows an unfusable ``first`` is a real Reduce again.
    lowered = _lower(
        ("groupby", ("lat",), {}), ("first", (), {}), ("mean", ("time",), {})
    )
    assert [type(n) for n in lowered] == [Opaque, Opaque, Reduce]
    assert lowered[-1].consumes == frozenset({"time"})


def test_fused_node_emits_both_calls_verbatim():
    (node,) = _lower(("groupby", ("time.month",), {}), ("mean", (), {"skipna": True}))
    assert emit([node]) == [
        Call(name="groupby", args=("time.month",)),
        Call(name="mean", kwargs=frozendict({"skipna": True})),
    ]


def test_lowering_stays_idempotent_with_fusion():
    once = _lower(
        ("groupby", ("time.month",), {}), ("mean", (), {}), ("cumsum", ("lat",), {})
    )
    assert to_lower_ir(once) == once
