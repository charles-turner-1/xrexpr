"""Tests for the op metadata table (PR 3).

The headline behaviour is the reduce/scan split — the thing the old bare
``AGGREGATIONS`` set could not express — plus the back-compat sets still derived
from the table.
"""

import pytest

from xrexpr.operations import (
    AGGREGATIONS,
    OP_TABLE,
    SELECTIONS,
    OpSpec,
    spec,
)

_KINDS = {"reduce", "scan", "select"}


@pytest.mark.parametrize(
    "name",
    [
        "reduce",
        "count",
        "all",
        "any",
        "max",
        "min",
        "mean",
        "prod",
        "sum",
        "std",
        "var",
        "median",
    ],
)
def test_reductions_consume_their_dim(name):
    assert spec(name) == OpSpec("reduce", True)


@pytest.mark.parametrize("name", ["cumsum", "cumprod", "diff"])
def test_scans_keep_their_dim(name):
    assert spec(name) == OpSpec("scan", False)


@pytest.mark.parametrize("name", ["sel", "isel"])
def test_selects(name):
    assert spec(name) == OpSpec("select", False)


def test_reduce_and_scan_are_distinguished():
    # the core fix: cumsum is no longer lumped in with mean
    assert spec("cumsum").kind != spec("mean").kind
    assert spec("mean").consumes_dim is True
    assert spec("cumsum").consumes_dim is False


def test_spec_unknown_returns_none():
    assert spec("rolling") is None


def test_every_spec_kind_is_valid():
    assert {s.kind for s in OP_TABLE.values()} == _KINDS


def test_selections_backcompat_unchanged():
    assert SELECTIONS == frozenset({"sel", "isel"})


def test_aggregations_backcompat_covers_reduces_and_scans():
    assert {"mean", "sum", "median"} <= AGGREGATIONS  # reduces
    assert {"cumsum", "cumprod", "diff"} <= AGGREGATIONS  # scans
    assert AGGREGATIONS.isdisjoint(SELECTIONS)
