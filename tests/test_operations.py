"""Tests for the op metadata table (PR 3).

The headline behaviour is the reduce/scan split — the thing the old bare
``AGGREGATIONS`` set could not express.
"""

import pytest

from xrexpr.operations import OP_TABLE, OpSpec, spec

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
