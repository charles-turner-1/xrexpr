"""Tests for the op metadata tables.

The headline behaviour is the reduce/scan split — reductions destroy their dim,
scans keep it — which a single flat "aggregations" set could not express.

``CONTEXT_METHODS`` lives here too, and is pinned against the two things it must agree
with: xarray itself, and the ``Literal`` that types the same names for mypy.
"""

from typing import get_args

import pytest
import xarray as xr

from xrexpr.ir import ContextOpenName
from xrexpr.operations import CONTEXT_METHODS, OP_TABLE, OpSpec, spec

_KINDS = {"reduce", "scan", "select", "rechunk"}


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


@pytest.mark.parametrize("name", sorted(CONTEXT_METHODS))
def test_every_context_method_exists_on_dataset(name):
    # A typo'd or xarray-removed name would silently stop opening a context, and the
    # chain would then be modelled as if its calls were Dataset-level.
    assert hasattr(xr.Dataset, name)


@pytest.mark.parametrize("name", sorted(CONTEXT_METHODS))
def test_no_context_method_is_also_tabulated(name):
    # ``to_opnode`` checks CONTEXT_METHODS *before* the kind dispatch, so a name in both
    # tables would have its OpSpec silently ignored. Neither table may claim the other's.
    assert spec(name) is None


def test_context_method_names_match_the_type_that_spells_them():
    # The frozenset here and the Literal in ``ir`` are one list written twice -- one for
    # the dispatch, one for the type checker. Pin them so neither can drift.
    assert set(get_args(ContextOpenName)) == set(CONTEXT_METHODS)
