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
    """Every tabulated reduction is kind ``reduce`` and destroys the dim it is given."""
    assert spec(name) == OpSpec("reduce", True)


@pytest.mark.parametrize("name", ["cumsum", "cumprod", "diff"])
def test_scans_keep_their_dim(name):
    """Every tabulated scan is kind ``scan`` and leaves the dim it is given in place."""
    assert spec(name) == OpSpec("scan", False)


@pytest.mark.parametrize("name", ["sel", "isel"])
def test_selects(name):
    """``sel``/``isel`` are kind ``select``, and the table claims no dim effect for them.

    Notes
    -----
    ``consumes_dim`` is ``False`` here because a select's dim removal depends on the
    *indexer*, not the method — ``to_opnode`` resolves it per call.
    """
    assert spec(name) == OpSpec("select", False)


def test_reduce_and_scan_are_distinguished():
    """A scan and a reduce disagree on both kind and dim effect.

    Notes
    -----
    The core fix this table exists for: ``cumsum`` is no longer lumped in with ``mean``,
    which is what let a rule reorder across an order-significant op.
    """
    assert spec("cumsum").kind != spec("mean").kind
    assert spec("mean").consumes_dim is True
    assert spec("cumsum").consumes_dim is False


def test_spec_unknown_returns_none():
    """An untabulated name has no spec, which is what routes it to ``Opaque``."""
    assert spec("rolling") is None


def test_every_spec_kind_is_valid():
    """The table uses exactly the four kinds ``to_opnode`` dispatches on — no more, no less.

    Notes
    -----
    A fifth kind added here without an arm in ``to_opnode`` would fall through to
    ``Opaque`` silently, so the set is pinned rather than merely bounded.
    """
    assert {s.kind for s in OP_TABLE.values()} == _KINDS


@pytest.mark.parametrize("name", sorted(CONTEXT_METHODS))
def test_every_context_method_exists_on_dataset(name):
    """Every name in ``CONTEXT_METHODS`` is really a method of ``xr.Dataset``.

    Notes
    -----
    A typo'd or xarray-removed name would silently stop opening a context, and the chain
    would then be modelled as if its calls were Dataset-level.
    """
    assert hasattr(xr.Dataset, name)


@pytest.mark.parametrize("name", sorted(CONTEXT_METHODS))
def test_no_context_method_is_also_tabulated(name):
    """The two tables are disjoint: a context opener has no ``OpSpec``.

    Notes
    -----
    ``to_opnode`` checks ``CONTEXT_METHODS`` *before* the kind dispatch, so a name in
    both tables would have its ``OpSpec`` silently ignored. Neither table may claim the
    other's.
    """
    assert spec(name) is None


def test_context_method_names_match_the_type_that_spells_them():
    """``CONTEXT_METHODS`` and the ``ContextOpenName`` literal name the same set.

    Notes
    -----
    The frozenset here and the ``Literal`` in ``ir`` are one list written twice — one for
    the dispatch, one for the type checker. Pin them so neither can drift.
    """
    assert set(get_args(ContextOpenName)) == set(CONTEXT_METHODS)
