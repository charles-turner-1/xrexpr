"""Tests for the op metadata table.

The headline behaviour is the reduce/scan split — reductions destroy their dim,
scans keep it — which a single flat "aggregations" set could not express. Each row is
now a *variant* rather than a record with a ``kind`` string, so that split is a type
distinction and ``to_opnode``'s dispatch is exhaustive under mypy.

``CONTEXT_METHODS`` lives here too, and is pinned against the one thing it must agree with
that no derivation can settle: xarray itself. Agreement with the ``Literal`` that types
the same names is not tested, because the rows are built from that ``Literal``.
"""

from typing import get_args

import pytest
import xarray as xr

from xrexpr.operations import (
    CONTEXT_METHODS,
    OP_TABLE,
    ContextSpec,
    OpSpec,
    ProjectSpec,
    RechunkSpec,
    ReduceSpec,
    ScanSpec,
    SelectSpec,
    spec,
)


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
def test_reductions_are_reduce_specs(name):
    """Every tabulated reduction is a ``ReduceSpec``, the variant that destroys its dim."""
    assert isinstance(spec(name), ReduceSpec)


@pytest.mark.parametrize("name", ["cumsum", "cumprod", "diff"])
def test_scans_are_scan_specs(name):
    """Every tabulated scan is a ``ScanSpec``, the variant that leaves its dim in place."""
    assert isinstance(spec(name), ScanSpec)


@pytest.mark.parametrize("name", ["sel", "isel"])
def test_selects(name):
    """``sel``/``isel`` are ``SelectSpec``s, which claim no dim effect of their own.

    Notes
    -----
    There is no ``consumes_dim`` to claim it with, because a select's dim removal depends
    on the *indexer*, not the method — ``to_opnode`` resolves it per call.
    """
    assert isinstance(spec(name), SelectSpec)


def test_reduce_and_scan_are_distinguished():
    """A scan and a reduce are different variants, not one record with different fields.

    Notes
    -----
    The core fix this table exists for: ``cumsum`` is no longer lumped in with ``mean``,
    which is what let a rule reorder across an order-significant op. Written as a type
    distinction, so the two cannot be confused by a mistyped string.
    """
    assert type(spec("cumsum")) is not type(spec("mean"))
    assert isinstance(spec("mean"), ReduceSpec)
    assert isinstance(spec("cumsum"), ScanSpec)


def test_reduce_takes_its_dim_spec_second():
    """``Dataset.reduce`` is the one reduction whose dim spec is not its first positional.

    Notes
    -----
    ``reduce(func, dim, ...)`` puts a *function* first (#96). The offset rides on the row
    that has the odd signature rather than in a second table keyed by name, so a second
    such method is one more field, not one more lookup.
    """
    assert spec("reduce").dim_arg == 1
    assert spec("mean").dim_arg == 0


def test_spec_unknown_returns_none():
    """An untabulated name has no spec, which is what routes it to ``Opaque``."""
    assert spec("where") is None


def test_leading_positional_methods_stay_out_of_the_op_table():
    """``quantile``/``sum_of_squares``/``sum_of_weights`` are untabulated → ``Opaque``.

    Notes
    -----
    ``lower._dim_call`` assumes a reduction's dim sits at ``_dim_arg(name)`` — 0, or 1 for
    ``reduce`` — and strips that positional out to re-derive the dim from the semantic field.
    A method whose first positional is something else (``quantile(q, dim=...)``) would have
    its ``q`` stripped as if it were the dim. Such a method must not enter ``OP_TABLE`` as a
    plain :class:`~xrexpr.operations.ReduceSpec` (or must carry the right ``dim_arg``); this
    pins that the known offenders haven't, so they replay verbatim and never reach
    ``_dim_call``. See ``lower._weighted_reduce``.
    """
    assert spec("quantile") is None
    assert spec("sum_of_squares") is None
    assert spec("sum_of_weights") is None


def test_getitem_is_tabulated_even_though_the_key_decides():
    """``__getitem__`` has a row, and that row only *nominates* the kind.

    Notes
    -----
    Every other row names the variant its call is; this one names the variant its call
    may be, because ``ds["tas"]`` is a ``Project`` and ``ds[mask]`` is an ``Opaque``.
    Tabulating it anyway is what keeps ``to_opnode``'s dispatch to a single ``match`` —
    the shape-of-the-key test is a *guard* on the arm rather than a special case bolted
    on after the dispatch, where a new variant could slip past it.
    """
    assert isinstance(spec("__getitem__"), ProjectSpec)


def test_every_spec_variant_is_tabulated():
    """The table uses exactly the variants ``to_opnode`` dispatches on — no more, no less.

    Notes
    -----
    mypy already rejects a *new* variant with no arm in ``to_opnode`` (the ``match`` is
    closed with ``assert_never``), which is what the old string ``kind`` could not do.
    This pins the other direction, which types cannot: a variant that exists but is never
    tabulated is an arm no call can reach, so it would go quietly dead.
    """
    assert {type(op) for op in OP_TABLE.values()} == set(get_args(OpSpec))


def test_the_table_is_keyed_by_the_name_each_row_carries():
    """No row can be filed under a key that disagrees with its own ``name``.

    Notes
    -----
    True by construction — the table is derived from the specs — and asserted because
    that construction is the reason ``to_opnode`` needs no ``cast``: it trusts a looked-up
    spec's ``name`` to be the name it looked up.
    """
    assert all(name == op.name for name, op in OP_TABLE.items())


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
def test_context_methods_are_context_specs(name):
    """A context opener is a row in the one table, like every other classified method.

    Notes
    -----
    It used to be a separate frozenset consulted *before* the kind dispatch, so a name in
    both tables would have had its ``OpSpec`` silently ignored — a hazard a test had to
    police. One table per name makes that unconstructible: ``CONTEXT_METHODS`` is now
    derived from these rows rather than pinned against them.
    """
    assert isinstance(spec(name), ContextSpec)


def test_chunk_is_a_rechunk_spec():
    """``chunk`` is its own variant: it touches chunk topology, never a dim."""
    assert isinstance(spec("chunk"), RechunkSpec)
