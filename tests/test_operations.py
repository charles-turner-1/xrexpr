"""Tests for the op metadata table.

The main behaviour is the reduce/scan split: reductions remove their dim, scans keep it.
Each row is its own *type* rather than a record with a ``kind`` string, so the split is
a type distinction and mypy checks that ``to_opnode``'s dispatch is exhaustive.

``CONTEXT_METHODS`` is tested here too, against the one thing no derivation can
guarantee: xarray itself. Agreement with the ``Literal`` that types the same names isn't
tested, because the rows are built from that ``Literal``.
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
    """``sel``/``isel`` are ``SelectSpec`` rows, which claim no dim effect of their own.

    Notes
    -----
    Whether a select removes a dim depends on the *indexer*, not the method, so the spec
    has nothing to say about it. ``to_opnode`` works it out per call.
    """
    assert isinstance(spec(name), SelectSpec)


def test_reduce_and_scan_are_distinguished():
    """A scan and a reduce are different variants, not one record with different fields.

    Notes
    -----
    Lumping ``cumsum`` in with ``mean`` would let a rule reorder across an
    order-significant op. Written as a type distinction, so the two cannot be confused
    by a mistyped string.
    """
    assert type(spec("cumsum")) is not type(spec("mean"))
    assert isinstance(spec("mean"), ReduceSpec)
    assert isinstance(spec("cumsum"), ScanSpec)


def test_reduce_takes_its_dim_spec_second():
    """``Dataset.reduce`` is the one reduction whose dim isn't its first positional argument.

    Notes
    -----
    ``reduce(func, dim, ...)`` puts a *function* first. The position is stored on the
    row that has the odd signature, not in a second name-keyed table, so another such
    method would be one more field, not one more lookup.
    """
    assert spec("reduce").dim_arg == 1
    assert spec("mean").dim_arg == 0


def test_spec_unknown_returns_none():
    """An untabulated name has no spec, which is what routes it to ``Opaque``."""
    assert spec("where") is None


def test_leading_positional_methods_stay_out_of_the_op_table():
    """``quantile``/``sum_of_squares``/``sum_of_weights`` aren't tabulated, so they're ``Opaque``.

    Notes
    -----
    ``lower._dim_call`` assumes a reduction's dim is at position ``_dim_arg(name)`` (0,
    or 1 for ``reduce``) and removes that positional to rebuild the dim from the node's
    field. A method whose first positional is something else (``quantile(q, dim=...)``)
    would have its ``q`` removed as if it were the dim. Such a method must not be added
    to ``OP_TABLE`` as a plain :class:`~xrexpr.operations.ReduceSpec` (or must have the
    right ``dim_arg``). This checks the known cases haven't been, so they replay
    verbatim and never reach ``_dim_call``.
    """
    assert spec("quantile") is None
    assert spec("sum_of_squares") is None
    assert spec("sum_of_weights") is None


def test_getitem_is_tabulated_even_though_the_key_decides():
    """``__getitem__`` has a row, and that row only *nominates* the kind.

    Notes
    -----
    Every other row names the kind its call is; this one names the kind its call may be,
    because ``ds["tas"]`` is a ``Project`` and ``ds[mask]`` is an ``Opaque``. Having a
    row anyway keeps ``to_opnode``'s dispatch to one ``match``: the key-shape test is a
    *guard* on that arm, rather than a special case added after the dispatch, where a
    new kind could slip past it.
    """
    assert isinstance(spec("__getitem__"), ProjectSpec)


def test_every_spec_variant_is_tabulated():
    """The table uses exactly the spec types ``to_opnode`` dispatches on, no more and no less.

    Notes
    -----
    mypy already rejects a *new* spec type with no arm in ``to_opnode`` (the ``match``
    ends in ``assert_never``). This checks the other direction, which types can't: a
    spec type that exists but has no row would be an arm no call can reach, so it would
    quietly go dead.
    """
    assert {type(op) for op in OP_TABLE.values()} == set(get_args(OpSpec))


def test_the_table_is_keyed_by_the_name_each_row_carries():
    """No row can be filed under a key that disagrees with its own ``name``.

    Notes
    -----
    True by construction (the table is built from the specs), and checked because that's
    why ``to_opnode`` needs no ``cast``: it trusts a looked-up spec's ``name`` to be the
    name it looked up.
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
    ``CONTEXT_METHODS`` is derived from these rows rather than kept as a separate set,
    so a name cannot sit in two tables with one of its specs silently ignored.
    """
    assert isinstance(spec(name), ContextSpec)


def test_chunk_is_a_rechunk_spec():
    """``chunk`` is its own variant: it touches chunk topology, never a dim."""
    assert isinstance(spec("chunk"), RechunkSpec)
