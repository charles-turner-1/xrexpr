"""Tests for the plan listing: ``xrexpr.explain.format_plan``.

``format_plan`` is a pure function of a lowered plan, so everything here builds nodes
directly and runs without a dataset; the end-to-end assertions — that what ``explain()``
prints is what ``collect()`` replays — live in ``test_accessor.py``.

Two kinds of assertion, and the distinction is deliberate. A handful pin the **whole
string** for a whole plan, because the format is a user-facing artefact the README quotes
and silent drift in it is a real regression. The rest pin **one fact per node kind**, so a
change in the surrounding layout does not have to be re-approved in a dozen places.
"""

import numpy as np
import pytest
import xarray as xr
from frozendict import frozendict

from xrexpr.explain import _annotate, format_plan
from xrexpr.ir import (
    ALL_DIMS,
    GroupedReduce,
    Opaque,
    Project,
    Rechunk,
    Reduce,
    Scan,
    Select,
    WeightedReduce,
    WindowedReduce,
)


def _isel(**indexer):
    """Build an ``isel`` node from dim keywords, with its indexer already normalised."""
    return Select(name="isel", kwargs=frozendict(indexer), indexer=frozendict(indexer))


def test_an_empty_plan_says_so():
    """A plan with no nodes renders as a one-line ``plan (0 ops)`` rather than a header."""
    assert format_plan([]) == "plan (0 ops)"


def test_a_plan_lists_one_numbered_line_per_node():
    """A two-node plan renders as a header plus one numbered line each, annotations included.

    Notes
    -----
    One of the whole-string assertions: the format is a user-facing artefact the README
    quotes, so silent drift in it is a real regression.
    """
    plan = [
        _isel(time=0),
        Reduce(
            name="mean", kwargs=frozendict({"dim": "lat"}), consumes=frozenset({"lat"})
        ),
    ]
    # ``emit`` derives the header from the semantic fields, so the rendered call is the
    # canonical spelling it replays as: the select's indexer as a positional dict, and the
    # reduce's dim positionally rather than as ``dim=``.
    assert format_plan(plan) == (
        "plan (2 ops):\n"
        "  1. Select  isel({'time': 0})\n"
        "  2. Reduce  mean('lat')  [consumes={lat}]"
    )


def test_a_fused_node_is_one_op_rendered_as_the_two_calls_it_replays_as():
    """A ``GroupedReduce`` counts as one op, and its line still shows both calls.

    Notes
    -----
    The point of formatting nodes rather than calls: ``groupby(...).mean()`` counts as
    *one* op, because one op is what it is — while the line still shows both calls, so it
    has not stopped answering "what will run".
    """
    grouped = GroupedReduce(
        name="groupby",
        group_dim="time",
        new_dim="month",
        reduce="mean",
        args=("time.month",),
        consumes=frozenset({"time"}),
    )
    assert format_plan([grouped]) == (
        "plan (1 ops):\n"
        "  1. GroupedReduce  groupby('time.month').mean()  [time -> month]"
    )


def test_a_bare_reduce_says_it_consumes_every_dim():
    """A ``Reduce`` carrying ``ALL_DIMS`` is annotated ``consumes=every dim``.

    Notes
    -----
    ``mean()`` on its own reads as if it did nothing to the dims; ``ALL_DIMS`` is
    precisely the fact the call cannot state.
    """
    text = format_plan([Reduce(name="mean", consumes=ALL_DIMS)])
    assert text == "plan (1 ops):\n  1. Reduce  mean()  [consumes=every dim]"


def test_an_opaque_node_is_visible_as_the_barrier_it_is():
    """An ``Opaque`` node is annotated as unmodelled rather than rendered like any other.

    Notes
    -----
    "Why did nothing move?" now has an answer in the output, instead of requiring the
    reader to know which methods the package models.
    """
    text = format_plan([Opaque(name="first")])
    assert text == (
        "plan (1 ops):\n  1. Opaque  first()  [not modelled -- nothing crosses it]"
    )


def test_a_weighted_reduce_names_its_weight_dims():
    """A ``WeightedReduce`` is annotated with both its weight dims and what it consumes.

    Notes
    -----
    The weights themselves elide (see the next test), so this is the only place the dims
    that decide projection pushdown appear at all.
    """
    weights = xr.DataArray(np.arange(3.0), dims="lat")
    node = WeightedReduce(
        name="weighted",
        reduce="mean",
        weight_dims=frozenset({"lat"}),
        args=(weights,),
        reduce_args=("time",),
        consumes=frozenset({"time"}),
    )
    assert "[weights over {lat}, consumes={time}]" in format_plan([node])


def test_an_array_argument_elides_instead_of_swallowing_the_listing():
    """A ``DataArray`` argument renders as ``<DataArray>``, keeping one line per node.

    Notes
    -----
    One line per node is the shape of the whole format, and a ``DataArray`` reprs as a
    multi-line block of values and coords. Pinned with a *following* node, so a regression
    shows up as the listing losing its structure, not merely as a long line.
    """
    weights = xr.DataArray(np.arange(3.0), dims="lat", coords={"lat": np.arange(3)})
    plan = [
        WeightedReduce(
            name="weighted",
            reduce="mean",
            weight_dims=frozenset({"lat"}),
            args=(weights,),
        ),
        Reduce(name="max", consumes=ALL_DIMS),
    ]
    text = format_plan(plan)
    assert "weighted(<DataArray>).mean()" in text
    assert len(text.splitlines()) == 3  # header + one line per node


def test_a_dask_backed_argument_elides_without_computing():
    """Formatting a dask-backed weights array elides it and leaves it lazy.

    Notes
    -----
    ``_format_arg`` reprs the payload to decide whether it is multi-line. That must stay
    free: xarray reprs a dask-backed variable as its graph summary, never its values —
    computing at plan time is the one thing this package promises never to do.
    """
    dask_array = pytest.importorskip("dask.array")
    weights = xr.DataArray(dask_array.arange(3.0, chunks=2), dims="lat")
    node = WeightedReduce(
        name="weighted", reduce="mean", weight_dims=frozenset({"lat"}), args=(weights,)
    )
    assert "weighted(<DataArray>)" in format_plan([node])
    assert weights.chunks is not None  # still lazy: nothing was realised to format it


def test_a_projection_renders_as_the_getitem_it_replays():
    """A ``Project`` renders in subscript form, ``[['temperature']]``, not as a call."""
    text = format_plan(
        [
            Project(
                name="__getitem__", args=(["temperature"],), variables=("temperature",)
            )
        ]
    )
    assert text == "plan (1 ops):\n  1. Project  [['temperature']]"


@pytest.mark.parametrize(
    "node",
    [
        _isel(time=0),
        Project(name="__getitem__", args=(["temperature"],)),
        Rechunk(name="chunk", args=({"time": 100},), chunks=frozendict({"time": 100})),
        Scan(name="cumsum", args=("time",)),
        WindowedReduce(
            name="rolling",
            reduce="mean",
            window=frozendict({"time": 3}),
            kwargs=frozendict({"time": 3}),
        ),
    ],
    ids=["select", "project", "rechunk", "scan", "windowed"],
)
def test_a_node_whose_call_says_everything_gets_no_annotation(node):
    """Select, project, rechunk, scan and windowed nodes annotate to the empty string.

    Notes
    -----
    The editorial rule this module exists to hold: annotate only what the *calls* do not
    already state. ``rolling(time=3)`` names its window and ``chunk({'time': 100})`` its
    chunks — restating either is noise, and noise is what stops a listing being read.
    """
    assert _annotate(node) == ""
