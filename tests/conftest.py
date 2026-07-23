"""Shared fixtures for the test suite.

Both fixtures were previously duplicated across four test modules; they are lifted here
so every module sees the same canonical dataset — a ``temperature`` variable over
``("time", "lat", "lon")`` sized ``(4, 3, 5)`` alongside an ``elevation`` variable over
``("lat", "lon")`` that is *missing* ``time``, with integer ``arange`` coords. The
second, ``time``-less variable is what the projection-pushdown rule must refuse to
reorder past a ``time`` reduction, so the metadata modules need it in scope too.

``ds`` carries random (seeded) values for the modules that evaluate results; ``schema``
wraps a zeros dataset of the same shape for the modules that only reason about metadata.
``tests/test_accessor.py`` shadows ``ds`` with a richer variant that adds a non-dimension
coord, to prove the rewrite preserves auxiliary coords.
"""

import numpy as np
import pytest
import xarray as xr

from xrexpr.schema import SchemaState


@pytest.fixture
def ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), rng.random((4, 3, 5))),
            # a second variable that is *missing* ``time``: projecting it can't be
            # pushed past a ``time`` reduction or selection
            "elevation": (("lat", "lon"), rng.random((3, 5))),
        },
        coords={
            "time": np.arange(4),
            "lat": np.arange(3),
            "lon": np.arange(5),
        },
    )


@pytest.fixture
def schema() -> SchemaState:
    ds = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.zeros((4, 3, 5))),
            "elevation": (("lat", "lon"), np.zeros((3, 5))),
        },
        coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(5)},
    )
    return SchemaState.from_dataset(ds)
