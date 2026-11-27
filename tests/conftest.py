"""Shared fixtures for the test suite.

Both fixtures live here so every module uses the same dataset rather than its own copy:

- ``temperature`` over ``("time", "lat", "lon")``, sized ``(4, 3, 5)``;
- ``elevation`` over ``("lat", "lon")``, *without* ``time``;
- integer ``arange`` coords.

The second variable, without ``time``, is the one the projection-pushdown rule must not
move past a ``time`` reduction, so the metadata-only modules need it too.

``ds`` has random (seeded) values, for modules that evaluate results. ``schema`` wraps a
zeros dataset of the same shape, for modules that only reason about metadata.
``tests/test_accessor.py`` overrides ``ds`` with a version that adds a non-dimension
coord, to check that rewrites keep auxiliary coords.
"""

import numpy as np
import pytest
import xarray as xr

from xrexpr.schema import SchemaState


@pytest.fixture
def ds() -> xr.Dataset:
    """The canonical dataset, with seeded random values, for modules that evaluate results.

    Returns
    -------
    xarray.Dataset
        ``temperature(time, lat, lon)`` sized ``(4, 3, 5)`` alongside
        ``elevation(lat, lon)``, which is *missing* ``time`` — the variable the
        projection-pushdown rule must refuse to reorder past a ``time`` reduction.
    """
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
    """The canonical schema, for modules that only reason about metadata.

    Returns
    -------
    SchemaState
        The snapshot of a zeros dataset shaped exactly like :func:`ds`, so the two
        fixtures agree on dims, sizes and variables without either evaluating anything.
    """
    ds = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.zeros((4, 3, 5))),
            "elevation": (("lat", "lon"), np.zeros((3, 5))),
        },
        coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(5)},
    )
    return SchemaState.from_dataset(ds)
