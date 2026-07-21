"""Shared fixtures for the test suite.

Both fixtures were previously duplicated verbatim across four test modules; they are
lifted here unchanged so every module sees the same canonical dataset — one
``temperature`` variable over ``("time", "lat", "lon")`` sized ``(4, 3, 5)``, with
integer ``arange`` coords.

``ds`` carries random (seeded) values for the modules that evaluate results; ``schema``
wraps a zeros dataset of the same shape for the modules that only reason about metadata.
"""

import numpy as np
import pytest
import xarray as xr

from xrexpr.schema import SchemaState


@pytest.fixture
def ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"temperature": (("time", "lat", "lon"), rng.random((4, 3, 5)))},
        coords={
            "time": np.arange(4),
            "lat": np.arange(3),
            "lon": np.arange(5),
        },
    )


@pytest.fixture
def schema() -> SchemaState:
    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.zeros((4, 3, 5)))},
        coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(5)},
    )
    return SchemaState.from_dataset(ds)
