# type: ignore
import inspect
import textwrap

import numpy as np
import pytest
import xarray as xr

from xrexpr.cst import InvalidExpressionError


def slow_func(ds: xr.Dataset) -> xr.Dataset:
    return ds.mean(dim="lat").mean(dim="lon").isel(time=0)


def fast_func(ds: xr.Dataset) -> xr.Dataset:
    return ds.isel(time=0).mean(dim="lat").mean(dim="lon")


from xarray.testing import assert_equal

from xrexpr.decorators import peek_rewritten_expr, rewrite_expr


def slow_func(ds: xr.Dataset) -> xr.Dataset:
    return ds.mean(dim="lat").mean(dim="lon").isel(time=0)


def fast_func(ds: xr.Dataset) -> xr.Dataset:
    return ds.isel(time=0).mean(dim="lat").mean(dim="lon")


def test_pushdown_optimization_readme():
    """Test that pushdown optimization works as expected from README examples."""
    # Create a test dataset similar to what would be used in the README
    np.random.seed(42)
    data = np.random.rand(10, 20, 30)  # time, lat, lon

    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(10),
            "lat": np.linspace(-90, 90, 20),
            "lon": np.linspace(-180, 180, 30),
        },
    )

    # Define the slow function from README

    # Test that the results are equivalent
    slow_result = slow_func(ds)
    fast_result = fast_func(ds)
    assert_equal(slow_result, fast_result)

    # Test the rewrite functionality
    rewritten_func = rewrite_expr(slow_func)
    rewritten_result = rewritten_func(ds)

    # Verify that rewritten function produces same result
    assert_equal(slow_result, rewritten_result)
    assert_equal(fast_result, rewritten_result)


@pytest.mark.xfail(reason="CST fails on locally defined functions, indentation error")
def test_local_func_pushdown_optimization_readme():
    """Test that pushdown optimization works as expected from README examples."""
    # Create a test dataset similar to what would be used in the README

    def slow_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.mean(dim="lat").mean(dim="lon").isel(time=0)

    def fast_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.isel(time=0).mean(dim="lat").mean(dim="lon")

    np.random.seed(42)
    data = np.random.rand(10, 20, 30)  # time, lat, lon

    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(10),
            "lat": np.linspace(-90, 90, 20),
            "lon": np.linspace(-180, 180, 30),
        },
    )

    # Define the slow function from README

    # Test that the results are equivalent
    slow_result = slow_func(ds)
    fast_result = fast_func(ds)
    assert_equal(slow_result, fast_result)

    # Test the rewrite functionality
    rewritten_func = rewrite_expr(slow_func)
    rewritten_result = rewritten_func(ds)

    # Verify that rewritten function produces same result
    assert_equal(slow_result, rewritten_result)
    assert_equal(fast_result, rewritten_result)


def test_peek_rewritten_expr():
    """Test that peek_rewritten_expr shows the expected reordering."""

    def slow_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.mean(dim="lat").mean(dim="lon").isel(time=0)

    def fast_func(ds: xr.Dataset) -> xr.Dataset:
        return ds.isel(time=0).mean(dim="lat").mean(dim="lon")

    # Get the rewritten expression
    rewritten_code = peek_rewritten_expr(slow_func)

    fast_src = inspect.getsource(fast_func.__code__)
    fast_src = textwrap.dedent(fast_src)
    assert rewritten_code.replace("def slow_func", "def fast_func") == fast_src


def test_multiple_operations_pushdown():
    """Test pushdown with multiple different operations."""
    # Create test dataset
    np.random.seed(123)
    data = np.random.rand(5, 10, 15, 8)  # time, lat, lon, level

    ds = xr.Dataset(
        {"var": (("time", "lat", "lon", "level"), data)},
        coords={
            "time": np.arange(5),
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(-180, 180, 15),
            "level": np.arange(8),
        },
    )

    # Complex operation that could benefit from reordering
    def complex_func(ds: xr.Dataset) -> xr.Dataset:
        return (
            ds.mean(dim="lat").sum(dim="level").isel(time=slice(0, 3)).mean(dim="lon")
        )

    # Test that rewritten version produces same result
    original_result = complex_func(ds)
    rewritten_func = rewrite_expr(complex_func)
    rewritten_result = rewritten_func(ds)

    assert_equal(original_result, rewritten_result)


def test_no_optimization_needed():
    """Test case where no optimization is needed (already optimal)."""

    def already_optimal(ds: xr.Dataset) -> xr.Dataset:
        return ds.isel(time=0).mean(dim="lat").mean(dim="lon")

    # Create test dataset
    np.random.seed(456)
    data = np.random.rand(10, 5, 8)
    ds = xr.Dataset({"data": (("time", "lat", "lon"), data)})

    original_result = already_optimal(ds)
    rewritten_func = rewrite_expr(already_optimal)
    rewritten_result = rewritten_func(ds)


def test_unoptimisable_isel():
    """Test case where isel cannot be pushed down due to dimension mismatch."""

    def unoptimisable(ds: xr.Dataset) -> xr.Dataset:
        # This gets rid of lon, so we can't optimise it. It is in fact
        # an invalid expression.
        return ds.mean(dim="lon").isel(lon=0)

    np.random.seed(42)
    data = np.random.rand(10, 20, 30)  # time, lat, lon

    ds = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(10),
            "lat": np.linspace(-90, 90, 20),
            "lon": np.linspace(-180, 180, 30),
        },
    )

    with pytest.raises(InvalidExpressionError):
        rewritten_func = rewrite_expr(unoptimisable)
