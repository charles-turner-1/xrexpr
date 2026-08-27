# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray[complete]@git+https://github.com/pydata/xarray.git@main",
#   "dask-array",
#   "dask",
#   "tabulate",
#
# ]
# ///
import time
from collections.abc import Callable
from typing import Any, Literal

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

BenchmarkFn = Callable[[xr.DataArray], Any]

type One = Literal[1]
type Two = Literal[2]
type TwoElementArray = np.ndarray[tuple[Two, *tuple[One, ...]], np.float64]


def naive(data: xr.DataArray) -> Any:
    """Do conversion before aggregation. Should be slowest."""
    return ((data * 1.8) + 32).mean().compute()


def algebraic_control(data: xr.DataArray) -> Any:
    """Do conversion after aggregation. Since we only have a single element array
    coming out of"""
    return (data.mean() * 1.8 + 32).compute()


def inplace_simple(data: xr.DataArray) -> Any:
    """
    Deepak's suggestion, straight up - no block kernel or anything, just do the
    operations in place.

    In theory, I think this should force everything to be done in place, but it
    still has the overhead of having to spit a full chunk out of each block,
    rather than just a 2 element array.
    """
    data *= 1.8
    data += 32
    return data.mean().compute()


def block_local_mapreduce(data: xr.DataArray) -> Any:
    stats = map_blocks_for(data)(
        _elide_kernel,
        data.data,
        dtype=np.float64,
        chunks=mean_and_count_chunks(data),
        new_axis=0,
    )
    return (stats[0].sum() / stats[1].sum()).compute()


def inplace_block_kernel(data: xr.DataArray) -> Any:
    stats = map_blocks_for(data)(
        _inplace_kernel,
        data.data,
        dtype=np.float64,
        chunks=mean_and_count_chunks(data),
        new_axis=0,
    )
    return (stats[0].sum() / stats[1].sum()).compute()


# For both of the _..._kernel functions, the only data coming out of each block
# is a 2 element array, rather than a full block. This should reduce the amount
# of data moving between blocks & speed things up a lot.
def _elide_kernel(block: np.ndarray) -> TwoElementArray:
    """Numpy should elide the intermediates here.

    Question - numpy should stop elision when chunks are are larger than 256KiB"""
    f = block * 1.8 + 32
    return np.array([f.sum(dtype=np.float64), f.size], dtype=np.float64).reshape(
        (2,) + (1,) * block.ndim
    )
    # (2,) is the new axis we specify in map_blocks


def _inplace_kernel(block: np.ndarray) -> TwoElementArray:
    """Deepak's suggestion, applied as a block kernel. Manually forces the operations to be in place."""
    f = block.copy()
    f *= 1.8
    f += 32
    return np.array([f.sum(dtype=np.float64), f.size], dtype=np.float64).reshape(
        (2,) + (1,) * block.ndim
    )


BENCHMARKS: tuple[tuple[str, BenchmarkFn], ...] = (
    ("naive", naive),
    ("block-local (Charles Version)", block_local_mapreduce),
    ("inplace (Deepak Version)", inplace_simple),
    ("inplace (Deepak Version, block local)", inplace_block_kernel),
    ("algebraic (control)", algebraic_control),
)


def map_blocks_for(data: xr.DataArray) -> Callable[..., Any]:
    """If the data is a dask-array, use the dask-array version of map_blocks, otherwise use the stock dask.array version."""
    import dask_array as new_dask_array

    if isinstance(data.data, new_dask_array.Array):
        return new_dask_array.map_blocks
    return da.map_blocks


def mean_and_count_chunks(data: xr.DataArray) -> tuple[tuple[int, ...], ...]:
    """Insert the extra dim as dim zero, with length two. Gets filled with sum, count. This is then each chunk in our
    post reduction array."""
    return ((2,),) + tuple(tuple(1 for _ in axis) for axis in data.data.chunks)


def build_stock_data(n: int, chunk_edge: int, seed: int) -> xr.DataArray:
    shape = (n, n, n)
    chunks = (chunk_edge, chunk_edge, chunk_edge)
    random = da.random.RandomState(seed)
    data = xr.DataArray(
        random.random_sample(shape, chunks=chunks),
        dims=("time", "y", "x"),
        name="temperature_c",
    ).persist()
    data.compute()  # Pregenerate the random array but keep data lazy for the benchmarks.
    return data


def register_dask_array() -> None:
    import dask_array as dexpr

    dexpr.xarray.register()


def build_dask_array_data(stock_data: xr.DataArray) -> xr.DataArray:
    import dask_array as dexpr

    chunks = tuple(axis[0] for axis in stock_data.data.chunks)
    data = xr.DataArray(
        dexpr.from_array(stock_data.data.compute(), chunks=chunks),
        dims=stock_data.dims,
        name=stock_data.name,
    ).persist()
    data.compute()
    return data


def benchmark_phase(phase: str, data: xr.DataArray, repeats: int) -> pd.DataFrame:
    rows = []

    for name, run_benchmark in BENCHMARKS:
        mutates_wrapper = run_benchmark is inplace_simple
        if mutates_wrapper:
            # The expression-level inplace case mutates the xarray wrapper. Prebuild
            # fresh wrappers around the same lazy array so allocation is not mixed into
            # the timing loop and later benchmarks still see the original Celsius data.
            warmup_data = data.copy(deep=False)
            run_inputs = [data.copy(deep=False) for _ in range(repeats)]
        else:
            warmup_data = data
            run_inputs = [data] * repeats

        run_benchmark(warmup_data)

        result, times = None, []
        for run_data in run_inputs:
            start = time.perf_counter()
            result = float(np.asarray(run_benchmark(run_data)))
            times.append(time.perf_counter() - start)

        rows.append(
            (
                phase,
                name,
                min(times),
                float(np.mean(times)),
                float(np.median(times)),
                result,
            )
        )

    table = pd.DataFrame(
        rows,
        columns=["phase", "case", "best_s", "mean_s", "median_s", "result"],
    )
    if not np.allclose(table["result"], table["result"].iloc[0], rtol=1e-6):
        raise AssertionError(f"{phase} benchmark results disagree:\n{table}")
    return table


def markdown_table(table: pd.DataFrame) -> str:
    table = table.drop(columns=["sort_speedup"])
    return table.to_markdown(
        index=False,
        floatfmt=(".6f", ".6f", ".6f", ".12g"),
    )


def main() -> None:
    n = 400
    chunk_edge = 200
    repeats = 5

    nbytes = n**3 * 8
    chunk_nbytes = chunk_edge**3 * 8

    print(
        f"Array: {n}^3 = {nbytes / 1e9:.2f} GB; "
        f"chunks: {chunk_edge}^3 = {chunk_nbytes / 1e6:.0f} MB; "
        f"repeats: {repeats}"
    )
    print("Note: stock means dask.array, not dask-array.\n\n")

    stock_data = build_stock_data(n, chunk_edge, seed=42)
    before = benchmark_phase("before dask-array", stock_data, repeats)
    baseline_median_s = float(before["median_s"].iloc[0])

    register_dask_array()
    stock_after_register = benchmark_phase(
        "stock after dask-array register", stock_data, repeats
    )

    dask_array_data = build_dask_array_data(stock_data)
    after = benchmark_phase("after dask-array", dask_array_data, repeats)

    table = pd.concat([before, stock_after_register, after], ignore_index=True)
    table["sort_speedup"] = baseline_median_s / table["median_s"]
    table["speedup_vs_pre_dask_naive"] = table["sort_speedup"].map(
        lambda speedup: f"{speedup:.2f}x"
    )

    if not np.allclose(table["result"], table["result"].iloc[0], rtol=1e-6):
        raise AssertionError(f"benchmark results disagree:\n{table}")

    table = table.sort_values("sort_speedup", ascending=False)
    print(markdown_table(table))


if __name__ == "__main__":
    main()
