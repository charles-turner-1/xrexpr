"""Run the marimo daskexpr map-reduce benchmark table linearly.

This is intentionally a direct translation of the main timing table in
``benchmarks/mapreduce_with_daskexpr.py``. It builds the stock-dask expressions first,
then registers dask-array and builds only the dask-array rewrite expression.

Run with:

    python benchmarks/mapreduce_marimo_order.py
"""

from __future__ import annotations

# ruff: noqa: D103
import argparse
import time
from typing import Any

import dask.array as da
import dask_array as dexpr
import numpy as np
import xarray as xr


def build_data(n: int, chunk_edge: int, seed: int) -> xr.DataArray:
    shape = (n, n, n)
    chunks = (chunk_edge, chunk_edge, chunk_edge)
    random = da.random.RandomState(seed)
    data = xr.DataArray(
        random.random_sample(shape, chunks=chunks),
        dims=("time", "y", "x"),
        name="temperature_c",
    )
    data = data.persist()
    data.compute()
    return data


def build_naive(da_x: xr.DataArray) -> Any:
    expression = ((da_x * 1.8) + 32).mean()
    expression.compute()
    return expression


def build_naive_numpy_elide(da_x: xr.DataArray) -> Any:
    expression = (da_x * np.float64(1.8) + np.int64(32)).mean()
    expression.compute()
    return expression


def build_chunk_local(da_x: xr.DataArray) -> Any:
    def fahrenheit_stats(block: np.ndarray) -> np.ndarray:
        f = block * 1.8 + 32
        return np.array([f.sum(dtype=np.float64), f.size], dtype=np.float64).reshape(
            2, 1, 1, 1
        )

    stats_chunks: tuple[tuple[int, ...], ...] = (
        (2,),
        tuple(1 for _ in da_x.data.chunks[0]),
        tuple(1 for _ in da_x.data.chunks[1]),
        tuple(1 for _ in da_x.data.chunks[2]),
    )
    stats = da.map_blocks(
        fahrenheit_stats, da_x.data, dtype=np.float64, chunks=stats_chunks, new_axis=0
    )
    expression = stats[0].sum() / stats[1].sum()
    expression.compute()
    return expression


def build_algebraic(da_x: xr.DataArray) -> Any:
    expression = da_x.mean() * 1.8 + 32
    expression.compute()
    return expression


def build_inplace(da_x: xr.DataArray) -> tuple[Any, int]:
    res = da_x.copy()
    res *= 1.8
    res += 32
    inplace_expr = res.mean()
    inplace_expr_tasks = len(inplace_expr.__dask_graph__())

    def inplace_stats(block: np.ndarray) -> np.ndarray:
        f = block.copy()
        f *= 1.8
        f += 32
        return np.array([f.sum(dtype=np.float64), f.size], dtype=np.float64).reshape(
            2, 1, 1, 1
        )

    stats_chunks = (
        (2,),
        tuple(1 for _ in da_x.data.chunks[0]),
        tuple(1 for _ in da_x.data.chunks[1]),
        tuple(1 for _ in da_x.data.chunks[2]),
    )
    stats = da.map_blocks(
        inplace_stats, da_x.data, dtype=np.float64, chunks=stats_chunks, new_axis=0
    )
    expression = stats[0].sum() / stats[1].sum()
    expression.compute()
    return expression, inplace_expr_tasks


def build_dask_array_rewrite(da_x: xr.DataArray) -> Any:
    dexpr.xarray.register()
    chunks = tuple(axis[0] for axis in da_x.data.chunks)
    dexpr_x = xr.DataArray(
        dexpr.from_array(da_x.data.compute(), chunks=chunks),
        dims=da_x.dims,
        name=da_x.name,
    ).persist()
    dexpr_x.compute()

    expression = ((dexpr_x * 1.8) + 32).mean()
    expression.compute()
    return expression


def benchmark(name: str, expression: Any, repeats: int) -> dict[str, object]:
    result, times = None, []
    for _ in range(repeats):
        start = time.perf_counter()
        result = float(np.asarray(expression.compute()))
        times.append(time.perf_counter() - start)
    return {
        "case": name,
        "best_s": min(times),
        "mean_s": float(np.mean(times)),
        "result": result,
    }


def build_rows(da_x: xr.DataArray, repeats: int) -> list[dict[str, object]]:
    naive = build_naive(da_x)
    naive_maybe_numpy_elided = build_naive_numpy_elide(da_x)
    blockwise_mean = build_chunk_local(da_x)
    algebraic = build_algebraic(da_x)
    dask_array_mean = build_dask_array_rewrite(da_x)
    inplace, inplace_expr_tasks = build_inplace(da_x)

    assert inplace_expr_tasks == len(naive.__dask_graph__()), (
        inplace_expr_tasks,
        len(naive.__dask_graph__()),
    )

    rows = [
        benchmark("naive", naive, repeats),
        benchmark("naive (numpy elide)", naive_maybe_numpy_elided, repeats),
        benchmark("chunk-local (Charles Optimise - Mapreduce)", blockwise_mean, repeats),
        benchmark("inplace (Deepak Optimise )", inplace, repeats),
        benchmark("algebraic (control)", algebraic, repeats),
        benchmark("dask-array (rewrite)", dask_array_mean, repeats),
    ]
    results = [row["result"] for row in rows]
    if not np.allclose(results, results[0], rtol=1e-6):
        raise AssertionError(f"benchmark results disagree: {results}")
    for row in rows:
        row["speedup_vs_naive"] = rows[0]["best_s"] / row["best_s"]
    return rows


def markdown_table(rows: list[dict[str, object]]) -> str:
    columns = ("case", "best_s", "mean_s", "result", "speedup_vs_naive")
    rendered_rows = [format_row(row, columns) for row in rows]
    widths = [
        max(len(column), *(len(rendered[column]) for rendered in rendered_rows))
        for column in columns
    ]
    lines = [
        "| "
        + " | ".join(pad(column, width) for column, width in zip(columns, widths))
        + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for rendered in rendered_rows:
        lines.append(
            "| "
            + " | ".join(
                pad(rendered[column], width) for column, width in zip(columns, widths)
            )
            + " |"
        )
    return "\n".join(lines)


def format_row(row: dict[str, object], columns: tuple[str, ...]) -> dict[str, str]:
    return {column: format_cell(column, row[column]) for column in columns}


def format_cell(column: str, value: object) -> str:
    if column in {"best_s", "mean_s"}:
        return f"{float(value):.6f}"
    if column == "speedup_vs_naive":
        return f"{float(value):.2f}x"
    if column == "result":
        return f"{float(value):.12g}"
    return str(value).replace("|", "\\|")


def pad(value: str, width: int) -> str:
    return value + " " * (width - len(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the marimo map-reduce benchmark table linearly."
    )
    parser.add_argument("--n", type=int, default=400, help="cube edge length")
    parser.add_argument(
        "--chunk-edge", type=int, default=200, help="cubic chunk edge length"
    )
    parser.add_argument("--repeats", type=int, default=5, help="timed runs per case")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nbytes = args.n**3 * 8
    chunk_nbytes = args.chunk_edge**3 * 8
    print(
        f"Array: {args.n}^3 = {nbytes / 1e9:.2f} GB; "
        f"chunks: {args.chunk_edge}^3 = {chunk_nbytes / 1e6:.0f} MB; "
        f"repeats: {args.repeats}"
    )
    print()
    da_x = build_data(args.n, args.chunk_edge, args.seed)
    print(markdown_table(build_rows(da_x, args.repeats)))


if __name__ == "__main__":
    main()
