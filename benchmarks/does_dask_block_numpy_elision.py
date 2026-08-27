# /// script
# requires-python = "==3.12"
# dependencies = [
#   "dask",
#   "numpy",
#   "pandas",
#   "tabulate",
# ]
# ///
"""Check whether dask prevents NumPy temporary elision inside a single chunk."""

from __future__ import annotations

# ruff: noqa: D101,D103
import argparse
import gc
import multiprocessing as mp
import platform
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dask.array as da
import numpy as np
import pandas as pd

CaseFn = Callable[[np.ndarray], tuple[str, Callable[[], Any]]]


@dataclass(frozen=True)
class CaseResult:
    case: str
    seconds: float
    peak_mb: float
    result: float


def pure_numpy(x: np.ndarray) -> tuple[str, Callable[[], Any]]:
    return "numpy: (x * 1.8 + 32).mean()", lambda: (x * 1.8 + 32).mean()


def pure_numpy_inplace(x: np.ndarray) -> tuple[str, Callable[[], Any]]:
    def run() -> Any:
        out = x.copy()
        out *= 1.8
        out += 32
        return out.mean()

    return "numpy: copy; *=; +=; mean", run


def dask_one_chunk_optimized(x: np.ndarray) -> tuple[str, Callable[[], Any]]:
    dx = da.from_array(x, chunks=x.shape)
    expr = ((dx * 1.8) + 32).mean()
    return (
        "dask one chunk: optimized",
        lambda: expr.compute(scheduler="single-threaded", optimize_graph=True),
    )


def dask_one_chunk_unoptimized(x: np.ndarray) -> tuple[str, Callable[[], Any]]:
    dx = da.from_array(x, chunks=x.shape)
    expr = ((dx * 1.8) + 32).mean()
    return (
        "dask one chunk: optimize_graph=False",
        lambda: expr.compute(scheduler="single-threaded", optimize_graph=False),
    )


def dask_one_chunk_augmented_assignment_optimized(
    x: np.ndarray,
) -> tuple[str, Callable[[], Any]]:
    dx = da.from_array(x, chunks=x.shape)
    expr = dx.copy()
    expr *= 1.8
    expr += 32
    mean = expr.mean()
    return (
        "dask one chunk: copy; *=; +=; optimized",
        lambda: mean.compute(scheduler="single-threaded", optimize_graph=True),
    )


def dask_one_chunk_augmented_assignment_unoptimized(
    x: np.ndarray,
) -> tuple[str, Callable[[], Any]]:
    dx = da.from_array(x, chunks=x.shape)
    expr = dx.copy()
    expr *= 1.8
    expr += 32
    mean = expr.mean()
    return (
        "dask one chunk: copy; *=; +=; optimize_graph=False",
        lambda: mean.compute(scheduler="single-threaded", optimize_graph=False),
    )


def dask_one_chunk_block_local(x: np.ndarray) -> tuple[str, Callable[[], Any]]:
    dx = da.from_array(x, chunks=x.shape)

    def stats(block: np.ndarray) -> np.ndarray:
        f = block * 1.8 + 32
        return np.array([f.sum(dtype=np.float64), f.size], dtype=np.float64)

    expr = da.map_blocks(
        stats,
        dx,
        dtype=np.float64,
        chunks=((2,),),
        drop_axis=(0, 1, 2),
        new_axis=0,
    )
    mean = expr[0] / expr[1]
    return (
        "dask one chunk: block-local sum/count",
        lambda: mean.compute(scheduler="single-threaded", optimize_graph=True),
    )


CASES: tuple[CaseFn, ...] = (
    pure_numpy,
    pure_numpy_inplace,
    dask_one_chunk_optimized,
    dask_one_chunk_unoptimized,
    dask_one_chunk_augmented_assignment_optimized,
    dask_one_chunk_augmented_assignment_unoptimized,
    dask_one_chunk_block_local,
)


def run_case(case_name: str, n: int, seed: int, queue: mp.Queue[CaseResult]) -> None:
    case_by_name = {case.__name__: case for case in CASES}
    case = case_by_name[case_name]

    rng = np.random.default_rng(seed)
    x = rng.random((n, n, n), dtype=np.float64)
    label, run = case(x)

    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = float(np.asarray(run()))
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    queue.put(
        CaseResult(
            case=label,
            seconds=seconds,
            peak_mb=peak / 1e6,
            result=result,
        )
    )


def measure(case: CaseFn, n: int, seed: int) -> CaseResult:
    context = mp.get_context("spawn")
    queue: mp.Queue[CaseResult] = context.Queue()
    process = context.Process(target=run_case, args=(case.__name__, n, seed, queue))
    process.start()
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(f"{case.__name__} failed with exit code {process.exitcode}")
    return queue.get()


def result_table(results: list[CaseResult], input_mb: float) -> pd.DataFrame:
    table = pd.DataFrame([result.__dict__ for result in results])
    baseline_peak = float(
        table.loc[table["case"].str.startswith("numpy:"), "peak_mb"].iloc[0]
    )
    table["peak_vs_numpy"] = table["peak_mb"] / baseline_peak
    table["peak_vs_input"] = table["peak_mb"] / input_mb
    return table.sort_values("peak_mb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NumPy temporary elision with one-chunk dask execution."
    )
    parser.add_argument("--n", type=int, default=400, help="cube edge length")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_mb = args.n**3 * 8 / 1e6
    print(f"Array: {args.n}^3 float64 = {input_mb:.1f} MB")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(
        "tracemalloc starts after the input array and dask graph are built; each case runs in a fresh process.\n"
    )

    results = [measure(case, args.n, args.seed) for case in CASES]
    table = result_table(results, input_mb)
    print(
        table.to_markdown(
            index=False,
            floatfmt=(".6f", ".1f", ".12g", ".2f", ".2f"),
        )
    )


if __name__ == "__main__":
    main()
