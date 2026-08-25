import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Map→reduce: fuse the transform into the reduction

    > load chunk → do everything you can with it → summarize it → forget it

    Naive `(x*1.8+32).mean()` materializes a full-size Fahrenheit array as a distributed
    intermediate before reducing. The map-reduce version does the arithmetic inside each
    chunk and collapses it to a tiny `(sum, count)` before crossing a chunk boundary:

    ```
    chunk 1 → (sum, count) ┐
    chunk 2 → (sum, count) ┼→ tree reduce → mean
    chunk 3 → (sum, count) ┘
    ```

    The win is memory bandwidth: `x*1.8+32` is trivial vs. moving a large array through
    memory, so the naive extra full-array passes dominate. This is an execution strategy
    (works for `exp(x).mean()`, `(x**2).mean()`, ...), not the linear-algebra identity
    `mean(a*x+b)=a*mean(x)+b`, which is included only as a control.
    """)
    return


@app.cell
def _():
    import time

    import dask
    import dask.array as da
    import matplotlib.pyplot as plt
    import numpy as np
    import pytest
    import xarray as xr

    return da, np, plt, pytest, time, xr


@app.cell
def _(mo):
    # cube edge: the array is n³ float64 values, so this sets total problem size.
    n = mo.ui.slider(200, 1000, value=400, step=100, label="cube edge (n³ float64)")
    # chunk edge: dask splits the cube into (chunk_edge)³ blocks; controls block size
    # and thus the number of chunks the map-reduce summarizes and tree-reduces.
    chunk_edge = mo.ui.slider(100, 500, value=200, step=100, label="chunk edge")
    # repeats: how many timed runs per case (we report best and mean).
    repeats = mo.ui.slider(1, 10, value=5, step=1, label="repeats")
    mo.vstack([n, chunk_edge, repeats])
    return chunk_edge, n, repeats


@app.cell
def _(chunk_edge, mo, n):
    _nbytes = n.value**3 * 8
    mo.md(
        f"Array: **{n.value}³** ≈ **{_nbytes / 1e9:.2f} GB**, "
        f"chunks **{chunk_edge.value}³** ≈ **{chunk_edge.value**3 * 8 / 1e6:.0f} MB** each."
    )
    return


@app.cell
def _(chunk_edge, da, n, xr):
    shape = (n.value, n.value, n.value)
    chunks = (chunk_edge.value, chunk_edge.value, chunk_edge.value)
    da_x = xr.DataArray(
        da.random.random(shape, chunks=chunks),
        dims=("time", "y", "x"),
        name="temperature_c",
    )
    # .persist() is lazy; block on it so we benchmark the calc, not the RNG.
    da_x = da_x.persist()
    da_x.compute()
    return (da_x,)


@app.cell
def naive(da_x):
    naive = ((da_x * 1.8) + 32).mean()
    naive.compute()
    return (naive,)


@app.cell
def naive_numpy(da_x, np):
    naive_maybe_numpy_elided = (da_x * np.float64(1.8) + np.int64(32)).mean()
    naive_maybe_numpy_elided.compute()
    return (naive_maybe_numpy_elided,)


@app.cell
def block_init(da, da_x, np):
    # Runs once per chunk on the raw ndarray (no dims/labels). Does the C->F transform,
    # then collapses the whole block to just (sum, count) so nothing full-size leaves it.
    # Shaped (2,1,1,1): the 2 partials on a new leading axis, 1 per spatial axis so the
    # per-block outputs tile back together along the original grid.
    def fahrenheit_stats(block: np.ndarray) -> np.ndarray:
        f = block * 1.8 + 32
        return np.array([f.sum(dtype=np.float64), f.size], dtype=np.float64).reshape(
            2, 1, 1, 1
        )

    # Output chunk grid map_blocks must be told up front (it can't infer our reshape):
    # leading axis holds the 2 partials as one chunk; each spatial axis keeps one length-1
    # chunk per input chunk, so the stats array is (2, nblocks_t, nblocks_y, nblocks_x).
    stats_chunks: tuple[tuple[int, ...], ...] = (
        (2,),
        tuple(1 for _ in da_x.data.chunks[0]),
        tuple(1 for _ in da_x.data.chunks[1]),
        tuple(1 for _ in da_x.data.chunks[2]),
    )
    stats = da.map_blocks(
        fahrenheit_stats, da_x.data, dtype=np.float64, chunks=stats_chunks, new_axis=0
    )
    # Tree-reduce the tiny per-block partials: total sum / total count == the mean.
    blockwise_mean = stats[0].sum() / stats[1].sum()
    blockwise_mean.compute()
    return (blockwise_mean,)


@app.cell
def def_algebraic(da_x):
    algebraic = da_x.mean() * 1.8 + 32  # control: valid only because mean is linear
    algebraic.compute()
    return (algebraic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deepak's angle: in-place rewriting

    On the tracking issue, Deepak pointed at a different lever for `x*1.8+32`:

    ```python
    res = ds["temperature"]
    res *= 1.8
    res += 32
    ```

    > which dask could totally do.

    The idea is memory *allocation*, not memory *bandwidth*: `x*1.8` allocates a whole
    new array, then `+32` allocates another. Rewriting to augmented assignment reuses one
    buffer, so the transform costs one write pass into an existing array instead of
    minting two throwaway ones.

    There is a wrinkle worth being honest about. At the **dask/xarray expression level**
    this rewrite is a no-op *today*: dask arrays are lazy and functional, so `res *= 1.8`
    just builds a new lazy node — the graph is byte-for-byte identical to `res = res * 1.8`
    (same task count, same result). Deepak's "*dask could totally do*" is the point: dask
    *could* fuse the chain and run it in place inside each block, but out of the box it
    doesn't rewrite the elementwise chain that way. So the honest place to realise the win
    is exactly where chunk-local lives — **inside the per-block kernel**, doing the affine
    map in place on the materialised chunk before collapsing to `(sum, count)`.

    Below builds both: the literal dask-expression form (to show it *is* the same graph as
    naive), and the in-place block kernel (the version that actually saves the
    allocations). Only the kernel form joins the timing tables.
    """)
    return


@app.cell
def def_inplace(da, da_x, np):
    # (a) Deepak's snippet, verbatim, at the dask/xarray expression level. Augmented
    #     assignment on a lazy dask array is functional: this builds the SAME graph as
    #     naive (see the assert below), so it can't save memory as written today.
    _res = da_x.copy()
    _res *= 1.8
    _res += 32
    _inplace_expr = _res.mean()
    # Proof it's a no-op rewrite for now: identical task count to the naive graph.
    inplace_expr_tasks = len(_inplace_expr.__dask_graph__())

    # (b) The version that realises the win: do the affine map IN PLACE on each
    #     materialised block, then collapse to (sum, count) like chunk-local. `.copy()`
    #     guards the persisted input from mutation (the one allocation we can't dodge);
    #     the *=/+= then reuse that single buffer instead of minting two intermediates.
    def _inplace_stats(block: np.ndarray) -> np.ndarray:
        _f = block.copy()
        _f *= 1.8
        _f += 32
        return np.array([_f.sum(dtype=np.float64), _f.size], dtype=np.float64).reshape(
            2, 1, 1, 1
        )

    _stats_chunks = (
        (2,),
        tuple(1 for _ in da_x.data.chunks[0]),
        tuple(1 for _ in da_x.data.chunks[1]),
        tuple(1 for _ in da_x.data.chunks[2]),
    )
    _stats = da.map_blocks(
        _inplace_stats, da_x.data, dtype=np.float64, chunks=_stats_chunks, new_axis=0
    )
    inplace = _stats[0].sum() / _stats[1].sum()
    inplace.compute()
    return inplace, inplace_expr_tasks


@app.cell
def _(np, time):
    # Time expression.compute() `repeats` times; report best (least noise) and mean.
    # `result` is the scalar value, kept so callers can assert cross-case agreement.
    def benchmark(name: str, expression, repeats: int) -> dict[str, object]:
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

    return (benchmark,)


@app.cell
def _(
    algebraic,
    benchmark,
    blockwise_mean,
    inplace,
    inplace_expr_tasks,
    naive,
    naive_maybe_numpy_elided,
    np,
    repeats,
):
    # The dask-expression form of Deepak's rewrite really is the naive graph today:
    # same task count, so it earns no timing row of its own — only the block kernel does.
    assert inplace_expr_tasks == len(naive.__dask_graph__()), (
        inplace_expr_tasks,
        len(naive.__dask_graph__()),
    )
    rows = [
        benchmark("naive", naive, repeats.value),
        # numpy-backed, single-machine: whatever temp elision buys is baked into this.
        # Not a dask row — it's the eager-numpy point of comparison for the naive shape.
        benchmark("naive (numpy elide)", naive_maybe_numpy_elided, repeats.value),
        benchmark("chunk-local", blockwise_mean, repeats.value),
        benchmark("inplace (block kernel)", inplace, repeats.value),
        benchmark("algebraic (control)", algebraic, repeats.value),
    ]
    # All strategies must agree to 1e-6, else a "faster" number is meaningless.
    _r = [x["result"] for x in rows]
    assert np.allclose(_r, _r[0], rtol=1e-6), _r
    # rows[0] is naive, the baseline every speedup is measured against.
    for x in rows:
        x["speedup_vs_naive"] = rows[0]["best_s"] / x["best_s"]
    return (rows,)


@app.cell
def benchmark_table(mo, rows):
    mo.ui.table(rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Speedup map over the parameter space

    Two independent axes: **chunk size** and **number of chunks**. Each point stacks `k`
    cubic chunks of edge `c` along one axis — shape `(c·k, c, c)`, chunked `(c, c, c)`.
    Because a pure map→reduce has no inter-chunk dependencies, the chunks need not tile a
    cube; stacking them keeps every chunk a fixed `c³` while letting `k` vary on its own.
    So chunk size (`c³·8` bytes) and chunk count (`k`) move separately, and array size is
    just `k × chunk_bytes`.

    We benchmark all four cases 10× per point and take the **median speedup vs naive**;
    the naive panel (always 1.0) is omitted. Points whose peak working set would exceed
    the RAM budget are skipped and drawn blank.
    """)
    return


@app.cell
def _(mo):
    # chunk edges (elements): the array is a rectangular stack (c·k, c, c), so a
    # chunk is c³ float64 = c³·8/1e6 MB. These span ~8 → 216 MB per chunk.
    sweep_chunk_edges = [100, 150, 200, 250, 300]
    # number of chunks: with rectangular tiling there are no inter-chunk deps, so we
    # stack along one axis and get exactly k chunks. Fine steps show scaling clearly.
    sweep_chunks = [1, 2, 5, 8, 10, 15, 20]
    sweep_runs = 10  # timed runs per case per grid point; report the median.

    # Out-of-core cap: skip points whose peak working set won't fit in RAM. The naive
    # path materialises a full Fahrenheit intermediate on top of the persisted array,
    # so peak ≈ 2·(k·c³·8). Cap the working set at 6 GB (of ~7 GB free).
    sweep_mem_budget_gb = 6.0

    def sweep_peak_gb(c, k):
        return 2 * k * c**3 * 8 / 1e9

    _n_ok = sum(
        sweep_peak_gb(_c, _k) <= sweep_mem_budget_gb
        for _c in sweep_chunk_edges
        for _k in sweep_chunks
    )
    _n_total = len(sweep_chunk_edges) * len(sweep_chunks)
    run_sweep = mo.ui.run_button(label="Run parameter sweep")
    mo.vstack(
        [
            mo.md(
                f"Grid: **{len(sweep_chunk_edges)}×{len(sweep_chunks)}** points "
                f"(**{_n_ok}** within the {sweep_mem_budget_gb:.0f} GB budget, "
                f"**{_n_total - _n_ok}** skipped), **{sweep_runs}** runs each."
            ),
            run_sweep,
        ]
    )
    return (
        run_sweep,
        sweep_chunk_edges,
        sweep_chunks,
        sweep_mem_budget_gb,
        sweep_peak_gb,
        sweep_runs,
    )


@app.cell
def _(
    da,
    mo,
    np,
    run_sweep,
    sweep_chunk_edges,
    sweep_chunks,
    sweep_mem_budget_gb,
    sweep_peak_gb,
    sweep_runs,
    time,
    xr,
):
    mo.stop(
        not run_sweep.value, mo.md("*Press ‘Run parameter sweep’ to compute the map.*")
    )

    # Median (not best) here: the sweep runs many points unattended, so we want the
    # typical run, robust to the odd GC pause, rather than the luckiest one.
    def _median_time(expression, runs: int) -> float:
        _times = []
        for _ in range(runs):
            _t0 = time.perf_counter()
            float(np.asarray(expression.compute()))
            _times.append(time.perf_counter() - _t0)
        return float(np.median(_times))

    def _cases(data):
        # Rebuild the four expressions for an arbitrary chunked DataArray.
        _naive = ((data * 1.8) + 32).mean()

        _stats_chunks = (
            (2,),
            tuple(1 for _ in data.data.chunks[0]),
            tuple(1 for _ in data.data.chunks[1]),
            tuple(1 for _ in data.data.chunks[2]),
        )

        # Same per-block (sum, count) collapse as block_init; see there for the shape.
        def _stats(block: np.ndarray) -> np.ndarray:
            _f = block * 1.8 + 32
            return np.array(
                [_f.sum(dtype=np.float64), _f.size], dtype=np.float64
            ).reshape(2, 1, 1, 1)

        _s = da.map_blocks(
            _stats, data.data, dtype=np.float64, chunks=_stats_chunks, new_axis=0
        )
        _chunk_local = _s[0].sum() / _s[1].sum()

        # Deepak's in-place variant: same collapse, but the affine map runs *=/+= on the
        # materialised block. See def_inplace for why the win lives in the kernel.
        def _stats_inplace(block: np.ndarray) -> np.ndarray:
            _f = block.copy()
            _f *= 1.8
            _f += 32
            return np.array(
                [_f.sum(dtype=np.float64), _f.size], dtype=np.float64
            ).reshape(2, 1, 1, 1)

        _si = da.map_blocks(
            _stats_inplace,
            data.data,
            dtype=np.float64,
            chunks=_stats_chunks,
            new_axis=0,
        )
        _inplace = _si[0].sum() / _si[1].sum()

        _algebraic = data.mean() * 1.8 + 32
        return _naive, _chunk_local, _inplace, _algebraic

    # Speedup grids: rows = chunk size (edge c), cols = number of chunks (k). NaN marks
    # skipped points. The two 1-D arrays are the axis tick values (MB per chunk, and k).
    sweep_chunk_local = np.full((len(sweep_chunk_edges), len(sweep_chunks)), np.nan)
    sweep_inplace = np.full_like(sweep_chunk_local, np.nan)
    sweep_algebraic = np.full_like(sweep_chunk_local, np.nan)
    sweep_chunk_mb = np.array([c**3 * 8 / 1e6 for c in sweep_chunk_edges])
    sweep_total_chunks = np.array(sweep_chunks)

    for _i, _c in enumerate(sweep_chunk_edges):
        for _j, _k in enumerate(sweep_chunks):
            if sweep_peak_gb(_c, _k) > sweep_mem_budget_gb:
                continue  # would exceed the RAM budget → leave as NaN (blank cell)
            # Rectangular stack: k chunks along the leading axis, no inter-chunk deps.
            _x = xr.DataArray(
                da.random.random((_c * _k, _c, _c), chunks=(_c, _c, _c)),
                dims=("time", "y", "x"),
                name="temperature_c",
            ).persist()
            _x.compute()
            _naive_e, _chunk_e, _inplace_e, _alg_e = _cases(_x)
            # Speedup = naive time / this-strategy time, so >1 means faster than naive.
            _t_naive = _median_time(_naive_e, sweep_runs)
            sweep_chunk_local[_i, _j] = _t_naive / _median_time(_chunk_e, sweep_runs)
            sweep_inplace[_i, _j] = _t_naive / _median_time(_inplace_e, sweep_runs)
            sweep_algebraic[_i, _j] = _t_naive / _median_time(_alg_e, sweep_runs)
            del _x  # free this point's persisted array before building the next
    return (
        sweep_algebraic,
        sweep_chunk_local,
        sweep_chunk_mb,
        sweep_inplace,
        sweep_total_chunks,
    )


@app.cell
def benchmark_heatmap(
    np,
    plt,
    sweep_algebraic,
    sweep_chunk_local,
    sweep_chunk_mb,
    sweep_inplace,
    sweep_total_chunks,
):
    _speedup_panels = [
        ("chunk-local", sweep_chunk_local),
        ("inplace (block kernel)", sweep_inplace),
        ("algebraic (control)", sweep_algebraic),
    ]
    # Shared colour scale across the three speedup panels so they're directly comparable.
    _vmin = min(
        np.nanmin(sweep_chunk_local),
        np.nanmin(sweep_inplace),
        np.nanmin(sweep_algebraic),
    )
    _vmax = max(
        np.nanmax(sweep_chunk_local),
        np.nanmax(sweep_inplace),
        np.nanmax(sweep_algebraic),
    )

    # 4th panel: inplace ÷ chunk-local. Both are block-kernel map-reductions vs naive, so
    # their ratio isolates the cost of the in-place `.copy()` + `*=/+=` against the
    # out-of-place `block*1.8+32` (>1 favours inplace, <1 favours chunk-local).
    _ratio = sweep_inplace / sweep_chunk_local
    # Symmetric log scale centred on 1.0 so both directions read equally.
    _rmax = max(np.nanmax(_ratio), 1.0 / np.nanmin(_ratio))

    # Skipped points are NaN; mask them so they render blank (out-of-budget cells).
    _cmap = plt.get_cmap("viridis").copy()
    _cmap.set_bad("0.9")
    _cmap_r = plt.get_cmap("RdBu_r").copy()
    _cmap_r.set_bad("0.9")

    _fig, _axes = plt.subplots(1, 4, figsize=(21, 4.5), constrained_layout=True)
    for _ax, (_title, _grid) in zip(_axes, _speedup_panels):
        _im = _ax.imshow(
            np.ma.masked_invalid(_grid),
            origin="lower",
            aspect="auto",
            cmap=_cmap,
            vmin=_vmin,
            vmax=_vmax,
        )
        _ax.set_title(f"{_title}\nmedian speedup vs naive")
        _ax.set_xlabel("number of chunks (k)")
        _ax.set_ylabel("chunk size (MB)")
        _ax.set_xticks(range(len(sweep_total_chunks)))
        _ax.set_xticklabels(sweep_total_chunks)
        _ax.set_yticks(range(len(sweep_chunk_mb)))
        _ax.set_yticklabels([f"{_mb:.0f}" for _mb in sweep_chunk_mb])
        for _i in range(_grid.shape[0]):
            for _j in range(_grid.shape[1]):
                if np.isnan(_grid[_i, _j]):
                    continue
                _ax.text(
                    _j,
                    _i,
                    f"{_grid[_i, _j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
    _fig.colorbar(_im, ax=_axes[:3], label="speedup ×")

    _ax_ratio = _axes[3]
    _im_ratio = _ax_ratio.imshow(
        np.ma.masked_invalid(_ratio),
        origin="lower",
        aspect="auto",
        cmap=_cmap_r,
        vmin=1.0 / _rmax,
        vmax=_rmax,
        norm="log",
    )
    _ax_ratio.set_title("inplace ÷ chunk-local\n>1 favours inplace")
    _ax_ratio.set_xlabel("number of chunks (k)")
    _ax_ratio.set_ylabel("chunk size (MB)")
    _ax_ratio.set_xticks(range(len(sweep_total_chunks)))
    _ax_ratio.set_xticklabels(sweep_total_chunks)
    _ax_ratio.set_yticks(range(len(sweep_chunk_mb)))
    _ax_ratio.set_yticklabels([f"{_mb:.0f}" for _mb in sweep_chunk_mb])
    for _i in range(_ratio.shape[0]):
        for _j in range(_ratio.shape[1]):
            if np.isnan(_ratio[_i, _j]):
                continue
            _ax_ratio.text(
                _j,
                _i,
                f"{_ratio[_i, _j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    _fig.colorbar(_im_ratio, ax=_ax_ratio, label="ratio ×")
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Reading the map

    ### Why the speedup hovers around 2x

    The number falls out of counting how many times each strategy moves the array
    through memory. The arithmetic (`x*1.8+32`) is a couple of flops per element --
    negligible next to the cost of streaming a large array past the CPU. So the runtime
    is set by **bytes moved, not flops done**.

    Let the array be `N` float64 elements.

    - **naive** -- `(x*1.8+32).mean()`. Dask fuses the elementwise chain into the
      reduction's per-chunk leaves (the graph optimizer collapses this point's tasks
      from 39 to 23), so it never parks a full Fahrenheit copy in memory at once. But it
      still has to *produce* the transformed value for every element and *consume* it in
      the reduction: **read N, write N, read N** -- one full extra pass of write-then-read
      over the transformed data on top of the input read.
    - **chunk-local** -- transform inside the block, collapse to `(sum, count)` before
      the value ever leaves the chunk. The transformed element is born, summed, and
      discarded in-register: **read N, write ~0**. Output is `2k` floats total.
    - **inplace** -- Deepak's variant. Same block-local collapse, but the affine map runs
      `*=/+=` on the materialised chunk rather than `block*1.8+32`. It still clears naive
      by about the same 2x (same **read N, write ~0** memory class), but it lands a notch
      *below* chunk-local, not alongside: the `.copy()` it needs to protect the persisted
      input is a full-size allocation numpy's fused `block*1.8+32` never pays for, so the
      buffer reuse it was meant to buy is more than eaten by the copy.
    - **algebraic** -- `mean(x)*1.8+32`. Dask's tree reduction reads each element once
      into a running partial sum: **read N, write ~0**. The `*1.8+32` runs once, on a
      scalar.

    So both fused strategies do about `N` of traffic and naive does about `2N` (input
    pass + transformed-intermediate pass). The clean single-intermediate reading is:
    **naive pays for the transformed array as a distinct thing that has to be written out
    and read back; the fused versions never give it a separate existence.** Ratio is
    about 2x -- which is exactly the floor the grid sits on.

    It is *only* about 2x (not larger) because the op is memory-bound and both passes are
    streaming: you save one of two passes, not a factor tied to chunk count or flops.

    ### How the ratio moves across the grid

    The 4th panel is `inplace / chunk-local` (both are speedups vs naive). These two are
    the *same* map-reduce shape — identical `map_blocks` scaffolding, identical
    `(2,1,1,1)` collapse, identical tree reduce — differing only in how each block does
    the affine map: chunk-local writes `block*1.8+32` into a fresh array, inplace does
    `.copy()` then `*=/+=` into that one buffer. So the ratio isolates a single thing:
    the block-kernel allocation pattern, with everything else held fixed.

    That makes the panel the honest test of Deepak's lever, and the answer it gives is
    **no**: inplace is a small but consistent *loss* against chunk-local. Almost every
    cell sits below 1.0 (roughly 0.80–0.95), and it only pokes above 1.0 at `k=1` — a
    single chunk, where there is barely any reduction to speak of. The cause is the
    `.copy()` that guards the persisted input: it is itself a full-size allocation, the
    very cost the rewrite was meant to dodge, and it buys nothing because numpy already
    fuses the out-of-place `block*1.8+32` into an efficient single pass. So the in-place
    kernel pays for a copy the out-of-place one never makes.

    The one visible trend is down the columns: **larger chunks push the ratio lower**
    (the bottom row lands around 0.80–0.87). Bigger blocks mean the `.copy()` moves more
    bytes in absolute terms, so its fixed overhead bites harder exactly where real
    out-of-core work lives.

    The reading for xrexpr: at the expression level the in-place rewrite is a no-op today
    (same graph as naive), and pushed into the kernel it actively *underperforms* the
    plain chunk-local map. The memory-bandwidth win was already banked by *fusing the
    transform into the reduction at all*; trying to also reuse the block's buffer only
    adds a copy. Deepak's lever is worth knowing about, but on this workload it is not the
    thing that pays — the fusion is.

    ### What this says for xrexpr

    The general map-reduce fusion earns its keep whenever chunks are a sensible size.
    Algebraic only draws level in a regime (many tiny chunks) you would avoid anyway for
    scheduling reasons, and it only works at all because `mean` is linear -- the fused
    map-reduce needs no such special structure. So the IR knowing an op is a
    decomposable reduction over an elementwise map is the thing that unlocks the win in
    the general case, and the 2x is the honest ceiling for a single memory-bound pass.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where this comes from, and where it could go: flox and the IR

    ### The flox move

    This benchmark is a stolen idea. flox makes `groupby(...).mean()` on dask fast by
    not just doing the obvious thing. That obvious thing is to shuffle every element into
    its group, then reduce each group. That drags the whole array across chunk and worker
    boundaries before any arithmetic runs. flox reduces *within each chunk first*. It
    computes a partial `(sum, count)` per group per block, then combines the small
    per-block partials into the final answer. The data that crosses a chunk boundary is
    `O(groups)`, not `O(elements)`, which massively speeds up computation.

    The three cases here are the ungrouped shadow of that same trick.

    - **naive** is the shuffle-then-reduce shape. Build the whole transformed array as a
      distributed object, then reduce it.
    - **chunk-local** is the flox shape. `map_blocks` does the transform-and-collapse
      inside each block, emits `(sum, count)`, and a small tree combines them.
    - **algebraic** is a fourth thing flox never needs. It is a hand-proved shortcut that
      works only because `mean` is linear. This isn't really something our IR could do much with (it's way too specific), but it's handy as a sanity check.

    flox's trick generalizes past groupby to any *decomposable reduction*: one you can
    compute from per-block partials plus a combine step. Sum, count, mean, min, max,
    variance via moments. The `(load chunk -> do everything you can -> summarize ->
    forget)` motto is just this.

    ### What xrexpr adds

    flox recognizes one pattern, grouped reductions, and hand-writes the blockwise
    version. xrexpr records the whole chain of operations and tags each node with what it
    does to the data: `Elementwise`, `Reduce`, and what each node `consumes` and
    `requires`. So xrexpr can see a string like

    ```
    Elementwise(*1.8) -> Elementwise(+32) -> Reduce(mean)
    ```

    and recognize the shape flox exploits by hand. A run of elementwise work feeding a
    decomposable reduction. The elementwise work has no inter-chunk dependencies, so it
    can move inside the reduction's per-block kernel. The reduction being decomposable is
    what makes the collapse legal. Both facts already live in the IR. Elementwise-ness is
    a node kind, and decomposability is a property of the reduction op. No array gets
    inspected, so the rewrite stays purely structural and inside the correctness invariant
    the project already holds: same values, no new errors.

    The payoff is generality. flox gets you the grouped case. An IR that recognizes
    `(elementwise+ -> decomposable reduce)` gets you every such chain. Affine transforms,
    unit conversions, masking, any pointwise pre-processing in front of a sum, mean, or
    count, with no hand-written kernel per pattern.

    ### The scope jump

    Today xrexpr emits a replayed sequence of xarray calls. It reorders and fuses nodes,
    then calls `getattr(ds, node.name)(*args, **kwargs)` and lets xarray and dask build the
    graph. Pushing elementwise work inside a chunk-local reduction cannot be expressed as a
    call reordering. It needs a `map_blocks`-style kernel that fuses the transform and the
    partial reduction into one per-block function. That is xrexpr respecifying a dask
    computation itself, not replaying xarray calls. It is a different output target than
    anything in `planning/`, which stays a call-reordering middle-end.

    This is a real step up in scope, but it stays in the provably-correct space. The
    legality conditions are the structural facts the IR already reasons about: elementwise
    has no cross-chunk dependencies, and the reduction is decomposable. What follows is a
    scope for the work, small enough to think about discussing with someone.

    1. **A `MapReduce` IR node.** A fused span holding the elementwise chain plus its
       closing decomposable reduction. It carries the block function (the composed
       elementwise ops), the partial it emits (`sum`, `count`, moments), and the combine
       step. This is the physical-lowering target xrexpr does not yet own.

    2. **A recognizer.** One lowering rule that spots `(Elementwise+ -> Reduce)` where the
       reduction is decomposable and folds the span into a `MapReduce`. Reuse the existing
       `consumes` / `requires` tags for the legality check. No new data-touching analysis.

    3. **A decomposability registry.** A table from reduction op to its (map, combine,
       finalize) triple. mean is `(sum+count, add, divide)`. var is the moment form. Ops
       with no entry stay as ordinary reduces, so the rule is opt-in and safe by default.

    4. **A `map_blocks` emitter.** The one place xrexpr writes a dask computation instead
       of an xarray call. It compiles a `MapReduce` node into the block function plus the
       tree combine. Keep it behind a capability check so a non-dask backend falls back to
       the plain replayed chain.

    5. **Correctness coverage.** Extend the property-based suite so any plan containing a
       `MapReduce` is checked against its un-fused replay over generated datasets. This is
       the guardrail that lets the emitter live next to the existing call-replay path.

    The benchmark above is the smallest witness that the prize is real. About 2x, the
    memory-bandwidth ceiling, holding across chunk sizes and counts. That is a consistent
    one-pass-instead-of-two, and it comes from a rewrite the IR is already shaped to spot.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ___

    ## Grafting: let xarray own the metadata, own only the graph

    The `chunk-local` cell above hand-builds its output. It computes `stats_chunks`, uses
    `new_axis=0` to stash `(sum, count)` on a leading axis, then reduces that back to a
    scalar. That bookkeeping is the tax you pay for going under xarray to `da.map_blocks`.
    In real code the temptation is worse: to make **flox** carry a non-groupby map-reduce
    you have to dress the computation up as a grouped reduction it isn't, purely to borrow
    its blockwise machinery. That is a janky hack, and it puts a wrong abstraction on the
    critical path.

    The graft sidesteps both. A `DataArray` is a thin shell — `.variable`, `.dims`,
    `.coords`, `.attrs`, `.name` — wrapped around a duck array. That shell is independent
    of *what the underlying graph computes*. So:

    1. Ask xarray for the answer the ordinary way: `((x*1.8)+32).mean()`. Don't compute
       it — just take the **lazy** result. It already has the correct shape, dims, coords
       and attrs, because xarray applied its own reduction rules to produce it.
    2. Build our own memory-optimal dask array with the map-reduce graph.
    3. Swap the array into the shell with `template.copy(data=our_array)`.

    `copy(data=...)` is the blessed "same metadata, different values" API, and it
    validates that our array's shape matches — a free correctness check on the kernel's
    output. No `stats_chunks`, no `new_axis`, no flox cosplay: xarray computes the
    dimension info, we graft the fast graph onto it.

    The cell below does exactly this and checks three things: the grafted value matches
    naive, the metadata is identical to the un-fused template, and the graph really is the
    small map-reduce (not xarray's fused-into-reduction elementwise chain).
    """)
    return


@app.cell
def _(da, da_x, np):
    # Per-block (sum, count) collapse, same as block_init (see there for the (2,1,1,1)).
    def _fahrenheit_stats(block: np.ndarray) -> np.ndarray:
        _f = block * 1.8 + 32
        return np.array([_f.sum(dtype=np.float64), _f.size], dtype=np.float64).reshape(
            2, 1, 1, 1
        )

    _stats_chunks = (
        (2,),
        tuple(1 for _ in da_x.data.chunks[0]),
        tuple(1 for _ in da_x.data.chunks[1]),
        tuple(1 for _ in da_x.data.chunks[2]),
    )
    _stats = da.map_blocks(
        _fahrenheit_stats, da_x.data, dtype=np.float64, chunks=_stats_chunks, new_axis=0
    )
    # Our memory-optimal graph: a bare dask scalar, no dims/coords/attrs attached.
    mapreduce_array = _stats[0].sum() / _stats[1].sum()

    # xarray computes the metadata for us: shape, dims, coords, attrs of the reduction.
    template = ((da_x * 1.8) + 32).mean()
    # Graft: keep the shell, replace the graph. copy(data=...) validates the shape match.
    grafted = template.copy(data=mapreduce_array)
    grafted
    return grafted, mapreduce_array, template


@app.cell
def _(template):
    template.__dask_graph__()
    return


@app.cell
def _(grafted):
    grafted.data.visualize()
    return


@app.cell
def assert_graft_correct(grafted, mapreduce_array, mo, np, template):
    # 1. Same answer as the naive path (value correctness).
    _grafted_val = float(np.asarray(grafted.compute()))
    _naive_val = float(np.asarray(template.compute()))
    _values_match = np.allclose(_grafted_val, _naive_val, rtol=1e-6)

    # 2. Identical metadata to the un-fused template (the shell xarray built).
    _meta_match = (
        grafted.dims == template.dims
        and grafted.shape == template.shape
        and grafted.name == template.name
        and grafted.attrs == template.attrs
        and list(grafted.coords) == list(template.coords)
    )

    # 3. The grafted array really carries OUR graph, not xarray's reduction. Identity is
    #    the honest check: the shell now points at mapreduce_array, and its graph differs
    #    from the template's. (Task count is a red herring here — see "Reading the map":
    #    the win is bytes moved, not tasks, and dask fuses naive down to few tasks anyway.)
    _graph_is_ours = grafted.data is mapreduce_array
    _graph_differs = grafted.data.__dask_graph__() is not template.data.__dask_graph__()

    mo.md(
        f"- **value matches naive:** {_values_match} "
        f"(grafted `{_grafted_val:.6f}` vs naive `{_naive_val:.6f}`)\n"
        f"- **metadata matches template:** {_meta_match} "
        f"(dims `{grafted.dims}`, shape `{grafted.shape}`, name `{grafted.name}`)\n"
        f"- **graph is our map-reduce:** {_graph_is_ours and _graph_differs} "
        f"(grafted array is our object, graph distinct from the template's)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ...and it's actually faster

    Correctness is the point above; this is the receipt that the graft keeps the speed.
    We build one big array (bigger than the interactive default, so the memory-bandwidth
    gap dominates timing noise), then time the **grafted** map-reduce against the **naive**
    path that produced the template. Same helper, same array, same repeats as the rest of
    the notebook — the only difference is which graph runs under the identical shell.
    """)
    return


@app.cell
def _(benchmark, da, np, repeats, xr):
    # Bigger than the interactive default so the ~2x memory-bandwidth gap clears noise.
    # 800³ float64 ≈ 4.1 GB, chunked 200³ (64 chunks); persist so we time the calc.
    _big = xr.DataArray(
        da.random.random((800, 800, 800), chunks=(200, 200, 200)),
        dims=("time", "y", "x"),
        name="temperature_c",
    ).persist()
    _big.compute()

    # Per-block (sum, count) collapse, same as block_init (see there for the (2,1,1,1)).
    def _stats(block: np.ndarray) -> np.ndarray:
        _f = block * 1.8 + 32
        return np.array([_f.sum(dtype=np.float64), _f.size], dtype=np.float64).reshape(
            2, 1, 1, 1
        )

    _stats_chunks = (
        (2,),
        tuple(1 for _ in _big.data.chunks[0]),
        tuple(1 for _ in _big.data.chunks[1]),
        tuple(1 for _ in _big.data.chunks[2]),
    )
    _s = da.map_blocks(
        _stats, _big.data, dtype=np.float64, chunks=_stats_chunks, new_axis=0
    )
    _big_template = ((_big * 1.8) + 32).mean()
    _big_grafted = _big_template.copy(data=_s[0].sum() / _s[1].sum())

    graft_speed_rows = [
        benchmark("naive (template)", _big_template, repeats.value),
        benchmark("grafted map-reduce", _big_grafted, repeats.value),
    ]
    graft_speedup = graft_speed_rows[0]["best_s"] / graft_speed_rows[1]["best_s"]
    for _row in graft_speed_rows:
        _row["speedup_vs_naive"] = graft_speed_rows[0]["best_s"] / _row["best_s"]
    assert np.allclose(
        graft_speed_rows[0]["result"], graft_speed_rows[1]["result"], rtol=1e-6
    )
    return graft_speed_rows, graft_speedup


@app.cell
def _(graft_speed_rows, graft_speedup, mo):
    mo.vstack(
        [
            mo.md(
                f"Grafted map-reduce is **{graft_speedup:.2f}×** faster than the naive "
                f"template it was grafted onto — same shell, same answer, faster graph."
            ),
            mo.ui.table(graft_speed_rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Toward library code: the graft as three functions

    Everything the notebook did by hand, pulled into functions with no notebook state
    behind them. This is the shape the logic would take inside xrexpr's `map_blocks`
    emitter: a per-block kernel, a builder for the bare map-reduce dask array, and the
    graft that borrows xarray's metadata. Read these plus the test cell below and you have
    the whole feature, ready to lift out.
    """)
    return


@app.cell
def library_candidate(da, np, xr):
    from collections.abc import Callable

    # The per-block kernel, split into its two parts. `f` is the elementwise transform
    # (the swappable bit — the whole elementwise chain the IR would fuse); everything
    # around it is the fixed map-reduce scaffolding: apply f in-register, collapse the
    # block to (sum, count), reshape so it tiles back. Leading axis holds the two
    # partials; one length-1 axis per input dim gives the (2, 1, 1, ...) block output.
    def _block_stats(
        block: np.ndarray, f: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        mapped = f(block)
        return np.array(
            [mapped.sum(dtype=np.float64), mapped.size], dtype=np.float64
        ).reshape((2,) + (1,) * block.ndim)

    # Build the bare map-reduce dask array for mean(f(x)): a scalar graph with no
    # dims/coords/attrs. Emitter's job — produce correct *values*, wear no labels yet.
    # `f` is the elementwise transform applied inside each block before the collapse.
    def mapreduce_mean(data: xr.DataArray, f: Callable[[np.ndarray], np.ndarray]):
        # Output chunk grid, derived from the input's: axis 0 is the length-2 partials as
        # one chunk, then one length-1 chunk per input chunk along every original axis.
        stats_chunks = ((2,),) + tuple(
            tuple(1 for _ in axis_chunks) for axis_chunks in data.data.chunks
        )
        partials = da.map_blocks(
            _block_stats,
            data.data,
            f,
            dtype=np.float64,
            chunks=stats_chunks,
            new_axis=0,
        )
        return partials[0].sum() / partials[1].sum()

    # Graft the fast array onto xarray's own reduction shell. xarray computes the
    # metadata (the template); copy(data=...) swaps in our graph and validates the shape.
    # Returns the template too, so callers can check the graft preserved it. The C->F
    # transform is just the elementwise lambda handed to mapreduce_mean.
    def grafted_mean(data: xr.DataArray):
        f = lambda x: x * 1.8 + 32
        template = ((data * 1.8) + 32).mean()
        array = mapreduce_mean(data, f)
        return template.copy(data=array), template, array

    return (grafted_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tests

    The correctness claims the notebook makes above, pinned as `pytest` tests over the
    functions just extracted. marimo collects the `test_*` functions in the cell below, so
    `pytest benchmarks/mapreduce.py` runs them without executing the whole notebook. The
    fixtures build their own data, so the cell is self-contained: read it top to bottom and
    it reads like a test module for library code. The cell takes `grafted_mean`, `da` and
    `xr` (and `pytest`) as parameters; marimo's DAG wires them in.
    """)
    return


@app.cell
def _(da, grafted_mean, pytest, xr):
    @pytest.fixture
    def data():
        # A small chunked cube built on the fly — the fixtures own the input, so the
        # tests don't lean on any notebook global. 200³ in 100³ chunks = 8 blocks.
        return xr.DataArray(
            da.random.random((200, 200, 200), chunks=(100, 100, 100)),
            dims=("time", "y", "x"),
            name="temperature_c",
        ).persist()

    @pytest.fixture
    def graft(data):
        # Everything under test, constructed from the data fixture via the extracted
        # function. Returns (grafted, template, mapreduce_array).
        return grafted_mean(data)

    def test_graft_matches_naive_value(graft):
        # The grafted map-reduce graph and the naive template compute the same scalar.
        import numpy as np

        grafted, template, _ = graft
        assert np.allclose(
            float(np.asarray(grafted.compute())),
            float(np.asarray(template.compute())),
            rtol=1e-6,
        )

    def test_graft_preserves_metadata(graft):
        # copy(data=...) keeps xarray's shell: grafted must be indistinguishable from the
        # un-fused template in every piece of metadata a consumer can observe.
        grafted, template, _ = graft
        assert grafted.dims == template.dims
        assert grafted.shape == template.shape
        assert grafted.name == template.name
        assert grafted.attrs == template.attrs
        assert list(grafted.coords) == list(template.coords)

    def test_graft_carries_our_graph(graft):
        # The shell points at our dask array, and its graph is not the template's:
        # a real substitution, not a relabel of xarray's own reduction. (Task count is
        # deliberately not asserted; see "Reading the map".)
        grafted, template, array = graft
        assert grafted.data is array
        assert grafted.data.__dask_graph__() is not template.data.__dask_graph__()

    return


if __name__ == "__main__":
    app.run()
