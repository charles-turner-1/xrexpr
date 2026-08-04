# xrexpr

## Environment: pixi, not uv

The project environment is **pixi** (configured in `pyproject.toml` under `[tool.pixi.*]`,
locked by `pixi.lock`), and it is what CI runs — `.github/workflows/ci.yml` uses
`setup-pixi` and `pixi run -e <env> test-cov`. Run everything through it:

| Task | Command |
| --- | --- |
| Tests | `pixi run test` |
| Tests with coverage | `pixi run test-cov` |
| A single file / test | `pixi run pytest tests/test_accessor.py -q` |
| A scratch snippet | `pixi run python -c "..."` |
| Lint | `pixi run ruff` |
| Format | `pixi run black` |
| Types | `pixi run mypy` |
| Hooks | `pixi run pre-commit` |

The default environment matches `test-py313` plus dev tooling and numbagg. The other
`[tool.pixi.environments]` entries exist to pin the matrix CI runs; reach for them
(`pixi run -e test-py310 test`, `pixi run -e test-nonumbagg test`) only when the change is
version- or optional-dependency-sensitive. `dask` is a *test* dependency only — xrexpr must
keep working without it, and the rechunk tests `importorskip` accordingly.
