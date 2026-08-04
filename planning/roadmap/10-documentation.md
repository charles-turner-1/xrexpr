# W10 — The documentation site: Sphinx + MyST, the migrated essays, the slimmed docstrings

*(2026-08-04. Promotes the docs job out of the README: the 2026-07-28 checkpoint recorded
"a real 'grouped and windowed operations' section is still owed, and is a docs job rather
than part of any workstream" (`00-assessment.md:222-226`), and #92/#93 paid that debt into
`README.md` because the README was the only rendered artefact there was. This workstream
gives the material a real home. Anchors checked against `main` on this date.)*

**Goal:** a hosted documentation site, in MyST markdown, with four levels — a user-facing
overview of what a plan is and how it is optimised, an explanation of the rewrite rules
and lowering, the internals essays that today live as module docstrings, and a generated
API reference — with the worked examples *executed at build time* so `explain()` output
in the docs can never drift from the code. Then, once the essays have a rendered home,
slim the module docstrings that made the source hard to read.

**Size:** 8 PRs. PRs 1–7 are additive and can interleave with open code work; PR 8
(the slimming) is destructive of in-code prose and lands last.

## Why a site, and why now

Roughly 55–70% of every module in `src/xrexpr/` is docstring. The *callable* docstrings
are enforced structure (ruff `D` with `convention = "numpy"`, `pyproject.toml:152-156`;
the `numpydoc-validation` hook, `pyproject.toml:241-256`) and they stay. The
*module-level* docstrings are something else: full design essays — the list-not-tree
argument (`ir.py:1-54`), the lowering contract (`lower.py:1-38`), the rule catalogue and
trust boundary (`optimize.py:1-51`), the two value taxonomies (`indexers.py:1-29`,
`chunks.py:1-52`). They are the best writing in the project and they are in the worst
place for it: unrendered, unlinkable, and sitting between the reader and the code.

The user-facing story has the same problem one level up: `README.md` carries the pitch,
the quickstart, the rewrite table *and* four `<details>` blocks of internals
(`README.md:190-281`) because there has been nowhere else to put them.

The fix is the standard one: a Sphinx site under `docs/`, MyST markdown throughout,
executed examples, hosted on Read the Docs. The essays migrate to `internals/` pages,
the README's collapsed blocks become links, and the module docstrings shrink to an
orientation paragraph plus a pointer.

## The toolchain, pinned

Do not relitigate these choices; each is one line of rationale away from its alternative.

| choice | pick | why |
|---|---|---|
| builder | `sphinx>=8` | current major; everything below supports it |
| markdown + execution | `myst-nb>=1.2` | MyST pages *and* executed code cells in one extension (it bundles `myst-parser` — do not list both) |
| theme | `pydata-sphinx-theme>=0.16` | xarray's own theme; "looks like the xarray docs" is the brief |
| page furniture | `sphinx-design`, `sphinx-copybutton` | landing-page cards; copy buttons on code blocks |
| docstring rendering | `sphinx.ext.autodoc` + `sphinx.ext.autosummary` + `sphinx.ext.napoleon` (numpy mode) | the docstrings are numpydoc-sectioned rST; napoleon translates and autodoc renders the ~340 `:func:`/`:class:` roles natively. **Not** the `numpydoc` Sphinx extension — validation is already the pre-commit hook's job, and two owners of one convention fight |
| diagrams | `sphinxcontrib-mermaid>=1.0` | MyST has no native mermaid; the `{mermaid}` directive works inside MyST pages and renders client-side |
| cross-refs | `sphinx.ext.intersphinx` (python, numpy, pandas, xarray, dask) + `sphinx.ext.viewcode` | the essays constantly point at xarray/dask concepts and at this package's own source |

**One dependency list, by construction.** pixi reads this repo's manifest from
`pyproject.toml` and exposes each `[project.optional-dependencies]` group as a feature —
that is how the existing `dev` extra (`pyproject.toml:34-45`) and `dev` feature
(`pyproject.toml:209-220`) merge. So the docs dependencies are declared **once**, as an
extra, and both pixi (locally, in CI) and Read the Docs (via pip) consume it:

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=8",
    "myst-nb>=1.2",
    "pydata-sphinx-theme>=0.16",
    "sphinx-design",
    "sphinx-copybutton",
    "sphinxcontrib-mermaid>=1.0",
    "ipykernel",  # myst-nb executes pages through a Jupyter kernel
    "dask",  # docs-only, same posture as the test feature (pyproject.toml:198-200):
             # the rechunking pages call .chunk(); xrexpr itself stays dask-free
]

[tool.pixi.environments]
docs = { features = ["docs"], solve-group = "default" }

[tool.pixi.feature.docs.tasks]
docs = { cmd = "sphinx-build -W --keep-going -b html docs docs/_build/html", description = "Build the docs (warnings are errors)" }
```

`solve-group = "default"` so the docs build against the same xarray the tests do. The
local loop is `pixi run -e docs docs`. **Deliberately absent: numbagg.** xarray
dispatches reductions to numbagg when it is importable and its presence changes results,
not just coverage (`ci.yml:22-25`) — the docs env is the no-numbagg arm so executed
output has one canonical form.

**Read the Docs builds via pip, not pixi.** RTD does not run pixi natively. The pip
route (`pip install .[docs]`) keeps RTD's managed build, PR previews and
`fail_on_warning`; because the pixi feature *is* the extra, only resolution differs
(pip vs conda-forge), not the dependency list. The fallback, written down now so the
first breakage doesn't relitigate it: if an RTD build ever fails where CI's lock-frozen
build passes and the cause traces to pip/conda version skew, switch `.readthedocs.yaml`
to `build.commands` (curl-install pixi, `pixi run -e docs docs`, copy
`docs/_build/html` to `$READTHEDOCS_OUTPUT/html`). Until then, pip.

**Sphinx must not see the memos.** *(Amended: the memos and this roadmap have since moved
to `planning/`, so `docs/` is the site and nothing else. The allowlist below is kept as
belt-and-braces — it is what keeps a stray note dropped into `docs/` from failing the
build — but it is no longer load-bearing.)* Under `-W`, every `.md` file Sphinx discovers
but no toctree references is a fatal warning. Allowlist the site directories rather than
blocklisting anything else — the allowlist cannot rot when the next file is added:

```python
include_patterns = [
    "index.md",
    "getting-started/**",
    "guide/**",
    "internals/**",
    "api/**",
]
```

## The site, page by page

Four levels, matching the brief: what it is → what it does to your chain → how it works
→ the reference. `(nb)` marks an executed page — a MyST notebook whose code cells run at
build time, with `nb_execution_mode = "force"` and `nb_execution_raise_on_error = True`
so a cell that errors is a build failure, not a stale page. An executed page is an
ordinary `.md` file with this frontmatter:

```markdown
---
file_format: mystnb
kernelspec:
  name: python3
---
```

```text
docs/
├── conf.py
├── index.md                      landing: the pitch (README.md:17-67) + sphinx-design
│                                 cards into the four sections
├── getting-started/
│   ├── installation.md           README.md:115-121; dask optional; Python 3.10+
│   └── quickstart.md       (nb)  the README story end-to-end on a synthetic dataset
│                                 (build one the way tests/conftest.py does — do not
│                                 ship a data file). Record → explain → collect,
│                                 executed. The 200x benchmark numbers stay *prose*,
│                                 quoted from the README: they are hardware-dependent
│                                 and the source file is untracked.
├── guide/
│   ├── concepts.md               what a plan is; record → lower → optimise → replay in
│   │                             user terms; diagram D1 (short form). Sources:
│   │                             README.md:123-133, accessor.py:1-24.
│   ├── reading-explain.md  (nb)  every explain() annotation decoded: [consumes={...}],
│   │                             [time -> month], Opaque "[not modelled]". Sources:
│   │                             README.md:69-113, explain.py:1-21.
│   ├── rewrites.md         (nb)  what it rewrites today (the README table), what it
│   │                             deliberately won't touch, the invariant **verbatim**
│   │                             (README.md:184-188), InvalidExpressionError, and the
│   │                             stop-failing case (README.md:164-183) executed.
│   ├── grouped-windowed-weighted.md (nb)  builder pairs are one operation; why
│   │                             (lowering, in user terms); what moves past them.
│   │                             Sources: README.md:240-267, lower.py:1-38 (top half).
│   └── rechunking.md       (nb)  what a chunk spec is, the None ≠ -1 table
│                                 (chunks.py:27-39), which specs a select crosses.
│                                 Sources: README.md:216-238. Needs dask; diagram D5.
├── internals/
│   ├── pipeline.md               the whole pipeline, stage by stage, with each stage's
│   │                             contract; D1 (full form). Sources: lower.py:1-38,
│   │                             accessor.py:1-24, schema.py:1-20 (the fold ownership).
│   ├── ir.md                     list-not-tree; kinds not methods; fat variants;
│   │                             symbolic dim sets and ALL_DIMS; immutability vs
│   │                             hashability; unary-only; FluentOp vs LoweredOp; D2.
│   │                             Source: ir.py:1-54.
│   ├── operations.md             OP_TABLE as a sum type over dispatch kinds; reduce
│   │                             destroys, scan keeps; nominate-vs-confirm. Source:
│   │                             operations.py:1-32.
│   ├── lowering.md               why a stage and not a rule; the fusion contract
│   │                             (semantics-preserving + idempotent); emit; the opaque
│   │                             fallback for unmodelled openers. Source: lower.py:1-38.
│   ├── optimiser.md              the rule catalogue; the fixpoint and its termination
│   │                             measure; dim_effect as the single dispatch site; the
│   │                             schema fold and the _trusted_prefix boundary; D3.
│   │                             Source: optimize.py:1-51.
│   ├── schema.md                 logical schema, no materialisation; variables as the
│   │                             store, dims derived; sizes as int | None; why the
│   │                             optimiser owns the fold. Source: schema.py:1-20.
│   ├── values.md                 the two value taxonomies and the shared design moves:
│   │                             classify as sole constructor, policy stays with the
│   │                             optimiser, an escape hatch per layer (Label /
│   │                             OpaqueChunk / Opaque); extent-dependence; D4, D5.
│   │                             Sources: indexers.py:1-29, chunks.py:1-52.
│   └── design-history.md         an annotated reading list: the origin memo, the
│                                 structural-dispatch series, the IR discussions, this
│                                 roadmap. **GitHub URLs, not relative links** — these
│                                 files are outside include_patterns, and a relative
│                                 link to an unbuilt file is a -W failure.
└── api/
    └── index.md                  autosummary tables in {eval-rst} blocks, grouped:
                                  top-level (LazyProxy, Explanation,
                                  InvalidExpressionError) / IR nodes / operation specs /
                                  schema / lowering / optimiser / indexers / chunks /
                                  explain. Stubs generate into api/generated/
                                  (gitignored, autosummary_generate = True).
```

The essay migration is **copy and adapt**, not transclude: the docs page is the new
canonical home and may reorder, join and extend what the docstring said, and prose that
only makes sense inside the source file ("see the match below") is rewritten. Two
mechanical rules for every migrated page:

1. Every rST role changes spelling: ``:func:`x``` → ``{func}`x```, and likewise
   `:class:`/`:meth:`/`:attr:`/`:mod:`/`:data:`. A missed one renders as literal text
   **with no warning** — this is the one failure `-W` cannot catch, so it gets its own
   grep in the Tests section.
2. The GitHub alert syntax in README-sourced material (`> [!WARNING]`, `README.md:3-9`)
   becomes MyST admonitions (`:::{warning}` under `myst_enable_extensions =
   ["colon_fence"]`).

## The diagrams, named

Five mermaid diagrams, each stated here so the implementer draws the mechanism and not a
decoration. Deliberately **no** diagram for select pushdown itself: an executed
before/after `explain()` pair shows it better and cannot drift.

- **D1 — the pipeline** (`guide/concepts.md` short; `internals/pipeline.md` full).
  `flowchart LR`: xarray calls → `to_opnode` → fluent IR → `to_lower_ir` → lowered IR →
  `optimize` → lowered IR → `emit` → `[Call]` → `_replay` → result. The full form labels
  each stage with its owning module and marks that every stage is pure metadata except
  `_replay` — the only box that touches data.
- **D2 — the two-level IR** (`internals/ir.md`). Two columns for `FluentOp` and
  `LoweredOp`: the shared variants (`Reduce`, `Select`, `Scan`, `Project`, `Rechunk`,
  `Opaque`), the fluent-only `ContextOpen`, the lowered-only fused kinds
  (`GroupedReduce`, `WindowedReduce`, `WeightedReduce`), and the `to_lower_ir` arrows —
  opener + closer fuse into one node; an unrecognised pair demotes to `Opaque`.
- **D3 — the fixpoint loop** (`internals/optimiser.md`). The driver: plan in → each rule
  in `_RULES` once → changed? loop, else done — annotated with the termination measure,
  and with `dim_effect` drawn as the single box both pushdown rules consult.
- **D4 — classifying an indexer** (`internals/values.md`). The `classify` decision tree
  to the six variants, each leaf annotated with `drops_dim`.
- **D5 — classifying a chunk spec** (`internals/values.md`, reused in
  `guide/rechunking.md`). The `classify_chunk` tree to the seven variants, grouped by
  the discriminant the optimiser matches on: extent-free (`SingleSize`, `FullDim`,
  `NoChange` — a select crosses) vs extent-dependent (`Auto`, `ByteSize`, `BlockSeq` —
  barrier) vs `OpaqueChunk`.

## The PRs

Every PR lands with the docs building clean — `pixi run -e docs docs` green under `-W`
from PR 1 onwards — and with the standard suite untouched.

### PR 1 — the skeleton and an executed quickstart

`pyproject.toml` gains the `docs` extra, environment and task (the block above) **and a
re-run `pixi lock`** — CI installs with `frozen: true`, so a manifest change without its
lock is a red build. `docs/conf.py` (all extensions, `include_patterns`, theme, napoleon
knobs, intersphinx mapping, `nb_execution_mode`/`nb_execution_raise_on_error`,
`myst_enable_extensions = ["colon_fence"]`). `docs/index.md`,
`docs/getting-started/installation.md`, `docs/getting-started/quickstart.md`.
`.gitignore` gains `docs/_build/` and `docs/api/generated/`.

**Proof:** `pixi run -e docs docs` exits 0; the built quickstart HTML contains
`explain()` output that was *produced by execution* (assert by eye in the diff: the
page source contains code cells, not pasted output).

### PR 2 — CI and Read the Docs

A `docs` job in `.github/workflows/ci.yml` mirroring the existing setup-pixi pattern
(`ci.yml:37-44`):

```yaml
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4.2.2
      - uses: prefix-dev/setup-pixi@v0.10.0
        with:
          environments: docs
          frozen: true
      - run: pixi run -e docs docs
      - uses: actions/upload-artifact@v4
        with:
          name: docs-html
          path: docs/_build/html
```

And `.readthedocs.yaml`:

```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.12"
sphinx:
  configuration: docs/conf.py
  fail_on_warning: true
python:
  install:
    - method: pip
      path: .
      extra_requirements: [docs]
```

**A named human step, outside any PR:** importing the project on readthedocs.org and
switching on pull-request previews are dashboard actions only the maintainer can take.
The PR description must say so and link the RTD import page.

**Proof:** the CI docs job is green and the artifact opens locally; once imported, the
RTD build is green.

### PR 3 — the user guide, first half

`guide/concepts.md`, `guide/reading-explain.md`, `guide/rewrites.md`; D1 short form.
Sourced from `README.md:69-188` — including the invariant quoted verbatim
(`README.md:184-188`) and the stop-failing example executed (`README.md:164-183`).
README is **not** edited in this PR.

**Proof:** build green; the role-grep (Tests §2) clean over `docs/guide/`.

### PR 4 — the user guide, second half

`guide/grouped-windowed-weighted.md`, `guide/rechunking.md`; D5. The first pages whose
cells need dask at execution time. In executed cells prefer `explain()` text and
`assert_equal` over dask reprs — chunked reprs are dask-version-sensitive and say
nothing the plan text doesn't.

**Proof:** build green with the cells executing (no `nb_execution_excludepatterns`
escape hatches).

### PR 5 — internals, first half

`internals/pipeline.md`, `internals/ir.md`, `internals/operations.md`,
`internals/lowering.md`; D1 full, D2. Copied and adapted from the module essays
(`lower.py:1-38`, `ir.py:1-54`, `operations.py:1-32`, `accessor.py:1-24`,
`schema.py:1-20` where the fold is explained). **The docstrings are not edited** — the
prose exists in both places until PR 8, and that transient duplication is this
workstream's stated policy, not an oversight.

**Proof:** build green; role-grep clean over `docs/internals/`.

### PR 6 — internals, second half

`internals/optimiser.md`, `internals/schema.md`, `internals/values.md`,
`internals/design-history.md`; D3, D4. Sources: `optimize.py:1-51`, `schema.py:1-20`,
`indexers.py:1-29`, `chunks.py:1-52`. `design-history.md` links the memos by GitHub URL
only.

**Proof:** as PR 5.

### PR 7 — the API reference

`api/index.md` with grouped `autosummary` tables (in `{eval-rst}` blocks);
`autosummary_generate = True`. The groups follow the module `__all__`s; `optimize` and
`explain` have no `__all__`, so list `optimize.optimize` and `explain.format_plan`
explicitly. Autosummary will surface every dangling cross-reference the docstrings
carry (roles pointing at private helpers, renamed symbols); fixing those
docstring-side is **in scope for this PR**, not deferred to PR 8.

**Proof:** every name in `xrexpr.__init__.__all__` (`__init__.py:13`) plus the
module-level publics renders a stub page; `viewcode` links resolve; build still `-W`
clean.

### PR 8 — slim the module docstrings and the README (last)

Only after PRs 5–6 are live on RTD. For each module in `src/xrexpr/`: diff the essay
against its docs page, confirm nothing is said in the docstring that the page lost, then
cut the module docstring to (a) the one-line summary, (b) one orientation paragraph —
what the module holds and the one invariant a reader editing it must know, (c) a pointer
to the docs page. Callable docstrings are untouched.

`README.md` in the same PR: the four `<details>` blocks (`README.md:190-281`) become
one short "Under the hood" paragraph of links into the site; the RTD badge joins the
top; the pitch, the benchmark numbers, the rewrite table and the two admonition banners
stay.

**Proof:** `pixi run pre-commit` green — ruff `D100` still requires the module
docstring to exist, and of the numpydoc checks only `GL02`/`GL03` touch module
docstrings (`pyproject.toml:241-256`; the `PR*`/`RT*`/`YD01` checks are signature
checks), so the hooks *permit* the slimming but do not verify the summaries stay true —
that is the review's job, stated here so nobody mistakes a green hook for the check.
Full suite, mypy, ruff unchanged; docs still `-W` clean (autodoc renders the slimmed
module docstrings on the API pages).

## Roadmap bookkeeping

`00-assessment.md` gains a W10 row in the workstreams table and a row in the handover
table's recommended order (this file's own rule: file the GitHub issue and link it in
the same pass, `00-assessment.md:282-284`). The standing comment rule
(`00-assessment.md:242-243`) interacts with this workstream twice: while W10 is open,
code may cite this file; after PR 8, the slimmed module docstrings point at the **site**
— not at this roadmap file, whose work will then be closed.

## Where this is most likely wrong (the honesty beat)

- **Executed output is hostage to the environment.** The docs env pins the no-numbagg
  arm and prefers `explain()` text over computed floats, but any numeric repr in an
  executed cell can drift under an xarray/dask bump. `nb_execution_raise_on_error`
  catches crashes; silent value drift is only caught by a human reading the RTD
  preview. Keep computed output out of cells wherever the plan text carries the point.
- **RTD's env is not CI's env.** pip resolution vs `pixi.lock` means a page can build
  in CI and fail on RTD. That is visible (both are required green) but annoying; the
  switch-to-pixi trigger is written in the toolchain section so the decision is
  pre-made.
- **The slimmed summaries have no guard.** After PR 8 nothing mechanical keeps a
  module's orientation paragraph true as the module evolves — the same maintenance
  burden the essays had, minus the bulk. Accepted, because the enforced callable
  docstrings carry the per-object truth.
- **The rST-role grep is the only net for the silent failure.** It lives in Tests and
  must survive any future restructuring of the site directories.
- **README/docs duplication is accepted and bounded.** After PR 8 the drift surface is
  two spots: the benchmark numbers (unexecutable — hardware-dependent, source file
  untracked) and the rewrite table. Named here so they are checked when the optimiser
  gains a rule.
- **What this workstream does not buy:** doctest-checking of examples inside
  docstrings; versioned docs (RTD serves `latest` until a release process exists — the
  `CHANGELOG.md` that `pyproject.toml:52` links is absent); and retiring the README's
  WARNING banner, which is the maintainer's confidence call, not a docs PR.
- **PR 8 vs open code work.** W5/W6 will edit the very docstrings PR 8 slims. Land PR 8
  after whichever of those is in flight has merged, or rebase deliberately — the diff
  is prose-only either way, but a mid-air collision on a 50-line essay is a miserable
  merge.

## Tests

1. **The build is the test.** Every PR: `pixi run -e docs docs` exits 0 with `-W
   --keep-going` — no warnings, all cells executed, no orphaned pages.
2. **No leftover rST roles in MyST pages.** From PR 3 onward:
   `grep -rn ':func:\|:class:\|:meth:\|:data:\|:mod:\|:attr:' docs/index.md docs/getting-started docs/guide docs/internals docs/api/index.md`
   returns nothing. (Generated stubs and `conf.py` are exempt; docstrings rendered via
   autodoc keep their rST roles on purpose.)
3. **Execution is real.** The quickstart and guide pages carry `file_format: mystnb`
   frontmatter and code cells; no page pastes `explain()` output as a plain fenced
   block where a cell could produce it.
4. **PR 8 only:** the existing suite passes unmodified; `pixi run pre-commit` green;
   for each module, the deleted essay paragraphs each have a home on a docs page
   (reviewed against the PR 5/6 diffs, not asserted).

## Acceptance criteria

- The site builds warning-free in CI (lock-frozen pixi) and on Read the Docs (pip), and
  RTD serves it with PR previews enabled.
- All four levels exist and are reachable from `index.md`: getting-started, guide,
  internals, API reference; the five named diagrams are present.
- After PR 8: no module docstring in `src/xrexpr/` exceeds ~20 lines; the README has no
  `<details>` blocks; `pixi run test`, `pixi run mypy`, `pixi run ruff` and
  `pixi run pre-commit` are all green.

## Verification commands

```
pixi run -e docs docs
pixi run python -m pytest tests -q
pixi run mypy
pixi run python -m ruff check src tests
pixi run pre-commit
```
