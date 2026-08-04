"""Sphinx configuration for the xrexpr documentation site.

The site is MyST markdown throughout, and the pages carrying worked examples are
*executed* at build time (``myst-nb``), so no ``explain()`` output in the docs can drift
from what the code actually prints. Warnings are errors: the local loop and CI both run
``sphinx-build -W --keep-going`` via ``pixi run -e docs docs``.

Notes
-----
Under ``-W``, every markdown file Sphinx discovers but no toctree references is a fatal
warning. ``docs/`` is the site and nothing else — the design memos and the roadmap live
in ``planning/`` — so the ``include_patterns`` allowlist is belt-and-braces rather than
load-bearing: it keeps a stray note dropped in here from failing a build, and it cannot
rot the way a blocklist of memo filenames would.
"""

from importlib.metadata import version as _installed_version

# Project information
# -------------------

project = "xrexpr"
author = "Charles Turner"
copyright = "2026, Charles Turner"
release = _installed_version("xrexpr")
version = release.split("+")[0]


# Sources Sphinx may read
# -----------------------
# An allowlist of the site itself; anything else that lands under ``docs/`` is invisible
# to the builder rather than fatal. See the module docstring.

include_patterns = [
    "index.md",
    "getting-started/**",
    "guide/**",
    "internals/**",
    "api/**",
]


# Extensions
# ----------
# ``myst_nb`` bundles ``myst_parser`` — listing both is an error, not a redundancy.
# Deliberately absent: the ``numpydoc`` Sphinx extension. Docstring validation is the
# pre-commit hook's job ([tool.numpydoc_validation] in pyproject.toml), and two owners of
# one convention fight; ``napoleon`` here only *renders* the numpydoc sections.

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]


# MyST
# ----
# ``colon_fence`` so README-sourced GitHub alerts (``> [!WARNING]``) can become
# ``:::{warning}`` admonitions, which nest inside directives without fence-counting.

myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3


# Execution (myst-nb)
# -------------------
# ``force`` so every build re-runs the cells rather than trusting a cache, and
# ``raise_on_error`` so a cell that throws is a build failure rather than a stale page.

nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 120


# API rendering
# -------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True


# Cross-references
# ----------------
# The essays constantly point at xarray and dask concepts, and at this package's own
# source (``viewcode``).

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}


# HTML output
# -----------

html_theme = "pydata_sphinx_theme"
html_title = "xrexpr"
html_theme_options = {
    "github_url": "https://github.com/charles-turner-1/xrexpr",
    "show_toc_level": 2,
    "navigation_with_keys": False,
}
