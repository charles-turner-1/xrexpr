"""Rewrite chained expressions on xarray datasets to improve performance.

Importing the package registers the ``.plan`` dataset accessor, so
``ds.plan.mean("lat").isel(time=0).collect()`` records the chain, optimises it and
replays the cheaper plan onto the dataset. See :mod:`xrexpr.accessor` for the entry
point and :mod:`xrexpr.optimize` for the rewrite rules.
"""

from . import _version
from . import accessor as accessor  # registers the ``.plan`` dataset accessor
from .exceptions import InvalidExpressionError

__all__ = ["InvalidExpressionError"]

__version__ = _version.get_versions()["version"]  # type: ignore
