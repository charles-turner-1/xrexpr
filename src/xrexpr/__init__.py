"""Rewrite chained expressions on xarray datasets to improve performance.

Importing the package registers the ``.plan`` accessor on ``Dataset`` and ``DataArray``
alike, so ``ds.plan.mean("lat").isel(time=0).collect()`` records the chain, optimises it
and replays the cheaper plan onto the object. See ``xrexpr.accessor`` for the entry
point and ``xrexpr.optimize`` for the rewrite rules.
"""

from . import _version
from . import accessor as accessor  # registers the ``.plan`` accessor on both types
from .exceptions import InvalidExpressionError

__all__ = ["InvalidExpressionError"]

__version__ = _version.get_versions()["version"]  # type: ignore
