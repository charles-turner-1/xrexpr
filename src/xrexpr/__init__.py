from . import _version
from . import accessor as accessor  # registers the ``.plan`` dataset accessor
from .exceptions import InvalidExpressionError

__all__ = ["InvalidExpressionError"]

__version__ = _version.get_versions()["version"]  # type: ignore
