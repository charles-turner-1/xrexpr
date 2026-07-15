from . import _version
from . import accessor as accessor  # registers the ``.plan`` dataset accessor

__version__ = _version.get_versions()["version"]  # type: ignore
