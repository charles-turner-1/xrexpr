# type: ignore
"""Shim so versioneer can supply the version; the real metadata is in pyproject.toml."""

from setuptools import setup

import versioneer

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
