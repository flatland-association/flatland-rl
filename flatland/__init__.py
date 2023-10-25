# -*- coding: utf-8 -*-

"""Top-level package for flatland."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("package-name")
except PackageNotFoundError:
    pass
