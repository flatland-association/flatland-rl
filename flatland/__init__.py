# -*- coding: utf-8 -*-

"""Top-level package for flatland."""

__version__ = "0.1.0"

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("flatland-rl")
except PackageNotFoundError:
    pass
