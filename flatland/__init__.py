# -*- coding: utf-8 -*-

"""Top-level package for flatland."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("flatland-rl")
except PackageNotFoundError:
    pass
