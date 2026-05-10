"""Compatibility shim for ``src.spatial``."""

import georeset.spatial as _impl
from georeset.spatial import *  # noqa: F403

__path__ = _impl.__path__
