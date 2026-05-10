"""Compatibility shim for ``src.visualization``."""

import georeset.visualization as _impl
from georeset.visualization import *  # noqa: F403

__path__ = _impl.__path__
