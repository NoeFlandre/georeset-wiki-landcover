"""Compatibility shim for ``src.analysis``."""

import georeset.analysis as _impl
from georeset.analysis import *  # noqa: F403

__path__ = _impl.__path__
