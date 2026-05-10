"""Compatibility shim for ``src.classification``."""

import georeset.classification as _impl
from georeset.classification import *  # noqa: F403

__path__ = _impl.__path__
