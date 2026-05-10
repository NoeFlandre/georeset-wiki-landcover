"""Compatibility shim for ``src.utils``."""

import georeset.utils as _impl
from georeset.utils import *  # noqa: F403

__path__ = _impl.__path__
