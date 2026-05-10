"""Compatibility shim for ``src.fetchers``."""

import georeset.fetchers as _impl
from georeset.fetchers import *  # noqa: F403

__path__ = _impl.__path__
