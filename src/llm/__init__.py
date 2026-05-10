"""Compatibility shim for ``src.llm``."""

import georeset.llm as _impl
from georeset.llm import *  # noqa: F403

__path__ = _impl.__path__
