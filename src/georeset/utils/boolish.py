"""Parse weakly typed boolean values without treating arbitrary strings as true."""

from __future__ import annotations

import math

TRUE_STRINGS = {"true", "t", "yes", "y", "oui", "1"}
FALSE_STRINGS = {"false", "f", "no", "n", "non", "0"}
NONE_STRINGS = {"", "nan", "none", "null"}


def parse_boolish(value: object) -> bool | None:
    """Return a boolean for known bool-like values, otherwise ``None``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, int | float):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_STRINGS:
            return True
        if normalized in FALSE_STRINGS:
            return False
        if normalized in NONE_STRINGS:
            return None
    return None
