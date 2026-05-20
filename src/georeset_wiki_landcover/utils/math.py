"""Small numeric helpers shared across metric code."""


def safe_div(num: float, den: float) -> float:
    """Return ``num / den`` with a stable zero-denominator convention."""
    return num / den if den else 0.0
