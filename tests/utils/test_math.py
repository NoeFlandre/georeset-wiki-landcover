from georeset_wiki_landcover.utils.math import safe_div


def test_safe_div_returns_quotient_for_nonzero_denominator() -> None:
    assert safe_div(3, 2) == 1.5


def test_safe_div_returns_zero_for_zero_denominator() -> None:
    assert safe_div(3, 0) == 0.0
