import math

from georeset.utils.boolish import parse_boolish


def test_parse_boolish_accepts_real_booleans_and_numeric_zero_one() -> None:
    assert parse_boolish(True) is True
    assert parse_boolish(False) is False
    assert parse_boolish(1) is True
    assert parse_boolish(0) is False
    assert parse_boolish(1.0) is True
    assert parse_boolish(0.0) is False


def test_parse_boolish_accepts_known_string_values_case_insensitively() -> None:
    for value in ["true", "T", "YES", "y", "oui", "1"]:
        assert parse_boolish(value) is True
    for value in ["false", "F", "NO", "n", "non", "0"]:
        assert parse_boolish(value) is False


def test_parse_boolish_returns_none_for_missing_or_unknown_values() -> None:
    for value in ["", "nan", "None", "NULL", None, math.nan, "maybe", "2"]:
        assert parse_boolish(value) is None
