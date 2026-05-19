import pytest

from georeset.cli.csv_args import parse_csv_ints, parse_csv_strings


def test_parse_csv_strings_strips_whitespace_and_drops_empty_parts() -> None:
    assert parse_csv_strings(" clip_base, ,dinov2_base ,, clip_large ") == [
        "clip_base",
        "dinov2_base",
        "clip_large",
    ]


def test_parse_csv_ints_strips_whitespace_and_drops_empty_parts() -> None:
    assert parse_csv_ints("320, 640,,2240 ") == [320, 640, 2240]


def test_parse_csv_ints_raises_for_non_integer_parts() -> None:
    with pytest.raises(ValueError):
        parse_csv_ints("320,large")
