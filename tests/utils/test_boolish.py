import math

import numpy as np
import pandas as pd

from georeset_wiki_landcover.utils.boolish import parse_boolish, parse_boolish_series


def test_parse_boolish_accepts_real_booleans_and_numeric_zero_one() -> None:
    assert parse_boolish(True) is True
    assert parse_boolish(False) is False
    assert parse_boolish(np.bool_(True)) is True
    assert parse_boolish(np.bool_(False)) is False
    assert parse_boolish(1) is True
    assert parse_boolish(0) is False
    assert parse_boolish(1.0) is True
    assert parse_boolish(0.0) is False
    assert parse_boolish(np.int64(1)) is True
    assert parse_boolish(np.int64(0)) is False
    assert parse_boolish(np.float64(1.0)) is True
    assert parse_boolish(np.float64(0.0)) is False


def test_parse_boolish_accepts_known_string_values_case_insensitively() -> None:
    for value in ["true", "T", "YES", "y", "oui", "on", "1"]:
        assert parse_boolish(value) is True
    for value in ["false", "F", "NO", "n", "non", "off", "0"]:
        assert parse_boolish(value) is False


def test_parse_boolish_returns_none_for_missing_or_unknown_values() -> None:
    for value in ["", "nan", "None", "NULL", None, math.nan, "maybe", "2"]:
        assert parse_boolish(value) is None


def test_parse_boolish_series_preserves_index_and_returns_bool_mask() -> None:
    values = pd.Series(
        ["true", "false", "oui", "non", "on", "off", None, "maybe", np.bool_(True)],
        index=pd.Index(["a", "b", "c", "d", "e", "f", "g", "h", "i"], name="pageid"),
    )

    parsed = parse_boolish_series(values)

    assert parsed.tolist() == [True, False, True, False, True, False, False, False, True]
    assert parsed.dtype == bool
    assert parsed.index.equals(values.index)
