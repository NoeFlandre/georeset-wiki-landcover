import pandas as pd

from georeset.text.record_access import is_missing, json_scalar, mapping_get


def test_mapping_get_reads_dict_and_series_with_default() -> None:
    assert mapping_get({"value": 3}, "value") == 3
    assert mapping_get(pd.Series({"value": 4}), "value") == 4
    assert mapping_get(None, "value", "missing") == "missing"
    assert mapping_get({}, "value", "missing") == "missing"


def test_is_missing_treats_none_blank_and_pandas_na_as_missing() -> None:
    assert is_missing(None)
    assert is_missing("")
    assert is_missing("  ")
    assert is_missing(pd.NA)
    assert not is_missing([""])
    assert not is_missing({"value": pd.NA})


def test_json_scalar_returns_json_safe_scalars_and_default_for_missing() -> None:
    assert json_scalar(pd.NA) is None
    assert json_scalar(pd.NA, default="missing") == "missing"
    assert json_scalar(" value ") == "value"
    assert json_scalar(pd.Series([3], dtype="int64").iloc[0]) == 3
    assert json_scalar(pd.Series([1.5], dtype="float64").iloc[0]) == 1.5
