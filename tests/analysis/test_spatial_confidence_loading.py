from pathlib import Path

import pandas as pd
import pytest

from georeset.analysis.spatial_confidence_loading import load_spatial_confidence


def test_load_csv_normalizes_pageid_to_string(tmp_path: Path) -> None:
    path = tmp_path / "spatial.csv"
    pd.DataFrame(
        {
            "pageid": [1, 2, "3"],
            "point_label_share_250m": [0.95, 0.1, 0.7],
            "point_label": ["31", "21", "31"],
        }
    ).to_csv(path, index=False)

    loaded = load_spatial_confidence(path)

    assert loaded["pageid"].tolist() == ["1", "2", "3"]


def test_load_csv_preserves_leading_zeros_in_pageid(tmp_path: Path) -> None:
    path = tmp_path / "spatial_leading_zeros.csv"
    path.write_text(
        "pageid,point_label_share_250m\n001,0.9\n002,0.8\n",
        encoding="utf-8",
    )

    loaded = load_spatial_confidence(path)

    assert loaded["pageid"].tolist() == ["001", "002"]


def test_load_csv_casts_point_label_to_string_when_present(tmp_path: Path) -> None:
    path = tmp_path / "spatial_point_label.csv"
    pd.DataFrame(
        {
            "pageid": ["1", "2"],
            "point_label": [31, 21],
            "point_label_share_250m": [0.9, 0.8],
        }
    ).to_csv(path, index=False)

    loaded = load_spatial_confidence(path)

    assert loaded["point_label"].tolist() == ["31", "21"]


def test_load_parquet_if_pyarrow_available(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    path = tmp_path / "spatial.parquet"
    pd.DataFrame(
        {
            "pageid": [1, 2],
            "point_label_share_250m": [0.82, 0.91],
            "dominant_matches_point_label_250m": ["TRUE", "FALSE"],
        }
    ).to_parquet(path, index=False)

    loaded = load_spatial_confidence(path)

    assert loaded["pageid"].tolist() == ["1", "2"]


def test_load_csv_missing_pageid_returns_empty_when_allowed(tmp_path: Path) -> None:
    path = tmp_path / "missing_pageid.csv"
    pd.DataFrame({"other": [1, 2], "point_label_share_250m": [0.1, 0.2]}).to_csv(
        path, index=False
    )

    loaded = load_spatial_confidence(path, allow_missing_pageid=True)

    assert loaded.empty


def test_load_csv_missing_pageid_raises_by_default(tmp_path: Path) -> None:
    path = tmp_path / "missing_pageid.csv"
    pd.DataFrame({"other": [1, 2], "point_label_share_250m": [0.1, 0.2]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="missing required column.*pageid"):
        load_spatial_confidence(path)


def test_coerces_dominant_match_columns_from_string_values_to_nullable_bool(tmp_path: Path) -> None:
    path = tmp_path / "dominant.csv"
    pd.DataFrame(
        {
            "pageid": ["1", "2", "3", "4", "5", "6", "7", "8"],
            "dominant_matches_point_label_250m": [
                "true",
                "FALSE",
                "True",
                "false",
                None,
                "",
                "1",
                "0",
            ],
            "dominant_matches_point_label_500m": [
                "FALSE",
                "true",
                "0",
                "1",
                "null",
                None,
                "maybe",
                "",
            ],
        }
    ).to_csv(path, index=False)

    loaded = load_spatial_confidence(path)

    assert str(loaded["dominant_matches_point_label_250m"].dtype) == "boolean"
    assert str(loaded["dominant_matches_point_label_500m"].dtype) == "boolean"

    mask_250 = loaded["dominant_matches_point_label_250m"].astype("boolean").fillna(False)
    mask_500 = loaded["dominant_matches_point_label_500m"].astype("boolean").fillna(False)

    assert mask_250.tolist() == [True, False, True, False, False, False, True, False]
    assert mask_500.tolist() == [False, True, False, True, False, False, False, False]


def test_numeric_point_label_columns_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "numeric.csv"
    frame = pd.DataFrame(
        {
            "pageid": [1, 2],
            "point_label_share_250m": [0.11, 0.44],
            "point_label_share_500m": [0.22, 0.55],
            "dominant_matches_point_label_250m": ["True", "False"],
            "point_label": ["31", "21"],
        }
    )
    frame.to_csv(path, index=False)

    loaded = load_spatial_confidence(path)

    assert loaded["point_label_share_250m"].tolist() == frame["point_label_share_250m"].tolist()
    assert (
        loaded["point_label_share_500m"].tolist()
        == frame["point_label_share_500m"].tolist()
    )
