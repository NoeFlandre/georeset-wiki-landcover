import pandas as pd

from georeset_wiki_landcover.analysis.pageid_frames import (
    dataframe_by_pageid,
    load_optional_pageid_csv,
)


def test_load_optional_pageid_csv_returns_empty_pageid_frame_for_missing_path(tmp_path):
    frame = load_optional_pageid_csv(tmp_path / "missing.csv")

    assert list(frame.columns) == ["pageid"]
    assert frame.empty


def test_load_optional_pageid_csv_returns_empty_pageid_frame_when_pageid_missing(tmp_path):
    path = tmp_path / "quality.csv"
    path.write_text("quality_score\n7.0\n", encoding="utf-8")

    frame = load_optional_pageid_csv(path)

    assert list(frame.columns) == ["pageid"]
    assert frame.empty


def test_load_optional_pageid_csv_normalizes_pageid_to_string(tmp_path):
    path = tmp_path / "quality.csv"
    path.write_text("pageid,quality_score\n100,7.0\n", encoding="utf-8")

    frame = load_optional_pageid_csv(path)

    assert frame.loc[0, "pageid"] == "100"


def test_dataframe_by_pageid_indexes_rows_by_string_pageid():
    frame = pd.DataFrame(
        [
            {"pageid": 100, "value": "first"},
            {"pageid": "200", "value": "second"},
        ]
    )

    indexed = dataframe_by_pageid(frame)

    assert set(indexed) == {"100", "200"}
    assert indexed["100"]["value"] == "first"
    assert indexed["200"]["value"] == "second"


def test_dataframe_by_pageid_returns_empty_when_pageid_missing():
    frame = pd.DataFrame([{"value": "first"}])

    assert dataframe_by_pageid(frame) == {}
