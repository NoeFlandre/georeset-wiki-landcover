import json
import os

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from georeset.utils.json_io import (
    write_csv_atomic,
    write_geojson_atomic,
    write_html_map_atomic,
    write_json_atomic,
    write_markdown_table_atomic,
    write_parquet_atomic,
    write_text_atomic,
)


def test_write_json_atomic_writes_complete_json(tmp_path):
    output_path = tmp_path / "nested" / "records.json"

    write_json_atomic(output_path, {"1": {"value": "ok"}}, indent=2, ensure_ascii=False)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"1": {"value": "ok"}}
    assert not list(output_path.parent.glob("*.tmp"))


def test_write_json_atomic_preserves_existing_file_when_serialization_fails(tmp_path):
    output_path = tmp_path / "records.json"
    output_path.write_text('{"old": true}', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(output_path, {"bad": object()})

    assert output_path.read_text(encoding="utf-8") == '{"old": true}'
    assert not list(tmp_path.glob("*.tmp"))


def test_write_json_atomic_uses_os_replace_after_temp_file_is_written(tmp_path, monkeypatch):
    output_path = tmp_path / "records.json"
    output_path.write_text('{"old": true}', encoding="utf-8")
    calls = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        with open(src, encoding="utf-8") as f:
            temp_payload = json.load(f)
        calls.append((src, dst, temp_payload))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)

    write_json_atomic(output_path, {"new": True})

    assert calls == [(calls[0][0], output_path, {"new": True})]
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"new": True}


def test_write_text_atomic_preserves_existing_file_when_replace_fails(tmp_path, monkeypatch):
    output_path = tmp_path / "summary.md"
    output_path.write_text("old\n", encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_text_atomic(output_path, "new\n")

    assert output_path.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_write_csv_atomic_writes_dataframe_via_atomic_text_helper(tmp_path):
    output_path = tmp_path / "nested" / "table.csv"
    frame = pd.DataFrame([{"label": "31", "score": 0.5}])

    write_csv_atomic(output_path, frame, index=False)

    assert output_path.read_text(encoding="utf-8") == "label,score\n31,0.5\n"
    assert not list(output_path.parent.glob("*.tmp"))


def test_write_markdown_table_atomic_escapes_cells_and_preserves_column_order(tmp_path):
    output_path = tmp_path / "nested" / "overview.md"

    write_markdown_table_atomic(
        output_path,
        title="Overview",
        rows=[{"label": "forest | water", "note": "line one\nline two"}],
        columns=["label", "note"],
    )

    assert output_path.read_text(encoding="utf-8") == (
        "# Overview\n"
        "\n"
        "| label | note |\n"
        "| --- | --- |\n"
        "| forest \\| water | line one<br>line two |\n"
    )
    assert not list(output_path.parent.glob("*.tmp"))


def test_write_geojson_atomic_uses_replace_after_temp_geojson_is_written(tmp_path, monkeypatch):
    output_path = tmp_path / "nested" / "polygons.geojson"
    frame = gpd.GeoDataFrame(
        [{"label": "31", "geometry": Point(7.0, 48.0)}],
        geometry="geometry",
        crs="EPSG:4326",
    )
    calls = []
    real_replace = os.replace

    def tracking_replace(src, dst):
        temp_frame = gpd.read_file(src)
        calls.append((src, dst, temp_frame.loc[0, "label"]))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", tracking_replace)

    write_geojson_atomic(output_path, frame)

    assert calls == [(calls[0][0], output_path, "31")]
    assert gpd.read_file(output_path).loc[0, "label"] == "31"
    assert not list(output_path.parent.glob("*.tmp*"))


def test_write_html_map_atomic_preserves_existing_file_when_save_fails(tmp_path):
    output_path = tmp_path / "map.html"
    output_path.write_text("old map", encoding="utf-8")

    class FailingMap:
        def save(self, path: str) -> None:
            Path(path).write_text("partial map", encoding="utf-8")
            raise RuntimeError("render failed")

    from pathlib import Path

    with pytest.raises(RuntimeError, match="render failed"):
        write_html_map_atomic(output_path, FailingMap())

    assert output_path.read_text(encoding="utf-8") == "old map"
    assert not list(tmp_path.glob("*.tmp*"))


def test_write_parquet_atomic_preserves_existing_file_when_replace_fails(tmp_path, monkeypatch):
    output_path = tmp_path / "table.parquet"
    output_path.write_bytes(b"old parquet")

    class FakeFrame:
        def to_parquet(self, path, **kwargs) -> None:
            Path(path).write_bytes(b"new parquet")

    from pathlib import Path

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_parquet_atomic(output_path, FakeFrame(), index=False)

    assert output_path.read_bytes() == b"old parquet"
    assert not list(tmp_path.glob("*.tmp*"))
