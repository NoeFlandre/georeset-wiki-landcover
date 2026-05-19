import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from georeset.utils import json_io
from georeset.utils.json_io import (
    markdown_table,
    read_json_file,
    resolve_table_columns,
    write_csv_atomic,
    write_dict_rows_csv_atomic,
    write_dict_rows_markdown_atomic,
    write_dict_rows_table_pair_atomic,
    write_geojson_atomic,
    write_html_map_atomic,
    write_json_atomic,
    write_markdown_table_atomic,
    write_npz_atomic,
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


def test_markdown_table_formats_rows_without_title() -> None:
    assert markdown_table(
        rows=[{"label": "forest | water", "note": "line one\nline two"}],
        columns=["label", "note"],
    ) == (
        "| label | note |\n"
        "| --- | --- |\n"
        "| forest \\| water | line one<br>line two |\n"
    )


def test_markdown_table_returns_no_rows_for_empty_input() -> None:
    assert markdown_table(rows=[], columns=["label", "note"]) == "No rows.\n"


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


def test_write_npz_atomic_writes_arrays_and_preserves_existing_file_when_replace_fails(
    tmp_path, monkeypatch
) -> None:
    output_path = tmp_path / "cache.npz"
    np.savez(output_path, pageids=np.array(["old"]), embeddings=np.array([[0.0]], dtype=np.float32))

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_npz_atomic(
            output_path,
            pageids=np.array(["new"]),
            embeddings=np.array([[1.0]], dtype=np.float32),
        )

    existing = np.load(output_path)
    assert existing["pageids"].tolist() == ["old"]
    assert existing["embeddings"].tolist() == [[0.0]]
    assert not list(tmp_path.glob("*.tmp*"))


def test_read_json_file_reads_from_pathlike_with_utf8(tmp_path):
    nested = tmp_path / "nested"
    path = nested / "records.json"
    nested.mkdir()
    path.write_text('{"résumé": ["value", 42]}', encoding="utf-8")

    payload = read_json_file(path)

    assert payload == {"résumé": ["value", 42]}


def test_read_json_file_propagates_json_decode_error(tmp_path):
    bad_path = tmp_path / "broken.json"
    bad_path.write_text('{"broken": true', encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_json_file(bad_path)


def test_resolve_table_columns_uses_union_all_rows_and_preserves_explicit_order_for_supplied_columns() -> None:
    rows = [{"b": 1, "a": 2}, {"c": 3, "b": 4}]

    assert resolve_table_columns(rows, ["b", "a"]) == ["b", "a", "c"]


def test_resolve_table_columns_without_columns_is_deterministic_union() -> None:
    rows = [{"b": 1, "a": 2}, {"c": 3, "b": 4}]

    assert resolve_table_columns(rows) == ["a", "b", "c"]


def test_write_dict_rows_csv_atomic_uses_union_columns_and_is_atomic(
    tmp_path, monkeypatch
) -> None:
    output_path = tmp_path / "rows.csv"
    captured: list[tuple[str, str]] = []

    def fake_write_text_atomic(
        path: str | os.PathLike[str],
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        captured.append((str(path), text))

    monkeypatch.setattr(json_io, "write_text_atomic", fake_write_text_atomic)

    write_dict_rows_csv_atomic(
        output_path,
        [{"a": "1", "b": "2"}, {"c": "3", "a": "4"}],
        columns=["c", "a"],
    )

    assert captured
    assert captured[0][0] == str(output_path)
    assert captured[0][1].splitlines()[0] == "c,a,b"


def test_write_dict_rows_csv_atomic_writes_empty_file_for_no_rows(tmp_path) -> None:
    output_path = tmp_path / "rows.csv"

    write_dict_rows_csv_atomic(output_path, [])

    assert output_path.read_text(encoding="utf-8") == ""


def test_write_dict_rows_markdown_atomic_resolves_columns_and_escapes_cells(tmp_path) -> None:
    output_path = tmp_path / "overview.md"

    write_dict_rows_markdown_atomic(
        output_path,
        title="Rows",
        rows=[
            {"label": "forest | water", "note": "line one\nline two", "extra": "A"},
            {"note": "third", "score": 1},
        ],
        columns=["label", "note"],
    )

    expected = (
        "# Rows\n\n"
        "| label | note | extra | score |\n"
        "| --- | --- | --- | --- |\n"
        "| forest \\| water | line one<br>line two | A |  |\n"
        "|  | third |  | 1 |\n"
    )
    assert output_path.read_text(encoding="utf-8") == expected


def test_write_dict_rows_table_pair_atomic_writes_csv_and_markdown(tmp_path) -> None:
    output_dir = tmp_path / "tables"

    write_dict_rows_table_pair_atomic(
        output_dir=output_dir,
        stem="overview",
        title="Overview",
        rows=[{"b": 2, "a": 1}, {"a": 3, "c": 4}],
        columns=["a"],
    )

    assert (output_dir / "overview.csv").read_text(encoding="utf-8").splitlines()[0] == "a,b,c"
    assert (output_dir / "overview.md").read_text(encoding="utf-8").startswith(
        "# Overview\n\n| a | b | c |"
    )
