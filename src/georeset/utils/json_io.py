"""Atomic file I/O helpers."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class HtmlMap(Protocol):
    """Minimal protocol implemented by Folium map objects."""

    def save(self, outfile: str) -> None:
        """Save HTML map content to a path."""


def write_text_atomic(
    path: str | os.PathLike[str],
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text via a same-directory temp file and atomic replacement."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def read_json_file(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> Any:
    """Read and parse a JSON file."""

    with open(path, encoding=encoding) as file:
        return json.load(file)


def write_json_atomic(
    path: str | os.PathLike[str],
    data: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
) -> None:
    """Write JSON via a same-directory temp file and atomic replacement."""
    text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii) + "\n"
    write_text_atomic(path, text)


def write_csv_atomic(
    path: str | os.PathLike[str],
    frame: Any,
    **to_csv_kwargs: Any,
) -> None:
    """Write a pandas-like DataFrame CSV via atomic text replacement."""
    write_text_atomic(path, frame.to_csv(**to_csv_kwargs))


def _markdown_cell(value: Any) -> str:
    return (
        str(value)
        .replace("|", "\\|")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "<br>")
    )


def write_markdown_table_atomic(
    path: str | os.PathLike[str],
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    columns: list[str] | None = None,
) -> None:
    """Write a simple Markdown table through atomic text replacement."""
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
    else:
        lines.extend(
            [
                "| " + " | ".join(_markdown_cell(column) for column in columns) + " |",
                "| " + " | ".join(["---"] * len(columns)) + " |",
            ]
        )
        for row in rows:
            lines.append(
                "| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |"
            )
    write_text_atomic(path, "\n".join(lines) + "\n")


def resolve_table_columns(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> list[str]:
    """Resolve table columns from explicit input and row key union."""
    if columns is None:
        return sorted({key for row in rows for key in row})
    explicit_columns: list[str] = list(dict.fromkeys(columns))
    extra_columns = sorted({key for row in rows for key in row} - set(explicit_columns))
    return explicit_columns + extra_columns


def write_dict_rows_csv_atomic(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] | None = None,
) -> None:
    """Write a list of row mappings to CSV through the atomic text helper."""
    if not rows:
        write_text_atomic(path, "")
        return
    output = io.StringIO()
    fieldnames = resolve_table_columns(rows, columns)
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, output.getvalue())


def write_dict_rows_markdown_atomic(
    path: str | os.PathLike[str],
    *,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    """Write a list of row mappings to Markdown through the atomic text helper."""
    resolved_columns = resolve_table_columns(rows, columns)
    write_markdown_table_atomic(
        path,
        title=title,
        rows=rows,
        columns=resolved_columns if resolved_columns else None,
    )


def write_dict_rows_table_pair_atomic(
    *,
    output_dir: str | os.PathLike[str],
    stem: str,
    title: str,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
) -> None:
    """Write matching CSV and Markdown row tables with the same columns."""
    base_dir = Path(output_dir)
    write_dict_rows_csv_atomic(base_dir / f"{stem}.csv", rows, columns=columns)
    write_dict_rows_markdown_atomic(
        base_dir / f"{stem}.md",
        title=title,
        rows=rows,
        columns=columns,
    )


def _write_path_atomic(
    path: str | os.PathLike[str],
    *,
    suffix: str,
    writer: Any,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        writer(temp_path)
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def write_geojson_atomic(path: str | os.PathLike[str], frame: Any) -> None:
    """Write a GeoDataFrame GeoJSON through a temp file and atomic replacement."""

    def writer(temp_path: Path) -> None:
        frame.to_file(temp_path, driver="GeoJSON")

    _write_path_atomic(path, suffix=".tmp.geojson", writer=writer)


def write_html_map_atomic(path: str | os.PathLike[str], html_map: HtmlMap) -> None:
    """Write a Folium-like HTML map through a temp file and atomic replacement."""

    def writer(temp_path: Path) -> None:
        html_map.save(str(temp_path))

    _write_path_atomic(path, suffix=".tmp.html", writer=writer)


def write_parquet_atomic(
    path: str | os.PathLike[str],
    frame: Any,
    **to_parquet_kwargs: Any,
) -> None:
    """Write a pandas-like DataFrame parquet file via atomic replacement."""

    def writer(temp_path: Path) -> None:
        frame.to_parquet(temp_path, **to_parquet_kwargs)

    _write_path_atomic(path, suffix=".tmp.parquet", writer=writer)
