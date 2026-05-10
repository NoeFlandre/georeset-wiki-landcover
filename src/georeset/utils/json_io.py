"""Atomic file I/O helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


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
