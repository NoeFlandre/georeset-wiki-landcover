import json
import os

import pytest

from src.utils.json_io import write_json_atomic


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
