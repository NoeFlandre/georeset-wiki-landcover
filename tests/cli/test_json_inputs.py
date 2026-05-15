import json
from pathlib import Path

import pytest

from georeset.cli.data.json_inputs import (
    index_json_records_by_pageid,
    read_optional_json_mapping,
    read_required_json_mapping,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_read_optional_json_mapping_returns_empty_for_missing_or_non_mapping(tmp_path):
    missing_path = tmp_path / "missing.json"
    non_mapping_path = tmp_path / "list.json"
    _write_json(non_mapping_path, [{"pageid": "100"}])

    assert read_optional_json_mapping(missing_path) == {}
    assert read_optional_json_mapping(non_mapping_path) == {}


def test_read_required_json_mapping_fails_loudly_for_non_mapping(tmp_path):
    path = tmp_path / "list.json"
    _write_json(path, [{"pageid": "100"}])

    with pytest.raises(ValueError, match="article contents"):
        read_required_json_mapping(path, description="article contents")


def test_index_json_records_by_pageid_uses_payload_pageid_when_present():
    indexed = index_json_records_by_pageid(
        {
            "stale-key": {"pageid": 100, "content": "from payload id"},
            "200": {"pageid": "", "content": "from key"},
            "bad": "not a mapping",
        }
    )

    assert set(indexed) == {"100", "200"}
    assert indexed["100"]["content"] == "from payload id"
    assert indexed["200"]["content"] == "from key"
