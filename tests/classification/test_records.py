from georeset.classification.records import build_prediction_record, should_skip_record


def test_should_skip_record_requires_ok_and_matching_fingerprint_by_default():
    record = {"parse_status": "ok", "metadata": {"fingerprint": "fp-1"}}

    assert should_skip_record(record, "fp-1", retry_failed=False) is True
    assert should_skip_record(record, "fp-2", retry_failed=False) is False
    assert should_skip_record({"parse_status": "error"}, "fp-1", retry_failed=False) is False
    assert should_skip_record(None, "fp-1", retry_failed=False) is False


def test_should_skip_record_keeps_any_ok_record_when_retrying_failures():
    record = {"parse_status": "ok", "metadata": {"fingerprint": "old-fp"}}

    assert should_skip_record(record, "new-fp", retry_failed=True) is True


def test_build_prediction_record_preserves_result_fields_and_adds_fingerprint():
    result = {
        "prediction": "31",
        "prediction_labels": ["31"],
        "parse_status": "ok",
        "raw_response": '{"label":"31"}',
        "error": None,
        "metadata": {"model": "m.gguf", "prompt": "..."},
    }

    record = build_prediction_record(
        pageid="100",
        title="Forêt",
        target="31",
        result=result,
        fingerprint="fp-1",
    )

    assert record == {
        "pageid": "100",
        "title": "Forêt",
        "target": "31",
        "prediction": "31",
        "prediction_labels": ["31"],
        "parse_status": "ok",
        "raw_response": '{"label":"31"}',
        "error": None,
        "metadata": {"model": "m.gguf", "prompt": "...", "fingerprint": "fp-1"},
    }


def test_build_prediction_record_accepts_extra_metadata():
    record = build_prediction_record(
        pageid="100",
        title="Forêt",
        target="31",
        result={"metadata": {}},
        fingerprint="fp-1",
        extra_metadata={"text_control": "shuffled", "shuffled_from_pageid": "200"},
    )

    assert record["prediction"] is None
    assert record["prediction_labels"] == []
    assert record["metadata"] == {
        "fingerprint": "fp-1",
        "text_control": "shuffled",
        "shuffled_from_pageid": "200",
    }
