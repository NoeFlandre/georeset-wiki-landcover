from georeset.classification.prediction_parser import (
    extract_allowed_labels,
    normalize_prediction_response,
)


def test_extract_allowed_labels_json_list():
    raw = '{"labels": ["31", "32"]}'
    assert extract_allowed_labels(raw, ["31", "32", "33"]) == ["31", "32"]


def test_extract_allowed_labels_json_string():
    raw = '{"label": "31, 32"}'
    assert extract_allowed_labels(raw, ["31", "32", "33"]) == ["31", "32"]


def test_extract_allowed_labels_raw_text():
    raw = "Je choisis 31 et 32."
    assert extract_allowed_labels(raw, ["31", "32", "33"]) == ["31", "32"]


def test_extract_allowed_labels_numeric_boundary():
    raw = "131"
    assert extract_allowed_labels(raw, ["31"]) == []


def test_extract_allowed_labels_word_boundary():
    raw = "woodland"
    assert extract_allowed_labels(raw, ["wood"]) == []


def test_normalize_prediction_response_valid_json():
    raw = '{"labels": ["31"]}'
    labels, error = normalize_prediction_response(raw, ["31"])
    assert labels == ["31"]
    assert error is None


def test_normalize_prediction_response_json_with_unknown_label_returns_error():
    raw = '{"labels": ["31", "99"]}'
    labels, error = normalize_prediction_response(raw, ["31"])
    assert labels == ["31"]
    assert error is not None
    assert "99" in error


def test_normalize_prediction_response_rejects_non_string_json_labels():
    raw = '{"labels": ["31", 32]}'

    labels, error = normalize_prediction_response(raw, ["31", "32"])

    assert labels == ["31"]
    assert error == "JSON list fields must contain only strings"


def test_normalize_prediction_response_invalid_json_but_valid_text():
    raw = "I predict 31."
    labels, error = normalize_prediction_response(raw, ["31"])
    assert labels == ["31"]
    assert error is None
