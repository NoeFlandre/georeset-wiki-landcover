from unittest.mock import MagicMock, patch

from georeset.classification.llm_classifier import MULTI_SCHEMA, SINGLE_SCHEMA, LLMClassifier


def test_classify_single_label_returns_ok_with_valid_response():
    mock_client = MagicMock()
    mock_client.complete_json.return_value = '{"labels": ["31"]}'
    classifier = LLMClassifier(
        model_path="model.gguf", seed=123, temperature=0.0, client=mock_client
    )

    result = classifier.classify_single_label(
        text="Une forêt dense.",
        allowed_labels=["21", "31"],
        label_descriptions={"21": "Arable land", "31": "Forests"},
        task="corine_level2",
        text_source="summary",
    )

    assert result["prediction"] == "31"
    assert result["prediction_labels"] == ["31"]
    assert result["parse_status"] == "ok"
    assert result["error"] is None
    assert result["metadata"]["model"] == "model.gguf"
    assert result["metadata"]["seed"] == 123
    assert result["metadata"]["temperature"] == 0.0
    assert result["metadata"]["task"] == "corine_level2"
    assert result["metadata"]["text_source"] == "summary"
    _, kwargs = mock_client.complete_json.call_args
    assert kwargs["schema"] == SINGLE_SCHEMA
    assert kwargs["temperature"] == 0.0


def test_single_label_schema_requires_exactly_one_label():
    labels_schema = SINGLE_SCHEMA["properties"]["labels"]

    assert labels_schema["minItems"] == 1
    assert labels_schema["maxItems"] == 1


def test_multilabel_schema_requires_at_least_one_label():
    labels_schema = MULTI_SCHEMA["properties"]["labels"]

    assert labels_schema["minItems"] == 1
    assert "maxItems" not in labels_schema


def test_classify_single_label_multiple_labels_returns_ambiguous():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"labels": ["31", "32"]}'}}]
    }
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Une forêt dense.",
        allowed_labels=["21", "31", "32"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )

    assert result["prediction"] is None
    assert result["prediction_labels"] == ["31", "32"]
    assert result["parse_status"] == "ambiguous"
    assert "multiple labels" in result["error"]


def test_classify_single_label_retries_empty_response_to_ok():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"labels": []}'}}]},
        {"choices": [{"message": {"content": '{"labels": ["31"]}'}}]},
    ]
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Une forêt dense.",
        allowed_labels=["21", "31"],
        label_descriptions={"21": "Arable land", "31": "Forests"},
        task="corine_level2",
        text_source="summary",
    )

    assert result["prediction"] == "31"
    assert result["prediction_labels"] == ["31"]
    assert result["parse_status"] == "ok"
    assert result["metadata"]["attempt_count"] == 2
    assert result["metadata"]["resolved_from_retry"] is True
    assert len(result["metadata"]["attempt_history"]) == 2
    assert (
        "Never return an empty list"
        in mock_llm.create_chat_completion.call_args_list[1].kwargs["messages"][1]["content"]
    )


def test_classify_single_label_retries_ambiguous_response_to_ok():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"labels": ["31", "32"]}'}}]},
        {"choices": [{"message": {"content": '{"labels": ["31"]}'}}]},
    ]
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Une forêt dense.",
        allowed_labels=["21", "31", "32"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )

    assert result["prediction"] == "31"
    assert result["prediction_labels"] == ["31"]
    assert result["parse_status"] == "ok"
    assert result["metadata"]["attempt_count"] == 2
    assert result["metadata"]["attempt_history"][0]["parse_status"] == "ambiguous"


def test_classify_single_label_keeps_ambiguous_when_retry_still_ambiguous():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"labels": ["31", "32"]}'}}]},
        {"choices": [{"message": {"content": '{"labels": ["21", "31"]}'}}]},
    ]
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Une forêt dense.",
        allowed_labels=["21", "31", "32"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )

    assert result["prediction"] is None
    assert result["prediction_labels"] == ["21", "31"]
    assert result["parse_status"] == "ambiguous"
    assert result["metadata"]["attempt_count"] == 2
    assert result["metadata"]["resolved_from_retry"] is False


def test_classify_single_label_comma_string_returns_ambiguous():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"label": "31, 32"}'}}]
    }
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Une forêt dense.",
        allowed_labels=["21", "31", "32"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )

    assert result["prediction"] is None
    assert result["prediction_labels"] == ["31", "32"]
    assert result["parse_status"] == "ambiguous"
    assert "multiple labels" in result["error"]


def test_classify_multilabel_deduplicates_and_sorts():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"labels": ["wood", "meadow", "wood"]}'}}]
    }
    classifier._llm = mock_llm

    result = classifier.classify_multilabel(
        text="Prairie avec bois.",
        allowed_labels=["meadow", "wood"],
        task="osm",
        text_source="summary",
    )
    assert result["prediction"] == ["meadow", "wood"]
    assert result["prediction_labels"] == ["meadow", "wood"]
    assert result["parse_status"] == "ok"


def test_classify_multilabel_mixed_returns_ok():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"label": "meadow, wood"}'}}]
    }
    classifier._llm = mock_llm

    result = classifier.classify_multilabel(
        text="Prairie.",
        allowed_labels=["meadow", "wood"],
        task="osm",
        text_source="summary",
    )
    assert result["prediction"] == ["meadow", "wood"]
    assert result["prediction_labels"] == ["meadow", "wood"]
    assert result["parse_status"] == "ok"


def test_classify_multilabel_retries_empty_response_to_ok():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"labels": []}'}}]},
        {"choices": [{"message": {"content": '{"labels": ["wood"]}'}}]},
    ]
    classifier._llm = mock_llm

    result = classifier.classify_multilabel(
        text="Bois.",
        allowed_labels=["meadow", "wood"],
        task="osm",
        text_source="summary",
    )

    assert result["prediction"] == ["wood"]
    assert result["prediction_labels"] == ["wood"]
    assert result["parse_status"] == "ok"
    assert result["metadata"]["attempt_count"] == 2
    assert result["metadata"]["resolved_from_retry"] is True


def test_classify_multilabel_keeps_error_when_retry_still_empty():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": '{"labels": []}'}}]},
        {"choices": [{"message": {"content": '{"labels": []}'}}]},
    ]
    classifier._llm = mock_llm

    result = classifier.classify_multilabel(
        text="Bois.",
        allowed_labels=["meadow", "wood"],
        task="osm",
        text_source="summary",
    )

    assert result["prediction"] is None
    assert result["prediction_labels"] == []
    assert result["parse_status"] == "error"
    assert result["metadata"]["attempt_count"] == 2
    assert result["metadata"]["resolved_from_retry"] is False


def test_invalid_label_returns_error_with_full_metadata():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"labels": ["residential"]}'}}]
    }
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Ville.",
        allowed_labels=["21", "31"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )
    assert result["parse_status"] == "error"
    assert "residential" in result["error"]
    assert result["raw_response"] == '{"labels": ["residential"]}'
    assert result["prediction"] is None
    assert result["metadata"]["prompt"]
    assert result["metadata"]["system_prompt"]
    assert result["metadata"]["allowed_labels"] == ["21", "31"]


def test_invalid_json_returns_error_with_full_metadata():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "not json at all"}}]
    }
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Texte.",
        allowed_labels=["21"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )
    assert result["parse_status"] == "error"
    assert "error" in result
    assert result["raw_response"] == "not json at all"
    assert result["metadata"]["prompt"]
    assert result["metadata"]["system_prompt"]
    assert result["metadata"]["allowed_labels"] == ["21"]


def test_infrastructure_exception_returns_error_with_full_metadata():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = RuntimeError("boom")
    classifier._llm = mock_llm

    result = classifier.classify_single_label(
        text="Texte.",
        allowed_labels=["21"],
        label_descriptions={},
        task="corine_level2",
        text_source="summary",
    )
    assert result["parse_status"] == "error"
    assert "boom" in result["error"]
    assert result["raw_response"] is None
    assert result["metadata"]["prompt"]
    assert result["metadata"]["system_prompt"]
    assert result["metadata"]["allowed_labels"] == ["21"]


def test_multilabel_infrastructure_exception_returns_error_with_full_metadata():
    classifier = LLMClassifier(model_path=None)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.side_effect = RuntimeError("boom")
    classifier._llm = mock_llm

    result = classifier.classify_multilabel(
        text="Texte.",
        allowed_labels=["meadow"],
        task="osm",
        text_source="summary",
    )

    assert result["parse_status"] == "error"
    assert "boom" in result["error"]
    assert result["raw_response"] is None
    assert result["metadata"]["prompt"]
    assert result["metadata"]["system_prompt"]
    assert result["metadata"]["allowed_labels"] == ["meadow"]


def test_gpu_optimization_enabled():
    classifier = LLMClassifier(model_path="model.gguf")
    mock_llama_cpp = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
        classifier._get_llm()
    _, kwargs = mock_llama_cpp.Llama.from_pretrained.call_args
    assert kwargs["n_gpu_layers"] == -1


def test_model_path_none_uses_default_filename():
    classifier = LLMClassifier(model_path=None)
    mock_llama_cpp = MagicMock()
    with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
        classifier._get_llm()
    _, kwargs = mock_llama_cpp.Llama.from_pretrained.call_args
    assert kwargs["filename"] == "Qwen3.6-27B-Q4_0.gguf"
