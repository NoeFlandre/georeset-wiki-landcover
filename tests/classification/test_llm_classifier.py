from unittest.mock import MagicMock, patch

from src.classification.llm_classifier import LLMClassifier


def test_classify_single_label_returns_ok_with_valid_response():
    classifier = LLMClassifier(model_path="model.gguf", seed=123, temperature=0.0)
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"labels": ["31"]}'}}]
    }
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
    assert result["error"] is None
    assert result["metadata"]["model"] == "model.gguf"
    assert result["metadata"]["seed"] == 123
    assert result["metadata"]["temperature"] == 0.0
    assert result["metadata"]["task"] == "corine_level2"
    assert result["metadata"]["text_source"] == "summary"
    _, kwargs = mock_llm.create_chat_completion.call_args
    assert kwargs["response_format"]["type"] == "json_object"
    assert kwargs["temperature"] == 0.0


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
        text="Texte.", allowed_labels=["21"], label_descriptions={},
        task="corine_level2", text_source="summary",
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
        text="Texte.", allowed_labels=["21"], label_descriptions={},
        task="corine_level2", text_source="summary",
    )
    assert result["parse_status"] == "error"
    assert "boom" in result["error"]
    assert result["raw_response"] is None
    assert result["metadata"]["prompt"]
    assert result["metadata"]["system_prompt"]
    assert result["metadata"]["allowed_labels"] == ["21"]


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
