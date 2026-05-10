from unittest.mock import MagicMock, patch

from src.llm.llama_client import DEFAULT_GGUF_FILENAME, DEFAULT_REPO_ID, LlamaChatClient


def test_llama_chat_client_lazy_loads_shared_runtime_config():
    client = LlamaChatClient(model_path=None, seed=123)
    mock_llama_cpp = MagicMock()
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": '{"ok": true}'}}]
    }
    mock_llama_cpp.Llama.from_pretrained.return_value = mock_llm

    with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
        result = client.complete_json(
            system_prompt="system",
            user_prompt="user",
            schema={"type": "object"},
            temperature=0.0,
            max_tokens=128,
        )
        second = client.complete_json(
            system_prompt="system",
            user_prompt="user again",
            schema={"type": "object"},
            temperature=0.0,
        )

    assert result == '{"ok": true}'
    assert second == '{"ok": true}'
    mock_llama_cpp.Llama.from_pretrained.assert_called_once_with(
        repo_id=DEFAULT_REPO_ID,
        filename=DEFAULT_GGUF_FILENAME,
        n_gpu_layers=-1,
        seed=123,
        n_ctx=8192,
    )
    _, kwargs = mock_llm.create_chat_completion.call_args
    assert kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user again"},
    ]
    assert kwargs["temperature"] == 0.0
    assert kwargs["seed"] == 123
    assert kwargs["max_tokens"] is None
    assert kwargs["response_format"] == {"type": "json_object", "schema": {"type": "object"}}
