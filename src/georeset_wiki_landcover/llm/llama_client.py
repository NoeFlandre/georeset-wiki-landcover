"""Shared llama-cpp chat completion client."""

from pathlib import Path
from typing import Any, Protocol, cast

from georeset_wiki_landcover.config import ModelSettings

DEFAULT_REPO_ID = "unsloth/Qwen3.6-27B-GGUF"
DEFAULT_GGUF_FILENAME = ModelSettings().model_path
DEFAULT_CONTEXT_WINDOW = 8192


class JsonChatClient(Protocol):
    """Protocol for schema-constrained chat completions."""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float,
        max_tokens: int | None = None,
    ) -> str:
        """Return the raw JSON string emitted by the model."""


class LlamaChatClient:
    """Lazy llama-cpp client with shared runtime defaults."""

    def __init__(
        self,
        model_path: str | None,
        seed: int = 42,
        repo_id: str | None = DEFAULT_REPO_ID,
        n_gpu_layers: int = -1,
        n_ctx: int = DEFAULT_CONTEXT_WINDOW,
    ):
        self.model_path = model_path
        self.seed = seed
        self.repo_id = repo_id
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._llm: Any | None = None

    @property
    def model_filename(self) -> str:
        return self.model_path if self.model_path else DEFAULT_GGUF_FILENAME

    @property
    def model_identity(self) -> dict[str, str | None]:
        return {
            "model": self.model_filename,
            "model_repo_id": self.repo_id,
        }

    def _get_llm(self) -> Any:
        if self._llm is None:
            import llama_cpp

            model_path = Path(self.model_filename)
            if model_path.exists():
                self._llm = llama_cpp.Llama(
                    model_path=str(model_path),
                    n_gpu_layers=self.n_gpu_layers,
                    seed=self.seed,
                    n_ctx=self.n_ctx,
                )
            else:
                self._llm = llama_cpp.Llama.from_pretrained(
                    repo_id=self.repo_id or DEFAULT_REPO_ID,
                    filename=self.model_filename,
                    n_gpu_layers=self.n_gpu_layers,
                    seed=self.seed,
                    n_ctx=self.n_ctx,
                )
        return self._llm

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        temperature: float,
        max_tokens: int | None = None,
    ) -> str:
        response = self._get_llm().create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            seed=self.seed,
            max_tokens=max_tokens,
            response_format={"type": "json_object", "schema": schema},
        )
        return cast(str, response["choices"][0]["message"]["content"])
