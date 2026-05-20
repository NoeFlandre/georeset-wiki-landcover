import json
from typing import Any

from georeset_wiki_landcover.classification.prediction_parser import normalize_prediction_response
from georeset_wiki_landcover.classification.types import PredictionResult
from georeset_wiki_landcover.llm.llama_client import (
    DEFAULT_GGUF_FILENAME,
    JsonChatClient,
    LlamaChatClient,
)

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 1,
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}
MULTI_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Tu es un assistant qui classe les articles Wikipedia français en catégories "
    "de couverture du sol. Réponds uniquement avec un objet JSON respectant le schéma "
    "fourni. Pas de Markdown, pas d'explication."
)


class LLMClassifier:
    """LLM-backed classifier.

    Error policy: JSON validation errors become structured parse records, and
    exceptions from the external LLM boundary become structured error records.
    Lower-level pure helpers should not catch broad exceptions.
    """

    def __init__(
        self,
        model_path: str | None,
        seed: int = 42,
        temperature: float = 0.0,
        model_repo_id: str | None = None,
        client: JsonChatClient | None = None,
    ):
        self.model_path = model_path
        self.model_repo_id = model_repo_id
        self.seed = seed
        self.temperature = temperature
        self._client = client or LlamaChatClient(
            model_path=model_path, seed=seed, repo_id=model_repo_id
        )

    def _get_llm(self) -> Any:
        if not isinstance(self._client, LlamaChatClient):
            raise TypeError("_get_llm is only available for the default LlamaChatClient")
        return self._client._get_llm()

    @property
    def _llm(self) -> Any | None:
        if isinstance(self._client, LlamaChatClient):
            return self._client._llm
        return None

    @_llm.setter
    def _llm(self, value: Any) -> None:
        if isinstance(self._client, LlamaChatClient):
            self._client._llm = value

    def _build_user_prompt(
        self,
        task: str,
        text_source: str,
        allowed_labels: list[str],
        label_descriptions: dict[str, str],
        text: str,
    ) -> str:
        lines = [
            f"task: {task}",
            f"text_source: {text_source}",
            f"allowed_labels: {allowed_labels}",
        ]
        if label_descriptions:
            lines.append(f"label_descriptions: {json.dumps(label_descriptions)}")
        truncated = text[:24000] if len(text) > 24000 else text
        lines.append(f"text: {truncated}")
        return "\n".join(lines)

    def _metadata(
        self, task: str, text_source: str, prompt: str, allowed_labels: list[str]
    ) -> dict[str, Any]:
        return {
            "task": task,
            "text_source": text_source,
            "model": self.model_path if self.model_path else DEFAULT_GGUF_FILENAME,
            "model_repo_id": self.model_repo_id,
            "seed": self.seed,
            "temperature": self.temperature,
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "allowed_labels": sorted(allowed_labels),
        }

    def _call_llm(self, user_prompt: str, schema: dict[str, Any]) -> str:
        return self._client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=schema,
            temperature=self.temperature,
        )

    def _build_retry_prompt(
        self,
        original_prompt: str,
        raw_response: str | None,
        error: str | None,
        task: str,
    ) -> str:
        if task == "corine_level2":
            instruction = (
                "Retry the classification. You must choose exactly one best label "
                "from allowed_labels. Never return an empty list. If uncertain, "
                "choose the closest valid label based only on the article text."
            )
        else:
            instruction = (
                "Retry the classification. You must choose at least one best label "
                "from allowed_labels. Never return an empty list. If uncertain, "
                "choose the closest valid label or labels based only on the article text."
            )
        return "\n".join(
            [
                original_prompt,
                "",
                "Previous invalid response:",
                raw_response or "",
                f"Previous error: {error or ''}",
                instruction,
                'Respond only with JSON like {"labels": ["label"]}.',
            ]
        )

    def _attempt_summary(self, result: PredictionResult) -> dict[str, Any]:
        return {
            "raw_response": result.get("raw_response"),
            "parse_status": result.get("parse_status"),
            "error": result.get("error"),
            "prediction_labels": result.get("prediction_labels", []),
        }

    def _attach_attempt_metadata(
        self, result: PredictionResult, attempts: list[PredictionResult]
    ) -> PredictionResult:
        metadata = dict(result.get("metadata", {}))
        metadata["attempt_count"] = len(attempts)
        metadata["attempt_history"] = [self._attempt_summary(attempt) for attempt in attempts]
        metadata["resolved_from_retry"] = len(attempts) > 1 and result.get("parse_status") == "ok"
        result["metadata"] = metadata
        return result

    def _error_result(
        self,
        error: str,
        raw_response: str | None,
        task: str,
        text_source: str,
        prompt: str,
        allowed_labels: list[str],
        prediction_labels: list[str] | None = None,
    ) -> PredictionResult:
        return {
            "prediction": None,
            "prediction_labels": prediction_labels or [],
            "parse_status": "error",
            "error": error,
            "raw_response": raw_response,
            "metadata": self._metadata(task, text_source, prompt, allowed_labels),
        }

    def _parse_single_label_response(
        self,
        raw_response: str,
        task: str,
        text_source: str,
        prompt: str,
        allowed_labels: list[str],
    ) -> PredictionResult:
        labels, error = normalize_prediction_response(raw_response, allowed_labels)

        if error:
            return self._error_result(
                error, raw_response, task, text_source, prompt, allowed_labels, labels
            )

        if len(labels) == 1:
            return {
                "prediction": labels[0],
                "prediction_labels": labels,
                "parse_status": "ok",
                "error": None,
                "raw_response": raw_response,
                "metadata": self._metadata(task, text_source, prompt, allowed_labels),
            }

        result = self._error_result(
            "multiple labels for single-label task",
            raw_response,
            task,
            text_source,
            prompt,
            allowed_labels,
            labels,
        )
        result["parse_status"] = "ambiguous"
        return result

    def _parse_multilabel_response(
        self,
        raw_response: str,
        task: str,
        text_source: str,
        prompt: str,
        allowed_labels: list[str],
    ) -> PredictionResult:
        labels, error = normalize_prediction_response(raw_response, allowed_labels)

        if error:
            # For multi-label OSM, unknown labels are non-fatal; keep only
            # in-scope labels and treat a fully unknown JSON payload as an
            # empty prediction if it is otherwise structurally valid JSON.
            if task == "osm" and error.startswith("Unknown labels in JSON:"):
                data = json.loads(raw_response)
                if isinstance(data, dict):
                    if "labels" in data and isinstance(data["labels"], list):
                        for item in data["labels"]:
                            if not isinstance(item, str):
                                return self._error_result(
                                    "JSON list fields must contain only strings",
                                    raw_response,
                                    task,
                                    text_source,
                                    prompt,
                                    allowed_labels,
                                    labels,
                                )
                    elif "label" in data and isinstance(data["label"], str):
                        # Keep behavior consistent with JSON list parsing for label
                        # CSV/text payloads.
                        pass
                    else:
                        return self._error_result(
                            error,
                            raw_response,
                            task,
                            text_source,
                            prompt,
                            allowed_labels,
                            labels,
                        )
                elif not isinstance(data, dict):
                    return self._error_result(
                        error,
                        raw_response,
                        task,
                        text_source,
                        prompt,
                        allowed_labels,
                        labels,
                    )

                return {
                    "prediction": labels,
                    "prediction_labels": labels,
                    "parse_status": "ok",
                    "error": None,
                    "raw_response": raw_response,
                    "metadata": self._metadata(task, text_source, prompt, allowed_labels),
                }

            return self._error_result(
                error, raw_response, task, text_source, prompt, allowed_labels, labels
            )

        return {
            "prediction": labels,
            "prediction_labels": labels,
            "parse_status": "ok",
            "error": None,
            "raw_response": raw_response,
            "metadata": self._metadata(task, text_source, prompt, allowed_labels),
        }

    def classify_single_label(
        self,
        text: str,
        allowed_labels: list[str],
        label_descriptions: dict[str, str],
        task: str,
        text_source: str,
    ) -> PredictionResult:
        user_prompt = self._build_user_prompt(
            task, text_source, allowed_labels, label_descriptions, text
        )
        raw_response = None
        attempts: list[PredictionResult] = []
        try:
            raw_response = self._call_llm(user_prompt, SINGLE_SCHEMA)
            first_result = self._parse_single_label_response(
                raw_response, task, text_source, user_prompt, allowed_labels
            )
            attempts.append(first_result)
            if first_result["parse_status"] == "ok":
                return self._attach_attempt_metadata(first_result, attempts)

            retry_prompt = self._build_retry_prompt(
                user_prompt,
                first_result.get("raw_response"),
                first_result.get("error"),
                task,
            )
            raw_response = self._call_llm(retry_prompt, SINGLE_SCHEMA)
            retry_result = self._parse_single_label_response(
                raw_response, task, text_source, retry_prompt, allowed_labels
            )
            attempts.append(retry_result)
            return self._attach_attempt_metadata(retry_result, attempts)
        except Exception as exc:
            result = self._error_result(
                str(exc), raw_response, task, text_source, user_prompt, allowed_labels
            )
            attempts.append(result)
            return self._attach_attempt_metadata(result, attempts)

    def classify_multilabel(
        self,
        text: str,
        allowed_labels: list[str],
        task: str,
        text_source: str,
    ) -> PredictionResult:
        user_prompt = self._build_user_prompt(task, text_source, allowed_labels, {}, text)
        raw_response = None
        attempts: list[PredictionResult] = []
        try:
            raw_response = self._call_llm(user_prompt, MULTI_SCHEMA)
            first_result = self._parse_multilabel_response(
                raw_response, task, text_source, user_prompt, allowed_labels
            )
            attempts.append(first_result)
            if first_result["parse_status"] == "ok":
                return self._attach_attempt_metadata(first_result, attempts)

            retry_prompt = self._build_retry_prompt(
                user_prompt,
                first_result.get("raw_response"),
                first_result.get("error"),
                task,
            )
            raw_response = self._call_llm(retry_prompt, MULTI_SCHEMA)
            retry_result = self._parse_multilabel_response(
                raw_response, task, text_source, retry_prompt, allowed_labels
            )
            attempts.append(retry_result)
            return self._attach_attempt_metadata(retry_result, attempts)
        except Exception as exc:
            result = self._error_result(
                str(exc), raw_response, task, text_source, user_prompt, allowed_labels
            )
            attempts.append(result)
            return self._attach_attempt_metadata(result, attempts)
