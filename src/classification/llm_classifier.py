import json
from typing import Any, cast

from src.classification.prediction_parser import normalize_prediction_response
from src.classification.types import PredictionResult

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
    "required": ["labels"],
    "additionalProperties": False,
}
MULTI_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "string"}}},
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

    def __init__(self, model_path: str | None, seed: int = 42, temperature: float = 0.0):
        self.model_path = model_path
        self.seed = seed
        self.temperature = temperature
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            import llama_cpp
            self._llm = llama_cpp.Llama.from_pretrained(
                repo_id="unsloth/Qwen3.6-27B-GGUF",
                filename=self.model_path if self.model_path else "Qwen3.6-27B-Q4_0.gguf",
                n_gpu_layers=-1,
                seed=self.seed,
                n_ctx=8192,
            )
        return self._llm

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
            "model": self.model_path if self.model_path else "Qwen3.6-27B-Q4_0.gguf",
            "seed": self.seed,
            "temperature": self.temperature,
            "prompt": prompt,
            "system_prompt": SYSTEM_PROMPT,
            "allowed_labels": sorted(allowed_labels),
        }

    def _call_llm(self, user_prompt: str, schema: dict[str, Any]) -> str:
        llm = self._get_llm()
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            seed=self.seed,
            response_format={"type": "json_object", "schema": schema},
        )
        return cast(str, response["choices"][0]["message"]["content"])

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
        user_prompt = self._build_user_prompt(
            task, text_source, allowed_labels, {}, text
        )
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
