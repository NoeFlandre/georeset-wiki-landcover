import json
from typing import Any

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
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
    ) -> dict:
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

    def _error_result(
        self,
        error: str,
        raw_response: str | None,
        task: str,
        text_source: str,
        prompt: str,
        allowed_labels: list[str],
    ) -> dict[str, Any]:
        return {
            "prediction": None,
            "parse_status": "error",
            "error": error,
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
    ) -> dict[str, Any]:
        user_prompt = self._build_user_prompt(
            task, text_source, allowed_labels, label_descriptions, text
        )
        raw_response = None
        try:
            llm = self._get_llm()
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                seed=self.seed,
                response_format={"type": "json_object", "schema": SINGLE_SCHEMA},
            )
            raw_response = response["choices"][0]["message"]["content"]
            payload = json.loads(raw_response)
            label = payload.get("label")
            if not isinstance(label, str):
                return self._error_result(
                    "label is not a string",
                    raw_response,
                    task,
                    text_source,
                    user_prompt,
                    allowed_labels,
                )
            if label not in allowed_labels:
                return self._error_result(
                    f"label '{label}' not in allowed labels",
                    raw_response,
                    task,
                    text_source,
                    user_prompt,
                    allowed_labels,
                )
            return {
                "prediction": label,
                "parse_status": "ok",
                "error": None,
                "raw_response": raw_response,
                "metadata": self._metadata(task, text_source, user_prompt, allowed_labels),
            }
        except Exception as exc:
            return self._error_result(
                str(exc),
                raw_response,
                task,
                text_source,
                user_prompt,
                allowed_labels,
            )

    def classify_multilabel(
        self,
        text: str,
        allowed_labels: list[str],
        task: str,
        text_source: str,
    ) -> dict[str, Any]:
        user_prompt = self._build_user_prompt(
            task, text_source, allowed_labels, {}, text
        )
        raw_response = None
        try:
            llm = self._get_llm()
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                seed=self.seed,
                response_format={"type": "json_object", "schema": MULTI_SCHEMA},
            )
            raw_response = response["choices"][0]["message"]["content"]
            payload = json.loads(raw_response)
            raw_labels = payload.get("labels")
            if not isinstance(raw_labels, list):
                return self._error_result(
                    "labels is not a list",
                    raw_response,
                    task,
                    text_source,
                    user_prompt,
                    allowed_labels,
                )
            seen, deduped = set(), []
            for lbl in raw_labels:
                if not isinstance(lbl, str):
                    return self._error_result(
                        f"invalid label type {type(lbl)}",
                        raw_response,
                        task,
                        text_source,
                        user_prompt,
                        allowed_labels,
                    )
                if lbl not in seen:
                    seen.add(lbl)
                    deduped.append(lbl)
            deduped.sort()
            invalid = [lbl for lbl in deduped if lbl not in allowed_labels]
            if invalid:
                return self._error_result(
                    f"labels {invalid} not in allowed labels",
                    raw_response,
                    task,
                    text_source,
                    user_prompt,
                    allowed_labels,
                )
            return {
                "prediction": deduped,
                "parse_status": "ok",
                "error": None,
                "raw_response": raw_response,
                "metadata": self._metadata(task, text_source, user_prompt, allowed_labels),
            }
        except Exception as exc:
            return self._error_result(
                str(exc),
                raw_response,
                task,
                text_source,
                user_prompt,
                allowed_labels,
            )
