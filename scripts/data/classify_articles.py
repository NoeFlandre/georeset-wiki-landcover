"""CLI wrapper for article text classification."""

from typing import Any, cast

from src.classification import runner as _runner
from src.classification.llm_classifier import LLMClassifier
from src.classification.text_sources import apply_shuffled_text_control

CLASSIFICATION_POLICY_VERSION = _runner.CLASSIFICATION_POLICY_VERSION
compute_metrics = _runner.compute_metrics
load_text_source = _runner.load_text_source
parse_args = _runner.parse_args


def prediction_fingerprint(
    task: str,
    text_source: str,
    model_path: str,
    seed: int,
    temperature: float,
    allowed_labels: list[str],
) -> str:
    previous_version = _runner.CLASSIFICATION_POLICY_VERSION
    _runner.CLASSIFICATION_POLICY_VERSION = CLASSIFICATION_POLICY_VERSION
    try:
        return _runner.prediction_fingerprint(
            task, text_source, model_path, seed, temperature, allowed_labels
        )
    finally:
        _runner.CLASSIFICATION_POLICY_VERSION = previous_version


__all__ = [
    "CLASSIFICATION_POLICY_VERSION",
    "LLMClassifier",
    "apply_shuffled_text_control",
    "compute_metrics",
    "load_text_source",
    "main",
    "parse_args",
    "prediction_fingerprint",
]


def main(argv: list[str] | None = None) -> None:
    """Run the classifier CLI, preserving monkeypatch compatibility for tests."""
    cast(Any, _runner).LLMClassifier = LLMClassifier
    _runner.main(argv)


if __name__ == "__main__":
    main()
