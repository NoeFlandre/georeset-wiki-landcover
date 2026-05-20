"""CLI wrapper for article text classification."""

import georeset.classification.runner as _runner
from georeset.classification.llm_classifier import LLMClassifier
from georeset.classification.text_sources import apply_shuffled_text_control

CLASSIFICATION_POLICY_VERSION = _runner.CLASSIFICATION_POLICY_VERSION
compute_metrics = _runner.compute_metrics
load_text_source = _runner.load_text_source
parse_args = _runner.parse_args
text_fingerprint = _runner.text_fingerprint


def prediction_fingerprint(
    task: str,
    text_source: str,
    model_path: str,
    seed: int,
    temperature: float,
    allowed_labels: list[str],
    model_repo_id: str | None = None,
) -> str:
    previous_version = _runner.CLASSIFICATION_POLICY_VERSION
    _runner.CLASSIFICATION_POLICY_VERSION = CLASSIFICATION_POLICY_VERSION
    try:
        return _runner.prediction_fingerprint(
            task, text_source, model_path, model_repo_id, seed, temperature, allowed_labels
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
    "text_fingerprint",
]


def main(argv: list[str] | None = None) -> None:
    """Run the classifier CLI."""

    def classifier_factory(
        model_path: str | None, model_repo_id: str | None, seed: int, temperature: float
    ) -> _runner.Classifier:
        return LLMClassifier(
            model_path=model_path,
            model_repo_id=model_repo_id,
            seed=seed,
            temperature=temperature,
        )

    _runner.main(argv, classifier_factory=classifier_factory)


if __name__ == "__main__":
    main()
