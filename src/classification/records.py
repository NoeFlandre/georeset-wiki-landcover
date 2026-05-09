"""Prediction record helpers for classification checkpoints."""

from typing import Any

from src.classification.types import ClassificationTarget, PredictionRecord, PredictionResult


def should_skip_record(
    record: dict[str, Any] | None, fingerprint: str, retry_failed: bool
) -> bool:
    if not record or record.get("parse_status") != "ok":
        return False
    return retry_failed or record.get("metadata", {}).get("fingerprint") == fingerprint


def build_prediction_record(
    *,
    pageid: str,
    title: str,
    target: ClassificationTarget,
    result: PredictionResult,
    fingerprint: str,
    extra_metadata: dict[str, Any] | None = None,
) -> PredictionRecord:
    metadata = {**result.get("metadata", {}), "fingerprint": fingerprint}
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "pageid": pageid,
        "title": title,
        "target": target,
        "prediction": result.get("prediction"),
        "prediction_labels": result.get("prediction_labels", []),
        "parse_status": result.get("parse_status"),
        "raw_response": result.get("raw_response"),
        "error": result.get("error"),
        "metadata": metadata,
    }
