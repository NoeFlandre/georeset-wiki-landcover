"""Shared type contracts for classification modules."""

from typing import Any, Literal, TypedDict

ClassificationTask = Literal["corine_level2", "osm"]
ParseStatus = Literal["ok", "error", "ambiguous"]
ClassificationTarget = str | list[str]


class PredictionResult(TypedDict, total=False):
    prediction: ClassificationTarget | None
    prediction_labels: list[str]
    parse_status: ParseStatus
    raw_response: str | None
    error: str | None
    metadata: dict[str, Any]


class PredictionRecord(TypedDict):
    pageid: str
    title: str
    target: ClassificationTarget
    prediction: ClassificationTarget | None
    prediction_labels: list[str]
    parse_status: ParseStatus | None
    raw_response: str | None
    error: str | None
    metadata: dict[str, Any]
