"""Shared JSON-facing record contracts.

These TypedDicts document persisted payload shapes. They intentionally mirror
the existing JSON keys and do not change serialization.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

ClassificationTask = Literal["corine_level2", "osm"]
ParseStatus = Literal["ok", "error", "ambiguous"]
ClassificationTarget = str | list[str]


class GeoBounds(TypedDict):
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float


class ArticleMeta(TypedDict, total=False):
    pageid: int
    title: str
    lat: float
    lon: float
    url: str


class ArticleContent(ArticleMeta, total=False):
    content: str


class SummaryMetadata(TypedDict, total=False):
    model: str
    seed: int
    temperature: float
    prompt: str
    system_prompt: str
    summary_mode: str


class SummaryRecord(ArticleContent, total=False):
    summary: str
    metadata: SummaryMetadata


class PerLabelMetric(TypedDict):
    support: int
    precision: float
    recall: float
    f1: float


class SingleLabelMetricResult(TypedDict):
    n_eligible: int
    n_predicted_ok: int
    n_parse_error: int
    coverage: float
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: dict[str, PerLabelMetric]
    task: NotRequired[str]
    text_source: NotRequired[str]
    allowed_labels: NotRequired[list[str]]
    labels_evaluated: NotRequired[list[str]]


class MultiLabelMetricResult(TypedDict):
    n_eligible: int
    n_predicted_ok: int
    n_parse_error: int
    coverage: float
    exact_match_accuracy: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: dict[str, PerLabelMetric]
    task: NotRequired[str]
    text_source: NotRequired[str]
    allowed_labels: NotRequired[list[str]]
    labels_evaluated: NotRequired[list[str]]


class SpatialSubsetMetricResult(TypedDict, total=False):
    n: int
    n_predicted_ok: int
    n_parse_error: int
    coverage: float
    accuracy: float
    balanced_accuracy: float
    exact_match_accuracy: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    jaccard: float
    hamming_loss: float
    majority_accuracy: float
    majority_balanced_accuracy: float
    majority_macro_f1: float
    majority_labelset_exact_match_accuracy: float
    empty_set_exact_match_accuracy: float
    delta_vs_majority_accuracy: float
    delta_vs_majority_balanced_accuracy: float
    delta_vs_majority_macro_f1: float


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
