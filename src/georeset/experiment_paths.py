"""Canonical locations for frozen experiment artifacts."""

from __future__ import annotations

from pathlib import Path

EXPERIMENTS_ROOT = Path("data/experiments")

EXPERIMENT_GROUPS: dict[str, str] = {
    "article_text_classification_e2e_v1": "001_qwen_e2e_shuffled_control",
    "article_text_classification_shuffled_control_v1": "001_qwen_e2e_shuffled_control",
    "article_text_classification_e2e_with_shuffled_control_v1": ("001_qwen_e2e_shuffled_control"),
    "corine_spatial_confidence_v1": "002_corine_spatial_confidence",
    "article_text_classification_spatial_confidence_v1": (
        "003_qwen_spatial_confidence_reevaluation"
    ),
    "article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0": (
        "004_gemma4_model_rerun_and_comparison"
    ),
    "article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0": (
        "004_gemma4_model_rerun_and_comparison"
    ),
    "model_comparison_qwen_vs_gemma4_31b_it_q4_0": ("004_gemma4_model_rerun_and_comparison"),
    "article_text_classification_landuse_evidence_v1__qwen3_6_27b_q4_0": (
        "005_landuse_evidence_summary"
    ),
    "article_text_classification_landuse_evidence_v1__gemma4_31b_it_q4_0": (
        "005_landuse_evidence_summary"
    ),
    "article_text_classification_landuse_evidence_spatial_confidence_v1__qwen3_6_27b_q4_0": (
        "005_landuse_evidence_summary"
    ),
    "article_text_classification_landuse_evidence_spatial_confidence_v1__gemma4_31b_it_q4_0": (
        "005_landuse_evidence_summary"
    ),
    "landuse_evidence_comparison_v1": "005_landuse_evidence_summary",
    "article_text_classification_relevance_stratified_v1": ("006_relevance_stratified_evaluation"),
    "article_text_classification_article_type_relevance_stratified_v1": (
        "007_article_type_relevance_stratified_evaluation"
    ),
    "article_text_supervision_quality_score_v1": "008_supervision_quality_score",
    "article_text_evidence_card_v1__qwen3_6_27b_q4_0": ("009_evidence_card_text_source"),
    "evidence_card_comparison_v1": "009_evidence_card_text_source",
    "article_text_evidence_highlights_v1__qwen3_6_27b_q4_0": ("010_evidence_highlighted_content"),
    "article_text_evidence_highlights_v1__gemma4_31b_it_q4_0": ("010_evidence_highlighted_content"),
    "evidence_highlights_comparison_v1": "010_evidence_highlighted_content",
    "article_text_retrieved_evidence_windows_v1__qwen3_6_27b_q4_0": (
        "011_retrieved_evidence_windows"
    ),
    "article_text_retrieved_evidence_windows_v1__gemma4_31b_it_q4_0": (
        "011_retrieved_evidence_windows"
    ),
    "retrieved_evidence_windows_comparison_v1": "011_retrieved_evidence_windows",
    "clip_linear_probe_weak_labels_v1": "012_clip_linear_probe_weak_labels",
    "article_text_subset_randomization_controls_v1": ("013_subset_randomization_controls"),
    "quality_weighted_multiscale_image_probe_v1": ("014_quality_weighted_multiscale_image_probe"),
}


def experiment_artifact_dir(artifact_id: str, *, root: Path = EXPERIMENTS_ROOT) -> Path:
    """Return the canonical directory for a frozen experiment artifact."""
    group = EXPERIMENT_GROUPS.get(artifact_id)
    if group is None:
        return root / artifact_id
    return root / group / artifact_id


def experiment_artifact_file(artifact_id: str, *parts: str, root: Path = EXPERIMENTS_ROOT) -> Path:
    """Return a child file or directory path inside a canonical artifact dir."""
    return experiment_artifact_dir(artifact_id, root=root).joinpath(*parts)


def resolve_existing_experiment_artifact_dir(
    artifact_id: str, *, root: Path = EXPERIMENTS_ROOT
) -> Path:
    """Resolve a frozen artifact, accepting legacy flat layouts for compatibility."""
    canonical = experiment_artifact_dir(artifact_id, root=root)
    if canonical.exists():
        return canonical
    legacy = root / artifact_id
    if legacy.exists():
        return legacy
    return canonical
