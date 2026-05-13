# Experiment Reports

This directory contains human-readable reports for GeoReset experiments. Reports
are grouped chronologically so the research path is clear as new experiments are
added.

## Reading Order

Start with [`000_experiment_summary.md`](000_experiment_summary.md) for a
short, plain-language overview of the full experiment path, methods, results,
and next steps.

1. [`001_qwen_e2e_shuffled_control/analysis.md`](001_qwen_e2e_shuffled_control/analysis.md)
   - First full article-text classification batch with Qwen.
   - Covers CORINE level-2 and OSM classification across summaries, no-place
     summaries, raw content, and shuffled controls.
2. [`002_corine_spatial_confidence/README.md`](002_corine_spatial_confidence/README.md)
   - Defines CORINE spatial-confidence diagnostics.
   - Explains buffer purity, EPSG:2154 area calculations, and why full CORINE
     including artificial classes is used.
3. [`003_qwen_spatial_confidence_reevaluation/analysis.md`](003_qwen_spatial_confidence_reevaluation/analysis.md)
   - Re-evaluates the Qwen predictions on spatially reliable subsets without
     rerunning the LLM.
   - Adds majority baselines, shuffled deltas by subset, and class-distribution
     diagnostics.
4. [`004_gemma4_model_rerun_and_comparison/analysis.md`](004_gemma4_model_rerun_and_comparison/analysis.md)
   - Repeats the same protocol with Gemma 4 31B IT Q4_0.
   - Compares Gemma against Qwen under the non-spatial and spatial-confidence
     evaluations.
5. [`005_landuse_evidence_summary/analysis.md`](005_landuse_evidence_summary/analysis.md)
   - Adds deterministic no-place land-use evidence summaries generated with Qwen.
   - Re-runs Qwen and Gemma classifiers on the evidence summaries and shuffled
     evidence controls, then compares non-spatial and spatial-confidence results.
6. [`006_relevance_stratified_evaluation/analysis.md`](006_relevance_stratified_evaluation/analysis.md)
   - Reuses frozen Qwen and Gemma predictions without rerunning any LLM.
   - Tests whether land-use evidence metadata is useful as a relevance filter
     for raw content, generic summaries, shuffled controls, and spatial subsets.
7. [`007_article_type_relevance_stratified_evaluation/analysis.md`](007_article_type_relevance_stratified_evaluation/analysis.md)
   - Fetches French Wikipedia category/page metadata without refetching content.
   - Tests whether category-derived article type explains where classification
     works, and how article type interacts with land-cover relevance and CORINE
     spatial confidence.
8. [`008_supervision_quality_score/analysis.md`](008_supervision_quality_score/analysis.md)
   - Re-scores the frozen Qwen and Gemma predictions with deterministic quality
     rules using evidence metadata, spatial confidence, and article type metadata.
   - Adds quality-bin and recommended-use partitions and recomputes subset metrics
     for CORINE and OSM.
9. `009_evidence_card_text_source/analysis.md`
   - Planned next report for deterministic evidence-card text sources.
   - Tests whether structured metadata cards improve over the previous
     land-use evidence summary and whether cards help raw content on high-quality
     subsets.

## Data Artifact Map

The reports cite artifacts under `data/experiments/`:

- `article_text_classification_e2e_v1/`
- `article_text_classification_e2e_with_shuffled_control_v1/`
- `article_text_classification_shuffled_control_v1/`
- `corine_spatial_confidence_v1/`
- `article_text_classification_spatial_confidence_v1/`
- `article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/`
- `article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/`
- `model_comparison_qwen_vs_gemma4_31b_it_q4_0/`
- `article_text_classification_landuse_evidence_v1__qwen3_6_27b_q4_0/`
- `article_text_classification_landuse_evidence_v1__gemma4_31b_it_q4_0/`
- `article_text_classification_landuse_evidence_spatial_confidence_v1__qwen3_6_27b_q4_0/`
- `article_text_classification_landuse_evidence_spatial_confidence_v1__gemma4_31b_it_q4_0/`
- `landuse_evidence_comparison_v1/`
- `article_text_classification_relevance_stratified_v1/`
- `article_text_classification_article_type_relevance_stratified_v1/`
- `article_text_supervision_quality_score_v1/`
- `article_text_evidence_card_v1__qwen3_6_27b_q4_0/`
- `evidence_card_comparison_v1/`

Working classifier checkpoints live separately under `data/classification/runs/`.
Those are useful for resumability, but the stable research outputs are the
frozen experiment directories above.

## Naming Convention

Use a numeric prefix for future report folders:

```text
005_short_experiment_name/
  analysis.md
```

Keep generated data in `data/experiments/<experiment_id>/` and keep narrative
analysis in `docs/experiments/<number>_<short_name>/`.
