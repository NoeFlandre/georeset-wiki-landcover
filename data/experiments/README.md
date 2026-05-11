# Frozen Experiment Artifacts

This directory contains stable, citable experiment outputs. Each subdirectory is
an experiment ID and should be treated as immutable once analysis reports cite
it.

## Current Chronology

1. `article_text_classification_e2e_v1/`
   - Early non-shuffled article-text classification batch.
2. `article_text_classification_e2e_with_shuffled_control_v1/`
   - Qwen full batch with primary text sources and shuffled controls.
3. `corine_spatial_confidence_v1/`
   - CORINE buffer-purity diagnostics reused by later reevaluations.
4. `article_text_classification_shuffled_control_v1/`
   - Shuffled-only extracted folder retained from the first shuffled-control
     organization pass.
5. `article_text_classification_spatial_confidence_v1/`
   - Qwen predictions reevaluated on CORINE spatial-confidence subsets.
6. `article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/`
   - Gemma rerun using the same protocol as the Qwen shuffled-control batch.
7. `article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/`
   - Gemma predictions reevaluated on the same spatial-confidence subsets.
8. `model_comparison_qwen_vs_gemma4_31b_it_q4_0/`
   - Direct Qwen-vs-Gemma comparison tables and summary.

## Rule of Thumb

- Use `data/classification/runs/` for resumable working checkpoints.
- Copy a completed run into `data/experiments/<experiment_id>/` when it becomes
  a frozen research artifact.
- Put human narrative analysis in `docs/experiments/<number>_<short_name>/`.
