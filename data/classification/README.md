# Classification Working Outputs

This directory stores resumable classifier outputs before they are copied into
frozen experiment folders under `data/experiments/`.

## Layout

- `runs/default/`: default scratch output directory for local or cluster
  classification runs.
- `runs/qwen3_6_27b_q4_0/`: working outputs from the Qwen classifier batch.
- `runs/gemma4_31b_it_q4_0/`: working outputs from the Gemma classifier batch,
  including cluster submission logs.

Each run folder contains flat files named:

- `{task}_{text_source}_predictions.json`
- `{task}_{text_source}_metrics.json`

The flat naming is intentional because the classifier resumes by task and text
source within a single chosen `--output-dir`.

## Frozen Results

Use `data/experiments/` for stable, citable analysis artifacts. Working outputs
in this directory can be regenerated or superseded by later runs.

Current frozen experiment folders include:

- `data/experiments/article_text_classification_e2e_with_shuffled_control_v1/`
- `data/experiments/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0/`
- `data/experiments/article_text_classification_spatial_confidence_v1/`
- `data/experiments/article_text_classification_spatial_confidence_v1__gemma4_31b_it_q4_0/`
- `data/experiments/model_comparison_qwen_vs_gemma4_31b_it_q4_0/`

## Notes

Historical duplicate browsing folders such as `primary/`, `shuffled_control/`,
and `by_task/` were removed after verifying they were byte-identical duplicates
of the Qwen run files now stored in `runs/qwen3_6_27b_q4_0/`.
