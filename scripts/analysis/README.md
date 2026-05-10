# Analysis Script Wrappers

This folder keeps backwards-compatible repository wrappers for packaged
analysis CLIs.

## Wrappers

- `run_corine_analysis.py` wraps
  `georeset.cli.analysis.run_corine_analysis`.
- `summarize_classification_experiment.py` wraps
  `georeset.cli.analysis.summarize_classification_experiment`.
- `evaluate_predictions_with_spatial_confidence.py` wraps
  `georeset.cli.analysis.evaluate_predictions_with_spatial_confidence`.

## Preferred Usage

Use the packaged entry points in new commands:

```bash
uv run georeset-run-corine-analysis
uv run georeset-summarize-classification-experiment --help
uv run georeset-evaluate-spatial-confidence --help
```
