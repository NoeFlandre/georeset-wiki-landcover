# Data CLIs

This package contains commands that create or update data artifacts used by the
experiments.

## Commands

- `filter_pipeline.py`: filters project data against CORINE/OSM spatial policy,
  optionally refetches OSM/Wikipedia inputs, prunes stale content/summaries, and
  regenerates maps/distribution outputs.
- `summarize_articles.py`: summarizes fetched Wikipedia article content with the
  configured local LLM backend. It supports `--summary-mode place` and
  `--summary-mode no_place`.
- `classify_articles.py`: runs article-text classification for CORINE or OSM,
  including shuffled controls and checkpoint-based resumability.
- `compute_corine_spatial_confidence.py`: computes CORINE buffer-purity
  diagnostics for the union of page IDs in a frozen classification experiment.

## Safety Expectations

- Use `--dry-run` or `--audit-only` when checking destructive/pruning behavior.
- Use isolated output directories for new model families or experiment variants.
- Do not change prompts, labels, summaries, seed, or temperature when the goal
  is model-family comparison only.
- Write artifacts atomically through `georeset.utils.json_io`.
