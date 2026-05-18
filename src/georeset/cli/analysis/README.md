# Analysis CLIs

This package contains command-line tools that read already-generated artifacts
and produce derived tables, summaries, or reevaluations.

## Files

- `__init__.py`: marks the command package; keep it side-effect free.
- `evaluate_article_type_relevance_stratified.py`: stratifies frozen prediction
  metrics by deterministic Wikipedia article type, land-cover relevance, and
  spatial confidence.
- `evaluate_evidence_card_experiment.py`: compares deterministic evidence-card
  classification outputs against previous summary/content and evidence-summary
  baselines.
- `evaluate_evidence_highlights_experiment.py`: compares deterministic
  evidence-highlighted raw-content outputs against previous text-source
  baselines.
- `evaluate_predictions_with_spatial_confidence.py`: joins frozen predictions to
  CORINE spatial-confidence data and recomputes metrics on fixed spatial
  subsets.
- `evaluate_relevance_stratified_predictions.py`: reevaluates frozen
  predictions by land-cover relevance and combined relevance/spatial filters.
- `evaluate_retrieved_evidence_windows_experiment.py`: compares retrieved
  evidence sentence windows against full, random, and shuffled text sources.
- `evaluate_supervision_quality_score.py`: computes analysis-only supervision
  quality scores and evaluates quality-bin subsets.
- `run_clip_linear_probe_experiment.py`: trains NumPy linear probes over cached
  frozen CLIP embeddings and split tiers.
- `run_clip_zero_shot_experiment.py`: evaluates zero-shot CLIP prompts on cached
  Sentinel patch embeddings.
- `run_corine_analysis.py`: fetches or loads project-scoped OSM polygons,
  computes CORINE distributions, and writes map/distribution artifacts.
- `summarize_classification_experiment.py`: regenerates overview tables,
  shuffled deltas, majority baselines, and experiment summaries from prediction
  artifacts.

## Output Discipline

These tools must not mutate frozen parent experiment inputs. Write new derived
artifacts into a separate output directory and include a manifest when creating
an experiment folder.
