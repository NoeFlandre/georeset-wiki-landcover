# Data CLIs

This package contains commands that create or update data artifacts used by the
experiments.

## Files

- `__init__.py`: marks the command package; keep it side-effect free.
- `build_clip_label_splits.py`: creates deterministic train/eval tiers for the
  CLIP weak-label experiment.
- `build_evidence_cards.py`: builds deterministic no-LLM evidence-card text
  sources from existing evidence, article-type, spatial-confidence, and quality
  metadata.
- `build_evidence_highlights.py`: builds deterministic no-LLM highlighted raw
  content from existing evidence metadata and article contents.
- `build_retrieved_evidence_windows.py`: builds deterministic sentence-window
  text sources around evidence matches plus random-window controls.
- `classify_articles.py`: runs article-text classification for CORINE or OSM,
  including shuffled controls and checkpoint-based resumability.
- `compute_corine_spatial_confidence.py`: computes CORINE buffer-purity
  diagnostics for frozen classification page IDs.
- `embed_clip_patches.py`: embeds cached Sentinel-2 patches with frozen CLIP
  image features.
- `fetch_sentinel_patches.py`: fetches Sentinel-2 RGB image patches for CLIP
  weak-label examples.
- `fetch_wikipedia_article_types.py`: fetches French Wikipedia category and
  page-property metadata for article-type diagnostics.
- `filter_pipeline.py`: filters project data against CORINE/OSM spatial policy,
  optionally refetches OSM/Wikipedia inputs, prunes stale artifacts, and
  regenerates maps/distribution outputs.
- `json_inputs.py`: shared JSON loading helpers for data-building CLIs.
- `summarize_articles.py`: summarizes fetched Wikipedia article content with the
  configured local LLM backend.
- `summarize_landuse_evidence.py`: extracts no-place, one- to three-sentence
  land-use evidence summaries and stores relevance/uncertainty metadata.

## Safety Expectations

- Use `--dry-run` or `--audit-only` when checking destructive/pruning behavior.
- Use isolated output directories for new model families or experiment variants.
- Do not change prompts, labels, summaries, seed, or temperature when the goal
  is model-family comparison only.
- Write artifacts atomically through `georeset_wiki_landcover.utils.json_io`.
