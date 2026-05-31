# CLI And Script Reference

This page lists the command-line surfaces declared in `pyproject.toml` and the
repository helper scripts. Commands marked "verified" were run in this checkout
while updating the docs. Other commands are listed because they are declared
entry points or existing scripts, but they may require synced `data/`, optional
dependency groups, network access, a model file, or Grid5000 access.

## Verified Smoke Commands

These commands were verified locally:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/reproduce_small.py \
  --output-dir build/reproducibility/small \
  --clean
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
```

They create and validate a synthetic no-network, no-LLM package.

The following static CI checks were also run while updating these docs:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run ruff check .
PYTHONDONTWRITEBYTECODE=1 uv run ruff format --check .
PYTHONDONTWRITEBYTECODE=1 uv run mypy src scripts
```

## Packaged Entry Points

`pyproject.toml` exposes these commands after `uv sync --group dev`.

| Command | Implementation | Purpose |
| --- | --- | --- |
| `georeset-wiki-landcover-snapshot` | `cli.dev.snapshot` | Print a quick local data snapshot. Requires local data for useful output. |
| `georeset-wiki-landcover-run-corine-analysis` | `cli.analysis.run_corine_analysis` | Compute CORINE-in-OSM distribution and maps. Requires CORINE and OSM inputs. |
| `georeset-wiki-landcover-filter-pipeline` | `cli.data.filter_pipeline` | Cascade non-artificial CORINE filtering through OSM/wiki/content/summary artifacts. Use `--dry-run` before mutation. |
| `georeset-wiki-landcover-summarize-articles` | `cli.data.summarize_articles` | Generate article summaries from `article_contents.json`. Requires `llm` dependencies and a model. |
| `georeset-wiki-landcover-summarize-landuse-evidence` | `cli.data.summarize_landuse_evidence` | Generate land-use evidence summaries. Requires `llm` dependencies and a model. |
| `georeset-wiki-landcover-classify-articles` | `cli.data.classify_articles` | Run CORINE/OSM article text classification with resumable checkpoints. Requires text inputs, geospatial inputs, and usually a model. |
| `georeset-wiki-landcover-compute-corine-spatial-confidence` | `cli.data.compute_corine_spatial_confidence` | Compute CORINE buffer-purity diagnostics for article labels. |
| `georeset-wiki-landcover-summarize-classification-experiment` | `cli.analysis.summarize_classification_experiment` | Build overview tables for a classification experiment directory. |
| `georeset-wiki-landcover-evaluate-spatial-confidence` | `cli.analysis.evaluate_predictions_with_spatial_confidence` | Reevaluate frozen predictions on spatial-confidence subsets. |
| `georeset-wiki-landcover-evaluate-relevance-stratified` | `cli.analysis.evaluate_relevance_stratified_predictions` | Reevaluate frozen predictions by relevance metadata. |
| `georeset-wiki-landcover-fetch-wikipedia-article-types` | `cli.data.fetch_wikipedia_article_types` | Fetch/cache Wikipedia article type metadata. |
| `georeset-wiki-landcover-evaluate-article-type-relevance-stratified` | `cli.analysis.evaluate_article_type_relevance_stratified` | Reevaluate predictions by article type/relevance metadata. |
| `georeset-wiki-landcover-build-evidence-cards` | `cli.data.build_evidence_cards` | Build article evidence cards from content, summaries, and quality metadata. |
| `georeset-wiki-landcover-evaluate-evidence-card-experiment` | `cli.analysis.evaluate_evidence_card_experiment` | Analyze evidence-card experiment outputs. |
| `georeset-wiki-landcover-build-evidence-highlights` | `cli.data.build_evidence_highlights` | Build content with evidence highlights. |
| `georeset-wiki-landcover-evaluate-evidence-highlights-experiment` | `cli.analysis.evaluate_evidence_highlights_experiment` | Analyze evidence-highlight experiment outputs. |
| `georeset-wiki-landcover-build-retrieved-evidence-windows` | `cli.data.build_retrieved_evidence_windows` | Build retrieved evidence windows and related text variants. |
| `georeset-wiki-landcover-evaluate-retrieved-evidence-windows-experiment` | `cli.analysis.evaluate_retrieved_evidence_windows_experiment` | Analyze retrieved evidence window runs. |
| `georeset-wiki-landcover-evaluate-subset-randomization-controls` | `cli.analysis.evaluate_subset_randomization_controls` | Analyze subset randomization control artifacts. |
| `georeset-wiki-landcover-build-clip-label-splits` | `cli.data.build_clip_label_splits` | Build CLIP weak-label split artifacts. |
| `georeset-wiki-landcover-fetch-sentinel-patches` | `cli.data.fetch_sentinel_patches` | Fetch Sentinel image patches. Requires vision dependencies and external imagery access. |
| `georeset-wiki-landcover-fetch-sentinel-multiscale-patches` | `cli.data.fetch_sentinel_multiscale_patches` | Fetch multi-scale Sentinel patches. Requires vision dependencies and external imagery access. |
| `georeset-wiki-landcover-build-image-probe-splits-v2` | `cli.data.build_image_probe_splits_v2` | Build image-probe split artifacts. |
| `georeset-wiki-landcover-embed-clip-patches` | `cli.data.embed_clip_patches` | Embed patch caches with CLIP. Requires vision dependencies and compute device. |
| `georeset-wiki-landcover-embed-image-patches` | `cli.data.embed_image_patches` | Embed patch caches with a configured encoder. |
| `georeset-wiki-landcover-run-clip-linear-probe-experiment` | `cli.analysis.run_clip_linear_probe_experiment` | Train/evaluate CLIP linear probes from split and embedding caches. |
| `georeset-wiki-landcover-run-clip-zero-shot-experiment` | `cli.analysis.run_clip_zero_shot_experiment` | Run CLIP zero-shot evaluation from split and embedding caches. |
| `georeset-wiki-landcover-run-quality-weighted-image-probe` | `cli.analysis.run_quality_weighted_image_probe` | Run quality-weighted image probe experiments. |
| `georeset-wiki-landcover-run-quality-weighted-image-zero-shot` | `cli.analysis.run_quality_weighted_image_zero_shot` | Run quality-weighted image zero-shot experiments. |
| `georeset-wiki-landcover-evaluate-image-probe-training-policy-controls` | `cli.analysis.evaluate_image_probe_training_policy_controls` | Evaluate image probe training policy controls. |

Use `PYTHONDONTWRITEBYTECODE=1 uv run <command> --help` to inspect the current
flags. The help output is generated from the active code, so prefer it over
copying flag lists into experiment logs by hand.

## Classification CLI Essentials

Core flags for `georeset-wiki-landcover-classify-articles`:

- `--task`: `corine_level2` or `osm`.
- `--text-source`: one of the text sources in
  `classification/text_sources.py`, including supported shuffled controls.
- input path flags for wiki articles, contents, summaries, evidence artifacts,
  OSM polygons, and CORINE polygons.
- `--output-dir`: run directory for predictions and metrics.
- `--model-path` and optional `--model-repo-id`.
- `--seed`, `--temperature`, `--limit`, and `--retry-failed`.

Outputs:

```text
<output-dir>/<task>_<text_source>_predictions.json
<output-dir>/<task>_<text_source>_metrics.json
```

## Repository Scripts

- `scripts/reproduce_small.py`: synthetic reproducibility package generator.
- `scripts/validate_artifacts.py`: small/full artifact validator.
- `scripts/data/*.py`, `scripts/analysis/*.py`, and `scripts/dev/*.py`: thin
  wrappers around packaged CLI modules.
- `scripts/cluster/*.sh`: Grid5000/Nancy submit, run, and sync scripts.

Cluster scripts are not portable smoke tests. They assume SSH/OAR/Grid5000
access and use environment variables documented in `docs/configuration.md`.

## Commands Requiring Care

- `georeset-wiki-landcover-filter-pipeline` mutates data artifacts unless
  `--dry-run` or `--audit-only` is used.
- Classification and summarization commands can be long-running and write
  resumable JSON checkpoints.
- Vision commands may require `uv sync --group dev --group vision`, CUDA or CPU
  device choices, Sentinel/Planetary Computer access, and local patch caches.
- Grid5000 sync scripts can poll repeatedly unless configured for one-shot sync;
  prefer `SYNC_ONCE=1`.
