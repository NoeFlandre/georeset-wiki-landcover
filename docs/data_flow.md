# Data Flow

This document describes the current data flow and artifact names used by the
code. See `docs/artifacts.md` for file-level validation rules.

## Core Inputs

Full project runs expect synced data under `data/`:

- `data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp` plus
  shapefile sidecars and `data/corine/bounds.json`.
- `data/wiki/wiki_articles.json`, a list of Wikipedia geosearch metadata with
  `pageid`, `lat`, and `lon`.
- `data/wiki/article_contents.json`, keyed by page ID.
- `data/osm/osm_project_polygons.geojson`, with `osm_id` and project-relevant
  land-cover tags.

The small reproduction path creates synthetic equivalents under
`build/reproducibility/small/data/` and does not require bucket data.

## Acquisition And Filtering Flow

```text
CORINE shapefile + bounds
  -> DataFetcher
  -> non-artificial CORINE filter
  -> OSM fetch/filter
  -> wiki metadata fetch/filter
  -> article content fetch
  -> summaries/evidence text variants
```

Important commands:

- `georeset-wiki-landcover-run-corine-analysis` reads CORINE/OSM inputs,
  computes `data/distribution/osm_corine_distribution.csv`, and writes maps.
- `georeset-wiki-landcover-filter-pipeline` removes artificial-surface cascade
  artifacts and prunes wiki content/summary JSONs to retained page IDs.
- `georeset-wiki-landcover-summarize-articles` writes place or no-place
  summaries.
- `georeset-wiki-landcover-summarize-landuse-evidence` writes
  `data/wiki/article_landuse_evidence_summaries.json`.

Remote Wikipedia, Overpass, Planetary Computer, Hugging Face, and Grid5000
dependencies are not reproducible from Git alone. Use synced bucket artifacts or
the synthetic smoke path for deterministic local checks.

## Classification Flow

```text
wiki_articles.json
  + selected text source JSON
  + CORINE polygons or OSM polygons
  -> build ground truth by pageid
  -> eligible pageids = text pageids intersect target pageids
  -> optional deterministic shuffled text reassignment
  -> LLM classifier
  -> per-pageid prediction checkpoint JSON
  -> metrics JSON
```

Output directory default:

```text
data/classification/runs/default/
  corine_level2_<text_source>_predictions.json
  corine_level2_<text_source>_metrics.json
  osm_<text_source>_predictions.json
  osm_<text_source>_metrics.json
```

Supported tasks:

- `corine_level2`: single-label target from CORINE `code_18[:2]`, excluding
  artificial classes for classification ground truth.
- `osm`: multi-label target from project-scoped `landuse` and `natural` tags.

Supported text sources are declared in
`classification/text_sources.py`. The current list includes summaries,
no-place summaries, content, land-use evidence summaries, evidence cards,
evidence-highlighted content, retrieved evidence windows, and deterministic
shuffled controls for selected sources.

## Analysis Flow

Frozen experiment inputs live under `data/experiments/`. `experiment_paths.py`
maps known experiment IDs to numbered experiment groups such as
`001_qwen_e2e_shuffled_control` and
`014_quality_weighted_multiscale_image_probe`.

Analysis commands read frozen prediction artifacts and write derived tables:

- `georeset-wiki-landcover-summarize-classification-experiment` writes
  classification overview tables for one experiment directory.
- `georeset-wiki-landcover-evaluate-spatial-confidence` joins frozen
  predictions to `spatial_confidence.csv` and writes subset metrics.
- Relevance, article-type, evidence-card, evidence-highlight, retrieved-window,
  supervision-quality, subset-randomization, and vision analysis commands write
  CSV/Markdown summaries and manifests under their output directories.

Most analysis commands are intended to be no-LLM reruns: they evaluate frozen
prediction artifacts rather than generating new model outputs.

## Vision Flow

```text
wiki/articles + image/split metadata
  -> Sentinel patch fetchers
  -> patch caches
  -> CLIP/image embedding caches (.npz)
  -> zero-shot or linear-probe experiments
  -> CSV/Markdown/manifest outputs
```

The shared embedding cache loader expects NPZ arrays named `pageids` and
`embeddings` with the same row count. Probe code can either fail on missing
embeddings or explicitly allow missing rows where a command supports that path.

## Cache And Staleness Rules

- Article content fetching is resumable and skips sane entries already written
  to `data/wiki/article_contents.json`.
- Classification checkpoints are resumable. A record is skipped only when
  `parse_status == "ok"`, the classification fingerprint matches, and
  `metadata.text_sha256` matches the current input text.
- `--retry-failed` keeps matching OK records and retries non-OK records.
- Small reproducibility outputs have manifest SHA-256 hashes. Re-run
  `scripts/validate_artifacts.py --profile small` to detect stale files.
- Full artifact validation is intentionally lightweight and does not compare
  local data to the Hugging Face bucket.

## Known Non-Determinism

- Remote Wikipedia and Overpass results can change.
- LLM outputs can vary by model file, quantization, backend, version, seed, and
  hardware even with low temperature.
- Sentinel imagery, cloud filtering, and Planetary Computer availability are
  external inputs.
- Grid5000 scheduling and GPU allocation are outside the repository.
