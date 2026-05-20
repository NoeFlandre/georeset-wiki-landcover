# Source Code Overview

`src/georeset_wiki_landcover/` contains the installable GeoReset Wiki Land-Cover Python package. The wheel
packages this tree only; top-level `scripts/` files are repository wrappers, not
the primary package API.

## Packages

- `georeset_wiki_landcover.fetchers`
  - `data_fetcher.py`: loads CORINE data from `data/corine/`, converts to WGS84,
    maps CORINE codes to labels, and exposes bounds/sampling helpers.
  - `wiki_fetcher.py`: fetches French Wikipedia geosearch metadata for the
    CORINE bounds, with optional CORINE and OSM polygon filters.
  - `wiki_content_fetcher.py`: fetches full Wikipedia article extracts from
    `data/wiki/wiki_articles.json`. Existing output is sanitized before resume;
    duplicate page IDs are skipped; progress is checkpointed after each batch.
  - `osm_fetcher.py`: fetches OSM land-cover polygons from Overpass using the
    project tag allowlist.

- `georeset_wiki_landcover.analysis`
  - `corine_polygon_stats.py`: computes CORINE class area/share distributions
    inside OSM polygons.
  - `distribution_summary.py`: summarizes distribution CSV outputs.

- `georeset_wiki_landcover.classification`
  - `labels.py`: CORINE level-2 and OSM label allowlists.
  - `ground_truth.py`: spatial joins that build CORINE single-label and OSM
    multi-label ground truth.
  - `prediction_parser.py`: conservative JSON/text normalization for model
    responses. It catches only JSON decoding failures and otherwise lets
    programmer errors surface.
  - `llm_classifier.py`: llama-cpp boundary for classification. This is the
    module that converts external inference exceptions into structured
    `parse_status="error"` records with metadata.
  - `records.py`: checkpoint record shape and resumability skip policy.
  - `runner.py`: reusable classification orchestration for CLI and tests.
  - `task_setup.py`: task-specific data loading, ground-truth construction, and
    label-description setup.
  - `text_sources.py`: primary and shuffled text-source policy.
  - `types.py`: shared typed contracts for prediction results and records.

- `georeset_wiki_landcover.visualization`
  - `map_visualizer.py`: writes Folium maps for CORINE polygons, Wikipedia
    article points, and OSM polygon overlays.

- `georeset_wiki_landcover.spatial`
  - `corine_confidence.py`: computes CORINE level-2 buffer purity around
    article coordinates in EPSG:2154. It uses the full CORINE dataset, including
    artificial classes, for spatial-confidence diagnostics.

- `georeset_wiki_landcover.utils`
  - `json_io.py`: atomic writers for JSON, text, CSV, GeoJSON, Folium HTML maps,
    and optional parquet outputs. Pipeline artifact writes should go through
    these helpers instead of direct `open(..., "w")`, `to_file`, `save`, or
    `to_parquet` calls.

- `georeset_wiki_landcover.cli`
  - `dev/snapshot.py`: prints a quick CORINE dataset snapshot.
  - `analysis/run_corine_analysis.py`: runs OSM/CORINE distribution and
    map generation.
  - `analysis/summarize_classification_experiment.py`: regenerates overview
    tables and README files for classification experiment folders.
  - `analysis/evaluate_predictions_with_spatial_confidence.py`: reevaluates
    frozen classification predictions on CORINE spatial-confidence subsets.
  - `data/classify_articles.py`: packaged article-text classification CLI.
  - `data/compute_corine_spatial_confidence.py`: packaged CLI for the
    `corine_spatial_confidence_v1` diagnostics.
  - `data/filter_pipeline.py`: filters/refetches project data artifacts and
    regenerates OSM/CORINE distribution outputs.
  - `data/summarize_articles.py`: summarizes fetched article content with an LLM.
    It supports `--summary-mode place` for normal summaries and
    `--summary-mode no_place` for summaries that suppress the described place
    name. It uses llama-cpp-python schema-constrained JSON generation and
    persists only public summaries, never private thinking fields.
  - Top-level `scripts/` modules are thin repository wrappers around these
    packaged CLI modules for backwards-compatible `python -m scripts...` usage.

## Data Contract

The package reads and writes local paths under `data/`, but `data/` is not part
of the Git repository. Sync it from the Hugging Face bucket before running
pipeline commands:

```bash
hf sync hf://buckets/NoeFlandre/georeset-wiki-landcover ./data
```

After generating or fetching data, sync it back:

```bash
hf sync ./data hf://buckets/NoeFlandre/georeset-wiki-landcover --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```

## Common Entry Points

```bash
uv run georeset-wiki-landcover-snapshot
uv run python -m georeset_wiki_landcover.fetchers.wiki_fetcher
uv run python -m georeset_wiki_landcover.fetchers.wiki_content_fetcher
uv run python -m georeset_wiki_landcover.visualization.map_visualizer
uv run georeset-wiki-landcover-run-corine-analysis
uv run georeset-wiki-landcover-summarize-articles
uv run georeset-wiki-landcover-classify-articles --help
uv run georeset-wiki-landcover-compute-corine-spatial-confidence --help
uv run georeset-wiki-landcover-evaluate-spatial-confidence --help
uv run georeset-wiki-landcover-summarize-classification-experiment --help
```

Use `PYTHONDONTWRITEBYTECODE=1` while developing if you want to avoid local
`__pycache__` churn.

## Quality Checks

The test suite enforces focused coverage for `georeset_wiki_landcover.classification` with
`pytest-cov` and a 95% fail-under threshold. Run the full local gate before
committing:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run ruff check .
PYTHONDONTWRITEBYTECODE=1 uv run ruff format --check .
PYTHONDONTWRITEBYTECODE=1 uv run mypy src scripts
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q
```

Exception handling policy for classification code:

- pure helpers catch narrow, expected exceptions only, such as
  `json.JSONDecodeError` in prediction parsing;
- the LLM boundary catches broad inference/runtime exceptions and records them
  as auditable prediction errors;
- task setup and geospatial loaders let IO/data errors fail loudly so bad inputs
  are fixed rather than silently converted into misleading metrics.
