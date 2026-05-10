# Source Code Overview

`src/georeset/` contains the GeoReset Python package. The package is organized around
small modules for fetching geospatial/text data, analyzing polygon overlap, and
writing visual checks.

## Packages

- `georeset.fetchers`
  - `data_fetcher.py`: loads CORINE data from `data/corine/`, converts to WGS84,
    maps CORINE codes to labels, and exposes bounds/sampling helpers.
  - `wiki_fetcher.py`: fetches French Wikipedia geosearch metadata for the
    CORINE bounds, with optional CORINE and OSM polygon filters.
  - `wiki_content_fetcher.py`: fetches full Wikipedia article extracts from
    `data/wiki/wiki_articles.json`. Existing output is sanitized before resume;
    duplicate page IDs are skipped; progress is checkpointed after each batch.
  - `osm_fetcher.py`: fetches OSM land-cover polygons from Overpass using the
    project tag allowlist.

- `georeset.analysis`
  - `corine_polygon_stats.py`: computes CORINE class area/share distributions
    inside OSM polygons.
  - `distribution_summary.py`: summarizes distribution CSV outputs.

- `georeset.classification`
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

- `georeset.visualization`
  - `map_visualizer.py`: writes Folium maps for CORINE polygons, Wikipedia
    article points, and OSM polygon overlays.

- top-level `scripts`
  - `scripts/dev/snapshot.py`: prints a quick CORINE dataset snapshot.
  - `scripts/analysis/run_corine_analysis.py`: runs OSM/CORINE distribution and
    map generation.
  - `scripts/data/summarize_articles.py`: summarizes fetched article content with an LLM.
    It supports `--summary-mode place` for normal summaries and
    `--summary-mode no_place` for summaries that suppress the described place
    name. It uses llama-cpp-python schema-constrained JSON generation and
    persists only public summaries, never private thinking fields.

## Data Contract

The package reads and writes local paths under `data/`, but `data/` is not part
of the Git repository. Sync it from the Hugging Face bucket before running
pipeline commands:

```bash
hf sync hf://buckets/NoeFlandre/georeset ./data
```

After generating or fetching data, sync it back:

```bash
hf sync ./data hf://buckets/NoeFlandre/georeset --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```

## Common Entry Points

```bash
uv run python -m scripts.dev.snapshot
uv run python -m georeset.fetchers.wiki_fetcher
uv run python -m georeset.fetchers.wiki_content_fetcher
uv run python -m georeset.visualization.map_visualizer
uv run python -m scripts.analysis.run_corine_analysis
uv run python -m scripts.data.summarize_articles
```

Use `PYTHONDONTWRITEBYTECODE=1` while developing if you want to avoid local
`__pycache__` churn.

## Quality Checks

The test suite enforces focused coverage for `georeset.classification` with
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
