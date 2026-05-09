# Source Code Overview

`src/` contains the GeoReset Python package. The package is organized around
small modules for fetching geospatial/text data, analyzing polygon overlap, and
writing visual checks.

## Packages

- `src.fetchers`
  - `data_fetcher.py`: loads CORINE data from `data/corine/`, converts to WGS84,
    maps CORINE codes to labels, and exposes bounds/sampling helpers.
  - `wiki_fetcher.py`: fetches French Wikipedia geosearch metadata for the
    CORINE bounds, with optional CORINE and OSM polygon filters.
  - `wiki_content_fetcher.py`: fetches full Wikipedia article extracts from
    `data/wiki/wiki_articles.json`. Existing output is sanitized before resume;
    duplicate page IDs are skipped; progress is checkpointed after each batch.
  - `osm_fetcher.py`: fetches OSM land-cover polygons from Overpass using the
    project tag allowlist.

- `src.analysis`
  - `corine_polygon_stats.py`: computes CORINE class area/share distributions
    inside OSM polygons.
  - `distribution_summary.py`: summarizes distribution CSV outputs.

- `src.visualization`
  - `map_visualizer.py`: writes Folium maps for CORINE polygons, Wikipedia
    article points, and OSM polygon overlays.

- `src.scripts`
  - `snapshot.py`: prints a quick CORINE dataset snapshot.
  - `run_corine_analysis.py`: runs OSM/CORINE distribution and map generation.
  - `summarize_articles.py`: summarizes fetched article content with an LLM.
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
uv run python -m src.scripts.snapshot
uv run python -m src.fetchers.wiki_fetcher
uv run python -m src.fetchers.wiki_content_fetcher
uv run python -m src.visualization.map_visualizer
uv run python -m src.scripts.run_corine_analysis
uv run python -m src.scripts.summarize_articles
```

Use `PYTHONDONTWRITEBYTECODE=1` while developing if you want to avoid local
`__pycache__` churn.
