# GeoReset

GeoReset experiments with whether text descriptions near land-cover polygons can
help an LLM infer the correct CORINE land-cover class.

- Project site: https://geo-reset.sylvainlobry.com/
- Code repository: https://github.com/NoeFlandre/georeset
- Data bucket: https://huggingface.co/buckets/NoeFlandre/georeset

## Code And Data Split

GitHub stores source code, tests, Docker configuration, and documentation.
Generated and downloaded project data lives in the Hugging Face bucket. Keep
`data/` out of Git.

Download or refresh data from Hugging Face:

```bash
hf sync hf://buckets/NoeFlandre/georeset ./data
```

Upload local data changes back to Hugging Face:

```bash
hf sync ./data hf://buckets/NoeFlandre/georeset --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```

Before committing code, check that no data files are staged:

```bash
git status --short
git ls-files data build
```

`git ls-files data build` should print nothing. If data was staged by mistake:

```bash
git rm -r --cached data build
git add .gitignore README.md src tests pyproject.toml uv.lock LICENSE Dockerfile .dockerignore
```

## Repository Layout

- `src/fetchers/data_fetcher.py`: loads CORINE shapefiles and exposes bounds,
  class labels, centroids, and samples.
- `src/fetchers/wiki_fetcher.py`: fetches French Wikipedia geosearch metadata
  inside the CORINE bounds and project polygon filters.
- `src/fetchers/wiki_content_fetcher.py`: fetches full Wikipedia extracts from
  page IDs. It sanitizes existing output, skips already-fetched entries, writes
  checkpoints after each batch, and can be stopped/resumed at any time.
- `src/fetchers/osm_fetcher.py`: fetches project-relevant OSM land-cover
  polygons from Overpass.
- `src/analysis/corine_polygon_stats.py`: computes CORINE class area/share
  distributions inside OSM polygons.
- `src/analysis/distribution_summary.py`: summarizes distribution outputs.
- `src/visualization/map_visualizer.py`: writes Folium map visualizations.
- `src/scripts/run_corine_analysis.py`: runs the OSM/CORINE distribution and
  map generation workflow.
- `src/scripts/snapshot.py`: prints a quick CORINE dataset snapshot.
- `src/scripts/summarize_articles.py`: summarizes fetched article content with
  a local LLM backend.
- `src/classification/`: label utilities, ground-truth builders, LLM
  classifier, and metrics for CORINE level-2 and OSM tag classification.
- `scripts/grid5000/submit_summarization.sh`: syncs the minimal repository
  state to Grid5000/Nancy, submits the summarization OAR job, and continuously
  syncs the resumable output back.

## Data Artifacts

These files are expected under `data/` after syncing the bucket:

- `data/corine/`: CORINE shapefile and bounds.
- `data/wiki/wiki_articles.json`: Wikipedia geosearch metadata.
- `data/wiki/article_contents.json`: resumable Wikipedia article content.
- `data/osm/osm_project_polygons.geojson`: project-relevant OSM polygons.
- `data/distribution/osm_corine_distribution.csv`: CORINE class area/share
  distribution inside OSM polygons.
- `data/maps/`: generated HTML visualizations.

The source CORINE data was downloaded from:
https://www.datagrandest.fr/geonetwork/srv/api/records/c0ccbf45-2620-4bde-93f8-869558e51d7e?language=fre

## Local Setup

Install dependencies with `uv`:

```bash
uv sync --all-groups
hf sync hf://buckets/NoeFlandre/georeset ./data
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest
```

Run only the resumable Wikipedia content fetcher tests:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/fetchers/test_wiki_content_fetcher.py -q
```

## Pipeline Commands

Print a quick dataset snapshot:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.scripts.snapshot
```

Fetch Wikipedia article metadata:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.fetchers.wiki_fetcher
```

Fetch full Wikipedia article content. This command is resumable: stop it with
`Ctrl-C`, then run it again and it will skip sane entries already saved in
`data/wiki/article_contents.json`.

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.fetchers.wiki_content_fetcher
hf sync ./data hf://buckets/NoeFlandre/georeset --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```

Regenerate the CORINE + Wikipedia article map:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.visualization.map_visualizer
```

Fetch/use OSM polygons, compute CORINE distributions, and generate the separate
CORINE + OSM map:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m src.scripts.run_corine_analysis
hf sync ./data hf://buckets/NoeFlandre/georeset --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```

## Grid5000 Article Summarization

The summarization path is prepared so a coding agent only needs to run one
local command:

```bash
bash scripts/grid5000/submit_summarization.sh
```

That script syncs the code and `data/wiki/article_contents.json` to Nancy,
submits `scripts/grid5000/run_summarization_job.sh` through OAR, and keeps
pulling `data/wiki/article_summaries.json` back locally every 30 seconds. The
remote job uses CUDA llama-cpp-python, installs `uv` if needed, and runs:

```bash
uv run python -m src.scripts.summarize_articles \
  --input-path data/wiki/article_contents.json \
  --output-path data/wiki/article_summaries.json
```

Optional environment overrides:

```bash
G5K_SITE=nancy G5K_REMOTE_DIR=georeset GEORESET_MODEL_PATH=Qwen3.6-27B-Q4_0.gguf \
  bash scripts/grid5000/submit_summarization.sh
```

## Article-Text Land-Cover Classification

Six classification runs are supported: 2 tasks (CORINE level-2 single-label, OSM multi-label) × 3 text sources (normal summary, no-place summary, raw article content).

Local runs:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m scripts.data.classify_articles \
  --task corine_level2 --text-source summary
PYTHONDONTWRITEBYTECODE=1 uv run python -m scripts.data.classify_articles \
  --task corine_level2 --text-source summary_no_place
PYTHONDONTWRITEBYTECODE=1 uv run python -m scripts.data.classify_articles \
  --task corine_level2 --text-source content
PYTHONDONTWRITEBYTECODE=1 uv run python -m scripts.data.classify_articles \
  --task osm --text-source summary
PYTHONDONTWRITEBYTECODE=1 uv run python -m scripts.data.classify_articles \
  --task osm --text-source summary_no_place
PYTHONDONTWRITEBYTECODE=1 uv run python -m scripts.data.classify_articles \
  --task osm --text-source content
```

Outputs:
- `data/classification/{task}_{text_source}_predictions.json`: per-article predictions with raw LLM response and full metadata including fingerprint.
- `data/classification/{task}_{text_source}_metrics.json`: aggregate metrics (n_eligible, n_predicted_ok, n_parse_error, coverage, accuracy/F1 scores, task, text_source, allowed_labels, labels_evaluated).

Resumability: articles with matching fingerprint and `parse_status=="ok"` are skipped; parse errors are re-run. Use `--limit N` for smoke testing.

Grid5000 runs:

```bash
GEORESET_CLASSIFICATION_TASK=corine_level2 \
GEORESET_CLASSIFICATION_TEXT_SOURCE=summary \
bash scripts/cluster/submit_classification.sh
```

## Docker

The Docker image contains code and Python dependencies only. It intentionally
does not bake in `data/`; mount local synced data at `/app/data`.

Build the image:

```bash
docker build -t georeset .
```

Run a quick container smoke test:

```bash
docker run --rm georeset
```

Run tests in Docker:

```bash
docker run --rm -v "$PWD/data:/app/data" georeset uv run pytest tests/fetchers/test_wiki_content_fetcher.py -q
```

Run a pipeline command in Docker:

```bash
hf sync hf://buckets/NoeFlandre/georeset ./data
docker run --rm -v "$PWD/data:/app/data" georeset uv run python -m src.fetchers.wiki_content_fetcher
hf sync ./data hf://buckets/NoeFlandre/georeset --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```

## OSM Scope

OSM fetching is intentionally restricted to project-relevant land-cover tags.
It excludes dense or unrelated tags such as buildings, amenities, commercial,
industrial, residential, and leisure features.

Included `landuse` values:

```text
farmland, farmyard, meadow, orchard, vineyard, forest, allotments,
plant_nursery, greenhouse_horticulture, grass
```

Included `natural` values:

```text
wood, scrub, grassland, wetland, heath, water, bare_rock, sand, scree,
shingle, beach, mud
```

## Publishing Workflow

Use this split every time:

1. Code/docs/tests go to GitHub.
2. Generated/downloaded artifacts go to the Hugging Face bucket.
3. Do not commit `data/`, `build/`, caches, or local environment files.

Code push:

```bash
git status --short
git add .gitignore README.md Dockerfile .dockerignore src tests pyproject.toml uv.lock LICENSE
git commit -m "Describe code change"
git push origin main
```

Data push:

```bash
hf sync ./data hf://buckets/NoeFlandre/georeset --delete --exclude '**/.DS_Store' --exclude '.DS_Store'
```
