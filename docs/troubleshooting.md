# Troubleshooting

This page lists failure modes that are visible in the current code and tests.
It focuses on errors that can otherwise produce confusing or scientifically
invalid outputs.

## Setup And Commands

### `uv` command not found

Install `uv` first, then run:

```bash
uv sync --group dev
```

Optional groups:

- `--group llm` for llama-cpp/Hugging Face model workflows.
- `--group vision` for Sentinel, CLIP, torch, rasterio, and related workflows.

### Entry point not found

Run commands through the project environment:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run georeset-wiki-landcover-classify-articles --help
```

If the entry point is still missing, rerun `uv sync --group dev`.

## Missing Or Incomplete Data

### `data/` is missing

The Git repository does not include generated/downloaded data. Either run the
synthetic smoke path:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/reproduce_small.py \
  --output-dir build/reproducibility/small \
  --clean
```

or sync the project bucket:

```bash
hf sync hf://buckets/NoeFlandre/georeset-wiki-landcover ./data
```

### `validate_artifacts.py --profile full` reports missing files

Check that `--root` points at `data`, not the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root data \
  --profile full
```

Missing `occupation_sol_2018.*` sidecars, `bounds.json`,
`wiki_articles.json`, `article_contents.json`, or
`osm_project_polygons.geojson` usually means the bucket sync is incomplete.

### Shapefile read failures

CORINE full inputs are a shapefile bundle, not just a `.shp`. Keep `.dbf`,
`.prj`, `.shx`, and any other sidecars with the shapefile. For tiny tests,
prefer GeoJSON fixtures.

## Classification Failures

### Zero eligible records

The classifier now fails before model construction or metrics writing when the
selected text records do not overlap task targets. Check:

- selected `--text-source`;
- matching page IDs across wiki metadata and text-source JSON;
- article coordinates;
- CORINE/OSM input paths and CRS;
- `--limit` if it is set.

### Empty label universe

Metrics fail when no true or predicted labels are available. This usually means
all records were filtered out, targets are empty, or an analysis subset has no
usable label data.

### Unknown or malformed model output

Classifier output validation rejects:

- unknown `parse_status` values;
- non-string `prediction_labels`;
- `parse_status="ok"` without a prediction;
- unknown labels outside the allowed label set;
- multiple/mismatched labels for `corine_level2`;
- non-object metadata.

For LLM-backed runs, inspect the per-pageid `raw_response`, `error`, and
`metadata.attempt_history` in the predictions JSON.

### Cached predictions are not reused

OK records are skipped only when both cache checks match:

- classification fingerprint: task, text source, model path, model repo ID,
  seed, temperature, allowed labels, and classification policy version;
- `metadata.text_sha256`: exact text sent to the classifier.

If text or config changed, the record is recomputed. Use `--retry-failed` to
retry only non-OK records while keeping matching OK records.

## Artifact Validation Failures

### Stale synthetic manifest hash

`scripts/reproduce_small.py` writes SHA-256 hashes in `manifest.json`.
Validation reports stale hashes when files changed after manifest creation.
Rerun with `--clean` or inspect the reported file.

### Duplicate wiki page IDs

Duplicate `pageid` values break page-ID keyed joins and caches. The artifact
validator reports duplicates for both small and full profiles. Fix the source
metadata rather than relying on last-wins dictionary behavior.

### Malformed wiki rows

Full validation requires each wiki row to be a JSON object with `pageid` and
numeric `lat`/`lon`. Non-numeric coordinates are excluded by downstream point
builders and can silently remove articles from joins.

### CORINE bounds rejected

`data/corine/bounds.json` must contain numeric `min_lon`, `min_lat`, `max_lon`,
and `max_lat`, with min values not exceeding max values.

## Geospatial And Metric Issues

### Unexpectedly low spatial-confidence coverage

Check that spatial-confidence `pageid` values are strings or string-coercible,
that the predictions and spatial-confidence CSV use the same page IDs, and that
the expected `point_label_share_*` or `dominant_matches_point_label_*` columns
exist.

### Boundary points behave unexpectedly

The project uses the shared point/polygon predicate from
`spatial/policy.py`. Boundary points are retained during filtering; CORINE
ground truth later excludes page IDs with multiple distinct level-2 labels.

### Bad geometries

Spatial-confidence code repairs CORINE geometries with `shapely.make_valid`.
If a vector file still fails to read, validate it independently with
`geopandas.read_file()` and check CRS metadata.

## Cluster And Remote Runs

### Grid5000 jobs do not sync back

Use one-shot syncs unless you intentionally need polling:

```bash
SYNC_ONCE=1 bash scripts/cluster/sync_classification.sh
SYNC_ONCE=1 bash scripts/cluster/sync_summaries.sh
```

Classification sync requires task/text-source/output-dir environment variables;
see `docs/configuration.md`.

### LLM model cannot be loaded

Set either a local model path or a Hugging Face repo ID depending on the code
path:

```bash
GEORESET_WIKI_LANDCOVER_MODEL_PATH=Qwen3.6-27B-Q4_0.gguf
GEORESET_WIKI_LANDCOVER_MODEL_REPO_ID=<repo-id>
```

Local LLM workflows require `uv sync --group dev --group llm`.

## When In Doubt

Run the deterministic smoke path and validator first. If that passes, isolate
the failing full workflow by validating `data/`, then running the relevant
command with `--limit`, `--dry-run`, or a small fixture where the command
supports it.
