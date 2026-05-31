# Artifacts

Generated and downloaded artifacts live under `data/` and are not committed to
Git. Reproducibility smoke outputs live under `build/reproducibility/` and are
also ignored by Git.

## Expected Directory Structure

After syncing the project bucket, the core directory layout is:

```text
data/
  corine/
    alsace_corine_land_use_2018/
      occupation_sol_2018.shp
      occupation_sol_2018.dbf
      occupation_sol_2018.prj
      occupation_sol_2018.shx
    bounds.json
  wiki/
    wiki_articles.json
    article_contents.json
    article_summaries.json
    article_summaries_no_place.json
    article_landuse_evidence_summaries.json
    article_evidence_cards.json
    article_evidence_highlights.json
    article_retrieved_evidence_windows.json
  osm/
    osm_project_polygons.geojson
  distribution/
    osm_corine_distribution.csv
  maps/
    corine_with_articles.html
    osm_corine_polygons.html
  classification/
    runs/
      <run-name>/
        <task>_<text_source>_predictions.json
        <task>_<text_source>_metrics.json
  experiments/
    <experiment-id>/
      ...
```

The synthetic reproducibility package uses a nested version of the same shape:

```text
build/reproducibility/small/
  manifest.json
  data/
    corine/synthetic_corine.geojson
    wiki/wiki_articles.json
    wiki/article_contents.json
    wiki/article_summaries.json
    wiki/article_summaries_no_place.json
    osm/osm_project_polygons.geojson
    classification/runs/small/
      corine_level2_summary_predictions.json
      corine_level2_summary_metrics.json
      osm_summary_predictions.json
      osm_summary_metrics.json
```

## Core Input Artifacts

- CORINE polygons: land-cover polygons with required `code_18` values. The full
  default path is
  `data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp`.
- CORINE bounds: `data/corine/bounds.json` with `min_lon`, `min_lat`,
  `max_lon`, and `max_lat`.
- Wikipedia metadata: `data/wiki/wiki_articles.json`, a JSON list with unique
  `pageid` values and article coordinates.
- Wikipedia contents: `data/wiki/article_contents.json`, keyed by page ID.
- OSM polygons: `data/osm/osm_project_polygons.geojson`, with project-relevant
  land-cover tags.

## Derived Artifacts

- Article summaries: resumable JSON objects under `data/wiki/`.
- Filtered distribution: `data/distribution/osm_corine_distribution.csv`.
- Maps: HTML outputs under `data/maps/`.
- Classification predictions: JSON objects keyed by page ID, preserving target,
  prediction, parse status, raw response, errors, and metadata.
- Classification metrics: aggregate JSON metrics for each task/text-source run.
- Experiment manifests: many analysis commands write `manifest.json`,
  `run_manifest.json`, or control manifests in their output directories.

## Prediction Metadata

Classification prediction records are expected to include:

- `pageid`;
- `title`;
- `target`;
- `prediction`;
- `prediction_labels`;
- `parse_status`;
- `raw_response`;
- `error`;
- `metadata`.

Important metadata keys include `fingerprint`, `text_sha256`, `model`,
`model_repo_id`, `seed`, `temperature`, `task`, `text_source`, and
`allowed_labels`. These fields support cache invalidation and auditability.

## Synthetic Manifest

`scripts/reproduce_small.py` writes `manifest.json` with:

- workflow name and mode;
- Python and package version;
- classification policy version;
- synthetic model name, seed, and temperature;
- input and output file lists;
- expected row counts;
- `artifact_sha256` hashes for generated inputs and outputs;
- known non-reproducible components for this package.

The small package intentionally has an empty `known_non_reproducible_components`
list because it uses synthetic data and a deterministic local classifier.

## Validation Profiles

Small profile:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
```

Checks:

- required files exist and are non-empty;
- JSON files parse;
- wiki page IDs are unique;
- content and summary keys match wiki page IDs;
- CORINE and OSM vector files are readable and have required columns;
- prediction records have required fields;
- metrics row counts match prediction records;
- manifest hashes match current artifact contents.

Full profile:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root data \
  --profile full
```

Checks:

- core synced input files exist and are non-empty;
- JSON files parse;
- CORINE bounds contain numeric `min_lon`, `min_lat`, `max_lon`, and `max_lat`
  values with minimums not exceeding maximums;
- wiki page IDs are unique;
- wiki rows include a page ID and numeric `lat`/`lon` coordinates;
- article content keys do not contain page IDs absent from wiki metadata;
- OSM GeoJSON is readable and has `osm_id`.

The full profile is a lightweight preflight check, not a complete scientific
reproduction audit.
