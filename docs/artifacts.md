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

## Evidence Card Version 2

New evidence cards use `metadata.version = 2`. Existing version 1 artifacts and
frozen experiment reports remain historical outputs; do not regenerate them as
part of this change. When intentionally building new cards, use a separate
`--output-path` to retain the earlier artifacts. Newly generated evidence-card
analysis manifests also report the current generator version as
`deterministic_card_version`; this field does not infer the version of frozen
prediction inputs or migrate them.

- Spatial boolean display uses the shared `parse_boolish` contract. Numeric
  `1.0`/`0.0` and strings `on`/`off`/`t`/`f` now render as `oui`/`non` instead of
  `inconnue`. Known strings are case-insensitive and trimmed. Missing and unknown
  values still render as `inconnue`; decimal strings `"1.0"`/`"0.0"` remain unknown.
  Raw spatial metadata is preserved; only its French display is interpreted.
- Evidence sentences and summaries use the same title-removal helper as
  highlights and retrieved windows. Exact and separator-varied matches remain
  case-insensitive and become `ce lieu`. Empty/whitespace-only titles skip
  replacement; punctuation-only titles replace literal matches only. All paths
  now collapse whitespace and strip the result, including empty and
  punctuation-only titles (which previously preserved internal whitespace in
  cards). Card headings, line breaks, bullets, and the appended article content
  retain their existing formatting rules.

These intentional changes can alter newly generated card text, character counts,
and text hashes. Classification already checks `metadata.text_sha256` when
reusing predictions, so changed text invalidates its corresponding cached
prediction without changing the classification policy version. The shared
helpers' behavior, highlight/window versions, CLI flags, filenames, schemas,
and experiment IDs are unchanged.

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

## Cache Behavior

- `data/wiki/article_contents.json` is a resumable content cache. The content
  fetcher sanitizes existing entries and skips entries that are already sane.
- Classification prediction files are resumable checkpoints. A page ID is
  skipped only when the existing record has `parse_status == "ok"`, the
  classification fingerprint still matches the current task/text/model/seed/
  temperature/label policy, and `metadata.text_sha256` still matches the exact
  input text.
- Non-OK classification records are retried on later runs. `--retry-failed`
  keeps matching OK records while retrying failures.
- `build/reproducibility/small/manifest.json` records SHA-256 hashes for the
  generated synthetic package. Small-profile validation recomputes those hashes.
- Vision embedding caches are NPZ files expected to contain `pageids` and
  `embeddings` arrays with the same row count.

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
- the synthetic profile matches the deterministic source coordinates, polygon
  geometries, CORINE/OSM labels, and summary text;
- the small synthetic manifest matches the expected workflow, input and output
  inventories, model settings, and empty non-reproducible-components list;
- synthetic prediction records match the complete deterministic page ID, title,
  target, prediction, response, error, parse status, and metadata fields;
- synthetic prediction metadata matches the deterministic model settings,
  allowed labels, run fingerprint, and source-text hash;
- metrics row counts match prediction records;
- the synthetic profile matches its fixed target/prediction pairs and complete
  deterministic metric objects;
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
