# Reproducibility

This guide gives two paths:

- Small smoke reproduction: fully synthetic, fast, and does not require private
  or large data, a model file, GPU access, Hugging Face authentication, or
  network fetches.
- Full project reproduction: uses synced project artifacts under `data/` and,
  for LLM or vision experiments, external model/GPU resources.

The small path is the minimum meaningful workflow for a new researcher because
it exercises geospatial inputs, article JSON inputs, classification targets,
prediction fingerprints, metrics, artifact manifests, and the artifact
validator.

## Setup From Clean Clone

```bash
git clone https://github.com/NoeFlandre/georeset-wiki-landcover.git
cd georeset-wiki-landcover
uv sync --group dev
```

Use Python 3.10 through `uv`; this is what CI installs.

Optional groups:

```bash
uv sync --group dev --group llm
uv sync --group dev --group vision
```

The `llm` group is needed for llama-cpp based summarization and classification.
The `vision` group is needed for Sentinel/CLIP workflows.

## Required Data Inputs

For the small smoke path: none. The script creates synthetic inputs under its
output directory.

For the full project path, sync the project bucket:

```bash
hf sync hf://buckets/NoeFlandre/georeset-wiki-landcover ./data
```

Expected core inputs after sync are documented in
[`docs/artifacts.md`](artifacts.md). The most important full-pipeline inputs are:

- `data/corine/alsace_corine_land_use_2018/occupation_sol_2018.shp` and sidecar
  files.
- `data/corine/bounds.json`.
- `data/wiki/wiki_articles.json`.
- `data/wiki/article_contents.json`.
- `data/osm/osm_project_polygons.geojson`.

## Optional Data Inputs

These are only needed for specific downstream analyses:

- `data/wiki/article_summaries.json`.
- `data/wiki/article_summaries_no_place.json`.
- `data/wiki/article_landuse_evidence_summaries.json`.
- `data/wiki/article_evidence_cards.json`.
- `data/wiki/article_evidence_highlights.json`.
- `data/wiki/article_retrieved_evidence_windows.json`.
- `data/experiments/**` frozen experiment directories.
- Sentinel patch caches and CLIP embedding caches for vision experiments.

## Tiny Smoke Test

Run the synthetic reproduction:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/reproduce_small.py \
  --output-dir build/reproducibility/small \
  --clean
```

Then validate the artifacts:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
```

Expected behavior:

- The first command writes synthetic inputs and deterministic classification
  outputs.
- The second command prints `Artifact validation passed: ...`.
- Outputs are under `build/reproducibility/small/`, which is ignored by Git.

The small workflow intentionally does not call an LLM. It uses a deterministic
local classifier so that the smoke test can run on a clean machine.

## Full Pipeline

Install dependencies and sync data first:

```bash
uv sync --group dev
hf sync hf://buckets/NoeFlandre/georeset-wiki-landcover ./data
```

Run the non-mutating filter-pipeline check:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run georeset-wiki-landcover-filter-pipeline --dry-run
```

Run the full filter pipeline only after confirming inputs are present and the
dry run looks sensible:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run georeset-wiki-landcover-filter-pipeline
```

Run a limited classification smoke on real synced inputs:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run georeset-wiki-landcover-classify-articles \
  --task corine_level2 \
  --text-source summary \
  --limit 5 \
  --output-dir data/classification/runs/local_smoke
```

Full LLM classification runs require a GGUF model path or a configured model
repo, and can be long-running:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run georeset-wiki-landcover-classify-articles \
  --task corine_level2 \
  --text-source summary \
  --output-dir data/classification/runs/<run-name>
```

See the repository `README.md` for the complete matrix of CORINE/OSM tasks,
text sources, shuffled controls, summarization jobs, and spatial-confidence
analyses.

## Validate Outputs

Validate the synthetic package:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
```

Validate presence and basic consistency of synced full inputs:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root data \
  --profile full
```

The full profile is intentionally lightweight: it checks core file existence,
JSON readability, duplicate wiki page IDs, stale article-content keys, and OSM
GeoJSON readability. It does not prove that every historical experiment can be
regenerated bit-for-bit.

## Cached Output Staleness

Classification predictions include a metadata fingerprint built from:

- task;
- text source;
- model path;
- optional model repo ID;
- seed;
- temperature;
- allowed labels;
- classification policy version.

Each prediction also stores `text_sha256`. The classifier skips cached records
only when the fingerprint and text hash still match.

The synthetic smoke package writes `manifest.json` with `artifact_sha256` hashes
for every generated input and output. `scripts/validate_artifacts.py --profile
small` recomputes those hashes. If a file changes after the manifest is written,
validation fails with a message like:

```text
ERROR: stale artifact hash for data/wiki/article_summaries.json: manifest=<old> current=<new>
```

For full project data, stale-cache detection is workflow-specific. Use
classification fingerprints for prediction caches, pipeline dry runs before
mutating shared artifacts, and bucket sync timestamps or checksums when
comparing local `data/` to Hugging Face.

## Hardware And Runtime

Observed locally in this checkout:

- Synthetic smoke reproduction: seconds on a laptop-class CPU.
- Artifact validation for the synthetic package: seconds.
- Full test suite: about 2 to 3 minutes locally.

Unknown or environment-dependent:

- Full filter pipeline runtime depends on local `data/` size and geospatial
  joins.
- LLM summarization and classification depend on model size, quantization,
  backend, CPU/GPU availability, and article count.
- Grid5000 jobs are configured for GPU execution and can request long wall times.
- Vision workflows depend on Sentinel/Planetary Computer access, cached patch
  sizes, embedding model size, and hardware.

## Known Non-Reproducible Components

- Remote Wikipedia and Overpass fetches can change over time. Use bucket-synced
  artifacts or frozen experiment directories when comparing results.
- LLM outputs may vary across model builds, hardware backends, quantization,
  llama-cpp versions, and decoding settings.
- Generated experiment reports may include absolute local paths from the machine
  that produced them.
- Hugging Face bucket contents are external to Git history.
- Grid5000 scheduling and hardware allocation are external to this repository.

## Troubleshooting

- `data/` is missing: run the small smoke path, or sync the Hugging Face bucket
  before full reproduction.
- `hf` is missing: install or authenticate Hugging Face Hub tooling before
  syncing bucket data.
- `validate_artifacts.py --profile full` reports missing files: the local bucket
  sync is incomplete or the command is pointed at the wrong root.
- `validate_artifacts.py --profile small` reports a stale hash: rerun
  `scripts/reproduce_small.py --clean`, or inspect which file changed after
  manifest creation.
- Classification fails to load a model: set
  `GEORESET_WIKI_LANDCOVER_MODEL_PATH` or pass `--model-path`; for local smoke
  without a model, use `scripts/reproduce_small.py`.
- Geospatial reads fail: verify that shapefile sidecar files were synced with
  the `.shp`, or use GeoJSON inputs for small tests.
- A real-data smoke run has zero eligible articles: verify that article
  coordinates overlap the CORINE or OSM inputs and that the selected text source
  has matching page IDs.
