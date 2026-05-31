# Architecture

This document maps the current repository structure to the research workflow.
It is derived from `pyproject.toml`, `src/georeset_wiki_landcover/`, `scripts/`,
and the test suite.

## Package Boundary

The installable package is `src/georeset_wiki_landcover/`. The wheel packaging
configuration in `pyproject.toml` includes only this package tree. Top-level
`scripts/` files are repository helpers and compatibility wrappers; reusable
logic should live under `src/georeset_wiki_landcover/`.

## Main Modules

| Path | Role |
| --- | --- |
| `config.py` | Frozen default data paths and model settings. `ModelSettings.from_env()` reads model path/repo overrides. |
| `contracts.py` | Shared typed contracts for article metadata, labels, predictions, and metric result shapes. |
| `experiment_paths.py` | Canonical mapping from experiment IDs to `data/experiments/<group>/<artifact-id>/`. |
| `fetchers/` | Data acquisition helpers for CORINE, Wikipedia metadata/content, OSM, article-type metadata, and LLM summaries. |
| `classification/` | Ground-truth builders, label policy, parser, LLM classifier adapter, resumable runner, cache fingerprints, and metrics. |
| `text/` | Evidence-card, evidence-highlight, retrieved-window, title-scrubbing, and text record helpers. |
| `spatial/` | CORINE spatial-confidence diagnostics and point/polygon join policy. |
| `analysis/` | Frozen prediction loaders, subset metric helpers, label-universe helpers, quality/spatial metadata loading, and distribution summaries. |
| `vision/` | Sentinel patch extraction, image encoder registry, CLIP/image embedding caches, weak labels, zero-shot, and linear-probe utilities. |
| `visualization/` | Folium map generation. |
| `cli/` | Packaged command implementations exposed through `[project.scripts]`. |
| `utils/` | Atomic JSON/CSV/GeoJSON/HTML/parquet/NPZ writers, page-ID indexing, bool parsing, and safe math helpers. |

## Workflow Layers

1. Data acquisition and filtering:
   `fetchers/`, `georeset-wiki-landcover-run-corine-analysis`,
   `georeset-wiki-landcover-filter-pipeline`.
2. Text preparation:
   `georeset-wiki-landcover-summarize-articles`,
   `georeset-wiki-landcover-summarize-landuse-evidence`,
   evidence-card/highlight/window builders.
3. Classification:
   `georeset-wiki-landcover-classify-articles` calls
   `classification.runner.main()`, which builds task ground truth, loads a text
   source, resumes cached predictions where fingerprints match, validates model
   outputs, writes per-record checkpoints, then writes metrics.
4. Analysis:
   `cli/analysis/*` commands read frozen prediction artifacts and write CSV,
   Markdown, JSON manifests, and summaries under `data/experiments/`.
5. Vision experiments:
   Sentinel patch fetchers and embedding commands write patch/embedding caches,
   then probe commands consume those caches.
6. Reproducibility:
   `scripts/reproduce_small.py` creates a synthetic, no-network, no-LLM package;
   `scripts/validate_artifacts.py` validates small or full artifact roots.

## Guardrails In Code

- Atomic writers in `utils/json_io.py` write temp files in the target directory
  before `os.replace`.
- Classification cache keys include task, text source, model path, model repo
  ID, seed, temperature, allowed labels, and `CLASSIFICATION_POLICY_VERSION`.
- Each classification record stores `metadata.text_sha256`; cached records are
  skipped only when the text hash still matches.
- Classification now fails early for zero eligible rows and empty label
  universes.
- Classifier outputs are validated before checkpointing: `parse_status`,
  prediction labels, single-label shape for CORINE, allowed-label membership,
  and metadata object shape.
- `scripts/validate_artifacts.py` checks required files, malformed JSON,
  duplicate wiki page IDs, stale synthetic manifest hashes, malformed wiki rows,
  CORINE bounds shape, prediction record schema, row-count invariants, and OSM
  GeoJSON readability for the supported profiles.

## Tests And CI

CI in `.github/workflows/ci.yml` installs Python 3.10 with `uv`, runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run pytest -q
```

The local pytest configuration measures coverage for
`georeset_wiki_landcover.classification` and requires at least 95%.

## Extension Rules

- Put shared behavior under `src/georeset_wiki_landcover/`; keep top-level
  scripts thin.
- Use the existing typed contracts before adding a new loosely typed record
  shape.
- Use `utils/json_io.py` atomic writers for artifacts.
- Add a focused test before changing parser, join, cache, metric, or artifact
  behavior.
- Preserve existing artifact filenames and CLI flags unless a migration is
  explicitly documented and tested.
- For new experiment IDs, update `experiment_paths.py` if the artifact belongs
  in a numbered `data/experiments/<group>/` directory.
