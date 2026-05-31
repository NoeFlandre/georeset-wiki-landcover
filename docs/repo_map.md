# Repo Map

This is a quick navigation map for agents. It complements
`docs/architecture.md`.

## Root

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Concise instructions for coding agents. |
| `README.md` | Human-facing setup, data sync, main workflows, Docker, publishing. |
| `pyproject.toml` | Package metadata, dependency groups, CLI entry points, ruff, mypy, pytest coverage config. |
| `uv.lock` | Locked Python dependency resolution. Update only when dependency changes require it. |
| `.github/workflows/ci.yml` | CI quality gate: ruff, format, mypy, pytest. |
| `Dockerfile`, `.dockerignore` | Dev/test container without bundled `data/`. |

## Package Tree

| Path | What To Look For |
| --- | --- |
| `src/georeset_wiki_landcover/config.py` | Default data paths and model env overrides. |
| `src/georeset_wiki_landcover/contracts.py` | Shared typed dictionaries and metric contracts. |
| `src/georeset_wiki_landcover/experiment_paths.py` | Experiment ID to artifact directory mapping. |
| `src/georeset_wiki_landcover/fetchers/` | CORINE, Wikipedia, OSM, and LLM summary fetch/build helpers. |
| `src/georeset_wiki_landcover/classification/` | Label policy, parser, LLM classifier, ground truth, metrics, runner, cache behavior. |
| `src/georeset_wiki_landcover/text/` | Evidence cards, highlights, retrieved windows, title scrubbing. |
| `src/georeset_wiki_landcover/spatial/` | Spatial-confidence metrics and point/polygon predicate. |
| `src/georeset_wiki_landcover/analysis/` | Frozen prediction loading, subset metrics, metadata joins, summaries. |
| `src/georeset_wiki_landcover/vision/` | Sentinel patches, embeddings, weak labels, probes, zero-shot utilities. |
| `src/georeset_wiki_landcover/cli/` | Packaged command implementations. |
| `src/georeset_wiki_landcover/utils/` | Atomic writers, page-ID indexing, bool parsing, safe division. |

## Scripts

| Path | Purpose |
| --- | --- |
| `scripts/reproduce_small.py` | Synthetic no-data/no-LLM reproducibility package. |
| `scripts/validate_artifacts.py` | Small/full artifact validation. |
| `scripts/data/*.py` | Thin wrappers around `georeset_wiki_landcover.cli.data.*`. |
| `scripts/analysis/*.py` | Thin wrappers around `georeset_wiki_landcover.cli.analysis.*`. |
| `scripts/dev/*.py` | Thin wrappers around `georeset_wiki_landcover.cli.dev.*`. |
| `scripts/cluster/*.sh` | Grid5000 submit/run/sync scripts. Require remote access; do not treat as local smoke tests. |

## Tests

| Path | Contracts Covered |
| --- | --- |
| `tests/classification/` | Prediction parsing, label policies, runner, cache records, metrics, ground-truth joins. |
| `tests/fetchers/` | Data fetcher behavior, OSM/Wikipedia boundaries, summary validation. |
| `tests/spatial/` | CORINE spatial-confidence geometry, entropy, invalid geometry handling. |
| `tests/analysis/` | Frozen prediction loading, subset metrics, experiment outputs and joins. |
| `tests/cli/` | CLI helper behavior for data/text builders. |
| `tests/scripts/` | Repository wrapper compatibility and cluster script contracts. |
| `tests/utils/` | Atomic writers, bool/math/article helpers. |
| `tests/vision/` | Sentinel patches, embedding caches, split policies, probes, zero-shot helpers. |
| `tests/test_packaging_smoke.py` | CI, Docker, wrappers, packaging, script contract smoke checks. |

## Documentation

| Path | Use |
| --- | --- |
| `docs/reproducibility.md` | Clean clone, small smoke path, full reproduction caveats. |
| `docs/artifacts.md` | Artifact layout, schemas, validation profiles, cache behavior. |
| `docs/configuration.md` | Dependency groups and environment variables. |
| `docs/cli.md` | Declared packaged entry points and repository scripts. |
| `docs/data_flow.md` | Workflow inputs and outputs. |
| `docs/troubleshooting.md` | Human-facing recovery guide. |
| `docs/common_failure_modes.md` | Agent-facing failure mode checklist. |
| `docs/agent_playbook.md` | Longer agent workflow guidance. |
| `docs/experiments/` | Frozen experiment reports and generated analysis pages. |

## Ignored Or External Data

- `data/`: full project artifacts synced through Hugging Face.
- `build/`: local synthetic reproduction outputs and build artifacts.
- `.venv/`, caches, coverage output, local `.env*`: local-only.

Do not add ignored artifact paths to Git without an explicit reason.
