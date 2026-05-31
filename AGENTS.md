# AGENTS.md

Guidance for Codex and other LLM coding agents working in this repository.

## Project Purpose

GeoReset Wiki Land-Cover tests whether geolocated French Wikipedia text can
support land-cover classification against CORINE level-2 and project-scoped OSM
labels. The repo contains code, tests, docs, and reproducibility helpers.
Downloaded/generated research data lives outside Git in the Hugging Face bucket.

## First Checks

```bash
git status --short --branch
rg --files | head
sed -n '1,220p' README.md
sed -n '1,220p' docs/architecture.md
sed -n '1,220p' docs/data_flow.md
sed -n '1,220p' docs/cli.md
```

There is no Makefile or justfile. Use `uv`.

## Setup And Verification

```bash
uv sync --frozen --group dev
uv lock --check
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/dev/check_repository_hygiene.py
PYTHONDONTWRITEBYTECODE=1 uv run ruff check .
PYTHONDONTWRITEBYTECODE=1 uv run ruff format --check .
PYTHONDONTWRITEBYTECODE=1 uv run mypy src scripts
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q
```

For a fast no-data smoke path:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/reproduce_small.py \
  --output-dir build/reproducibility/small \
  --clean
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
```

Optional dependency groups:

- `uv sync --group dev --group llm` for llama-cpp/Hugging Face model work.
- `uv sync --group dev --group vision` for Sentinel/CLIP/torch workflows.

## Core Invariants

- Keep `data/`, `build/`, caches, virtualenvs, and local `.env*` files out of
  Git.
- Do not upload, delete, or rewrite Hugging Face bucket data unless explicitly
  asked.
- Preserve public CLI flags, artifact filenames, JSON/CSV schemas, and
  experiment IDs unless the change is intentional, documented, and tested.
- Classification cache validity depends on task, text source, model path, model
  repo ID, seed, temperature, allowed labels, policy version, and
  `metadata.text_sha256`.
- CORINE classification is single-label. OSM classification is multi-label.
- Zero-row scientific states should fail clearly rather than produce plausible
  metrics.
- Use atomic artifact writers from `src/georeset_wiki_landcover/utils/json_io.py`.
- Put reusable Python logic in `src/georeset_wiki_landcover/`; keep top-level
  `scripts/` wrappers thin.

## Do Not Edit Casually

- `data/` and `build/`: generated/synced artifacts, ignored by Git.
- `docs/experiments/`: frozen experiment reports; update only when regenerating
  or correcting a specific report.
- `docs/diagrams/*` generated outputs: update source and rendered files
  together when needed.
- `.github/workflows/ci.yml`, `pyproject.toml`, `uv.lock`, Docker files, and
  cluster scripts: these are workflow contracts; inspect tests before changing.
- `CLASSIFICATION_POLICY_VERSION`: bump only for intentional cache-invalidating
  classification policy changes.

## Style Preferences

- Python 3.10+, typed package, ruff formatting, mypy over `src scripts`.
- Prefer small, focused changes with regression tests around parsers, joins,
  filters, metrics, cache invalidation, and artifact validation.
- Prefer packaged `georeset-wiki-landcover-*` entry points in docs and
  automation. Top-level `scripts/data`, `scripts/analysis`, and `scripts/dev`
  are compatibility wrappers.
- Use tiny synthetic fixtures instead of private or large data.
- Use `rg`/`rg --files` for search.

## Safe Change Process

1. Inspect the relevant module, tests, docs, and artifact assumptions.
2. Run the smallest relevant baseline check before edits.
3. Add or update focused tests for behavior changes.
4. Keep diffs small; avoid speculative rewrites.
5. Run targeted tests, then the relevant quality gate.
6. Report exactly what was verified and what remains unverified.

## What Not To Do

- Do not commit `data/`, `build/`, model files, caches, or secrets.
- Run `scripts/dev/check_repository_hygiene.py` before staging workflow or
  artifact changes.
- Do not silently skip malformed scientific inputs unless existing code already
  documents that policy.
- Do not broaden network, Grid5000, bucket, or model side effects without user
  confirmation.
- Do not rewrite architecture or move artifact locations for cosmetic reasons.
- Do not claim a command works unless you ran it or clearly mark it unverified.
- Do not push or create releases unless the user explicitly asks.

Longer references:

- `docs/agent_playbook.md`
- `docs/repo_map.md`
- `docs/common_failure_modes.md`
- `docs/reproducibility.md`
- `docs/artifacts.md`
- `docs/troubleshooting.md`
