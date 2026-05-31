# Common Failure Modes For Agents

This checklist is tuned for coding agents making changes in this repo. It is
not a replacement for `docs/troubleshooting.md`; it focuses on mistakes agents
can introduce.

## Data And Artifact Mistakes

### Accidentally staging generated data

Risk: commits include bucket data, model outputs, or synthetic build products.

Checks:

```bash
git status --short
git ls-files data build
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/dev/check_repository_hygiene.py
```

Expected: `git ls-files data build` prints nothing and the hygiene check passes.
The hygiene check fails if tracked files live under `data/` or `build/`, if
tracked `.env*` files are present, if tracked cache files are present, or if a
tracked file exceeds the configured size limit.

### Uploading or deleting bucket data unintentionally

Risk: `hf sync --delete` changes shared research artifacts.

Policy: only run upload/delete syncs when the user explicitly asks. For ordinary
code/docs tasks, do not mutate the Hugging Face bucket.

### Editing frozen experiment reports casually

Risk: docs no longer match frozen artifacts.

Policy: treat `docs/experiments/` as report outputs. Update only for a specific
report correction/regeneration task and document what was regenerated.

## Classification And Metrics Mistakes

### Producing metrics over zero valid rows

Risk: plausible-looking metrics for an invalid scientific state.

Current guardrail: classification runner fails on zero eligible records and
empty label universes. Preserve this behavior and add tests for new metric
paths.

### Invalid model output reaches checkpoints

Risk: unknown labels or malformed records contaminate resumable prediction
files.

Current guardrail: classifier results are validated before checkpointing.
Tests live in `tests/classification/test_runner.py` and
`tests/classification/test_prediction_parser.py`.

### Cache invalidation is incomplete

Risk: stale predictions are reused after text/config/model/policy changes.

Current guardrail: prediction fingerprints include task, text source, model
path, model repo ID, seed, temperature, labels, and policy version; records also
store `metadata.text_sha256`.

If classification semantics change, evaluate whether to bump
`CLASSIFICATION_POLICY_VERSION`.

### CORINE/OSM label policy drift

Risk: single-label CORINE or multi-label OSM assumptions break downstream
metrics.

Tests to inspect:

- `tests/classification/test_ground_truth.py`
- `tests/classification/test_labels.py`
- `tests/classification/test_metrics.py`
- `tests/analysis/test_evaluation_metrics.py`

## Join And Geospatial Mistakes

### Page-ID type mismatches

Risk: joins silently drop rows because one artifact uses integers and another
uses strings.

Current pattern: normalize page IDs to strings in loaders/indexers. Add tests
for new joins.

### CRS or boundary policy changes

Risk: article points join to different polygons.

Current pattern: geospatial builders normalize to WGS84 or metric CRS as needed
and use the shared predicate in `src/georeset_wiki_landcover/spatial/policy.py`.
Boundary points are retained during filtering; ambiguous CORINE ground truth is
excluded later.

### Invalid geometries

Risk: spatial-confidence areas are wrong or operations fail late.

Current pattern: spatial-confidence code uses `shapely.make_valid`. Keep tiny
in-memory geometry tests for changes.

## CLI And Script Mistakes

### Moving reusable logic into `scripts/`

Risk: package users and tests cannot import shared behavior.

Policy: put reusable logic under `src/georeset_wiki_landcover/`; keep
`scripts/data`, `scripts/analysis`, and `scripts/dev` as thin wrappers.

### Breaking packaged entry points

Risk: README, CI, Docker, or cluster jobs call stale commands.

Checks:

- `pyproject.toml` `[project.scripts]`
- `tests/test_packaging_smoke.py`
- `docs/cli.md`

### Running cluster scripts as local tests

Risk: unintended SSH/OAR side effects.

Policy: inspect and test cluster scripts through existing script tests unless
the user explicitly asks to submit/sync Grid5000 jobs.

## Documentation Mistakes

### Claiming unverified commands work

Risk: future researchers waste time on stale instructions.

Policy: either run the command, cite that it is declared by the current code, or
mark it unverified/environment-dependent.

### Duplicating long command lists in multiple places

Risk: docs drift.

Policy: keep root docs concise and link to `docs/cli.md`,
`docs/reproducibility.md`, `docs/artifacts.md`, and `docs/configuration.md`.

## Dependency Mistakes

### Using the wrong package manager

Risk: lockfile and CI drift.

Policy: use `uv`; do not introduce pip/conda instructions unless explicitly
needed and documented as alternative.

Run `uv lock --check` before finalizing dependency or CI changes so lockfile
drift fails locally before it reaches CI.

### Adding heavy dependencies casually

Risk: CI, Docker, and clean-clone setup become slower or brittle.

Policy: prefer existing dependencies. If a new dependency is necessary, update
`pyproject.toml`, `uv.lock`, docs, and tests.
