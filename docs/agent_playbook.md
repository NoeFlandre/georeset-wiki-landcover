# Agent Playbook

This playbook is for future coding-agent runs. It expands the root
`AGENTS.md` with repo-specific workflow advice.

## Start Of Run Checklist

1. Check the branch and worktree:

   ```bash
   git status --short --branch
   ```

2. Read the local guidance:

   ```bash
   sed -n '1,220p' AGENTS.md
   sed -n '1,220p' docs/architecture.md
   sed -n '1,220p' docs/data_flow.md
   sed -n '1,220p' docs/cli.md
   ```

3. Identify whether the task touches code, docs, data artifacts, cluster
   scripts, or frozen experiment reports.
4. Inspect the nearest tests before editing. The tests are organized by package
   area and document expected behavior.

## Dependency And Command Policy

Use `uv`. CI installs Python 3.10 and runs:

```bash
uv sync --frozen --group dev
uv lock --check
uv run python scripts/dev/check_repository_hygiene.py
uv run georeset-wiki-landcover-classify-articles --help
uv run python scripts/reproduce_small.py \
  --output-dir build/reproducibility/small \
  --clean
uv run python scripts/validate_artifacts.py \
  --root build/reproducibility/small \
  --profile small
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run pytest -q
```

Use `PYTHONDONTWRITEBYTECODE=1` for local commands when possible to avoid
`__pycache__` churn:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest tests/classification/test_runner.py -q --no-cov
```

Use optional groups only when needed:

- `llm`: llama-cpp/Hugging Face model workflows.
- `vision`: Sentinel, CLIP, torch, rasterio, and image probe workflows.

## Choosing Verification

For docs-only changes:

- Run `ruff check .`, `ruff format --check .`, and any command that the docs
  newly claim was verified.
- Run the small reproduction path if docs mention reproducibility or artifacts.

For classification changes:

- Start with the closest tests in `tests/classification/`.
- Include parser, runner, records, metrics, and ground-truth tests when cache,
  output schema, or label behavior changes.
- Run full `pytest -q` before finalizing behavior changes because coverage is
  scoped to `georeset_wiki_landcover.classification`.

For geospatial changes:

- Use tiny GeoJSON or in-memory GeoDataFrames.
- Run `tests/classification/test_ground_truth.py`,
  `tests/spatial/test_corine_confidence.py`, and affected CLI/script tests.

For artifact/reproducibility changes:

- Run:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 uv run python scripts/reproduce_small.py \
    --output-dir build/reproducibility/small \
    --clean
  PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_artifacts.py \
    --root build/reproducibility/small \
    --profile small
  ```

For cluster script changes:

- Inspect `tests/scripts/test_classification_cluster_scripts.py`,
  `tests/scripts/test_summarization_cluster_scripts.py`, and
  `tests/test_packaging_smoke.py`.
- Do not run Grid5000 commands unless the user explicitly asks and credentials
  are available.

## Artifact Rules

- `data/` is bucket-synced and ignored by Git.
- `build/` contains local smoke outputs and build artifacts; keep it out of Git.
- Use `scripts/validate_artifacts.py --profile small` for synthetic packages.
- Use `scripts/validate_artifacts.py --root data --profile full` only when
  synced full data is present. It is a lightweight preflight, not proof of full
  reproduction.
- Use `utils/json_io.py` atomic writers for new artifacts.

Do not stage generated artifacts accidentally. Before commit:

```bash
git status --short
git ls-files data build
PYTHONDONTWRITEBYTECODE=1 uv run python scripts/dev/check_repository_hygiene.py
```

The second command should print nothing, and the hygiene check should pass.

## Research Contracts To Preserve

- CORINE level-2 classification is single-label.
- OSM classification is multi-label.
- Artificial CORINE classes are excluded from classification ground truth but
  can remain evidence in spatial-confidence diagnostics.
- Shuffled text controls change text assignment only, not targets or labels.
- Frozen experiment directories under `data/experiments/` are inputs for
  analysis reruns. Avoid mutating them unless that is the task.
- Analysis commands generally evaluate frozen predictions and should not rerun
  an LLM unless explicitly designed to do so.

## Common Change Patterns

### Add A New CLI

1. Put implementation under `src/georeset_wiki_landcover/cli/...`.
2. Put reusable logic outside the CLI wrapper if it will be tested or reused.
3. Add a `[project.scripts]` entry in `pyproject.toml`.
4. Add or update tests that exercise argument parsing and output contracts.
5. Update `docs/cli.md` and related docs.

### Add A New Artifact

1. Define exact path, schema, and whether it is source, derived, or cache.
2. Use atomic writers.
3. Add validator coverage if it belongs to the reproducibility package or full
   preflight.
4. Document it in `docs/artifacts.md` and `docs/data_flow.md`.

### Change Classification Behavior

1. Add a regression test first.
2. Decide whether cache invalidation is required.
3. If required, bump `CLASSIFICATION_POLICY_VERSION`.
4. Confirm prediction metadata and metrics remain interpretable.
5. Update docs that mention cache or output schema.

## Reporting Uncertainty

Be explicit about:

- commands you did not run;
- workflows requiring unavailable private/synced data;
- external services not exercised locally;
- assumptions inferred from tests rather than from live data;
- behavior intentionally preserved even if it looks imperfect.

Use phrases like "not verified in this environment" rather than implying a full
pipeline was tested when only the small synthetic path ran.
