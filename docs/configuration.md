# Configuration

This repository keeps most defaults in code and exposes runtime overrides
through CLI flags and environment variables. Prefer explicit CLI flags in
research logs so runs can be reconstructed.

## Dependency Groups

```bash
uv sync --group dev
uv sync --group dev --group llm
uv sync --group dev --group vision
```

- `dev`: tests, linting, type checks, and non-LLM pipeline code.
- `llm`: llama-cpp and Hugging Face Hub dependencies for local LLM workflows.
- `vision`: Sentinel/CLIP dependencies for image workflows.

## Data Sync

```bash
hf sync hf://buckets/NoeFlandre/georeset-wiki-landcover ./data
hf sync ./data hf://buckets/NoeFlandre/georeset-wiki-landcover --delete \
  --exclude '**/.DS_Store' --exclude '.DS_Store'
```

`data/` is intentionally ignored by Git. Generated local smoke outputs should go
under `build/reproducibility/`, which is also ignored.

## Core Environment Variables

| Variable | Used By | Default | Notes |
| --- | --- | --- | --- |
| `PYTHONDONTWRITEBYTECODE` | Python commands | unset | Use `1` in reproducibility commands to avoid `__pycache__` churn. |
| `GEORESET_WIKI_LANDCOVER_MODEL_PATH` | LLM classification and summarization | `Qwen3.6-27B-Q4_0.gguf` | Override with a local GGUF model path. |
| `GEORESET_WIKI_LANDCOVER_MODEL_REPO_ID` | LLM classification and land-use evidence summarization | unset | Optional Hugging Face repo ID for model loading. |
| `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR` | Classification cluster scripts | `data/classification/runs/default` | Prefer stable run-specific directories. |
| `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK` | Classification cluster scripts | required | `corine_level2` or `osm`. |
| `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE` | Classification cluster scripts | required | Any supported text source, including shuffled controls. |
| `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEMPERATURE` | Classification cluster scripts | `0.0` | Also available as CLI `--temperature`. |
| `GEORESET_WIKI_LANDCOVER_EXTRA_ARGS` | Classification cluster job | unset | Extra CLI args split by the shell script. |
| `GEORESET_WIKI_LANDCOVER_AUTO_SYNC` | Cluster submit scripts | `0` | Keep disabled unless repeated SSH polling is intentional. |
| `GEORESET_WIKI_LANDCOVER_SUMMARY_OUTPUT` | Summary sync scripts | `data/wiki/article_summaries.json` | Path synced back from summary jobs. |
| `GEORESET_WIKI_LANDCOVER_JOB_CACHE_DIR` | Cluster job scripts | `${TMPDIR:-/tmp}/georeset_${OAR_JOB_ID:-manual}` | Per-job cache path for selected remote jobs. |
| `SYNC_ONCE` | Cluster sync scripts | `0` | Set to `1` for one manual sync pass. |

## Land-Use Evidence Summary Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_INPUT_PATH` | `data/wiki/article_contents.json` | Input JSON for evidence summaries on Grid5000. |
| `GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_OUTPUT_PATH` | `data/wiki/article_landuse_evidence_summaries.json` | Output JSON path. |
| `GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_SEED` | `42` | Also record CLI `--seed` in run logs. |
| `GEORESET_WIKI_LANDCOVER_LANDUSE_EVIDENCE_TEMPERATURE` | `0.0` | Keep deterministic when possible. |

## Grid5000 Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `G5K_ACCESS_HOST` | `nflandre@access.grid5000.fr` | SSH access host. |
| `G5K_SITE` | `nancy` | Grid5000 site. |
| `G5K_REMOTE_DIR` | `georeset_wiki_landcover` | Remote project directory name. |
| `G5K_REMOTE_USER` | derived from `G5K_ACCESS_HOST` | Remote user. |
| `G5K_REMOTE_HOME` | `/home/${G5K_REMOTE_USER}` | Remote home directory. |
| `G5K_REMOTE_PROJECT_DIR` | `${G5K_REMOTE_HOME}/${G5K_REMOTE_DIR}` | Full remote project path. |
| `G5K_OAR_QUEUE` | `production` | Submit queue; scripts validate allowed characters. |
| `G5K_OAR_TYPES` | unset | Optional OAR `-t` types. |
| `G5K_OAR_PROPERTIES` | script-specific | GPU memory properties for some jobs. |
| `G5K_LANDUSE_EVIDENCE_WALLTIME` | `20:00:00` | Walltime for land-use evidence summary submission. |

## CLI Flags To Record

For classification:

- `--task`;
- `--text-source`;
- input paths;
- `--output-dir`;
- `--model-path`;
- `--model-repo-id`;
- `--seed`;
- `--temperature`;
- `--limit`;
- `--retry-failed`.

For summarization and evidence extraction:

- input and output paths;
- `--summary-mode`;
- model path or repo ID;
- seed;
- temperature.

For validation:

- `scripts/validate_artifacts.py --root ... --profile small`;
- `scripts/validate_artifacts.py --root data --profile full`.

## Defaults In Code

`src/georeset_wiki_landcover/config.py` defines default data paths and model
settings. Classification cache fingerprints also include the classification
policy version from `src/georeset_wiki_landcover/classification/runner.py`.

When changing defaults, update this document and prefer a test that proves old
caches fail or skip intentionally.

## More References

- `docs/cli.md` lists packaged entry points and repository scripts.
- `docs/data_flow.md` explains which artifacts each workflow reads and writes.
- `docs/troubleshooting.md` lists common invalid states and the checks that
  catch them.
