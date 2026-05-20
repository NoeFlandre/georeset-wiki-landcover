# Grid5000 Cluster Scripts

This folder contains shell launchers for remote GPU jobs on Grid5000/Nancy.

## Classification

- `submit_classification.sh`: syncs code/data, submits one classification OAR
  job, and exits unless `GEORESET_WIKI_LANDCOVER_AUTO_SYNC=1`.
- `run_classification_job.sh`: remote job script that creates an isolated
  `.venv_${OAR_JOB_ID}`, installs `dev` + `llm` dependencies, and runs
  `georeset-wiki-landcover-classify-articles`.
- `sync_classification.sh`: one-shot or interval sync of predictions/metrics
  with JSON validation. Prefer `SYNC_ONCE=1`.

Useful environment variables:

- `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK`
- `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE`
- `GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR`
- `GEORESET_WIKI_LANDCOVER_MODEL_PATH`
- `GEORESET_WIKI_LANDCOVER_MODEL_REPO_ID`
- `GEORESET_WIKI_LANDCOVER_EXTRA_ARGS`
- `G5K_ACCESS_HOST`, `G5K_SITE`, `G5K_REMOTE_DIR`, `G5K_REMOTE_PROJECT_DIR`

By default, classification outputs go to
`data/classification/runs/default/`. For named experiments, set
`GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR`, for example
`data/classification/runs/gemma4_31b_it_q4_0`.

## Summarization

- `submit_summarization.sh`: submits the place-summary job.
- `run_summarization_job.sh`: remote place-summary job.
- `run_summarization_no_place.sh`: remote no-place summary job.
- `run_landuse_evidence_summarization_job.sh`: remote land-use evidence
  summary job.
- `sync_summaries.sh`: one-shot or interval sync of summary JSON.
- `submit_landuse_evidence_summarization.sh`: remote land-use evidence summary
  job submitter (manual sync by default).

## SSH Discipline

Do not create frequent polling loops by default. The administrator previously
flagged excessive frontend SSH logging, so use one-shot status/sync commands or
long intervals.
