# GPU Classification on Grid5000 Guide

This document outlines the workflow, commands, and lessons learned for running GPU classification jobs on the Grid5000 cluster for the GeoReset project.

## Prerequisites

- SSH access to Grid5000 (`access.grid5000.fr`)
- A model downloaded or available in the remote environment (default expected: `Qwen3.6-27B-Q4_0.gguf`)

## Overview

The job submission process copies the local repository and data to the Grid5000 cluster, submits an OAR job requesting GPU resources, and provides a synchronization script to pull results back locally as they are generated.

## How to Submit a Job

Use the provided cluster submission script: `scripts/cluster/submit_classification.sh`.

You must define the task and text source via environment variables.

Example to submit a `corine_level2` classification using `summary` texts:

```bash
GEORESET_CLASSIFICATION_TASK=corine_level2 \
GEORESET_CLASSIFICATION_TEXT_SOURCE=summary \
bash scripts/cluster/submit_classification.sh
```

### What the Script Does:
1. Connects to `nflandre@access.grid5000.fr`, then SSHes into the `nancy` site.
2. Synchronizes the local codebase (excluding ignored directories like `.venv`) and necessary data (`data/osm`, `data/corine`) via `rsync`.
3. Creates a `wrapper.sh` script that exports necessary variables (see "Lessons Learned") and submits it to the OAR scheduler using `oarsub`.
4. Outputs the `OAR_JOB_ID` and commands to monitor the job.

## Monitoring the Job

Once submitted, you will receive an `OAR_JOB_ID` (e.g., `6407544`).

**Check Job Status:**
```bash
ssh -o BatchMode=yes nflandre@access.grid5000.fr "ssh nancy 'oarstat -j <OAR_JOB_ID>'"
```
*(Look at the `S` column: `W` = Waiting, `R` = Running, `F` = Finished, `E` = Error)*

**Tail Stderr Logs (useful for debugging and progress):**
```bash
ssh -o BatchMode=yes nflandre@access.grid5000.fr "ssh nancy 'tail -f /home/nflandre/georeset/OAR_<OAR_JOB_ID>.err'"
```

**Tail Stdout Logs:**
```bash
ssh -o BatchMode=yes nflandre@access.grid5000.fr "ssh nancy 'tail -f /home/nflandre/georeset/OAR_<OAR_JOB_ID>.out'"
```

## Synchronizing Outputs

As the job runs, it writes predictions (and eventually metrics) to the `data/classification/` directory on the Grid5000 node. 

The `submit_classification.sh` script automatically starts a `sync_classification.sh` loop after submission to pull outputs every 20 seconds. If this sync loop is interrupted, you can resume it manually:

```bash
GEORESET_CLASSIFICATION_TASK=corine_level2 \
GEORESET_CLASSIFICATION_TEXT_SOURCE=summary \
bash scripts/cluster/sync_classification.sh
```

### Expected Output Files
- `data/classification/<task>_<text_source>_predictions.json`: Contains the individual prediction records. This file will grow incrementally.
- `data/classification/<task>_<text_source>_metrics.json`: Contains the final evaluation metrics. This file is generated only after the job finishes completely.

## Lessons Learned & Technical Details

1. **Environment Variable Propagation with OAR:**
   - Standard `oarsub` commands sometimes fail to inherit inline environment variables passed via `env`. 
   - **Fix Applied:** We modified `scripts/cluster/submit_classification.sh` to generate a `wrapper.sh` script on the fly. The script dynamically injects `export` statements right after the `#!/usr/bin/env bash` line, preserving all `#OAR` directives. This ensures variables like `GEORESET_CLASSIFICATION_TASK` are correctly available inside the GPU job.
2. **Missing `~/.bashrc` warning:**
   - OAR stderr logs may print: `/var/lib/oar/.batch_job_bashrc: line 5: /home/nflandre/.bashrc: No such file or directory`. This is harmless and does not affect the job execution, as the `run_classification_job.sh` manages its own environment using `uv`.
3. **Data Verification:**
   - The JSON prediction files can be quite large. A quick way to verify them locally without dumping them to terminal is to use Python:
     ```bash
     python -c '
     import json
     with open("data/classification/corine_level2_summary_predictions.json") as f:
         d = json.load(f)
         print("Total:", len(d))
         print("Errors:", sum(1 for i in d.values() if isinstance(i, dict) and i.get("parse_status") == "error"))
     '
     ```