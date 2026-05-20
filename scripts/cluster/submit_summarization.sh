#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset_wiki_landcover}"
REMOTE_USER="${G5K_REMOTE_USER:-${ACCESS_HOST%@*}}"
REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"
REMOTE_ACCESS_DIR="${SITE}/${REMOTE_DIR}"
OUTPUT_PATH="data/wiki/article_summaries.json"
JOB_SCRIPT="scripts/cluster/run_summarization_job.sh"
AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"

mkdir -p data/wiki

echo "Preparing remote directory ${SITE}/${REMOTE_DIR}"
ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${REMOTE_ACCESS_DIR}"

echo "Syncing repository to Grid5000"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'build' \
  --exclude 'dist' \
  --exclude '*.egg-info' \
  --exclude 'data/corine' \
  --exclude 'data/maps' \
  --exclude 'data/osm' \
  --exclude 'data/distribution' \
  --exclude 'OAR_*' \
  --exclude '*.db' \
  ./ "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/"

echo "Submitting OAR job"
SUBMIT_OUTPUT="$(
  ssh -o BatchMode=yes "${ACCESS_HOST}" "
    ssh ${SITE} 'cd \"${REMOTE_PROJECT_DIR}\" && chmod +x \"${JOB_SCRIPT}\" && \
    env \
      GEORESET_WIKI_LANDCOVER_MODEL_PATH=\"${GEORESET_WIKI_LANDCOVER_MODEL_PATH:-Qwen3.6-27B-Q4_0.gguf}\" \
      G5K_REMOTE_DIR=\"${REMOTE_DIR}\" \
      G5K_REMOTE_PROJECT_DIR=\"${REMOTE_PROJECT_DIR}\" \
      oarsub -S ./\"${JOB_SCRIPT}\"'
  ")"
echo "${SUBMIT_OUTPUT}"

JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | sed -n 's/.*OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [ -z "${JOB_ID}" ]; then
  echo "Could not parse OAR job id from submission output." >&2
  exit 1
fi

echo "Submitted OAR job ${JOB_ID}"
echo "Watch status: ssh -o BatchMode=yes ${ACCESS_HOST} \"ssh ${SITE} 'oarstat -j ${JOB_ID}'\""
echo "Watch stderr: ssh -o BatchMode=yes ${ACCESS_HOST} \"ssh ${SITE} 'tail -f ${REMOTE_PROJECT_DIR}/OAR_${JOB_ID}.err'\""
if [ "${AUTO_SYNC}" != "1" ]; then
  echo "Auto-sync disabled to avoid repeated SSH polling. Run sync_summaries.sh manually when needed."
  echo "Manual sync: GEORESET_WIKI_LANDCOVER_SUMMARY_OUTPUT=${OUTPUT_PATH} SYNC_ONCE=1 bash scripts/cluster/sync_summaries.sh"
  exit 0
fi

echo "Syncing ${OUTPUT_PATH}; press Ctrl+C to stop syncing after the job finishes."

G5K_ACCESS_HOST="${ACCESS_HOST}" G5K_SITE="${SITE}" G5K_REMOTE_DIR="${REMOTE_DIR}" \
  G5K_REMOTE_HOME="${REMOTE_HOME}" G5K_REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR}" \
  GEORESET_WIKI_LANDCOVER_SUMMARY_OUTPUT="${OUTPUT_PATH}" \
  SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}" \
  bash scripts/cluster/sync_summaries.sh
