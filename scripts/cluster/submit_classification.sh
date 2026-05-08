#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
TASK="${GEORESET_CLASSIFICATION_TASK:?Set GEORESET_CLASSIFICATION_TASK}"
TEXT_SOURCE="${GEORESET_CLASSIFICATION_TEXT_SOURCE:?Set GEORESET_CLASSIFICATION_TEXT_SOURCE}"
OUTPUT_PREFIX="data/classification/${TASK}_${TEXT_SOURCE}"
JOB_SCRIPT="scripts/cluster/run_classification_job.sh"

echo "Preparing remote directory ${SITE}/${REMOTE_DIR}"
ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${SITE}/${REMOTE_DIR}"

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
  --exclude 'data/distribution' \
  --exclude '*.db' \
  ./ "${ACCESS_HOST}:${SITE}/${REMOTE_DIR}/"

echo "Syncing data directories to Grid5000"
rsync -az \
  data/wiki \
  data/osm \
  data/corine \
  "${ACCESS_HOST}:${SITE}/${REMOTE_DIR}/data/"

ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${SITE}/${REMOTE_DIR}/data/classification"

echo "Submitting OAR job"
SUBMIT_OUTPUT="$(
  ssh -o BatchMode=yes "${ACCESS_HOST}" "
    ssh ${SITE} 'cd ${REMOTE_DIR} && chmod +x ${JOB_SCRIPT} && \
    GEORESET_CLASSIFICATION_TASK=${TASK} \
    GEORESET_CLASSIFICATION_TEXT_SOURCE=${TEXT_SOURCE} \
    GEORESET_MODEL_PATH=\${GEORESET_MODEL_PATH:-Qwen3.6-27B-Q4_0.gguf} \
    oarsub -S ./${JOB_SCRIPT}'
  ")"
echo "${SUBMIT_OUTPUT}"

JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | sed -n 's/.*OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [ -z "${JOB_ID}" ]; then
  echo "Could not parse OAR job id from submission output." >&2
  exit 1
fi

echo "Submitted OAR job ${JOB_ID}"
echo "Watch status: ssh -o BatchMode=yes ${ACCESS_HOST} \"ssh ${SITE} 'oarstat -j ${JOB_ID}'\""
echo "Watch stderr: ssh -o BatchMode=yes ${ACCESS_HOST} \"ssh ${SITE} 'tail -f /home/nflandre/${REMOTE_DIR}/OAR_${JOB_ID}.err'\""
echo "Syncing ${OUTPUT_PREFIX}_predictions.json and _metrics.json; press Ctrl+C to stop."

GEORESET_CLASSIFICATION_TASK="${TASK}" \
GEORESET_CLASSIFICATION_TEXT_SOURCE="${TEXT_SOURCE}" \
G5K_ACCESS_HOST="${ACCESS_HOST}" \
G5K_SITE="${SITE}" \
G5K_REMOTE_DIR="${REMOTE_DIR}" \
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-20}" \
bash scripts/cluster/sync_classification.sh
