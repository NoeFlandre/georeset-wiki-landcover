#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
REMOTE_USER="${G5K_REMOTE_USER:-${ACCESS_HOST%@*}}"
REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"
REMOTE_ACCESS_DIR="${SITE}/${REMOTE_DIR}"
TASK="${GEORESET_CLASSIFICATION_TASK:?Set GEORESET_CLASSIFICATION_TASK}"
TEXT_SOURCE="${GEORESET_CLASSIFICATION_TEXT_SOURCE:?Set GEORESET_CLASSIFICATION_TEXT_SOURCE}"
OUTPUT_DIR="${GEORESET_CLASSIFICATION_OUTPUT_DIR:-data/classification}"
OUTPUT_PREFIX="${OUTPUT_DIR}/${TASK}_${TEXT_SOURCE}"
JOB_SCRIPT="scripts/cluster/run_classification_job.sh"
AUTO_SYNC="${GEORESET_AUTO_SYNC:-0}"

case "${TASK}:${TEXT_SOURCE}" in
  *[!A-Za-z0-9_:.-]*)
    echo "Invalid classification task or text source." >&2
    exit 1
    ;;
esac

echo "Preparing remote directory ${SITE}/${REMOTE_DIR}"
ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${REMOTE_ACCESS_DIR}"

echo "Syncing repository to Grid5000"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv*' \
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
  --exclude 'data/classification' \
  --exclude 'OAR_*' \
  --exclude 'wrapper*.sh' \
  --exclude '*.db' \
  ./ "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/"

echo "Syncing data directories to Grid5000"
rsync -az \
  data/wiki \
  data/osm \
  data/corine \
  "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/data/"

ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Syncing existing classification outputs to Grid5000 for resumable retry"
rsync -az \
  "${OUTPUT_DIR}/" \
  "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}/"

echo "Submitting OAR job"
SUBMIT_OUTPUT="$(
  ssh -o BatchMode=yes "${ACCESS_HOST}" "
    ssh ${SITE} 'cd \"${REMOTE_PROJECT_DIR}\" && chmod +x \"${JOB_SCRIPT}\" && \
      oarsub -q production \
        -l host=1/gpu=1,walltime=20:00:00 \
        -p \"gpu_mem>=32000\" \
        -O OAR_%jobid%.out \
        -E OAR_%jobid%.err \
        \"env GEORESET_CLASSIFICATION_TASK=\\\"${TASK}\\\" \
        GEORESET_CLASSIFICATION_TEXT_SOURCE=\\\"${TEXT_SOURCE}\\\" \
        GEORESET_MODEL_PATH=\\\"${GEORESET_MODEL_PATH:-Qwen3.6-27B-Q4_0.gguf}\\\" \
        GEORESET_MODEL_REPO_ID=\\\"${GEORESET_MODEL_REPO_ID:-}\\\" \
        GEORESET_CLASSIFICATION_TEMPERATURE=\\\"${GEORESET_CLASSIFICATION_TEMPERATURE:-0.0}\\\" \
        GEORESET_EXTRA_ARGS=\\\"${GEORESET_EXTRA_ARGS:-}\\\" \
        GEORESET_CLASSIFICATION_OUTPUT_DIR=\\\"${OUTPUT_DIR}\\\" \
        G5K_REMOTE_DIR=\\\"${REMOTE_DIR}\\\" \
        G5K_REMOTE_PROJECT_DIR=\\\"${REMOTE_PROJECT_DIR}\\\" \
        bash ./\\\"${JOB_SCRIPT}\\\"\"'
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
  echo "Auto-sync disabled to avoid repeated SSH polling. Run sync_classification.sh manually when needed."
  echo "Manual sync: GEORESET_CLASSIFICATION_TASK=${TASK} GEORESET_CLASSIFICATION_TEXT_SOURCE=${TEXT_SOURCE} GEORESET_CLASSIFICATION_OUTPUT_DIR=${OUTPUT_DIR} SYNC_ONCE=1 bash scripts/cluster/sync_classification.sh"
  exit 0
fi

echo "Syncing ${OUTPUT_PREFIX}_predictions.json and _metrics.json; press Ctrl+C to stop."

GEORESET_CLASSIFICATION_TASK="${TASK}" \
GEORESET_CLASSIFICATION_TEXT_SOURCE="${TEXT_SOURCE}" \
GEORESET_CLASSIFICATION_OUTPUT_DIR="${OUTPUT_DIR}" \
G5K_ACCESS_HOST="${ACCESS_HOST}" \
G5K_SITE="${SITE}" \
G5K_REMOTE_DIR="${REMOTE_DIR}" \
G5K_REMOTE_HOME="${REMOTE_HOME}" \
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-20}" \
bash scripts/cluster/sync_classification.sh
