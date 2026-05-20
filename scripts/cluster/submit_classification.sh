#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset_wiki_landcover}"
REMOTE_USER="${G5K_REMOTE_USER:-${ACCESS_HOST%@*}}"
REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"
REMOTE_ACCESS_DIR="${SITE}/${REMOTE_DIR}"
TASK="${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK:?Set GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK}"
TEXT_SOURCE="${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE:?Set GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE}"
OUTPUT_DIR="${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR:-data/classification/runs/default}"
OUTPUT_PREFIX="${OUTPUT_DIR}/${TASK}_${TEXT_SOURCE}"
JOB_SCRIPT="scripts/cluster/run_classification_job.sh"
AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"
OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-gpu_mem>=32000}"
OAR_QUEUE="${G5K_OAR_QUEUE:-production}"
OAR_TYPES="${G5K_OAR_TYPES:-}"
OAR_TYPE_FLAGS=""

case "${OAR_QUEUE}" in
  *[!A-Za-z0-9_.:-]*)
    echo "Invalid OAR queue: ${OAR_QUEUE}." >&2
    exit 1
    ;;
esac

if [ -n "${OAR_TYPES}" ]; then
  for OAR_TYPE in ${OAR_TYPES}; do
    case "${OAR_TYPE}" in
      *[!A-Za-z0-9_.:-]*)
        echo "Invalid OAR type: ${OAR_TYPE}." >&2
        exit 1
        ;;
    esac
    OAR_TYPE_FLAGS="${OAR_TYPE_FLAGS} -t ${OAR_TYPE}"
  done
fi

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
      oarsub -q \"${OAR_QUEUE}\" \
        ${OAR_TYPE_FLAGS} \
        -l host=1/gpu=1,walltime=20:00:00 \
        -p \"${OAR_PROPERTIES}\" \
        -O OAR_%jobid%.out \
        -E OAR_%jobid%.err \
        \"env GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK=\\\"${TASK}\\\" \
        GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE=\\\"${TEXT_SOURCE}\\\" \
        GEORESET_WIKI_LANDCOVER_MODEL_PATH=\\\"${GEORESET_WIKI_LANDCOVER_MODEL_PATH:-Qwen3.6-27B-Q4_0.gguf}\\\" \
        GEORESET_WIKI_LANDCOVER_MODEL_REPO_ID=\\\"${GEORESET_WIKI_LANDCOVER_MODEL_REPO_ID:-}\\\" \
        GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEMPERATURE=\\\"${GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEMPERATURE:-0.0}\\\" \
        GEORESET_WIKI_LANDCOVER_EXTRA_ARGS=\\\"${GEORESET_WIKI_LANDCOVER_EXTRA_ARGS:-}\\\" \
        GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR=\\\"${OUTPUT_DIR}\\\" \
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
  echo "Manual sync: GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK=${TASK} GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE=${TEXT_SOURCE} GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR=${OUTPUT_DIR} SYNC_ONCE=1 bash scripts/cluster/sync_classification.sh"
  exit 0
fi

echo "Syncing ${OUTPUT_PREFIX}_predictions.json and _metrics.json; press Ctrl+C to stop."

GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TASK="${TASK}" \
GEORESET_WIKI_LANDCOVER_CLASSIFICATION_TEXT_SOURCE="${TEXT_SOURCE}" \
GEORESET_WIKI_LANDCOVER_CLASSIFICATION_OUTPUT_DIR="${OUTPUT_DIR}" \
G5K_ACCESS_HOST="${ACCESS_HOST}" \
G5K_SITE="${SITE}" \
G5K_REMOTE_DIR="${REMOTE_DIR}" \
G5K_REMOTE_HOME="${REMOTE_HOME}" \
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-20}" \
bash scripts/cluster/sync_classification.sh
