#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset_wiki_landcover}"
REMOTE_USER="${G5K_REMOTE_USER:-${ACCESS_HOST%@*}}"
REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"
REMOTE_ACCESS_DIR="${SITE}/${REMOTE_DIR}"
JOB_SCRIPT="scripts/cluster/run_clip_linear_probe_job.sh"
OUTPUT_DIR="${CLIP_OUTPUT_DIR:-data/experiments/012_clip_linear_probe_weak_labels/clip_linear_probe_weak_labels_v1}"
AUTO_SYNC="${GEORESET_WIKI_LANDCOVER_AUTO_SYNC:-0}"
OAR_QUEUE="${G5K_OAR_QUEUE:-production}"
OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-gpu_mem>=16000}"

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

echo "Syncing experiment input data to Grid5000"
ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p \
  ${REMOTE_ACCESS_DIR}/data/experiments/001_qwen_e2e_shuffled_control \
  ${REMOTE_ACCESS_DIR}/data/experiments/004_gemma4_model_rerun_and_comparison \
  ${REMOTE_ACCESS_DIR}/data/experiments/008_supervision_quality_score"
rsync -az \
  data/experiments/008_supervision_quality_score/article_text_supervision_quality_score_v1 \
  "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/data/experiments/008_supervision_quality_score/"
rsync -az \
  data/experiments/001_qwen_e2e_shuffled_control/article_text_classification_e2e_with_shuffled_control_v1 \
  "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/data/experiments/001_qwen_e2e_shuffled_control/"
rsync -az \
  data/experiments/004_gemma4_model_rerun_and_comparison/article_text_classification_e2e_with_shuffled_control_v1__gemma4_31b_it_q4_0 \
  "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/data/experiments/004_gemma4_model_rerun_and_comparison/"

ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p ${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Syncing existing CLIP outputs to Grid5000 for resumable retry"
rsync -az \
  "${OUTPUT_DIR}/" \
  "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}/"

echo "Submitting OAR job"
SUBMIT_OUTPUT="$(
  ssh -o BatchMode=yes "${ACCESS_HOST}" "
    ssh ${SITE} 'cd \"${REMOTE_PROJECT_DIR}\" && chmod +x \"${JOB_SCRIPT}\" && \
      oarsub -q \"${OAR_QUEUE}\" \
        -l host=1/gpu=1,walltime=${CLIP_WALLTIME:-20:00:00} \
        -p \"${OAR_PROPERTIES}\" \
        -O OAR_%jobid%.out \
        -E OAR_%jobid%.err \
        \"env CLIP_OUTPUT_DIR=\\\"${OUTPUT_DIR}\\\" \
        CLIP_MODEL_NAME=\\\"${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}\\\" \
        CLIP_EVAL_PER_CLASS=\\\"${CLIP_EVAL_PER_CLASS:-5}\\\" \
        CLIP_TRAIN_PER_CLASS=\\\"${CLIP_TRAIN_PER_CLASS:-80}\\\" \
        CLIP_PATCH_SIZE=\\\"${CLIP_PATCH_SIZE:-224}\\\" \
        CLIP_CLOUD_COVER=\\\"${CLIP_CLOUD_COVER:-25}\\\" \
        CLIP_DATETIME_RANGE=\\\"${CLIP_DATETIME_RANGE:-2022-04-01/2022-10-31}\\\" \
        CLIP_BATCH_SIZE=\\\"${CLIP_BATCH_SIZE:-32}\\\" \
        CLIP_LINEAR_EPOCHS=\\\"${CLIP_LINEAR_EPOCHS:-600}\\\" \
        CLIP_LINEAR_LEARNING_RATE=\\\"${CLIP_LINEAR_LEARNING_RATE:-0.1}\\\" \
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
  echo "Auto-sync disabled. Run:"
  echo "rsync -az ${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}/ ${OUTPUT_DIR}/"
  exit 0
fi

while true; do
  rsync -az "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}/" "${OUTPUT_DIR}/" || true
  [ -s "${OUTPUT_DIR}/linear_probe_metrics.csv" ] && [ -s "${OUTPUT_DIR}/summary.md" ] && break
  sleep "${SYNC_INTERVAL_SECONDS:-300}"
done

echo "Synced CLIP outputs:"
ls -lh "${OUTPUT_DIR}/label_splits.csv" \
  "${OUTPUT_DIR}/sentinel_patches_rgb.npz" \
  "${OUTPUT_DIR}/clip_embeddings.npz" \
  "${OUTPUT_DIR}/linear_probe_metrics.csv" \
  "${OUTPUT_DIR}/zero_shot_clip_metrics.csv" \
  "${OUTPUT_DIR}/summary.md"
