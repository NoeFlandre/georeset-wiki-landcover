#!/usr/bin/env bash
set -euo pipefail

ACCESS_HOST="${G5K_ACCESS_HOST:-nflandre@access.grid5000.fr}"
SITE="${G5K_SITE:-nancy}"
REMOTE_DIR="${G5K_REMOTE_DIR:-georeset_wiki_landcover}"
REMOTE_USER="${G5K_REMOTE_USER:-${ACCESS_HOST%@*}}"
REMOTE_HOME="${G5K_REMOTE_HOME:-/home/${REMOTE_USER}}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${REMOTE_HOME}/${REMOTE_DIR}}"
REMOTE_ACCESS_DIR="${SITE}/${REMOTE_DIR}"
OUTPUT_DIR="${IMAGE_PROBE_OUTPUT_DIR:-data/experiments/014_quality_weighted_multiscale_image_probe/quality_weighted_multiscale_image_probe_v1}"
REMOTE_ACCESS_OUTPUT_DIR="${REMOTE_ACCESS_DIR}/${OUTPUT_DIR}"
REMOTE_PROJECT_OUTPUT_DIR="${REMOTE_PROJECT_DIR}/${OUTPUT_DIR}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_SITE_HOST="${SITE}"
OAR_QUEUE="${G5K_OAR_QUEUE:-production}"
OAR_PROPERTIES="${G5K_OAR_PROPERTIES:-gpu_mem>=32000}"
OAR_WALLTIME="${IMAGE_PROBE_WALLTIME:-48:00:00}"

ssh -o BatchMode=yes "${ACCESS_HOST}" "mkdir -p '${REMOTE_ACCESS_DIR}'"

rsync -az --delete \
  --exclude '.git' \
  --exclude '.venv*' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '.mypy_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'build' \
  --exclude 'dist' \
  --exclude '*.egg-info' \
  --exclude 'data/corine' \
  --exclude 'data/maps' \
  --exclude 'data/distribution' \
  --exclude 'data/classification' \
  --exclude 'data/experiments/014_quality_weighted_multiscale_image_probe' \
  --exclude 'OAR_*' \
  --exclude 'wrapper*.sh' \
  --exclude '*.db' \
  "${LOCAL_ROOT}/" "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/"

for input_dir in \
  "data/experiments/001_qwen_e2e_shuffled_control" \
  "data/experiments/004_gemma4_model_rerun_and_comparison" \
  "data/experiments/008_supervision_quality_score"; do
  if [[ -d "${LOCAL_ROOT}/${input_dir}" ]]; then
    ssh "${ACCESS_HOST}" "mkdir -p '${REMOTE_ACCESS_DIR}/${input_dir%/*}'"
    rsync -az "${LOCAL_ROOT}/${input_dir}/" "${ACCESS_HOST}:${REMOTE_ACCESS_DIR}/${input_dir}/"
  fi
done

if [[ -d "${LOCAL_ROOT}/${OUTPUT_DIR}" ]]; then
  ssh "${ACCESS_HOST}" "mkdir -p '${REMOTE_ACCESS_OUTPUT_DIR}'"
  rsync -az "${LOCAL_ROOT}/${OUTPUT_DIR}/" "${ACCESS_HOST}:${REMOTE_ACCESS_OUTPUT_DIR}/"
fi

SUBMIT_OUTPUT="$(
  ssh "${ACCESS_HOST}" "ssh ${REMOTE_SITE_HOST} 'cd ${REMOTE_PROJECT_DIR} && chmod +x ./scripts/cluster/run_quality_weighted_image_probe_job.sh && oarsub \
    -q \"${OAR_QUEUE}\" \
    -l host=1/gpu=1,walltime=${OAR_WALLTIME} \
    -p \"${OAR_PROPERTIES}\" \
    -O OAR_%jobid%.out \
    -E OAR_%jobid%.err \
    \"env IMAGE_PROBE_ENCODERS=\\\"${IMAGE_PROBE_ENCODERS:-clip_base}\\\" \
    IMAGE_PROBE_WINDOWS=\\\"${IMAGE_PROBE_WINDOWS:-320,2240}\\\" \
    IMAGE_PROBE_RUN_CONTROLS=\\\"${IMAGE_PROBE_RUN_CONTROLS:-0}\\\" \
    IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION=\\\"${IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION:-0}\\\" \
    IMAGE_PROBE_OUTPUT_DIR=\\\"${OUTPUT_DIR}\\\" \
    G5K_REMOTE_PROJECT_DIR=\\\"${REMOTE_PROJECT_DIR}\\\" \
    bash ./scripts/cluster/run_quality_weighted_image_probe_job.sh\"'"
)"
echo "${SUBMIT_OUTPUT}"

JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | sed -n 's/.*OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -n "${JOB_ID}" ]]; then
  echo "Submitted OAR job ${JOB_ID}"
  echo "Watch status: ssh ${ACCESS_HOST} \"ssh ${REMOTE_SITE_HOST} 'oarstat -j ${JOB_ID}'\""
  echo "Watch stderr: ssh ${ACCESS_HOST} \"ssh ${REMOTE_SITE_HOST} 'tail -f ${REMOTE_PROJECT_DIR}/OAR_${JOB_ID}.err'\""
fi

cat <<EOF
Submitted Experiment 014 to Grid5000.

MVP defaults:
  IMAGE_PROBE_ENCODERS=clip_base
  IMAGE_PROBE_WINDOWS=320,2240
  IMAGE_PROBE_RUN_CONTROLS=0
  IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION=0

Full run example:
  IMAGE_PROBE_ENCODERS=clip_base,clip_large,dinov2_base IMAGE_PROBE_WINDOWS=320,640,1280,2240 ./scripts/cluster/submit_quality_weighted_image_probe.sh

Patch-validation-only example:
  IMAGE_PROBE_STOP_AFTER_PATCH_VALIDATION=1 ./scripts/cluster/submit_quality_weighted_image_probe.sh

Controls run example:
  IMAGE_PROBE_RUN_CONTROLS=1 ./scripts/cluster/submit_quality_weighted_image_probe.sh

Watch:
  ssh ${ACCESS_HOST} "ssh ${REMOTE_SITE_HOST} 'oarstat -u ${REMOTE_USER}'"

Sync back:
  rsync -az ${ACCESS_HOST}:${REMOTE_ACCESS_OUTPUT_DIR}/ ${LOCAL_ROOT}/${OUTPUT_DIR}/
EOF

if [[ "${IMAGE_PROBE_SYNC_AFTER_SUBMIT:-0}" == "1" ]]; then
  rsync -az "${ACCESS_HOST}:${REMOTE_ACCESS_OUTPUT_DIR}/" "${LOCAL_ROOT}/${OUTPUT_DIR}/"
fi
