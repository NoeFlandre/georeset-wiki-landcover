#!/usr/bin/env bash
#OAR -q production
#OAR -l host=1/gpu=1,walltime=20:00:00
#OAR -p gpu_mem>=16000
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

set -euo pipefail

REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${HOME}/${REMOTE_DIR}}"
cd "${REMOTE_PROJECT_DIR}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export CLIP_OUTPUT_DIR="${CLIP_OUTPUT_DIR:-data/experiments/clip_linear_probe_weak_labels_v1}"
export CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
JOB_CACHE_DIR="${GEORESET_JOB_CACHE_DIR:-${TMPDIR:-/tmp}/georeset_${OAR_JOB_ID:-manual}}"
export HF_HOME="${HF_HOME:-${JOB_CACHE_DIR}/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${JOB_CACHE_DIR}/uv}"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${CLIP_OUTPUT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

export UV_PROJECT_ENVIRONMENT="${REMOTE_PROJECT_DIR}/.venv_clip_${OAR_JOB_ID:-manual}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"

uv sync --group dev --group vision

uv run georeset-build-clip-label-splits \
  --output-path "${CLIP_OUTPUT_DIR}/label_splits.csv" \
  --eval-per-class "${CLIP_EVAL_PER_CLASS:-5}" \
  --train-per-class "${CLIP_TRAIN_PER_CLASS:-80}"

uv run georeset-fetch-sentinel-patches \
  --splits-path "${CLIP_OUTPUT_DIR}/label_splits.csv" \
  --output-path "${CLIP_OUTPUT_DIR}/sentinel_patches_rgb.npz" \
  --patch-size "${CLIP_PATCH_SIZE:-224}" \
  --cloud-cover "${CLIP_CLOUD_COVER:-25}" \
  --datetime-range "${CLIP_DATETIME_RANGE:-2022-04-01/2022-10-31}"

uv run georeset-embed-clip-patches \
  --patches-path "${CLIP_OUTPUT_DIR}/sentinel_patches_rgb.npz" \
  --output-path "${CLIP_OUTPUT_DIR}/clip_embeddings.npz" \
  --model-name "${CLIP_MODEL_NAME}" \
  --device "${CLIP_DEVICE:-cuda}" \
  --batch-size "${CLIP_BATCH_SIZE:-32}"

uv run georeset-run-clip-linear-probe-experiment \
  --splits-path "${CLIP_OUTPUT_DIR}/label_splits.csv" \
  --embeddings-path "${CLIP_OUTPUT_DIR}/clip_embeddings.npz" \
  --output-dir "${CLIP_OUTPUT_DIR}" \
  --epochs "${CLIP_LINEAR_EPOCHS:-600}" \
  --learning-rate "${CLIP_LINEAR_LEARNING_RATE:-0.1}"

uv run georeset-run-clip-zero-shot-experiment \
  --splits-path "${CLIP_OUTPUT_DIR}/label_splits.csv" \
  --embeddings-path "${CLIP_OUTPUT_DIR}/clip_embeddings.npz" \
  --output-dir "${CLIP_OUTPUT_DIR}" \
  --model-name "${CLIP_MODEL_NAME}" \
  --device "${CLIP_DEVICE:-cuda}"
