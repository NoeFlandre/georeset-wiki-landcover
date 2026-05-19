#!/usr/bin/env bash
#OAR -n georeset_quality_weighted_image_probe
#OAR -q production
#OAR -l host=1/gpu=1,walltime=48:00:00
#OAR -p gpu_mem>=32000
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${HOME}/${REMOTE_DIR}}"
OUTPUT_DIR="${IMAGE_PROBE_OUTPUT_DIR:-data/experiments/014_quality_weighted_multiscale_image_probe/quality_weighted_multiscale_image_probe_v1}"
WINDOWS="${IMAGE_PROBE_WINDOWS:-320,2240}"
ENCODERS="${IMAGE_PROBE_ENCODERS:-clip_base}"
RUN_CONTROLS="${IMAGE_PROBE_RUN_CONTROLS:-0}"
DEVICE="${IMAGE_PROBE_DEVICE:-cuda}"
BATCH_SIZE="${IMAGE_PROBE_BATCH_SIZE:-32}"
EPOCHS="${IMAGE_PROBE_EPOCHS:-600}"
N_BOOTSTRAP="${IMAGE_PROBE_N_BOOTSTRAP:-1000}"
N_CONTROL_DRAWS="${IMAGE_PROBE_N_CONTROL_DRAWS:-100}"

cd "${REMOTE_PROJECT_DIR}"

export PYTHONDONTWRITEBYTECODE=1
JOB_CACHE_DIR="${GEORESET_JOB_CACHE_DIR:-${TMPDIR:-/tmp}/georeset_${OAR_JOB_ID:-manual}}"
export HF_HOME="${HF_HOME:-${JOB_CACHE_DIR}/hf}"
export TORCH_HOME="${TORCH_HOME:-${JOB_CACHE_DIR}/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${JOB_CACHE_DIR}/uv}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${UV_CACHE_DIR}" "${OUTPUT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

uv sync --group dev --group vision

uv run georeset-build-image-probe-splits-v2 --output-dir "${OUTPUT_DIR}"
uv run georeset-fetch-sentinel-multiscale-patches --output-dir "${OUTPUT_DIR}" --window-m "${WINDOWS}"

IFS=',' read -r -a WINDOW_ARRAY <<< "${WINDOWS}"
IFS=',' read -r -a ENCODER_ARRAY <<< "${ENCODERS}"
for window_m in "${WINDOW_ARRAY[@]}"; do
  window_padded="$(printf "%04d" "${window_m}")"
  patches_path="${OUTPUT_DIR}/sentinel_rgb_window_${window_padded}m.npz"
  for encoder in "${ENCODER_ARRAY[@]}"; do
    uv run georeset-embed-image-patches \
      --patches-path "${patches_path}" \
      --output-path "${OUTPUT_DIR}/embeddings_${encoder}_window_${window_padded}m.npz" \
      --encoder "${encoder}" \
      --device "${DEVICE}" \
      --batch-size "${BATCH_SIZE}"
  done
done

uv run georeset-run-quality-weighted-image-probe \
  --output-dir "${OUTPUT_DIR}" \
  --encoders "${ENCODERS}" \
  --windows "${WINDOWS}" \
  --epochs "${EPOCHS}" \
  --n-bootstrap "${N_BOOTSTRAP}"

if [[ "${RUN_CONTROLS}" == "1" ]]; then
  uv run georeset-evaluate-image-probe-training-policy-controls \
    --output-dir "${OUTPUT_DIR}" \
    --encoders "${ENCODERS}" \
    --windows "${WINDOWS}" \
    --epochs "${EPOCHS}" \
    --n-draws "${N_CONTROL_DRAWS}"
fi
