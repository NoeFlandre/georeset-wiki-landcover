#!/usr/bin/env bash
#OAR -q production
#OAR -l host=1/gpu=1,walltime=20:00:00
#OAR -p gpu_mem>=32000
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

set -euo pipefail

REMOTE_DIR="${G5K_REMOTE_DIR:-georeset}"
REMOTE_PROJECT_DIR="${G5K_REMOTE_PROJECT_DIR:-${HOME}/${REMOTE_DIR}}"
cd "${REMOTE_PROJECT_DIR}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export GEORESET_MODEL_PATH="${GEORESET_MODEL_PATH:-Qwen3.6-27B-Q4_0.gguf}"
export GEORESET_MODEL_REPO_ID="${GEORESET_MODEL_REPO_ID:-}"

TASK="${GEORESET_CLASSIFICATION_TASK:?Set GEORESET_CLASSIFICATION_TASK}"
TEXT_SOURCE="${GEORESET_CLASSIFICATION_TEXT_SOURCE:?Set GEORESET_CLASSIFICATION_TEXT_SOURCE}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v module >/dev/null 2>&1; then
  module load cuda || true
fi

export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
export UV_PROJECT_ENVIRONMENT="${REMOTE_PROJECT_DIR}/.venv_${OAR_JOB_ID}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"

EXTRA_ARGS=()
if [ -n "${GEORESET_EXTRA_ARGS:-}" ]; then
  read -r -a EXTRA_ARGS <<< "${GEORESET_EXTRA_ARGS}"
fi
MODEL_REPO_ARGS=()
if [ -n "${GEORESET_MODEL_REPO_ID:-}" ]; then
  MODEL_REPO_ARGS=(--model-repo-id "${GEORESET_MODEL_REPO_ID}")
fi

uv sync --group dev --group llm

uv run georeset-classify-articles \
  --task "${TASK}" \
  --text-source "${TEXT_SOURCE}" \
  --model-path "${GEORESET_MODEL_PATH}" \
  "${MODEL_REPO_ARGS[@]}" \
  --temperature "${GEORESET_CLASSIFICATION_TEMPERATURE:-0.0}" \
  "${EXTRA_ARGS[@]}"
