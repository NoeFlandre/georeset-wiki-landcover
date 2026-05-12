#!/usr/bin/env bash
#OAR -q production
#OAR -l host=1/gpu=1,walltime=2:00:00
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

if [ -z "${GEORESET_LANDUSE_EVIDENCE_INPUT_PATH:-}" ]; then
  export GEORESET_LANDUSE_EVIDENCE_INPUT_PATH="data/wiki/article_contents.json"
fi
if [ -z "${GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH:-}" ]; then
  export GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH="data/wiki/article_landuse_evidence_summaries.json"
fi
if [ -z "${GEORESET_LANDUSE_EVIDENCE_SEED:-}" ]; then
  export GEORESET_LANDUSE_EVIDENCE_SEED="42"
fi
if [ -z "${GEORESET_LANDUSE_EVIDENCE_TEMPERATURE:-}" ]; then
  export GEORESET_LANDUSE_EVIDENCE_TEMPERATURE="0.0"
fi

echo "Starting land-use evidence summarization job on $(hostname)"
echo "Input: ${GEORESET_LANDUSE_EVIDENCE_INPUT_PATH}"
echo "Output: ${GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH}"
echo "Model: ${GEORESET_MODEL_PATH}"

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

uv sync --group dev --group llm

MODEL_REPO_ARGS=()
if [ -n "${GEORESET_MODEL_REPO_ID}" ]; then
  MODEL_REPO_ARGS=(--model-repo-id "${GEORESET_MODEL_REPO_ID}")
fi

uv run georeset-summarize-landuse-evidence \
  --input-path "${GEORESET_LANDUSE_EVIDENCE_INPUT_PATH}" \
  --output-path "${GEORESET_LANDUSE_EVIDENCE_OUTPUT_PATH}" \
  --model-path "${GEORESET_MODEL_PATH}" \
  "${MODEL_REPO_ARGS[@]}" \
  --seed "${GEORESET_LANDUSE_EVIDENCE_SEED}" \
  --temperature "${GEORESET_LANDUSE_EVIDENCE_TEMPERATURE}"

