#!/usr/bin/env bash
#OAR -q besteffort
#OAR -l host=1/gpu=1,walltime=10:00:00
#OAR -O /home/nflandre/georeset/OAR_%jobid%.out
#OAR -E /home/nflandre/georeset/OAR_%jobid%.err

set -euo pipefail
cd /home/nflandre/georeset

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export GEORESET_MODEL_PATH="${GEORESET_MODEL_PATH:-Qwen3.6-27B-Q4_0.gguf}"

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

uv sync --all-groups
uv pip install --no-cache-dir huggingface_hub llama-cpp-python

uv run python -m scripts.data.classify_articles \
  --task "${TASK}" \
  --text-source "${TEXT_SOURCE}" \
  --model-path "${GEORESET_MODEL_PATH}"
