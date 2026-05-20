# LLM Package

This package contains shared local-LLM infrastructure.

## Files

- `__init__.py`: exports shared local-LLM helpers without loading a model.
- `llama_client.py`: lazy `llama-cpp-python` JSON chat client used by article
  summarization, evidence extraction, and classification.

## Model Identity

The client supports two loading modes:

- Hugging Face Hub GGUF loading with `repo_id` and `filename`.
- Local GGUF loading when `model_path` points to an existing file.

Classification fingerprints and metadata record model path/filename, optional
model repo ID, seed, temperature, task, text source, allowed labels, and
classification policy version. This keeps model-family comparisons isolated and
auditable.

## Dependency Boundary

`llama-cpp-python` is in the optional `llm` dependency group. CI and normal
development use `uv sync --group dev`; GPU/LLM jobs use
`uv sync --group dev --group llm`.
