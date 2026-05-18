# Vision Package

Reusable helpers for Sentinel-2 patch classification experiments. The package
keeps image fetching, CLIP embedding, zero-shot evaluation, weak-label split
construction, and linear probing separate from CLI orchestration.

## Files

- `__init__.py`: marks the package; keep it free of model or network loading.
- `clip_embedding_cache.py`: shared helpers for loading, validating, and
  stacking cached CLIP embedding artifacts.
- `clip_embeddings.py`: embeds cached Sentinel-2 image patches with frozen CLIP
  image features.
- `clip_transformers.py`: small typed adapters around optional Transformers CLIP
  output objects.
- `clip_weak_labels.py`: builds deterministic CORINE weak-label train/eval
  tiers from quality, relevance, spatial, and model-agreement signals.
- `clip_zero_shot.py`: evaluates zero-shot CLIP text prompts against cached
  Sentinel patch embeddings.
- `linear_probe.py`: small NumPy softmax linear probe for frozen image
  embeddings.
- `sentinel_patches.py`: fetches and caches Sentinel-2 RGB patches around
  article coordinates.

## Design Boundaries

- Keep model and image-processing primitives here.
- Keep command-line parsing in `georeset.cli.data` and `georeset.cli.analysis`.
- Keep cached image arrays and embeddings under experiment artifact storage, not
  in Git.
- Treat frozen CLIP embeddings as reusable artifacts; training code should not
  refetch Sentinel patches or recompute embeddings unless explicitly requested.
