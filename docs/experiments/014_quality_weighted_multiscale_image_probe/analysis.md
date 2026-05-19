# Experiment 014: Quality-Weighted Multiscale Image Probe

## Question

Experiment 012 showed that hard filtering weak labels can hurt a downstream CLIP
linear probe because it removes rare-class training examples. This experiment
tests the next-stage alternative: keep broad training coverage, but weight
examples by quality, relevance, spatial purity, and text-model agreement.

The experiment also tests whether Sentinel-2 physical crop scale matters by
comparing 320m, 640m, 1280m, and 2240m windows resized to the same encoder input
size.

## Staged execution

First run the MVP path on Grid5000:

```bash
IMAGE_PROBE_ENCODERS=clip_base \
IMAGE_PROBE_WINDOWS=320,2240 \
IMAGE_PROBE_RUN_CONTROLS=0 \
./scripts/cluster/submit_quality_weighted_image_probe.sh
```

Then run the full main probe:

```bash
IMAGE_PROBE_ENCODERS=clip_base,clip_large,dinov2_base \
IMAGE_PROBE_WINDOWS=320,640,1280,2240 \
IMAGE_PROBE_RUN_CONTROLS=0 \
./scripts/cluster/submit_quality_weighted_image_probe.sh
```

Run random training controls after the main probe identifies relevant
encoder/window/policy combinations:

```bash
IMAGE_PROBE_RUN_CONTROLS=1 ./scripts/cluster/submit_quality_weighted_image_probe.sh
```

## Key safeguards

- Evaluation rows are selected from `quality_spatial`.
- Qwen/Gemma agreement is used only for training weights and the
  `text_spatial_agreement` training tier.
- The main evaluation set is not selected by text-model agreement.
- Generated artifacts are under `data/experiments/...` and must not be committed.

## Metrics

Headline metrics are:

- `balanced_accuracy_supported`
- `macro_f1_supported`

Continuity metrics retained for comparison with prior reports:

- `balanced_accuracy_allowed`
- `macro_f1_allowed`

Supported metrics average only over labels present in the evaluation set.
Allowed metrics average over the full allowed label universe, including labels
with zero support in that evaluation split.

## Scale audit fields

Each Sentinel-2 NPZ patch cache stores:

- `window_m`
- `source_pixels`
- `native_pixel_size_m`
- `resize_method`
- `output_size`
- `stac_item_id`
- `eo_cloud_cover`

These fields make the physical crop-scale comparison auditable.

## Artifacts

- `image_probe_splits_v2.csv`
- `sample_weights.csv`
- `split_manifest.json`
- `sentinel_rgb_window_0320m.npz`, etc.
- `embeddings_<encoder>_window_<window>m.npz`
- `weighted_probe_metrics.csv`
- `weighted_probe_predictions.csv`
- `per_class_metrics.csv`
- `bootstrap_confidence_intervals.csv`
- `confusion_matrices.json`
- `image_probe_random_training_controls.csv`

The final results section should be filled only after generated artifacts are
available and inspected.
