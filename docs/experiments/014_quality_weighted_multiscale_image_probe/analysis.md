# Experiment 014: Quality-Weighted Multiscale Image Probe

## Question

Experiment 012 showed that hard filtering weak CORINE labels can hurt a
downstream CLIP linear probe because it removes rare-class training examples.
Experiment 014 tests the next alternative:

- keep broad image-training coverage;
- encode Sentinel-2 crops at explicit physical scales;
- use article relevance, spatial purity, supervision quality, and Qwen/Gemma
  agreement as soft training weights;
- compare soft weighting with hard-filtered tiers and broad unweighted training;
- keep the same 35-example strict evaluation design used by Experiment 012.

This is an image experiment, not a new LLM experiment. It does not rerun prompts,
change labels, or change article predictions.

## Critical Implementation Note

The first MVP run was invalid. The multiscale patch fetcher passed WGS84
longitude/latitude directly to `rasterio.dataset.index()`, but Sentinel-2 raster
assets use projected raster CRSs. Because reads were `boundless=True`, the wrong
coordinates silently produced all-black patches.

The bug was fixed by reprojecting each article point from `EPSG:4326` into the
asset CRS before calling `dataset.index()`. The corrected run also adds patch
validation artifacts before embeddings are computed:

- `patch_stats.csv`
- `patch_contact_sheet.png`
- `patch_validation_manifest.json`

The corrected validation passed:

| window | patches | source pixels | all-zero patches | mean pixel | pixel std |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320 m | 1,251 | 32 | 0 | 132.788 | 34.880 |
| 2240 m | 1,251 | 224 | 0 | 128.356 | 38.231 |

The 320 m and 2240 m arrays are not identical: mean absolute pixel difference is
`26.758`, and the contact sheet visually confirms centered, non-black crops at
both scales.

## MVP Scope

The completed corrected MVP used:

- encoder: `openai/clip-vit-base-patch32`;
- windows: 320 m and 2240 m;
- Sentinel-2 bands: B04/B03/B02;
- native pixel size: 10 m;
- output size: 224 x 224;
- resize method: bilinear;
- strict evaluation split: 35 examples, 5 per supported CORINE class;
- training rows after excluding strict eval: 1,216 for broad policies;
- linear-probe epochs: 600;
- L2 grid: `1e-5`, `1e-4`, `1e-3`, `1e-2`;
- bootstrap intervals: 1,000 resamples.

Qwen/Gemma agreement is used only as a training signal, not as the main
evaluation selector. The strict evaluation set is selected from the
quality-spatial subset, independent of text-model agreement.

## Zero-Shot Baseline

Zero-shot CLIP is weak on the strict evaluation split:

| window | accuracy | supported balanced accuracy | supported macro-F1 |
| ---: | ---: | ---: | ---: |
| 320 m | 0.200 | 0.200 | 0.152 |
| 2240 m | 0.200 | 0.200 | 0.224 |

This confirms that the trained linear probe is adding information beyond
out-of-the-box CLIP text-image alignment.

## Strict Evaluation Results

Headline metric: `balanced_accuracy_supported`. The allowed-label metrics are
identical here because the strict split contains support for the same seven
labels used in this MVP comparison.

Best policy per window:

| window | best policy | L2 | train n | accuracy | supported balanced accuracy | supported macro-F1 | 95% CI balanced accuracy | 95% CI macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 320 m | `text_agreement_soft_weighted` | 0.01 | 1,216 | 0.686 | 0.686 | 0.663 | 0.560-0.817 | 0.507-0.785 |
| 2240 m | `text_agreement_soft_weighted` | 0.00001 | 1,216 | 0.686 | 0.686 | 0.685 | 0.538-0.837 | 0.505-0.827 |

Best strict-split score by training policy:

| window | policy | train n | supported balanced accuracy | supported macro-F1 |
| ---: | --- | ---: | ---: | ---: |
| 320 m | `text_agreement_soft_weighted` | 1,216 | 0.686 | 0.663 |
| 320 m | `all_quality_weighted_class_balanced` | 1,216 | 0.600 | 0.569 |
| 320 m | `all_quality_weighted` | 1,216 | 0.571 | 0.524 |
| 320 m | `all_unweighted` | 1,216 | 0.543 | 0.502 |
| 320 m | `spatial_only_unweighted` | 611 | 0.543 | 0.520 |
| 320 m | `quality_spatial_hard` | 286 | 0.400 | 0.344 |
| 320 m | `agreement_hard` | 152 | 0.429 | 0.358 |
| 2240 m | `text_agreement_soft_weighted` | 1,216 | 0.686 | 0.685 |
| 2240 m | `all_unweighted` | 1,216 | 0.686 | 0.677 |
| 2240 m | `spatial_soft_weighted` | 1,216 | 0.657 | 0.642 |
| 2240 m | `all_quality_weighted_class_balanced` | 1,216 | 0.629 | 0.617 |
| 2240 m | `all_quality_weighted` | 1,216 | 0.600 | 0.588 |
| 2240 m | `quality_spatial_hard` | 286 | 0.543 | 0.505 |
| 2240 m | `spatial_only_unweighted` | 611 | 0.543 | 0.520 |
| 2240 m | `agreement_hard` | 152 | 0.400 | 0.363 |

## Interpretation

The corrected MVP reverses the invalid first conclusion. The pipeline is not
collapsing to chance. With valid Sentinel crops, the trained CLIP linear probe
beats zero-shot CLIP by a large margin:

- best trained 320 m: `0.686` supported balanced accuracy versus zero-shot
  `0.200`;
- best trained 2240 m: `0.686` supported balanced accuracy versus zero-shot
  `0.200`;
- 2240 m gives the strongest strict-split macro-F1 (`0.685`), slightly above
  320 m (`0.663`).

The most important modeling result is about hard filters. Hard-filtered training
tiers still underperform broad training:

- `quality_spatial_hard` uses only 286 training rows and reaches `0.400` at
  320 m and `0.543` at 2240 m;
- `agreement_hard` uses only 152 training rows and reaches `0.429` at 320 m and
  `0.400` at 2240 m;
- the best policies keep all 1,216 train rows and use soft weights.

This supports the Experiment 014 design premise: for this data size, quality and
agreement signals are more useful as weights than as hard filters.

The scale result is mixed but informative. Both windows tie on strict balanced
accuracy, but 2240 m has slightly better macro-F1 and stronger broad unweighted
performance. The wider crop likely provides useful land-cover context for some
classes, but the MVP is too small to claim a definitive scale winner.

## Robustness Notes

Repeated evaluation splits are much higher on average than the fixed strict
split, and spatial-block folds are lower:

- repeated split means for `text_agreement_soft_weighted`: about `0.893` at
  320 m and `0.900` at 2240 m supported balanced accuracy;
- spatial-block means for the same policy: about `0.495` at 320 m and `0.533`
  at 2240 m.

This gap means the fixed strict split is not the whole story. Random repeated
splits can be optimistic, while spatial-block folds expose geographic
generalization difficulty. The final publication-grade claim should therefore
report all three views: strict comparability, repeated split stability, and
spatial-block generalization.

## Current Conclusion

Experiment 014 is now operational and scientifically useful after the CRS fix.
The corrected MVP shows:

1. valid Sentinel-2 patch extraction at both scales;
2. non-identical, normalized CLIP embeddings for 320 m and 2240 m;
3. zero-shot CLIP remains weak;
4. trained CLIP linear probes reach `0.686` strict supported balanced accuracy;
5. soft weighting is safer than hard filtering;
6. 2240 m is at least competitive with 320 m and slightly stronger on macro-F1.

Next, run the planned full main grid: `clip_base`, `clip_large`, and
`dinov2_base` across 320 m, 640 m, 1280 m, and 2240 m. Random training controls
should be run only after the full grid identifies the encoder/window/policy
comparisons that matter.
