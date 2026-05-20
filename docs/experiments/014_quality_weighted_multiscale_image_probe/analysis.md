# Experiment 014: Quality-Weighted Multiscale Image Probe

## Executive Summary

Experiment 014 is the corrected multiscale Sentinel-2 image-probe experiment.
It asks whether weak labels derived from GeoReset's Wikipedia/CORINE pipeline can
train a useful image classifier when quality signals are used as soft weights
instead of strict hard filters.

The corrected MVP result is positive:

- zero-shot CLIP reaches only `0.200` supported balanced accuracy at both 320 m
  and 2240 m;
- the best trained CLIP linear probe reaches `0.686` supported balanced accuracy
  at both scales;
- 2240 m has the strongest strict-split macro-F1 (`0.685`);
- hard-filtered tiers remain weaker than broad training because they remove too
  many training examples;
- soft text-agreement weighting helps clearly at 320 m, but at 2240 m it
  essentially ties broad all-unweighted training.

The important conclusion is therefore not "soft weighting always wins." The
stronger and better-supported conclusion is:

> Broad training coverage is essential. Quality, relevance, spatial confidence,
> and text-model agreement are useful training signals, but at this data size
> they should not be used mainly as hard filters.

## Why This Experiment Exists

Experiment 012 was the first downstream image experiment. It fetched Sentinel-2
RGB patches around geolocated Wikipedia articles, embedded them with frozen
`openai/clip-vit-base-patch32`, and trained a simple linear classifier to predict
CORINE level-2 land-cover classes.

Experiment 012 found a useful signal: broad weak-label training reached `0.600`
balanced accuracy on the fixed 35-example strict evaluation split, compared with
`0.200` for zero-shot CLIP.

But it also showed a problem. The cleanest hard-filtered training tier performed
worse because it removed too many examples, especially minority classes. That
made the next research question clear:

Can we preserve broad training coverage while still using relevance, spatial
confidence, quality, and Qwen/Gemma agreement to reduce weak-label noise?

Experiment 014 tests that idea.

## What Was Performed

The corrected MVP run used:

- task: CORINE level-2 single-label image classification;
- image source: Sentinel-2 L2A RGB crops from Microsoft Planetary Computer;
- bands: B04/B03/B02;
- encoder: frozen `openai/clip-vit-base-patch32`;
- image embedding size: 512;
- crop windows: 320 m and 2240 m;
- Sentinel native pixel size: 10 m;
- source pixels:
  - 320 m -> 32 source pixels;
  - 2240 m -> 224 source pixels;
- resize: bilinear to 224 x 224;
- strict eval split: 35 examples, 5 per supported CORINE class;
- broad train rows after excluding strict eval: 1,216;
- linear probe: NumPy softmax classifier on frozen CLIP embeddings;
- training epochs: 600;
- L2 grid: `1e-5`, `1e-4`, `1e-3`, `1e-2`;
- bootstrap intervals: 1,000 resamples.

The L2 grid is exploratory in this MVP: there is no separate validation split,
so "best" rows below mean best observed evaluation rows, not a final
hyperparameter-selected estimate.

No LLMs were rerun. No prompts, text sources, or labels were changed. Qwen/Gemma
predictions were reused only as frozen metadata for training weights and
training-tier definitions.

## The Critical CRS Bug And Fix

The first MVP run was invalid. The multiscale patch fetcher passed WGS84
longitude/latitude directly to `rasterio.dataset.index()`. Sentinel-2 raster
assets are stored in projected raster coordinate reference systems, so
`dataset.index()` expects coordinates in the raster asset CRS, not WGS84.

Because patch reads used `boundless=True`, the bug did not crash. It silently
returned fill pixels, which made the Sentinel patches all black. The previous
`0.143` result must therefore be ignored.

The fix reprojects every article point from `EPSG:4326` into the raster asset CRS
before calling `dataset.index()`. We also added three safeguards:

- all-zero newly fetched patches fail immediately;
- all-zero existing patch caches fail immediately;
- each patch run writes numeric and visual validation artifacts before embedding.

Corrected validation:

| Window | Patches | Source pixels | All-zero patches | Mean pixel | Pixel std |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 320 m | 1,251 | 32 | 0 | 132.788 | 34.880 |
| 2240 m | 1,251 | 224 | 0 | 128.356 | 38.231 |

The 320 m and 2240 m patch arrays differ for the same pageids:

```text
mean absolute patch difference = 26.758
exact equal patch count = 0
```

The contact sheet also confirms the expected visual pattern: 320 m crops show
local detail, while 2240 m crops show broader landscape context.

## What The Classifier Does

The classifier predicts CORINE level-2 land-cover labels from Sentinel-2 image
crops. Text is not used at inference time.

The image pipeline is:

1. take a geolocated Wikipedia article coordinate;
2. fetch a Sentinel-2 RGB crop around that coordinate;
3. resize the physical crop to 224 x 224;
4. embed the image with frozen CLIP;
5. train a linear classifier on top of the frozen image embedding;
6. predict the CORINE level-2 label.

The text-derived signals enter only during training-policy construction. They
help decide how much to weight a training example or whether it belongs to a hard
training tier.

## What Soft Weighting Means

A hard filter makes a binary decision:

```text
keep this example / drop this example
```

Soft weighting keeps broad coverage but changes how much each example
contributes to the training loss:

```text
cleaner examples count more / noisier examples count less
```

This matters because the dataset is small and class-imbalanced. Hard filters can
remove noisy labels, but they can also remove rare-class examples and visual
diversity. A small clean set can be worse than a larger slightly noisy set.

Experiment 014 compares both strategies directly.

## Weight Signals

The sample weights use four families of metadata:

1. **Land-cover relevance**

   Articles with medium/high explicit land-cover evidence are more trusted than
   articles with none/low evidence.

2. **Uncertainty**

   High uncertainty downweights examples.

3. **Spatial confidence**

   Higher `point_label_share_250m` indicates that the article coordinate sits in
   a locally coherent CORINE region.

4. **Qwen/Gemma agreement**

   If both frozen text classifiers predict the same CORINE label as the spatial
   point label, the example receives stronger training support.

Important: Qwen/Gemma agreement is not used to select the strict evaluation set.
It is used only for training weights and training tiers. This avoids selecting
the evaluation set with the same text models whose metadata is being tested.

## Training Policies

The experiment compares broad training, hard filters, and soft weighting.

| Policy | Type | Train rows | Meaning |
| --- | --- | ---: | --- |
| `all_unweighted` | broad baseline | 1,216 | Use every train row equally. |
| `spatial_only_unweighted` | hard filter | 611 | Keep only spatially coherent rows. |
| `quality_spatial_hard` | hard filter | 286 | Keep spatially coherent, relevant, high-quality, low-uncertainty rows. |
| `agreement_hard` | hard filter | 152 | Keep only rows where Qwen, Gemma, and CORINE agree. |
| `all_quality_weighted` | soft weight | 1,216 | Use all rows with relevance/spatial/quality/agreement weights. |
| `all_quality_weighted_class_balanced` | soft weight | 1,216 | Same but normalized within classes. |
| `spatial_soft_weighted` | soft weight | 1,216 | Use all rows but emphasize spatially coherent rows. |
| `text_agreement_soft_weighted` | soft weight | 1,216 | Use all rows but emphasize text agreement plus spatial/relevance signals. |

The hard-filter rows show the core risk: `agreement_hard` is clean but has only
152 training examples. That is too little for robust class coverage.

## Zero-Shot CLIP Baseline

Zero-shot CLIP means no training on GeoReset Sentinel/CORINE data. The model
classifies images using CLIP text-image similarity to class prompts.

| Window | Accuracy | Supported balanced accuracy | Supported macro-F1 |
| ---: | ---: | ---: | ---: |
| 320 m | 0.200 | 0.200 | 0.152 |
| 2240 m | 0.200 | 0.200 | 0.224 |

Zero-shot CLIP is weak for this task. That is expected: CORINE level-2
remote-sensing labels are specialized, and the images are Sentinel-2 overhead
crops rather than ordinary web photos.

## Strict Evaluation Results

The headline metric is `balanced_accuracy_supported`, which averages recall only
over labels present in the evaluation split. In this strict split, it matches the
allowed-label balanced accuracy because all seven supported labels are present.

Best policy per window:

| Window | Best policy | L2 | Train rows | Accuracy | Supported balanced accuracy | Supported macro-F1 | Balanced-accuracy 95% CI | Macro-F1 95% CI |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 320 m | `text_agreement_soft_weighted` | 0.01 | 1,216 | 0.686 | 0.686 | 0.663 | 0.560-0.817 | 0.507-0.785 |
| 2240 m | `text_agreement_soft_weighted` | 0.00001 | 1,216 | 0.686 | 0.686 | 0.685 | 0.538-0.837 | 0.505-0.827 |

These results are far above zero-shot CLIP:

| Method | 320 m balanced accuracy | 2240 m balanced accuracy |
| --- | ---: | ---: |
| zero-shot CLIP | 0.200 | 0.200 |
| best trained linear probe | 0.686 | 0.686 |

## Policy Comparison

### 320 m

| Policy | Train rows | Supported balanced accuracy | Supported macro-F1 |
| --- | ---: | ---: | ---: |
| `text_agreement_soft_weighted` | 1,216 | 0.686 | 0.663 |
| `all_quality_weighted_class_balanced` | 1,216 | 0.600 | 0.569 |
| `all_quality_weighted` | 1,216 | 0.571 | 0.524 |
| `all_unweighted` | 1,216 | 0.543 | 0.502 |
| `spatial_only_unweighted` | 611 | 0.543 | 0.520 |
| `spatial_soft_weighted` | 1,216 | 0.543 | 0.534 |
| `agreement_hard` | 152 | 0.429 | 0.358 |
| `quality_spatial_hard` | 286 | 0.400 | 0.344 |

At 320 m, soft text-agreement weighting clearly helps over all-unweighted
training: `0.686` versus `0.543` supported balanced accuracy.

### 2240 m

| Policy | Train rows | Supported balanced accuracy | Supported macro-F1 |
| --- | ---: | ---: | ---: |
| `text_agreement_soft_weighted` | 1,216 | 0.686 | 0.685 |
| `all_unweighted` | 1,216 | 0.686 | 0.677 |
| `spatial_soft_weighted` | 1,216 | 0.657 | 0.642 |
| `all_quality_weighted_class_balanced` | 1,216 | 0.629 | 0.617 |
| `all_quality_weighted` | 1,216 | 0.600 | 0.588 |
| `quality_spatial_hard` | 286 | 0.543 | 0.505 |
| `spatial_only_unweighted` | 611 | 0.543 | 0.520 |
| `agreement_hard` | 152 | 0.400 | 0.363 |

At 2240 m, the story is different. `all_unweighted` ties
`text_agreement_soft_weighted` on supported balanced accuracy (`0.686`) and is
only slightly lower on macro-F1 (`0.677` versus `0.685`). This means we should
not claim that soft weighting universally improves the classifier.

The robust conclusion is that broad coverage matters most; soft weighting can
help, especially at local scale, but the clear loser is aggressive hard
filtering.

## Scale Interpretation

The strict split does not show a balanced-accuracy winner between 320 m and
2240 m:

```text
320 m best balanced accuracy  = 0.686
2240 m best balanced accuracy = 0.686
```

But 2240 m has slightly stronger macro-F1:

```text
320 m best macro-F1  = 0.663
2240 m best macro-F1 = 0.685
```

The wider window also performs very well without any soft weighting:

```text
2240 m all_unweighted balanced accuracy = 0.686
2240 m all_unweighted macro-F1          = 0.677
```

This suggests that broader physical context may make the task easier for CLIP
features. But the MVP has only two scales and one encoder, so the correct next
step is the planned full grid rather than a scale claim.

## Repeated And Spatial-Block Results

Repeated evaluation splits are alternative quality-spatial samples with their
own pageids excluded from training before fitting. The split CSV now writes
explicit `train_for_<eval_split>` rows so that downstream scripts do not need to
infer this exclusion rule from runner code. For
`text_agreement_soft_weighted`, the corrected leakage-guarded means are:

| Window | Repeated mean supported balanced accuracy | Repeated mean supported macro-F1 |
| ---: | ---: | ---: |
| 320 m | 0.631 | 0.619 |
| 2240 m | 0.631 | 0.625 |

Spatial-block folds are lower:

| Window | Spatial-block mean supported balanced accuracy | Spatial-block mean supported macro-F1 |
| ---: | ---: | ---: |
| 320 m | 0.467 | 0.472 |
| 2240 m | 0.537 | 0.524 |

This is an important caution. Random repeated splits are no longer inflated by
direct train/eval pageid overlap, but they can still be easier than geographic
holdout because nearby places may appear in both train and evaluation.
Spatial-block folds test a harder form of generalization. They show that the
classifier has learned useful visual signal, but geographic generalization is
still not solved.

## What Claims Are Safe

Safe claims:

- The first black-patch run was invalid and has been corrected.
- Corrected Sentinel-2 crops are non-black and scale-specific.
- Trained CLIP linear probes beat zero-shot CLIP strongly on the strict split.
- Hard filters underperform broad training in this MVP.
- Soft text-agreement weighting helps clearly at 320 m.
- At 2240 m, all-unweighted broad training is essentially tied with the best
  soft-weighted policy.
- Repeated-split metrics now exclude each eval split's pageids from training.
- Spatial-block performance is lower than random repeated split
  performance, so geographic generalization remains a major concern.

Claims to avoid:

- "Soft weighting universally wins."
- "2240 m is definitively the best scale."
- "The classifier is publication-ready."
- "Random split performance alone proves generalization."
- "The best L2/policy rows are final model-selection estimates."

## Current Conclusion

Experiment 014 is now a valid, useful image experiment. The corrected MVP shows
that weakly supervised Sentinel-2 image probing is viable when the patch
extraction is correct. It also sharpens the lesson from Experiment 012: do not
throw away too much data. At this scale, broad weak-label coverage is more
important than strict hard filtering.

The best next experiment is the full main grid:

- encoders: `clip_base`, `clip_large`, `dinov2_base`;
- windows: 320 m, 640 m, 1280 m, 2240 m;
- policies: broad unweighted, soft weighted, class-balanced weighted, and hard
  tiers;
- then random training controls for the comparisons that matter.
