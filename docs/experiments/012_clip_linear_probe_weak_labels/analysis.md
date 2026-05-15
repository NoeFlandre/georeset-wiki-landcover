# CLIP linear probe over weak CORINE labels

## Question

Can the quality filters developed so far produce a useful weak-supervision set
for satellite-image classification?

This experiment tests that directly. Instead of asking an LLM to predict CORINE
from article text, it uses the article coordinates to fetch Sentinel-2 RGB
patches, embeds those patches with frozen CLIP image features, and trains a
small linear probe to predict the CORINE level-2 label.

## Setup

- Experiment id: `clip_linear_probe_weak_labels_v1`
- Image source: Sentinel-2 L2A RGB from Microsoft Planetary Computer
- Date range: `2022-04-01/2022-10-31`
- Cloud filter: `<25`
- Patch size: `224x224`
- Image model: `openai/clip-vit-base-patch32`, frozen image encoder
- Classifier: NumPy softmax linear probe
- Zero-shot baseline: same frozen CLIP model, no training, averaged text prompts
  for each CORINE label description
- Evaluation: fixed strict split, 5 examples per class, 35 examples total

The split builder creates one fixed evaluation set from spatially reliable and
quality-screened articles, then excludes those pageids from every training tier.
This keeps the comparison focused on the training-label policy.

Training tiers:

- `all`: all available CORINE weak labels after excluding eval pageids
- `spatial_only`: `point_label_share_250m >= 0.8` and dominant 250 m label
  matches the point label
- `quality_spatial`: `spatial_only` plus medium/high text relevance, high or
  very-high quality, and no high uncertainty
- `text_spatial_agreement`: `quality_spatial` plus Qwen and Gemma both agreeing
  with the CORINE point label

## Split sizes

| split | tier | examples |
| --- | --- | ---: |
| eval | eval_strict | 35 |
| train | all | 482 |
| train | spatial_only | 357 |
| train | quality_spatial | 184 |
| train | text_spatial_agreement | 142 |

The strictest tier is much smaller and has very few minority-class examples. In
particular, `text_spatial_agreement` has only 1-4 training examples for several
non-forest classes.

## Results

### Linear probe

| tier | train | eval | accuracy | balanced accuracy | macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 482 | 35 | 0.600 | 0.600 | 0.586 |
| spatial_only | 357 | 35 | 0.571 | 0.571 | 0.589 |
| quality_spatial | 184 | 35 | 0.514 | 0.514 | 0.490 |
| text_spatial_agreement | 142 | 35 | 0.400 | 0.400 | 0.363 |

The strongest result is the permissive `all` tier by accuracy and balanced
accuracy. `spatial_only` is nearly tied on macro-F1 and uses fewer examples, so
it remains useful as a conservative alternative. The stricter quality and
text-agreement tiers underperform.

### Out-of-the-box CLIP

| method | train | eval | accuracy | balanced accuracy | macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero-shot CLIP | 0 | 35 | 0.200 | 0.200 | 0.224 |
| frozen CLIP + linear probe, all weak labels | 482 | 35 | 0.600 | 0.600 | 0.586 |
| frozen CLIP + linear probe, spatial-only labels | 357 | 35 | 0.571 | 0.571 | 0.589 |

The trained linear probe is much stronger than out-of-the-box zero-shot CLIP on
the same frozen image embeddings and the same strict evaluation split. The best
linear probe improves balanced accuracy by `+0.400` over zero-shot CLIP and
macro-F1 by `+0.362`.

## Interpretation

This result does not invalidate the text-quality and agreement filters. It shows
that they are too expensive as hard filters for this downstream image model at
the current data size. CLIP linear probing needs broad class coverage. Removing
noisy labels also removes many of the rare labels, and the rare-class data loss
hurts more than the expected label-cleaning benefit.

The image baseline is also intentionally simple: frozen CLIP features and a
linear head. That is a good first test because it is cheap, deterministic, and
easy to debug. The zero-shot result shows that generic CLIP text-image alignment
does not map cleanly onto these Sentinel-2 CORINE classes without local
supervision. But this is still not a full Sentinel-specialized model, so the
result should be read as a weak-supervision data-policy test, not as the ceiling
for satellite patch classification.

## Conclusion

For the next image experiment, do not train only on the strict
`text_spatial_agreement` set. The best next step is to keep a broad training set
and use the quality signals as soft weights or sampling controls, while keeping
the strict set for evaluation or calibration. A Sentinel-native encoder would
also be a stronger follow-up than further tightening the weak-label filters.

## Artifacts

- `data/experiments/clip_linear_probe_weak_labels_v1/label_splits.csv`
- `data/experiments/clip_linear_probe_weak_labels_v1/sentinel_patches_rgb.npz`
- `data/experiments/clip_linear_probe_weak_labels_v1/clip_embeddings.npz`
- `data/experiments/clip_linear_probe_weak_labels_v1/linear_probe_metrics.csv`
- `data/experiments/clip_linear_probe_weak_labels_v1/linear_probe_predictions.csv`
- `data/experiments/clip_linear_probe_weak_labels_v1/zero_shot_clip_metrics.csv`
- `data/experiments/clip_linear_probe_weak_labels_v1/zero_shot_clip_predictions.csv`
- `data/experiments/clip_linear_probe_weak_labels_v1/zero_shot_clip_summary.md`
- `data/experiments/clip_linear_probe_weak_labels_v1/summary.md`

## Artifact inventory

Generated data artifacts are kept under ignored `data/experiments/` storage, not
in git. The local completed run produced:

| artifact | bytes | sha256 |
| --- | ---: | --- |
| `clip_embeddings.npz` | 1,660,360 | `fb8932e791888752b045ddec3a112b5bae5b1ff760ed9e68e7c8cfb5f8c53e52` |
| `label_splits.csv` | 192,574 | `16a56c1c6ce2afc7a95c84ff0a3e016004d49cc4ed93e163886ae184b4b8fcf5` |
| `linear_probe_metrics.csv` | 332 | `97ea65d30715b6290d438939cd59426d44db89187d144737c063c479af4ad723` |
| `linear_probe_predictions.csv` | 3,914 | `fdce79665767dc76f7b6ff2396edfe8a18e259dc63d336308559222bfa0b7fb0` |
| `sentinel_patches_rgb.npz` | 120,147,394 | `797dd39a8ddd5709c59b7c8844c1c5d58163ac6f297ebd609beb3a1012085f49` |
| `summary.md` | 492 | `4016d213e49b4617bd582e84fb483e2b819408e742571dcd42e8202cff87a815` |
| `zero_shot_clip_metrics.csv` | 109 | `6dffedd6396d4387a02107b9f1569ede2eae351e2201083b5d4b09853a6436fa` |
| `zero_shot_clip_predictions.csv` | 1,040 | `e6bee40765fba1270f47728400127e17c946971122803d12c6d310439b83466d` |
| `zero_shot_clip_summary.md` | 118 | `ff87b336a5a864ad98faa2217f94d709d86bfdebea6f91c1e7268245fab062f3` |

Completion checks:

- The Sentinel cache contains 798 unique pageids and patches with shape
  `(798, 224, 224, 3)`.
- The CLIP embedding cache contains 798 pageids and embeddings with shape
  `(798, 512)`.
- The zero-shot baseline uses the same 35-example strict evaluation split as the
  linear-probe comparison.
- No Grid5000 jobs remained active after the run.
