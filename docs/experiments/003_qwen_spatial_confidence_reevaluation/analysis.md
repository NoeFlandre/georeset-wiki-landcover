# Article-Text Classification Spatial Confidence v1: Verified Analysis

This note analyzes the frozen outputs in:

- `data/experiments/corine_spatial_confidence_v1/`
- `data/experiments/article_text_classification_spatial_confidence_v1/`

The analysis is based on the generated CSV artifacts only. No LLM predictions were rerun.

## 1. Spatial Confidence Worked and Reveals Noisy Supervision

The spatial-confidence table contains 1,251 articles. The subset sizes confirm that point-in-polygon CORINE labels are often spatially fragile:

| Spatial condition | Articles kept | Share kept |
| --- | ---: | ---: |
| all available spatial confidence | 1,251 | 100.0% |
| dominant matches point label at 250 m | 1,082 | 86.5% |
| dominant matches point label at 500 m | 968 | 77.4% |
| point label share at 250 m >= 0.8 | 646 | 51.6% |
| point label share at 250 m >= 0.9 | 518 | 41.4% |
| point label share at 500 m >= 0.8 | 419 | 33.5% |

Only about half of the points are clean CORINE examples at 250 m under the `point_label_share >= 0.8` rule, and only about one third are clean at 500 m. This is an important result by itself: Wikipedia point coordinates are noisy supervision for land-cover labels unless spatial reliability is measured or filtered.

## 2. High Spatial Confidence Improves CORINE Text Signal

For `corine_level2/content`, high-purity subsets improve balanced accuracy and increase the aligned-vs-shuffled gap:

| Subset | n | Accuracy | Balanced accuracy | Macro-F1 | Shuffled balanced accuracy | Delta vs shuffled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all available spatial confidence | 1,251 | 0.293 | 0.281 | 0.270 | 0.067 | +0.213 |
| point label share 250 m >= 0.8 | 646 | 0.350 | 0.352 | 0.285 | 0.071 | +0.281 |
| point label share 250 m >= 0.9 | 518 | 0.351 | 0.354 | 0.270 | 0.070 | +0.284 |
| point label share 500 m >= 0.8 | 419 | 0.334 | 0.302 | 0.213 | 0.067 | +0.235 |

The key comparison is not only that aligned content improves from balanced accuracy `0.281` to `0.352` on the 250 m high-purity subset. It is also that shuffled content stays low, around `0.071`. That pattern supports the interpretation that article text contains real land-cover signal, and the signal becomes clearer when the spatial ground truth is more reliable.

## 3. Raw Accuracy Still Loses to the Majority Baseline Because Filtering Increases Imbalance

For `corine_level2/content`, raw accuracy remains below the majority-class accuracy baseline:

| Subset | Model accuracy | Majority accuracy | Delta |
| --- | ---: | ---: | ---: |
| all available spatial confidence | 0.293 | 0.379 | -0.086 |
| point label share 250 m >= 0.8 | 0.350 | 0.502 | -0.152 |
| point label share 250 m >= 0.9 | 0.351 | 0.569 | -0.218 |
| point label share 500 m >= 0.8 | 0.334 | 0.659 | -0.325 |

This is explained by changing class balance. High-purity filtering makes the subset increasingly forest-dominated:

| Subset | Forest support | Forest share |
| --- | ---: | ---: |
| all available spatial confidence | 474 | 37.9% |
| point label share 250 m >= 0.8 | 324 | 50.2% |
| point label share 250 m >= 0.9 | 295 | 56.9% |
| point label share 500 m >= 0.8 | 276 | 65.9% |

Raw accuracy is therefore not the right primary metric for CORINE here. It rewards majority-class concentration too strongly, especially after spatial filtering.

## 4. The Model Beats Majority on Balanced Metrics

The more meaningful CORINE comparison is against majority baselines under balanced accuracy and macro-F1:

| Subset | Delta vs majority accuracy | Delta vs majority balanced accuracy | Delta vs majority macro-F1 |
| --- | ---: | ---: | ---: |
| all available spatial confidence | -0.086 | +0.170 | +0.209 |
| point label share 250 m >= 0.8 | -0.152 | +0.241 | +0.211 |
| point label share 250 m >= 0.9 | -0.218 | +0.243 | +0.190 |
| point label share 500 m >= 0.8 | -0.325 | +0.191 | +0.125 |

The correct reading is not that the model fails because it loses to majority accuracy. The correct reading is that the model loses under raw accuracy because of class imbalance, but substantially beats majority under balanced accuracy and macro-F1. The text signal is real, but uneven across classes.

## 5. Raw Content Is Stronger Than the Existing Summaries

For CORINE, raw content is the strongest text source in the main settings:

| Subset | Text source | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| all available spatial confidence | summary | 0.237 | 0.255 | 0.244 |
| all available spatial confidence | summary no place | 0.232 | 0.243 | 0.235 |
| all available spatial confidence | content | 0.293 | 0.281 | 0.270 |
| point label share 250 m >= 0.8 | summary | 0.268 | 0.294 | 0.241 |
| point label share 250 m >= 0.8 | summary no place | 0.262 | 0.286 | 0.236 |
| point label share 250 m >= 0.8 | content | 0.350 | 0.352 | 0.285 |

This suggests that the current generic summaries remove useful land-cover evidence. A likely next experiment is not only a stronger classifier, but a better extraction step: a `landscape_evidence_summary` designed to preserve cues about terrain, agriculture, forests, water, and named landscape features.

## 6. Per-Class Behavior Is Uneven and Scientifically Informative

For `corine_level2/content`, comparing all articles to `point_label_share_250m >= 0.8`:

| Class | Meaning | Support all -> high purity | F1 all -> high purity | Interpretation |
| --- | --- | ---: | ---: | --- |
| 21 | Arable land | 209 -> 100 | 0.292 -> 0.310 | Slightly better, still moderate |
| 22 | Permanent crops | 137 -> 81 | 0.673 -> 0.811 | Very strong |
| 23 | Pastures | 156 -> 50 | 0.117 -> 0.044 | Very weak |
| 24 | Heterogeneous agriculture | 183 -> 65 | 0.208 -> 0.225 | Still weak |
| 31 | Forests | 474 -> 324 | 0.381 -> 0.451 | Better |
| 32 | Shrub/herbaceous | 50 -> 14 | 0.327 -> 0.391 | Better, but low support |
| 51 | Inland waters | 42 -> 12 | 0.431 -> 0.333 | High recall, unstable precision |

The strongest classes are permanent crops, forests, water, and shrub/herbaceous classes, with support caveats for the rarer classes. The weakest classes are pastures and heterogeneous agricultural areas. This is plausible: Wikipedia text often names forests, vineyards, rivers, lakes, mountains, or natural areas, but generic arable land and pasture are less likely to be described in article text in a way that maps cleanly to CORINE.

The research claim should therefore be class-aware: geolocated text is not uniformly useful for all land-cover classes. It is more useful for semantically salient and text-described landscape classes than for generic agricultural land-cover classes.

## 7. The 500 m High-Purity Subset Is Useful as a Diagnostic, Not as the Main Subset

The `point_label_share_500m >= 0.8` subset keeps only 419 articles and is heavily forest-dominated: 276 of 419 articles are class 31, or 65.9%. Its balanced accuracy remains above shuffled, but macro-F1 drops to `0.213`.

The better main high-confidence subset is `point_label_share_250m >= 0.8`. It improves balanced accuracy and shuffled deltas while retaining 646 articles and more class diversity.

## 8. OSM Also Improves on Spatially Reliable CORINE Subsets, but Remains Hard

The CORINE-derived spatial-confidence filter also improves OSM content metrics:

| Subset | n | Exact match | Micro-F1 | Macro-F1 | Jaccard | Majority label-set exact match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all available spatial confidence | 275 | 0.164 | 0.197 | 0.158 | 0.190 | 0.207 |
| point label share 250 m >= 0.8 | 160 | 0.225 | 0.249 | 0.204 | 0.245 | 0.244 |
| point label share 500 m >= 0.8 | 124 | 0.218 | 0.237 | 0.159 | 0.237 | 0.282 |

Exact-match accuracy is still below the most-frequent-label-set baseline, including on the 250 m high-purity subset (`0.225` vs `0.244`). However, aligned content is much stronger than shuffled content. On `point_label_share_250m >= 0.8`, OSM content exact match is `0.225`, shuffled content exact match is `0.063`, and the delta is `+0.163`.

The OSM task is therefore not solved, but the text signal exists. Exact-match multi-label evaluation is harsh, so micro-F1, macro-F1, Jaccard, and per-label behavior should be read alongside exact match.

## 9. Scientific Conclusion

The spatial-confidence experiment supports a stronger and more accurate interpretation than the earlier raw-majority framing:

Wikipedia article text contains useful land-cover information, but the signal is clearer when spatial ground truth is reliable. Point-in-polygon CORINE labels are noisy, and high-purity CORINE neighborhoods produce stronger aligned-vs-shuffled performance. The model does not beat majority under raw accuracy because high-confidence subsets are increasingly majority-class dominated, especially by forests. But it does beat majority under balanced accuracy and macro-F1, especially for raw content.

The main limitations are class imbalance, generic summaries, article relevance, and land-cover classes that are not naturally expressed in Wikipedia text. The next high-impact experiment should focus on extracting landscape evidence from article text rather than relying on generic summaries.
