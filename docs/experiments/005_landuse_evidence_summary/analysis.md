# Land-use Evidence Summary Experiment Comparison

## Source Artifacts
- Qwen land-use experiment: `data/experiments/005_landuse_evidence_summary/article_text_classification_landuse_evidence_v1__qwen3_6_27b_q4_0`
- Gemma land-use experiment: `data/experiments/005_landuse_evidence_summary/article_text_classification_landuse_evidence_v1__gemma4_31b_it_q4_0`
- Qwen land-use spatial: `data/experiments/005_landuse_evidence_summary/article_text_classification_landuse_evidence_spatial_confidence_v1__qwen3_6_27b_q4_0`
- Gemma land-use spatial: `data/experiments/005_landuse_evidence_summary/article_text_classification_landuse_evidence_spatial_confidence_v1__gemma4_31b_it_q4_0`

## Validation
- Non-spatial files are parseable and include four rows per model (corine/OSM × aligned/shuffled).
- All land-use rows have `n_parse_error = 0` and `coverage = 1.0` in both overview metrics.
- Spatial reevaluations were completed for both models, and output files exist in both land-use spatial dirs.

## Primary comparisons

### Qwen: land-use vs generic
- corine_level2 / content: primary delta -0.0965 (0.2807 -> 0.1842).
- corine_level2 / summary: primary delta -0.0706 (0.2548 -> 0.1842).
- corine_level2 / summary_no_place: primary delta -0.0737 (0.2579 -> 0.1842).
- osm / content: primary delta -0.0545 (0.1636 -> 0.1091).
- osm / summary: primary delta -0.0364 (0.1455 -> 0.1091).
- osm / summary_no_place: primary delta -0.0545 (0.1636 -> 0.1091).

### Gemma: land-use vs generic
- corine_level2 / content: primary delta -0.0533 (0.2714 -> 0.2181).
- corine_level2 / summary: primary delta -0.0177 (0.2358 -> 0.2181).
- corine_level2 / summary_no_place: primary delta -0.0051 (0.2225 -> 0.2181).
- osm / content: primary delta -0.1127 (0.2145 -> 0.1018).
- osm / summary: primary delta -0.1382 (0.2400 -> 0.1018).
- osm / summary_no_place: primary delta -0.1418 (0.2473 -> 0.1018).

## Aligned minus shuffled (land-use text)
- qwen / corine_level2: 0.1370
- qwen / osm: 0.0473
- gemma / corine_level2: 0.1657
- gemma / osm: 0.0145

## Spatial all vs point_label_share_250m_ge_0.8
- qwen / corine_level2: balanced accuracy change 0.0661, exact match change , macro F1 change 0.0287, micro F1 change , Jaccard change .
- qwen / osm: balanced accuracy change , exact match change 0.0347, macro F1 change 0.0077, micro F1 change 0.0396, Jaccard change 0.0398.
- gemma / corine_level2: balanced accuracy change 0.0494, exact match change , macro F1 change 0.0052, micro F1 change , Jaccard change .
- gemma / osm: balanced accuracy change , exact match change 0.0107, macro F1 change 0.0081, micro F1 change 0.0193, Jaccard change 0.0160.

## Qwen vs Gemma (non-spatial, land-use summary)
- corine_level2: gemma minus qwen primary 0.0339, exact match , macro F1 0.0160, micro F1 .
- osm: gemma minus qwen primary -0.0073, exact match -0.0073, macro F1 -0.0277, micro F1 -0.0170.

## Qwen vs Gemma (spatial, land-use summary)
- corine_level2 / all_available_spatial_confidence: balanced accuracy 0.0339, exact match , macro F1 0.0160, micro F1 , Jaccard .
- corine_level2 / point_label_share_250m_ge_0.8: balanced accuracy 0.0171, exact match , macro F1 -0.0075, micro F1 , Jaccard .
- osm / all_available_spatial_confidence: balanced accuracy , exact match -0.0073, macro F1 -0.0277, micro F1 -0.0170, Jaccard -0.0150.
- osm / point_label_share_250m_ge_0.8: balanced accuracy , exact match -0.0312, macro F1 -0.0274, micro F1 -0.0372, Jaccard -0.0389.

## CORINE per-class behavior (point_label_share_250m_ge_0.8, land-use summary)

| label | qwen_f1 | gemma_f1 | qwen_minus_gemma_f1 |
| --- | ---: | ---: | ---: |
| 21 | 0.1955 | 0.1379 | 0.0576 |
| 22 | 0.7838 | 0.7891 | -0.0053 |
| 23 | 0.0278 | 0.0312 | -0.0035 |
| 24 | 0.0759 | 0.1163 | -0.0403 |
| 31 | 0.5214 | 0.5344 | -0.0131 |
| 32 | 0.2222 | 0.1509 | 0.0713 |
| 33 | 0.0000 | 0.0000 | 0.0000 |
| 41 | 0.0000 | 0.0000 | 0.0000 |
| 51 | 0.2609 | 0.2597 | 0.0011 |

## OSM metrics availability
- In non-spatial outputs, only `exact_match_accuracy`, `macro_f1`, and `micro_f1` are available for OSM.
- Jaccard is available in spatial outputs (`overview_spatial_subsets.csv`) and is reported above.
