# CORINE Spatial Confidence v1

Point-in-polygon CORINE labels are fragile supervision for article-level land-cover classification. A Wikipedia coordinate may sit near a boundary, on a small polygon, or in a location whose surrounding landscape is mixed. This experiment measures that spatial reliability without rerunning the LLM.

The diagnostic buffers each article coordinate at 100 m, 250 m, 500 m, and 1000 m. For each radius, it computes area-weighted CORINE level-2 label shares inside the buffer. `point_label_share` is the primary confidence variable because it asks whether the original ground-truth label assigned to the point dominates the surrounding area.

All buffering and area calculations use EPSG:2154. The full CORINE dataset is used, including artificial classes, because nearby urban/artificial land is ambiguity evidence and should not be hidden.

The downstream spatial-subset evaluation joins this confidence table to the frozen `article_text_classification_e2e_with_shuffled_control_v1` predictions. It does not change prompts, summaries, model temperature, model outputs, or any previous frozen files.

Interpretation:

- If aligned text improves on high-purity subsets while shuffled text remains low, then the text signal is stronger when the spatial ground truth is reliable.
- If performance does not improve on high-purity subsets, the main bottleneck is probably not spatial ambiguity; next steps should be article relevance filtering, land-cover evidence extraction, prompt improvement, or a stronger LLM ceiling test.
- If many articles have low purity, that is itself an important GEO-ReSeT result: Wikipedia point coordinates are noisy supervision for land-cover labels unless filtered.
