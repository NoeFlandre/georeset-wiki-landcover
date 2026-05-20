# article_text_subset_randomization_controls_v1

This analysis-only experiment compares filtered article subsets against Monte Carlo random controls. It does not rerun any LLMs, prompts, labels, spatial confidence, or GPU jobs.

## What The Controls Mean

- `random_same_n`: same parent experiment, model, task, text source, subset metadata availability, and subset size.
- `random_same_target_distribution`: same comparison universe, plus the same CORINE target counts or OSM target-label-set counts.

The comparison universe is intentionally explicit: random controls sample only from rows that could have passed the same metadata-availability requirements as the observed subset.

Rows with `n < 30` are marked `unstable_small_n=true`. They are reported, but should be treated as diagnostic rather than definitive.

## Main Results

For CORINE raw content, the key filters beat both random controls for both models. Qwen reaches 0.374 vs target-matched mean 0.280 on `relevance_medium_high` and 0.409 vs target-matched mean 0.281 on `relevance_medium_high_and_spatial_250m_ge_0.8`. Gemma reaches 0.352 vs target-matched mean 0.271 and 0.384 vs target-matched mean 0.272 on the same two subsets.

Quality-plus-spatial behaves like a strong proxy for relevance plus spatial confidence. It beats both controls for CORINE, but does not exceed the combined relevance+spatial subset.

For OSM, results are more mixed. Qwen raw content on `relevance_medium_high_and_spatial_250m_ge_0.8` reaches 0.291 vs target-matched mean 0.231, exact match 0.261. Gemma reaches 0.294 vs target-matched mean 0.295, exact match 0.261 on the same subset, which is not distinguishable from target-matched controls in the headline table. The Qwen OSM `recommended_use_evaluation_only` row is 0.407 vs target-matched mean 0.353, exact match 0.360 (unstable small n), so it is diagnostic only.

## Interpretation

The strongest previous CORINE claim survives: when article relevance and spatial label reliability are credible, Wikipedia text signal is stronger than expected from subset size or target class composition alone.

The OSM claim should be phrased more carefully. OSM text signal exists for some Qwen content subsets, but target composition and small support explain more of the apparent improvement than they do for CORINE.

Aligned-vs-shuffled CORINE deltas also survive random delta controls. For raw content, both Qwen and Gemma CORINE deltas on relevance/spatial headline subsets are above the random 97.5% interval. OSM shuffled deltas are weaker and less consistent, especially for Gemma.

## Headline Control Flags

- beats_same_n: 8
- beats_target_matched: 23
- below_random: 2
- not_distinguishable: 15
