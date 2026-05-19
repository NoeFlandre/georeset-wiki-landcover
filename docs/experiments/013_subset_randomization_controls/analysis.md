# Experiment 013: Subset Randomization Controls

## Purpose

This analysis tests whether earlier filtered-subset gains are stronger than
random subsets with the same support. It does not rerun LLMs, prompts, labels,
spatial confidence, or GPU jobs. It only reuses frozen predictions and existing
metadata.

The core question is:

> When a filter improves CORINE balanced accuracy or OSM Jaccard, is that
> improvement better than random subsets of the same size and target
> distribution?

Outputs are in
`data/experiments/013_subset_randomization_controls/article_text_subset_randomization_controls_v1/`.

## Controls

Each random comparison uses the same universe as the observed subset:

- same parent experiment
- same model
- same task
- same text source
- same subset-specific metadata availability

This is important. For example, a spatial subset is compared only against rows
with spatial metadata, not against rows that could never have passed a spatial
filter.

Two Monte Carlo controls were computed with `n_draws=1000` and `seed=42`:

- `random_same_n`: random rows with the same subset size.
- `random_same_target_distribution`: random rows with the same CORINE target
  counts, or the same OSM target-label-set counts.

Rows with `n < 30` are marked `unstable_small_n=true`. Their metrics are still
reported, but they should be treated as diagnostic rather than definitive.

## Headline Results

The table below focuses on the requested headline setting: Qwen/Gemma, raw
`content`, CORINE balanced accuracy, and OSM Jaccard plus exact match.

| model | task | subset | n | observed | exact match | same-n mean | target-matched mean | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen | CORINE | `relevance_medium_high` | 576 | 0.374 |  | 0.280 | 0.280 | beats target-matched |
| Qwen | CORINE | `point_label_share_250m_ge_0.8` | 646 | 0.352 |  | 0.281 | 0.280 | beats target-matched |
| Qwen | CORINE | `relevance_medium_high_and_spatial_250m_ge_0.8` | 321 | 0.409 |  | 0.280 | 0.281 | beats target-matched |
| Qwen | CORINE | `quality_high_or_very_high_and_spatial_250m_ge_0.8` | 388 | 0.394 |  | 0.280 | 0.281 | beats target-matched |
| Qwen | CORINE | `recommended_use_training` | 195 | 0.385 |  | 0.282 | 0.278 | beats target-matched |
| Qwen | CORINE | `recommended_use_evaluation_only` | 126 | 0.473 |  | 0.278 | 0.282 | beats target-matched |
| Gemma | CORINE | `relevance_medium_high` | 576 | 0.352 |  | 0.271 | 0.271 | beats target-matched |
| Gemma | CORINE | `point_label_share_250m_ge_0.8` | 646 | 0.326 |  | 0.271 | 0.271 | beats target-matched |
| Gemma | CORINE | `relevance_medium_high_and_spatial_250m_ge_0.8` | 321 | 0.384 |  | 0.271 | 0.272 | beats target-matched |
| Gemma | CORINE | `quality_high_or_very_high_and_spatial_250m_ge_0.8` | 388 | 0.375 |  | 0.271 | 0.272 | beats target-matched |
| Gemma | CORINE | `recommended_use_training` | 195 | 0.338 |  | 0.272 | 0.270 | beats target-matched |
| Gemma | CORINE | `recommended_use_evaluation_only` | 126 | 0.472 |  | 0.270 | 0.271 | beats target-matched |
| Qwen | OSM | `relevance_medium_high` | 153 | 0.236 | 0.196 | 0.190 | 0.215 | beats same-n only |
| Qwen | OSM | `point_label_share_250m_ge_0.8` | 160 | 0.245 | 0.225 | 0.189 | 0.206 | beats target-matched |
| Qwen | OSM | `relevance_medium_high_and_spatial_250m_ge_0.8` | 92 | 0.291 | 0.261 | 0.190 | 0.231 | beats target-matched |
| Qwen | OSM | `quality_high_or_very_high_and_spatial_250m_ge_0.8` | 110 | 0.257 | 0.227 | 0.191 | 0.218 | beats same-n only |
| Qwen | OSM | `recommended_use_training` | 67 | 0.248 | 0.224 | 0.191 | 0.184 | not distinguishable |
| Qwen | OSM | `recommended_use_evaluation_only` | 25 | 0.407 | 0.360 | 0.188 | 0.353 | beats same-n only, small n |
| Gemma | OSM | `relevance_medium_high` | 153 | 0.247 | 0.196 | 0.251 | 0.272 | below random |
| Gemma | OSM | `point_label_share_250m_ge_0.8` | 160 | 0.286 | 0.256 | 0.252 | 0.281 | not distinguishable |
| Gemma | OSM | `relevance_medium_high_and_spatial_250m_ge_0.8` | 92 | 0.294 | 0.261 | 0.254 | 0.295 | not distinguishable |
| Gemma | OSM | `quality_high_or_very_high_and_spatial_250m_ge_0.8` | 110 | 0.269 | 0.236 | 0.252 | 0.300 | not distinguishable |
| Gemma | OSM | `recommended_use_training` | 67 | 0.249 | 0.224 | 0.252 | 0.266 | below random |
| Gemma | OSM | `recommended_use_evaluation_only` | 25 | 0.415 | 0.360 | 0.258 | 0.385 | not distinguishable, small n |

## Interpretation

High-relevance CORINE subsets survive both random controls. For raw content, Qwen
and Gemma medium/high relevance rows are well above both the same-size and
target-matched random means. This supports the earlier claim that relevance is
selecting genuinely more informative examples, not only a different class mix.

High spatial confidence also survives both controls for CORINE. The 250 m
spatial-purity subset beats target-matched controls for both models. The
combined relevance plus 250 m spatial subset is strongest: Qwen reaches 0.409
balanced accuracy versus a 0.281 target-matched random mean, and Gemma reaches
0.384 versus 0.272.

Quality-plus-spatial behaves like a strong proxy for relevance plus spatial
confidence on CORINE. It beats both controls for both models, but it does not
exceed the best relevance+spatial row. This suggests the quality score is useful
as a compact selection rule, not a fundamentally different signal.

Recommended-use subsets need more careful wording. `recommended_use_training`
is strong for CORINE but not definitive for OSM. `recommended_use_evaluation_only`
has high scores, but OSM has only 25 examples and is marked
`unstable_small_n=true`.

OSM is mixed. Qwen raw content has strong evidence on the combined
relevance+spatial subset, where Jaccard is 0.291 versus 0.231 for the
target-matched mean. But some OSM improvements beat same-size controls without
beating target-matched controls, which means class composition explains part of
the apparent gain. Gemma OSM content does not survive target matching in the
headline rows.

## Shuffled Deltas

The CORINE aligned-vs-shuffled result remains strong after random delta controls.
For raw content:

| model | task | subset | observed delta | random mean | random 97.5% | percentile |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | CORINE | `relevance_medium_high` | 0.297 | 0.214 | 0.246 | 100.0 |
| Qwen | CORINE | `point_label_share_250m_ge_0.8` | 0.281 | 0.214 | 0.241 | 100.0 |
| Qwen | CORINE | `relevance_medium_high_and_spatial_250m_ge_0.8` | 0.326 | 0.213 | 0.261 | 100.0 |
| Gemma | CORINE | `relevance_medium_high` | 0.279 | 0.206 | 0.232 | 100.0 |
| Gemma | CORINE | `point_label_share_250m_ge_0.8` | 0.260 | 0.205 | 0.231 | 100.0 |
| Gemma | CORINE | `relevance_medium_high_and_spatial_250m_ge_0.8` | 0.317 | 0.206 | 0.250 | 100.0 |

For OSM, the delta result is weaker. Qwen spatial-only OSM content has a delta
above the 97.5% random interval, but the relevance-only and combined subsets do
not. Gemma OSM deltas are not stronger than random controls in the headline
rows.

## What Should Change In The Claims

The strongest previous claim becomes more robust:

> For CORINE, geolocated Wikipedia text is most predictive when both article
> relevance and spatial label reliability are credible, and this is not explained
> away by support size or target class composition.

The OSM claims should be softened:

> OSM gains are text-linked for some Qwen content subsets, but target composition
> and small sample sizes explain more of the apparent improvement than they do
> for CORINE.

Recommended-use evaluation-only rows should be described as promising diagnostics,
not final evidence, whenever `unstable_small_n=true`.

## Files

- `observed_subset_metrics.csv/md`: all observed subset metrics.
- `random_same_n_controls.csv/md`: same-size Monte Carlo controls.
- `random_target_matched_controls.csv/md`: target-distribution-matched controls.
- `shuffled_delta_random_controls.csv/md`: aligned-vs-shuffled delta controls.
- `subset_class_distribution.csv/md`: class or label-set distribution by subset.
- `significant_filter_summary.csv/md`: compact headline table.
- `manifest.json`: run configuration, inputs, subset definitions, universe rules,
  and skipped rows.
