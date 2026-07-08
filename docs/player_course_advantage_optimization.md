# Parameter optimization & variants (#73, #74)

Tools to improve the player-course advantage model **honestly** — tuning on a
validation split without touching held-out test, and adding shrinkage/ensemble
variants that only win if they win on validation.

> ⚠️ Golf per-hole samples are small and noisy. Do **not** over-tune. Selecting a
> parameter set on the same data you report is leakage; always hold out a test
> split. On synthetic data no recommendation is emitted unless you opt in
> (`--allow-synthetic`).

## Validation-split optimizer (#73)

Grid-searches on `train ∪ validation` seasons, selects on **validation only**,
then reports the recommended params on the held-out **test** seasons.

```bash
python scripts/optimize_player_course_advantage_params.py \
    --history data/player_course_advantage/private/normalized_history.csv \
    --similar-holes courses/_index --course augusta_national \
    --results path/to/private/event_outcomes.csv \
    --train 2019 2020 2021 --validation 2022 --test 2023 \
    --config baseline fairway_heavy green_heavy hazard_heavy \
    --output data/player_course_advantage/analysis_runs/2026_07_08_real/opt \
    --data-source real
```

Optimizes `top_n`, `weight_method`, v2.5 `config`, lookback `W`, recency `m`,
`min_holes_covered`, `min_occurrences_per_hole`, `aggregate`. Outputs:
`optimization_summary.csv`, `parameter_rankings.csv`, `validation_metrics.csv`,
`test_metrics.csv` (if test supplied), `recommended_params.json`, markdown.

Rules enforced: train/validation/test must be disjoint; NaN/low-coverage sets rank
last; the recommendation is withheld for synthetic data unless
`--allow-synthetic`.

## Shrinkage & ensemble variants (#74)

For robustness when the similar-hole signal is thin:

```python
from pipeline.modeling.player_course_advantage import (
    default_variants, compare_variants, render_variant_summary, select_blend_alpha,
)
variants = default_variants(baseline_cols=["recent_form_advantage"])
comp = compare_variants(predictions, variants,
                        validation_seasons=[2022], test_seasons=[2023])
print(render_variant_summary(comp, data_source="real"))
```

Variants: `similar_only` (raw), `shrunk_zero`, `shrunk_by_holes`,
`shrunk_by_raw_occ`, `shrunk_by_weighted_occ`, and `blend_<baseline>`. Shrinkage
pulls low-coverage predictions toward zero (high-coverage barely moves). Blend
weights are chosen on validation (`select_blend_alpha`). **If a variant does not
beat the plain model on validation, the summary says to keep the simple model.**
Test metrics are reported for honesty, never used for selection.
