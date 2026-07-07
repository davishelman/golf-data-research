# player_course_advantage

A downstream model over **v2.5** similar-hole sets: given an upcoming course `C`
and a player `p`, estimate `p`'s per-hole and per-course **advantage** from how
`p` historically scored on holes that look like each of `C`'s 18 holes.

- **Not** a similarity model. It *reads* v2.5 similarity (`total_score` / rank)
  and never mutates v2 or v2.5 outputs.
- **Positive-is-good** primary outcome: `field_avg_score - player_score`
  (adjusts for hole difficulty and field/day conditions).

## Status

Built so far — the **scorer and backtest remain deferred**:

- **#30 / #31** — spec + historical hole-score input schema & validation.
- **#32** — similar-hole set loader (`similar_holes.py`): reads v2.5 result CSVs
  into normalized per-target-hole similar-hole sets.

Links:

- Full spec, notation, formulas, defaults, leakage warnings:
  [`docs/player_course_advantage.md`](../../../docs/player_course_advantage.md)
- Input contract + validation: [`schema.py`](schema.py)
  (`validate_hole_score_history`, `AdvantageParams`, `DEFAULT_PARAMS`,
  `field_adjusted_advantage`).
- Similar-hole loader: [`similar_holes.py`](similar_holes.py)
  (`load_similar_hole_sets`, `add_similarity_weights`, `parse_v25_hole_id`).

## Similar-hole loader (#32)

```python
from pipeline.modeling.player_course_advantage import load_similar_hole_sets

# root = a "courses/_index" (local) or artifact bundle root
df = load_similar_hole_sets(root, "augusta_national", config_name="baseline",
                            top_n=10, weight_method="rank_decay")
```

Returns one row per (target hole, candidate hole) with a `similarity_weight` that
sums to 1.0 within each target hole. Weight methods: `rank_decay` (default),
`inverse_score`, `softmax_score`, `uniform`. Path resolution reuses
`pipeline.modeling.pointcloud.demo`, so both the local-index and artifact-bundle
layouts work. Component score columns (`fairway_score`, …) are preserved when the
source CSV has them, and their absence does not break loading.

## Experimental defaults

`n = 10` similar holes/target · `W = 5`-year lookback · `m = 0.85` recency decay.
Placeholders from the sketch — not yet calibrated. See `AdvantageParams`.
