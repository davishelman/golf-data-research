# player_course_advantage

A downstream model over **v2.5** similar-hole sets: given an upcoming course `C`
and a player `p`, estimate `p`'s per-hole and per-course **advantage** from how
`p` historically scored on holes that look like each of `C`'s 18 holes.

- **Not** a similarity model. It *reads* v2.5 similarity (`total_score` / rank)
  and never mutates v2 or v2.5 outputs.
- **Positive-is-good** primary outcome: `field_avg_score - player_score`
  (adjusts for hole difficulty and field/day conditions).

## Status

Built so far — the **backtest remains deferred**:

- **#30 / #31** — spec + historical hole-score input schema & validation.
- **#32** — similar-hole set loader (`similar_holes.py`): reads v2.5 result CSVs
  into normalized per-target-hole similar-hole sets.
- **#33** — recency-weighted advantage scorer (`scorer.py`): turns the schema +
  similar-hole sets into player-hole and player-course advantages.
- **#39** — diagnostics / explanation outputs (`diagnostics.py`): reconciling
  breakdowns of *why* a score came out the way it did.

Links:

- Full spec, notation, formulas, defaults, leakage warnings:
  [`docs/player_course_advantage.md`](../../../docs/player_course_advantage.md)
- Input contract + validation: [`schema.py`](schema.py)
  (`validate_hole_score_history`, `AdvantageParams`, `DEFAULT_PARAMS`,
  `field_adjusted_advantage`).
- Similar-hole loader: [`similar_holes.py`](similar_holes.py)
  (`load_similar_hole_sets`, `add_similarity_weights`, `parse_v25_hole_id`).
- Advantage scorer: [`scorer.py`](scorer.py)
  (`score_player_holes`, `score_player_course`, `recency_weight`,
  `filter_history_for_prediction_window`).
- Diagnostics / explanation: [`diagnostics.py`](diagnostics.py)
  (`explain_player_course`, `contribution_rows`, `PlayerCourseExplanation`).

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

## Advantage scorer (#33)

```python
from pipeline.modeling.player_course_advantage import (
    load_similar_hole_sets, score_player_course,
)

sim = load_similar_hole_sets(root, "augusta_national")
per_hole, summary = score_player_course(
    history, sim, player_id="p123", target_course_slug="augusta_national",
    predict_season=2024,           # only strictly-past seasons are eligible
    aggregate="sum",               # or "mean"
)
```

- **Model.** `advantage(p, h)` is the similarity- and recency-weighted mean of
  `field_avg_score - player_score` over the player's occurrences on `h`'s similar
  holes; the course number sums (default) or averages the covered holes.
- **Leakage guard.** Eligible history is `predict_season - W <= year < predict_season`
  — never the target season or later — with `age = (predict_season - 1) - year` so
  the prior season has recency weight `1.0`. Same-course prior history is excluded
  unless `include_current_course_history=True`.
- **Coverage.** Holes below `min_occurrences_per_hole` get `hole_advantage = NaN`
  (never a fabricated 0) with a `reason`; a course with fewer than
  `min_holes_covered` covered holes is withheld. `score_player_holes` returns the
  per-hole detail; `score_player_course` adds a course-summary dict.
- Pure, deterministic, Streamlit-free. **v0** — still needs the backtest (#35).

## Diagnostics / explanation (#39)

```python
from pipeline.modeling.player_course_advantage import explain_player_course

expl = explain_player_course(history, sim, "p123", "augusta_national", 2024)
expl.top_target_holes(5)       # holes driving the course number
expl.top_similar_holes(5)      # (target, similar) pairs by weighted contribution
expl.occurrence_year_counts    # coverage by season
expl.low_coverage              # withheld holes + reason
expl.to_records()              # serializable for notebooks / UI
```

Read-only companion to the scorer. It re-derives the same
`similarity_weight · recency_weight` contributions the scorer aggregates, so the
breakdowns **reconcile exactly**: a hole's similar-hole contributions sum to its
advantage, and covered holes sum to the course advantage. No Streamlit formatting
leaks into scorer internals.

## Experimental defaults

`n = 10` similar holes/target · `W = 5`-year lookback · `m = 0.85` recency decay.
Placeholders from the sketch — not yet calibrated. See `AdvantageParams`.
