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
- **#34** — batch tournament-field ranking (`batch.py`): rank a whole field for a
  course, Python API + CLI.
- **#44 / #45** — artifact export/load (`artifact_export.py`) + an end-to-end
  synthetic smoke test proving the full pipeline runs offline.
- **#35** — retrospective backtest framework (`backtest.py`): leakage-guarded
  walk-forward evaluation with rank-correlation / hit-rate / lift metrics.
- **#38** — simple comparison baselines (`baselines.py`): drop-in rankers +
  side-by-side backtest comparison.
- **#36** — parameter sweep (`sweep.py`): reproducible grid search around the
  backtest with train/validation split logic.
- **#46** — historical data sourcing plan
  ([`docs/player_course_advantage_data_sourcing.md`](../../../docs/player_course_advantage_data_sourcing.md)):
  sources, ID mapping, normalization path, and the **blocker** — no real per-hole
  data is wired yet, so all numbers remain synthetic.
- **#37** — evaluation notebook
  ([`notebooks/player_course_advantage_backtest.ipynb`](../../../notebooks/player_course_advantage_backtest.ipynb)):
  offline, synthetic-labelled walkthrough of the whole pipeline (scorer →
  ranking → diagnostics → backtest → baselines → sweep).
- **#40** — Streamlit UI integration plan
  ([`docs/player_course_advantage_ui_plan.md`](../../../docs/player_course_advantage_ui_plan.md)):
  how the advantage view is added as an additive tab (implementation is #47).
- **#47** — Streamlit ranking view (`ui.py` + additive section 7 in `app.py`):
  rankings, coverage badges, and per-hole contribution detail from artifacts or a
  synthetic demo. Streamlit-free helpers keep it testable.

Links:

- Full spec, notation, formulas, defaults, leakage warnings:
  [`docs/player_course_advantage.md`](../../../docs/player_course_advantage.md)
- Metrics contract + acceptance gates:
  [`docs/player_course_advantage_metrics.md`](../../../docs/player_course_advantage_metrics.md)
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
- Batch field ranking: [`batch.py`](batch.py)
  (`score_tournament_field`, `rank_field`, `export_field_ranking`, CLI `main`).
- Artifact export/load: [`artifact_export.py`](artifact_export.py)
  (`assemble_field_outputs`, `export_advantage_run`, `load_advantage_run`).
- Backtest framework: [`backtest.py`](backtest.py)
  (`run_backtest`, `BacktestResult`, `spearman_corr`, `top_k_hit_rate`).
- Comparison baselines: [`baselines.py`](baselines.py)
  (`compare_baselines`, `model_beats_baselines`, `BASELINES`).
- Parameter sweep: [`sweep.py`](sweep.py)
  (`run_sweep`, `SweepGrid`, `SweepResult.recommend`, `export_sweep`).
- Streamlit view helpers: [`ui.py`](ui.py)
  (`synthetic_demo_view`, `load_ranking_view`, `discover_advantage_runs`,
  `format_ranking_for_display`) — pure, drives `app.py` section 7.

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

## Batch field ranking (#34)

```python
from pipeline.modeling.player_course_advantage import (
    load_similar_hole_sets, score_tournament_field,
)

sim = load_similar_hole_sets(root, "augusta_national")
ranking = score_tournament_field(
    history, sim, field,               # field: list of player_ids or a DataFrame
    target_course_slug="augusta_national", predict_season=2024,
)
```

One deterministic row per player (valid players first, then descending
`course_advantage`, higher coverage, `player_id` tie-break); low-coverage players
are ranked last and flagged, never dropped or faked. Also runnable as a CLI:

```bash
python -m pipeline.modeling.player_course_advantage.batch \
    --root courses/_index --history history.csv \
    --course augusta_national --predict-season 2024 --out data/player_course_advantage/run1
```

`export_field_ranking(...)` writes `player_rankings.csv` + a `manifest.json`;
generated outputs live under `data/player_course_advantage/` and are gitignored.
Per-player *why* comes from the #39 diagnostics module.

## Artifacts (#44) and the end-to-end smoke test (#45)

```python
from pipeline.modeling.player_course_advantage import (
    assemble_field_outputs, export_advantage_run, load_advantage_run,
)

outs = assemble_field_outputs(history, sim, field, "augusta_national", 2024)
run = export_advantage_run(
    outs["rankings"], hole_details=outs["hole_details"], diagnostics=outs["diagnostics"],
    target_course_slug="augusta_national", config_name="baseline",
    predict_season=2024, params=DEFAULT_PARAMS,
)
loaded = load_advantage_run(run.run_dir)
```

Run layout under `data/player_course_advantage/<run_id>/` (gitignored):

```
player_rankings.csv       one row per player (batch ranking)
player_hole_details.csv   one row per (player, target hole)
diagnostics.csv           per (player, target hole, similar hole) contributions
backtest_summary.csv      optional — written only when #35 supplies it
parameters.json           AdvantageParams + aggregate mode
manifest.json             run id, model version, timestamps, input/output counts
```

Only ids/scores/diagnostics are written — **never raw point-cloud geometry**
(guarded on export). The #45 smoke test exercises the whole chain (fake v2.5 CSV
→ loader → scorer → ranking → diagnostics → export/load) offline, with no real
`courses/` outputs, no network, and no Streamlit.

## Backtest (#35)

```python
from pipeline.modeling.player_course_advantage import run_backtest

result = run_backtest(
    history, sim, results,             # results: actual outcomes per (event, player)
    outcome_col="finish_rank", higher_is_better=False,   # lower finish = better
)
result.summary          # pooled + mean Spearman/Pearson, hit-rate, lift, coverage
result.per_event        # one row of metrics per historical event
result.to_markdown()    # optional report
```

Each event is scored with `predict_season = <event season>`, so the scorer admits
only `predict_season - W <= year < predict_season` — **the leakage guard**. Each
event freezes its own window; nothing is pooled and scored in-sample. Metrics:
Spearman / Pearson (mean-of-events and pooled), top-k hit-rate, top-k lift,
optional MAE/RMSE, and coverage. Correlations are computed without a SciPy
dependency.

**v0 plumbing only.** It reports honest metrics on whatever labels it's given
(synthetic in tests) and makes **no** predictive-validity claim on real data —
that needs the #46 data sourcing first. The `backtest_summary.csv` slot in the
artifact layout is where a run's metrics land.

## Baselines (#38)

```python
from pipeline.modeling.player_course_advantage import (
    compare_baselines, model_beats_baselines,
)

table = compare_baselines(history, sim, results, outcome_col="finish_rank",
                          higher_is_better=False)   # one metrics row per ranker
beats = model_beats_baselines(table)                # strict, NaN-safe verdict
```

Each baseline (`null`, `recent_form`, `course_history`, `same_par`,
`season_average`) is a **drop-in ranker** with the model's signature/output, so
the backtest evaluates them on identical events via its `ranker=` hook. All share
the same leakage-guarded prediction window. `model_beats_baselines` only returns
`True` when the model *strictly* exceeds every baseline — the model does **not**
automatically win, and on synthetic data it frequently ties. A v2
feature-similarity baseline is intentionally left out (it would couple this layer
to v2 internals). Real superiority is unproven until #46 supplies leakage-free
labels.

## Parameter sweep (#36)

```python
from pipeline.modeling.player_course_advantage import SweepGrid, run_sweep

grid = SweepGrid(recency_decay=(0.7, 0.85, 1.0), lookback_years=(3, 5),
                 min_holes_covered=(8, 12))
res = run_sweep(history, results, provider,   # provider(config, top_n, weight_method) -> sim
                grid=grid, outcome_col="finish_rank", higher_is_better=False,
                validation_seasons=[2024])    # hold out for honest selection
res.table                       # metrics per parameter set (train + val_ columns)
res.recommend()                 # best row clearing coverage/pairs guards
res.recommended_params()        # -> AdvantageParams
res.warnings                    # in-sample / low-coverage / low-pairs flags
```

Loader-side knobs (`top_n`, `weight_method`, v2.5 `config_name`) come from the
memoized `provider`; scorer-side knobs (`W`, `m`, coverage thresholds, aggregate)
go into `AdvantageParams`. Each backtest is leakage-free per event. Selecting the
best row on the same events is in-sample tuning, so **pass `validation_seasons`**:
metrics split into train / `val_*`, `recommend` chooses on validation, and a
warning fires when no split is given. `recommend` also refuses rows below the
coverage / scored-pairs guards so a near-empty field can't win. Real sweep
outputs are gitignored (`export_sweep`).

## Experimental defaults

`n = 10` similar holes/target · `W = 5`-year lookback · `m = 0.85` recency decay.
Placeholders from the sketch — not yet calibrated. See `AdvantageParams`.
