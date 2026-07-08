# Real-data analysis runner (#72)

One repeatable command that turns a **validated canonical history** + v2.5
similar-hole sets + actual event outcomes into the full analysis bundle. It
sources no data — you supply it — and writes only under your output directory.

> Predictive claims require `--data-source real` with a validated real history
> **and** real event outcomes. With synthetic inputs the report says predictive
> validity was not evaluated.

## Inputs

- `--history` — canonical CSV from the [ingestion adapter](player_course_advantage_real_data_ingestion.md)
  (must pass `validate_hole_score_history`).
- `--similar-holes` + `--course` — v2.5 results root and target course slug.
- `--results` — actual outcomes: one row per (event, player) with
  `event_id, predict_season, target_course_slug, player_id, <outcome_col>`
  (e.g. `finish_rank`).
- `--output` — bundle directory (keep under gitignored `data/player_course_advantage/`).

## Run

```bash
python scripts/run_player_course_advantage_real_analysis.py \
    --history  data/player_course_advantage/private/normalized_history.csv \
    --similar-holes courses/_index --course augusta_national \
    --results  path/to/private/event_outcomes.csv \
    --output   data/player_course_advantage/analysis_runs/2026_07_08_real \
    --data-source real --run-sweep --run-benchmarks
```

## Pipeline & outputs

1. validate history (refuses predictive evaluation on failure),
2. data-health report — **stops early** if input coverage < `--min-coverage`,
3. leakage-free backtest, 4. baseline comparison + lift, 5. optional sweep,
6. calibration/reliability, 7. optional benchmarks, 8. evaluation report + manifest.

Files written to `--output`: `analysis_manifest.json`, `data_health_summary.csv`,
`coverage_by_event.csv`, `backtest_predictions.csv`, `backtest_per_event.csv`,
`baseline_comparison.csv`, `baseline_lift.csv`, `sweep_summary.csv` (if enabled),
`calibration_table.csv`, `reliability_by_coverage.csv`, `benchmark_summary.csv`
(if enabled), `analysis_report.md`, `missing_outputs.json` (why anything was skipped).

Nothing is committed. If real outcomes are absent you cannot run the backtest —
the runner will tell you.
