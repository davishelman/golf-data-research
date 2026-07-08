# Running the player-course advantage analysis bundle

A short how-to for running the analysis stack end-to-end and producing a bundle
of outputs for review. **This is a system/plumbing validation, not a predictive
evaluation.**

> ⚠️ **SYNTHETIC — validates plumbing only, not predictive edge.** No real
> historical player-by-hole score table is wired into this repo yet (blocker:
> [data sourcing plan](player_course_advantage_data_sourcing.md), #46). Until real,
> leakage-free data exists, every number the bundle produces is synthetic and
> carries **no** predictive-validity claim. See the
> [metrics contract](player_course_advantage_metrics.md) for the acceptance gates.

## Prerequisites

```bash
pip install -r requirements-dev.txt      # numpy, pandas, matplotlib, pytest, …
python -m pytest tests/test_player_course_advantage_*.py -q
```

## Where outputs go

Write everything under the **gitignored** analysis directory (never committed):

```
data/player_course_advantage/analysis_runs/<timestamp>/
```

`data/player_course_advantage/` is in `.gitignore`, so generated CSVs/JSON/markdown
there stay out of version control.

## What to run (existing modules only)

The bundle calls the shipped modules — no model logic is reimplemented. A minimal
synthetic run:

```python
from pathlib import Path
import pandas as pd
from pipeline.modeling.player_course_advantage import (
    validate_hole_score_history, assemble_field_outputs, explain_player_course,
    export_advantage_run, build_data_health_report, run_backtest, compare_baselines,
    baseline_lift_summary, run_sweep, SweepGrid, rank_ablation, ablation_effects,
    calibration_table, reliability_by_coverage, build_evaluation_summary,
    export_evaluation_report, run_benchmarks, AdvantageParams,
)

out = Path("data/player_course_advantage/analysis_runs/<timestamp>"); out.mkdir(parents=True, exist_ok=True)

# 1. Build SYNTHETIC inputs (fake course / similar holes / history / field / results).
#    Reuse pipeline.modeling.player_course_advantage.benchmarks.synthetic_inputs
#    for a quick sized dataset, or construct your own valid frames.
from pipeline.modeling.player_course_advantage.benchmarks import synthetic_inputs
sim, history, field, results = synthetic_inputs(n_players=30, n_events=5)
params = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=9)

# 2. Validate + health-check inputs FIRST (stop if red / too sparse).
validate_hole_score_history(history)
health = build_data_health_report(history, sim)

# 3. Ranking + diagnostics + artifact export.
outs = assemble_field_outputs(history, sim, field, "augusta_national", 2024, params=params)
expl = explain_player_course(history, sim, outs["rankings"].iloc[0]["player_id"],
                             "augusta_national", 2024, params=params)
run = export_advantage_run(outs["rankings"], hole_details=outs["hole_details"],
                           diagnostics=outs["diagnostics"], target_course_slug="augusta_national",
                           config_name="baseline", predict_season=2024, params=params, root=out)

# 4. Backtest + baselines + lift + sweep + calibration.
bt = run_backtest(history, sim, results, outcome_col="finish_rank", higher_is_better=False, params=params)
comp = compare_baselines(history, sim, results, outcome_col="finish_rank", higher_is_better=False, params=params)
lift = baseline_lift_summary(comp)
sweep = run_sweep(history, results, lambda c, n, w: sim, grid=SweepGrid(recency_decay=(0.7, 1.0)),
                  outcome_col="finish_rank", higher_is_better=False)
cal = calibration_table(bt.predictions)
rel = reliability_by_coverage(bt.predictions)

# 5. Evaluation report + benchmarks (label data_source SYNTHETIC).
summary = build_evaluation_summary(bt, comp, sweep, health, data_source="synthetic")
export_evaluation_report(summary, out)
bench = run_benchmarks(field_sizes=(10, 50), event_counts=(1, 5))
bench.to_csv(out / "benchmark_summary.csv", index=False)
```

Benchmarks also have a CLI:

```bash
python scripts/benchmark_player_course_advantage.py --field-sizes 10 50 150 --out \
    data/player_course_advantage/analysis_runs/<timestamp>/benchmark_summary.csv
```

## Honesty checklist

- Label every output `SYNTHETIC — validates plumbing only, not predictive edge`.
- `model_beats_baselines` / `baseline_lift_summary` will often report the model does
  **not** beat baselines on synthetic data — report that as-is.
- Do **not** commit anything under `analysis_runs/`, `.env`, `courses/`, or raw data.

## To run a *real* evaluation (blocked)

Source and normalize real per-hole scores into `validate_hole_score_history` (see the
[data sourcing plan](player_course_advantage_data_sourcing.md)), run
`build_data_health_report` first, then the same steps above with
`data_source="real"`. Until then: **no real predictive evaluation is possible
because no validated historical player-by-hole score table exists.**
