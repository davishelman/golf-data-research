# Error analysis & insight reports (#75, #76)

The last layer: turn analysis outputs into **insight** — where the model works,
where it fails, whether it beats baselines, and what to try next — honestly.

## Error analysis (#75)

Slices backtest predictions to localize performance.

```python
from pipeline.modeling.player_course_advantage import build_error_analysis
ea = build_error_analysis(backtest_result.predictions, baseline_comparison)
ea["by_course"]            # Spearman per course
ea["by_coverage_bucket"]   # per coverage band
ea["by_config"]            # per v2.5 config (if present)
ea["worst_events"], ea["best_events"]
ea["failure_modes"]        # plain-language problems (coverage first)
ea["next_experiments"]     # what to try
```

Breaks down by course, season, event, coverage/occurrence bucket, score quintile,
and — **when the columns are present** — v2.5 config, par, and hole number.
Missing optional metadata is skipped gracefully. `identify_failure_modes` calls
out low coverage *before* performance, and says plainly when the model beats no
baselines.

## Insight report (#76)

```python
from pipeline.modeling.player_course_advantage import (
    render_insight_report, insight_verdict, load_and_render,
)
md = load_and_render("data/player_course_advantage/analysis_runs/<ts>")  # from a run dir
```

**Verdict scale:**

- ⚪ **gray** — synthetic run; predictive validity **not** evaluated.
- 🔴 **red** — real data but coverage too low, non-positive, or beats no baselines.
- 🟡 **yellow** — real, adequate coverage, mixed baseline lift / unstable.
- 🟢 **green** — real, adequate coverage, positive and beats **all** baselines.

The report always states real-vs-synthetic and whether predictive claims are
allowed, puts coverage limitations ahead of performance when data is sparse,
reports model-vs-baselines, calibration/reliability, best/worst events, failure
modes, and next experiments — and, for synthetic runs, ends with the exact real-data
blocker. It's concise and interview-ready. Generated reports are never committed.
