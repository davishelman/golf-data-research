"""Portfolio-ready insight report (issue #76).

Turns analysis outputs into an honest, concise, interview-ready markdown report:
does the model beat baselines, where does it work/fail, is it calibrated, is
coverage sufficient, and what to try next. It **never** claims predictive validity
from synthetic data, puts coverage ahead of performance when data is sparse, and
says plainly when the model loses to baselines.

Verdict scale: 🟢 green / 🟡 yellow / 🔴 red for real data; ⚪ gray when synthetic
(predictive validity not evaluated). Pure, Streamlit-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, Path]

VERDICTS = ("green", "yellow", "red", "gray")


def insight_verdict(
    data_source: str,
    *,
    coverage: Optional[float] = None,
    spearman_pooled: Optional[float] = None,
    beats_all_baselines: Optional[bool] = None,
    wins_vs_baselines_count: Optional[int] = None,
    min_coverage: float = 0.5,
) -> str:
    """The executive verdict.

    ``gray`` for synthetic (validity not evaluated); ``red`` for real data that is
    too sparse, non-positive, or beats no baselines; ``green`` for positive +
    beats-all + adequate coverage; ``yellow`` otherwise.
    """
    if data_source != "real":
        return "gray"
    if coverage is not None and not pd.isna(coverage) and coverage < min_coverage:
        return "red"  # coverage gates everything
    if wins_vs_baselines_count == 0:
        return "red"
    if spearman_pooled is not None and not pd.isna(spearman_pooled) and spearman_pooled <= 0:
        return "red"
    if beats_all_baselines and (spearman_pooled or 0) > 0:
        return "green"
    return "yellow"


def render_insight_report(
    *,
    data_source: str,
    spearman_pooled: Optional[float] = None,
    spearman_mean: Optional[float] = None,
    coverage: Optional[float] = None,
    event_coverage: Optional[float] = None,
    beats_all_baselines: Optional[bool] = None,
    wins_vs_baselines_count: Optional[int] = None,
    n_baselines_comparable: Optional[int] = None,
    calibration: Optional[dict] = None,
    stability: Optional[dict] = None,
    error_analysis: Optional[dict] = None,
    variant_note: Optional[str] = None,
    real_data_blocker: Optional[str] = None,
    min_coverage: float = 0.5,
) -> str:
    """Render the full markdown insight report."""
    real = data_source == "real"
    verdict = insight_verdict(
        data_source, coverage=coverage, spearman_pooled=spearman_pooled,
        beats_all_baselines=beats_all_baselines,
        wins_vs_baselines_count=wins_vs_baselines_count, min_coverage=min_coverage)
    icon = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚪"}[verdict]

    L = ["# Player-course advantage — insight report", ""]
    L.append(f"## Executive verdict: {icon} {verdict.upper()}")
    L.append(f"- Data source: **{data_source}**")
    if real:
        L.append("- Predictive claims: **allowed** (real, leakage-free evaluation).")
    else:
        L.append("- Predictive claims: **not allowed** — synthetic run; predictive "
                 "validity was **not** evaluated (plumbing only).")

    # Coverage first when sparse.
    L += ["", "## Data health & coverage"]
    L.append(f"- Scored-pair coverage: {_fmt(coverage)}"
             + (f"  ·  event coverage: {_fmt(event_coverage)}" if event_coverage is not None else ""))
    if real and coverage is not None and not pd.isna(coverage) and coverage < min_coverage:
        L.append("- ⚠️ **Coverage is too low to trust performance** — treat metrics below "
                 "as provisional and fix coverage first.")

    L += ["", "## Model vs baselines"]
    if wins_vs_baselines_count is not None:
        L.append(f"- Beats {wins_vs_baselines_count}/{n_baselines_comparable} baselines; "
                 f"beats_all={beats_all_baselines}.")
        if real and wins_vs_baselines_count == 0:
            L.append("- 🔴 The model beats **no** baselines — do not claim it adds value; "
                     "prefer the simple baseline.")
    L.append(f"- Spearman (pooled / mean): {_fmt(spearman_pooled)} / {_fmt(spearman_mean)}")

    if error_analysis:
        best = error_analysis.get("best_events")
        worst = error_analysis.get("worst_events")
        L += ["", "## Best / worst events"]
        L.append(f"- best: {_events(best)}")
        L.append(f"- worst: {_events(worst)}")

    L += ["", "## Calibration & reliability"]
    if calibration:
        L.append(f"- monotonicity: {_fmt(calibration.get('monotonicity_score'))}"
                 f"  ·  slope: {_fmt(calibration.get('calibration_slope'))}")
    else:
        L.append("- not computed.")

    if stability:
        L += ["", "## Stability"]
        for k, v in stability.items():
            L.append(f"- {k}: {_fmt(v)}")

    L += ["", "## Parameter / variant sensitivity"]
    L.append(variant_note or "- not evaluated in this run.")

    L += ["", "## Likely failure modes"]
    modes = (error_analysis or {}).get("failure_modes", [])
    L += ([f"- {m}" for m in modes] or ["- None flagged."])

    L += ["", "## Recommended next experiments"]
    nexts = (error_analysis or {}).get("next_experiments", [])
    L += ([f"- {x}" for x in nexts] or ["- Validate on more seasons/courses."])

    if not real:
        L += ["", "## Blocker for real predictive claims",
              f"- {real_data_blocker or 'No validated real historical player-by-hole score table is present. Predictive validity cannot be evaluated until real, leakage-free data is ingested (see the data sourcing plan).'}"]
    return "\n".join(L)


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    try:
        if pd.isna(v):
            return "NaN"
    except (TypeError, ValueError):
        return str(v)
    return f"{float(v):.3f}" if isinstance(v, (int, float)) else str(v)


def _events(frame) -> str:
    if frame is None or getattr(frame, "empty", True):
        return "n/a"
    return ", ".join(
        f"{r['event_id']}({r['spearman']:.2f})" for _, r in frame.head(3).iterrows()
    )


def load_and_render(run_dir: PathLike, *, min_coverage: float = 0.5) -> str:
    """Render an insight report from an analysis-run directory (manifest + tables)."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "analysis_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no analysis_manifest.json in {run_dir}")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    error_analysis = None
    preds_path = run_dir / "backtest_predictions.csv"
    comp_path = run_dir / "baseline_comparison.csv"
    if preds_path.exists():
        from .error_analysis import build_error_analysis
        preds = pd.read_csv(preds_path)
        comp = pd.read_csv(comp_path) if comp_path.exists() else None
        error_analysis = build_error_analysis(preds, comp)

    return render_insight_report(
        data_source=m.get("data_source", "synthetic"),
        spearman_pooled=m.get("spearman_pooled"),
        coverage=m.get("coverage"),
        beats_all_baselines=m.get("beats_all_baselines"),
        wins_vs_baselines_count=m.get("wins_vs_baselines_count"),
        error_analysis=error_analysis,
        min_coverage=min_coverage,
    )


__all__ = ["VERDICTS", "insight_verdict", "render_insight_report", "load_and_render"]
