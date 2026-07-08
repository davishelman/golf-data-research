"""Model evaluation report generator (issue #59).

Turns the shipped analysis outputs — a ``BacktestResult`` (#35), a
``compare_baselines`` table (#38), an optional ``SweepResult`` (#36), an optional
loaded run (#44), and an optional data-health report (#62) — into **hard metrics
tables** and a portable report. It computes nothing new about the model; it
*aggregates and labels* what those layers already produced.

Honest by construction: every summary carries ``data_source`` (``synthetic`` vs
``real``), the markdown always includes a caveat when not real, and it never
asserts the model beats baselines unless the baseline lift (from #60) says so.
Export writes only to a caller-supplied path — never a tracked default. Pure,
Streamlit-free, synthetic-testable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .ablation import MODEL_NAME, baseline_lift_summary
from .data_health import summarize_backtest_coverage

PathLike = Union[str, Path]


def summarize_per_event_metrics(backtest_result) -> pd.DataFrame:
    """Per-event metrics with best/worst-by-Spearman flags."""
    pe = backtest_result.per_event.copy()
    if not pe.empty and "spearman" in pe.columns and pe["spearman"].notna().any():
        best = pe["spearman"].max()
        worst = pe["spearman"].min()
        pe["is_best_event"] = pe["spearman"] == best
        pe["is_worst_event"] = pe["spearman"] == worst
    return pe


def _best_worst_events(pe: pd.DataFrame) -> dict:
    if pe.empty or "spearman" not in pe.columns or not pe["spearman"].notna().any():
        return {"best": None, "worst": None}

    def _pick(idx):
        row = pe.loc[idx]
        return {"event_id": row.get("event_id"),
                "spearman": float(row["spearman"]) if pd.notna(row["spearman"]) else None}

    return {"best": _pick(pe["spearman"].idxmax()), "worst": _pick(pe["spearman"].idxmin())}


def summarize_coverage(backtest_result, data_health: Optional[dict] = None) -> dict:
    """Backtest coverage (reuses #62) plus any data-health warnings."""
    preds = getattr(backtest_result, "predictions", None)
    cov = summarize_backtest_coverage(preds) if preds is not None and len(preds) else {}
    if data_health:
        cov = {**cov, "data_health_warnings": list(data_health.get("warnings", []))}
    return cov


def summarize_stability(backtest_result) -> dict:
    """Spread of per-event Spearman across seasons and courses (lower = steadier)."""
    pe = backtest_result.per_event
    out: dict = {}
    if pe.empty or "spearman" not in pe.columns:
        return out
    for col, name in (("predict_season", "season"), ("target_course_slug", "course")):
        if col in pe.columns and pe[col].nunique() > 1:
            by = pe.groupby(col)["spearman"].mean()
            out[f"spearman_std_across_{name}s"] = float(by.std(ddof=0))
            out[f"spearman_range_across_{name}s"] = float(by.max() - by.min())
    return out


def summarize_baseline_lift(
    baseline_comparison: pd.DataFrame,
    *,
    metric: str = "spearman_pooled",
    model_name: str = MODEL_NAME,
) -> dict:
    """Delegates to the #60 lift summary (single source of truth)."""
    return baseline_lift_summary(baseline_comparison, metric=metric, model_name=model_name)


def build_evaluation_summary(
    backtest_result,
    baseline_comparison: Optional[pd.DataFrame] = None,
    sweep=None,
    data_health: Optional[dict] = None,
    *,
    data_source: str = "synthetic",
    primary_metric: str = "spearman_pooled",
    model_name: str = MODEL_NAME,
) -> dict:
    """Assemble the evaluation summary dict from the available analysis outputs."""
    pe = summarize_per_event_metrics(backtest_result)
    summary = {
        "data_source": data_source,
        "primary_metric": primary_metric,
        "backtest_summary": dict(backtest_result.summary),
        "coverage": summarize_coverage(backtest_result, data_health),
        "stability": summarize_stability(backtest_result),
        "best_worst_events": _best_worst_events(pe),
    }
    if baseline_comparison is not None and not baseline_comparison.empty:
        summary["baseline_lift"] = summarize_baseline_lift(
            baseline_comparison, metric=primary_metric, model_name=model_name)
        summary["baseline_comparison"] = baseline_comparison.to_dict("records")
    if sweep is not None:
        best = sweep.recommend() if hasattr(sweep, "recommend") else None
        summary["sweep"] = {
            "recommended": (best.to_dict() if best is not None else None),
            "warnings": list(getattr(sweep, "warnings", [])),
        }
    return summary


def make_metrics_manifest(summary: dict) -> dict:
    """Flatten the summary into a compact scalar manifest (for tracking/diffing)."""
    bt = summary.get("backtest_summary", {})
    cov = summary.get("coverage", {})
    lift = summary.get("baseline_lift", {})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": summary.get("data_source"),
        "primary_metric": summary.get("primary_metric"),
    }
    for k in ("spearman_pooled", "spearman_mean", "pearson_pooled", "n_pairs"):
        if k in bt:
            manifest[k] = bt[k]
    for k in ("coverage", "event_coverage"):
        if k in cov:
            manifest[k] = cov[k]
    for k in ("wins_vs_baselines_count", "beats_all_baselines", "model_rank"):
        if k in lift:
            manifest[k] = lift[k]
    return manifest


def render_evaluation_markdown(summary: dict) -> str:
    """A concise, honest markdown report (always carries a data-source caveat)."""
    bt = summary.get("backtest_summary", {})
    cov = summary.get("coverage", {})
    src = summary.get("data_source", "synthetic")
    lines = [
        "# Player-course advantage — evaluation report",
        "",
        f"- **Data source:** `{src}`"
        + ("" if src == "real" else "  ⚠️ synthetic — **not** a predictive-validity claim."),
        f"- Spearman (pooled / mean): "
        f"{bt.get('spearman_pooled', float('nan')):.3f} / "
        f"{bt.get('spearman_mean', float('nan')):.3f}",
        f"- Coverage: {cov.get('coverage', float('nan'))}  ·  scored pairs: "
        f"{bt.get('n_pairs', 0)}",
    ]
    if "baseline_lift" in summary:
        lift = summary["baseline_lift"]
        verdict = "beats all comparable baselines" if lift.get("beats_all_baselines") \
            else "does NOT beat all baselines"
        lines.append(
            f"- Baseline lift: model {verdict} "
            f"({lift.get('wins_vs_baselines_count')}/{lift.get('n_baselines_comparable')} "
            f"wins, rank {lift.get('model_rank')}/{lift.get('n_rankers')})."
        )
    bw = summary.get("best_worst_events", {})
    if bw.get("best"):
        lines.append(f"- Best event: {bw['best']}  ·  worst: {bw['worst']}")
    return "\n".join(lines)


def export_evaluation_report(summary: dict, out_dir: PathLike) -> dict[str, Path]:
    """Write the summary JSON, metrics manifest, and markdown to ``out_dir``.

    Writes only under the supplied path — never a tracked default. Returns the
    written file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "evaluation_summary.json"
    manifest_path = out / "metrics_manifest.json"
    report_path = out / "evaluation_report.md"

    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(make_metrics_manifest(summary), indent=2, default=str), encoding="utf-8")
    report_path.write_text(render_evaluation_markdown(summary), encoding="utf-8")
    return {"summary": summary_path, "manifest": manifest_path, "report": report_path}


__all__ = [
    "summarize_per_event_metrics",
    "summarize_coverage",
    "summarize_stability",
    "summarize_baseline_lift",
    "build_evaluation_summary",
    "make_metrics_manifest",
    "render_evaluation_markdown",
    "export_evaluation_report",
]
