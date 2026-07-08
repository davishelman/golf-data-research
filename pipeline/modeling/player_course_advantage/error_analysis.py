"""Error analysis by course, hole type, and coverage (issue #75).

Slices backtest performance to find *where* the player-course advantage model
works and where it fails — by course, season, event, coverage bucket, similar-hole
count, score bucket, v2.5 config, par/hole (when that metadata is present), and
low-coverage reason. It names failure modes honestly (including "the model does
not beat baselines" and "low coverage explains this") and suggests next
experiments.

Metrics are computed at the level the backtest supports: Spearman(advantage,
realized performance) within each group of covered (player, event) rows. Optional
metadata columns are used only when present. Pure pandas, Streamlit-free.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .ablation import baseline_lift_summary
from .backtest import spearman_corr

_MIN_GROUP = 2  # need ≥2 covered rows for a rank correlation


def _covered(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions
    if "covered" in df.columns:
        df = df[df["covered"].fillna(False)]
    for c in ("course_advantage", "performance"):
        if c in df.columns:
            df = df[df[c].notna()]
    return df


def add_analysis_buckets(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with coverage / occurrence / score buckets for grouping."""
    df = predictions.copy()
    if "holes_covered" in df.columns and "total_target_holes" in df.columns:
        total = pd.to_numeric(df["total_target_holes"], errors="coerce").replace(0, np.nan)
        frac = pd.to_numeric(df["holes_covered"], errors="coerce") / total
        df["coverage_bucket"] = pd.cut(
            frac, [-0.01, 0.25, 0.5, 0.75, 1.01],
            labels=["<25%", "25-50%", "50-75%", "75%+"],
        )
    if "total_raw_occurrences" in df.columns:
        df["raw_occ_bucket"] = pd.cut(
            pd.to_numeric(df["total_raw_occurrences"], errors="coerce"),
            [-0.01, 3, 10, 30, np.inf], labels=["0-3", "4-10", "11-30", "30+"],
        )
    cov = _covered(df)
    if len(cov) >= 5 and cov["course_advantage"].nunique() >= 5:
        try:
            df.loc[cov.index, "score_quintile"] = pd.qcut(
                cov["course_advantage"], 5, labels=False, duplicates="drop")
        except ValueError:
            pass
    return df


def performance_by(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per-group Spearman(advantage, performance) + counts for one dimension."""
    cols = [column, "n", "n_covered", "spearman", "mean_performance"]
    if column not in predictions.columns:
        return pd.DataFrame(columns=cols)
    cov = _covered(predictions)
    rows = []
    for key, g in cov.groupby(column, observed=True):
        rows.append({
            column: key,
            "n": int(len(g)),
            "n_covered": int(len(g)),
            "spearman": (spearman_corr(g["course_advantage"], g["performance"])
                         if len(g) >= _MIN_GROUP else float("nan")),
            "mean_performance": float(g["performance"].mean()),
        })
    return pd.DataFrame(rows, columns=cols).sort_values(column).reset_index(drop=True)


def best_worst_events(predictions: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best and worst events by per-event Spearman (needs ``event_id``)."""
    by_event = performance_by(predictions, "event_id")
    ranked = by_event.dropna(subset=["spearman"]).sort_values("spearman", ascending=False)
    return ranked.head(k).reset_index(drop=True), ranked.tail(k).iloc[::-1].reset_index(drop=True)


def identify_failure_modes(
    predictions: pd.DataFrame,
    baseline_comparison: Optional[pd.DataFrame] = None,
    *,
    primary_metric: str = "spearman_pooled",
    low_coverage_threshold: float = 0.5,
) -> tuple[list, list]:
    """Return ``(failure_modes, next_experiments)`` — honest, plain-language findings."""
    modes: list[str] = []
    nexts: list[str] = []

    # Coverage first — unreliable scores outrank performance talk.
    n = int(len(predictions))
    covered = int(_covered(predictions).shape[0])
    cov_rate = (covered / n) if n else float("nan")
    if n and cov_rate < low_coverage_threshold:
        modes.append(
            f"Low coverage: only {cov_rate:.0%} of (player, event) pairs were scored — "
            "treat any performance number as unreliable until coverage improves.")
        nexts.append("Increase history depth or relax coverage thresholds before trusting metrics.")

    by_course = performance_by(predictions, "target_course_slug")
    weak = by_course[by_course["spearman"] < 0]
    if not weak.empty:
        modes.append("Model is anti-correlated with outcomes on: "
                     + ", ".join(str(c) for c in weak["target_course_slug"].tolist()[:5]))
        nexts.append("Inspect similar-hole quality / v2.5 config for the weak courses.")

    if baseline_comparison is not None and not baseline_comparison.empty:
        lift = baseline_lift_summary(baseline_comparison, metric=primary_metric)
        if lift["n_baselines_comparable"] and lift["wins_vs_baselines_count"] == 0:
            modes.append("Model beats NO baselines on the primary metric — the simple "
                         "baselines are at least as good; do not claim value.")
            nexts.append("Prefer the best baseline, or add shrinkage/blend variants (#74).")
        elif not lift["beats_all_baselines"]:
            modes.append(
                f"Model beats only {lift['wins_vs_baselines_count']}/"
                f"{lift['n_baselines_comparable']} baselines — mixed evidence.")
            nexts.append("Try validation-split optimization (#73) and ensemble variants (#74).")

    if not modes:
        nexts.append("Coverage and baseline lift look reasonable — validate on more "
                     "seasons/courses before strong claims.")
    return modes, nexts


def build_error_analysis(
    predictions: pd.DataFrame,
    baseline_comparison: Optional[pd.DataFrame] = None,
    *,
    dimensions: Sequence[str] = ("target_course_slug", "predict_season",
                                 "coverage_bucket", "config_name", "par", "hole_number"),
    primary_metric: str = "spearman_pooled",
) -> dict:
    """Full error-analysis bundle: per-dimension tables, best/worst events, failures.

    Optional dimensions absent from ``predictions`` are skipped gracefully.
    """
    enriched = add_analysis_buckets(predictions)
    by_dim = {d: performance_by(enriched, d) for d in dimensions if d in enriched.columns}
    best, worst = best_worst_events(enriched)
    modes, nexts = identify_failure_modes(
        enriched, baseline_comparison, primary_metric=primary_metric)

    summary_rows = []
    for dim, frame in by_dim.items():
        if frame.empty:
            continue
        summary_rows.append({
            "dimension": dim,
            "groups": int(len(frame)),
            "mean_spearman": float(frame["spearman"].mean(skipna=True)),
            "worst_spearman": float(frame["spearman"].min(skipna=True)),
        })
    return {
        "summary": pd.DataFrame(summary_rows, columns=["dimension", "groups",
                                                       "mean_spearman", "worst_spearman"]),
        "by_dimension": by_dim,
        "by_course": by_dim.get("target_course_slug", pd.DataFrame()),
        "by_coverage_bucket": by_dim.get("coverage_bucket", pd.DataFrame()),
        "by_config": by_dim.get("config_name", pd.DataFrame()),
        "best_events": best,
        "worst_events": worst,
        "failure_modes": modes,
        "next_experiments": nexts,
    }


def render_failure_modes(error_analysis: dict, *, data_source: str = "synthetic") -> str:
    """Markdown for ``failure_modes.md``."""
    lines = ["# Failure modes & next experiments", ""]
    if data_source != "real":
        lines.append("> Synthetic data — these are structural checks, not real findings.\n")
    modes = error_analysis.get("failure_modes", [])
    lines.append("## Failure modes")
    lines += ([f"- {m}" for m in modes] or ["- None flagged."])
    lines.append("")
    lines.append("## Recommended next experiments")
    lines += [f"- {x}" for x in error_analysis.get("next_experiments", [])]
    return "\n".join(lines)


__all__ = [
    "add_analysis_buckets",
    "performance_by",
    "best_worst_events",
    "identify_failure_modes",
    "build_error_analysis",
    "render_failure_modes",
]
