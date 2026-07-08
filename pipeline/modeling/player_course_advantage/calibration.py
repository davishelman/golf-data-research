"""Calibration & reliability analysis (issue #61).

Two questions beyond rank correlation:

1. **Calibration** — do *larger* predicted advantages correspond to *better*
   realized outcomes? Bucket predictions, compare per-bucket mean prediction vs
   mean performance, and check monotonicity / slope.
2. **Reliability** — are higher-coverage predictions more trustworthy? Split by
   holes covered / raw / weighted occurrences and compare rank correlation.

Operates on a backtest predictions frame (``course_advantage``, ``performance``,
``covered``, ``holes_covered``, ``total_raw_occurrences``,
``total_weighted_occurrences``). Degrades gracefully on small/sparse samples
(returns ``NaN`` + warnings, never crashes). Synthetic monotonic data scores
well, scrambled data worse — but this is **not** a calibration claim on real data.
Pure, Streamlit-free.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .backtest import spearman_corr

_MIN_ROWS = 4  # below this, calibration is not meaningful


def _covered(predictions: pd.DataFrame) -> pd.DataFrame:
    """Covered rows with a usable prediction + performance."""
    df = predictions
    if "covered" in df.columns:
        df = df[df["covered"].fillna(False)]
    keep = df
    for col in ("course_advantage", "performance"):
        if col in keep.columns:
            keep = keep[keep[col].notna()]
    return keep


def bucket_predictions(
    predictions: pd.DataFrame, *, n_buckets: int = 5, score_col: str = "course_advantage"
) -> pd.DataFrame:
    """Return covered rows with a ``bucket`` index (0 = lowest predicted advantage).

    Uses quantile bins; collapses to fewer bins when advantage has few distinct
    values. Empty/degenerate input yields an empty frame (with a ``bucket`` column).
    """
    cov = _covered(predictions).copy()
    if cov.empty or cov[score_col].nunique() < 2:
        cov["bucket"] = pd.Series(dtype="int")
        return cov
    q = min(n_buckets, cov[score_col].nunique())
    try:
        cov["bucket"] = pd.qcut(cov[score_col], q=q, labels=False, duplicates="drop")
    except ValueError:
        cov["bucket"] = 0
    cov["bucket"] = cov["bucket"].astype(int)
    return cov


def calibration_table(
    predictions: pd.DataFrame, *, n_buckets: int = 5, score_col: str = "course_advantage"
) -> pd.DataFrame:
    """Per-bucket mean prediction vs mean realized performance (+ counts/coverage)."""
    cols = ["bucket", "n", "mean_prediction", "mean_performance", "mean_holes_covered"]
    bucketed = bucket_predictions(predictions, n_buckets=n_buckets, score_col=score_col)
    if bucketed.empty:
        return pd.DataFrame(columns=cols)
    hc = bucketed["holes_covered"] if "holes_covered" in bucketed.columns else np.nan
    bucketed = bucketed.assign(_hc=hc)
    grouped = bucketed.groupby("bucket").agg(
        n=(score_col, "size"),
        mean_prediction=(score_col, "mean"),
        mean_performance=("performance", "mean"),
        mean_holes_covered=("_hc", "mean"),
    ).reset_index()
    return grouped[cols]


def monotonicity_score(table: pd.DataFrame) -> float:
    """Rank correlation between bucket mean prediction and mean performance.

    ``+1`` = perfectly monotonic (bigger advantage → better outcome); ``NaN`` when
    there are fewer than two buckets.
    """
    if table is None or len(table) < 2:
        return float("nan")
    return spearman_corr(table["mean_prediction"], table["mean_performance"])


def calibration_slope(
    predictions: pd.DataFrame, *, score_col: str = "course_advantage"
) -> float:
    """OLS slope of ``performance`` on predicted advantage (``NaN`` if under-determined)."""
    cov = _covered(predictions)
    if len(cov) < 2:
        return float("nan")
    x = cov[score_col].to_numpy(dtype=float)
    y = cov["performance"].to_numpy(dtype=float)
    if np.std(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def reliability_by_coverage(
    predictions: pd.DataFrame,
    *,
    by: str = "holes_covered",
    threshold: Optional[float] = None,
) -> dict:
    """Rank correlation among high- vs low-coverage predictions.

    Splits covered rows at ``threshold`` (default: the median of ``by``) and
    reports Spearman in each half. If the model is reliable, the high-coverage
    half should correlate at least as strongly. Returns ``NaN`` correlations when a
    half is too small.
    """
    cov = _covered(predictions)
    out = {"by": by, "threshold": None, "n_high": 0, "n_low": 0,
           "high_spearman": float("nan"), "low_spearman": float("nan")}
    if cov.empty or by not in cov.columns:
        return out
    thr = float(cov[by].median()) if threshold is None else float(threshold)
    high = cov[cov[by] >= thr]
    low = cov[cov[by] < thr]
    out["threshold"] = thr
    out["n_high"], out["n_low"] = int(len(high)), int(len(low))
    if len(high) >= 2:
        out["high_spearman"] = spearman_corr(high["course_advantage"], high["performance"])
    if len(low) >= 2:
        out["low_spearman"] = spearman_corr(low["course_advantage"], low["performance"])
    return out


def render_calibration_summary(
    predictions: pd.DataFrame, *, n_buckets: int = 5, data_source: str = "synthetic"
) -> dict:
    """Bundle calibration + reliability into one dict with warnings + markdown."""
    cov = _covered(predictions)
    warnings: list[str] = []
    if len(cov) < _MIN_ROWS:
        warnings.append(
            f"only {len(cov)} covered predictions — calibration is not meaningful "
            f"below {_MIN_ROWS} rows; treat slope/monotonicity as noise."
        )
    table = calibration_table(predictions, n_buckets=n_buckets)
    slope = calibration_slope(predictions)
    mono = monotonicity_score(table)
    reliability = reliability_by_coverage(predictions)

    md = "\n".join([
        "## Calibration & reliability",
        f"- slope(performance ~ advantage): {slope:.3f}",
        f"- monotonicity (bucket rank corr): {mono:.3f}",
        f"- reliability by {reliability['by']}: high={reliability['high_spearman']:.3f} "
        f"vs low={reliability['low_spearman']:.3f}",
        f"> data_source=`{data_source}`"
        + ("" if data_source == "real" else " — synthetic, not a calibration claim."),
    ])
    return {
        "data_source": data_source,
        "n_covered": int(len(cov)),
        "calibration_table": table,
        "calibration_slope": slope,
        "monotonicity_score": mono,
        "reliability_by_coverage": reliability,
        "warnings": warnings,
        "markdown": md,
    }


__all__ = [
    "bucket_predictions",
    "calibration_table",
    "monotonicity_score",
    "calibration_slope",
    "reliability_by_coverage",
    "render_calibration_summary",
]
