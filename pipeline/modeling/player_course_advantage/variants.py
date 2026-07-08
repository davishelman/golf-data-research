"""Baseline-ensemble & shrinkage variants (issue #74).

When the similar-hole signal is thin or noisy, a raw advantage can be an
over-confident extreme. These variants make the score more robust — shrinking
low-coverage predictions toward zero and blending with simple baselines — while
staying interpretable (component weights are explicit) and honest (blend/ensemble
weights are chosen on **validation only**, never on held-out test).

All functions are pure and operate on a backtest-style predictions frame
(``course_advantage``, ``performance``, ``covered``, ``holes_covered``,
``total_raw_occurrences``, ``total_weighted_occurrences`` [+ optional baseline
advantage columns]). If a variant loses to the plain model, the report says so.
Streamlit-free.
"""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import pandas as pd

from .backtest import spearman_corr

Variant = Callable[[pd.DataFrame], pd.Series]
BASE_COL = "course_advantage"


# --------------------------------------------------------------------------- #
# Shrinkage / blend primitives (pure)
# --------------------------------------------------------------------------- #
def shrink_toward_zero(advantage: pd.Series, factor: float = 0.5) -> pd.Series:
    """Multiply by ``factor`` in [0, 1] — 0 kills the signal, 1 is the raw model."""
    return advantage.astype(float) * float(factor)


def shrink_by_count(advantage: pd.Series, counts: pd.Series, k: float = 5.0) -> pd.Series:
    """James-Stein-style shrink: ``advantage · count / (count + k)``.

    Low counts (thin coverage) are pulled hard toward zero; high counts are barely
    touched. ``k`` sets the half-shrink point.
    """
    counts = pd.to_numeric(counts, errors="coerce").astype(float)
    return advantage.astype(float) * (counts / (counts + float(k)))


def blend(advantage: pd.Series, baseline: pd.Series, alpha: float = 0.5) -> pd.Series:
    """Convex blend ``alpha·advantage + (1-alpha)·baseline`` (alpha in [0, 1])."""
    return float(alpha) * advantage.astype(float) + (1.0 - float(alpha)) * baseline.astype(float)


# --------------------------------------------------------------------------- #
# Built-in variants
# --------------------------------------------------------------------------- #
def default_variants(baseline_cols: Sequence[str] = ()) -> dict[str, Variant]:
    """The standard variant set (plus a blend per supplied baseline column)."""
    variants: dict[str, Variant] = {
        "similar_only": lambda d: d[BASE_COL].astype(float),
        "shrunk_zero": lambda d: shrink_toward_zero(d[BASE_COL], 0.5),
        "shrunk_by_holes": lambda d: shrink_by_count(d[BASE_COL], d["holes_covered"], 5.0),
        "shrunk_by_raw_occ": lambda d: shrink_by_count(d[BASE_COL], d["total_raw_occurrences"], 10.0),
        "shrunk_by_weighted_occ": lambda d: shrink_by_count(d[BASE_COL], d["total_weighted_occurrences"], 10.0),
    }
    for col in baseline_cols:
        variants[f"blend_{col}"] = (lambda c: (lambda d: blend(d[BASE_COL], d[c], 0.5)))(col)
    return variants


def variant_components() -> pd.DataFrame:
    """Interpretable description of each built-in variant's shrink/blend weights."""
    return pd.DataFrame([
        {"variant": "similar_only", "kind": "raw", "param": None},
        {"variant": "shrunk_zero", "kind": "shrink_constant", "param": 0.5},
        {"variant": "shrunk_by_holes", "kind": "shrink_by_holes_covered", "param": 5.0},
        {"variant": "shrunk_by_raw_occ", "kind": "shrink_by_raw_occurrences", "param": 10.0},
        {"variant": "shrunk_by_weighted_occ", "kind": "shrink_by_weighted_occurrences", "param": 10.0},
    ])


# --------------------------------------------------------------------------- #
# Comparison (validation / test)
# --------------------------------------------------------------------------- #
def _covered(df: pd.DataFrame) -> pd.Series:
    return df["covered"].fillna(False) if "covered" in df.columns else pd.Series(True, index=df.index)


def _metric(df: pd.DataFrame, score: pd.Series) -> float:
    mask = _covered(df) & df["performance"].notna() & score.notna()
    if int(mask.sum()) < 2:
        return float("nan")
    return spearman_corr(score[mask].to_numpy(dtype=float),
                         df.loc[mask, "performance"].to_numpy(dtype=float))


def compare_variants(
    predictions: pd.DataFrame,
    variants: Optional[Mapping[str, Variant]] = None,
    *,
    validation_seasons: Sequence[int],
    test_seasons: Optional[Sequence[int]] = None,
    season_col: str = "predict_season",
) -> pd.DataFrame:
    """Score each variant on validation (and optional test) seasons; rank by validation.

    The metric is Spearman(variant advantage, realized performance) over covered
    rows. Selection should use ``validation_metric``; ``test_metric`` is reported
    for honesty, never for selection.
    """
    variants = dict(variants) if variants is not None else default_variants()
    seasons = pd.to_numeric(predictions[season_col], errors="coerce")
    val = predictions[seasons.isin(set(int(s) for s in validation_seasons))]
    test = (predictions[seasons.isin(set(int(s) for s in test_seasons))]
            if test_seasons else None)

    rows = []
    for name, fn in variants.items():
        row = {"variant": name,
               "validation_metric": _metric(val, fn(val)),
               "n_validation": int(_covered(val).sum())}
        if test is not None:
            row["test_metric"] = _metric(test, fn(test))
            row["n_test"] = int(_covered(test).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "validation_metric", ascending=False, na_position="last").reset_index(drop=True)


def best_variant_by_validation(comparison: pd.DataFrame) -> Optional[str]:
    """Name of the top variant by validation metric (``None`` if all NaN)."""
    valid = comparison.dropna(subset=["validation_metric"])
    return None if valid.empty else str(valid.iloc[0]["variant"])


def select_blend_alpha(
    predictions: pd.DataFrame,
    baseline_col: str,
    *,
    validation_seasons: Sequence[int],
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    season_col: str = "predict_season",
) -> dict:
    """Pick the model/baseline blend weight on **validation only**.

    Returns ``{best_alpha, best_validation_metric, table}``. ``alpha=1`` is the
    plain model, ``alpha=0`` the pure baseline.
    """
    seasons = pd.to_numeric(predictions[season_col], errors="coerce")
    val = predictions[seasons.isin(set(int(s) for s in validation_seasons))]
    table, best = [], (None, float("-inf"))
    for a in alphas:
        m = _metric(val, blend(val[BASE_COL], val[baseline_col], a))
        table.append({"alpha": a, "validation_metric": m})
        if pd.notna(m) and m > best[1]:
            best = (a, m)
    return {
        "best_alpha": best[0],
        "best_validation_metric": best[1] if best[0] is not None else float("nan"),
        "table": pd.DataFrame(table),
    }


def render_variant_summary(comparison: pd.DataFrame, *, data_source: str = "synthetic") -> str:
    """Honest markdown: did any variant beat the plain model on validation?"""
    best = best_variant_by_validation(comparison)
    base = comparison[comparison["variant"] == "similar_only"]
    base_val = float(base["validation_metric"].iloc[0]) if not base.empty else float("nan")
    lines = ["## Variant comparison", ""]
    if best is None:
        lines.append("- No variant produced a valid validation metric.")
    else:
        best_val = float(comparison.iloc[0]["validation_metric"])
        if best == "similar_only" or not (best_val > base_val):
            lines.append(f"- The **plain model** (`similar_only`) is best on validation "
                         f"({base_val:.3f}); shrinkage/blend did **not** help — keep the simple model.")
        else:
            lines.append(f"- `{best}` beats the plain model on validation "
                         f"({best_val:.3f} vs {base_val:.3f}) — shrinkage/blend helped.")
    if data_source != "real":
        lines.append("")
        lines.append("> Synthetic data — not a predictive-validity claim.")
    return "\n".join(lines)


__all__ = [
    "shrink_toward_zero",
    "shrink_by_count",
    "blend",
    "default_variants",
    "variant_components",
    "compare_variants",
    "best_variant_by_validation",
    "select_blend_alpha",
    "render_variant_summary",
]
