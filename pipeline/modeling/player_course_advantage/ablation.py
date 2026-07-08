"""Baseline lift & ablation analysis (issue #60).

Two honest questions:

1. **Lift** — does the model beat simple baselines, and by how much? Given a
   ``compare_baselines`` table, compute per-baseline lift on a primary metric,
   how many baselines the model strictly beats, and its rank.
2. **Ablation** — which parts of the model matter? Given a ``run_sweep`` table,
   rank the parameter/config settings and summarize what helped or hurt.

Reuses the shipped ``compare_baselines`` / ``run_sweep`` outputs rather than
recomputing, is NaN-safe throughout, and **never counts a tie as a win**. Pure,
Streamlit-free, synthetic-testable — a positive lift on synthetic data proves
nothing and is labelled as such by callers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

MODEL_NAME = "similar_hole_model"


# --------------------------------------------------------------------------- #
# Baseline lift
# --------------------------------------------------------------------------- #
def baseline_lift_table(
    comparison: pd.DataFrame,
    *,
    metric: str = "spearman_pooled",
    model_name: str = MODEL_NAME,
) -> pd.DataFrame:
    """Per-baseline lift on ``metric``: ``model_value - baseline_value`` + win flag.

    A ``model_wins`` is only ``True`` when both values are non-NaN and the model is
    **strictly** greater (ties are not wins). Returns one row per non-model ranker.
    """
    cols = ["baseline", "baseline_value", "model_value", "lift", "model_wins"]
    if "ranker" not in comparison.columns or model_name not in set(comparison["ranker"]):
        return pd.DataFrame(columns=cols)
    model_val = comparison.loc[comparison["ranker"] == model_name, metric].iloc[0]

    rows = []
    for _, r in comparison[comparison["ranker"] != model_name].iterrows():
        bval = r.get(metric)
        both = pd.notna(model_val) and pd.notna(bval)
        rows.append({
            "baseline": r["ranker"],
            "baseline_value": bval,
            "model_value": model_val,
            "lift": (model_val - bval) if both else float("nan"),
            "model_wins": bool(both and model_val > bval),
        })
    return pd.DataFrame(rows, columns=cols)


def baseline_lift_summary(
    comparison: pd.DataFrame,
    *,
    metric: str = "spearman_pooled",
    model_name: str = MODEL_NAME,
) -> dict:
    """Scalar lift verdict: wins count, whether the model beats *all* baselines, rank.

    ``beats_all_baselines`` requires at least one comparable baseline and a strict
    win over every comparable one — honest by construction.
    """
    tbl = baseline_lift_table(comparison, metric=metric, model_name=model_name)
    comparable = int(tbl["lift"].notna().sum()) if not tbl.empty else 0
    wins = int(tbl["model_wins"].sum()) if not tbl.empty else 0

    model_rank = None
    if "ranker" in comparison.columns and metric in comparison.columns:
        ranked = comparison.dropna(subset=[metric]).sort_values(
            metric, ascending=False
        ).reset_index(drop=True)
        hit = ranked.index[ranked["ranker"] == model_name].tolist()
        model_rank = int(hit[0] + 1) if hit else None

    model_val = (
        comparison.loc[comparison["ranker"] == model_name, metric].iloc[0]
        if model_name in set(comparison.get("ranker", [])) else float("nan")
    )
    return {
        "metric": metric,
        "model_value": (float(model_val) if pd.notna(model_val) else float("nan")),
        "wins_vs_baselines_count": wins,
        "n_baselines_comparable": comparable,
        "beats_all_baselines": bool(comparable > 0 and wins == comparable),
        "model_rank": model_rank,
        "n_rankers": int(len(comparison)),
    }


# --------------------------------------------------------------------------- #
# Ablation over a sweep table
# --------------------------------------------------------------------------- #
def _sweep_table(sweep) -> pd.DataFrame:
    return sweep.table if hasattr(sweep, "table") else sweep


def rank_ablation(
    sweep, *, primary_metric: str = "spearman_pooled"
) -> pd.DataFrame:
    """Rank every swept setting by ``primary_metric`` (best first, NaN last)."""
    table = _sweep_table(sweep).copy()
    if primary_metric not in table.columns:
        raise ValueError(f"sweep table has no metric column {primary_metric!r}")
    table["ablation_rank"] = table[primary_metric].rank(ascending=False, method="min")
    return table.sort_values(
        primary_metric, ascending=False, na_position="last"
    ).reset_index(drop=True)


def ablation_effects(
    sweep,
    *,
    primary_metric: str = "spearman_pooled",
    param_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-parameter effect: best value and metric spread across that knob.

    A large ``spread`` means the model is sensitive to that knob (it *matters*); a
    tiny spread means it barely moves the metric.
    """
    table = _sweep_table(sweep)
    default_params = ("top_n", "weight_method", "config_name", "lookback_years",
                      "recency_decay", "min_occurrences_per_hole",
                      "min_holes_covered", "aggregate")
    cols = [c for c in (param_cols or default_params) if c in table.columns]
    rows = []
    for col in cols:
        if table[col].nunique(dropna=True) < 2:
            continue  # knob wasn't varied — no effect to report
        g = table.groupby(col)[primary_metric].mean()
        has = g.notna().any()
        rows.append({
            "param": col,
            "best_value": (g.idxmax() if has else None),
            "worst_value": (g.idxmin() if has else None),
            "spread": (float(g.max() - g.min()) if has else float("nan")),
        })
    return pd.DataFrame(rows, columns=["param", "best_value", "worst_value", "spread"])


def render_ablation_markdown(
    lift_summary: dict, effects: Optional[pd.DataFrame] = None
) -> str:
    lines = ["## Baseline lift & ablation", ""]
    verdict = "beats **all** comparable baselines" if lift_summary.get("beats_all_baselines") \
        else "does **not** beat all baselines"
    lines.append(
        f"- On `{lift_summary['metric']}`, the model {verdict} "
        f"({lift_summary['wins_vs_baselines_count']}/{lift_summary['n_baselines_comparable']} "
        f"wins; rank {lift_summary['model_rank']} of {lift_summary['n_rankers']})."
    )
    if effects is not None and not effects.empty:
        top = effects.sort_values("spread", ascending=False).iloc[0]
        lines.append(
            f"- Most influential knob: `{top['param']}` (metric spread "
            f"{top['spread']:.3f}, best at `{top['best_value']}`)."
        )
    lines.append("")
    lines.append("> Synthetic unless a real, leakage-free label set is supplied — "
                 "lift here is not evidence of predictive validity.")
    return "\n".join(lines)


__all__ = [
    "MODEL_NAME",
    "baseline_lift_table",
    "baseline_lift_summary",
    "rank_ablation",
    "ablation_effects",
    "render_ablation_markdown",
]
