"""Validation-split parameter optimizer (issue #73).

Selects player-course advantage parameters **honestly**: it tunes on validation
seasons only, never touching held-out test seasons, and refuses to recommend
parameters from synthetic data unless explicitly told it's a demo. It wraps the
existing leakage-guarded sweep/backtest — it adds no new model math.

Golf samples are small and noisy; the docs warn against over-tuning. NaN /
low-coverage parameter sets always rank below valid ones. Streamlit-free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from .backtest import run_backtest
from .schema import AdvantageParams
from .sweep import SweepGrid, run_sweep

PathLike = Union[str, Path]


class OptimizationError(ValueError):
    """Raised on an unusable optimization request (bad split, empty grid, …)."""


@dataclass(frozen=True)
class OptimizationResult:
    """Ranked parameter settings, the validation-selected recommendation, and
    (optional) held-out test metrics."""

    rankings: pd.DataFrame
    recommended_params: Optional[dict]
    validation_metrics: pd.DataFrame
    test_metrics: Optional[dict]
    primary_metric: str
    data_source: str
    warnings: list = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# Player-course advantage parameter optimization", ""]
        lines.append(f"- primary metric (validation): `{self.primary_metric}`")
        lines.append(f"- data source: `{self.data_source}`")
        if self.recommended_params is None:
            lines.append("- **No recommendation** — no eligible parameter set "
                         "(synthetic without demo opt-in, or all low-coverage).")
        else:
            picks = ", ".join(f"{k}={v}" for k, v in self.recommended_params.items())
            lines.append(f"- **Recommended:** {picks}")
        if self.test_metrics is not None:
            lines.append(f"- held-out test {self.primary_metric}: "
                         f"{self.test_metrics.get(self.primary_metric)}")
        for w in self.warnings:
            lines.append(f"> ⚠️ {w}")
        lines.append("")
        lines.append("> Golf samples are small; do not over-tune. Test metrics were "
                     "**not** used for selection.")
        return "\n".join(lines)


def optimize_parameters(
    history: pd.DataFrame,
    results: pd.DataFrame,
    similar_holes_provider,
    grid: SweepGrid,
    *,
    train_seasons: Sequence[int],
    validation_seasons: Sequence[int],
    test_seasons: Optional[Sequence[int]] = None,
    outcome_col: str = "finish_rank",
    higher_is_better: bool = False,
    primary_metric: str = "spearman_mean",
    top_k: Sequence[int] = (10, 20),
    data_source: str = "synthetic",
    allow_synthetic_recommendation: bool = False,
    season_col: str = "predict_season",
    min_coverage: float = 0.0,
    min_pairs: int = 1,
) -> OptimizationResult:
    """Grid-search on train+validation seasons, select on validation, report on test.

    Held-out ``test_seasons`` are excluded from the sweep entirely and only scored
    with the recommended params. NaN / low-coverage settings rank last. The
    recommendation is withheld for synthetic data unless
    ``allow_synthetic_recommendation`` is set.
    """
    train = set(int(s) for s in train_seasons)
    val = set(int(s) for s in validation_seasons)
    test = set(int(s) for s in (test_seasons or []))
    if val & test or train & test or train & val:
        raise OptimizationError("train/validation/test season sets must be disjoint")
    if not val:
        raise OptimizationError("validation_seasons is required for honest selection")

    seasons = pd.to_numeric(results[season_col], errors="coerce")
    sweep_results = results[seasons.isin(train | val)]
    if sweep_results.empty:
        raise OptimizationError("no results rows fall in train ∪ validation seasons")

    sweep = run_sweep(
        history, sweep_results, similar_holes_provider, grid=grid,
        outcome_col=outcome_col, higher_is_better=higher_is_better, top_k=top_k,
        validation_seasons=list(val),
    )
    val_metric = f"val_{primary_metric}"
    if val_metric not in sweep.table.columns:
        raise OptimizationError(
            f"validation metric {val_metric!r} not in sweep table; "
            f"available: {list(sweep.table.columns)}"
        )

    rankings = _rank(sweep.table, val_metric)
    warnings = list(sweep.warnings)

    recommended = None
    best_row = _eligible_best(rankings, val_metric, "val_coverage", "val_n_pairs",
                              min_coverage, min_pairs)
    synthetic_block = (data_source != "real") and not allow_synthetic_recommendation
    if synthetic_block:
        warnings.append("data_source is not 'real' — recommendation withheld "
                        "(set allow_synthetic_recommendation=True for a demo).")
    elif best_row is None:
        warnings.append("no parameter set cleared the coverage/pairs guards on validation.")
    else:
        recommended = _params_dict(best_row)

    test_metrics = None
    if recommended is not None and test:
        test_metrics = _score_on_test(
            history, results, similar_holes_provider, recommended, test,
            outcome_col, higher_is_better, top_k, season_col, primary_metric)

    val_cols = ["ablation_rank"] if "ablation_rank" in rankings.columns else []
    validation_metrics = rankings[[c for c in rankings.columns
                                   if c.startswith("val_") or c in _PARAM_COLS or c in val_cols]]
    return OptimizationResult(
        rankings=rankings, recommended_params=recommended,
        validation_metrics=validation_metrics, test_metrics=test_metrics,
        primary_metric=primary_metric, data_source=data_source, warnings=warnings,
    )


_PARAM_COLS = ("top_n", "weight_method", "config_name", "lookback_years",
               "recency_decay", "min_occurrences_per_hole", "min_holes_covered", "aggregate")


def _rank(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = table.copy()
    t["ablation_rank"] = t[metric].rank(ascending=False, method="min")
    return t.sort_values(metric, ascending=False, na_position="last").reset_index(drop=True)


def _eligible_best(rankings, metric, cov_col, pairs_col, min_coverage, min_pairs):
    df = rankings
    mask = df[metric].notna()
    if cov_col in df.columns:
        mask = mask & (df[cov_col] >= min_coverage)
    if pairs_col in df.columns:
        mask = mask & (df[pairs_col] >= min_pairs)
    elig = df[mask]
    return None if elig.empty else elig.iloc[0]


def _params_dict(row) -> dict:
    return {c: (int(row[c]) if c in ("top_n", "lookback_years",
                                     "min_occurrences_per_hole", "min_holes_covered")
                else row[c])
            for c in _PARAM_COLS if c in row.index}


def _score_on_test(history, results, provider, recommended, test, outcome_col,
                   higher_is_better, top_k, season_col, primary_metric) -> dict:
    seasons = pd.to_numeric(results[season_col], errors="coerce")
    test_results = results[seasons.isin(test)]
    params = AdvantageParams(
        n=int(recommended.get("top_n", 10)),
        lookback_years=int(recommended.get("lookback_years", 5)),
        recency_decay=float(recommended.get("recency_decay", 0.85)),
        min_occurrences_per_hole=int(recommended.get("min_occurrences_per_hole", 3)),
        min_holes_covered=int(recommended.get("min_holes_covered", 12)),
    )
    sim = provider(recommended.get("config_name", "baseline"),
                   int(recommended.get("top_n", 10)),
                   recommended.get("weight_method", "rank_decay"))
    bt = run_backtest(history, sim, test_results, outcome_col=outcome_col,
                      higher_is_better=higher_is_better, params=params,
                      config_name=recommended.get("config_name", "baseline"),
                      aggregate=recommended.get("aggregate", "sum"), top_k=top_k)
    return {primary_metric: bt.summary.get(primary_metric),
            "coverage": bt.summary.get("coverage"), "n_pairs": bt.summary.get("n_pairs")}


def export_optimization(result: OptimizationResult, out_dir: PathLike) -> dict[str, Path]:
    """Write optimization tables + recommendation + markdown to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    result.rankings.to_csv(out / "optimization_summary.csv", index=False)
    result.rankings.to_csv(out / "parameter_rankings.csv", index=False)
    result.validation_metrics.to_csv(out / "validation_metrics.csv", index=False)
    paths["optimization_summary"] = out / "optimization_summary.csv"
    paths["parameter_rankings"] = out / "parameter_rankings.csv"
    paths["validation_metrics"] = out / "validation_metrics.csv"
    if result.test_metrics is not None:
        pd.DataFrame([result.test_metrics]).to_csv(out / "test_metrics.csv", index=False)
        paths["test_metrics"] = out / "test_metrics.csv"
    (out / "recommended_params.json").write_text(
        json.dumps(result.recommended_params, indent=2, default=str), encoding="utf-8")
    (out / "optimization_report.md").write_text(result.to_markdown(), encoding="utf-8")
    paths["recommended_params"] = out / "recommended_params.json"
    paths["report"] = out / "optimization_report.md"
    return paths


__all__ = [
    "OptimizationError",
    "OptimizationResult",
    "optimize_parameters",
    "export_optimization",
]
