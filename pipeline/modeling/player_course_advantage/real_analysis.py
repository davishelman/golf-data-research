"""Real-data analysis runner: ingestion → health → evaluation (issue #72).

One repeatable call that takes a **validated canonical history** plus v2.5
similar-hole sets and actual event outcomes, and produces the full analysis
bundle (data health, leakage-free backtest, baseline comparison, calibration,
optional sweep/benchmarks, evaluation report + manifest).

It does **not** source real data — it consumes what the caller provides. It
refuses predictive evaluation when history validation fails, stops early when
input coverage is below a configured minimum, writes only under the caller's
output directory, and labels ``data_source=real`` **only** when the caller says
so. Pure orchestration over shipped modules; Streamlit-free.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .backtest import run_backtest
from .baselines import compare_baselines
from .ablation import baseline_lift_summary, baseline_lift_table
from .benchmarks import run_benchmarks as _run_benchmarks
from .calibration import calibration_table, reliability_by_coverage
from .data_health import build_data_health_report, data_health_to_frames
from .evaluation import build_evaluation_summary, render_evaluation_markdown
from .schema import DEFAULT_PARAMS, AdvantageParams, validate_hole_score_history
from .sweep import SweepGrid, run_sweep as _run_sweep

PathLike = Union[str, Path]

# The predictive outputs that only exist once the backtest runs.
_PREDICTIVE_OUTPUTS = (
    "backtest_predictions.csv", "backtest_per_event.csv", "coverage_by_event.csv",
    "baseline_comparison.csv", "baseline_lift.csv", "calibration_table.csv",
    "reliability_by_coverage.csv",
)


def _write_csv(path: Path, df: pd.DataFrame, produced: dict) -> None:
    df.to_csv(path, index=False)
    produced[path.name] = int(len(df))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _pre_coverage(health: dict) -> float:
    sim = health.get("similarity_coverage", {})
    denom = sim.get("candidate_holes") or 0
    return (sim.get("candidate_holes_with_history", 0) / denom) if denom else 0.0


def run_real_analysis(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: PathLike,
    *,
    data_source: str = "synthetic",
    params: AdvantageParams = DEFAULT_PARAMS,
    config_name: str = "baseline",
    outcome_col: str = "finish_rank",
    higher_is_better: bool = False,
    aggregate: str = "sum",
    top_k: tuple = (10, 20),
    min_coverage: float = 0.0,
    do_sweep: bool = False,
    do_benchmarks: bool = False,
) -> dict:
    """Run the analysis bundle to ``output_dir`` and return the manifest dict.

    Raises
    :class:`~pipeline.modeling.player_course_advantage.schema.SchemaError` if
    ``history`` is invalid (predictive evaluation is refused). If similarity
    coverage < ``min_coverage`` the run **stops early** after the data-health
    step. Never mutates inputs; writes only under ``output_dir``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    produced: dict[str, int] = {}
    missing: dict[str, str] = {}

    # 1. Validate — refuse predictive evaluation on invalid history.
    validate_hole_score_history(history)

    # 2. Data health (cheap, pre-backtest).
    health = build_data_health_report(history, similar_holes, data_source=data_source)
    _write_csv(out / "data_health_summary.csv",
               data_health_to_frames(health)["data_quality_summary"], produced)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "real" if data_source == "real" else "synthetic",
        "data_source": data_source,
        "real_data_available": data_source == "real",
        "history_rows": int(len(history)),
        "similar_hole_rows": int(len(similar_holes)),
        "players": int(history["player_id"].nunique()) if "player_id" in history else 0,
        "courses": int(history["course_slug"].nunique()) if "course_slug" in history else 0,
        "events": int(results["event_id"].nunique()) if "event_id" in results else 0,
        "min_coverage": min_coverage,
        "input_coverage": _pre_coverage(health),
        "warnings": list(health.get("warnings", [])),
    }

    # 3. Early stop if inputs don't connect well enough to trust anything.
    if manifest["input_coverage"] < min_coverage:
        manifest["stopped_early"] = True
        manifest["stop_reason"] = (
            f"input coverage {manifest['input_coverage']:.3f} < min_coverage {min_coverage}"
        )
        for name in _PREDICTIVE_OUTPUTS + ("sweep_summary.csv", "benchmark_summary.csv"):
            missing[name] = manifest["stop_reason"]
        _write_json(out / "missing_outputs.json", missing)
        _write_json(out / "analysis_manifest.json", manifest)
        (out / "analysis_report.md").write_text(
            _report_md(data_source, health, None, None, None, stopped=manifest["stop_reason"]),
            encoding="utf-8")
        return manifest

    # 4. Leakage-free backtest.
    bt = run_backtest(history, similar_holes, results, outcome_col=outcome_col,
                      higher_is_better=higher_is_better, params=params,
                      config_name=config_name, aggregate=aggregate, top_k=top_k)
    _write_csv(out / "backtest_predictions.csv", bt.predictions, produced)
    _write_csv(out / "backtest_per_event.csv", bt.per_event, produced)
    cov_cols = [c for c in ("event_id", "predict_season", "target_course_slug",
                            "n_field", "n_covered", "coverage") if c in bt.per_event.columns]
    _write_csv(out / "coverage_by_event.csv", bt.per_event[cov_cols], produced)

    # 5. Baselines + lift.
    comp = compare_baselines(history, similar_holes, results, outcome_col=outcome_col,
                             higher_is_better=higher_is_better, params=params, top_k=top_k)
    _write_csv(out / "baseline_comparison.csv", comp, produced)
    _write_csv(out / "baseline_lift.csv", baseline_lift_table(comp), produced)
    lift = baseline_lift_summary(comp)

    # 6. Optional parameter sweep.
    sweep = None
    if do_sweep:
        grid = SweepGrid(recency_decay=(params.recency_decay, 1.0),
                         lookback_years=(params.lookback_years,),
                         min_holes_covered=(params.min_holes_covered,))
        sweep = _run_sweep(history, results, lambda c, n, w: similar_holes, grid=grid,
                           outcome_col=outcome_col, higher_is_better=higher_is_better, top_k=top_k)
        _write_csv(out / "sweep_summary.csv", sweep.table, produced)
    else:
        missing["sweep_summary.csv"] = "sweep not requested (do_sweep=False)"

    # 7. Calibration / reliability.
    cal = calibration_table(bt.predictions)
    _write_csv(out / "calibration_table.csv", cal, produced)
    rel = reliability_by_coverage(bt.predictions)
    _write_csv(out / "reliability_by_coverage.csv", pd.DataFrame([rel]), produced)

    # 8. Evaluation summary + human report.
    summary = build_evaluation_summary(bt, comp, sweep, health, data_source=data_source)
    (out / "analysis_report.md").write_text(
        _report_md(data_source, health, bt, comp, summary, lift=lift), encoding="utf-8")
    produced["analysis_report.md"] = 1

    # 9. Optional benchmarks.
    if do_benchmarks:
        bench = _run_benchmarks(field_sizes=(10,), event_counts=(1,))
        _write_csv(out / "benchmark_summary.csv", bench, produced)
    else:
        missing["benchmark_summary.csv"] = "benchmarks not requested (do_benchmarks=False)"

    manifest.update({
        "stopped_early": False,
        "spearman_pooled": bt.summary.get("spearman_pooled"),
        "coverage": bt.summary.get("coverage"),
        "wins_vs_baselines_count": lift.get("wins_vs_baselines_count"),
        "beats_all_baselines": lift.get("beats_all_baselines"),
        "outputs": sorted(produced),
    })
    _write_json(out / "missing_outputs.json", missing)
    _write_json(out / "analysis_manifest.json", manifest)
    return manifest


def _report_md(data_source, health, bt, comp, summary, *, lift=None, stopped=None) -> str:
    real = data_source == "real"
    lines = ["# Player-course advantage analysis report", ""]
    lines.append(f"- **Data source:** `{data_source}`"
                 + ("" if real else "  ⚠️ synthetic — validates plumbing only, not predictive edge."))
    if stopped:
        lines += ["", f"> ⛔ **Stopped early:** {stopped}",
                  "> Fix input coverage before running predictive evaluation."]
        return "\n".join(lines)
    if summary is not None:
        lines.append("")
        lines.append(render_evaluation_markdown(summary))
    if not real:
        lines += ["", "> Predictive validity was **not** evaluated (synthetic data)."]
    return "\n".join(lines)


__all__ = ["run_real_analysis"]
