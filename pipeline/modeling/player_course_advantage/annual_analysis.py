"""Batch annual real-analysis runner (issue #82).

Runs the full player-course advantage evaluation across **every supported annual
course that has validated data**, then aggregates the results into one report that
answers: which courses have enough data, which are missing scores/outcomes, where
the model beats (or loses to) baselines, whether higher predicted advantage tracks
better outcomes, and which courses are failure modes.

It orchestrates the shipped per-course runner (:func:`.real_analysis.run_real_analysis`)
plus calibration / error-analysis helpers — no new model math. Sources no data;
writes only under the caller's output directory; makes no predictive claim on
synthetic data. Streamlit-free.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from .calibration import calibration_slope, calibration_table, monotonicity_score
from .course_targets import supported_targets
from .error_analysis import build_error_analysis
from .real_analysis import run_real_analysis
from .schema import DEFAULT_PARAMS, AdvantageParams, SchemaError

PathLike = Union[str, Path]
SimilarHolesProvider = Callable[[str], Optional[pd.DataFrame]]

_STATUS_COLUMNS = ("course_slug", "supported", "status", "n_events", "coverage",
                   "spearman_pooled", "beats_all_baselines", "reason")


def run_annual_analysis(
    targets: pd.DataFrame,
    history: pd.DataFrame,
    outcomes: pd.DataFrame,
    similar_holes_provider: SimilarHolesProvider,
    out_root: PathLike,
    *,
    data_source: str = "synthetic",
    params: AdvantageParams = DEFAULT_PARAMS,
    config_name: str = "baseline",
    outcome_col: str = "finish_position",
    higher_is_better: bool = False,
    top_k: tuple = (10, 20),
    min_coverage: float = 0.0,
    do_sweep: bool = False,
    season_col: str = "season",
    course_col: str = "course_slug",
) -> dict:
    """Evaluate every supported course with data; write per-course + aggregate outputs.

    ``similar_holes_provider(course_slug)`` returns that course's v2.5 similar-hole
    set (or ``None``). ``outcomes`` is the actual event-results table (with
    ``event_id``, ``season``, ``course_slug``, ``player_id``, ``outcome_col``).
    Returns the aggregate manifest dict. Never mutates inputs.
    """
    out = Path(out_root)
    (out / "per_course").mkdir(parents=True, exist_ok=True)

    status_rows: list[dict] = []
    model_rows: list[dict] = []
    baseline_frames: list[pd.DataFrame] = []
    health_frames: list[pd.DataFrame] = []
    calib_rows: list[dict] = []
    error_rows: list[dict] = []
    failure_modes: dict[str, list] = {}
    sweep_frames: list[pd.DataFrame] = []

    supported = set(supported_targets(targets)["course_slug"])
    for r in targets.itertuples(index=False):
        slug = getattr(r, "course_slug")
        rec = {"course_slug": slug, "supported": slug in supported, "status": "unsupported",
               "n_events": 0, "coverage": None, "spearman_pooled": None,
               "beats_all_baselines": None, "reason": ""}
        if slug not in supported:
            rec["reason"] = "no v2.5 similarity outputs"
            status_rows.append(rec)
            continue

        sim = similar_holes_provider(slug)
        if sim is None or len(sim) == 0:
            rec.update(status="no_similarity", reason="similar-hole set unavailable")
            status_rows.append(rec)
            continue

        results_c = _course_results(outcomes, slug, course_col, season_col)
        rec["n_events"] = int(results_c["event_id"].nunique()) if len(results_c) else 0
        if results_c.empty:
            rec.update(status="no_event_outcomes",
                       reason="no event outcomes — real backtest blocked")
            status_rows.append(rec)
            continue

        course_dir = out / "per_course" / slug
        try:
            manifest_c = run_real_analysis(
                history, sim, results_c, course_dir, data_source=data_source,
                params=params, config_name=config_name, outcome_col=outcome_col,
                higher_is_better=higher_is_better, top_k=top_k,
                min_coverage=min_coverage, do_sweep=do_sweep)
        except SchemaError as exc:
            rec.update(status="validation_failed", reason=f"history invalid: {exc}")
            status_rows.append(rec)
            continue

        rec["coverage"] = manifest_c.get("coverage")
        rec["spearman_pooled"] = manifest_c.get("spearman_pooled")
        rec["beats_all_baselines"] = manifest_c.get("beats_all_baselines")
        rec["status"] = "low_coverage" if manifest_c.get("stopped_early") else "evaluated"
        if manifest_c.get("stopped_early"):
            rec["reason"] = manifest_c.get("stop_reason", "coverage below minimum")
        status_rows.append(rec)

        _aggregate_course(course_dir, slug, model_rows, baseline_frames, health_frames,
                          calib_rows, error_rows, failure_modes, sweep_frames)

    manifest = _write_aggregate(
        out, status_rows, model_rows, baseline_frames, health_frames, calib_rows,
        error_rows, failure_modes, sweep_frames, data_source, do_sweep)
    return manifest


def _course_results(outcomes, slug, course_col, season_col) -> pd.DataFrame:
    if course_col not in outcomes.columns:
        return pd.DataFrame()
    c = outcomes[outcomes[course_col].astype(str) == slug].copy()
    if c.empty:
        return c
    c["target_course_slug"] = slug
    if season_col in c.columns:
        c["predict_season"] = pd.to_numeric(c[season_col], errors="coerce")
    if "event_id" not in c.columns:
        c["event_id"] = c.get("event_name", slug).astype(str) + "_" + c["predict_season"].astype(str)
    return c


def _aggregate_course(course_dir, slug, model_rows, baseline_frames, health_frames,
                      calib_rows, error_rows, failure_modes, sweep_frames) -> None:
    preds_p = course_dir / "backtest_predictions.csv"
    comp_p = course_dir / "baseline_comparison.csv"
    health_p = course_dir / "data_health_summary.csv"
    sweep_p = course_dir / "sweep_summary.csv"

    if health_p.exists():
        h = pd.read_csv(health_p); h.insert(0, "course_slug", slug); health_frames.append(h)
    if sweep_p.exists():
        s = pd.read_csv(sweep_p); s.insert(0, "course_slug", slug); sweep_frames.append(s)
    if comp_p.exists():
        c = pd.read_csv(comp_p); c.insert(0, "course_slug", slug); baseline_frames.append(c)
    if not preds_p.exists():
        return
    preds = pd.read_csv(preds_p)
    table = calibration_table(preds)
    calib_rows.append({
        "course_slug": slug,
        "monotonicity": monotonicity_score(table),
        "calibration_slope": calibration_slope(preds),
    })
    comp = pd.read_csv(comp_p) if comp_p.exists() else None
    ea = build_error_analysis(preds, comp)
    failure_modes[slug] = ea["failure_modes"]
    by_course = ea["by_course"]
    model_rows.append({
        "course_slug": slug,
        "spearman": float(by_course["spearman"].mean(skipna=True)) if not by_course.empty else float("nan"),
        "n_failure_modes": len(ea["failure_modes"]),
    })
    error_rows.append({"course_slug": slug, "failure_modes": "; ".join(ea["failure_modes"]) or "none"})


def _write_aggregate(out, status_rows, model_rows, baseline_frames, health_frames,
                     calib_rows, error_rows, failure_modes, sweep_frames,
                     data_source, do_sweep) -> dict:
    status = pd.DataFrame(status_rows, columns=list(_STATUS_COLUMNS))
    missing: dict[str, str] = {}

    status.to_csv(out / "annual_course_status.csv", index=False)
    _concat_or_note(health_frames, out / "annual_data_health_summary.csv", missing)
    _concat_or_note(baseline_frames, out / "annual_baseline_comparison.csv", missing)
    _rows_or_note(model_rows, out / "annual_model_metrics.csv", missing,
                  ["course_slug", "spearman", "n_failure_modes"])
    _rows_or_note(calib_rows, out / "annual_calibration_summary.csv", missing,
                  ["course_slug", "monotonicity", "calibration_slope"])
    _rows_or_note(error_rows, out / "annual_error_analysis_summary.csv", missing,
                  ["course_slug", "failure_modes"])
    if do_sweep:
        _concat_or_note(sweep_frames, out / "annual_optimization_summary.csv", missing)
    else:
        missing["annual_optimization_summary.csv"] = "sweep not requested (do_sweep=False)"

    # per-course metrics.csv
    for row in status_rows:
        if row["status"] in ("evaluated", "low_coverage"):
            cd = out / "per_course" / row["course_slug"]
            cd.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(cd / "metrics.csv", index=False)

    report = _annual_report(status, calib_rows, failure_modes, data_source, do_sweep)
    (out / "annual_insight_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "analysis_type": "real" if data_source == "real" else "synthetic",
        "n_targets": int(len(status)),
        "status_counts": {k: int(v) for k, v in status["status"].value_counts().items()},
        "evaluated_courses": status.loc[status["status"] == "evaluated", "course_slug"].tolist(),
    }
    (out / "annual_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (out / "missing_outputs.json").write_text(json.dumps(missing, indent=2), encoding="utf-8")
    return manifest


def _concat_or_note(frames, path, missing):
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    else:
        missing[path.name] = "no courses produced this output"


def _rows_or_note(rows, path, missing, cols):
    if rows:
        pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    else:
        missing[path.name] = "no courses produced this output"


def _annual_report(status, calib_rows, failure_modes, data_source, do_sweep) -> str:
    real = data_source == "real"
    ev = status[status["status"] == "evaluated"]
    beats = ev[ev["beats_all_baselines"] == True]  # noqa: E712
    loses = ev[ev["beats_all_baselines"] == False]  # noqa: E712
    mono = [c["monotonicity"] for c in calib_rows if pd.notna(c.get("monotonicity"))]
    mean_mono = (sum(mono) / len(mono)) if mono else float("nan")

    def _slugs(df):
        return ", ".join(df["course_slug"].tolist()) or "none"

    L = ["# Annual player-course advantage — aggregate insight report", ""]
    L.append(f"- Data source: **{data_source}**"
             + ("" if real else "  ⚠️ synthetic — predictive validity **not** evaluated."))
    L += ["", "## Which courses have enough data?",
          f"- evaluated: {_slugs(ev)}"]
    L += ["", "## Data gaps",
          f"- missing event outcomes: {_slugs(status[status['status']=='no_event_outcomes'])}",
          f"- missing similarity: {_slugs(status[status['status']=='no_similarity'])}",
          f"- low coverage / insufficient: {_slugs(status[status['status']=='low_coverage'])}",
          f"- mapping/validation failures: {_slugs(status[status['status']=='validation_failed'])}",
          f"- unsupported (no v2.5): {_slugs(status[status['status']=='unsupported'])}"]
    L += ["", "## Model vs baselines",
          f"- beats all baselines on: {_slugs(beats)}",
          f"- does NOT beat all baselines on: {_slugs(loses)}"]
    L += ["", "## Best v2.5 config",
          "- single-config run; run with sweep/optimization to compare presets."
          if not do_sweep else "- see annual_optimization_summary.csv."]
    L += ["", "## Does higher advantage track better outcomes?",
          f"- mean calibration monotonicity across courses: {mean_mono:.3f}"
          if mono else "- not computable (insufficient covered data)."]
    L += ["", "## Optimization",
          "- Insufficient real data for reliable optimization." if not real
          else "- run the validation-split optimizer per course for tuning."]
    fm = [f"- {c}: {'; '.join(m)}" for c, m in failure_modes.items() if m]
    L += ["", "## Failure modes"] + (fm or ["- none flagged."])
    if not real:
        L += ["", "> Synthetic run — no predictive claims. Provide validated real "
              "per-hole scores + event outcomes to evaluate predictive validity."]
    return "\n".join(L)


__all__ = ["run_annual_analysis"]
