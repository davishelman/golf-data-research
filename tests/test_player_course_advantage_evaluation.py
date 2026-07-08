"""Tests for the evaluation report generator (issue #59).

Builds a real synthetic BacktestResult + baseline comparison via the shipped
modules, then checks the summary, manifest, markdown caveat, and export.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    build_data_health_report,
    build_evaluation_summary,
    compare_baselines,
    export_evaluation_report,
    make_metrics_manifest,
    render_evaluation_markdown,
    run_backtest,
)

TARGET_COURSE = "augusta_national"
PARAMS = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=3)


def _sim(n_holes=3):
    return pd.DataFrame([{
        "target_course_slug": TARGET_COURSE, "target_hole_number": h,
        "target_hole_id": f"{TARGET_COURSE}:{h}", "candidate_course_slug": "other",
        "candidate_hole_number": h, "candidate_hole_id": f"other:{h}",
        "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
    } for h in range(1, n_holes + 1)])


def _history(n_holes=3):
    rows, tid = [], 0
    for pid, outcome in {"pA": 3, "pB": 2, "pC": 1}.items():
        for yr in (2021, 2022, 2023):
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "tournament_id": f"{yr}-{tid}", "year": yr, "round": 1,
                    "hole_number": h, "course_slug": "other", "hole_id_v25": f"other:{h}",
                    "par": 4, "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    return pd.DataFrame(rows)


def _results():
    frames = []
    for s in (2023, 2024):
        frames.append(pd.DataFrame({
            "event_id": [f"E{s}"] * 3, "predict_season": [s] * 3,
            "target_course_slug": [TARGET_COURSE] * 3,
            "player_id": ["pA", "pB", "pC"], "finish_rank": [1, 2, 3],
        }))
    return pd.concat(frames, ignore_index=True)


def _bt_and_comp():
    sim, hist, results = _sim(), _history(), _results()
    bt = run_backtest(hist, sim, results, outcome_col="finish_rank",
                      higher_is_better=False, params=PARAMS, top_k=(1,))
    comp = compare_baselines(hist, sim, results, outcome_col="finish_rank",
                             higher_is_better=False, params=PARAMS, top_k=(1,))
    return bt, comp


def test_build_evaluation_summary():
    bt, comp = _bt_and_comp()
    summary = build_evaluation_summary(bt, comp, data_source="synthetic")
    assert summary["data_source"] == "synthetic"
    assert "spearman_pooled" in summary["backtest_summary"]
    assert "baseline_lift" in summary
    assert "coverage" in summary and "stability" in summary
    assert summary["best_worst_events"]["best"] is not None


def test_baseline_lift_reflects_comparison():
    bt, comp = _bt_and_comp()
    summary = build_evaluation_summary(bt, comp)
    lift = summary["baseline_lift"]
    # honest: on this synthetic data baselines using the same rows tie the model.
    assert lift["wins_vs_baselines_count"] <= lift["n_baselines_comparable"]
    assert isinstance(lift["beats_all_baselines"], bool)


def test_coverage_includes_data_health_warnings():
    bt, comp = _bt_and_comp()
    health = build_data_health_report(_history(), _sim(), bt.predictions)
    summary = build_evaluation_summary(bt, comp, data_health=health)
    assert "data_health_warnings" in summary["coverage"]


def test_markdown_has_synthetic_caveat():
    bt, comp = _bt_and_comp()
    md = render_evaluation_markdown(build_evaluation_summary(bt, comp, data_source="synthetic"))
    assert "synthetic" in md.lower()
    assert "not" in md.lower() and "predictive" in md.lower()


def test_real_label_omits_synthetic_caveat():
    bt, comp = _bt_and_comp()
    md = render_evaluation_markdown(build_evaluation_summary(bt, comp, data_source="real"))
    assert "not a predictive-validity claim" not in md.lower()


def test_metrics_manifest_is_scalar():
    bt, comp = _bt_and_comp()
    manifest = make_metrics_manifest(build_evaluation_summary(bt, comp))
    assert manifest["data_source"] == "synthetic"
    assert "spearman_pooled" in manifest
    assert all(not isinstance(v, (list, dict)) for v in manifest.values())


def test_export_writes_files(tmp_path):
    bt, comp = _bt_and_comp()
    summary = build_evaluation_summary(bt, comp)
    paths = export_evaluation_report(summary, tmp_path / "eval")
    for key in ("summary", "manifest", "report"):
        assert paths[key].exists()
    loaded = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert loaded["data_source"] == "synthetic"


def test_no_mutation_of_comparison():
    bt, comp = _bt_and_comp()
    before = comp.copy()
    build_evaluation_summary(bt, comp)
    pd.testing.assert_frame_equal(comp, before)
