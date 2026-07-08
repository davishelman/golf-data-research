"""Tests for the batch annual real-analysis runner (issue #82). Synthetic only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import AdvantageParams
from pipeline.modeling.player_course_advantage.annual_analysis import run_annual_analysis

COURSE_A, COURSE_B, COURSE_C = "course_a", "course_b", "course_c"
PLAYERS = {"pA": 3.0, "pB": 2.0, "pC": 1.0, "pD": 0.5}
PARAMS = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=6)


def _targets():
    def row(slug, supported):
        return {"course_slug": slug, "course_name": slug.title(), "event_name": "",
                "tour": "PGA Tour", "annual_status": "recurring",
                "supported_in_repo": supported, "has_v25_geometry": True,
                "has_similarity_outputs": supported, "expected_holes": 6,
                "source_priority": "byo_csv",
                "real_data_status": "missing" if supported else "unsupported", "notes": ""}
    return pd.DataFrame([row(COURSE_A, True), row(COURSE_B, True), row(COURSE_C, False)])


def _sim(course, n_holes=6):
    return pd.DataFrame([{
        "target_course_slug": course, "target_hole_number": h,
        "target_hole_id": f"{course}:{h}", "candidate_course_slug": "src",
        "candidate_hole_number": h, "candidate_hole_id": f"src:{h}",
        "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
    } for h in range(1, n_holes + 1)])


def _history(n_holes=6, years=(2021, 2022, 2023)):
    rows, tid = [], 0
    for pid, outcome in PLAYERS.items():
        for yr in years:
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "tournament_id": f"{yr}-{tid}", "year": yr, "round": 1,
                    "hole_number": h, "course_slug": "src", "hole_id_v25": f"src:{h}",
                    "par": 4, "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    return pd.DataFrame(rows)


def _outcomes(course=COURSE_A, season=2024):
    order = sorted(PLAYERS, key=lambda p: PLAYERS[p], reverse=True)
    finish = {p: r for r, p in enumerate(order, start=1)}
    return pd.DataFrame({
        "event_id": [f"E{season}"] * len(PLAYERS), "event_name": ["Syn"] * len(PLAYERS),
        "season": [season] * len(PLAYERS), "course_slug": [course] * len(PLAYERS),
        "player_id": list(PLAYERS), "player_name": list(PLAYERS),
        "finish_position": [finish[p] for p in PLAYERS],
    })


def _provider():
    sims = {COURSE_A: _sim(COURSE_A), COURSE_B: _sim(COURSE_B)}
    return lambda c: sims.get(c)


def _run(tmp_path, outcomes=None):
    return run_annual_analysis(
        _targets(), _history(), outcomes if outcomes is not None else _outcomes(),
        _provider(), tmp_path / "annual", data_source="synthetic", params=PARAMS,
        outcome_col="finish_position", higher_is_better=False, top_k=(2,))


# --------------------------------------------------------------------------- #
def test_end_to_end_aggregate_outputs(tmp_path):
    manifest = _run(tmp_path)
    out = tmp_path / "annual"
    for f in ("annual_analysis_manifest.json", "annual_course_status.csv",
              "annual_model_metrics.csv", "annual_baseline_comparison.csv",
              "annual_calibration_summary.csv", "annual_error_analysis_summary.csv",
              "annual_data_health_summary.csv", "annual_insight_report.md",
              "missing_outputs.json"):
        assert (out / f).exists(), f"missing {f}"
    assert (out / "per_course" / COURSE_A / "analysis_report.md").exists()
    assert (out / "per_course" / COURSE_A / "metrics.csv").exists()
    assert manifest["analysis_type"] == "synthetic"


def test_status_classification(tmp_path):
    _run(tmp_path)
    st = pd.read_csv(tmp_path / "annual" / "annual_course_status.csv").set_index("course_slug")
    assert st.loc[COURSE_A, "status"] == "evaluated"          # supported + outcomes + data
    assert st.loc[COURSE_B, "status"] == "no_event_outcomes"  # supported, no outcomes
    assert st.loc[COURSE_C, "status"] == "unsupported"        # no similarity


def test_report_answers_questions_and_is_synthetic(tmp_path):
    _run(tmp_path)
    md = (tmp_path / "annual" / "annual_insight_report.md").read_text(encoding="utf-8").lower()
    for q in ("which courses have enough data", "data gaps", "model vs baselines",
              "higher advantage", "failure modes"):
        assert q in md
    assert "synthetic" in md and "not" in md  # no predictive claim
    assert "insufficient real data for reliable optimization" in md  # synthetic


def test_writes_only_under_out_root(tmp_path):
    _run(tmp_path)
    out = tmp_path / "annual"
    assert all(p.is_relative_to(tmp_path) for p in out.rglob("*"))


def test_no_outcomes_at_all_blocks_all_courses(tmp_path):
    empty = pd.DataFrame(columns=["event_id", "event_name", "season", "course_slug",
                                  "player_id", "player_name", "finish_position"])
    manifest = _run(tmp_path, outcomes=empty)
    counts = manifest["status_counts"]
    assert counts.get("evaluated", 0) == 0
    assert counts.get("no_event_outcomes", 0) >= 2


def test_does_not_mutate_inputs(tmp_path):
    targets, history, outcomes = _targets(), _history(), _outcomes()
    t_b, h_b, o_b = targets.copy(), history.copy(), outcomes.copy()
    run_annual_analysis(targets, history, outcomes, _provider(), tmp_path / "a",
                        data_source="synthetic", params=PARAMS,
                        outcome_col="finish_position", top_k=(2,))
    pd.testing.assert_frame_equal(targets, t_b)
    pd.testing.assert_frame_equal(history, h_b)
    pd.testing.assert_frame_equal(outcomes, o_b)
