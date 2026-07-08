"""Tests for the data coverage / quality health report (issue #62).

Fake data only. Verifies the report counts problems without crashing on
imperfect input (missing field_avg, duplicate grain, invalid ids, uncovered
candidates), and summarizes backtest coverage when predictions are supplied.
"""

from __future__ import annotations

import pandas as pd

from pipeline.modeling.player_course_advantage import (
    build_data_health_report,
    data_health_to_frames,
    summarize_backtest_coverage,
    summarize_history_quality,
    summarize_similarity_coverage,
)


def _clean_history(n_players=3, n_holes=4, course="src"):
    rows, tid = [], 0
    for p in range(n_players):
        for h in range(1, n_holes + 1):
            rows.append({
                "player_id": f"p{p}", "tournament_id": "t1", "year": 2023, "round": 1,
                "hole_number": h, "course_slug": course, "hole_id_v25": f"{course}:{h}",
                "par": 4, "player_score": 4, "field_avg_score": 4.1,
            })
            tid += 1
    return pd.DataFrame(rows)


def _sim(n_holes=4, target="augusta_national", cand="src"):
    return pd.DataFrame([{
        "target_course_slug": target, "target_hole_number": h,
        "target_hole_id": f"{target}:{h}", "candidate_course_slug": cand,
        "candidate_hole_number": h, "candidate_hole_id": f"{cand}:{h}",
        "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
    } for h in range(1, n_holes + 1)])


# --------------------------------------------------------------------------- #
# History quality
# --------------------------------------------------------------------------- #
def test_clean_history_quality():
    q = summarize_history_quality(_clean_history())
    assert q["players"] == 3 and q["courses"] == 1 and q["seasons"] == 1
    assert q["missing_field_avg_score_rate"] == 0.0
    assert q["duplicate_grain_count"] == 0
    assert q["invalid_hole_id_count"] == 0


def test_missing_field_avg_score_detected():
    h = _clean_history()
    h.loc[0, "field_avg_score"] = None
    q = summarize_history_quality(h)
    assert q["missing_field_avg_score_rate"] > 0


def test_missing_field_avg_column_entirely():
    h = _clean_history().drop(columns=["field_avg_score"])
    q = summarize_history_quality(h)
    assert q["missing_field_avg_score_rate"] == 1.0


def test_duplicate_grain_counted():
    h = _clean_history()
    dup = pd.concat([h, h.iloc[[0]]], ignore_index=True)
    assert summarize_history_quality(dup)["duplicate_grain_count"] >= 2


def test_invalid_hole_id_counted():
    h = _clean_history()
    h.loc[0, "hole_id_v25"] = "wrong_course:99"
    assert summarize_history_quality(h)["invalid_hole_id_count"] == 1


# --------------------------------------------------------------------------- #
# Similarity coverage
# --------------------------------------------------------------------------- #
def test_similarity_coverage_all_present():
    c = summarize_similarity_coverage(_sim(4), _clean_history(n_holes=4))
    assert c["target_holes"] == 4
    assert c["candidate_holes"] == 4
    assert c["candidate_holes_missing_history"] == 0


def test_candidate_holes_missing_history_counted():
    # Similar set references src:1..6 but history only has src:1..4.
    c = summarize_similarity_coverage(_sim(6), _clean_history(n_holes=4))
    assert c["candidate_holes_missing_history"] == 2
    assert c["candidate_holes_with_history"] == 4


# --------------------------------------------------------------------------- #
# Backtest coverage
# --------------------------------------------------------------------------- #
def _predictions():
    return pd.DataFrame({
        "event_id": ["E1", "E1", "E1", "E2", "E2"],
        "predict_season": [2024, 2024, 2024, 2023, 2023],
        "target_course_slug": ["augusta_national"] * 5,
        "player_id": ["pA", "pB", "pC", "pA", "pB"],
        "holes_covered": [9, 9, 0, 9, 0],
        "low_coverage": [False, False, True, False, True],
        "reason": [None, None, "below_min_holes_covered", None, "no_player_history"],
        "covered": [True, True, False, True, False],
    })


def test_backtest_coverage_summary():
    bt = summarize_backtest_coverage(_predictions())
    assert bt["player_event_pairs"] == 5
    assert bt["player_event_pairs_scored"] == 3
    assert bt["coverage"] == 0.6
    # E1 has 2 covered (>=2 -> computable), E2 has 1 (<2).
    assert bt["event_coverage"] == 0.5
    assert bt["low_coverage_reason_counts"]["below_min_holes_covered"] == 1
    assert bt["low_coverage_reason_counts"]["no_player_history"] == 1
    assert set(bt["coverage_by_season"]) == {"2024", "2023"}


# --------------------------------------------------------------------------- #
# Full report + frames
# --------------------------------------------------------------------------- #
def test_report_without_predictions_runs_before_backtest():
    report = build_data_health_report(_clean_history(), _sim(4))
    assert report["backtest_coverage"] is None
    assert report["data_source"] == "synthetic"
    assert report["warnings"] == []  # clean inputs


def test_report_collects_warnings_on_imperfect_data():
    h = _clean_history()
    h.loc[0, "field_avg_score"] = None
    report = build_data_health_report(h, _sim(6), _predictions())
    joined = " ".join(report["warnings"])
    assert "field_avg_score" in joined
    assert "candidate holes" in joined  # src:5,6 uncovered
    frames = data_health_to_frames(report)
    assert "data_quality_summary" in frames
    assert not frames["low_coverage_reasons"].empty


def test_report_does_not_crash_on_empty():
    report = build_data_health_report(pd.DataFrame(), pd.DataFrame())
    assert report["history_quality"]["rows"] == 0
