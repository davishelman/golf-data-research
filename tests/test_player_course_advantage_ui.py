"""Tests for the player-course advantage UI helpers (issue #47).

Streamlit-free — mirrors the `pointcloud.demo` pattern so the view logic is
unit-testable without a browser. Covers display formatting, coverage labels,
per-player detail, and the artifact discover/load round-trip. No real data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    assemble_field_outputs,
    export_advantage_run,
)
from pipeline.modeling.player_course_advantage import ui as pcaui

TARGET_COURSE = "augusta_national"


def _synthetic(n_holes=9):
    sim = pd.DataFrame([{
        "target_course_slug": TARGET_COURSE, "target_hole_number": h,
        "target_hole_id": f"{TARGET_COURSE}:{h}",
        "candidate_course_slug": "src", "candidate_hole_number": h,
        "candidate_hole_id": f"src:{h}", "rank": 1, "total_score": 1.0,
        "similarity_weight": 1.0, "weight_method": "manual", "config_name": "baseline",
    } for h in range(1, n_holes + 1)])
    rows, tid = [], 0
    for pid, outcome in {"pA": 2.0, "pB": 1.0}.items():
        for yr in (2022, 2023):
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "player_name": pid.upper(), "tournament_id": f"{yr}-{tid}",
                    "year": yr, "round": 1, "hole_number": h, "course_slug": "src",
                    "hole_id_v25": f"src:{h}", "par": 4,
                    "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    return sim, pd.DataFrame(rows), ["pA", "pB"], n_holes


# --------------------------------------------------------------------------- #
# Display formatting + coverage labels
# --------------------------------------------------------------------------- #
def test_format_ranking_and_coverage_labels():
    ranking = pd.DataFrame([
        {"rank": 1, "player_id": "pA", "player_name": "PA", "course_advantage": 18.6666,
         "course_advantage_mean": 2.07, "holes_covered": 9, "total_target_holes": 9,
         "total_raw_occurrences": 18, "total_weighted_occurrences": 18.0,
         "low_coverage": False, "reason": None, "config_name": "baseline",
         "target_course_slug": TARGET_COURSE},
        {"rank": 2, "player_id": "pB", "player_name": "PB", "course_advantage": None,
         "course_advantage_mean": None, "holes_covered": 3, "total_target_holes": 9,
         "total_raw_occurrences": 3, "total_weighted_occurrences": 3.0,
         "low_coverage": True, "reason": "below_min_holes_covered", "config_name": "baseline",
         "target_course_slug": TARGET_COURSE},
    ])
    disp = pcaui.format_ranking_for_display(ranking)
    assert list(disp.columns) == list(pcaui.RANKING_DISPLAY_COLUMNS)
    assert disp.loc[0, "course_advantage"] == pytest.approx(18.667)
    assert disp.loc[0, "coverage"] == "✓ 9/9"
    assert "withheld" in disp.loc[1, "coverage"]
    assert "below_min_holes_covered" in disp.loc[1, "coverage"]


def test_format_ranking_empty():
    disp = pcaui.format_ranking_for_display(pd.DataFrame())
    assert list(disp.columns) == list(pcaui.RANKING_DISPLAY_COLUMNS)
    assert disp.empty


# --------------------------------------------------------------------------- #
# Synthetic demo view + per-player detail
# --------------------------------------------------------------------------- #
def test_synthetic_demo_view():
    view = pcaui.synthetic_demo_view()
    assert view["data_source"] == "SYNTHETIC"   # the app warns on this
    rd = view["ranking_display"]
    assert not rd.empty
    assert set(rd["player_id"]) == {"demo_ace", "demo_good", "demo_mid", "demo_low"}
    assert "coverage" in rd.columns


def test_player_hole_detail_filters_to_player():
    sim, history, field, n = _synthetic()
    outs = assemble_field_outputs(history, sim, field, TARGET_COURSE, 2024,
                                  params=AdvantageParams(min_holes_covered=n))
    detail = pcaui.player_hole_detail(outs["hole_details"], outs["diagnostics"], "pA")
    assert len(detail["per_hole"]) == n
    assert not detail["top_similar"].empty
    # Only pA's similar-hole contributions (diagnostics carry player_id).
    pa_diag = outs["diagnostics"][outs["diagnostics"]["player_id"] == "pA"]
    assert len(detail["top_similar"]) <= len(pa_diag)


# --------------------------------------------------------------------------- #
# Artifact discover + load round-trip
# --------------------------------------------------------------------------- #
def test_discover_and_load_ranking_view(tmp_path):
    sim, history, field, n = _synthetic()
    outs = assemble_field_outputs(history, sim, field, TARGET_COURSE, 2024,
                                  params=AdvantageParams(min_holes_covered=n))
    export_advantage_run(
        outs["rankings"], hole_details=outs["hole_details"], diagnostics=outs["diagnostics"],
        target_course_slug=TARGET_COURSE, config_name="baseline", predict_season=2024,
        params=AdvantageParams(min_holes_covered=n),
        root=tmp_path / "data" / "player_course_advantage",
    )
    runs = pcaui.discover_advantage_runs([tmp_path])
    assert len(runs) == 1
    view = pcaui.load_ranking_view(runs[0])
    assert not view["ranking_display"].empty
    assert set(view["ranking_display"]["player_id"]) == {"pA", "pB"}
    # per-player detail works off the loaded artifact
    detail = pcaui.player_hole_detail(view["hole_details"], view["diagnostics"], "pA")
    assert len(detail["per_hole"]) == n


def test_discover_ignores_non_run_dirs(tmp_path):
    d = tmp_path / "data" / "player_course_advantage" / "not_a_run"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text('{"kind": "something_else"}', encoding="utf-8")
    assert pcaui.discover_advantage_runs([tmp_path]) == []
