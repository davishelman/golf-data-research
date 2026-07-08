"""Tests for error analysis by course / hole type / coverage (issue #75)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.modeling.player_course_advantage.error_analysis import (
    build_error_analysis,
    identify_failure_modes,
    performance_by,
)


def _predictions(*, with_metadata=True, low_coverage=False, n_events=2, players=6):
    rows = []
    for e in range(n_events):
        for p in range(players):
            adv = float(p)                      # advantage increases with p
            perf = float(p)                     # performance matches -> good model
            rows.append({
                "event_id": f"E{e}", "predict_season": 2023 + e,
                "target_course_slug": "augusta_national" if e == 0 else "pebble_beach",
                "player_id": f"p{p}", "course_advantage": adv, "performance": perf,
                "covered": not (low_coverage and p >= 2),  # drop most rows if low_coverage
                "holes_covered": 9, "total_target_holes": 9,
                "total_raw_occurrences": 20, "low_coverage": False, "reason": None,
            })
            if with_metadata:
                rows[-1]["config_name"] = "baseline"
                rows[-1]["par"] = 4
                rows[-1]["hole_number"] = 1
    return pd.DataFrame(rows)


def _baseline_comparison(model=0.9):
    return pd.DataFrame({
        "ranker": ["similar_hole_model", "null", "recent_form"],
        "spearman_pooled": [model, 0.1, 0.5],
        "coverage": [1.0, 1.0, 1.0], "n_pairs": [12, 12, 12],
    })


# --------------------------------------------------------------------------- #
def test_performance_by_course():
    tbl = performance_by(_predictions(), "target_course_slug")
    assert set(tbl["target_course_slug"]) == {"augusta_national", "pebble_beach"}
    assert (tbl["spearman"] > 0.99).all()  # perfect synthetic ranking


def test_build_with_full_metadata():
    ea = build_error_analysis(_predictions(with_metadata=True), _baseline_comparison())
    dims = set(ea["by_dimension"])
    assert {"target_course_slug", "coverage_bucket", "config_name", "par"} <= dims
    assert not ea["by_course"].empty


def test_build_with_missing_optional_metadata():
    ea = build_error_analysis(_predictions(with_metadata=False), _baseline_comparison())
    # config/par absent -> skipped gracefully, no crash
    assert ea["by_config"].empty
    assert "target_course_slug" in ea["by_dimension"]


def test_identifies_model_underperforming_baselines():
    # model spearman below both baselines -> beats none
    comp = _baseline_comparison(model=0.05)
    modes, nexts = identify_failure_modes(_predictions(), comp)
    assert any("beats NO baselines" in m or "beats no baselines" in m.lower() for m in modes)
    assert nexts


def test_identifies_low_coverage_failure_mode():
    modes, _ = identify_failure_modes(_predictions(low_coverage=True), _baseline_comparison())
    assert any("coverage" in m.lower() for m in modes)


def test_failure_modes_empty_when_healthy():
    modes, nexts = identify_failure_modes(_predictions(), _baseline_comparison(model=0.9))
    # model beats all baselines, full coverage, positive -> no failure modes
    assert modes == []
    assert nexts  # still recommends validating further
