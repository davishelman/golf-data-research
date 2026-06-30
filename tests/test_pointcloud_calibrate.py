"""Tests for v2.5 score calibration decomposition + summary."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pipeline.modeling.pointcloud.calibrate import (
    decompose_components,
    summarize_calibration,
)
from pipeline.modeling.pointcloud.config import config_from_dict
from tests.test_pointcloud_config import _valid_payload


def _config():
    return config_from_dict(_valid_payload())


def _results():
    """Two pairs: one normal, one where water is a missing-surface penalty (50.0)."""
    return pd.DataFrame([
        {
            "target_hole_id": "t:1", "candidate_hole_id": "c:1",
            "total_score": 6.9,
            "fairway_score": 10.0, "green_score": 8.0, "bunker_score": 4.0,
            "water_score": None, "tee_score": 2.0,
            "yardage_penalty": 0.5, "elevation_penalty": 0.3,
            "missing_surface_penalty": 0.0,
        },
        {
            "target_hole_id": "t:1", "candidate_hole_id": "c:2",
            "total_score": 999.0,
            "fairway_score": 5.0, "green_score": 5.0, "bunker_score": 5.0,
            "water_score": 50.0, "tee_score": 5.0,  # water == missing penalty
            "yardage_penalty": 0.0, "elevation_penalty": 0.0,
            "missing_surface_penalty": 7.5,
        },
    ])


def test_decompose_components_arithmetic():
    config = _config()
    dec = decompose_components(_results(), config)
    first = dec.iloc[0]
    # weights: fairway .30, green .25, bunker .25, water .15, tee .05
    assert first["contrib_fairway"] == pytest.approx(3.0)
    assert first["contrib_green"] == pytest.approx(2.0)
    assert first["contrib_bunker"] == pytest.approx(1.0)
    assert first["contrib_water"] == pytest.approx(0.0)   # NaN -> 0
    assert first["contrib_tee"] == pytest.approx(0.1)
    assert first["contrib_yardage"] == pytest.approx(0.5)
    assert first["contrib_elevation"] == pytest.approx(0.3)
    assert first["recomputed_total"] == pytest.approx(6.9)


def test_decompose_shares_sum_to_one_when_total_nonzero():
    dec = decompose_components(_results(), _config())
    share_cols = [c for c in dec.columns if c.startswith("share_")]
    for _, row in dec.iterrows():
        if row["recomputed_total"] > 0:
            assert sum(row[c] for c in share_cols) == pytest.approx(1.0, abs=1e-9)


def test_summarize_calibration_detects_missing_dominance():
    config = _config()
    dec = decompose_components(_results(), config)
    summary = summarize_calibration(dec, config)
    assert summary["n_pairs"] == 2
    assert summary["config_name"] == config.config_name
    # The second pair's water contribution (7.5) equals weight*missing_penalty,
    # but it's a small share of 999; missing dominance fraction stays modest.
    assert 0.0 <= summary["missing_surface_dominant_fraction"] <= 1.0
    assert "mean_component_shares" in summary
    assert set(summary["flags"]) == {
        "missing_penalties_may_dominate",
        "penalties_may_dominate",
        "elevation_possibly_double_counted",
    }


def test_missing_surface_share_identifies_penalty_rows():
    # A pair dominated by a missing-surface penalty: bunker missing (weight .25 *
    # penalty 40 = 10) against tiny real scores elsewhere.
    config = _config()
    results = pd.DataFrame([{
        "target_hole_id": "t:1", "candidate_hole_id": "c:9",
        "total_score": 10.2,
        "fairway_score": 0.1, "green_score": 0.1, "bunker_score": 40.0,  # missing
        "water_score": None, "tee_score": 0.1,
        "yardage_penalty": 0.0, "elevation_penalty": 0.0,
        "missing_surface_penalty": 10.0,
    }])
    dec = decompose_components(results, config)
    summary = summarize_calibration(dec, config)
    # bunker contributes .25*40 = 10 of ~10.05 total -> missing dominates.
    assert summary["missing_surface_dominant_fraction"] == pytest.approx(1.0)
    assert summary["flags"]["missing_penalties_may_dominate"] is True


def test_summarize_handles_empty_results():
    config = _config()
    empty = pd.DataFrame(columns=[
        "target_hole_id", "candidate_hole_id", "total_score",
        "fairway_score", "green_score", "bunker_score", "water_score", "tee_score",
        "yardage_penalty", "elevation_penalty", "missing_surface_penalty",
    ])
    dec = decompose_components(empty, config)
    summary = summarize_calibration(dec, config)
    assert summary["n_pairs"] == 0
    assert summary["elevation_double_count_correlation"] is None
