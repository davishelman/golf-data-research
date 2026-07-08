"""Tests for calibration & reliability analysis (issue #61).

Synthetic predictions only. Monotonic data should calibrate well; scrambled data
worse; tiny samples return NaN/warnings without crashing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.modeling.player_course_advantage import (
    bucket_predictions,
    calibration_slope,
    calibration_table,
    monotonicity_score,
    reliability_by_coverage,
    render_calibration_summary,
)


def _predictions(advantage, performance, holes_covered=None, n_holes=9):
    n = len(advantage)
    return pd.DataFrame({
        "course_advantage": advantage,
        "performance": performance,
        "covered": [True] * n,
        "holes_covered": holes_covered if holes_covered is not None else [n_holes] * n,
        "total_raw_occurrences": [10] * n,
        "total_weighted_occurrences": [10.0] * n,
    })


def _monotonic(n=40):
    adv = np.linspace(0, 10, n)
    perf = adv + np.random.RandomState(0).normal(0, 0.1, n)  # noisy but monotonic
    return _predictions(adv, perf)


# --------------------------------------------------------------------------- #
# Monotonic vs scrambled
# --------------------------------------------------------------------------- #
def test_monotonic_data_calibrates_well():
    preds = _monotonic()
    table = calibration_table(preds, n_buckets=5)
    assert len(table) == 5
    assert monotonicity_score(table) > 0.8      # bigger advantage -> better outcome
    assert calibration_slope(preds) > 0


def test_scrambled_data_degrades():
    preds = _monotonic()
    scrambled = preds.copy()
    scrambled["performance"] = np.random.RandomState(1).permutation(
        scrambled["performance"].to_numpy())
    mono_ok = monotonicity_score(calibration_table(preds))
    mono_bad = monotonicity_score(calibration_table(scrambled))
    assert mono_bad < mono_ok
    assert abs(calibration_slope(scrambled)) < calibration_slope(preds)


# --------------------------------------------------------------------------- #
# Small samples degrade gracefully
# --------------------------------------------------------------------------- #
def test_small_sample_returns_warning_not_crash():
    preds = _predictions([1.0, 2.0], [1.0, 2.0])
    summary = render_calibration_summary(preds)
    assert summary["warnings"]  # flagged as too small
    # constant/near-degenerate buckets still don't crash
    assert "calibration_slope" in summary


def test_degenerate_advantage_no_crash():
    preds = _predictions([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    table = calibration_table(preds)   # single bucket (all equal advantage)
    assert np.isnan(monotonicity_score(table))


# --------------------------------------------------------------------------- #
# Reliability by coverage
# --------------------------------------------------------------------------- #
def test_reliability_split_by_coverage():
    # high-coverage rows are cleanly monotonic; low-coverage rows are noise.
    adv = list(np.linspace(0, 10, 10)) + list(np.linspace(0, 10, 10))
    perf = list(np.linspace(0, 10, 10)) + list(np.random.RandomState(2).normal(0, 1, 10))
    holes = [9] * 10 + [2] * 10
    preds = _predictions(adv, perf, holes_covered=holes)
    rel = reliability_by_coverage(preds, by="holes_covered")
    assert rel["n_high"] == 10 and rel["n_low"] == 10
    assert rel["high_spearman"] > rel["low_spearman"]


# --------------------------------------------------------------------------- #
# No mutation
# --------------------------------------------------------------------------- #
def test_no_mutation_of_inputs():
    preds = _monotonic()
    before = preds.copy()
    bucket_predictions(preds)
    calibration_table(preds)
    calibration_slope(preds)
    reliability_by_coverage(preds)
    pd.testing.assert_frame_equal(preds, before)
