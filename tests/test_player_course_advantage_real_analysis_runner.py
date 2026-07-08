"""Tests for the real-data analysis runner (issue #72). Synthetic only."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import AdvantageParams, run_real_analysis
from pipeline.modeling.player_course_advantage.benchmarks import synthetic_inputs
from pipeline.modeling.player_course_advantage.schema import SchemaError

PARAMS = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=9)


def _inputs():
    sim, history, field, results = synthetic_inputs(n_players=12, n_events=3)
    return sim, history, results


def test_runner_end_to_end(tmp_path):
    sim, history, results = _inputs()
    manifest = run_real_analysis(history, sim, results, tmp_path / "run",
                                 data_source="synthetic", params=PARAMS,
                                 top_k=(2,), do_sweep=True, do_benchmarks=True)
    out = tmp_path / "run"
    for f in ("analysis_manifest.json", "data_health_summary.csv", "coverage_by_event.csv",
              "backtest_predictions.csv", "backtest_per_event.csv", "baseline_comparison.csv",
              "baseline_lift.csv", "calibration_table.csv", "reliability_by_coverage.csv",
              "sweep_summary.csv", "benchmark_summary.csv", "analysis_report.md",
              "missing_outputs.json"):
        assert (out / f).exists(), f"missing output {f}"
    assert manifest["analysis_type"] == "synthetic"
    assert manifest["stopped_early"] is False
    # synthetic report must not claim predictive validity
    assert "synthetic" in (out / "analysis_report.md").read_text(encoding="utf-8").lower()


def test_runner_rejects_invalid_history(tmp_path):
    sim, history, results = _inputs()
    bad = history.drop(columns=["field_avg_score"])
    with pytest.raises(SchemaError):
        run_real_analysis(bad, sim, results, tmp_path / "run", params=PARAMS)


def test_runner_writes_only_under_output_dir(tmp_path):
    sim, history, results = _inputs()
    run_real_analysis(history, sim, results, tmp_path / "run", params=PARAMS, top_k=(2,))
    # everything created lives under the run dir
    assert (tmp_path / "run").is_dir()
    assert all(p.is_relative_to(tmp_path) for p in (tmp_path / "run").rglob("*"))


def test_runner_stops_early_on_low_coverage(tmp_path):
    sim, history, results = _inputs()
    manifest = run_real_analysis(history, sim, results, tmp_path / "run",
                                 params=PARAMS, min_coverage=1.1)  # impossible -> stop
    assert manifest["stopped_early"] is True
    out = tmp_path / "run"
    assert (out / "data_health_summary.csv").exists()
    assert not (out / "backtest_predictions.csv").exists()  # predictive step skipped
    missing = json.loads((out / "missing_outputs.json").read_text(encoding="utf-8"))
    assert "backtest_predictions.csv" in missing


def test_runner_does_not_mutate_inputs(tmp_path):
    sim, history, results = _inputs()
    sim_b, hist_b, res_b = sim.copy(), history.copy(), results.copy()
    run_real_analysis(history, sim, results, tmp_path / "run", params=PARAMS, top_k=(2,))
    pd.testing.assert_frame_equal(sim, sim_b)
    pd.testing.assert_frame_equal(history, hist_b)
    pd.testing.assert_frame_equal(results, res_b)
