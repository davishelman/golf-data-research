"""Smoke tests for the runtime/scalability benchmarks (issue #63).

Tiny sizes so it runs fast. Verifies the benchmark table shape, optional export,
and no input mutation — not the absolute timings.
"""

from __future__ import annotations

import pandas as pd

from pipeline.modeling.player_course_advantage import (
    benchmark_components,
    export_benchmarks,
    run_benchmarks,
)
from pipeline.modeling.player_course_advantage.benchmarks import (
    BENCHMARK_COLUMNS,
    synthetic_inputs,
)


def test_benchmark_components_shape():
    df = benchmark_components(field_size=4, n_events=1)
    for col in BENCHMARK_COLUMNS:
        assert col in df.columns
    assert set(df["component"]) >= {
        "score_player_course", "score_tournament_field", "explain_player_course",
        "run_backtest",
    }
    assert (df["seconds"] >= 0).all()


def test_benchmark_components_with_export(tmp_path):
    df = benchmark_components(field_size=4, n_events=1, export_root=tmp_path / "runs")
    assert "export_advantage_run" in set(df["component"])
    row = df[df["component"] == "export_advantage_run"].iloc[0]
    assert row["artifact_bytes"] > 0


def test_run_benchmarks_small():
    df = run_benchmarks(field_sizes=(3, 6), event_counts=(1, 2))
    assert not df.empty
    assert set(df["field_size"]) >= {3, 6}


def test_export_benchmarks_csv_and_json(tmp_path):
    df = run_benchmarks(field_sizes=(3,), event_counts=(1,))
    csv = export_benchmarks(df, tmp_path / "b.csv")
    js = export_benchmarks(df, tmp_path / "b.json")
    assert csv.exists() and js.exists()
    assert len(pd.read_csv(csv)) == len(df)


def test_synthetic_inputs_do_not_mutate_across_calls():
    a = synthetic_inputs(4, 1)
    b = synthetic_inputs(4, 1)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
