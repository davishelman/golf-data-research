"""Tests for the insight report + verdict (issue #76)."""

from __future__ import annotations

from pipeline.modeling.player_course_advantage.insight_report import (
    insight_verdict,
    render_insight_report,
)


# --------------------------------------------------------------------------- #
# Verdict logic
# --------------------------------------------------------------------------- #
def test_synthetic_is_gray():
    assert insight_verdict("synthetic", coverage=1.0, spearman_pooled=0.9,
                           beats_all_baselines=True, wins_vs_baselines_count=4) == "gray"


def test_real_success_is_green():
    assert insight_verdict("real", coverage=0.8, spearman_pooled=0.4,
                           beats_all_baselines=True, wins_vs_baselines_count=4) == "green"


def test_real_failure_beats_no_baselines_is_red():
    assert insight_verdict("real", coverage=0.8, spearman_pooled=0.2,
                           beats_all_baselines=False, wins_vs_baselines_count=0) == "red"


def test_low_coverage_is_red_before_performance():
    # even with a great spearman, low coverage gates to red
    assert insight_verdict("real", coverage=0.2, spearman_pooled=0.9,
                           beats_all_baselines=True, wins_vs_baselines_count=4) == "red"


def test_mixed_is_yellow():
    assert insight_verdict("real", coverage=0.8, spearman_pooled=0.3,
                           beats_all_baselines=False, wins_vs_baselines_count=2) == "yellow"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_synthetic_report_says_no_predictive_claim():
    md = render_insight_report(data_source="synthetic", spearman_pooled=0.9,
                               coverage=1.0, beats_all_baselines=True,
                               wins_vs_baselines_count=4)
    low = md.lower()
    assert "⚪" in md or "gray" in low
    assert "not allowed" in low and "not" in low and "evaluated" in low
    assert "blocker" in low


def test_real_failure_report_is_honest():
    md = render_insight_report(data_source="real", spearman_pooled=0.2, coverage=0.8,
                               beats_all_baselines=False, wins_vs_baselines_count=0,
                               n_baselines_comparable=4)
    low = md.lower()
    assert "🔴" in md or "red" in low
    assert "no" in low and "baseline" in low  # says it beats no baselines


def test_low_coverage_report_flags_coverage_first():
    md = render_insight_report(data_source="real", spearman_pooled=0.9, coverage=0.2,
                               beats_all_baselines=True, wins_vs_baselines_count=4)
    assert "coverage is too low" in md.lower()


def test_real_success_report_is_green_and_allows_claims():
    md = render_insight_report(data_source="real", spearman_pooled=0.4, coverage=0.8,
                               beats_all_baselines=True, wins_vs_baselines_count=4,
                               n_baselines_comparable=4)
    low = md.lower()
    assert "🟢" in md or "green" in low
    assert "allowed" in low


def test_load_and_render_from_run_dir(tmp_path):
    # End-to-end: real-analysis runner -> insight report loads its outputs.
    from pipeline.modeling.player_course_advantage import AdvantageParams, run_real_analysis
    from pipeline.modeling.player_course_advantage.benchmarks import synthetic_inputs
    from pipeline.modeling.player_course_advantage.insight_report import load_and_render

    sim, history, field, results = synthetic_inputs(n_players=12, n_events=3)
    run_real_analysis(history, sim, results, tmp_path / "run", data_source="synthetic",
                      params=AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=9),
                      top_k=(2,))
    md = load_and_render(tmp_path / "run")
    assert "insight report" in md.lower()
    assert "gray" in md.lower() or "⚪" in md  # synthetic run
