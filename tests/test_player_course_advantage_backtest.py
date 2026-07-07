"""Tests for the retrospective backtest framework (issue #35).

Fake data only. Two priorities: (1) the leakage guard — planted future/target
season rows must never change a prediction; (2) the metrics — on a perfectly
ordered synthetic field the correlations and hit-rates are exactly known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    BacktestError,
    SchemaError,
    pearson_corr,
    run_backtest,
    spearman_corr,
    top_k_hit_rate,
    top_k_lift,
)

TARGET_COURSE = "augusta_national"
LOOSE = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=1)


# --------------------------------------------------------------------------- #
# Fake data builders
# --------------------------------------------------------------------------- #
def make_similar_holes(n_holes=3, target_course=TARGET_COURSE):
    rows = []
    for h in range(1, n_holes + 1):
        rows.append({
            "target_course_slug": target_course,
            "target_hole_number": h,
            "target_hole_id": f"{target_course}:{h}",
            "candidate_course_slug": "other",
            "candidate_hole_number": h,
            "candidate_hole_id": f"other:{h}",
            "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
            "weight_method": "manual", "config_name": "baseline",
        })
    return pd.DataFrame(rows)


def make_history(player_outcomes, n_holes=3, year=2023, field_avg=4):
    """player_outcomes: {player_id: per_hole_outcome}. One occurrence per hole/season.

    ``field_avg`` is raised for blowout outcomes so ``player_score = field_avg -
    outcome`` stays >= 1 and passes schema validation.
    """
    rows = []
    tid = 0
    for pid, outcome in player_outcomes.items():
        for h in range(1, n_holes + 1):
            rows.append({
                "player_id": pid, "tournament_id": f"{year}-T{tid}", "year": year,
                "round": 1, "hole_number": h, "course_slug": "other",
                "hole_id_v25": f"other:{h}", "par": 4,
                "player_score": field_avg - outcome, "field_avg_score": field_avg,
            })
            tid += 1
    return pd.DataFrame(rows)


def _results(finish_ranks, event="E24", season=2024, course=TARGET_COURSE, col="finish_rank"):
    return pd.DataFrame({
        "event_id": [event] * len(finish_ranks),
        "predict_season": [season] * len(finish_ranks),
        "target_course_slug": [course] * len(finish_ranks),
        "player_id": list(finish_ranks.keys()),
        col: list(finish_ranks.values()),
    })


# --------------------------------------------------------------------------- #
# Pure metric helpers
# --------------------------------------------------------------------------- #
def test_pearson_corr():
    assert pearson_corr([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson_corr([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(pearson_corr([1, 1, 1], [1, 2, 3]))  # constant
    assert np.isnan(pearson_corr([1], [1]))              # too few


def test_spearman_corr_is_rank_based():
    # Monotonic but nonlinear -> Spearman 1.0, Pearson < 1.0.
    x, y = [1, 2, 3, 4], [1, 4, 9, 16]
    assert spearman_corr(x, y) == pytest.approx(1.0)
    assert pearson_corr(x, y) < 1.0


def test_top_k_hit_rate():
    assert top_k_hit_rate([3, 2, 1], [3, 2, 1], 1) == pytest.approx(1.0)
    assert top_k_hit_rate([3, 2, 1], [3, 2, 1], 2) == pytest.approx(1.0)
    assert top_k_hit_rate([3, 2, 1], [1, 2, 3], 1) == pytest.approx(0.0)


def test_top_k_lift():
    # Model's #1 pick has performance 10 vs field mean 0 -> lift 10.
    assert top_k_lift([3, 2, 1], [10, 0, -10], 1) == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Happy path: perfectly ordered field
# --------------------------------------------------------------------------- #
def test_perfect_prediction_metrics():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results({"pA": 1, "pB": 2, "pC": 3})  # lower finish = better
    res = run_backtest(
        hist, sim, results, outcome_col="finish_rank", higher_is_better=False,
        params=LOOSE, top_k=(1, 2),
    )
    assert res.summary["spearman_pooled"] == pytest.approx(1.0)
    assert res.summary["pearson_pooled"] == pytest.approx(1.0)
    assert res.per_event.iloc[0]["spearman"] == pytest.approx(1.0)
    assert res.summary["hit_rate_1_mean"] == pytest.approx(1.0)
    assert res.summary["lift_1_mean"] > 0
    assert res.summary["n_events"] == 1
    assert res.summary["n_pairs"] == 3


def test_higher_is_better_outcome():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    # strokes_gained: higher is better, and it matches the advantage order.
    results = _results({"pA": 3, "pB": 2, "pC": 1}, col="strokes_gained")
    res = run_backtest(
        hist, sim, results, outcome_col="strokes_gained", higher_is_better=True,
        params=LOOSE, top_k=(1,),
    )
    assert res.summary["spearman_pooled"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def test_future_and_target_season_rows_do_not_change_predictions():
    sim = make_similar_holes(3)
    base_hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results({"pA": 1, "pB": 2, "pC": 3})

    base = run_backtest(base_hist, sim, results, outcome_col="finish_rank",
                        higher_is_better=False, params=LOOSE, top_k=(1,))
    base_pC = base.predictions.set_index("player_id").loc["pC", "course_advantage"]

    # Plant target-season (2024) and future (2025) blowout rows for pC
    # (field_avg raised so the huge outcome keeps player_score valid).
    leaky = pd.concat([
        base_hist,
        make_history({"pC": 100}, year=2024, field_avg=104),
        make_history({"pC": 100}, year=2025, field_avg=104),
    ], ignore_index=True)
    leaked = run_backtest(leaky, sim, results, outcome_col="finish_rank",
                          higher_is_better=False, params=LOOSE, top_k=(1,))
    leaked_pC = leaked.predictions.set_index("player_id").loc["pC", "course_advantage"]

    # Prediction is unchanged: the post-cutoff rows were excluded.
    assert leaked_pC == pytest.approx(base_pC)
    assert leaked.predictions.set_index("player_id").loc["pC", "rank"] == 3
    assert leaked.summary["spearman_pooled"] == pytest.approx(1.0)


def test_out_of_window_history_is_excluded():
    sim = make_similar_holes(3)
    # Only ancient history (2015) -> outside the 5y window for a 2024 event.
    hist = make_history({"pA": 3, "pB": 2, "pC": 1}, year=2015)
    results = _results({"pA": 1, "pB": 2, "pC": 3})
    res = run_backtest(hist, sim, results, outcome_col="finish_rank",
                       higher_is_better=False, params=LOOSE, top_k=(1,))
    # No eligible history -> everyone withheld -> no scored pairs.
    assert res.summary["n_pairs"] == 0
    assert (~res.predictions["covered"]).all()


# --------------------------------------------------------------------------- #
# Coverage handling
# --------------------------------------------------------------------------- #
def test_low_coverage_players_excluded_from_metrics():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})  # pD has no history
    results = _results({"pA": 1, "pB": 2, "pC": 3, "pD": 4})
    res = run_backtest(hist, sim, results, outcome_col="finish_rank",
                       higher_is_better=False, params=LOOSE, top_k=(1,))
    ev = res.per_event.iloc[0]
    assert ev["n_field"] == 4
    assert ev["n_covered"] == 3
    assert ev["coverage"] == pytest.approx(0.75)
    pd_row = res.predictions.set_index("player_id").loc["pD"]
    assert bool(pd_row["covered"]) is False
    assert bool(pd_row["low_coverage"]) is True
    # Metrics still perfect on the three covered players.
    assert ev["spearman"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Multi-event, determinism, report, errors, no-mutation
# --------------------------------------------------------------------------- #
def test_multi_event_and_determinism():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    r1 = _results({"pA": 1, "pB": 2, "pC": 3}, event="E1")
    r2 = _results({"pA": 1, "pB": 2, "pC": 3}, event="E2")
    results = pd.concat([r2, r1], ignore_index=True)  # deliberately out of order
    a = run_backtest(hist, sim, results, outcome_col="finish_rank",
                     higher_is_better=False, params=LOOSE, top_k=(1,))
    b = run_backtest(hist, sim, results, outcome_col="finish_rank",
                     higher_is_better=False, params=LOOSE, top_k=(1,))
    pd.testing.assert_frame_equal(a.per_event, b.per_event)
    # Events reported in deterministic (season, event_id) order.
    assert a.per_event["event_id"].tolist() == ["E1", "E2"]
    assert a.summary["n_events"] == 2


def test_error_metrics_optional():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results({"pA": 1, "pB": 2, "pC": 3})
    res = run_backtest(hist, sim, results, outcome_col="finish_rank",
                       higher_is_better=False, params=LOOSE, top_k=(1,), error_metrics=True)
    assert "mae" in res.per_event.columns and "rmse" in res.per_event.columns


def test_to_markdown_report():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results({"pA": 1, "pB": 2, "pC": 3})
    res = run_backtest(hist, sim, results, outcome_col="finish_rank",
                       higher_is_better=False, params=LOOSE, top_k=(1,))
    md = res.to_markdown()
    assert isinstance(md, str) and "backtest" in md.lower()


def test_missing_results_column_raises():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3})
    results = _results({"pA": 1}).drop(columns=["finish_rank"])
    with pytest.raises(BacktestError):
        run_backtest(hist, sim, results, outcome_col="finish_rank",
                     higher_is_better=False, params=LOOSE)


def test_invalid_history_raises_schema_error():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3}).drop(columns=["field_avg_score"])
    results = _results({"pA": 1})
    with pytest.raises(SchemaError):
        run_backtest(hist, sim, results, outcome_col="finish_rank",
                     higher_is_better=False, params=LOOSE)


def test_backtest_does_not_mutate_inputs():
    sim = make_similar_holes(3)
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results({"pA": 1, "pB": 2, "pC": 3})
    sim_b, hist_b, res_b = sim.copy(), hist.copy(), results.copy()
    run_backtest(hist, sim, results, outcome_col="finish_rank",
                 higher_is_better=False, params=LOOSE, top_k=(1,))
    pd.testing.assert_frame_equal(sim, sim_b)
    pd.testing.assert_frame_equal(hist, hist_b)
    pd.testing.assert_frame_equal(results, res_b)
