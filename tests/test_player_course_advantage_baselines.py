"""Tests for the simple comparison baselines (issue #38).

Fake data only. Verifies each baseline's rule, that they are drop-in rankers for
the backtest, the side-by-side comparison table, and the honest
``model_beats_baselines`` logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    BASELINES,
    AdvantageParams,
    SchemaError,
    compare_baselines,
    model_beats_baselines,
    run_backtest,
)
from pipeline.modeling.player_course_advantage.batch import FIELD_RANKING_COLUMNS
from pipeline.modeling.player_course_advantage.baselines import (
    MODEL_NAME,
    course_history_baseline,
    null_baseline,
    recent_form_baseline,
    same_par_baseline,
    season_average_baseline,
)

TARGET_COURSE = "augusta_national"
PREDICT_SEASON = 2024
LOOSE = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=1)


def make_sim(n_holes=3, cand_course="other", target_course=TARGET_COURSE):
    rows = []
    for h in range(1, n_holes + 1):
        rows.append({
            "target_course_slug": target_course, "target_hole_number": h,
            "target_hole_id": f"{target_course}:{h}",
            "candidate_course_slug": cand_course, "candidate_hole_number": h,
            "candidate_hole_id": f"{cand_course}:{h}",
            "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
            "weight_method": "manual", "config_name": "baseline",
        })
    return pd.DataFrame(rows)


def hist(rows):
    out = []
    for i, r in enumerate(rows):
        slug, num = r["hole_id_v25"].split(":")
        rec = {
            "player_id": r["player_id"], "tournament_id": r.get("tournament_id", f"T{i}"),
            "year": r.get("year", 2023), "round": r.get("round", 1),
            "hole_number": int(num), "course_slug": slug, "hole_id_v25": r["hole_id_v25"],
            "par": r.get("par", 4), "player_score": r["player_score"],
            "field_avg_score": r["field_avg_score"],
        }
        out.append(rec)
    return pd.DataFrame(out)


def _rows_for(player, hole_ids, outcome, *, year=2023, par=4, field_avg=4):
    return [
        {"player_id": player, "year": year, "hole_id_v25": hid,
         "player_score": field_avg - outcome, "field_avg_score": field_avg, "par": par}
        for hid in hole_ids
    ]


# --------------------------------------------------------------------------- #
# Shape / drop-in ranker
# --------------------------------------------------------------------------- #
def test_baseline_output_shape():
    sim = make_sim(3)
    h = hist(_rows_for("pA", ["x:1", "x:2", "x:3"], 2))
    out = recent_form_baseline(h, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert list(out.columns) == list(FIELD_RANKING_COLUMNS)


def test_baseline_is_drop_in_ranker_for_backtest():
    sim = make_sim(3)
    h = hist(
        _rows_for("pA", ["other:1", "other:2", "other:3"], 3)
        + _rows_for("pB", ["other:1", "other:2", "other:3"], 1)
    )
    results = pd.DataFrame({
        "event_id": ["E"] * 2, "predict_season": [2024] * 2,
        "target_course_slug": [TARGET_COURSE] * 2,
        "player_id": ["pA", "pB"], "finish_rank": [1, 2],
    })
    res = run_backtest(h, sim, results, outcome_col="finish_rank",
                       higher_is_better=False, params=LOOSE, top_k=(1,),
                       ranker=recent_form_baseline)
    assert res.summary["n_pairs"] == 2


# --------------------------------------------------------------------------- #
# Each baseline's rule
# --------------------------------------------------------------------------- #
def test_recent_form_ranks_by_general_form():
    sim = make_sim(3)
    h = hist(
        _rows_for("pA", ["a:1", "a:2", "a:3"], 2)   # +2 avg
        + _rows_for("pB", ["b:1", "b:2", "b:3"], 1)  # +1 avg
    )
    out = recent_form_baseline(h, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert out["player_id"].tolist() == ["pA", "pB"]
    assert out.loc[out["player_id"] == "pA", "course_advantage"].iloc[0] == pytest.approx(2.0)


def test_course_history_only_uses_target_course():
    sim = make_sim(3)
    h = hist(
        # pA has target-course history; pB only plays elsewhere.
        _rows_for("pA", [f"{TARGET_COURSE}:1", f"{TARGET_COURSE}:2", f"{TARGET_COURSE}:3"], 2)
        + _rows_for("pB", ["elsewhere:1", "elsewhere:2", "elsewhere:3"], 1)
    )
    out = course_history_baseline(h, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON,
                                  params=AdvantageParams(min_occurrences_per_hole=1))
    pa = out.set_index("player_id").loc["pA"]
    pb = out.set_index("player_id").loc["pB"]
    assert pa["course_advantage"] == pytest.approx(2.0)
    assert bool(pb["low_coverage"]) is True          # no target-course history
    assert pd.isna(pb["course_advantage"])


def test_same_par_baseline_uses_only_matching_par():
    # Target hole 1's candidate is a par-3 hole -> target par set = {3}.
    sim = pd.DataFrame([{
        "target_course_slug": TARGET_COURSE, "target_hole_number": 1,
        "target_hole_id": f"{TARGET_COURSE}:1",
        "candidate_course_slug": "src", "candidate_hole_number": 3,
        "candidate_hole_id": "src:3", "rank": 1, "total_score": 1.0,
        "similarity_weight": 1.0, "weight_method": "manual", "config_name": "baseline",
    }])
    h = hist(
        _rows_for("pA", ["src:3", "src:7", "src:9"], 2, par=3)      # par-3: good (+2)
        + _rows_for("pA", ["big:1", "big:2", "big:3"], -5, par=5)   # par-5: bad, must be ignored
    )
    out = same_par_baseline(h, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON,
                            params=AdvantageParams(min_occurrences_per_hole=1))
    # Only the par-3 rows count -> advantage is the positive +2, not dragged down.
    assert out.loc[0, "course_advantage"] == pytest.approx(2.0)


def test_season_average_ranks_by_raw_score():
    sim = make_sim(3)
    h = hist(
        _rows_for("pA", ["a:1", "a:2", "a:3"], 2)   # player_score 2
        + _rows_for("pB", ["b:1", "b:2", "b:3"], 0)  # player_score 4
    )
    out = season_average_baseline(h, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    # Lower raw score is better -> pA first; score = -mean(player_score).
    assert out["player_id"].tolist() == ["pA", "pB"]
    assert out.loc[out["player_id"] == "pA", "course_advantage"].iloc[0] == pytest.approx(-2.0)


def test_null_baseline_is_constant_zero():
    sim = make_sim(3)
    h = hist(
        _rows_for("pA", ["a:1", "a:2", "a:3"], 2)
        + _rows_for("pB", ["b:1", "b:2", "b:3"], 1)
    )
    out = null_baseline(h, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert (out["course_advantage"] == 0.0).all()


# --------------------------------------------------------------------------- #
# Leakage still enforced in baselines
# --------------------------------------------------------------------------- #
def test_baseline_respects_prediction_window():
    sim = make_sim(3)
    base = hist(_rows_for("pA", ["a:1", "a:2", "a:3"], 2, year=2023))
    out_base = recent_form_baseline(base, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)

    leaky = pd.concat([
        base,
        hist(_rows_for("pA", ["a:1", "a:2", "a:3"], 100, year=2024, field_avg=104)),
    ], ignore_index=True)
    out_leaky = recent_form_baseline(leaky, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert out_leaky.loc[0, "course_advantage"] == pytest.approx(out_base.loc[0, "course_advantage"])


# --------------------------------------------------------------------------- #
# Comparison table + honest verdict
# --------------------------------------------------------------------------- #
def test_compare_baselines_table():
    sim = make_sim(3)
    h = hist(
        _rows_for("pA", ["other:1", "other:2", "other:3"], 3)
        + _rows_for("pB", ["other:1", "other:2", "other:3"], 2)
        + _rows_for("pC", ["other:1", "other:2", "other:3"], 1)
    )
    results = pd.DataFrame({
        "event_id": ["E"] * 3, "predict_season": [2024] * 3,
        "target_course_slug": [TARGET_COURSE] * 3,
        "player_id": ["pA", "pB", "pC"], "finish_rank": [1, 2, 3],
    })
    table = compare_baselines(h, sim, results, outcome_col="finish_rank",
                              higher_is_better=False, params=LOOSE, top_k=(1,))
    rankers = set(table["ranker"])
    assert MODEL_NAME in rankers
    assert set(BASELINES).issubset(rankers)
    assert "spearman_pooled" in table.columns
    # sorted best-first (NaN last).
    vals = table["spearman_pooled"].dropna().tolist()
    assert vals == sorted(vals, reverse=True)


def test_model_beats_baselines_logic():
    df = pd.DataFrame({
        "ranker": [MODEL_NAME, "null", "recent_form"],
        "spearman_pooled": [0.9, 0.1, 0.5],
    })
    assert model_beats_baselines(df) is True

    tie = pd.DataFrame({
        "ranker": [MODEL_NAME, "recent_form"],
        "spearman_pooled": [0.5, 0.5],
    })
    assert model_beats_baselines(tie) is False  # not strictly greater

    nan_model = pd.DataFrame({
        "ranker": [MODEL_NAME, "recent_form"],
        "spearman_pooled": [np.nan, 0.3],
    })
    assert model_beats_baselines(nan_model) is False


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_invalid_history_raises_schema_error():
    sim = make_sim(3)
    h = hist(_rows_for("pA", ["a:1"], 2)).drop(columns=["field_avg_score"])
    with pytest.raises(SchemaError):
        recent_form_baseline(h, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)


def test_baseline_does_not_mutate_inputs():
    sim = make_sim(3)
    h = hist(_rows_for("pA", ["a:1", "a:2", "a:3"], 2))
    sim_b, h_b = sim.copy(), h.copy()
    recent_form_baseline(h, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    pd.testing.assert_frame_equal(sim, sim_b)
    pd.testing.assert_frame_equal(h, h_b)
