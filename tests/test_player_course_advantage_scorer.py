"""Tests for the recency-weighted player-course advantage scorer (issue #33).

Fake, hand-verifiable data only — no real ``courses/`` outputs, no network, no
v2.5 scorer. Covers recency weighting, the leakage window filter, the weighted
mean (similarity x recency), coverage gating / withholding, course aggregation,
current-course exclusion, determinism, and no-mutation guarantees.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    DEFAULT_PARAMS,
    AdvantageParams,
    SchemaError,
    filter_history_for_prediction_window,
    recency_weight,
    score_player_course,
    score_player_holes,
)
from pipeline.modeling.player_course_advantage.scorer import (
    REASON_BELOW_MIN_HOLES_COVERED,
    REASON_BELOW_MIN_OCCURRENCES,
    REASON_NO_ELIGIBLE_HISTORY,
    REASON_NO_PLAYER_HISTORY,
)

TARGET_COURSE = "augusta_national"
PREDICT_SEASON = 2024

# Loose params for arithmetic tests: every hole with >=1 occurrence is "covered",
# so hand-computed advantages are not masked by coverage gating.
LOOSE = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=1)


# --------------------------------------------------------------------------- #
# Fake data builders
# --------------------------------------------------------------------------- #
def make_similar_holes(
    pairs: dict[int, list[tuple[str, float]]], target_course: str = TARGET_COURSE
) -> pd.DataFrame:
    """One row per (target hole, candidate) — mirrors load_similar_hole_sets output.

    ``pairs`` maps a target hole number to ``[(candidate_hole_id, similarity_weight)]``.
    """
    rows = []
    for hn, cands in pairs.items():
        for rank, (cand_id, weight) in enumerate(cands, start=1):
            cslug, cnum = cand_id.split(":")
            rows.append({
                "target_course_slug": target_course,
                "target_hole_number": hn,
                "target_hole_id": f"{target_course}:{hn}",
                "candidate_course_slug": cslug,
                "candidate_hole_number": int(cnum),
                "candidate_hole_id": cand_id,
                "rank": rank,
                "total_score": float(rank),
                "similarity_weight": weight,
                "weight_method": "manual",
                "config_name": "baseline",
            })
    return pd.DataFrame(rows)


def make_history(rows: list[dict]) -> pd.DataFrame:
    """Build a schema-valid hole-score history from compact row specs.

    Each spec needs ``player_id, year, hole_id_v25, player_score, field_avg_score``;
    ``round``, ``tournament_id``, ``par``, ``player_name``, ``field_adjusted_score``
    are optional. ``tournament_id`` defaults to a unique value so the occurrence
    grain stays unique.
    """
    out = []
    for i, r in enumerate(rows):
        slug, hnum = r["hole_id_v25"].split(":")
        rec = {
            "player_id": r["player_id"],
            "tournament_id": r.get("tournament_id", f"T{i}"),
            "year": r["year"],
            "round": r.get("round", 1),
            "hole_number": int(hnum),
            "course_slug": slug,
            "hole_id_v25": r["hole_id_v25"],
            "par": r.get("par", 4),
            "player_score": r["player_score"],
            "field_avg_score": r["field_avg_score"],
        }
        for opt in ("player_name", "field_adjusted_score"):
            if opt in r:
                rec[opt] = r[opt]
        out.append(rec)
    return pd.DataFrame(out)


def _adv_for_hole(per_hole: pd.DataFrame, hole_number: int):
    return per_hole.loc[per_hole["target_hole_number"] == hole_number].iloc[0]


# --------------------------------------------------------------------------- #
# 1-2. recency_weight
# --------------------------------------------------------------------------- #
def test_recency_weight_prior_season_is_one():
    # age = (2024 - 1) - 2023 = 0 -> weight 1.0 regardless of decay.
    assert recency_weight(2023, 2024, 0.85) == 1.0
    assert recency_weight(2023, 2024, 0.5) == 1.0


def test_recency_weight_decays_by_m_pow_age():
    for age in range(0, 5):
        year = (PREDICT_SEASON - 1) - age
        assert recency_weight(year, PREDICT_SEASON, 0.85) == pytest.approx(0.85 ** age)


# --------------------------------------------------------------------------- #
# 3-5. Prediction-window filter (leakage guard)
# --------------------------------------------------------------------------- #
def _year_df(years: list[int]) -> pd.DataFrame:
    return pd.DataFrame({"year": years, "player_id": ["p"] * len(years)})


def test_filter_excludes_target_season():
    out = filter_history_for_prediction_window(_year_df([2024]), 2024, 5)
    assert out.empty


def test_filter_excludes_future_years():
    out = filter_history_for_prediction_window(_year_df([2025, 2030]), 2024, 5)
    assert out.empty


def test_filter_includes_only_window():
    df = _year_df([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    out = filter_history_for_prediction_window(df, 2024, 5)
    # Eligible: 2019 <= year < 2024.
    assert sorted(out["year"].tolist()) == [2019, 2020, 2021, 2022, 2023]


# --------------------------------------------------------------------------- #
# 6-7. Sign of the advantage
# --------------------------------------------------------------------------- #
def test_positive_advantage_when_player_beats_field():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},  # outcome +1
    ])
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    assert _adv_for_hole(per_hole, 1)["hole_advantage"] == pytest.approx(1.0)


def test_negative_advantage_when_player_worse_than_field():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 5, "field_avg_score": 4},  # outcome -1
    ])
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    assert _adv_for_hole(per_hole, 1)["hole_advantage"] == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# 8. Both similarity and recency weights applied
# --------------------------------------------------------------------------- #
def test_similarity_and_recency_weights_both_applied():
    params = AdvantageParams(min_occurrences_per_hole=1, recency_decay=0.5)
    sim = make_similar_holes({1: [("riviera:5", 0.6), ("tpc:9", 0.4)]})
    hist = make_history([
        # candidate riviera:5, prior season (age 0 -> rec 1.0), outcome +10
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 1, "field_avg_score": 11},
        # candidate tpc:9, age 2 -> rec 0.5**2 = 0.25, outcome +2
        {"player_id": "p", "year": 2021, "hole_id_v25": "tpc:9",
         "player_score": 4, "field_avg_score": 6},
    ])
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params)

    w_a = 0.6 * 1.0            # sim x recency for riviera:5
    w_b = 0.4 * (0.5 ** 2)     # sim x recency for tpc:9
    expected = (w_a * 10 + w_b * 2) / (w_a + w_b)
    assert _adv_for_hole(per_hole, 1)["hole_advantage"] == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 9. Multiple occurrences on the same similar hole
# --------------------------------------------------------------------------- #
def test_multiple_occurrences_same_hole_averaged():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},  # +1
        {"player_id": "p", "year": 2023, "round": 2, "hole_id_v25": "riviera:5",
         "player_score": 2, "field_avg_score": 4},  # +2
    ])
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    # Same year -> equal recency, sim weight equal -> plain mean (1 + 2) / 2.
    row = _adv_for_hole(per_hole, 1)
    assert row["hole_advantage"] == pytest.approx(1.5)
    assert row["raw_occurrences"] == 2
    assert row["similar_holes_used"] == 1


# --------------------------------------------------------------------------- #
# 10. Multiple similar holes roll into one target-hole advantage
# --------------------------------------------------------------------------- #
def test_multiple_similar_holes_roll_up():
    sim = make_similar_holes({1: [("riviera:5", 0.75), ("tpc:9", 0.25)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},  # +1
        {"player_id": "p", "year": 2023, "hole_id_v25": "tpc:9",
         "player_score": 5, "field_avg_score": 4},  # -1
    ])
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    # 0.75*(+1) + 0.25*(-1) = 0.5, weights sum to 1.0.
    row = _adv_for_hole(per_hole, 1)
    assert row["hole_advantage"] == pytest.approx(0.5)
    assert row["similar_holes_used"] == 2


# --------------------------------------------------------------------------- #
# 11-12. Course aggregation
# --------------------------------------------------------------------------- #
def _full_course_fixture():
    """18 target holes; hole h has one occurrence with outcome exactly h."""
    sim = make_similar_holes({h: [(f"other:{h}", 1.0)] for h in range(1, 19)})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": f"other:{h}",
         "player_score": 1, "field_avg_score": 1 + h}
        for h in range(1, 19)
    ])
    return sim, hist


def test_course_aggregate_sum_is_default():
    sim, hist = _full_course_fixture()
    per_hole, summary = score_player_course(
        hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    assert summary["holes_covered"] == 18
    assert summary["total_target_holes"] == 18
    # sum(1..18) = 171 ; mean = 9.5.
    assert summary["course_advantage"] == pytest.approx(171.0)
    assert summary["course_advantage_mean"] == pytest.approx(9.5)


def test_course_aggregate_mean():
    sim, hist = _full_course_fixture()
    _, summary = score_player_course(
        hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE, aggregate="mean"
    )
    assert summary["course_advantage"] == pytest.approx(9.5)
    assert summary["course_advantage_mean"] == pytest.approx(9.5)


# --------------------------------------------------------------------------- #
# 13-15. Coverage gating / withholding
# --------------------------------------------------------------------------- #
def test_low_coverage_hole_is_withheld_not_zero():
    # Default min_occurrences_per_hole = 3; only 2 occurrences -> withheld.
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
        {"player_id": "p", "year": 2023, "round": 2, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
    ])
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON)
    row = _adv_for_hole(per_hole, 1)
    assert row["low_coverage"]
    assert pd.isna(row["hole_advantage"])  # NOT 0.0
    assert row["raw_occurrences"] == 2
    assert row["reason"] == REASON_BELOW_MIN_OCCURRENCES


def test_course_withheld_when_below_min_holes_covered():
    # 5 covered holes, default min_holes_covered = 12 -> course withheld.
    sim = make_similar_holes({h: [(f"other:{h}", 1.0)] for h in range(1, 6)})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": f"other:{h}",
         "player_score": 3, "field_avg_score": 4}
        for h in range(1, 6)
    ])
    params = AdvantageParams(min_occurrences_per_hole=1)  # holes covered, course not
    _, summary = score_player_course(
        hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=params
    )
    assert summary["holes_covered"] == 5
    assert summary["course_advantage"] is None
    assert summary["course_advantage_mean"] is None
    assert summary["low_coverage"] is True
    assert summary["reason"] == REASON_BELOW_MIN_HOLES_COVERED


def test_player_with_no_history_is_low_coverage():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "someone_else", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
    ])
    per_hole, summary = score_player_course(
        hist, sim, "ghost", TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    assert summary["low_coverage"] is True
    assert summary["reason"] == REASON_NO_PLAYER_HISTORY
    assert summary["course_advantage"] is None
    assert bool(_adv_for_hole(per_hole, 1)["low_coverage"]) is True
    assert _adv_for_hole(per_hole, 1)["reason"] == REASON_NO_PLAYER_HISTORY


def test_player_with_only_out_of_window_history():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2010, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},  # far outside the 5y window
    ])
    _, summary = score_player_course(
        hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    assert summary["reason"] == REASON_NO_ELIGIBLE_HISTORY
    assert summary["low_coverage"] is True


# --------------------------------------------------------------------------- #
# 16-17. include_current_course_history
# --------------------------------------------------------------------------- #
def _same_course_fixture():
    # Target hole 1's similar hole is another hole *on the target course itself*.
    sim = make_similar_holes({1: [(f"{TARGET_COURSE}:7", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": f"{TARGET_COURSE}:7",
         "player_score": 2, "field_avg_score": 4},  # outcome +2
    ])
    return sim, hist


def test_current_course_history_excluded_by_default():
    sim, hist = _same_course_fixture()
    params = AdvantageParams(min_occurrences_per_hole=1)  # default include=False
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params)
    row = _adv_for_hole(per_hole, 1)
    assert row["raw_occurrences"] == 0
    assert bool(row["low_coverage"]) is True
    assert pd.isna(row["hole_advantage"])


def test_current_course_history_included_when_enabled():
    sim, hist = _same_course_fixture()
    params = AdvantageParams(
        min_occurrences_per_hole=1, include_current_course_history=True
    )
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params)
    row = _adv_for_hole(per_hole, 1)
    assert row["raw_occurrences"] == 1
    assert row["hole_advantage"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# 18-19. field_adjusted_score present vs computed
# --------------------------------------------------------------------------- #
def test_existing_field_adjusted_score_used():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4, "field_adjusted_score": 1.0},
    ])
    assert "field_adjusted_score" in hist.columns
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    assert _adv_for_hole(per_hole, 1)["hole_advantage"] == pytest.approx(1.0)


def test_missing_field_adjusted_score_is_computed():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
    ])
    assert "field_adjusted_score" not in hist.columns
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    assert _adv_for_hole(per_hole, 1)["hole_advantage"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 20. Coverage diagnostics present
# --------------------------------------------------------------------------- #
def test_output_includes_coverage_diagnostics():
    sim, hist = _full_course_fixture()
    per_hole, summary = score_player_course(
        hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    for col in ("weighted_occurrences", "raw_occurrences", "similar_holes_used",
                "low_coverage", "reason"):
        assert col in per_hole.columns
    for key in ("holes_covered", "total_target_holes", "total_raw_occurrences",
                "total_weighted_occurrences", "low_coverage", "reason"):
        assert key in summary
    assert summary["total_raw_occurrences"] == 18
    assert summary["total_weighted_occurrences"] == pytest.approx(18.0)


# --------------------------------------------------------------------------- #
# 21. Deterministic ordering
# --------------------------------------------------------------------------- #
def test_output_ordering_is_deterministic():
    sim, hist = _full_course_fixture()
    a = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    b = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    pd.testing.assert_frame_equal(a, b)
    expected = a.sort_values(["target_hole_number", "target_hole_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, expected)


# --------------------------------------------------------------------------- #
# 22. Invalid history raises SchemaError
# --------------------------------------------------------------------------- #
def test_invalid_history_raises_schema_error():
    sim = make_similar_holes({1: [("riviera:5", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
    ]).drop(columns=["field_avg_score"])  # break the required contract
    with pytest.raises(SchemaError):
        score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)


# --------------------------------------------------------------------------- #
# 23. No mutation of inputs
# --------------------------------------------------------------------------- #
def test_scorer_does_not_mutate_inputs():
    sim, hist = _full_course_fixture()
    sim_before, hist_before = sim.copy(), hist.copy()
    score_player_course(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    pd.testing.assert_frame_equal(sim, sim_before)
    pd.testing.assert_frame_equal(hist, hist_before)


# --------------------------------------------------------------------------- #
# Default params sanity: DEFAULT_PARAMS is untouched by the loose fixtures.
# --------------------------------------------------------------------------- #
def test_default_params_unchanged():
    assert DEFAULT_PARAMS.min_occurrences_per_hole == 3
    assert DEFAULT_PARAMS.min_holes_covered == 12
