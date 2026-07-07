"""Tests for player-course advantage diagnostics / explanation outputs (issue #39).

Fake data only. The central guarantee is *reconciliation*: the explainer
re-derives the same contributions the scorer aggregates, so its breakdowns must
sum back to the scorer's hole and course advantages. Also covers missing-data
explanations, serializability, and no-mutation.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    SchemaError,
    contribution_rows,
    explain_player_course,
    score_player_holes,
)
from pipeline.modeling.player_course_advantage.scorer import REASON_NO_PLAYER_HISTORY

TARGET_COURSE = "augusta_national"
PREDICT_SEASON = 2024
LOOSE = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=1)


# --------------------------------------------------------------------------- #
# Fake data builders (mirror the scorer tests)
# --------------------------------------------------------------------------- #
def make_similar_holes(pairs, target_course=TARGET_COURSE):
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


def make_history(rows):
    out = []
    for i, r in enumerate(rows):
        slug, hnum = r["hole_id_v25"].split(":")
        out.append({
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
        })
    return pd.DataFrame(out)


def _fixture():
    # riviera:5 is shared across both target holes; 2022 row exercises recency.
    sim = make_similar_holes({
        1: [("riviera:5", 0.6), ("tpc:9", 0.4)],
        2: [("riviera:5", 0.5), ("pebble:3", 0.5)],
    })
    hist = make_history([
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},   # +1
        {"player_id": "p", "year": 2022, "round": 1, "hole_id_v25": "riviera:5",
         "player_score": 5, "field_avg_score": 4},   # -1
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "tpc:9",
         "player_score": 2, "field_avg_score": 4},    # +2
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "pebble:3",
         "player_score": 4, "field_avg_score": 4},    # 0
    ])
    return sim, hist


def _explain(sim, hist, aggregate="sum", params=LOOSE):
    return explain_player_course(
        hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=params, aggregate=aggregate
    )


# --------------------------------------------------------------------------- #
# Atomic contribution frame
# --------------------------------------------------------------------------- #
def test_contribution_rows_expose_both_weights():
    sim, hist = _fixture()
    rows = contribution_rows(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    for col in ("similarity_weight", "recency_weight", "row_weight", "contribution"):
        assert col in rows.columns
    # row_weight is exactly similarity_weight * recency_weight.
    assert (
        (rows["row_weight"] - rows["similarity_weight"] * rows["recency_weight"]).abs().max()
        < 1e-12
    )
    # 2022 row (age 1) has recency < 1; 2023 rows (age 0) have recency == 1.
    assert (rows.loc[rows["year"] == 2023, "recency_weight"] == 1.0).all()
    assert (rows.loc[rows["year"] == 2022, "recency_weight"] < 1.0).all()


# --------------------------------------------------------------------------- #
# Reconciliation with the scorer
# --------------------------------------------------------------------------- #
def test_contributions_reconcile_hole_advantage():
    sim, hist = _fixture()
    rows = contribution_rows(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    per_hole = score_player_holes(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)

    for thid, grp in rows.groupby("target_hole_id"):
        expected = grp["contribution"].sum() / grp["row_weight"].sum()
        actual = per_hole.loc[
            per_hole["target_hole_id"] == thid, "hole_advantage"
        ].iloc[0]
        assert actual == pytest.approx(expected)


def test_similar_hole_weighted_contributions_sum_to_hole_advantage():
    sim, hist = _fixture()
    expl = _explain(sim, hist)
    sc = expl.similar_hole_contributions
    per_hole = expl.hole_contributions

    # fractional weights sum to 1 within each target hole ...
    frac = sc.groupby("target_hole_id")["fractional_weight"].sum()
    assert (frac - 1.0).abs().max() < 1e-9
    # ... and weighted_contribution sums to that hole's advantage.
    rolled = sc.groupby("target_hole_id")["weighted_contribution"].sum()
    for thid, val in rolled.items():
        adv = per_hole.loc[per_hole["target_hole_id"] == thid, "hole_advantage"].iloc[0]
        assert adv == pytest.approx(val)


def test_covered_holes_sum_to_course_advantage():
    sim, hist = _fixture()
    expl = _explain(sim, hist, aggregate="sum")
    covered = expl.hole_contributions[expl.hole_contributions["course_contribution"].notna()]
    assert covered["course_contribution"].sum() == pytest.approx(
        expl.course_summary["course_advantage"]
    )


def test_mean_aggregate_reconciles():
    sim, hist = _fixture()
    expl = _explain(sim, hist, aggregate="mean")
    covered = expl.hole_contributions[expl.hole_contributions["course_contribution"].notna()]
    assert covered["course_contribution"].mean() == pytest.approx(
        expl.course_summary["course_advantage"]
    )


# --------------------------------------------------------------------------- #
# Occurrence-by-year diagnostics
# --------------------------------------------------------------------------- #
def test_occurrence_year_counts_reconcile():
    sim, hist = _fixture()
    expl = _explain(sim, hist)
    rows = contribution_rows(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    yc = expl.occurrence_year_counts
    assert int(yc["occurrences"].sum()) == len(rows)
    # riviera:5 under target hole 1 spans 2022 and 2023 -> two year rows.
    t1_riviera_years = rows[(rows["target_hole_number"] == 1) & (rows["candidate_hole_id"] == "riviera:5")]
    assert set(t1_riviera_years["year"]) == {2022, 2023}


# --------------------------------------------------------------------------- #
# Top-k explanation views
# --------------------------------------------------------------------------- #
def test_top_target_and_similar_holes_sorted():
    sim, hist = _fixture()
    expl = _explain(sim, hist)
    top_h = expl.top_target_holes(k=2)
    assert list(top_h["course_contribution"].abs()) == sorted(
        top_h["course_contribution"].abs(), reverse=True
    )
    top_s = expl.top_similar_holes(k=3)
    assert list(top_s["weighted_contribution"].abs()) == sorted(
        top_s["weighted_contribution"].abs(), reverse=True
    )


# --------------------------------------------------------------------------- #
# Missing-data explanations
# --------------------------------------------------------------------------- #
def test_low_coverage_reasons_are_clear():
    # min_occurrences_per_hole default is 3; hole 2 has only 1 occurrence.
    sim = make_similar_holes({1: [("riviera:5", 1.0)], 2: [("pebble:3", 1.0)]})
    hist = make_history([
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
        {"player_id": "p", "year": 2023, "round": 2, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
        {"player_id": "p", "year": 2022, "round": 1, "hole_id_v25": "riviera:5",
         "player_score": 3, "field_avg_score": 4},
        {"player_id": "p", "year": 2023, "round": 1, "hole_id_v25": "pebble:3",
         "player_score": 4, "field_avg_score": 4},
    ])
    expl = explain_player_course(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON)
    lc = expl.low_coverage
    assert set(lc["target_hole_number"]) == {2}
    assert lc.iloc[0]["reason"] == "below_min_occurrences"
    assert (lc["reason"].notna()).all()


def test_no_player_history_explanation():
    sim, hist = _fixture()
    expl = explain_player_course(hist, sim, "ghost", TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert expl.course_summary["reason"] == REASON_NO_PLAYER_HISTORY
    assert expl.similar_hole_contributions.empty
    assert expl.occurrence_year_counts.empty
    # Every target hole is explained as low-coverage.
    assert set(expl.low_coverage["target_hole_number"]) == {1, 2}
    assert (expl.low_coverage["reason"] == REASON_NO_PLAYER_HISTORY).all()


# --------------------------------------------------------------------------- #
# Serializability, validation, no mutation
# --------------------------------------------------------------------------- #
def test_explanation_is_serializable():
    sim, hist = _fixture()
    expl = _explain(sim, hist)
    records = expl.to_records()
    for key in ("hole_contributions", "similar_hole_contributions",
                "occurrence_year_counts", "low_coverage", "course_summary"):
        assert key in records
    assert isinstance(records["hole_contributions"], list)
    # Round-trips through JSON (numpy scalars stringified as a smoke check).
    json.dumps(records, default=str, allow_nan=True)


def test_invalid_history_raises_schema_error():
    sim, hist = _fixture()
    bad = hist.drop(columns=["field_avg_score"])
    with pytest.raises(SchemaError):
        explain_player_course(bad, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE)


def test_diagnostics_do_not_mutate_inputs():
    sim, hist = _fixture()
    sim_before, hist_before = sim.copy(), hist.copy()
    explain_player_course(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    contribution_rows(hist, sim, "p", TARGET_COURSE, PREDICT_SEASON, LOOSE)
    pd.testing.assert_frame_equal(sim, sim_before)
    pd.testing.assert_frame_equal(hist, hist_before)
