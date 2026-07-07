"""Tests for batch tournament-field ranking (issue #34).

Fake field, fake history, fake similar-hole sets. Covers deterministic ordering,
low-coverage flagging, field-input shapes, the export manifest, and the CLI —
all offline, no real ``courses/`` outputs, no committed generated rankings.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    FIELD_RANKING_COLUMNS,
    AdvantageParams,
    SchemaError,
    export_field_ranking,
    score_tournament_field,
)
from pipeline.modeling.player_course_advantage import batch as batch_mod
from pipeline.modeling.pointcloud.export_similarity import RESULTS_FILENAME

TARGET_COURSE = "augusta_national"
PREDICT_SEASON = 2024
LOOSE = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=1)


# --------------------------------------------------------------------------- #
# Fake data builders
# --------------------------------------------------------------------------- #
def make_similar_holes(n_holes=18, target_course=TARGET_COURSE):
    """Each target hole h has one candidate other:h with weight 1.0."""
    rows = []
    for h in range(1, n_holes + 1):
        rows.append({
            "target_course_slug": target_course,
            "target_hole_number": h,
            "target_hole_id": f"{target_course}:{h}",
            "candidate_course_slug": "other",
            "candidate_hole_number": h,
            "candidate_hole_id": f"other:{h}",
            "rank": 1,
            "total_score": 1.0,
            "similarity_weight": 1.0,
            "weight_method": "manual",
            "config_name": "baseline",
        })
    return pd.DataFrame(rows)


def make_history(specs):
    """specs: list of (player_id, per_hole_outcome, n_holes[, player_name])."""
    rows = []
    tid = 0
    for spec in specs:
        pid, outcome, n_holes = spec[0], spec[1], spec[2]
        pname = spec[3] if len(spec) > 3 else None
        for h in range(1, n_holes + 1):
            rec = {
                "player_id": pid,
                "tournament_id": f"T{tid}",
                "year": 2023,
                "round": 1,
                "hole_number": h,
                "course_slug": "other",
                "hole_id_v25": f"other:{h}",
                "par": 4,
                "player_score": 4 - outcome,  # field 4 -> outcome = 4 - player_score
                "field_avg_score": 4,
            }
            if pname is not None:
                rec["player_name"] = pname
            rows.append(rec)
            tid += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Shape / columns
# --------------------------------------------------------------------------- #
def test_ranking_has_expected_columns_in_order():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18)])
    ranking = score_tournament_field(hist, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert list(ranking.columns) == list(FIELD_RANKING_COLUMNS)


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #
def test_ranking_orders_by_advantage_then_low_coverage_last():
    sim = make_similar_holes(18)
    # pA sum=36, pB sum=18, pC no history (withheld).
    hist = make_history([("pA", 2, 18), ("pB", 1, 18)])
    ranking = score_tournament_field(
        hist, sim, ["pC", "pB", "pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    assert ranking["player_id"].tolist() == ["pA", "pB", "pC"]
    assert ranking["rank"].tolist() == [1, 2, 3]
    assert ranking.loc[ranking["player_id"] == "pA", "course_advantage"].iloc[0] == pytest.approx(36.0)
    assert ranking.loc[ranking["player_id"] == "pB", "course_advantage"].iloc[0] == pytest.approx(18.0)


def test_low_coverage_player_ranked_last_and_flagged():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18)])
    ranking = score_tournament_field(
        hist, sim, ["pA", "ghost"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    ghost = ranking.loc[ranking["player_id"] == "ghost"].iloc[0]
    assert bool(ghost["low_coverage"]) is True
    assert ghost["reason"] == "no_player_history"
    assert pd.isna(ghost["course_advantage"])
    assert int(ghost["rank"]) == 2  # ranked last, still ranked


def test_tie_break_by_player_id():
    sim = make_similar_holes(18)
    # Same advantage & coverage for two players -> player_id ascending decides.
    hist = make_history([("pZ", 1, 18), ("pA", 1, 18)])
    ranking = score_tournament_field(
        hist, sim, ["pZ", "pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE
    )
    assert ranking["player_id"].tolist() == ["pA", "pZ"]


def test_ranking_is_deterministic():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18), ("pB", 1, 18)])
    a = score_tournament_field(hist, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    b = score_tournament_field(hist, sim, ["pB", "pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# Field input shapes
# --------------------------------------------------------------------------- #
def test_field_as_list_and_dataframe_equivalent():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18), ("pB", 1, 18)])
    as_list = score_tournament_field(hist, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    as_df = score_tournament_field(
        hist, sim, pd.DataFrame({"player_id": ["pA", "pB"]}),
        TARGET_COURSE, PREDICT_SEASON, params=LOOSE,
    )
    pd.testing.assert_frame_equal(as_list, as_df)


def test_field_dataframe_supplies_player_name():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18)])  # no player_name in history
    field = pd.DataFrame({"player_id": ["pA"], "player_name": ["Alice Ace"]})
    ranking = score_tournament_field(hist, sim, field, TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert ranking.loc[0, "player_name"] == "Alice Ace"


def test_history_player_name_preferred_when_present():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18, "History Name")])
    field = pd.DataFrame({"player_id": ["pA"], "player_name": ["Field Name"]})
    ranking = score_tournament_field(hist, sim, field, TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    assert ranking.loc[0, "player_name"] == "History Name"


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_empty_field_raises():
    sim = make_similar_holes(2)
    hist = make_history([("pA", 2, 2)])
    with pytest.raises(batch_mod.AdvantageScorerError):
        score_tournament_field(hist, sim, [], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)


def test_unknown_aggregate_raises():
    sim = make_similar_holes(2)
    hist = make_history([("pA", 2, 2)])
    with pytest.raises(batch_mod.AdvantageScorerError):
        score_tournament_field(
            hist, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE, aggregate="median"
        )


def test_invalid_history_raises_schema_error():
    sim = make_similar_holes(2)
    hist = make_history([("pA", 2, 2)]).drop(columns=["field_avg_score"])
    with pytest.raises(SchemaError):
        score_tournament_field(hist, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)


def test_no_mutation_of_inputs():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18), ("pB", 1, 18)])
    sim_before, hist_before = sim.copy(), hist.copy()
    score_tournament_field(hist, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    pd.testing.assert_frame_equal(sim, sim_before)
    pd.testing.assert_frame_equal(hist, hist_before)


def test_mean_aggregate():
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18)])  # every hole advantage 2 -> mean 2
    ranking = score_tournament_field(
        hist, sim, ["pA"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE, aggregate="mean"
    )
    assert ranking.loc[0, "course_advantage"] == pytest.approx(2.0)
    assert ranking.loc[0, "course_advantage_mean"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Export + manifest
# --------------------------------------------------------------------------- #
def test_export_writes_csv_and_manifest(tmp_path):
    sim = make_similar_holes(18)
    hist = make_history([("pA", 2, 18), ("pB", 1, 18)])
    ranking = score_tournament_field(hist, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)

    out = tmp_path / "run"
    paths = export_field_ranking(
        ranking, out, target_course_slug=TARGET_COURSE, config_name="baseline",
        predict_season=PREDICT_SEASON, params=LOOSE, history_rows=len(hist),
    )
    assert paths["ranking"].exists() and paths["manifest"].exists()

    reloaded = pd.read_csv(paths["ranking"])
    assert reloaded["player_id"].tolist() == ["pA", "pB"]

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["target_course_slug"] == TARGET_COURSE
    assert manifest["n_players"] == 2
    assert manifest["n_covered"] == 2
    assert manifest["history_rows"] == len(hist)
    assert manifest["params"]["min_holes_covered"] == 1
    assert "created_at" in manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_v25(root, n_holes=3, course=TARGET_COURSE):
    rows = [
        {"target_hole_id": f"{course}:{h}", "candidate_hole_id": f"other:{h}",
         "rank": 1, "total_score": 1.0}
        for h in range(1, n_holes + 1)
    ]
    cfg = root / "pointcloud_similarity" / "baseline"
    cfg.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg / RESULTS_FILENAME, index=False)


def test_cli_end_to_end(tmp_path):
    _write_v25(tmp_path, n_holes=3)
    hist = make_history([("pA", 2, 3), ("pB", 1, 3)])
    hist_csv = tmp_path / "history.csv"
    hist.to_csv(hist_csv, index=False)
    out_dir = tmp_path / "out"

    ranking = batch_mod.main([
        "--root", str(tmp_path),
        "--history", str(hist_csv),
        "--course", TARGET_COURSE,
        "--predict-season", str(PREDICT_SEASON),
        "--min-occurrences", "1",
        "--min-holes", "1",
        "--out", str(out_dir),
    ])
    assert ranking["player_id"].tolist() == ["pA", "pB"]
    assert (out_dir / "player_rankings.csv").exists()
    assert (out_dir / "manifest.json").exists()


def test_cli_default_field_is_all_history_players(tmp_path):
    _write_v25(tmp_path, n_holes=3)
    hist = make_history([("pA", 2, 3), ("pB", 1, 3)])
    hist_csv = tmp_path / "history.csv"
    hist.to_csv(hist_csv, index=False)

    ranking = batch_mod.main([
        "--root", str(tmp_path),
        "--history", str(hist_csv),
        "--course", TARGET_COURSE,
        "--predict-season", str(PREDICT_SEASON),
        "--min-occurrences", "1",
        "--min-holes", "1",
    ])
    assert set(ranking["player_id"]) == {"pA", "pB"}
