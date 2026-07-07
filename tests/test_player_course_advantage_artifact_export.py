"""Tests for player-course advantage artifact export/load (issue #44).

Fake data written to temp dirs only — no committed generated artifacts. Covers
the assemble -> export -> load round-trip, manifest/parameters metadata, the
optional backtest slot, and the geometry-rejection guard.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    ArtifactExportError,
    assemble_field_outputs,
    export_advantage_run,
    load_advantage_run,
    make_run_id,
)
from pipeline.modeling.player_course_advantage import artifact_export as ax

TARGET_COURSE = "augusta_national"
PREDICT_SEASON = 2024
LOOSE = AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=1)


def make_similar_holes(n_holes=6, target_course=TARGET_COURSE):
    rows = []
    for h in range(1, n_holes + 1):
        for cand in (("riviera", h), ("tpc", h)):
            cslug, cnum = cand
            rows.append({
                "target_course_slug": target_course,
                "target_hole_number": h,
                "target_hole_id": f"{target_course}:{h}",
                "candidate_course_slug": cslug,
                "candidate_hole_number": cnum,
                "candidate_hole_id": f"{cslug}:{cnum}",
                "rank": 1 if cslug == "riviera" else 2,
                "total_score": 1.0 if cslug == "riviera" else 2.0,
                "similarity_weight": 0.6 if cslug == "riviera" else 0.4,
                "weight_method": "manual",
                "config_name": "baseline",
            })
    return pd.DataFrame(rows)


def make_history(specs, n_holes=6):
    rows = []
    tid = 0
    for pid, outcome in specs:
        for h in range(1, n_holes + 1):
            for cslug in ("riviera", "tpc"):
                rows.append({
                    "player_id": pid, "tournament_id": f"T{tid}", "year": 2023,
                    "round": 1, "hole_number": h, "course_slug": cslug,
                    "hole_id_v25": f"{cslug}:{h}", "par": 4,
                    "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    return pd.DataFrame(rows)


def _outputs():
    sim = make_similar_holes(6)
    hist = make_history([("pA", 2), ("pB", 1)], n_holes=6)
    outs = assemble_field_outputs(hist, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=LOOSE)
    return sim, hist, outs


# --------------------------------------------------------------------------- #
# assemble_field_outputs
# --------------------------------------------------------------------------- #
def test_assemble_shapes_and_player_id():
    _, _, outs = _outputs()
    assert outs["rankings"]["player_id"].tolist() == ["pA", "pB"]
    # 2 players x 6 target holes.
    assert len(outs["hole_details"]) == 2 * 6
    assert "player_id" in outs["diagnostics"].columns
    assert set(outs["diagnostics"]["player_id"]) == {"pA", "pB"}


def test_run_id_is_unique_and_prefixed():
    a, b = make_run_id(), make_run_id()
    assert a != b and a.startswith("run-")


# --------------------------------------------------------------------------- #
# export / load round-trip
# --------------------------------------------------------------------------- #
def test_export_and_load_roundtrip(tmp_path):
    _, hist, outs = _outputs()
    run = export_advantage_run(
        outs["rankings"], hole_details=outs["hole_details"], diagnostics=outs["diagnostics"],
        target_course_slug=TARGET_COURSE, config_name="baseline",
        predict_season=PREDICT_SEASON, params=LOOSE, root=tmp_path,
        input_counts={"history_rows": len(hist)},
    )
    for fname in (ax.RANKINGS_FILE, ax.HOLE_DETAILS_FILE, ax.DIAGNOSTICS_FILE,
                  ax.PARAMETERS_FILE, ax.MANIFEST_FILE):
        assert (run.run_dir / fname).exists()
    assert not (run.run_dir / ax.BACKTEST_SUMMARY_FILE).exists()  # not supplied

    loaded = load_advantage_run(run.run_dir)
    # CSV round-trips don't preserve an all-null object column's dtype/None-vs-NaN.
    pd.testing.assert_frame_equal(
        loaded.rankings.fillna(""), outs["rankings"].fillna(""), check_dtype=False
    )
    assert loaded.manifest["target_course_slug"] == TARGET_COURSE
    assert loaded.manifest["output_counts"]["n_players"] == 2
    assert loaded.manifest["input_counts"]["history_rows"] == len(hist)
    assert loaded.parameters["min_holes_covered"] == 1
    assert loaded.parameters["aggregate"] == "sum"
    assert loaded.backtest_summary is None


def test_manifest_has_reproducibility_metadata(tmp_path):
    _, _, outs = _outputs()
    run = export_advantage_run(
        outs["rankings"], target_course_slug=TARGET_COURSE, config_name="baseline",
        predict_season=PREDICT_SEASON, params=LOOSE, root=tmp_path,
    )
    m = json.loads((run.run_dir / ax.MANIFEST_FILE).read_text(encoding="utf-8"))
    for key in ("run_id", "created_at", "model_version", "aggregate",
                "files", "output_counts", "has_backtest"):
        assert key in m
    assert m["has_backtest"] is False
    assert m["model_version"].startswith("player_course_advantage")


# --------------------------------------------------------------------------- #
# optional backtest slot (forward-compat with #35)
# --------------------------------------------------------------------------- #
def test_backtest_summary_written_when_supplied(tmp_path):
    _, _, outs = _outputs()
    fake_backtest = pd.DataFrame({"metric": ["spearman"], "value": [0.42]})
    run = export_advantage_run(
        outs["rankings"], backtest_summary=fake_backtest,
        target_course_slug=TARGET_COURSE, config_name="baseline",
        predict_season=PREDICT_SEASON, params=LOOSE, root=tmp_path,
    )
    assert (run.run_dir / ax.BACKTEST_SUMMARY_FILE).exists()
    loaded = load_advantage_run(run.run_dir)
    assert loaded.manifest["has_backtest"] is True
    pd.testing.assert_frame_equal(loaded.backtest_summary, fake_backtest)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_geometry_columns_rejected(tmp_path):
    _, _, outs = _outputs()
    bad_details = outs["hole_details"].copy()
    bad_details["candidate_point_cloud_xyz"] = "1,2,3"
    with pytest.raises(ArtifactExportError):
        export_advantage_run(
            outs["rankings"], hole_details=bad_details,
            target_course_slug=TARGET_COURSE, config_name="baseline",
            predict_season=PREDICT_SEASON, params=LOOSE, root=tmp_path,
        )


def test_load_missing_manifest_raises(tmp_path):
    with pytest.raises(ArtifactExportError):
        load_advantage_run(tmp_path / "does_not_exist")


def test_export_does_not_mutate_frames(tmp_path):
    _, _, outs = _outputs()
    before = {k: v.copy() for k, v in outs.items()}
    export_advantage_run(
        outs["rankings"], hole_details=outs["hole_details"], diagnostics=outs["diagnostics"],
        target_course_slug=TARGET_COURSE, config_name="baseline",
        predict_season=PREDICT_SEASON, params=LOOSE, root=tmp_path,
    )
    for k in outs:
        pd.testing.assert_frame_equal(outs[k], before[k])
