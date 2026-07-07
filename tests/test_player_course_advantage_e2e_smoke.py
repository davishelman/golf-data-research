"""End-to-end synthetic smoke test for player-course advantage (issue #45).

Proves the whole pipeline runs offline from fake v2.5 result CSVs all the way to
an exported/reloaded artifact:

    fake v2.5 CSV -> load_similar_hole_sets -> score -> batch ranking
                  -> diagnostics -> artifact export -> artifact load

No real ``courses/`` outputs, no network, no Hugging Face token, no Streamlit,
and no raw point-cloud geometry — only CSV columns of ids/scores.
"""

from __future__ import annotations

import pandas as pd

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    assemble_field_outputs,
    export_advantage_run,
    load_advantage_run,
    load_similar_hole_sets,
)
from pipeline.modeling.pointcloud.export_similarity import RESULTS_FILENAME

TARGET_COURSE = "augusta_national"
PREDICT_SEASON = 2024
N_HOLES = 4
CANDIDATE_COURSE = "riviera"


def _write_fake_v25(root, n_holes=N_HOLES, n_candidates=4):
    """A minimal but realistic v2.5 result CSV (ids + scores + component columns)."""
    rows = []
    for h in range(1, n_holes + 1):
        for c in range(1, n_candidates + 1):
            rows.append({
                "model_version": "v2_5_chamfer_v1",
                "config_name": "baseline",
                "target_hole_id": f"{TARGET_COURSE}:{h}",
                "candidate_hole_id": f"{CANDIDATE_COURSE}:{c}",
                "rank": c,
                "total_score": float(c),
                # component *scores* (not geometry) — loader carries these along.
                "fairway_score": 0.1 * c, "green_score": 0.2 * c,
            })
    cfg = root / "pointcloud_similarity" / "baseline"
    cfg.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg / RESULTS_FILENAME, index=False)


def _fake_history(players, n_holes=N_HOLES, n_candidates=3):
    """Each player has one occurrence on each selected candidate hole per season."""
    rows = []
    tid = 0
    for pid, outcome in players.items():
        for c in range(1, n_candidates + 1):
            rows.append({
                "player_id": pid, "player_name": pid.upper(),
                "tournament_id": f"T{tid}", "year": 2023, "round": 1,
                "hole_number": c, "course_slug": CANDIDATE_COURSE,
                "hole_id_v25": f"{CANDIDATE_COURSE}:{c}", "par": 4,
                "player_score": 4 - outcome, "field_avg_score": 4,
            })
            tid += 1
    return pd.DataFrame(rows)


def test_end_to_end_synthetic_pipeline(tmp_path):
    # 1. fake v2.5 outputs on disk -> loader
    v25_root = tmp_path / "v25"
    _write_fake_v25(v25_root)
    sim = load_similar_hole_sets(v25_root, TARGET_COURSE, top_n=3)
    assert len(sim) == N_HOLES * 3
    # loader carried the component score column but no geometry.
    assert "fairway_score" in sim.columns

    # 2. fake history + field -> ranking + hole details + diagnostics
    # Every candidate riviera:1..3 is shared across all target holes, so each
    # target hole sees 3 occurrences (>= default min_occurrences_per_hole).
    history = _fake_history({"pA": 2, "pB": 1, "pC": -1})
    params = AdvantageParams(min_holes_covered=1)  # keep default min_occurrences (3)
    outs = assemble_field_outputs(
        history, sim, ["pC", "pB", "pA"], TARGET_COURSE, PREDICT_SEASON, params=params
    )
    ranking = outs["rankings"]

    # deterministic order: best advantage first, worst last.
    assert ranking["player_id"].tolist() == ["pA", "pB", "pC"]
    assert ranking["rank"].tolist() == [1, 2, 3]
    assert ranking["low_coverage"].tolist() == [False, False, False]

    # coverage diagnostics present and populated.
    for col in ("holes_covered", "total_raw_occurrences", "total_weighted_occurrences"):
        assert col in ranking.columns
    assert (ranking["holes_covered"] == N_HOLES).all()
    assert len(outs["hole_details"]) == 3 * N_HOLES

    # 3. export -> load round-trip
    run = export_advantage_run(
        ranking, hole_details=outs["hole_details"], diagnostics=outs["diagnostics"],
        target_course_slug=TARGET_COURSE, config_name="baseline",
        predict_season=PREDICT_SEASON, params=params,
        root=tmp_path / "artifacts",
        input_counts={"history_rows": len(history), "similar_hole_rows": len(sim)},
    )
    loaded = load_advantage_run(run.run_dir)

    # CSV reloads an all-null 'reason' column as float NaN; normalize before compare.
    pd.testing.assert_frame_equal(
        loaded.rankings.fillna(""), ranking.fillna(""), check_dtype=False
    )
    assert loaded.manifest["output_counts"]["n_players"] == 3
    assert loaded.manifest["output_counts"]["n_covered"] == 3
    assert loaded.manifest["input_counts"]["similar_hole_rows"] == len(sim)
    assert loaded.manifest["has_backtest"] is False

    # No raw geometry columns anywhere in the exported artifacts.
    for frame in (loaded.rankings, loaded.hole_details, loaded.diagnostics):
        assert not any(
            marker in str(c).lower()
            for c in frame.columns
            for marker in ("point_cloud", "xyz", "mesh", "vertices")
        )


def test_pipeline_is_deterministic(tmp_path):
    v25_root = tmp_path / "v25"
    _write_fake_v25(v25_root)
    sim = load_similar_hole_sets(v25_root, TARGET_COURSE, top_n=3)
    history = _fake_history({"pA": 2, "pB": 1})
    params = AdvantageParams(min_holes_covered=1)
    a = assemble_field_outputs(history, sim, ["pA", "pB"], TARGET_COURSE, PREDICT_SEASON, params=params)
    b = assemble_field_outputs(history, sim, ["pB", "pA"], TARGET_COURSE, PREDICT_SEASON, params=params)
    pd.testing.assert_frame_equal(a["rankings"], b["rankings"])
    pd.testing.assert_frame_equal(a["hole_details"], b["hole_details"])
