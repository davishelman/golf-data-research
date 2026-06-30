"""Tests for v2.5 point-cloud similarity validation reports."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.pointcloud.export_similarity import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    SIMILARITY_RESULTS_COLUMNS,
)
from pipeline.modeling.pointcloud.validate_similarity import (
    OVERLAP_COLUMNS,
    RANK_COMPARISON_FILENAME,
    TOP_MATCHES_COLUMNS,
    VALIDATION_MANIFEST_FILENAME,
    config_overlap_summary,
    load_result_dir,
    rank_comparison,
    run_validation,
    sanitize_hole_id,
    top_matches_for_target,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _result_row(target, candidate, rank, total_score, config_name):
    """A single similarity_results.csv row with all required columns."""
    return {
        "model_version": "v2_5_chamfer_v1",
        "config_name": config_name,
        "config_hash": "deadbeef",
        "target_hole_id": target,
        "candidate_hole_id": candidate,
        "rank": rank,
        "total_score": total_score,
        "fairway_score": total_score * 0.5,
        "green_score": total_score * 0.3,
        "bunker_score": total_score * 0.2,
        "water_score": None,
        "tee_score": 1.0,
        "yardage_penalty": 0.1,
        "elevation_penalty": 0.0,
        "missing_surface_penalty": 0.0,
        "filter_reason": "PASS",
    }


def _write_result_dir(base, config_name, rows):
    """Write a fake batch result dir (similarity_results.csv + manifest.json)."""
    d = base / config_name
    d.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=list(SIMILARITY_RESULTS_COLUMNS))
    frame.to_csv(d / RESULTS_FILENAME, index=False)
    manifest = {
        "model_version": "v2_5_chamfer_v1", "config_name": config_name,
        "config_hash": "deadbeef", "created_at": "2026-01-01T00:00:00+00:00",
        "top_n": 25, "total_written_rows": len(rows),
    }
    (d / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _two_config_dirs(base):
    """baseline + hazard_heavy with partly-overlapping candidate sets for T."""
    target = "augusta_national:13"
    baseline = [
        _result_row(target, "c1:1", 1, 10.0, "baseline"),
        _result_row(target, "c2:1", 2, 20.0, "baseline"),
        _result_row(target, "c3:1", 3, 30.0, "baseline"),
        # a different target, to be filtered out:
        _result_row("augusta_national:1", "x9:1", 1, 5.0, "baseline"),
    ]
    hazard = [
        _result_row(target, "c2:1", 1, 11.0, "hazard_heavy"),  # shared
        _result_row(target, "c3:1", 2, 12.0, "hazard_heavy"),  # shared
        _result_row(target, "c4:1", 3, 13.0, "hazard_heavy"),  # unique
    ]
    _write_result_dir(base, "baseline", baseline)
    _write_result_dir(base, "hazard_heavy", hazard)
    return target


# --------------------------------------------------------------------------- #
# Sanitize + load
# --------------------------------------------------------------------------- #

def test_sanitize_hole_id():
    assert sanitize_hole_id("augusta_national:13") == "augusta_national__13"


def test_load_result_dir(tmp_path):
    _two_config_dirs(tmp_path)
    df, manifest, name = load_result_dir(tmp_path / "baseline")
    assert name == "baseline"
    assert manifest["config_name"] == "baseline"
    assert list(df.columns) == list(SIMILARITY_RESULTS_COLUMNS)


def test_missing_results_csv_raises_clear_error(tmp_path):
    (tmp_path / "empty_cfg").mkdir()
    with pytest.raises(FileNotFoundError, match="similarity_results.csv"):
        load_result_dir(tmp_path / "empty_cfg")


# --------------------------------------------------------------------------- #
# Filtering + top-N
# --------------------------------------------------------------------------- #

def test_top_matches_filters_to_target_and_limits(tmp_path):
    target = _two_config_dirs(tmp_path)
    df, _, name = load_result_dir(tmp_path / "baseline")
    tm = top_matches_for_target(df, target, top_n=2, config_name=name)
    assert list(tm.columns) == list(TOP_MATCHES_COLUMNS)
    assert len(tm) == 2  # top_n respected
    assert set(tm["target_hole_id"]) == {target}  # other target filtered out
    assert list(tm["candidate_hole_id"]) == ["c1:1", "c2:1"]  # sorted by score
    assert list(tm["rank"]) == [1, 2]


def test_top_matches_empty_for_unknown_target(tmp_path):
    _two_config_dirs(tmp_path)
    df, _, name = load_result_dir(tmp_path / "baseline")
    tm = top_matches_for_target(df, "nonexistent:1", top_n=10, config_name=name)
    assert tm.empty
    assert list(tm.columns) == list(TOP_MATCHES_COLUMNS)


# --------------------------------------------------------------------------- #
# Overlap
# --------------------------------------------------------------------------- #

def test_config_overlap_summary(tmp_path):
    target = _two_config_dirs(tmp_path)
    dfs = {}
    for n in ("baseline", "hazard_heavy"):
        df, _, name = load_result_dir(tmp_path / n)
        dfs[name] = top_matches_for_target(df, target, 10, name)

    overlap = config_overlap_summary(dfs, top_n=10)
    assert list(overlap.columns) == list(OVERLAP_COLUMNS)
    assert len(overlap) == 1  # one pair
    row = overlap.iloc[0]
    assert row["config_a"] == "baseline"
    assert row["config_b"] == "hazard_heavy"
    # baseline {c1,c2,c3} vs hazard {c2,c3,c4}: overlap 2, union 4 -> 0.5
    assert row["overlap_count"] == 2
    assert row["union_count"] == 4
    assert row["jaccard_similarity"] == pytest.approx(0.5)
    assert row["shared_candidates"] == "c2:1|c3:1"


# --------------------------------------------------------------------------- #
# Rank comparison
# --------------------------------------------------------------------------- #

def test_rank_comparison_shared_and_unique(tmp_path):
    target = _two_config_dirs(tmp_path)
    dfs = {}
    for n in ("baseline", "hazard_heavy"):
        df, _, name = load_result_dir(tmp_path / n)
        dfs[name] = top_matches_for_target(df, target, 10, name)

    rc = rank_comparison(dfs)
    assert "rank_baseline" in rc.columns
    assert "rank_hazard_heavy" in rc.columns
    assert "total_score_baseline" in rc.columns
    expected_cols = {
        "candidate_hole_id", "rank_baseline", "rank_hazard_heavy",
        "total_score_baseline", "total_score_hazard_heavy",
        "configs_present_count", "best_rank", "worst_rank", "rank_spread",
    }
    assert set(rc.columns) == expected_cols

    by_cand = rc.set_index("candidate_hole_id")
    # Shared candidate present in both configs.
    assert by_cand.loc["c2:1", "configs_present_count"] == 2
    assert by_cand.loc["c2:1", "best_rank"] == 1   # rank 2 baseline, rank 1 hazard
    assert by_cand.loc["c2:1", "worst_rank"] == 2
    assert by_cand.loc["c2:1", "rank_spread"] == 1
    # Unique candidate present in one config only.
    assert by_cand.loc["c1:1", "configs_present_count"] == 1
    assert by_cand.loc["c4:1", "configs_present_count"] == 1
    assert pd.isna(by_cand.loc["c1:1", "rank_hazard_heavy"])

    # Deterministic ordering: most-shared first, then best_rank, then id.
    assert rc.iloc[0]["candidate_hole_id"] in {"c2:1", "c3:1"}
    assert list(rc["configs_present_count"]) == sorted(
        rc["configs_present_count"], reverse=True
    )


# --------------------------------------------------------------------------- #
# Full run
# --------------------------------------------------------------------------- #

def test_run_validation_writes_outputs_and_manifest(tmp_path):
    target = _two_config_dirs(tmp_path)
    out = tmp_path / "_validation_out"
    manifest = run_validation(
        target, result_dirs=[tmp_path / "baseline", tmp_path / "hazard_heavy"],
        top_n=10, output_dir=out,
    )
    # Files exist.
    assert (out / "top_matches_baseline.csv").exists()
    assert (out / "top_matches_hazard_heavy.csv").exists()
    assert (out / "config_overlap_summary.csv").exists()
    assert (out / RANK_COMPARISON_FILENAME).exists()
    assert (out / VALIDATION_MANIFEST_FILENAME).exists()
    assert (out / "top_matches.md").exists()

    # Manifest contents.
    on_disk = json.loads((out / VALIDATION_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert on_disk["target_hole_id"] == target
    assert on_disk["sanitized_target_hole_id"] == "augusta_national__13"
    assert on_disk["configs"] == ["baseline", "hazard_heavy"]
    assert on_disk["top_n"] == 10
    assert on_disk["match_counts"] == {"baseline": 3, "hazard_heavy": 3}
    assert "baseline" in on_disk["source_manifests"]


def test_run_validation_default_sanitized_path(tmp_path):
    target = _two_config_dirs(tmp_path)
    manifest = run_validation(
        target, result_dirs=[tmp_path / "baseline"],
        top_n=10, courses_root=tmp_path, write_markdown=False,
    )
    # Default path uses the sanitized id under _validation/.
    assert manifest["output_dir"].replace("\\", "/").endswith(
        "_index/pointcloud_similarity/_validation/augusta_national__13"
    )


def test_run_validation_empty_target_is_clean(tmp_path):
    _two_config_dirs(tmp_path)
    out = tmp_path / "_val_empty"
    manifest = run_validation(
        "no_such_course:7", result_dirs=[tmp_path / "baseline"],
        top_n=10, output_dir=out,
    )
    assert manifest["match_counts"] == {"baseline": 0}
    tm = pd.read_csv(out / "top_matches_baseline.csv")
    assert tm.empty
    rc = pd.read_csv(out / RANK_COMPARISON_FILENAME)
    assert rc.empty


def test_run_validation_requires_exactly_one_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        run_validation("augusta_national:13", output_dir=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        run_validation("augusta_national:13", configs=["baseline"],
                       result_dirs=[tmp_path / "baseline"], output_dir=tmp_path)


def test_run_validation_overwrite_guard(tmp_path):
    target = _two_config_dirs(tmp_path)
    out = tmp_path / "_val_guard"
    run_validation(target, result_dirs=[tmp_path / "baseline"], output_dir=out)
    with pytest.raises(FileExistsError):
        run_validation(target, result_dirs=[tmp_path / "baseline"], output_dir=out)
    # Overwrite succeeds.
    run_validation(target, result_dirs=[tmp_path / "baseline"], output_dir=out,
                   overwrite=True)
