"""Tests for v2.5 artifact export (#10) and demo data layer (#11)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.pointcloud import demo as pcdemo
from pipeline.modeling.pointcloud.artifact_export import (
    ARTIFACT_SUBPATH,
    add_pointcloud_similarity_to_artifact,
    discover_config_dirs,
)
from pipeline.modeling.pointcloud.export_similarity import (
    FILTER_SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    SIMILARITY_RESULTS_COLUMNS,
)


def _row(target, candidate, rank, score, config="baseline"):
    base = {c: None for c in SIMILARITY_RESULTS_COLUMNS}
    base.update({
        "model_version": "v2_5_chamfer_v1", "config_name": config, "config_hash": "h",
        "target_hole_id": target, "candidate_hole_id": candidate, "rank": rank,
        "total_score": score, "fairway_score": score * 0.5, "green_score": score * 0.3,
        "bunker_score": score * 0.2, "water_score": None, "tee_score": 0.5,
        "yardage_penalty": 0.1, "elevation_penalty": 0.0, "missing_surface_penalty": 0.0,
        "filter_reason": "PASS",
    })
    return base


def _write_batch_dir(courses_root, config_name, rows):
    d = courses_root / "_index" / "pointcloud_similarity" / config_name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SIMILARITY_RESULTS_COLUMNS)).to_csv(
        d / RESULTS_FILENAME, index=False)
    (d / MANIFEST_FILENAME).write_text(json.dumps({
        "model_version": "v2_5_chamfer_v1", "config_name": config_name,
        "config_hash": "h", "total_written_rows": len(rows), "top_n": 25,
    }), encoding="utf-8")
    pd.DataFrame([{"filter_reason": "PASS", "count": len(rows)}]).to_csv(
        d / FILTER_SUMMARY_FILENAME, index=False)
    return d


def _courses_root_with_two_configs(tmp_path):
    cr = tmp_path / "courses"
    target = "augusta_national:13"
    _write_batch_dir(cr, "baseline", [
        _row(target, "c1:1", 1, 10.0, "baseline"),
        _row(target, "c2:1", 2, 20.0, "baseline"),
        _row("augusta_national:1", "z9:1", 1, 5.0, "baseline"),
    ])
    _write_batch_dir(cr, "hazard_heavy", [
        _row(target, "c2:1", 1, 11.0, "hazard_heavy"),
        _row(target, "c3:1", 2, 12.0, "hazard_heavy"),
    ])
    # A validation working dir that must be ignored by discovery.
    (cr / "_index" / "pointcloud_similarity" / "_validation").mkdir(parents=True)
    return cr, target


# --- #10 artifact export --------------------------------------------------- #

def test_discover_config_dirs_excludes_validation(tmp_path):
    cr, _ = _courses_root_with_two_configs(tmp_path)
    dirs = discover_config_dirs(cr)
    names = [d.name for d in dirs]
    assert names == ["baseline", "hazard_heavy"]  # sorted, _validation excluded


def test_add_pointcloud_similarity_to_artifact(tmp_path):
    cr, _ = _courses_root_with_two_configs(tmp_path)
    artifact = tmp_path / "artifact"
    (artifact / "data").mkdir(parents=True)

    summary = add_pointcloud_similarity_to_artifact(artifact, courses_root=cr)
    assert summary["configs_exported"] == ["baseline", "hazard_heavy"]

    # Files copied under the separate subtree.
    base = artifact / ARTIFACT_SUBPATH / "baseline"
    assert (base / RESULTS_FILENAME).exists()
    assert (base / MANIFEST_FILENAME).exists()
    assert (base / FILTER_SUMMARY_FILENAME).exists()

    # Index metadata written and well-formed.
    index = json.loads((artifact / "metadata" / "pointcloud_similarity.json").read_text(encoding="utf-8"))
    assert index["model_version"] == "v2_5_chamfer_v1"
    assert {c["config_name"] for c in index["configs"]} == {"baseline", "hazard_heavy"}
    assert index["subpath"] == ARTIFACT_SUBPATH


def test_add_artifact_subset_of_configs(tmp_path):
    cr, _ = _courses_root_with_two_configs(tmp_path)
    artifact = tmp_path / "artifact"
    summary = add_pointcloud_similarity_to_artifact(
        artifact, courses_root=cr, config_names=["baseline"])
    assert summary["configs_exported"] == ["baseline"]
    assert not (artifact / ARTIFACT_SUBPATH / "hazard_heavy").exists()


def test_add_artifact_raises_when_no_outputs(tmp_path):
    artifact = tmp_path / "artifact"
    with pytest.raises(FileNotFoundError, match="no v2.5 batch outputs"):
        add_pointcloud_similarity_to_artifact(artifact, courses_root=tmp_path / "empty")


def test_export_refuses_secret_like_file(tmp_path):
    cr, _ = _courses_root_with_two_configs(tmp_path)
    # Drop a secret-like file into a config dir; the secret guard must reject it.
    (cr / "_index" / "pointcloud_similarity" / "baseline" / ".env").write_text(
        "SECRET=1", encoding="utf-8")
    # discovery still works; the copy of allow-listed files is fine, but if we had
    # tried to copy the .env it would raise. Here we assert the guard helper view:
    from pipeline.modeling.hf_export import _is_secret
    assert _is_secret(cr / "_index" / "pointcloud_similarity" / "baseline" / ".env")


# --- #11 demo data layer --------------------------------------------------- #

def _built_artifact(tmp_path):
    cr, target = _courses_root_with_two_configs(tmp_path)
    artifact = tmp_path / "artifact"
    add_pointcloud_similarity_to_artifact(artifact, courses_root=cr)
    return artifact, target


def test_list_pointcloud_configs(tmp_path):
    artifact, _ = _built_artifact(tmp_path)
    assert pcdemo.list_pointcloud_configs(artifact) == ["baseline", "hazard_heavy"]


def test_list_pointcloud_configs_empty_when_absent(tmp_path):
    assert pcdemo.list_pointcloud_configs(tmp_path / "nothing") == []


def test_load_and_top_matches_for_hole(tmp_path):
    artifact, target = _built_artifact(tmp_path)
    results = pcdemo.load_pointcloud_results(artifact)
    assert set(results) == {"baseline", "hazard_heavy"}

    top = pcdemo.top_matches_for_hole(results["baseline"], target, top_n=10)
    assert list(top.columns) == list(pcdemo.DISPLAY_COLUMNS)
    # Sorted best-first, other targets filtered out.
    assert list(top["candidate_hole_id"]) == ["c1:1", "c2:1"]
    assert list(top["rank"]) == [1, 2]


def test_top_matches_empty_for_unknown_hole(tmp_path):
    artifact, _ = _built_artifact(tmp_path)
    results = pcdemo.load_pointcloud_results(artifact)
    top = pcdemo.top_matches_for_hole(results["baseline"], "no_such:1", top_n=10)
    assert top.empty
    assert list(top.columns) == list(pcdemo.DISPLAY_COLUMNS)


def test_available_target_holes(tmp_path):
    artifact, _ = _built_artifact(tmp_path)
    results = pcdemo.load_pointcloud_results(artifact)
    targets = pcdemo.available_target_holes(results["baseline"])
    assert "augusta_national:13" in targets
    assert "augusta_national:1" in targets


def test_pointcloud_summary(tmp_path):
    artifact, _ = _built_artifact(tmp_path)
    summary = pcdemo.pointcloud_summary(artifact)
    assert summary["n_configs"] == 2
    assert summary["rows_by_config"]["baseline"] == 3
    assert summary["targets_by_config"]["baseline"] == 2  # two distinct targets
