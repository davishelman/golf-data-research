"""Tests for v2.5 batch + single-target export, using a fake in-memory loader.

No real course artifacts are required: a :class:`FakeLoader` supplies metadata
and surface points directly, so these tests stay fast and deterministic.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.pointcloud.config import config_from_dict
from pipeline.modeling.pointcloud.export_similarity import (
    FILTER_SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    SIMILARITY_RESULTS_COLUMNS,
    rank_similar_holes,
    run_batch_export,
)
from pipeline.modeling.pointcloud.schemas import (
    HoleMetadata,
    SurfacePoint,
    make_pc_hole_id,
)
from tests.test_pointcloud_config import _valid_payload


# --------------------------------------------------------------------------- #
# Fixtures / fakes
# --------------------------------------------------------------------------- #

def _config():
    return config_from_dict(_valid_payload())


def _all_surface_points(hole_id, *, dx=0.0, dy=0.0, dz=0.0, n=4):
    """All-surface n*n grid for one hole, optionally translated."""
    pts = []
    for surface in ("fairway", "green", "bunker", "water", "tee"):
        for i in range(n):
            for j in range(n):
                pts.append(SurfacePoint(
                    hole_id=hole_id, surface=surface,
                    x_lateral_m=float(i) + dx, y_down_hole_m=float(j) + dy,
                    z_relative_m=dz,
                ))
    return pts


def _meta(slug, number=1, par=4, yards=440.0):
    return HoleMetadata(
        hole_id=make_pc_hole_id(slug, number), course_slug=slug, hole_number=number,
        par=par, yards=yards,
        has_tee=True, has_green=True, has_fairway=True, has_bunker=True, has_water=True,
    )


class FakeLoader:
    """In-memory :class:`PointCloudArtifactLoader` for tests."""

    def __init__(self, metadata: dict[str, HoleMetadata], points: dict[str, list[SurfacePoint]]):
        self._metadata = metadata
        self._points = points
        self.load_points_calls: list[str] = []

    def load_metadata(self) -> dict[str, HoleMetadata]:
        return dict(self._metadata)

    def load_points(self, hole_id: str) -> list[SurfacePoint]:
        self.load_points_calls.append(hole_id)
        return list(self._points.get(hole_id, []))


def _field_loader():
    """A small field: 1 target + 3 eligible + 2 that fail filtering."""
    target = _meta("aaa_course", 1, par=4, yards=440.0)
    near = _meta("bbb_course", 1, par=4, yards=445.0)        # eligible (Δ5)
    mid = _meta("ccc_course", 1, par=4, yards=450.0)         # eligible (Δ10)
    far_id = _meta("ddd_course", 1, par=4, yards=600.0)      # YARDAGE_TOO_DIFFERENT
    wrong_par = _meta("eee_course", 1, par=5, yards=440.0)   # DIFFERENT_PAR

    metadata = {h.hole_id: h for h in (target, near, mid, far_id, wrong_par)}
    points = {
        target.hole_id: _all_surface_points(target.hole_id),
        near.hole_id: _all_surface_points(near.hole_id, dx=1.0),   # closest geometry
        mid.hole_id: _all_surface_points(mid.hole_id, dx=5.0),     # further geometry
        far_id.hole_id: _all_surface_points(far_id.hole_id),
        wrong_par.hole_id: _all_surface_points(wrong_par.hole_id),
    }
    return FakeLoader(metadata, points), target.hole_id


# --------------------------------------------------------------------------- #
# 1. writes a result row for passing candidates
# --------------------------------------------------------------------------- #

def test_batch_writes_rows_for_passing_candidates(tmp_path):
    loader, _ = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25)

    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    assert list(results.columns) == list(SIMILARITY_RESULTS_COLUMNS)
    assert len(results) > 0
    # Only the same-par, in-window candidates appear (bbb, ccc) for each target.
    cands = set(results["candidate_hole_id"])
    assert "ddd_course:1" not in cands  # yardage too different
    assert "eee_course:1" not in cands  # different par
    # filter_reason on written rows is always PASS (they were scored).
    assert set(results["filter_reason"]) == {"PASS"}


def test_batch_writes_all_three_artifacts(tmp_path):
    loader, _ = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25)
    assert (tmp_path / RESULTS_FILENAME).exists()
    assert (tmp_path / FILTER_SUMMARY_FILENAME).exists()
    assert (tmp_path / MANIFEST_FILENAME).exists()


# --------------------------------------------------------------------------- #
# 2. self-comparisons skipped by default
# --------------------------------------------------------------------------- #

def test_self_comparisons_skipped_by_default(tmp_path):
    loader, _ = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25)
    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    self_rows = results[results["target_hole_id"] == results["candidate_hole_id"]]
    assert self_rows.empty


def test_include_self_adds_self_rows(tmp_path):
    loader, _ = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25, include_self=True)
    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    self_rows = results[results["target_hole_id"] == results["candidate_hole_id"]]
    assert not self_rows.empty
    # A hole compared to itself is maximally similar -> total_score 0, rank 1.
    for _, row in self_rows.iterrows():
        assert row["total_score"] == pytest.approx(0.0, abs=1e-9)
        assert row["rank"] == 1


# --------------------------------------------------------------------------- #
# 3. top_n respected per target hole
# --------------------------------------------------------------------------- #

def test_top_n_respected_per_target(tmp_path):
    loader, _ = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=1)
    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    per_target = results.groupby("target_hole_id").size()
    assert (per_target <= 1).all()
    # Each target with >=1 eligible candidate keeps exactly its single best.
    assert (per_target == 1).all()


# --------------------------------------------------------------------------- #
# 4. deterministic sort order
# --------------------------------------------------------------------------- #

def test_results_sorted_total_score_then_candidate_id(tmp_path):
    loader, target_id = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25)
    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    sub = results[results["target_hole_id"] == target_id].reset_index(drop=True)
    # Ranks are 1..N in order.
    assert list(sub["rank"]) == list(range(1, len(sub) + 1))
    # Sorted by (total_score asc, candidate_hole_id asc).
    expected = sub.sort_values(["total_score", "candidate_hole_id"]).reset_index(drop=True)
    assert list(sub["candidate_hole_id"]) == list(expected["candidate_hole_id"])


def test_ties_broken_by_candidate_id(tmp_path):
    # Two candidates with identical geometry + yardage -> identical total_score.
    target = _meta("mmm_course", 1, yards=440.0)
    zzz = _meta("zzz_course", 1, yards=440.0)
    aaa = _meta("aaa_course", 1, yards=440.0)
    metadata = {h.hole_id: h for h in (target, zzz, aaa)}
    pts = {h.hole_id: _all_surface_points(h.hole_id) for h in (target, zzz, aaa)}
    loader = FakeLoader(metadata, pts)

    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25)
    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    sub = results[results["target_hole_id"] == target.hole_id].reset_index(drop=True)
    assert sub["total_score"].nunique() == 1  # genuine tie
    assert list(sub["candidate_hole_id"]) == ["aaa_course:1", "zzz_course:1"]


# --------------------------------------------------------------------------- #
# 5. filter summary includes failed candidate reasons
# --------------------------------------------------------------------------- #

def test_filter_summary_includes_failure_reasons(tmp_path):
    loader, _ = _field_loader()
    run_batch_export(_config(), loader, output_dir=tmp_path, top_n=25)
    summary = pd.read_csv(tmp_path / FILTER_SUMMARY_FILENAME)
    assert list(summary.columns) == ["filter_reason", "count"]
    reasons = dict(zip(summary["filter_reason"], summary["count"]))
    assert "PASS" in reasons
    assert "DIFFERENT_PAR" in reasons
    assert "YARDAGE_TOO_DIFFERENT" in reasons
    assert all(c > 0 for c in reasons.values())


# --------------------------------------------------------------------------- #
# 6. manifest contents
# --------------------------------------------------------------------------- #

def test_manifest_contents(tmp_path):
    loader, _ = _field_loader()
    config = _config()
    returned = run_batch_export(config, loader, output_dir=tmp_path, top_n=25,
                                config_path=tmp_path / "fake_config.yaml")
    on_disk = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == returned

    assert on_disk["config_name"] == config.config_name
    assert on_disk["config_hash"] == config.config_hash
    assert on_disk["model_version"] == config.model_version
    assert on_disk["total_targets"] == 5
    assert on_disk["top_n"] == 25
    assert on_disk["include_self"] is False
    # scored pairs == number of PASS rows in the filter summary.
    assert on_disk["total_scored_pairs"] == on_disk["filter_reason_counts"]["PASS"]
    results = pd.read_csv(tmp_path / RESULTS_FILENAME)
    assert on_disk["total_written_rows"] == len(results)


def test_limit_targets_and_overwrite(tmp_path):
    loader, _ = _field_loader()
    config = _config()
    m1 = run_batch_export(config, loader, output_dir=tmp_path, top_n=25, limit_targets=2)
    assert m1["total_targets"] == 2

    # Re-running without overwrite should refuse.
    with pytest.raises(FileExistsError):
        run_batch_export(config, loader, output_dir=tmp_path, top_n=25)

    # With overwrite it succeeds and replaces.
    m2 = run_batch_export(config, loader, output_dir=tmp_path, top_n=25, overwrite=True)
    assert m2["total_targets"] == 5


# --------------------------------------------------------------------------- #
# 7. single-target mode still works
# --------------------------------------------------------------------------- #

def test_single_target_mode_still_works():
    loader, target_id = _field_loader()
    results = rank_similar_holes(target_id, _config(), loader, top_n=10,
                                 exclude_same_course=True)
    assert len(results) >= 1
    cands = [r.candidate_hole_id for r in results]
    assert target_id not in cands  # never returns self
    # Deterministic, ascending total_score.
    scores = [r.total_score for r in results]
    assert scores == sorted(scores)
