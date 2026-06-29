"""Tests for v2.5 Chamfer distance and surface-aware scoring."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.modeling.pointcloud.chamfer import (
    EmptySurfaceError,
    symmetric_chamfer_distance,
)
from pipeline.modeling.pointcloud.config import config_from_dict
from pipeline.modeling.pointcloud.schemas import HoleMetadata, SurfacePoint, make_pc_hole_id
from pipeline.modeling.pointcloud.score import score_pair
from tests.test_pointcloud_config import _valid_payload


def _config():
    return config_from_dict(_valid_payload())


def _grid(surface, hole_id, *, dx=0.0, dy=0.0, dz=0.0, n=5):
    """A small n*n grid of SurfacePoints, optionally translated by (dx, dy, dz)."""
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append(SurfacePoint(
                hole_id=hole_id, surface=surface,
                x_lateral_m=float(i) + dx,
                y_down_hole_m=float(j) + dy,
                z_relative_m=dz,
            ))
    return pts


# --- 7. Chamfer is 0 for identical point sets ------------------------------ #

def test_chamfer_zero_for_identical_sets():
    pts = [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    d = symmetric_chamfer_distance(pts, pts, 1.0, 1.0, 1.0)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_chamfer_empty_both_is_zero():
    assert symmetric_chamfer_distance([], [], 1.0, 1.0, 1.0) == 0.0


def test_chamfer_one_empty_raises():
    with pytest.raises(EmptySurfaceError):
        symmetric_chamfer_distance([[0.0, 0.0, 0.0]], [], 1.0, 1.0, 1.0)


# --- 8. Chamfer increases for shifted point sets --------------------------- #

def test_chamfer_increases_with_shift():
    a = [[float(i), 0.0, 0.0] for i in range(10)]
    near = [[float(i) + 0.5, 0.0, 0.0] for i in range(10)]
    far = [[float(i) + 5.0, 0.0, 0.0] for i in range(10)]
    d_near = symmetric_chamfer_distance(a, near, 1.0, 1.0, 1.0)
    d_far = symmetric_chamfer_distance(a, far, 1.0, 1.0, 1.0)
    assert d_near > 0.0
    assert d_far > d_near


def test_chamfer_z_weight_amplifies_vertical_offset():
    a = [[0.0, 0.0, 0.0]]
    b = [[0.0, 0.0, 1.0]]
    d_flat = symmetric_chamfer_distance(a, b, 1.0, 1.0, 1.0)
    d_weighted = symmetric_chamfer_distance(a, b, 1.0, 1.0, 2.0)
    assert d_weighted == pytest.approx(2.0 * d_flat)


# --- 9. scoring: identical clouds score lower than shifted ----------------- #

def _meta(slug, number=1, par=4, yards=440.0):
    return HoleMetadata(
        hole_id=make_pc_hole_id(slug, number), course_slug=slug, hole_number=number,
        par=par, yards=yards,
        has_tee=True, has_green=True, has_fairway=True, has_bunker=True, has_water=True,
    )


def _all_surface_points(hole_id, *, dx=0.0, dy=0.0, dz=0.0):
    pts = []
    for surface in ("fairway", "green", "bunker", "water", "tee"):
        pts.extend(_grid(surface, hole_id, dx=dx, dy=dy, dz=dz))
    return pts


def test_scoring_identical_lower_than_shifted():
    cfg = _config()
    target = _meta("course_a")
    twin = _meta("course_b")
    shifted = _meta("course_c")

    t_pts = _all_surface_points(target.hole_id)
    twin_pts = _all_surface_points(twin.hole_id)               # identical geometry
    shifted_pts = _all_surface_points(shifted.hole_id, dx=10.0)  # translated away

    score_twin = score_pair(target, twin, t_pts, twin_pts, cfg).total_score
    score_shifted = score_pair(target, shifted, t_pts, shifted_pts, cfg).total_score

    assert score_twin < score_shifted
    # Identical geometry + identical yardage/elevation -> exactly zero.
    assert score_twin == pytest.approx(0.0, abs=1e-9)


def test_scoring_records_component_scores():
    cfg = _config()
    target = _meta("course_a")
    cand = _meta("course_b")
    result = score_pair(
        target, cand,
        _all_surface_points(target.hole_id),
        _all_surface_points(cand.hole_id, dx=3.0),
        cfg,
    )
    assert result.fairway_score is not None
    assert result.green_score is not None
    assert result.tee_score is not None
    assert result.model_version == cfg.model_version
    assert result.config_hash == cfg.config_hash
    # No geometry leaked into the result row.
    assert "points" not in result.to_row()


def test_scoring_applies_missing_surface_penalty():
    cfg = _config()
    target = _meta("course_a")
    cand = _meta("course_b")

    # Target has water points; candidate has none -> water missing penalty applies.
    t_pts = _all_surface_points(target.hole_id)
    cand_pts = [p for p in _all_surface_points(cand.hole_id) if p.surface != "water"]

    result = score_pair(target, cand, t_pts, cand_pts, cfg)
    assert result.water_score == pytest.approx(cfg.surface_missing_penalties["water"])
    assert result.missing_surface_penalty == pytest.approx(
        cfg.surface_weights["water"] * cfg.surface_missing_penalties["water"]
    )


def test_scoring_yardage_penalty_increases_total():
    cfg = _config()
    target = _meta("course_a", yards=440.0)
    same = _meta("course_b", yards=440.0)
    longer = _meta("course_c", yards=445.0)
    pts_t = _all_surface_points(target.hole_id)

    s_same = score_pair(target, same, pts_t, _all_surface_points(same.hole_id), cfg)
    s_long = score_pair(target, longer, pts_t, _all_surface_points(longer.hole_id), cfg)
    assert s_long.yardage_penalty == pytest.approx(5.0 * cfg.penalties.yardage_weight)
    assert s_long.total_score > s_same.total_score


def test_scoring_elevation_penalty_only_when_both_present():
    cfg = _config()
    target = HoleMetadata(
        hole_id=make_pc_hole_id("course_a", 1), course_slug="course_a", hole_number=1,
        par=4, yards=440.0, has_tee=True, has_green=True, has_fairway=True,
        has_bunker=True, has_water=True, tee_elevation_m=10.0, green_elevation_m=20.0,
    )
    cand = HoleMetadata(
        hole_id=make_pc_hole_id("course_b", 1), course_slug="course_b", hole_number=1,
        par=4, yards=440.0, has_tee=True, has_green=True, has_fairway=True,
        has_bunker=True, has_water=True, tee_elevation_m=10.0, green_elevation_m=14.0,
    )
    pts_t = _all_surface_points(target.hole_id)
    pts_c = _all_surface_points(cand.hole_id)
    result = score_pair(target, cand, pts_t, pts_c, cfg)
    # |(20-10) - (14-10)| * 0.10 = 6 * 0.10 = 0.6
    assert result.elevation_penalty == pytest.approx(0.6)

    # Candidate missing green elevation -> no elevation penalty.
    cand_no_elev = HoleMetadata(
        hole_id=make_pc_hole_id("course_c", 1), course_slug="course_c", hole_number=1,
        par=4, yards=440.0, has_tee=True, has_green=True, has_fairway=True,
        has_bunker=True, has_water=True, tee_elevation_m=10.0, green_elevation_m=None,
    )
    result2 = score_pair(target, cand_no_elev, pts_t,
                         _all_surface_points(cand_no_elev.hole_id), cfg)
    assert result2.elevation_penalty == 0.0
