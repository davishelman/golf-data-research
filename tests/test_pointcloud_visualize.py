"""Tests for v2.5 visual inspection helpers (pure helpers + render smoke)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pipeline.modeling.pointcloud.export_similarity import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    SIMILARITY_RESULTS_COLUMNS,
)
from pipeline.modeling.pointcloud.schemas import SurfacePoint, make_pc_hole_id
from pipeline.modeling.pointcloud.visualize_similarity import (
    SURFACE_COLORS,
    build_comparison_figure,
    compute_bounds,
    run_visual_report,
    score_breakdown_text,
    surface_points_xy,
)


def _points(hole_id, *, dx=0.0):
    pts = []
    for surface in ("fairway", "green", "bunker", "water", "tee"):
        for i in range(3):
            pts.append(SurfacePoint(
                hole_id=hole_id, surface=surface,
                x_lateral_m=float(i) + dx, y_down_hole_m=float(i) * 2.0, z_relative_m=0.0,
            ))
    return pts


class FakeLoader:
    def __init__(self, points: dict[str, list[SurfacePoint]]):
        self._points = points

    def load_metadata(self):  # not used by visualizer
        return {}

    def load_points(self, hole_id):
        return list(self._points.get(hole_id, []))


# --- pure helpers ---------------------------------------------------------- #

def test_surface_points_xy_filters_by_surface():
    pts = _points("a:1")
    xs, ys = surface_points_xy(pts, "green")
    assert len(xs) == 3
    assert len(ys) == 3
    # No points from other surfaces leak in (all x in {0,1,2}).
    assert set(xs.tolist()) == {0.0, 1.0, 2.0}


def test_surface_points_xy_empty_surface():
    xs, ys = surface_points_xy(_points("a:1"), surface="nonexistent")
    assert xs.size == 0 and ys.size == 0


def test_compute_bounds_padding():
    pts = _points("a:1")
    xmin, xmax, ymin, ymax = compute_bounds(pts, pad=5.0)
    assert xmin <= 0.0 - 5.0 + 1e-9
    assert xmax >= 2.0 + 5.0 - 1e-9
    assert ymin <= 0.0 - 5.0 + 1e-9


def test_compute_bounds_empty_is_unit_box():
    assert compute_bounds([], pad=3.0) == (-3.0, 3.0, -3.0, 3.0)


def test_score_breakdown_text_handles_missing():
    text = score_breakdown_text({
        "total_score": 12.345, "fairway_score": 1.0, "green_score": None,
        "bunker_score": float("nan"), "water_score": 2.0, "tee_score": 0.5,
        "yardage_penalty": 0.1, "elevation_penalty": 0.0, "missing_surface_penalty": 0.0,
    })
    assert "total: 12.345" in text
    assert "green: —" in text     # None -> em dash
    assert "bunker: —" in text    # NaN -> em dash
    assert "water: 2.000" in text


def test_surface_colors_cover_all_surfaces():
    assert set(SURFACE_COLORS) == {"fairway", "green", "bunker", "water", "tee"}


# --- render smoke (Agg, no pixel snapshot) --------------------------------- #

def test_build_comparison_figure_smoke():
    target_pts = _points("t:1")
    candidates = [
        ("c:1", _points("c:1", dx=1.0), {"rank": 1, "total_score": 3.0,
                                          "fairway_score": 1.0, "green_score": 1.0,
                                          "bunker_score": 1.0, "water_score": None,
                                          "tee_score": 0.5, "yardage_penalty": 0.0,
                                          "elevation_penalty": 0.0, "missing_surface_penalty": 0.0}),
    ]
    fig = build_comparison_figure("t:1", target_pts, candidates)
    # One target axis + one candidate axis.
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_run_visual_report_writes_png(tmp_path):
    target = make_pc_hole_id("aaa_course", 1)
    cand = make_pc_hole_id("bbb_course", 1)
    # Minimal batch result dir with one row for the target.
    rdir = tmp_path / "baseline"
    rdir.mkdir()
    row = {c: None for c in SIMILARITY_RESULTS_COLUMNS}
    row.update({
        "model_version": "v2_5_chamfer_v1", "config_name": "baseline",
        "config_hash": "x", "target_hole_id": target, "candidate_hole_id": cand,
        "rank": 1, "total_score": 2.0, "fairway_score": 1.0, "green_score": 1.0,
        "bunker_score": 0.0, "water_score": 0.0, "tee_score": 0.0,
        "yardage_penalty": 0.0, "elevation_penalty": 0.0, "missing_surface_penalty": 0.0,
        "filter_reason": "PASS",
    })
    pd.DataFrame([row], columns=list(SIMILARITY_RESULTS_COLUMNS)).to_csv(
        rdir / RESULTS_FILENAME, index=False)
    (rdir / MANIFEST_FILENAME).write_text(json.dumps({"config_name": "baseline"}),
                                          encoding="utf-8")

    loader = FakeLoader({target: _points(target), cand: _points(cand, dx=1.0)})
    png = run_visual_report(
        target, "baseline", loader=loader, result_dir=rdir,
        output_dir=tmp_path / "out", top_n=6,
    )
    assert png.exists()
    assert png.stat().st_size > 0
