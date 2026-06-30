"""Tests for the dedicated per-surface point-cloud artifact loader + writer."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.pointcloud.schemas import (
    HoleMetadata,
    SurfacePoint,
    make_pc_hole_id,
)
from pipeline.modeling.pointcloud.surface_loader import (
    SURFACE_POINT_COLUMNS,
    SurfacePointArtifactLoader,
    export_surface_artifact,
)


class FakeLoader:
    def __init__(self, metadata, points):
        self._metadata = metadata
        self._points = points

    def load_metadata(self):
        return dict(self._metadata)

    def load_points(self, hole_id):
        return list(self._points.get(hole_id, []))


def _source_loader():
    a = make_pc_hole_id("aaa_course", 1)
    b = make_pc_hole_id("bbb_course", 1)
    metadata = {
        a: HoleMetadata(hole_id=a, course_slug="aaa_course", hole_number=1, par=4,
                        yards=440.0, has_tee=True, has_green=True, has_fairway=True,
                        has_bunker=True, has_water=False, tee_elevation_m=10.0,
                        green_elevation_m=13.0, course_name="AAA"),
        b: HoleMetadata(hole_id=b, course_slug="bbb_course", hole_number=1, par=4,
                        yards=450.0, has_tee=True, has_green=True, has_fairway=True,
                        has_bunker=False, has_water=False),
    }
    points = {
        a: [
            SurfacePoint(a, "fairway", 1.0, 2.0, 0.5, point_weight=2.0),
            SurfacePoint(a, "green", 0.0, 50.0, 1.0),
            SurfacePoint(a, "bunker", -3.0, 30.0, 0.0),
            SurfacePoint(a, "tee", 0.0, 0.0, 0.0),
        ],
        b: [
            SurfacePoint(b, "fairway", 2.0, 3.0, 0.2),
            SurfacePoint(b, "green", 1.0, 55.0, 0.8),
            SurfacePoint(b, "tee", 0.0, 0.0, 0.0),
        ],
    }
    return FakeLoader(metadata, points), a, b


@pytest.mark.parametrize("ext", [".parquet", ".csv"])
def test_export_then_load_roundtrip(tmp_path, ext):
    src, a, b = _source_loader()
    points_path = tmp_path / f"surface_points{ext}"
    meta_path = tmp_path / f"hole_metadata{ext}"

    stats = export_surface_artifact(src, points_path, meta_path)
    assert stats["holes"] == 2
    assert stats["points"] == 7

    loader = SurfacePointArtifactLoader(points_path, meta_path)
    meta = loader.load_metadata()
    assert set(meta) == {a, b}
    # Metadata round-trips.
    assert meta[a].par == 4
    assert meta[a].yards == pytest.approx(440.0)
    assert meta[a].course_name == "AAA"
    assert meta[a].tee_elevation_m == pytest.approx(10.0)
    assert meta[a].has_bunker is True
    assert meta[b].has_bunker is False

    # Points round-trip, including a non-default point_weight.
    a_points = loader.load_points(a)
    assert len(a_points) == 4
    fairway = next(p for p in a_points if p.surface == "fairway")
    assert fairway.point_weight == pytest.approx(2.0)
    assert fairway.x_lateral_m == pytest.approx(1.0)


def test_has_flags_derived_from_points_when_absent(tmp_path):
    # Metadata table without has_* columns -> flags derived from the points table.
    src, a, b = _source_loader()
    points_path = tmp_path / "p.csv"
    meta_path = tmp_path / "m.csv"
    export_surface_artifact(src, points_path, meta_path)

    # Strip the has_* columns to force derivation.
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df[[c for c in meta_df.columns if not c.startswith("has_")]]
    meta_df.to_csv(meta_path, index=False)

    loader = SurfacePointArtifactLoader(points_path, meta_path)
    meta = loader.load_metadata()
    # Hole a has fairway/green/bunker/tee points but no water.
    assert meta[a].has_fairway is True
    assert meta[a].has_bunker is True
    assert meta[a].has_water is False
    # Hole b has no bunker points.
    assert meta[b].has_bunker is False


def test_point_weight_defaults_to_one_when_missing(tmp_path):
    points_path = tmp_path / "p.csv"
    meta_path = tmp_path / "m.csv"
    a = make_pc_hole_id("aaa_course", 1)
    pd.DataFrame([
        {"hole_id": a, "surface": "fairway", "x_lateral_m": 1.0,
         "y_down_hole_m": 2.0, "z_relative_m": 0.0},
    ]).to_csv(points_path, index=False)
    pd.DataFrame([
        {"hole_id": a, "course_slug": "aaa_course", "hole_number": 1,
         "par": 4, "yards": 440.0},
    ]).to_csv(meta_path, index=False)

    loader = SurfacePointArtifactLoader(points_path, meta_path)
    pts = loader.load_points(a)
    assert pts[0].point_weight == pytest.approx(1.0)


def test_missing_required_point_column_raises(tmp_path):
    points_path = tmp_path / "p.csv"
    meta_path = tmp_path / "m.csv"
    # Missing y_down_hole_m.
    pd.DataFrame([{"hole_id": "a:1", "surface": "fairway",
                   "x_lateral_m": 1.0, "z_relative_m": 0.0}]).to_csv(points_path, index=False)
    pd.DataFrame([{"hole_id": "a:1", "course_slug": "a", "hole_number": 1,
                   "par": 4, "yards": 440.0}]).to_csv(meta_path, index=False)
    loader = SurfacePointArtifactLoader(points_path, meta_path)
    with pytest.raises(KeyError, match="y_down_hole_m"):
        loader.load_points("a:1")


def test_surface_point_columns_constant():
    assert SURFACE_POINT_COLUMNS == (
        "hole_id", "surface", "x_lateral_m", "y_down_hole_m", "z_relative_m", "point_weight",
    )
