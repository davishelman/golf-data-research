"""End-to-end integration on a synthetic course (no OSM / OpenTopography)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import exports, orchestrator
from pipeline.config import CourseConfig
from pipeline.paths import CoursePaths, HolePaths
from pipeline.schemas import RunOptions
from tests import synthetic


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    e0, n0 = synthetic.utm_origin()

    monkeypatch.setattr(orchestrator, "fetch_osm_source",
                        lambda course: synthetic.make_osm_source(e0, n0))

    def fake_dem(bounds_projected, src_crs, dem_path, dem_type, force=False):
        synthetic.write_ramp_dem(Path(dem_path), e0, n0)
        return Path(dem_path), "local"

    monkeypatch.setattr(orchestrator, "download_course_dem", fake_dem)

    course = CourseConfig(
        course_slug="synthetic_course",
        course_name="Synthetic Course",
        lat=synthetic.ANCHOR_LAT, lon=synthetic.ANCHOR_LON,
        country="US", dem_type="USGS1m", par=72, holes_count=18,
        hole_buffer_meters=100.0,
    )
    options = RunOptions(
        courses_root=tmp_path, skip_plots=True,
        point_sampling_resolution_m=5.0, write_parquet=True,
    )
    result = orchestrator.run_course(course, options)
    return result, tmp_path, course


def test_course_processed_with_18_holes(synthetic_run):
    result, root, course = synthetic_run
    assert result.status == "processed"
    assert result.processed_holes == 18

    paths = CoursePaths.for_slug(course.course_slug, courses_root=root)
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["status"] == "processed"
    assert manifest["detected_holes"] == 18
    assert len(manifest["holes"]) == 18
    assert manifest["projected_crs"]


def test_each_hole_has_artifacts(synthetic_run):
    _, root, course = synthetic_run
    paths = CoursePaths.for_slug(course.course_slug, courses_root=root)
    for n in range(1, 19):
        hp = HolePaths.for_hole(paths, n)
        assert hp.terrain_summary.exists(), f"missing terrain summary hole {n}"
        assert hp.label_map.exists()
        assert hp.hole_points_jsonl.exists()
        assert hp.hole_points_compact.exists()


def test_tee_maps_to_origin_and_green_downrange(synthetic_run):
    _, root, course = synthetic_run
    paths = CoursePaths.for_slug(course.course_slug, courses_root=root)
    hp = HolePaths.for_hole(paths, 1)
    records = [json.loads(l) for l in hp.hole_points_jsonl.read_text().splitlines() if l.strip()]
    assert records

    # A grid point close to the tee anchor exists (tee ~ (0,0,0)).
    min_origin_dist = min((r["x_rel_m"] ** 2 + r["y_rel_m"] ** 2) ** 0.5 for r in records)
    assert min_origin_dist < 10.0

    # The green is downrange: aligned +Y reaches well beyond the tee.
    max_y_aligned = max(r["y_aligned_m"] for r in records if r["y_aligned_m"] is not None)
    assert max_y_aligned > 250.0

    # All aligned points have x roughly centered (corridor is narrow vs length).
    labels = {r["label"] for r in records}
    assert {"fairway", "tee", "green", "rough_inferred"}.issubset(labels)


def test_terrain_summary_fields(synthetic_run):
    _, root, course = synthetic_run
    paths = CoursePaths.for_slug(course.course_slug, courses_root=root)
    summary = json.loads(HolePaths.for_hole(paths, 1).terrain_summary.read_text())
    for key in ("tee_elevation_m", "green_elevation_m", "net_elevation_change_m",
                "min_elevation_m", "max_elevation_m", "mean_elevation_m",
                "avg_slope_deg", "avg_slope_percent"):
        assert key in summary
    # Green is north (higher) than tee on the ramp -> net positive.
    assert summary["net_elevation_change_m"] > 0


def test_aggregate_index(synthetic_run):
    _, root, _ = synthetic_run
    written = exports.build_aggregate_index(root, write_parquet=True, write_duckdb=True)
    assert "all_holes_csv" in written
    csv_text = Path(written["all_holes_csv"]).read_text()
    # header + 18 hole rows.
    assert len([ln for ln in csv_text.splitlines() if ln.strip()]) == 19


def test_cached_source_reprocesses(synthetic_run, monkeypatch):
    # A second run (no refetch) must load cached source layers and still process
    # — guards the hole_number round-trip through source GeoJSON.
    result, root, course = synthetic_run

    def boom(course):
        raise AssertionError("fetch_osm_source should not be called on cache hit")

    monkeypatch.setattr(orchestrator, "fetch_osm_source", boom)
    monkeypatch.setattr(orchestrator, "download_course_dem",
                        lambda **kw: (kw["dem_path"], "local"))

    options = RunOptions(courses_root=root, skip_plots=True,
                         point_sampling_resolution_m=5.0, write_parquet=False)
    second = orchestrator.run_course(course, options)
    assert second.status == "processed"
    assert second.processed_holes == 18


def test_dirty_course_skipped(tmp_path, monkeypatch):
    e0, n0 = synthetic.utm_origin()

    # Drop hole 7 so only 17 are detected.
    def short_source(course):
        src = synthetic.make_osm_source(e0, n0)
        feats = src.features
        mask = ~((feats["golf"] == "hole") & (feats["ref"] == "7"))
        src.features = feats[mask].copy()
        return src

    monkeypatch.setattr(orchestrator, "fetch_osm_source", short_source)
    monkeypatch.setattr(orchestrator, "download_course_dem",
                        lambda **kw: (kw["dem_path"], "local"))

    course = CourseConfig(course_slug="dirty_course", course_name="Dirty",
                          lat=synthetic.ANCHOR_LAT, lon=synthetic.ANCHOR_LON,
                          country="US", holes_count=18)
    result = orchestrator.run_course(course, RunOptions(courses_root=tmp_path, skip_plots=True))
    assert result.status == "skipped"

    paths = CoursePaths.for_slug("dirty_course", courses_root=tmp_path)
    quality = json.loads(paths.quality_report.read_text())
    assert quality["status"] == "skipped"
    assert any(i["code"] == "EXPECTED_HOLES_MISMATCH" for i in quality["issues"])
