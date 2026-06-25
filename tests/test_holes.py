from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, box

from pipeline.config import CourseConfig
from pipeline.osm.boundary import BoundarySelection
from pipeline.osm.fetch import OsmSource
from pipeline.osm.holes import detect_main_holes

CRS = "EPSG:32617"


def _boundary():
    poly = box(0, 0, 1000, 1000)
    gdf = gpd.GeoDataFrame({"x": [1]}, geometry=[poly], crs=CRS)
    return BoundarySelection(boundary=gdf, geometry=poly, metadata={})


def _source(records, geoms):
    feats = gpd.GeoDataFrame(records, geometry=geoms, crs=CRS)
    return OsmSource(features=feats, crs=CRS, osm_id_col=None, element_col=None)


def _course(holes_count=4):
    return CourseConfig(course_slug="t", course_name="T", lat=0, lon=0,
                        holes_count=holes_count)


def test_strict_18_like_validation_missing_hole():
    # Holes 1,2,3 present (hole 4 missing) with holes_count=4.
    records = [{"golf": "hole", "ref": str(n)} for n in (1, 2, 3)]
    geoms = [LineString([(100 * n, 0), (100 * n, 200)]) for n in (1, 2, 3)]
    res = detect_main_holes(_course(4), _source(records, geoms), _boundary())
    assert res.detected == 3
    assert res.missing_refs == [4]
    assert res.is_clean is False


def test_exactly_expected_is_clean():
    records = [{"golf": "hole", "ref": str(n)} for n in (1, 2, 3, 4)]
    geoms = [LineString([(100 * n, 0), (100 * n, 200)]) for n in (1, 2, 3, 4)]
    res = detect_main_holes(_course(4), _source(records, geoms), _boundary())
    assert res.detected == 4
    assert res.is_clean is True


def test_duplicate_ref_keeps_longest():
    # Hole 2 appears twice; the longer centerline must win.
    records = [
        {"golf": "hole", "ref": "1"},
        {"golf": "hole", "ref": "2"},       # short
        {"golf": "hole", "ref": "2"},       # long (should be kept)
        {"golf": "hole", "ref": "3"},
        {"golf": "hole", "ref": "4"},
    ]
    geoms = [
        LineString([(100, 0), (100, 200)]),
        LineString([(200, 0), (200, 50)]),    # length 50
        LineString([(200, 0), (200, 400)]),   # length 400
        LineString([(300, 0), (300, 200)]),
        LineString([(400, 0), (400, 200)]),
    ]
    res = detect_main_holes(_course(4), _source(records, geoms), _boundary())
    assert res.detected == 4
    assert 2 in res.duplicate_refs
    kept_hole2 = res.main_holes[res.main_holes["hole_number"] == 2].iloc[0]
    assert abs(kept_hole2.geometry.length - 400.0) < 1e-6
