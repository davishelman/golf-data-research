from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, box

from pipeline.osm.assignment import assign_layer_to_hole

CRS = "EPSG:32617"

LINE1 = LineString([(0, 0), (0, 300)])
LINE2 = LineString([(300, 0), (300, 300)])
HOLE_LINES = {1: LINE1, 2: LINE2}


def _fairways():
    records = [
        {"ref": "1"},     # A: belongs to hole 1 by ref, near hole 1
        {"ref": None},    # B: untagged, near hole 2
        {"ref": "2"},     # C: ref says hole 2, but sits near hole 1
    ]
    geoms = [
        box(-10, 140, 10, 160),     # near hole 1 centerline
        box(290, 140, 310, 160),    # near hole 2 centerline
        box(-10, 190, 10, 210),     # near hole 1 centerline
    ]
    return gpd.GeoDataFrame(records, geometry=geoms, crs=CRS)


def test_ref_feature_goes_to_matching_hole():
    out = assign_layer_to_hole("fairways", _fairways(), 1, LINE1.buffer(100), HOLE_LINES)
    assert len(out) == 1
    assert out.iloc[0]["assignment_method"] == "ref"


def test_wrong_ref_feature_is_dropped():
    # Feature C (ref=2) sits in hole 1's buffer but must NOT be assigned to hole 1.
    out = assign_layer_to_hole("fairways", _fairways(), 1, LINE1.buffer(100), HOLE_LINES)
    methods = set(out["assignment_method"])
    assert "ref" in methods
    assert len(out) == 1  # only feature A


def test_untagged_feature_assigned_by_nearest_centerline():
    # Only feature B (untagged, near hole 2) intersects hole 2's buffer; it is
    # assigned by nearest centerline. Feature C (ref=2) sits at hole 1 and never
    # enters hole 2's corridor.
    out = assign_layer_to_hole("fairways", _fairways(), 2, LINE2.buffer(100), HOLE_LINES)
    assert len(out) == 1
    assert out.iloc[0]["assignment_method"] == "nearest_centerline"


def test_ref_feature_assigned_even_across_buffers():
    # A feature whose ref names hole 2 but which overlaps hole 2's buffer is kept
    # by ref. Place a ref=2 fairway inside hole 2's corridor.
    import geopandas as gpd
    from shapely.geometry import box as _box
    g = gpd.GeoDataFrame({"ref": ["2"]}, geometry=[_box(290, 90, 310, 110)], crs=CRS)
    out = assign_layer_to_hole("fairways", g, 2, LINE2.buffer(100), HOLE_LINES)
    assert len(out) == 1
    assert out.iloc[0]["assignment_method"] == "ref"


def test_shared_layer_uses_overlap():
    water = gpd.GeoDataFrame({"ref": [None]}, geometry=[box(-5, 140, 5, 160)], crs=CRS)
    out = assign_layer_to_hole("water", water, 1, LINE1.buffer(100), HOLE_LINES)
    assert len(out) == 1
    assert out.iloc[0]["assignment_method"] == "shared_overlap"
