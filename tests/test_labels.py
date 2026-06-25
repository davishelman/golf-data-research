from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, box

from pipeline.features.labels import classify_points

CRS = "EPSG:32617"


def _layers():
    fairway = gpd.GeoDataFrame(geometry=[box(0, 0, 100, 100)], crs=CRS)
    # bunker + green both sit *inside* the fairway to test priority override.
    bunker = gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20)], crs=CRS)
    green = gpd.GeoDataFrame(geometry=[box(80, 80, 95, 95)], crs=CRS)
    return {"fairways": fairway, "bunkers": bunker, "greens": green}


def _points(coords):
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in coords], crs=CRS)


def test_priority_bunker_over_fairway():
    pts = _points([(15, 15)])  # inside bunker (and fairway)
    labels, ids, sources, conf = classify_points(pts, _layers(), infer_rough=True)
    assert labels[0] == "bunker"


def test_priority_green_over_fairway():
    pts = _points([(88, 88)])  # inside green (and fairway)
    labels, *_ = classify_points(pts, _layers(), infer_rough=True)
    assert labels[0] == "green"


def test_fairway_only_point():
    pts = _points([(50, 50)])
    labels, *_ = classify_points(pts, _layers(), infer_rough=True)
    assert labels[0] == "fairway"


def test_unlabeled_point_inferred_rough():
    pts = _points([(200, 200)])  # outside everything
    labels, ids, sources, conf = classify_points(pts, _layers(), infer_rough=True)
    assert labels[0] == "rough_inferred"
    assert sources[0] == "inferred:background"


def test_unlabeled_point_unknown_when_disabled():
    pts = _points([(200, 200)])
    labels, *_ = classify_points(pts, _layers(), infer_rough=False)
    assert labels[0] == "unknown"
