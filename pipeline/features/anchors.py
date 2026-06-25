"""Stage 6 — select tee and green anchors for a hole.

The centerline endpoints are the most reliable anchors. When assigned tee/green
features exist they refine the anchor: the tee feature nearest the centerline
start, and the green feature nearest the centerline end. Selection method and a
confidence score are always recorded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import LineString, Point

from ..geometry import representative_point
from ..logging_config import get_logger
from ..raster.sampling import sample_raster_at_points
from ..schemas import HoleAnchors

log = get_logger("features.anchors")


def _nearest_feature_point(gdf: Optional[gpd.GeoDataFrame], target: Point) -> Optional[Point]:
    if gdf is None or gdf.empty:
        return None
    best_pt: Optional[Point] = None
    best_d: Optional[float] = None
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        rp = representative_point(geom)
        d = rp.distance(target)
        if best_d is None or d < best_d:
            best_d, best_pt = d, rp
    return best_pt


def select_anchors(
    hole_line: LineString,
    tees: Optional[gpd.GeoDataFrame],
    greens: Optional[gpd.GeoDataFrame],
    projected_dem_path: Path,
) -> HoleAnchors:
    coords = list(hole_line.coords)
    start = Point(coords[0][0], coords[0][1])
    end = Point(coords[-1][0], coords[-1][1])

    tee_pt = _nearest_feature_point(tees, start)
    if tee_pt is not None:
        tee_method = "nearest_centerline_start_tee"
        tee_conf = 0.85
    else:
        tee_pt = start
        tee_method = "centerline_start"
        tee_conf = 0.6

    green_pt = _nearest_feature_point(greens, end)
    if green_pt is not None:
        green_method = "nearest_centerline_end_green"
        green_conf = 0.85
    else:
        green_pt = end
        green_method = "centerline_end"
        green_conf = 0.6

    tee_elev, green_elev = sample_raster_at_points(
        projected_dem_path, [(tee_pt.x, tee_pt.y), (green_pt.x, green_pt.y)]
    )

    return HoleAnchors(
        tee_point=tee_pt,
        green_point=green_pt,
        centerline=hole_line,
        tee_elevation_m=float(tee_elev),
        green_elevation_m=float(green_elev),
        tee_selection_method=tee_method,
        green_selection_method=green_method,
        confidence=round((tee_conf + green_conf) / 2.0, 3),
    )
