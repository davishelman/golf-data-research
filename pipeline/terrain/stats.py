"""Build the per-hole TerrainSummary from a metric DEM + slope + anchors."""

from __future__ import annotations

from typing import Optional

import numpy as np
from shapely.geometry import LineString

from ..schemas import HoleAnchors, HoleIdentity, TerrainSummary


def _finite_stat(fn, arr: np.ndarray) -> Optional[float]:
    if not np.any(np.isfinite(arr)):
        return None
    return float(fn(arr))


def build_terrain_summary(
    identity: HoleIdentity,
    course_name: str,
    hole_line: LineString,
    dem: np.ndarray,
    slope_deg: np.ndarray,
    slope_pct: np.ndarray,
    anchors: HoleAnchors,
    dem_type: str,
    dem_source: str,
    raster_resolution_m: Optional[float],
    quality_flags: Optional[list[str]] = None,
) -> TerrainSummary:
    tee_elev = anchors.tee_elevation_m
    green_elev = anchors.green_elevation_m
    tee_ok = tee_elev is not None and np.isfinite(tee_elev)
    green_ok = green_elev is not None and np.isfinite(green_elev)
    net = (green_elev - tee_elev) if (tee_ok and green_ok) else None

    min_e = _finite_stat(np.nanmin, dem)
    max_e = _finite_stat(np.nanmax, dem)
    mean_e = _finite_stat(np.nanmean, dem)
    elev_range = (max_e - min_e) if (min_e is not None and max_e is not None) else None

    return TerrainSummary(
        hole_id=identity.hole_id,
        course_slug=identity.course_slug,
        course_name=course_name,
        hole_number=identity.hole_number,
        hole_name=identity.name,
        par=identity.par,
        handicap=identity.handicap,
        hole_length_m=float(hole_line.length),
        tee_elevation_m=float(tee_elev) if tee_ok else None,
        green_elevation_m=float(green_elev) if green_ok else None,
        net_elevation_change_m=net,
        abs_elevation_change_m=abs(net) if net is not None else None,
        min_elevation_m=min_e,
        max_elevation_m=max_e,
        mean_elevation_m=mean_e,
        elevation_range_m=elev_range,
        avg_slope_deg=_finite_stat(np.nanmean, slope_deg),
        max_slope_deg=_finite_stat(np.nanmax, slope_deg),
        avg_slope_percent=_finite_stat(np.nanmean, slope_pct),
        max_slope_percent=_finite_stat(np.nanmax, slope_pct),
        dem_type=dem_type,
        dem_source=dem_source,
        raster_resolution_m=raster_resolution_m,
        tee_selection_method=anchors.tee_selection_method,
        green_selection_method=anchors.green_selection_method,
        quality_flags=quality_flags or [],
    )
