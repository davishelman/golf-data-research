"""Shared geometry helpers (CRS-agnostic; callers must pass metric geometries).

Kept dependency-light: only shapely. No project imports, so anything can use it.
"""

from __future__ import annotations

from typing import Optional

from shapely.geometry import LineString, Point


def ensure_linestring(geom) -> LineString:
    """Coerce a (possibly Multi*) geometry into a single LineString.

    Concatenates sub-part coordinates in order. Raises on unsupported types.
    """
    if isinstance(geom, LineString):
        return geom
    try:
        coords: list = []
        for part in geom.geoms:  # MultiLineString / GeometryCollection
            coords.extend(list(part.coords))
        return LineString(coords)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unsupported hole geometry type: {type(geom)}") from exc


def nearest_hole_number(geom, hole_lines: dict[int, LineString]) -> Optional[int]:
    """Return the hole number whose centerline is closest to ``geom``.

    Ties resolve to the smaller hole number for determinism.
    """
    best_hole: Optional[int] = None
    best_dist: Optional[float] = None
    for hole_number in sorted(hole_lines):
        d = geom.distance(hole_lines[hole_number])
        if best_dist is None or d < best_dist:
            best_dist, best_hole = d, hole_number
    return best_hole


def representative_point(geom) -> Point:
    """A guaranteed-inside point for polygons; the geometry itself if a Point."""
    if isinstance(geom, Point):
        return geom
    try:
        return geom.representative_point()
    except Exception:  # noqa: BLE001
        c = geom.centroid
        return Point(c.x, c.y)
