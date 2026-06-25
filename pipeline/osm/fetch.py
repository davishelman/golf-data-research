"""Stage 1 — fetch OSM features and normalize to a metric (UTM) CRS.

osmnx already caches Overpass responses under ``cache/`` so re-runs are cheap.
This module's job is to return a single projected GeoDataFrame of all relevant
features plus the chosen UTM CRS; splitting into layers happens in ``layers.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import geopandas as gpd

from ..config import CourseConfig
from ..constants import OSM_TAGS
from ..logging_config import get_logger

log = get_logger("osm.fetch")


@dataclass
class OsmSource:
    """Result of an OSM fetch, normalized to a metric CRS."""

    features: gpd.GeoDataFrame  # all features, projected to UTM
    crs: object                 # the projected (UTM) CRS
    osm_id_col: Optional[str]   # column holding OSM ids, if discoverable
    element_col: Optional[str]  # column holding element type, if discoverable


_ID_COLS = ("osmid", "id", "element_id")
_ELEMENT_COLS = ("element", "element_type")


def _normalize_index(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Flatten osmnx's (element, id) MultiIndex into columns when present."""
    if gdf.index.nlevels > 1 or gdf.index.name:
        try:
            gdf = gdf.reset_index()
        except Exception:  # noqa: BLE001
            pass
    return gdf


def _find_col(gdf: gpd.GeoDataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def fetch_osm_source(course: CourseConfig) -> OsmSource:
    """Fetch OSM features around the course and project to its UTM zone."""
    import osmnx as ox  # imported lazily so tests never need osmnx

    log.info("fetching OSM features at (%s, %s) radius=%sm",
             course.lat, course.lon, course.search_radius_m)
    raw = ox.features_from_point(
        center_point=(course.lat, course.lon),
        tags=OSM_TAGS,
        dist=course.search_radius_m,
    )
    if raw.empty:
        raise RuntimeError("OSM returned no features. Check lat/lon/search_radius_m.")

    raw = _normalize_index(raw)
    log.info("raw OSM features: %d", len(raw))

    target_crs = raw.estimate_utm_crs()
    features = raw.to_crs(target_crs)

    return OsmSource(
        features=features,
        crs=target_crs,
        osm_id_col=_find_col(features, _ID_COLS),
        element_col=_find_col(features, _ELEMENT_COLS),
    )
