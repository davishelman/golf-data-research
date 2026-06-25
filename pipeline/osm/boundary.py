"""Stage 2 — select the main course boundary polygon.

Prefers an explicit ``osm_relation_id`` when configured; otherwise falls back to
scoring ``leisure=golf_course`` polygons by proximity to the configured point and
area. Always records how the decision was made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import geopandas as gpd
from shapely.geometry import Point

from ..config import CourseConfig
from ..logging_config import get_logger
from .fetch import OsmSource

log = get_logger("osm.boundary")


@dataclass
class BoundarySelection:
    boundary: gpd.GeoDataFrame  # single-row GeoDataFrame
    geometry: Any               # unioned boundary geometry (projected)
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.metadata


def _course_center(course: CourseConfig, crs) -> Point:
    return (
        gpd.GeoSeries.from_xy([course.lon], [course.lat], crs="EPSG:4326")
        .to_crs(crs)
        .iloc[0]
    )


def _candidates(source: OsmSource) -> gpd.GeoDataFrame:
    feats = source.features
    if "leisure" not in feats.columns:
        return feats.iloc[0:0].copy()
    return feats[feats["leisure"] == "golf_course"].copy()


def _try_relation_id(
    candidates: gpd.GeoDataFrame, source: OsmSource, relation_id: int
) -> Optional[gpd.GeoDataFrame]:
    """Best-effort match of a candidate boundary to an OSM relation id."""
    if candidates.empty or source.osm_id_col is None:
        return None
    id_col = source.osm_id_col
    try:
        ids = candidates[id_col].astype("Int64")
    except Exception:  # noqa: BLE001
        return None
    match = candidates[ids == relation_id]
    if match.empty:
        return None
    # If we can tell element type, prefer relation/way rows.
    if source.element_col and source.element_col in match.columns:
        rel = match[match[source.element_col].astype(str).str.contains("relation", case=False)]
        if not rel.empty:
            return rel.head(1).copy()
    return match.head(1).copy()


def select_boundary(course: CourseConfig, source: OsmSource) -> BoundarySelection:
    candidates = _candidates(source)
    candidate_count = int(len(candidates))
    if candidates.empty:
        raise ValueError("No leisure=golf_course polygon found near the configured point.")

    center = _course_center(course, source.crs)

    # 1. Relation-id-aware selection.
    if course.osm_relation_id is not None:
        chosen = _try_relation_id(candidates, source, int(course.osm_relation_id))
        if chosen is not None and not chosen.empty:
            geom = chosen.geometry.union_all()
            log.info("boundary selected via osm_relation_id=%s", course.osm_relation_id)
            return BoundarySelection(
                boundary=chosen,
                geometry=geom,
                metadata={
                    "selection_method": "osm_relation_id",
                    "osm_relation_id": int(course.osm_relation_id),
                    "fallback_used": False,
                    "candidate_count": candidate_count,
                },
            )
        log.warning("osm_relation_id=%s not matched; falling back to scoring",
                    course.osm_relation_id)

    # 2. Scoring fallback: nearest to center, then largest.
    scored = candidates.copy()
    scored["area_m2"] = scored.geometry.area
    scored["dist_to_center"] = scored.geometry.distance(center)
    scored = scored.sort_values(["dist_to_center", "area_m2"], ascending=[True, False])
    chosen = scored.head(1).copy()
    geom = chosen.geometry.union_all()
    log.info("boundary selected via scoring: area=%.0f m^2 dist=%.0f m",
             float(chosen["area_m2"].iloc[0]), float(chosen["dist_to_center"].iloc[0]))

    return BoundarySelection(
        boundary=chosen,
        geometry=geom,
        metadata={
            "selection_method": "scoring",
            "osm_relation_id": (int(course.osm_relation_id)
                                if course.osm_relation_id is not None else None),
            "fallback_used": course.osm_relation_id is not None,
            "candidate_count": candidate_count,
            "selected_area_m2": round(float(chosen["area_m2"].iloc[0]), 1),
            "selected_dist_to_center_m": round(float(chosen["dist_to_center"].iloc[0]), 1),
        },
    )
