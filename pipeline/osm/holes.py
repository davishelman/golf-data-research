"""Stage 3 — detect main holes and enforce strict cleanliness.

Holes come from ``golf=hole`` centerlines tagged with a ``ref`` hole number.
Duplicates are resolved deterministically (inside boundary, longest, nearest
center). The course is validated against the expected hole count.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import geopandas as gpd

from ..config import CourseConfig
from ..geometry import ensure_linestring
from ..logging_config import get_logger
from .boundary import BoundarySelection
from .fetch import OsmSource

log = get_logger("osm.holes")


def parse_ref_tokens(value) -> Optional[list[int]]:
    """Parse an OSM ``ref`` into a list of hole-number ints, or None.

    Examples:
        "1"        -> [1]
        "9;10"     -> [9, 10]
        "9, 10"    -> [9, 10]
        None       -> None
        "None"     -> None
        float nan  -> None
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "none" or s.lower() == "nan":
        return None
    tokens = [t for t in re.split(r"[^0-9]+", s) if t]
    if not tokens:
        return None
    return [int(t) for t in tokens]


def parse_hole_number(value) -> Optional[int]:
    """The single hole number a centerline's ref denotes (first token)."""
    tokens = parse_ref_tokens(value)
    return tokens[0] if tokens else None


@dataclass
class HoleDetectionResult:
    main_holes: gpd.GeoDataFrame
    expected: int
    detected: int
    missing_refs: list[int] = field(default_factory=list)
    duplicate_refs: list[int] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.detected == self.expected and not self.missing_refs

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_holes": self.expected,
            "detected_holes": self.detected,
            "missing_refs": self.missing_refs,
            "duplicate_refs": self.duplicate_refs,
            "rejected_candidates": self.rejected,
        }


def detect_main_holes(
    course: CourseConfig,
    source: OsmSource,
    boundary: BoundarySelection,
) -> HoleDetectionResult:
    feats = source.features
    expected = course.holes_count

    if "golf" not in feats.columns:
        return HoleDetectionResult(_empty_like(feats), expected, 0,
                                   missing_refs=list(range(1, expected + 1)))

    holes = feats[feats["golf"] == "hole"].copy()
    # Keep linear geometries only (coerce Multi* later).
    holes = holes[holes.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if holes.empty:
        return HoleDetectionResult(_empty_like(feats), expected, 0,
                                   missing_refs=list(range(1, expected + 1)))

    # Inside-boundary flag and hole number.
    bgeom = boundary.geometry
    holes["_inside"] = holes.geometry.intersects(bgeom)
    holes["_hole_number"] = (
        holes["ref"].apply(parse_hole_number) if "ref" in holes.columns else None
    )
    holes = holes[holes["_hole_number"].notna()].copy()
    holes["_hole_number"] = holes["_hole_number"].astype(int)

    # Restrict to valid range.
    in_range = holes[(holes["_hole_number"] >= 1) & (holes["_hole_number"] <= expected)].copy()
    if in_range.empty:
        return HoleDetectionResult(_empty_like(feats), expected, 0,
                                   missing_refs=list(range(1, expected + 1)))

    in_range["_length"] = in_range.geometry.length
    center = boundary.boundary.geometry.union_all().centroid
    in_range["_dist_center"] = in_range.geometry.distance(center)

    # Deterministic dedupe: prefer inside boundary, then longest, then nearest.
    in_range = in_range.sort_values(
        ["_hole_number", "_inside", "_length", "_dist_center"],
        ascending=[True, False, False, True],
    )
    duplicate_refs = sorted(
        {int(n) for n in in_range["_hole_number"].value_counts().loc[lambda s: s > 1].index}
    )
    chosen = in_range.drop_duplicates(subset="_hole_number", keep="first").copy()

    # Record rejected duplicates for the quality report.
    rejected: list[dict[str, Any]] = []
    kept_idx = set(chosen.index)
    for idx, row in in_range.iterrows():
        if idx in kept_idx:
            continue
        rejected.append({
            "hole_number": int(row["_hole_number"]),
            "length_m": round(float(row["_length"]), 1),
            "inside_boundary": bool(row["_inside"]),
            "reason": "duplicate_ref_lower_priority",
        })

    detected_numbers = sorted(int(n) for n in chosen["_hole_number"].tolist())
    missing = [n for n in range(1, expected + 1) if n not in detected_numbers]

    main_holes = _build_main_holes(chosen, source.crs)

    log.info("detected %d/%d holes (missing=%s duplicates=%s)",
             len(detected_numbers), expected, missing, duplicate_refs)

    return HoleDetectionResult(
        main_holes=main_holes,
        expected=expected,
        detected=len(detected_numbers),
        missing_refs=missing,
        duplicate_refs=duplicate_refs,
        rejected=rejected,
    )


def _build_main_holes(chosen: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    rows = []
    geoms = []
    for _, row in chosen.sort_values("_hole_number").iterrows():
        rows.append({
            "hole_number": int(row["_hole_number"]),
            "name": _get(row, "name"),
            "par": _get(row, "par"),
            "handicap": _get(row, "handicap"),
        })
        geoms.append(ensure_linestring(row.geometry))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=crs)
    return gdf.reset_index(drop=True)


def hole_lines_map(main_holes: gpd.GeoDataFrame) -> dict[int, Any]:
    """{hole_number: centerline LineString} for nearest-hole assignment."""
    return {
        int(r["hole_number"]): ensure_linestring(r.geometry)
        for _, r in main_holes.iterrows()
    }


def _get(row, col):
    if col in row.index:
        v = row[col]
        if v is None:
            return None
        try:
            if isinstance(v, float) and math.isnan(v):
                return None
        except TypeError:
            pass
        return v
    return None


def _empty_like(feats: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"hole_number": [], "name": [], "par": [], "handicap": []},
        geometry=[], crs=feats.crs,
    )
