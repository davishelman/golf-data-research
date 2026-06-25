"""Core data models for the pipeline.

Plain stdlib dataclasses (not Pydantic) so the models can hold shapely
geometries directly and stay import-light. Each model that becomes an artifact
exposes a ``to_dict`` producing JSON-ready output with a ``schema_version``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from shapely.geometry import LineString, Point

from .constants import LABEL_IDS, METERS_TO_YARDS, SCHEMA_VERSION

# Re-export CourseConfig from config so callers can `from .schemas import CourseConfig`.
from .config import CourseConfig  # noqa: F401  (intentional re-export)


def make_hole_id(course_slug: str, hole_number: int) -> str:
    """Stable hole identifier, e.g. ``augusta_national__01``."""
    return f"{course_slug}__{hole_number:02d}"


@dataclass(frozen=True)
class RunOptions:
    """All runtime knobs, mostly driven by CLI flags."""

    courses_root: Any = None  # pathlib.Path; Any to avoid importing Path here
    # Source refresh
    refetch_osm: bool = False
    redownload_dem: bool = False
    rebuild_points: bool = False
    # Validation
    strict_18: bool = True
    # Plotting
    skip_plots: bool = False
    only_plots: bool = False
    # Exports
    export_parquet: bool = False
    # Point-cloud generation
    point_sampling_resolution_m: float = 1.0
    max_points_per_hole: int = 250_000
    enable_aligned_coordinates: bool = True
    infer_rough_from_background: bool = True
    write_jsonl: bool = True
    write_compact_json: bool = True
    write_parquet: bool = True


@dataclass(frozen=True)
class HoleIdentity:
    course_slug: str
    hole_number: int
    hole_id: str
    par: Optional[int] = None
    handicap: Optional[int] = None
    name: Optional[str] = None

    @classmethod
    def build(cls, course_slug: str, hole_number: int, par=None,
              handicap=None, name=None) -> "HoleIdentity":
        return cls(
            course_slug=course_slug,
            hole_number=hole_number,
            hole_id=make_hole_id(course_slug, hole_number),
            par=par, handicap=handicap, name=name,
        )


@dataclass(frozen=True)
class HoleAnchors:
    """The tee and green anchors used as the origin / aim for transforms."""

    tee_point: Point
    green_point: Point
    centerline: LineString
    tee_elevation_m: float
    green_elevation_m: float
    tee_selection_method: str
    green_selection_method: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tee_xy": [self.tee_point.x, self.tee_point.y],
            "green_xy": [self.green_point.x, self.green_point.y],
            "tee_elevation_m": _r(self.tee_elevation_m),
            "green_elevation_m": _r(self.green_elevation_m),
            "tee_selection_method": self.tee_selection_method,
            "green_selection_method": self.green_selection_method,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class TerrainSummary:
    hole_id: str
    course_slug: str
    course_name: str
    hole_number: int
    hole_name: Optional[str]
    par: Optional[int]
    handicap: Optional[int]
    hole_length_m: float
    tee_elevation_m: Optional[float]
    green_elevation_m: Optional[float]
    net_elevation_change_m: Optional[float]
    abs_elevation_change_m: Optional[float]
    min_elevation_m: Optional[float]
    max_elevation_m: Optional[float]
    mean_elevation_m: Optional[float]
    elevation_range_m: Optional[float]
    avg_slope_deg: Optional[float]
    max_slope_deg: Optional[float]
    avg_slope_percent: Optional[float]
    max_slope_percent: Optional[float]
    dem_type: str
    dem_source: str
    raster_resolution_m: Optional[float]
    tee_selection_method: str
    green_selection_method: str
    quality_flags: list[str] = field(default_factory=list)

    @property
    def hole_length_yd(self) -> float:
        return round(self.hole_length_m * METERS_TO_YARDS, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "hole_id": self.hole_id,
            "course_slug": self.course_slug,
            "course_name": self.course_name,
            "hole_number": self.hole_number,
            "hole_name": self.hole_name,
            "par": self.par,
            "handicap": self.handicap,
            "hole_length_m": _r(self.hole_length_m),
            "hole_length_yd": self.hole_length_yd,
            "tee_elevation_m": _r(self.tee_elevation_m),
            "green_elevation_m": _r(self.green_elevation_m),
            "net_elevation_change_m": _r(self.net_elevation_change_m),
            "abs_elevation_change_m": _r(self.abs_elevation_change_m),
            "min_elevation_m": _r(self.min_elevation_m),
            "max_elevation_m": _r(self.max_elevation_m),
            "mean_elevation_m": _r(self.mean_elevation_m),
            "elevation_range_m": _r(self.elevation_range_m),
            "avg_slope_deg": _r(self.avg_slope_deg),
            "max_slope_deg": _r(self.max_slope_deg),
            "avg_slope_percent": _r(self.avg_slope_percent),
            "max_slope_percent": _r(self.max_slope_percent),
            "dem_type": self.dem_type,
            "dem_source": self.dem_source,
            "raster_resolution_m": _r(self.raster_resolution_m),
            "tee_selection_method": self.tee_selection_method,
            "green_selection_method": self.green_selection_method,
            "quality_flags": self.quality_flags,
        }

    def to_flat_row(self) -> dict[str, Any]:
        """Flattened dict for CSV / Parquet (hole_length_yd materialized)."""
        d = self.to_dict()
        d.pop("schema_version", None)
        d["hole_length_yd"] = self.hole_length_yd
        d["quality_flags"] = ";".join(self.quality_flags)
        return d


@dataclass(frozen=True)
class FeaturePoint:
    hole_id: str
    point_id: int
    x_abs_m: float
    y_abs_m: float
    z_abs_m: float
    x_rel_m: float
    y_rel_m: float
    z_rel_m: float
    x_aligned_m: Optional[float]
    y_aligned_m: Optional[float]
    label: str
    label_id: int
    source: str
    confidence: float

    def to_jsonl_record(self) -> dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "point_id": self.point_id,
            "x_abs_m": _r(self.x_abs_m),
            "y_abs_m": _r(self.y_abs_m),
            "z_abs_m": _r(self.z_abs_m),
            "x_rel_m": _r(self.x_rel_m),
            "y_rel_m": _r(self.y_rel_m),
            "z_rel_m": _r(self.z_rel_m),
            "x_aligned_m": _r(self.x_aligned_m),
            "y_aligned_m": _r(self.y_aligned_m),
            "label": self.label,
            "label_id": self.label_id,
            "source": self.source,
            "confidence": round(self.confidence, 3),
        }

    def to_compact(self) -> list[float | int]:
        """[x_aligned, y_aligned, z_rel, label_id] (falls back to rel xy)."""
        x = self.x_aligned_m if self.x_aligned_m is not None else self.x_rel_m
        y = self.y_aligned_m if self.y_aligned_m is not None else self.y_rel_m
        return [_r(x), _r(y), _r(self.z_rel_m), self.label_id]


def _r(v: Optional[float], ndigits: int = 3) -> Optional[float]:
    """Round, preserving None."""
    if v is None:
        return None
    try:
        return round(float(v), ndigits)
    except (TypeError, ValueError):
        return None


# Convenience for callers that only need the label id.
def label_id_for(label: str) -> int:
    return LABEL_IDS.get(label, LABEL_IDS["unknown"])
