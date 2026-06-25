"""Course configuration: the immutable input model + loaders.

`CourseConfig` is kept here (rather than in `schemas.py`) so that `schemas` can
re-export it without an import cycle. The JSON format in `config/courses.json` is
unchanged for backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


DEFAULT_HOLE_BUFFER_METERS: float = 100.0
DEFAULT_HOLES_COUNT: int = 18
DEFAULT_SEARCH_RADIUS_M: int = 1500
DEM_BY_COUNTRY: dict[str, str] = {
    "US": "USGS1m",
    # All other countries fall back to a global DEM.
}
DEFAULT_DEM_TYPE_GLOBAL: str = "COP30"


@dataclass(frozen=True)
class CourseConfig:
    course_slug: str
    course_name: str
    lat: float
    lon: float
    osm_relation_id: Optional[int] = None
    search_radius_m: int = DEFAULT_SEARCH_RADIUS_M
    country: Optional[str] = None
    dem_type: Optional[str] = None  # resolved via resolve_dem_type()
    par: Optional[int] = None
    holes_count: int = DEFAULT_HOLES_COUNT
    hole_buffer_meters: float = DEFAULT_HOLE_BUFFER_METERS
    extras: dict[str, Any] = field(default_factory=dict)  # any unknown keys

    @classmethod
    def from_dict(cls, d: dict) -> "CourseConfig":
        known = {
            "course_slug", "course_name", "lat", "lon",
            "osm_relation_id", "search_radius_m", "country", "dem_type",
            "par", "holes_count", "hole_buffer_meters",
        }
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            course_slug=d["course_slug"],
            course_name=d["course_name"],
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            osm_relation_id=_opt_int(d.get("osm_relation_id")),
            search_radius_m=int(d.get("search_radius_m") or DEFAULT_SEARCH_RADIUS_M),
            country=d.get("country"),
            dem_type=d.get("dem_type"),
            par=_opt_int(d.get("par")),
            holes_count=int(d.get("holes_count") or DEFAULT_HOLES_COUNT),
            hole_buffer_meters=float(d.get("hole_buffer_meters") or DEFAULT_HOLE_BUFFER_METERS),
            extras=extras,
        )

    def resolve_dem_type(self) -> str:
        """Return the effective DEM dataset name to request."""
        if self.dem_type:
            return self.dem_type
        if self.country and self.country.upper() in DEM_BY_COUNTRY:
            return DEM_BY_COUNTRY[self.country.upper()]
        return DEFAULT_DEM_TYPE_GLOBAL


def _opt_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_courses(path: Path) -> list[CourseConfig]:
    """Load a JSON file (list or ``{"courses": [...]}``) of course configs."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and "courses" in data:
        entries = data["courses"]
    else:
        raise ValueError(f"{path}: expected a list or an object with a 'courses' key.")
    return [CourseConfig.from_dict(e) for e in entries]


def find_course(courses: list[CourseConfig], slug: str) -> CourseConfig:
    for c in courses:
        if c.course_slug == slug:
            return c
    available = ", ".join(c.course_slug for c in courses)
    raise KeyError(f"Course '{slug}' not found in config. Available: {available}")
