"""Data contracts for the v2.5 point-cloud similarity layer.

These are deliberately small, pure dataclasses with no geo / sklearn / scipy
imports, so they stay importable and unit-testable from the light stack. The
scorer consumes *these* clean shapes — it never parses raw OSM, DEMs, or course
APIs (that is the job of the upstream geometry/DEM pipeline).

Coordinate convention (matches the existing tee-relative, green-aligned compact
point clouds written by :mod:`pipeline.features.point_cloud`):

* ``x_lateral_m``   — signed lateral offset from the tee->green axis (+ = right).
* ``y_down_hole_m`` — distance down the hole from the tee toward the green (+Y).
* ``z_relative_m``  — elevation relative to the tee anchor.

Identifier contract
-------------------
``hole_id = "{course_slug}:{hole_number}"`` (e.g. ``"augusta_national:13"``).
Note this is intentionally *distinct* from the v2 feature-table id
(``augusta_national__01``); v2.5 is an additive layer and owns its own id space.
Use :func:`make_pc_hole_id` / :func:`parse_pc_hole_id` rather than formatting by
hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: Canonical surface types compared by the v2.5 model, in a stable order.
KNOWN_SURFACES: tuple[str, ...] = ("fairway", "green", "bunker", "water", "tee")

#: Map from the existing point-cloud ``label`` names to v2.5 surface names. Labels
#: not present here (rough, trees, cartpath, sand, unknown) are ignored by v2.5.
LABEL_TO_SURFACE: dict[str, str] = {
    "fairway": "fairway",
    "green": "green",
    "bunker": "bunker",
    "water": "water",
    "tee": "tee",
}


def make_pc_hole_id(course_slug: str, hole_number: int) -> str:
    """Build a v2.5 hole id: ``"{course_slug}:{hole_number}"``."""
    return f"{course_slug}:{int(hole_number)}"


def parse_pc_hole_id(hole_id: str) -> tuple[str, int]:
    """Split a v2.5 hole id back into ``(course_slug, hole_number)``.

    Splits on the last ``":"`` so course slugs that themselves contain a colon
    are tolerated. Raises ``ValueError`` on a malformed id.
    """
    if ":" not in hole_id:
        raise ValueError(
            f"invalid v2.5 hole_id {hole_id!r}; expected 'course_slug:hole_number'"
        )
    slug, _, num = hole_id.rpartition(":")
    if not slug or not num.isdigit():
        raise ValueError(
            f"invalid v2.5 hole_id {hole_id!r}; expected 'course_slug:hole_number'"
        )
    return slug, int(num)


@dataclass(frozen=True)
class HoleMetadata:
    """Clean, model-ready metadata for a single hole.

    Carries only what candidate filtering and the penalty terms need — no
    geometry. The ``has_*`` flags let candidate filtering enforce
    ``required_surfaces`` without loading any point cloud.
    """

    hole_id: str
    course_slug: str
    hole_number: int
    par: int
    yards: float
    has_tee: bool = False
    has_green: bool = False
    has_fairway: bool = False
    has_bunker: bool = False
    has_water: bool = False
    course_name: Optional[str] = None
    tee_elevation_m: Optional[float] = None
    green_elevation_m: Optional[float] = None

    def has_surface(self, surface: str) -> bool:
        """Whether this hole has the named surface (per its ``has_*`` flags)."""
        flag = {
            "tee": self.has_tee,
            "green": self.has_green,
            "fairway": self.has_fairway,
            "bunker": self.has_bunker,
            "water": self.has_water,
        }.get(surface)
        if flag is None:
            raise ValueError(f"unknown surface {surface!r}; known: {KNOWN_SURFACES}")
        return bool(flag)


@dataclass(frozen=True)
class SurfacePoint:
    """A single normalized, surface-tagged point in tee-relative space.

    ``point_weight`` is reserved for future weighted-Chamfer experiments; the
    v1 Chamfer implementation treats every point equally and ignores it.
    """

    hole_id: str
    surface: str
    x_lateral_m: float
    y_down_hole_m: float
    z_relative_m: float
    point_weight: float = 1.0


@dataclass(frozen=True)
class SimilarityResult:
    """One target->candidate comparison. Stores IDs + scores only (no geometry).

    Lower ``total_score`` means *more* similar. Component scores are retained for
    explainability; surface component fields are ``None`` when that surface was
    not scored for the pair (both sides empty), and otherwise hold the surface's
    contribution (Chamfer distance when both sides have points, or the configured
    missing-surface penalty when exactly one side has them).
    """

    model_version: str
    config_name: str
    config_hash: str
    target_hole_id: str
    candidate_hole_id: str
    total_score: float
    yardage_penalty: float = 0.0
    elevation_penalty: float = 0.0
    missing_surface_penalty: float = 0.0
    filter_reason: str = "PASS"
    fairway_score: Optional[float] = None
    green_score: Optional[float] = None
    bunker_score: Optional[float] = None
    water_score: Optional[float] = None
    tee_score: Optional[float] = None
    #: Per-surface raw contributions keyed by surface name (full detail; the
    #: typed ``*_score`` fields above mirror the five known surfaces).
    surface_scores: dict[str, Optional[float]] = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flatten to a dict suitable for a DataFrame / CSV row (no geometry)."""
        return {
            "model_version": self.model_version,
            "config_name": self.config_name,
            "config_hash": self.config_hash,
            "target_hole_id": self.target_hole_id,
            "candidate_hole_id": self.candidate_hole_id,
            "total_score": self.total_score,
            "fairway_score": self.fairway_score,
            "green_score": self.green_score,
            "bunker_score": self.bunker_score,
            "water_score": self.water_score,
            "tee_score": self.tee_score,
            "yardage_penalty": self.yardage_penalty,
            "elevation_penalty": self.elevation_penalty,
            "missing_surface_penalty": self.missing_surface_penalty,
            "filter_reason": self.filter_reason,
        }


@dataclass(frozen=True)
class CandidateFilterResult:
    """Outcome of cheap pre-scoring candidate filtering."""

    passed: bool
    reason: str


__all__ = [
    "KNOWN_SURFACES",
    "LABEL_TO_SURFACE",
    "make_pc_hole_id",
    "parse_pc_hole_id",
    "HoleMetadata",
    "SurfacePoint",
    "SimilarityResult",
    "CandidateFilterResult",
]
