"""Dedicated per-surface point-cloud artifact loader for v2.5 similarity.

This is a second implementation of the
:class:`~pipeline.modeling.pointcloud.export_similarity.PointCloudArtifactLoader`
protocol, reading from a *purpose-built* per-surface artifact rather than the
existing compact per-hole JSON clouds. :class:`CompactArtifactLoader` is left
untouched and remains the default; this loader is additive and the scorer is
unchanged.

Artifact schema
---------------
Two tables (Parquet or CSV, chosen by file extension):

* **points** — one row per normalized surface point:

  ====================  =======  ====================================
  column                dtype    meaning
  ====================  =======  ====================================
  ``hole_id``           str      ``"{course_slug}:{hole_number}"``
  ``surface``           str      one of :data:`KNOWN_SURFACES`
  ``x_lateral_m``       float    signed lateral offset from tee->green axis
  ``y_down_hole_m``     float    distance down the hole from the tee
  ``z_relative_m``      float    elevation relative to the tee anchor
  ``point_weight``      float    per-point weight (default 1.0)
  ====================  =======  ====================================

* **metadata** — one row per hole, matching :class:`HoleMetadata` fields:
  ``hole_id, course_slug, hole_number, par, yards`` (required) and optional
  ``course_name, tee_elevation_m, green_elevation_m`` plus optional explicit
  ``has_tee/has_green/has_fairway/has_bunker/has_water`` flags. When the ``has_*``
  flags are absent they are derived from which surfaces actually appear in the
  points table.

The companion :func:`export_surface_artifact` builds this artifact from any
existing loader (e.g. :class:`CompactArtifactLoader`), giving a concrete path
from today's compact clouds to the dedicated store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ...logging_config import get_logger
from .schemas import KNOWN_SURFACES, HoleMetadata, SurfacePoint

log = get_logger("modeling.pointcloud.surface_loader")

#: Required + optional columns of the points table.
SURFACE_POINT_COLUMNS: tuple[str, ...] = (
    "hole_id", "surface", "x_lateral_m", "y_down_hole_m", "z_relative_m", "point_weight",
)

#: Required metadata columns; ``has_*`` and elevation/name are optional.
METADATA_REQUIRED_COLUMNS: tuple[str, ...] = (
    "hole_id", "course_slug", "hole_number", "par", "yards",
)

_HAS_FLAGS = {
    "tee": "has_tee", "green": "has_green", "fairway": "has_fairway",
    "bunker": "has_bunker", "water": "has_water",
}


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"artifact table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported artifact extension {path.suffix!r} (use .parquet or .csv)")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported artifact extension {path.suffix!r} (use .parquet or .csv)")


class SurfacePointArtifactLoader:
    """Loader over a dedicated per-surface point-cloud artifact (points + metadata).

    Both tables are read once and indexed; ``load_points`` then serves each hole
    from memory. Implements the ``PointCloudArtifactLoader`` protocol structurally
    (``load_metadata`` + ``load_points``).
    """

    def __init__(self, points_path: Path, metadata_path: Path) -> None:
        self.points_path = Path(points_path)
        self.metadata_path = Path(metadata_path)
        self._points_by_hole: Optional[dict[str, list[SurfacePoint]]] = None
        self._surfaces_by_hole: dict[str, set[str]] = {}

    # -- internal ----------------------------------------------------------- #

    def _ensure_points(self) -> dict[str, list[SurfacePoint]]:
        if self._points_by_hole is not None:
            return self._points_by_hole
        df = _read_table(self.points_path)
        missing = set(SURFACE_POINT_COLUMNS[:-1]) - set(df.columns)  # point_weight optional
        if missing:
            raise KeyError(f"points table missing required column(s): {sorted(missing)}")
        if "point_weight" not in df.columns:
            df = df.assign(point_weight=1.0)

        by_hole: dict[str, list[SurfacePoint]] = {}
        surfaces: dict[str, set[str]] = {}
        for row in df.itertuples(index=False):
            surface = str(row.surface)
            if surface not in KNOWN_SURFACES:
                continue  # ignore non-modeled surfaces defensively
            hole_id = str(row.hole_id)
            by_hole.setdefault(hole_id, []).append(SurfacePoint(
                hole_id=hole_id, surface=surface,
                x_lateral_m=float(row.x_lateral_m),
                y_down_hole_m=float(row.y_down_hole_m),
                z_relative_m=float(row.z_relative_m),
                point_weight=float(row.point_weight),
            ))
            surfaces.setdefault(hole_id, set()).add(surface)
        self._points_by_hole = by_hole
        self._surfaces_by_hole = surfaces
        return by_hole

    # -- contract ----------------------------------------------------------- #

    def load_metadata(self) -> dict[str, HoleMetadata]:
        self._ensure_points()
        df = _read_table(self.metadata_path)
        missing = set(METADATA_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise KeyError(f"metadata table missing required column(s): {sorted(missing)}")

        out: dict[str, HoleMetadata] = {}
        for row in df.itertuples(index=False):
            hole_id = str(row.hole_id)
            present = self._surfaces_by_hole.get(hole_id, set())

            def flag(surface: str) -> bool:
                col = _HAS_FLAGS[surface]
                if hasattr(row, col) and not pd.isna(getattr(row, col)):
                    return bool(getattr(row, col))
                return surface in present

            out[hole_id] = HoleMetadata(
                hole_id=hole_id,
                course_slug=str(row.course_slug),
                course_name=_opt_str(getattr(row, "course_name", None)),
                hole_number=int(row.hole_number),
                par=int(row.par),
                yards=float(row.yards),
                has_tee=flag("tee"), has_green=flag("green"), has_fairway=flag("fairway"),
                has_bunker=flag("bunker"), has_water=flag("water"),
                tee_elevation_m=_opt_float(getattr(row, "tee_elevation_m", None)),
                green_elevation_m=_opt_float(getattr(row, "green_elevation_m", None)),
            )
        return out

    def load_points(self, hole_id: str) -> list[SurfacePoint]:
        return list(self._ensure_points().get(hole_id, []))


def _opt_float(v) -> Optional[float]:
    if v is None or pd.isna(v):
        return None
    return float(v)


def _opt_str(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return str(v)


# --------------------------------------------------------------------------- #
# Writer: build the dedicated artifact from any existing loader
# --------------------------------------------------------------------------- #

def export_surface_artifact(loader, points_path: Path, metadata_path: Path) -> dict[str, int]:
    """Materialize a dedicated per-surface artifact from ``loader``.

    Iterates the loader's metadata, pulls each hole's points, and writes the
    points + metadata tables (Parquet or CSV by extension). Returns
    ``{"holes": n, "points": n}``. The source loader is read-only.
    """
    metadata = loader.load_metadata()

    point_rows: list[dict] = []
    for hole_id in sorted(metadata):
        for p in loader.load_points(hole_id):
            point_rows.append({
                "hole_id": p.hole_id, "surface": p.surface,
                "x_lateral_m": p.x_lateral_m, "y_down_hole_m": p.y_down_hole_m,
                "z_relative_m": p.z_relative_m, "point_weight": p.point_weight,
            })
    points_df = pd.DataFrame(point_rows, columns=list(SURFACE_POINT_COLUMNS))

    meta_rows: list[dict] = []
    for hole_id in sorted(metadata):
        m = metadata[hole_id]
        meta_rows.append({
            "hole_id": m.hole_id, "course_slug": m.course_slug,
            "course_name": m.course_name, "hole_number": m.hole_number,
            "par": m.par, "yards": m.yards,
            "has_tee": m.has_tee, "has_green": m.has_green, "has_fairway": m.has_fairway,
            "has_bunker": m.has_bunker, "has_water": m.has_water,
            "tee_elevation_m": m.tee_elevation_m, "green_elevation_m": m.green_elevation_m,
        })
    metadata_df = pd.DataFrame(meta_rows)

    _write_table(points_df, points_path)
    _write_table(metadata_df, metadata_path)
    log.info("surface artifact: %d holes, %d points -> %s, %s",
             len(metadata), len(points_df), points_path, metadata_path)
    return {"holes": len(metadata), "points": len(points_df)}


__all__ = [
    "SurfacePointArtifactLoader",
    "export_surface_artifact",
    "SURFACE_POINT_COLUMNS",
    "METADATA_REQUIRED_COLUMNS",
]
