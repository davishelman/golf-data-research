"""Stage 8 — generate the per-hole tee-relative labeled 3D point cloud.

Deterministic raster-grid sampling: every DEM cell center inside the hole buffer
becomes a point, classified by the priority labeler, then expressed in absolute,
tee-relative, and (optionally) tee->green-aligned coordinates.

Artifacts written:
  features/hole_points.jsonl          — one JSON record per point (streamed)
  features/hole_points_compact.json   — compact [x, y, z_rel, label_id] arrays
  features/hole_points.parquet        — columnar (if pyarrow available)
  features/label_map.json             — id -> label name
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import geopandas as gpd
import numpy as np
import shapely

from ..constants import LABEL_MAP_JSON, SCHEMA_VERSION
from ..logging_config import get_logger
from ..schemas import HoleAnchors, HoleIdentity, RunOptions
from ..storage import json_io, parquet_io
from .labels import classify_points
from .transforms import aligned_arrays, alignment_angle, relative_arrays
from ..raster.sampling import cell_center_points, raster_resolution_m, stride_for_resolution

log = get_logger("features.point_cloud")


@dataclass
class PointCloudResult:
    num_points: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    point_limit_reached: bool = False
    sampling_resolution_m: float = 0.0
    stride: int = 1
    jsonl_path: Optional[str] = None
    compact_path: Optional[str] = None
    parquet_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_points": self.num_points,
            "label_counts": self.label_counts,
            "point_limit_reached": self.point_limit_reached,
            "sampling_resolution_m": self.sampling_resolution_m,
            "stride": self.stride,
        }


def _round(v, nd=3):
    if v is None:
        return None
    f = float(v)
    if not math.isfinite(f):
        return None
    return round(f, nd)


def generate_point_cloud(
    identity: HoleIdentity,
    dem: np.ndarray,
    transform,
    dem_crs,
    buffer_geom,
    anchors: HoleAnchors,
    layers: dict[str, gpd.GeoDataFrame],
    hole_paths,  # HolePaths
    options: RunOptions,
) -> PointCloudResult:
    hole_paths.ensure()
    json_io.save_json(LABEL_MAP_JSON, hole_paths.label_map)

    res_m = raster_resolution_m(transform)
    stride = stride_for_resolution(transform, options.point_sampling_resolution_m)
    result = PointCloudResult(
        sampling_resolution_m=round(res_m * stride, 3), stride=stride
    )

    xs, ys, zs = cell_center_points(dem, transform, stride)
    if xs.size == 0:
        _write_empty(identity, anchors, hole_paths, options, result)
        return result

    # Keep only cell centers inside the hole buffer (vectorized contains).
    pts = shapely.points(xs, ys)
    inside = shapely.contains(buffer_geom, pts)
    xs, ys, zs, pts = xs[inside], ys[inside], zs[inside], pts[inside]
    if xs.size == 0:
        _write_empty(identity, anchors, hole_paths, options, result)
        return result

    # Guardrail: cap points per hole with deterministic even subsampling.
    if xs.size > options.max_points_per_hole:
        sel = np.unique(np.linspace(0, xs.size - 1, options.max_points_per_hole).astype(int))
        xs, ys, zs, pts = xs[sel], ys[sel], zs[sel], pts[sel]
        result.point_limit_reached = True
        log.warning("hole %s reached max_points_per_hole=%d",
                    identity.hole_id, options.max_points_per_hole)

    points_gdf = gpd.GeoDataFrame(geometry=pts, crs=dem_crs)
    labels, label_ids, sources, confidences = classify_points(
        points_gdf, layers, options.infer_rough_from_background
    )

    # Coordinate transforms.
    tee = anchors.tee_point
    tee_z = anchors.tee_elevation_m
    x_rel, y_rel, z_rel = relative_arrays(xs, ys, zs, tee.x, tee.y, tee_z)
    if options.enable_aligned_coordinates:
        angle = alignment_angle(tee.x, tee.y, anchors.green_point.x, anchors.green_point.y)
        x_al, y_al = aligned_arrays(x_rel, y_rel, angle)
    else:
        angle = 0.0
        x_al = y_al = None

    # Stream JSONL + accumulate compact + parquet columns.
    label_counts: dict[str, int] = {}
    compact_points: list[list] = []
    parquet_records: list[dict] = [] if (options.write_parquet and parquet_io.parquet_available()) else []

    with json_io.JsonlWriter(hole_paths.hole_points_jsonl) as writer:
        for i in range(xs.size):
            lab = labels[i]
            label_counts[lab] = label_counts.get(lab, 0) + 1
            xa = _round(x_al[i]) if x_al is not None else None
            ya = _round(y_al[i]) if y_al is not None else None
            rec = {
                "hole_id": identity.hole_id,
                "point_id": i,
                "x_abs_m": _round(xs[i]),
                "y_abs_m": _round(ys[i]),
                "z_abs_m": _round(zs[i]),
                "x_rel_m": _round(x_rel[i]),
                "y_rel_m": _round(y_rel[i]),
                "z_rel_m": _round(z_rel[i]),
                "x_aligned_m": xa,
                "y_aligned_m": ya,
                "label": lab,
                "label_id": int(label_ids[i]),
                "source": sources[i],
                "confidence": round(float(confidences[i]), 3),
            }
            if options.write_jsonl:
                writer.write(rec)
            if options.write_compact_json:
                cx = xa if xa is not None else _round(x_rel[i])
                cy = ya if ya is not None else _round(y_rel[i])
                compact_points.append([cx, cy, _round(z_rel[i]), int(label_ids[i])])
            if parquet_records is not None and options.write_parquet:
                parquet_records.append(rec)

    result.num_points = int(xs.size)
    result.label_counts = label_counts
    if options.write_jsonl:
        result.jsonl_path = str(hole_paths.hole_points_jsonl)

    if options.write_compact_json:
        _write_compact(identity, anchors, angle, compact_points, hole_paths)
        result.compact_path = str(hole_paths.hole_points_compact)

    if options.write_parquet and parquet_io.parquet_available() and parquet_records:
        p = parquet_io.write_points_parquet(parquet_records, hole_paths.hole_points_parquet)
        if p is not None:
            result.parquet_path = str(p)

    log.info("hole %s: %d points %s", identity.hole_id, result.num_points, label_counts)
    return result


def _write_compact(identity, anchors, angle, compact_points, hole_paths) -> None:
    tee = anchors.tee_point
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hole_id": identity.hole_id,
        "course_slug": identity.course_slug,
        "hole_number": identity.hole_number,
        "coordinate_system": "tee_relative_aligned_meters",
        "origin": {
            "type": "selected_tee_anchor",
            "x_abs_m": _round(tee.x),
            "y_abs_m": _round(tee.y),
            "z_abs_m": _round(anchors.tee_elevation_m),
        },
        "alignment": {
            "enabled": True,
            "axis": "+Y_toward_green",
            "rotation_degrees": round(math.degrees(angle), 3),
        },
        "label_map": LABEL_MAP_JSON,
        "points": compact_points,
    }
    json_io.save_json(payload, hole_paths.hole_points_compact, indent=0)


def _write_empty(identity, anchors, hole_paths, options, result) -> None:
    """Write empty-but-valid artifacts so downstream readers never choke."""
    with json_io.JsonlWriter(hole_paths.hole_points_jsonl):
        pass
    if options.write_compact_json:
        _write_compact(identity, anchors, 0.0, [], hole_paths)
    log.warning("hole %s: no points generated (empty DEM/buffer)", identity.hole_id)
