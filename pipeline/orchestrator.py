"""Course orchestration — readable, staged pipeline.

    fetch -> boundary -> detect/validate holes -> feature layers -> course DEM
    -> per hole: assign features, clip DEM, terrain stats, anchors, point cloud,
       (optional) plots
    -> course manifest + quality report (+ legacy summary)

Per-hole and per-course failures are isolated and recorded in the quality report
rather than crashing the batch. Dirty courses (e.g. != expected holes) are
skipped with an explicit report unless ``--allow-dirty``.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import geopandas as gpd
import numpy as np

from .config import CourseConfig
from .constants import FEATURE_LAYERS, LAYER_HOLE_CENTERLINES, SCHEMA_VERSION
from .geometry import ensure_linestring
from .logging_config import get_logger
from .paths import CoursePaths, HolePaths
from .quality import (
    CODE_COARSE_DEM_RESOLUTION,
    CODE_DEM_DOWNLOAD_FAILED,
    CODE_EXPECTED_HOLES_MISMATCH,
    CODE_GREEN_ELEVATION_NAN,
    CODE_HOLE_PROCESSING_FAILED,
    CODE_LOW_DEM_COVERAGE,
    CODE_NO_COURSE_BOUNDARY,
    CODE_NO_HOLE_FEATURES,
    CODE_PLOTTING_FAILED,
    CODE_POINT_LIMIT_REACHED,
    CODE_TEE_ELEVATION_NAN,
    QualityReport,
)
from .schemas import HoleIdentity, RunOptions, make_hole_id
from .storage import json_io
from .terrain import build_terrain_summary

from .osm.assignment import AssignmentSummary, assign_layer_to_hole
from .osm.boundary import select_boundary
from .osm.fetch import fetch_osm_source
from .osm.holes import detect_main_holes, hole_lines_map
from .osm.layers import (
    build_feature_layers,
    load_source_layers,
    save_source_layers,
    source_layers_exist,
)
from .raster.clip import clip_dem_to_buffer, reproject_raster_to_crs
from .raster.dem import DemDownloadError, download_course_dem
from .raster.sampling import raster_resolution_m, read_dem_masked
from .raster.slope import compute_slope
from .features.anchors import select_anchors
from .features.point_cloud import generate_point_cloud

log = get_logger("orchestrator")


@dataclass
class CourseRunResult:
    course_slug: str
    status: str
    processed_holes: int = 0
    expected_holes: int = 0
    manifest_path: Optional[str] = None
    quality_path: Optional[str] = None
    message: str = ""
    per_hole: list[dict] = field(default_factory=list)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_course(course: CourseConfig, options: RunOptions) -> CourseRunResult:
    courses_root = options.courses_root
    paths = CoursePaths.for_slug(course.course_slug, courses_root=courses_root)
    paths.ensure()
    report = QualityReport(course.course_slug)

    log.info("=== course start: %s (%s)", course.course_name, course.course_slug)

    # ---- Source: fetch or load from cache ----------------------------------
    try:
        layers, main_holes, boundary_gdf = _get_source(course, paths, options, report)
    except _SkipCourse as skip:
        return _write_skipped(course, paths, report, skip.message)
    except Exception as exc:  # noqa: BLE001
        log.exception("source stage failed for %s", course.course_slug)
        report.error(CODE_NO_HOLE_FEATURES, f"source stage failed: {exc}")
        return _write_skipped(course, paths, report, str(exc), status="failed")

    projected_crs = main_holes.crs
    hole_lines = hole_lines_map(main_holes)

    # ---- Course DEM --------------------------------------------------------
    dem_type = course.resolve_dem_type()
    course_dem_path = paths.course_dem()
    dem_source = "local"
    if not options.only_plots:
        try:
            course_dem_path, dem_source = _ensure_course_dem(
                course, main_holes, projected_crs, course_dem_path, dem_type, options
            )
        except DemDownloadError as exc:
            report.error(CODE_DEM_DOWNLOAD_FAILED, str(exc))
            return _write_skipped(course, paths, report, str(exc), status="failed")

    # ---- Per-hole loop -----------------------------------------------------
    per_hole: list[dict] = []
    for _, hole_row in main_holes.sort_values("hole_number").iterrows():
        hole_number = int(hole_row["hole_number"])
        try:
            result = _process_hole(
                course, paths, hole_row, projected_crs, layers, hole_lines,
                course_dem_path, dem_type, dem_source, options, report,
            )
            per_hole.append(result)
        except Exception as exc:  # noqa: BLE001
            log.exception("hole %d failed for %s", hole_number, course.course_slug)
            report.error(
                CODE_HOLE_PROCESSING_FAILED, f"hole {hole_number} failed: {exc}",
                hole_id=make_hole_id(course.course_slug, hole_number),
            )

    if not per_hole:
        return _write_skipped(course, paths, report,
                              "no holes processed", status="failed")

    # ---- Course-wide plot --------------------------------------------------
    if not options.skip_plots:
        try:
            from . import plotting
            plotting.plot_course_overview(per_hole, course.course_name, paths.course_overview)
        except Exception as exc:  # noqa: BLE001
            report.warning(CODE_PLOTTING_FAILED, f"course overview failed: {exc}")

    # ---- Manifests + summaries --------------------------------------------
    manifest = _build_manifest(course, paths, projected_crs, dem_type, dem_source,
                               len(main_holes), per_hole, report)
    json_io.save_json(manifest, paths.manifest)
    json_io.save_json(_legacy_course_summary(course, dem_type, dem_source, per_hole),
                      paths.course_summary)
    json_io.save_json(report.to_dict(), paths.quality_report)

    log.info("=== course done: %s (%d/%d holes)",
             course.course_slug, len(per_hole), len(main_holes))
    return CourseRunResult(
        course_slug=course.course_slug,
        status="processed",
        processed_holes=len(per_hole),
        expected_holes=len(main_holes),
        manifest_path=str(paths.manifest),
        quality_path=str(paths.quality_report),
        per_hole=per_hole,
    )


# ---------------------------------------------------------------------------
# Source stage
# ---------------------------------------------------------------------------


class _SkipCourse(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _get_source(course, paths, options, report):
    """Return (layers, main_holes, boundary_gdf), fetching or loading cache."""
    if source_layers_exist(paths) and not options.refetch_osm:
        log.info("using cached source layers at %s", paths.source)
        layers, main_holes, boundary_gdf = load_source_layers(paths)
        if "hole_number" in main_holes.columns and not main_holes.empty:
            detected = main_holes["hole_number"].nunique()
            if options.strict_18 and detected != course.holes_count:
                raise _SkipCourse(
                    f"cached source has {detected} holes, expected {course.holes_count}"
                )
            return layers, main_holes, boundary_gdf
        raise _SkipCourse("cached source layers are unusable; re-run with --refetch-osm")

    source = fetch_osm_source(course)

    try:
        boundary = select_boundary(course, source)
    except ValueError as exc:
        report.error(CODE_NO_COURSE_BOUNDARY, str(exc))
        raise _SkipCourse(str(exc))
    json_io.save_json(boundary.to_dict(), paths.boundary_selection)

    detection = detect_main_holes(course, source, boundary)
    if detection.duplicate_refs:
        report.warning(
            "DUPLICATE_HOLE_REFS",
            f"duplicate hole refs resolved: {detection.duplicate_refs}",
            details=detection.to_dict(),
        )
    if options.strict_18 and not detection.is_clean:
        report.error(
            CODE_EXPECTED_HOLES_MISMATCH,
            f"expected {detection.expected} holes, detected {detection.detected}",
            details=detection.to_dict(),
        )
        raise _SkipCourse(
            f"expected {detection.expected} holes, detected {detection.detected}"
        )
    if not detection.is_clean:
        report.warning(
            CODE_EXPECTED_HOLES_MISMATCH,
            f"processing dirty course (allow-dirty): detected {detection.detected}",
            details=detection.to_dict(),
        )

    main_holes = detection.main_holes
    if main_holes.empty:
        raise _SkipCourse("no usable hole centerlines detected")

    layers = build_feature_layers(source, boundary)
    save_source_layers(paths, layers, main_holes, boundary)
    return layers, main_holes, boundary.boundary


# ---------------------------------------------------------------------------
# DEM stage
# ---------------------------------------------------------------------------


def _ensure_course_dem(course, main_holes, projected_crs, dem_path, dem_type, options):
    buffered = main_holes.copy()
    buffered["geometry"] = buffered.geometry.apply(
        lambda g: ensure_linestring(g).buffer(course.hole_buffer_meters)
    )
    bounds = tuple(buffered.total_bounds)
    return download_course_dem(
        bounds_projected=bounds, src_crs=projected_crs, dem_path=dem_path,
        dem_type=dem_type, force=options.redownload_dem,
    )


# ---------------------------------------------------------------------------
# Per-hole stage
# ---------------------------------------------------------------------------


def _process_hole(course, paths, hole_row, projected_crs, layers, hole_lines,
                  course_dem_path, dem_type, dem_source, options, report) -> dict:
    hole_number = int(hole_row["hole_number"])
    identity = HoleIdentity.build(
        course.course_slug, hole_number,
        par=_clean_int(hole_row.get("par")),
        handicap=_clean_int(hole_row.get("handicap")),
        name=_clean_str(hole_row.get("name")),
    )
    hp = HolePaths.for_hole(paths, hole_number)
    hp.ensure()

    hole_line = ensure_linestring(hole_row.geometry)
    buffer_geom = hole_line.buffer(course.hole_buffer_meters)
    log.info("[%s] '%s' par=%s length=%.1f m",
             identity.hole_id, identity.name or "", identity.par, hole_line.length)

    # ---- Assign + clip per-hole feature layers ----
    assign_summary = AssignmentSummary()
    assigned: dict[str, gpd.GeoDataFrame] = {}
    for layer in FEATURE_LAYERS:
        clipped = assign_layer_to_hole(
            layer, layers.get(layer), hole_number, buffer_geom, hole_lines, assign_summary
        )
        assigned[layer] = clipped
        json_io.save_geojson(clipped, hp.vectors / f"{layer}.geojson")

    json_io.save_geojson(
        gpd.GeoDataFrame({"hole_number": [hole_number]}, geometry=[hole_line], crs=projected_crs),
        hp.vectors / f"{LAYER_HOLE_CENTERLINES}.geojson",
    )
    json_io.save_geojson(
        gpd.GeoDataFrame({"hole_number": [hole_number]}, geometry=[buffer_geom], crs=projected_crs),
        hp.vectors / "hole_buffer.geojson",
    )
    json_io.save_json(assign_summary.to_dict(), hp.assignment_report)

    # ---- DEM clip + reproject ----
    if not options.only_plots or not hp.projected_dem.exists():
        buffer_wgs84 = (
            gpd.GeoSeries([buffer_geom], crs=projected_crs).to_crs("EPSG:4326").iloc[0]
        )
        clip_dem_to_buffer(course_dem_path, buffer_wgs84, hp.clipped_dem)
        reproject_raster_to_crs(hp.clipped_dem, projected_crs, hp.projected_dem)

    dem, transform, dem_crs = read_dem_masked(hp.projected_dem)
    if not np.any(np.isfinite(dem)):
        raise RuntimeError("processed DEM has no valid pixels")
    slope_deg, slope_pct = compute_slope(dem, transform)
    res_m = raster_resolution_m(transform)

    # Coverage = finite pixels vs. pixels expected *inside the buffer* (not the
    # raster bbox), so diagonal corridors are not falsely flagged.
    finite_count = int(np.isfinite(dem).sum())
    expected_cells = max(buffer_geom.area / (res_m ** 2), 1.0)
    coverage = min(finite_count / expected_cells, 1.0)

    # ---- Anchors ----
    anchors = select_anchors(hole_line, assigned.get("tees"), assigned.get("greens"),
                             hp.projected_dem)
    json_io.save_json(anchors.to_dict(), hp.anchors)

    # ---- Quality flags ----
    flags: list[str] = []
    if anchors.tee_elevation_m is None or not np.isfinite(anchors.tee_elevation_m):
        flags.append(CODE_TEE_ELEVATION_NAN)
        report.warning(CODE_TEE_ELEVATION_NAN, "tee elevation NaN", hole_id=identity.hole_id)
    if anchors.green_elevation_m is None or not np.isfinite(anchors.green_elevation_m):
        flags.append(CODE_GREEN_ELEVATION_NAN)
        report.warning(CODE_GREEN_ELEVATION_NAN, "green elevation NaN", hole_id=identity.hole_id)
    if coverage < 0.5:
        flags.append(CODE_LOW_DEM_COVERAGE)
        report.warning(CODE_LOW_DEM_COVERAGE,
                       f"DEM coverage {coverage:.0%} of buffer", hole_id=identity.hole_id)
    if res_m > 2.0:
        flags.append(CODE_COARSE_DEM_RESOLUTION)

    # ---- Terrain summary ----
    summary = build_terrain_summary(
        identity, course.course_name, hole_line, dem, slope_deg, slope_pct,
        anchors, dem_type, dem_source, res_m, flags,
    )
    json_io.save_json(summary.to_dict(), hp.terrain_summary)

    # ---- Point cloud ----
    pc_result = None
    points_exist = hp.hole_points_jsonl.exists()
    if not options.only_plots and (options.rebuild_points or not points_exist):
        pc_result = generate_point_cloud(
            identity, dem, transform, dem_crs, buffer_geom, anchors,
            assigned, hp, options,
        )
        if pc_result.point_limit_reached:
            summary.quality_flags.append(CODE_POINT_LIMIT_REACHED)
            report.info(CODE_POINT_LIMIT_REACHED,
                        f"hole {hole_number} hit point cap", hole_id=identity.hole_id)
            json_io.save_json(summary.to_dict(), hp.terrain_summary)

    # ---- Plots ----
    if not options.skip_plots:
        try:
            from . import plotting
            distances, elevations = plotting.profile_arrays(hp.projected_dem, hole_line)
            plotting.render_hole_plots(
                dem, slope_deg, transform, dem_crs, assigned, hole_line,
                distances, elevations, summary.to_dict(), hp.plots,
            )
        except Exception as exc:  # noqa: BLE001
            report.warning(CODE_PLOTTING_FAILED,
                           f"hole {hole_number} plots failed: {exc}", hole_id=identity.hole_id)

    return {
        "identity": identity,
        "summary": summary,
        "dem": dem,
        "transform": transform,
        "dem_crs": dem_crs,
        "overlays": assigned,
        "hole_line": hole_line,
        "points": pc_result.to_dict() if pc_result else None,
        "points_path": (hp.relpath_from_course(paths, hp.hole_points_jsonl)
                        if hp.hole_points_jsonl.exists() else None),
        "summary_path": hp.relpath_from_course(paths, hp.terrain_summary),
    }


# ---------------------------------------------------------------------------
# Manifest / summary builders
# ---------------------------------------------------------------------------


def _build_manifest(course, paths, projected_crs, dem_type, dem_source,
                    detected, per_hole, report) -> dict[str, Any]:
    holes = []
    for d in per_hole:
        ident: HoleIdentity = d["identity"]
        holes.append({
            "hole_number": ident.hole_number,
            "hole_id": ident.hole_id,
            "status": "processed",
            "summary_path": d["summary_path"],
            "points_path": d["points_path"],
            "num_points": (d["points"]["num_points"] if d["points"] else None),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "course_slug": course.course_slug,
        "course_name": course.course_name,
        "status": "processed",
        "country": course.country,
        "par": course.par,
        "expected_holes": course.holes_count,
        "detected_holes": detected,
        "processed_holes": len(per_hole),
        "dem_type": dem_type,
        "dem_source": dem_source,
        "projected_crs": str(projected_crs),
        "hole_buffer_meters": course.hole_buffer_meters,
        "created_at_utc": _utc_now(),
        "quality_flags": report.flags,
        "holes": holes,
    }


def _legacy_course_summary(course, dem_type, dem_source, per_hole) -> dict[str, Any]:
    summaries = [d["summary"] for d in per_hole]
    return {
        "course": course.course_name,
        "course_slug": course.course_slug,
        "country": course.country,
        "holes_processed": [s.hole_number for s in summaries],
        "hole_buffer_meters": course.hole_buffer_meters,
        "dem_type": dem_type,
        "dem_source": dem_source,
        "total_length_m": round(sum(s.hole_length_m for s in summaries), 2),
        "total_length_yd": round(sum(s.hole_length_yd for s in summaries), 2),
        "total_par": sum(s.par for s in summaries if s.par is not None),
        "holes": [s.to_dict() for s in summaries],
    }


def _write_skipped(course, paths, report, message, status="skipped") -> CourseRunResult:
    report.status = status
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "course_slug": course.course_slug,
        "course_name": course.course_name,
        "status": status,
        "country": course.country,
        "expected_holes": course.holes_count,
        "detected_holes": 0,
        "processed_holes": 0,
        "reason": message,
        "created_at_utc": _utc_now(),
        "quality_flags": report.flags,
        "holes": [],
    }
    json_io.save_json(manifest, paths.manifest)
    json_io.save_json(report.to_dict(), paths.quality_report)
    log.warning("course %s %s: %s", course.course_slug, status, message)
    return CourseRunResult(
        course_slug=course.course_slug, status=status, processed_holes=0,
        expected_holes=course.holes_count, manifest_path=str(paths.manifest),
        quality_path=str(paths.quality_report), message=message,
    )


def _clean_int(v) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        s = str(v).strip()
        return int(float(s)) if s else None
    except Exception:  # noqa: BLE001
        return None


def _clean_str(v) -> Optional[str]:
    if v is None:
        return None
    try:
        if isinstance(v, float) and not np.isfinite(v):
            return None
    except TypeError:
        pass
    s = str(v).strip()
    return s or None
