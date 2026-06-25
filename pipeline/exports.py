"""Aggregate, cross-course exports into courses/_index/.

Reads each course's manifest + per-hole terrain summaries and point parquet,
producing:
  all_holes.csv / all_holes.parquet        — one row per processed hole
  all_hole_points.parquet                  — every hole's points concatenated
  all_courses_manifest.json                — index of course manifests
  golf.duckdb                              — DuckDB views over the parquet files
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .logging_config import get_logger
from .paths import COURSES_ROOT, CoursePaths, HolePaths, IndexPaths
from .storage import duckdb_writer, json_io, parquet_io

log = get_logger("exports")

_PREFERRED_COLUMNS = (
    "course_slug", "course_name", "country", "hole_buffer_meters",
    "hole_id", "hole_number", "hole_name", "par", "handicap",
    "hole_length_m", "hole_length_yd",
    "tee_elevation_m", "green_elevation_m",
    "net_elevation_change_m", "abs_elevation_change_m",
    "min_elevation_m", "max_elevation_m", "mean_elevation_m", "elevation_range_m",
    "avg_slope_deg", "max_slope_deg", "avg_slope_percent", "max_slope_percent",
    "dem_type", "dem_source", "raster_resolution_m",
    "tee_selection_method", "green_selection_method", "quality_flags",
)


def _iter_course_manifests(courses_root: Path):
    for manifest_path in sorted(courses_root.glob("*/course_manifest.json")):
        try:
            yield manifest_path, json_io.read_json(manifest_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable manifest %s (%s)", manifest_path, exc)


def collect_hole_rows(courses_root: Path = COURSES_ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path, manifest in _iter_course_manifests(courses_root):
        if manifest.get("status") != "processed":
            continue
        course_root = manifest_path.parent
        course_extras = {
            "country": manifest.get("country"),
            "hole_buffer_meters": manifest.get("hole_buffer_meters"),
            "course_name": manifest.get("course_name"),
        }
        for hole in manifest.get("holes", []):
            sp = hole.get("summary_path")
            if not sp:
                continue
            summary_file = course_root / sp
            if not summary_file.exists():
                continue
            try:
                summary = json_io.read_json(summary_file)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping summary %s (%s)", summary_file, exc)
                continue
            summary.pop("schema_version", None)
            if isinstance(summary.get("quality_flags"), list):
                summary["quality_flags"] = ";".join(summary["quality_flags"])
            rows.append({**summary, **course_extras})
    return rows


def _to_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, dict)):
        return json_io.json.dumps(value, ensure_ascii=False)
    return str(value)


def write_all_holes_csv(rows: list[dict], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(_PREFERRED_COLUMNS)
    seen = set(columns)
    for r in rows:
        for k in r:
            if k not in seen:
                columns.append(k)
                seen.add(k)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _to_csv_value(r.get(c)) for c in columns})
    log.info("wrote %d hole rows -> %s", len(rows), out_path)
    return out_path


def _ordered_rows(rows: list[dict]) -> list[dict]:
    """Reorder each row dict so preferred columns come first (for parquet)."""
    out = []
    for r in rows:
        ordered = {c: r.get(c) for c in _PREFERRED_COLUMNS if c in r}
        for k, v in r.items():
            if k not in ordered:
                ordered[k] = v
        out.append(ordered)
    return out


def build_aggregate_index(
    courses_root: Path = COURSES_ROOT,
    write_parquet: bool = True,
    write_duckdb: bool = True,
) -> dict[str, str]:
    index = IndexPaths.for_root(courses_root)
    index.ensure()
    written: dict[str, str] = {}

    rows = collect_hole_rows(courses_root)
    write_all_holes_csv(rows, index.all_holes_csv)
    written["all_holes_csv"] = str(index.all_holes_csv)

    if write_parquet and parquet_io.parquet_available():
        p = parquet_io.write_rows_parquet(_ordered_rows(rows), index.all_holes_parquet)
        if p:
            written["all_holes_parquet"] = str(p)
        # Concatenate per-hole point parquet files.
        point_files: list[Path] = []
        for manifest_path, manifest in _iter_course_manifests(courses_root):
            if manifest.get("status") != "processed":
                continue
            cp = CoursePaths.for_slug(manifest["course_slug"], courses_root=courses_root)
            for hole in manifest.get("holes", []):
                hn = hole.get("hole_number")
                if hn is None:
                    continue
                pq = HolePaths.for_hole(cp, int(hn)).hole_points_parquet
                if pq.exists():
                    point_files.append(pq)
        if point_files:
            dest = parquet_io.concat_parquet(point_files, index.all_hole_points_parquet)
            if dest:
                written["all_hole_points_parquet"] = str(dest)

    # Aggregate course manifest index.
    manifests = [m for _, m in _iter_course_manifests(courses_root)]
    json_io.save_json(
        {"schema_version": "1.0.0", "courses": [
            {k: m.get(k) for k in ("course_slug", "course_name", "status",
                                   "detected_holes", "processed_holes",
                                   "expected_holes", "dem_type")}
            for m in manifests
        ]},
        index.all_courses_manifest,
    )
    written["all_courses_manifest"] = str(index.all_courses_manifest)

    if write_duckdb and duckdb_writer.duckdb_available():
        db = duckdb_writer.build_database(
            index.duckdb,
            holes_parquet=index.all_holes_parquet if index.all_holes_parquet.exists() else None,
            points_parquet=(index.all_hole_points_parquet
                            if index.all_hole_points_parquet.exists() else None),
        )
        if db:
            written["duckdb"] = str(db)

    log.info("aggregate index written: %s", list(written))
    return written
