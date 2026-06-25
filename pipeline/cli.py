"""CLI: read course config and run the staged pipeline for one or all courses.

Heavy imports (geopandas via orchestrator/exports, shapely via schemas) are
deferred into the branches that need them, so light-weight commands — notably
``--build-hole-features`` / ``--build-hole-similarity`` — run without the geo
stack installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parent.parent
    _env = _repo_root / ".env"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

from .config import find_course, load_courses
from .logging_config import configure_logging, get_logger
from .paths import COURSES_ROOT, DEFAULT_COURSES_JSON

log = get_logger("cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Run the golf course terrain + point-cloud pipeline.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_COURSES_JSON,
                   help=f"Path to courses JSON (default: {DEFAULT_COURSES_JSON})")
    p.add_argument("--courses-root", type=Path, default=COURSES_ROOT,
                   help=f"Output root for all course data (default: {COURSES_ROOT})")
    p.add_argument("--course", "-c", action="append", default=[],
                   help="Course slug to process. Repeatable.")
    p.add_argument("--all", action="store_true", help="Process every course in the config.")
    p.add_argument("--list", action="store_true", help="List config slugs and exit.")

    p.add_argument("--refetch-osm", action="store_true", help="Force re-fetch of OSM data.")
    p.add_argument("--redownload-dem", action="store_true", help="Force re-download of the DEM.")
    p.add_argument("--rebuild-points", action="store_true",
                   help="Regenerate point clouds even if they already exist.")

    strict = p.add_mutually_exclusive_group()
    strict.add_argument("--strict-18", dest="strict_18", action="store_true", default=True,
                        help="Skip courses without exactly the expected hole count (default).")
    strict.add_argument("--allow-dirty", dest="strict_18", action="store_false",
                        help="Process courses even if the hole count is off.")

    plots = p.add_mutually_exclusive_group()
    plots.add_argument("--skip-plots", action="store_true", help="Generate data without plots.")
    plots.add_argument("--only-plots", action="store_true",
                       help="Only render plots from existing per-hole data.")

    p.add_argument("--export-csv", action="store_true",
                   help="Build/refresh the aggregate CSV index.")
    p.add_argument("--export-parquet", action="store_true",
                   help="Also build Parquet + DuckDB aggregate exports.")

    # --- modeling (decoupled data-science phase) ---
    p.add_argument("--build-hole-features", action="store_true",
                   help="Build courses/_index/hole_features.parquet and exit.")
    p.add_argument("--build-hole-similarity", action="store_true",
                   help="Build hole clusters + similarity examples and exit.")
    p.add_argument("--clusters", type=int, default=8, help="Cluster count for similarity (default 8).")
    p.add_argument("--neighbors", type=int, default=10, help="Neighbors per hole (default 10).")
    p.add_argument("--no-umap", action="store_true", help="Skip UMAP embedding in similarity.")

    p.add_argument("--point-resolution", type=float, default=1.0,
                   help="Target point sampling resolution in meters (default 1.0).")
    p.add_argument("--max-points", type=int, default=250_000,
                   help="Max points per hole guardrail (default 250000).")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR.")
    return p


def _run_modeling(args) -> int:
    """Run the decoupled modeling steps. Returns a process exit code."""
    try:
        if args.build_hole_features:
            from .modeling.hole_feature_builder import build_hole_features
            path = build_hole_features(args.courses_root)
            log.info("hole features -> %s", path)
        if args.build_hole_similarity:
            from .modeling.export_similarity import build_hole_similarity
            written = build_hole_similarity(
                args.courses_root, n_clusters=args.clusters,
                n_neighbors=args.neighbors, use_umap=not args.no_umap,
            )
            for k, v in written.items():
                log.info("%s -> %s", k, v)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    return 0


def _run_processing(args, courses) -> int:
    from . import exports
    from .orchestrator import run_course
    from .schemas import RunOptions

    targets = courses if args.all else [find_course(courses, s) for s in args.course]
    options = RunOptions(
        courses_root=args.courses_root,
        refetch_osm=args.refetch_osm,
        redownload_dem=args.redownload_dem,
        rebuild_points=args.rebuild_points,
        strict_18=args.strict_18,
        skip_plots=args.skip_plots,
        only_plots=args.only_plots,
        export_parquet=args.export_parquet,
        point_sampling_resolution_m=args.point_resolution,
        max_points_per_hole=args.max_points,
    )
    log.info("processing %d course(s) into %s", len(targets), args.courses_root)

    processed = skipped = failed = 0
    failures: list[tuple[str, str]] = []
    for course in targets:
        try:
            result = run_course(course, options)
        except Exception as exc:  # noqa: BLE001
            log.exception("unhandled failure for %s", course.course_slug)
            failed += 1
            failures.append((course.course_slug, str(exc)))
            continue
        if result.status == "processed":
            processed += 1
        elif result.status == "skipped":
            skipped += 1
        else:
            failed += 1
            failures.append((course.course_slug, result.message))

    try:
        exports.build_aggregate_index(
            args.courses_root, write_parquet=args.export_parquet,
            write_duckdb=args.export_parquet,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("aggregate index build failed: %s", exc)

    log.info("complete: %d processed, %d skipped, %d failed", processed, skipped, failed)
    for slug, msg in failures:
        log.warning("  - %s: %s", slug, msg)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    # Modeling commands are decoupled from the geo stack and need no course config.
    if args.build_hole_features or args.build_hole_similarity:
        return _run_modeling(args)

    if not args.config.exists():
        log.error("config file not found: %s", args.config)
        return 2

    courses = load_courses(args.config)

    if args.list:
        for c in courses:
            tag = f"  ({c.country})" if c.country else ""
            print(f"  {c.course_slug:42s} {c.course_name}{tag}")
        return 0

    has_targets = bool(args.all or args.course)

    # Export-only invocation (no processing targets).
    if (args.export_csv or args.export_parquet) and not has_targets:
        from . import exports
        exports.build_aggregate_index(
            args.courses_root, write_parquet=args.export_parquet,
            write_duckdb=args.export_parquet,
        )
        return 0

    if not has_targets:
        parser.print_help()
        log.error("specify --course <slug> (repeatable), --all, or a --build-hole-* command")
        return 2

    return _run_processing(args, courses)


if __name__ == "__main__":
    raise SystemExit(main())
