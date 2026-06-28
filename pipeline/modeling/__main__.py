"""Modeling CLI — decoupled from the geo pipeline (no geopandas/rasterio needed).

Usage:
    python -m pipeline.modeling features      # build hole_features.parquet/csv
    python -m pipeline.modeling similarity    # build clusters + similarity examples
    python -m pipeline.modeling all           # both, in order
    python -m pipeline.modeling visual-check --hole-id augusta_national__01 \
        --same-par --exclude-same-course --n 4   # save a side-by-side comparison PNG
    python -m pipeline.modeling hf-export --tier lite --output hf_artifact_lite
    python -m pipeline.modeling hf-export --tier full --output hf_artifact_full \
        --include-point-parquet      # build a Hugging Face upload folder
    python -m pipeline.modeling similarity-modes            # all golf modes -> CSVs
    python -m pipeline.modeling similarity-modes --mode off_the_tee
"""

from __future__ import annotations

import argparse

from ..logging_config import configure_logging, get_logger
from ..paths import COURSES_ROOT

log = get_logger("modeling.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.modeling",
        description="Build hole feature rows, find similar golf holes, and visually compare them.",
    )
    p.add_argument("command",
                   choices=["features", "similarity", "all", "visual-check",
                            "hf-export", "similarity-modes"],
                   help="Which step to run.")
    # similarity-modes options
    p.add_argument("--mode", default=None,
                   help="Single golf mode for 'similarity-modes' (default: all GOLF_MODES). "
                        "One of: overall_v2, off_the_tee, approach, green_complex, "
                        "hazard, terrain, shot_shape.")
    # hf-export options
    p.add_argument("--tier", choices=["lite", "full"], default="lite",
                   help="Hugging Face artifact tier (hf-export; default lite).")
    p.add_argument("--include-point-parquet", action="store_true",
                   help="hf-export full tier: also copy per-hole point Parquet files.")
    p.add_argument("--include-all-points", action="store_true",
                   help="hf-export full tier: also copy the ~1 GB all_hole_points.parquet.")
    p.add_argument("--courses-root", type=type(COURSES_ROOT), default=COURSES_ROOT,
                   help=f"Root of course outputs (default: {COURSES_ROOT}).")
    p.add_argument("--clusters", type=int, default=8, help="Number of clusters (default 8).")
    p.add_argument("--neighbors", type=int, default=10,
                   help="Nearest neighbors per hole (default 10).")
    p.add_argument("--pca-components", type=int, default=10,
                   help="PCA components to validate/fit (default 10).")
    p.add_argument("--no-umap", action="store_true", help="Skip UMAP embedding.")
    # visual-check options
    p.add_argument("--hole-id", help="Query hole id for visual-check, e.g. augusta_national__01.")
    p.add_argument("--n", type=int, default=3, help="Number of neighbors to plot (visual-check).")
    p.add_argument("--same-par", action="store_true", help="Restrict neighbors to the query's par.")
    p.add_argument("--exclude-same-course", action="store_true",
                   help="Exclude neighbors from the query's course.")
    p.add_argument("--color-by", choices=["label", "elevation"], default="label",
                   help="Color points by surface label or by elevation (visual-check).")
    p.add_argument("--max-points", type=int, default=40_000,
                   help="Max points plotted per hole (visual-check).")
    p.add_argument("--output", default=None, help="Output PNG path (visual-check).")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR.")
    return p


def _run_visual_check(args) -> int:
    import pandas as pd

    from ..paths import IndexPaths
    from .similarity import build_feature_matrix, feature_columns, similar_holes
    from .visual_compare import save_hole_comparison

    if not args.hole_id:
        log.error("visual-check requires --hole-id")
        return 1

    index = IndexPaths.for_root(args.courses_root)
    if not index.hole_features_parquet.exists():
        log.error("%s not found. Run: python -m pipeline.modeling features", index.hole_features_parquet)
        return 1

    feat = pd.read_parquet(index.hole_features_parquet)
    X, _, _ = build_feature_matrix(feat, feature_columns(feat))
    neighbors = similar_holes(feat, X, args.hole_id, k=args.n,
                              exclude_same_course=args.exclude_same_course,
                              same_par=args.same_par)
    hole_ids = [args.hole_id] + list(neighbors["similar_hole_id"])
    titles = [f"{h}\n(query)" if h == args.hole_id else h for h in hole_ids]

    if args.output:
        out = args.output
    else:
        suffix = "_".join(
            ["same_par"] * args.same_par + ["cross_course"] * args.exclude_same_course
        ) or "nearest"
        out = index.visual_checks / f"{args.hole_id}_{suffix}.png"

    path = save_hole_comparison(args.courses_root, hole_ids, out, titles=titles,
                                color_by=args.color_by, max_points=args.max_points)
    log.info("visual-check -> %s", path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)

    try:
        if args.command in ("features", "all"):
            from .hole_feature_builder import build_hole_features
            path = build_hole_features(args.courses_root)
            log.info("features -> %s", path)
        if args.command in ("similarity", "all"):
            from .export_similarity import build_hole_similarity
            written = build_hole_similarity(
                args.courses_root,
                n_clusters=args.clusters,
                n_neighbors=args.neighbors,
                pca_components=args.pca_components,
                use_umap=not args.no_umap,
            )
            for k, v in written.items():
                log.info("%s -> %s", k, v)
        if args.command == "hf-export":
            from .hf_export import build_hf_artifact
            summary = build_hf_artifact(
                args.tier, args.output, courses_root=args.courses_root,
                include_point_parquet=args.include_point_parquet,
                include_all_points=args.include_all_points,
            )
            log.info("hf-export (%s) -> %s | %s across %d files",
                     summary["tier"], summary["output_dir"],
                     summary["total_human"], summary["total_files"])
        if args.command == "similarity-modes":
            from .export_similarity import build_similarity_modes
            from .similarity import GOLF_MODES
            modes = (args.mode,) if args.mode else GOLF_MODES
            esc = True if args.exclude_same_course else None
            written = build_similarity_modes(
                args.courses_root, modes=modes, n_neighbors=args.neighbors,
                exclude_same_course=esc,
            )
            for k, v in written.items():
                log.info("%s -> %s", k, v)
        if args.command == "visual-check":
            return _run_visual_check(args)
    except (ImportError, FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
