"""Modeling CLI — decoupled from the geo pipeline (no geopandas/rasterio needed).

Usage:
    python -m pipeline.modeling features      # build hole_features.parquet/csv
    python -m pipeline.modeling similarity    # build clusters + similarity examples
    python -m pipeline.modeling all           # both, in order
"""

from __future__ import annotations

import argparse
import sys

from ..logging_config import configure_logging, get_logger
from ..paths import COURSES_ROOT

log = get_logger("modeling.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.modeling",
        description="Build hole feature rows and find similar golf holes.",
    )
    p.add_argument("command", choices=["features", "similarity", "all"],
                   help="Which step to run.")
    p.add_argument("--courses-root", type=type(COURSES_ROOT), default=COURSES_ROOT,
                   help=f"Root of course outputs (default: {COURSES_ROOT}).")
    p.add_argument("--clusters", type=int, default=8, help="Number of clusters (default 8).")
    p.add_argument("--neighbors", type=int, default=10,
                   help="Nearest neighbors per hole (default 10).")
    p.add_argument("--pca-components", type=int, default=10,
                   help="PCA components to validate/fit (default 10).")
    p.add_argument("--no-umap", action="store_true", help="Skip UMAP embedding.")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR.")
    return p


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
    except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
