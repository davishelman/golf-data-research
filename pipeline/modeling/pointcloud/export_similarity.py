"""CLI / entry point for v2.5 point-cloud similarity (single-target + batch).

Single target (one hole vs. the field):

    python -m pipeline.modeling.pointcloud.export_similarity \\
        --config configs/similarity/pointcloud_chamfer_v1.yaml \\
        --target-hole-id augusta_national:13 \\
        --top-n 10

Batch (every hole vs. the field, written to artifact files):

    python -m pipeline.modeling.pointcloud.export_similarity \\
        --config configs/similarity/pointcloud_chamfer_v1.yaml \\
        --all --top-n 25

Responsibilities here are orchestration only: load a validated config, pull
clean metadata + normalized points from an *artifact loader*, filter candidates,
score the survivors, rank deterministically, and emit results (IDs + scores,
never geometry).

The loader is the single integration seam between v2.5 and the existing
geometry/DEM pipeline. :class:`PointCloudArtifactLoader` is the contract;
:class:`CompactArtifactLoader` is a working implementation over the artifacts the
pipeline already writes (per-hole ``hole_points_compact.json`` + the
``hole_features`` table). Swapping in a different store (HF artifact, parquet
point clouds, a service) means writing one more loader — scoring is untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol

import pandas as pd

from ...logging_config import get_logger
from ...paths import COURSES_ROOT, CoursePaths, HolePaths, IndexPaths
from .candidate_filter import filter_candidate
from .config import PointCloudSimilarityConfig, load_config
from .schemas import (
    LABEL_TO_SURFACE,
    HoleMetadata,
    SimilarityResult,
    SurfacePoint,
    make_pc_hole_id,
    parse_pc_hole_id,
)
from .score import score_pair

log = get_logger("modeling.pointcloud.export")

#: Column order for ``similarity_results.csv`` (rank inserted after the ids).
SIMILARITY_RESULTS_COLUMNS: tuple[str, ...] = (
    "model_version", "config_name", "config_hash",
    "target_hole_id", "candidate_hole_id", "rank", "total_score",
    "fairway_score", "green_score", "bunker_score", "water_score", "tee_score",
    "yardage_penalty", "elevation_penalty", "missing_surface_penalty", "filter_reason",
)

#: Output file names written per config run.
RESULTS_FILENAME = "similarity_results.csv"
FILTER_SUMMARY_FILENAME = "filter_summary.csv"
MANIFEST_FILENAME = "manifest.json"

PointsGetter = Callable[[str], list[SurfacePoint]]


# --------------------------------------------------------------------------- #
# Loader contract + concrete implementation
# --------------------------------------------------------------------------- #

class PointCloudArtifactLoader(Protocol):
    """Source of clean v2.5 metadata + normalized surface points.

    Implementations own *all* I/O and any mapping from the existing artifact
    layout into v2.5's contracts. The scorer/CLI only ever sees the dataclasses.
    """

    def load_metadata(self) -> dict[str, HoleMetadata]:
        """Return ``{hole_id: HoleMetadata}`` for every available hole."""
        ...

    def load_points(self, hole_id: str) -> list[SurfacePoint]:
        """Return the normalized, surface-tagged points for one hole."""
        ...


class CompactArtifactLoader:
    """Loader over the pipeline's existing compact point clouds + feature table.

    Metadata comes from ``hole_features.parquet`` (par, yardage, elevation).
    Surface presence and normalized points come from each hole's
    ``hole_points_compact.json`` (already tee-relative + green-aligned). v2.5 hole
    ids (``slug:number``) are derived from ``course_slug`` + ``hole_number``; the
    v2 feature-table id (``slug__NN``) is kept only to locate files.

    Elevation mapping (documented assumption): the feature table exposes only
    ``tee_to_green_elevation_change``. We set ``tee_elevation_m`` from the compact
    cloud's tee-anchor ``origin.z_abs_m`` and ``green_elevation_m = tee +
    tee_to_green_elevation_change`` so the scorer's ``Δ(green-tee)`` term is exact
    without needing a separate absolute green elevation.
    """

    def __init__(
        self,
        courses_root: Path = COURSES_ROOT,
        features_path: Optional[Path] = None,
    ) -> None:
        self.courses_root = Path(courses_root)
        index = IndexPaths.for_root(self.courses_root)
        self.features_path = Path(features_path) if features_path else index.hole_features_parquet
        self._features: Optional[pd.DataFrame] = None
        # hole_id -> (course_slug, hole_number); built alongside metadata.
        self._locator: dict[str, tuple[str, int]] = {}

    # -- internal helpers --------------------------------------------------- #

    def _features_df(self) -> pd.DataFrame:
        if self._features is None:
            if not self.features_path.exists():
                raise FileNotFoundError(
                    f"{self.features_path} not found. Build the feature table first:\n"
                    "    python -m pipeline.modeling features"
                )
            self._features = pd.read_parquet(self.features_path)
        return self._features

    def _compact_path(self, course_slug: str, hole_number: int) -> Path:
        course = CoursePaths.for_slug(course_slug, self.courses_root)
        return HolePaths.for_hole(course, hole_number).hole_points_compact

    def _read_compact(self, course_slug: str, hole_number: int) -> Optional[dict]:
        path = self._compact_path(course_slug, hole_number)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # -- contract ----------------------------------------------------------- #

    def load_metadata(self) -> dict[str, HoleMetadata]:
        df = self._features_df()
        required = {"course_slug", "hole_number", "par", "hole_length_yd"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"feature table missing required column(s): {sorted(missing)}")

        out: dict[str, HoleMetadata] = {}
        for row in df.itertuples(index=False):
            slug = getattr(row, "course_slug")
            number = int(getattr(row, "hole_number"))
            hole_id = make_pc_hole_id(slug, number)
            self._locator[hole_id] = (slug, number)

            compact = self._read_compact(slug, number)
            present = _surfaces_present(compact)
            tee_elev, green_elev = _elevations(compact, getattr(row, "tee_to_green_elevation_change", None))

            out[hole_id] = HoleMetadata(
                hole_id=hole_id,
                course_slug=slug,
                course_name=getattr(row, "course_name", None),
                hole_number=number,
                par=int(getattr(row, "par")),
                yards=float(getattr(row, "hole_length_yd")),
                has_tee="tee" in present,
                has_green="green" in present,
                has_fairway="fairway" in present,
                has_bunker="bunker" in present,
                has_water="water" in present,
                tee_elevation_m=tee_elev,
                green_elevation_m=green_elev,
            )
        return out

    def load_points(self, hole_id: str) -> list[SurfacePoint]:
        if hole_id not in self._locator:
            # Allow loading points without a prior full metadata pass.
            slug, number = parse_pc_hole_id(hole_id)
        else:
            slug, number = self._locator[hole_id]

        compact = self._read_compact(slug, number)
        if not compact:
            return []
        label_map = {int(k): v for k, v in compact.get("label_map", {}).items()}
        points: list[SurfacePoint] = []
        for rec in compact.get("points", []):
            x, y, z, label_id = rec[0], rec[1], rec[2], int(rec[3])
            surface = LABEL_TO_SURFACE.get(label_map.get(label_id, ""))
            if surface is None:
                continue  # rough / trees / cartpath / sand / unknown -> not modeled
            points.append(SurfacePoint(
                hole_id=hole_id, surface=surface,
                x_lateral_m=float(x), y_down_hole_m=float(y), z_relative_m=float(z),
            ))
        return points


def _surfaces_present(compact: Optional[dict]) -> set[str]:
    if not compact:
        return set()
    label_map = {int(k): v for k, v in compact.get("label_map", {}).items()}
    present: set[str] = set()
    for rec in compact.get("points", []):
        surface = LABEL_TO_SURFACE.get(label_map.get(int(rec[3]), ""))
        if surface is not None:
            present.add(surface)
    return present


def _elevations(compact: Optional[dict], tee_to_green_change) -> tuple[Optional[float], Optional[float]]:
    if not compact:
        return None, None
    tee_z = (compact.get("origin") or {}).get("z_abs_m")
    if tee_z is None:
        return None, None
    tee_z = float(tee_z)
    if tee_to_green_change is None or pd.isna(tee_to_green_change):
        return tee_z, None
    return tee_z, tee_z + float(tee_to_green_change)


# --------------------------------------------------------------------------- #
# Ranking core (shared by single-target and batch)
# --------------------------------------------------------------------------- #

def _rank_for_target(
    target_id: str,
    metadata: dict[str, HoleMetadata],
    get_points: PointsGetter,
    config: PointCloudSimilarityConfig,
    top_n: int,
    *,
    include_self: bool = False,
    exclude_same_course: bool = False,
) -> tuple[list[tuple[int, SimilarityResult]], Counter]:
    """Filter, score, and rank candidates for one target hole.

    Returns ``(ranked, reason_counts)`` where:

    * ``ranked`` is ``[(rank, SimilarityResult), ...]`` — at most ``top_n`` rows,
      sorted by ``(total_score ascending, candidate_hole_id ascending)`` and
      ranked 1..N.
    * ``reason_counts`` counts the candidate-filter outcome for *every* evaluated
      candidate (the ``PASS`` count includes all scored pairs, pre-truncation),
      so a batch run can aggregate a filter summary.

    Candidates are iterated in sorted id order for determinism. The target hole
    itself is skipped unless ``include_self``; same-course candidates are skipped
    when ``exclude_same_course``. Neither skip contributes to ``reason_counts``.
    """
    target = metadata[target_id]
    target_points = get_points(target_id)

    reason_counts: Counter = Counter()
    scored: list[SimilarityResult] = []

    for cand_id in sorted(metadata):
        candidate = metadata[cand_id]
        if cand_id == target_id and not include_self:
            continue
        if exclude_same_course and cand_id != target_id and candidate.course_slug == target.course_slug:
            continue

        decision = filter_candidate(target, candidate, config)
        reason_counts[decision.reason] += 1
        if not decision.passed:
            continue

        result = score_pair(
            target, candidate, target_points, get_points(cand_id), config,
            filter_reason=decision.reason,
        )
        scored.append(result)

    scored.sort(key=lambda r: (r.total_score, r.candidate_hole_id))
    ranked = [(i, r) for i, r in enumerate(scored[:top_n], start=1)]
    return ranked, reason_counts


def rank_similar_holes(
    target_hole_id: str,
    config: PointCloudSimilarityConfig,
    loader: PointCloudArtifactLoader,
    top_n: int = 10,
    *,
    exclude_same_course: bool = True,
) -> list[SimilarityResult]:
    """The ``top_n`` holes most similar to one target (most similar first).

    Backward-compatible single-target entry point. Candidates that fail filtering
    are skipped; ranking is deterministic
    (``total_score`` then ``candidate_hole_id``).
    """
    metadata = loader.load_metadata()
    if target_hole_id not in metadata:
        raise KeyError(
            f"target hole_id {target_hole_id!r} not found among "
            f"{len(metadata)} loaded holes."
        )
    ranked, _ = _rank_for_target(
        target_hole_id, metadata, loader.load_points, config, top_n,
        include_self=False, exclude_same_course=exclude_same_course,
    )
    return [r for _rank, r in ranked]


def results_to_frame(results: Iterable[SimilarityResult]) -> pd.DataFrame:
    """Tabular view of ranked results (IDs + scores; no geometry)."""
    rows = [r.to_row() for r in results]
    return pd.DataFrame(rows)


def _caching_points_getter(loader: PointCloudArtifactLoader) -> PointsGetter:
    """Wrap ``loader.load_points`` with an in-memory cache (each hole read once).

    Batch ranking touches every hole's points O(num_targets) times; caching turns
    that back into a single read per hole without changing the loader contract.
    """
    cache: dict[str, list[SurfacePoint]] = {}

    def get(hole_id: str) -> list[SurfacePoint]:
        points = cache.get(hole_id)
        if points is None:
            points = loader.load_points(hole_id)
            cache[hole_id] = points
        return points

    return get


# --------------------------------------------------------------------------- #
# Batch export
# --------------------------------------------------------------------------- #

def _source_artifact_paths(loader: PointCloudArtifactLoader) -> dict[str, str]:
    """Best-effort provenance for the manifest (empty for opaque loaders)."""
    if isinstance(loader, CompactArtifactLoader):
        return {
            "courses_root": str(loader.courses_root),
            "features_path": str(loader.features_path),
        }
    return {}


def run_batch_export(
    config: PointCloudSimilarityConfig,
    loader: PointCloudArtifactLoader,
    *,
    output_dir: Path,
    config_path: Optional[Path] = None,
    top_n: int = 25,
    include_self: bool = False,
    exclude_same_course: bool = False,
    overwrite: bool = False,
    limit_targets: Optional[int] = None,
) -> dict[str, object]:
    """Score every target hole against the field and write artifact files.

    Writes ``similarity_results.csv``, ``filter_summary.csv`` and
    ``manifest.json`` into ``output_dir``. Returns the manifest dict. Raises
    :class:`FileExistsError` if results already exist and ``overwrite`` is False.

    The run is deterministic: targets are processed in sorted id order, and each
    target's candidates are ranked by ``(total_score, candidate_hole_id)``.
    """
    output_dir = Path(output_dir)
    results_path = output_dir / RESULTS_FILENAME
    if results_path.exists() and not overwrite:
        raise FileExistsError(
            f"{results_path} already exists; pass overwrite=True (--overwrite) to replace it."
        )

    metadata = loader.load_metadata()
    target_ids = sorted(metadata)
    if limit_targets is not None:
        target_ids = target_ids[:limit_targets]
    get_points = _caching_points_getter(loader)

    rows: list[dict] = []
    reason_counts: Counter = Counter()
    total_scored_pairs = 0

    for target_id in target_ids:
        ranked, target_reasons = _rank_for_target(
            target_id, metadata, get_points, config, top_n,
            include_self=include_self, exclude_same_course=exclude_same_course,
        )
        reason_counts.update(target_reasons)
        total_scored_pairs += target_reasons.get("PASS", 0)
        for rank, result in ranked:
            row = result.to_row()
            row["rank"] = rank
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)

    results_frame = pd.DataFrame(rows, columns=list(SIMILARITY_RESULTS_COLUMNS))
    results_frame.to_csv(results_path, index=False)

    filter_frame = _filter_summary_frame(reason_counts)
    filter_frame.to_csv(output_dir / FILTER_SUMMARY_FILENAME, index=False)

    manifest = {
        "model_version": config.model_version,
        "config_name": config.config_name,
        "config_hash": config.config_hash,
        "config_path": str(config_path) if config_path is not None else config.source_path or None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_n": top_n,
        "include_self": include_self,
        "exclude_same_course": exclude_same_course,
        "total_targets": len(target_ids),
        "total_scored_pairs": total_scored_pairs,
        "total_written_rows": len(rows),
        "filter_reason_counts": dict(sorted(reason_counts.items())),
        "source_artifact_paths": _source_artifact_paths(loader),
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    log.info(
        "batch export '%s': %d targets, %d scored pairs, %d rows -> %s",
        config.config_name, len(target_ids), total_scored_pairs, len(rows), output_dir,
    )
    return manifest


def _filter_summary_frame(reason_counts: Counter) -> pd.DataFrame:
    """Summarize candidate-filter outcomes: one row per reason, sorted.

    Columns: ``filter_reason``, ``count``. Sorted by descending count then
    reason for a stable, readable ordering.
    """
    rows = [{"filter_reason": reason, "count": int(count)}
            for reason, count in reason_counts.items()]
    frame = pd.DataFrame(rows, columns=["filter_reason", "count"])
    if not frame.empty:
        frame = frame.sort_values(
            ["count", "filter_reason"], ascending=[False, True]
        ).reset_index(drop=True)
    return frame


def default_output_dir(config: PointCloudSimilarityConfig, courses_root: Path = COURSES_ROOT) -> Path:
    """``courses/_index/pointcloud_similarity/<config_name>/`` for a config."""
    return IndexPaths.for_root(courses_root).pointcloud_similarity_dir / config.config_name


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.modeling.pointcloud.export_similarity",
        description="Rank holes by v2.5 point-cloud Chamfer similarity, for a "
                    "single target or in batch across every hole.",
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to a v2.5 similarity YAML config.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target-hole-id",
                      help="Single-target mode: rank the field against this hole "
                           "(e.g. 'augusta_national:13').")
    mode.add_argument("--all", action="store_true",
                      help="Batch mode: rank every hole and write artifact files.")

    parser.add_argument("--top-n", type=int, default=10,
                        help="Neighbors to keep per target (default: 10).")
    parser.add_argument("--courses-root", type=Path, default=COURSES_ROOT,
                        help="Root of the courses/ artifact tree.")
    parser.add_argument("--features", type=Path, default=None,
                        help="Override path to hole_features.parquet.")

    # Single-target-only knobs.
    parser.add_argument("--include-same-course", action="store_true",
                        help="(single-target) allow candidates from the target's own course.")
    parser.add_argument("--out", type=Path, default=None,
                        help="(single-target) optional CSV path for ranked results.")

    # Batch-only knobs.
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="(batch) output dir (default: "
                             "courses/_index/pointcloud_similarity/<config_name>/).")
    parser.add_argument("--include-self", action="store_true",
                        help="(batch) include each hole's self-comparison (default: off).")
    parser.add_argument("--overwrite", action="store_true",
                        help="(batch) overwrite existing output files.")
    parser.add_argument("--limit-targets", type=int, default=None,
                        help="(batch) only process the first N target holes (testing).")
    return parser


def _run_single(args, config: PointCloudSimilarityConfig, loader: PointCloudArtifactLoader) -> int:
    results = rank_similar_holes(
        args.target_hole_id, config, loader, top_n=args.top_n,
        exclude_same_course=not args.include_same_course,
    )
    if not results:
        log.warning("no eligible candidates for %s (check par/yardage filters and "
                    "that point clouds exist).", args.target_hole_id)
        return 1

    frame = results_to_frame(results)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        log.info("wrote %d rows -> %s", len(frame), args.out)
    return 0


def _run_batch(args, config: PointCloudSimilarityConfig, loader: PointCloudArtifactLoader) -> int:
    output_dir = args.output_dir or default_output_dir(config, args.courses_root)
    manifest = run_batch_export(
        config, loader,
        output_dir=output_dir, config_path=args.config, top_n=args.top_n,
        include_self=args.include_self, overwrite=args.overwrite,
        limit_targets=args.limit_targets,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    log.info("loaded config '%s' (%s, hash=%s)",
             config.config_name, config.model_version, config.config_hash[:12])

    loader = CompactArtifactLoader(courses_root=args.courses_root, features_path=args.features)
    if args.all:
        return _run_batch(args, config, loader)
    return _run_single(args, config, loader)


if __name__ == "__main__":
    sys.exit(main())
