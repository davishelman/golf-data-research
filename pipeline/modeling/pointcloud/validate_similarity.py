"""v2.5 point-cloud similarity validation / comparison reports.

Reads the batch outputs written by
:mod:`pipeline.modeling.pointcloud.export_similarity` (one
``similarity_results.csv`` + ``manifest.json`` per config) and produces
target-hole-specific comparison artifacts so a human can see how the different
scoring presets (baseline / hazard-heavy / green-heavy / fairway-heavy) rank the
field for a given hole.

It never re-scores anything and never touches point-cloud geometry — it is a
pure read/compare/report layer over existing result files.

CLI
---
By config name (resolved under ``courses/_index/pointcloud_similarity/<name>/``)::

    python -m pipeline.modeling.pointcloud.validate_similarity \\
        --target-hole-id augusta_national:13 \\
        --configs baseline hazard_heavy green_heavy fairway_heavy --top-n 10

By explicit result directories::

    python -m pipeline.modeling.pointcloud.validate_similarity \\
        --target-hole-id augusta_national:13 \\
        --result-dirs courses/_index/pointcloud_similarity/baseline ...

Outputs land under
``courses/_index/pointcloud_similarity/_validation/<sanitized_target_hole_id>/``
where ``augusta_national:13`` sanitizes to ``augusta_national__13``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ...logging_config import get_logger
from ...paths import COURSES_ROOT, IndexPaths
from .export_similarity import MANIFEST_FILENAME, RESULTS_FILENAME

log = get_logger("modeling.pointcloud.validate")

#: Columns kept in each ``top_matches_<config>.csv`` (stable order).
TOP_MATCHES_COLUMNS: tuple[str, ...] = (
    "config_name", "rank", "target_hole_id", "candidate_hole_id", "total_score",
    "fairway_score", "green_score", "bunker_score", "water_score", "tee_score",
    "yardage_penalty", "elevation_penalty", "missing_surface_penalty",
)

#: Columns in ``config_overlap_summary.csv``.
OVERLAP_COLUMNS: tuple[str, ...] = (
    "config_a", "config_b", "top_n",
    "overlap_count", "union_count", "jaccard_similarity", "shared_candidates",
)

VALIDATION_DIRNAME = "_validation"
TOP_MATCHES_PREFIX = "top_matches_"
OVERLAP_FILENAME = "config_overlap_summary.csv"
RANK_COMPARISON_FILENAME = "rank_comparison.csv"
VALIDATION_MANIFEST_FILENAME = "validation_manifest.json"
TOP_MATCHES_MD_FILENAME = "top_matches.md"


def sanitize_hole_id(hole_id: str) -> str:
    """Make a hole id filesystem-safe: ``augusta_national:13`` -> ``augusta_national__13``."""
    return hole_id.replace(":", "__")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def resolve_result_dir(config_name: str, courses_root: Path = COURSES_ROOT) -> Path:
    """Default batch-output dir for a config name."""
    return IndexPaths.for_root(courses_root).pointcloud_similarity_dir / config_name


def load_result_dir(result_dir: Path) -> tuple[pd.DataFrame, Optional[dict], str]:
    """Load one batch result dir.

    Returns ``(results_df, manifest_or_none, config_name)``. The config name comes
    from the manifest when present, else the directory name. Raises
    :class:`FileNotFoundError` with a clear message if ``similarity_results.csv``
    is missing.
    """
    result_dir = Path(result_dir)
    results_path = result_dir / RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Build batch outputs first:\n"
            "    python -m pipeline.modeling.pointcloud.export_similarity "
            "--config <cfg> --all"
        )
    df = pd.read_csv(results_path)

    manifest: Optional[dict] = None
    manifest_path = result_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    config_name = (manifest or {}).get("config_name") or result_dir.name
    return df, manifest, config_name


# --------------------------------------------------------------------------- #
# Per-target extraction + comparisons (pure)
# --------------------------------------------------------------------------- #

def top_matches_for_target(
    results: pd.DataFrame, target_hole_id: str, top_n: int, config_name: str
) -> pd.DataFrame:
    """Filter results to one target hole, keep the best ``top_n``, normalize cols.

    Sorted by ``(total_score asc, candidate_hole_id asc)`` for determinism and
    re-ranked 1..N. Returns an empty (typed) frame if the target is absent.
    """
    sub = results[results["target_hole_id"] == target_hole_id].copy()
    if sub.empty:
        return pd.DataFrame(columns=list(TOP_MATCHES_COLUMNS))

    sub = sub.sort_values(["total_score", "candidate_hole_id"]).head(top_n).reset_index(drop=True)
    sub["config_name"] = config_name
    sub["rank"] = range(1, len(sub) + 1)
    for col in TOP_MATCHES_COLUMNS:
        if col not in sub.columns:
            sub[col] = pd.NA
    return sub[list(TOP_MATCHES_COLUMNS)]


def config_overlap_summary(
    top_by_config: dict[str, pd.DataFrame], top_n: int
) -> pd.DataFrame:
    """Pairwise Jaccard overlap of top-match candidate sets across configs.

    One row per unordered config pair (sorted by name). ``shared_candidates`` is
    a ``|``-joined sorted list of the intersecting candidate hole ids.
    """
    names = sorted(top_by_config)
    cand_sets = {n: set(top_by_config[n]["candidate_hole_id"]) for n in names}

    rows: list[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = cand_sets[a], cand_sets[b]
            inter = sa & sb
            union = sa | sb
            jaccard = (len(inter) / len(union)) if union else 0.0
            rows.append({
                "config_a": a,
                "config_b": b,
                "top_n": top_n,
                "overlap_count": len(inter),
                "union_count": len(union),
                "jaccard_similarity": round(jaccard, 6),
                "shared_candidates": "|".join(sorted(inter)),
            })
    return pd.DataFrame(rows, columns=list(OVERLAP_COLUMNS))


def rank_comparison(top_by_config: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-candidate rank + score across configs, with spread summary columns.

    Columns: ``candidate_hole_id``, one ``rank_<config>`` per config, one
    ``total_score_<config>`` per config, then ``configs_present_count``,
    ``best_rank``, ``worst_rank``, ``rank_spread``. Sorted by
    ``(configs_present_count desc, best_rank asc, candidate_hole_id asc)``.
    """
    names = sorted(top_by_config)
    rank_cols = [f"rank_{n}" for n in names]
    score_cols = [f"total_score_{n}" for n in names]

    # candidate -> {config: (rank, score)}
    per_candidate: dict[str, dict[str, tuple[int, float]]] = {}
    for name in names:
        for row in top_by_config[name].itertuples(index=False):
            per_candidate.setdefault(row.candidate_hole_id, {})[name] = (
                int(row.rank), float(row.total_score)
            )

    out_cols = ["candidate_hole_id", *rank_cols, *score_cols,
                "configs_present_count", "best_rank", "worst_rank", "rank_spread"]
    if not per_candidate:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict] = []
    for cand, by_config in per_candidate.items():
        row: dict = {"candidate_hole_id": cand}
        present_ranks: list[int] = []
        for name in names:
            if name in by_config:
                r, s = by_config[name]
                row[f"rank_{name}"] = r
                row[f"total_score_{name}"] = s
                present_ranks.append(r)
            else:
                row[f"rank_{name}"] = pd.NA
                row[f"total_score_{name}"] = pd.NA
        row["configs_present_count"] = len(present_ranks)
        row["best_rank"] = min(present_ranks)
        row["worst_rank"] = max(present_ranks)
        row["rank_spread"] = max(present_ranks) - min(present_ranks)
        rows.append(row)

    frame = pd.DataFrame(rows, columns=out_cols)
    frame = frame.sort_values(
        ["configs_present_count", "best_rank", "candidate_hole_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# Markdown (optional, human-facing)
# --------------------------------------------------------------------------- #

def render_markdown(
    target_hole_id: str,
    top_by_config: dict[str, pd.DataFrame],
    overlap: pd.DataFrame,
) -> str:
    """Compact human-readable summary of the per-config top matches."""
    lines = [f"# Point-cloud similarity validation — `{target_hole_id}`", ""]
    for name in sorted(top_by_config):
        tm = top_by_config[name]
        lines.append(f"## {name}")
        if tm.empty:
            lines.append("_no matches for this target in this config._")
            lines.append("")
            continue
        lines.append("| rank | candidate | total | fairway | green | bunker | water | tee |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for r in tm.itertuples(index=False):
            def fmt(v):
                return "" if pd.isna(v) else f"{float(v):.3f}"
            lines.append(
                f"| {r.rank} | {r.candidate_hole_id} | {fmt(r.total_score)} | "
                f"{fmt(r.fairway_score)} | {fmt(r.green_score)} | {fmt(r.bunker_score)} | "
                f"{fmt(r.water_score)} | {fmt(r.tee_score)} |"
            )
        lines.append("")
    if not overlap.empty:
        lines.append("## Config overlap (Jaccard of top-N candidate sets)")
        lines.append("| config_a | config_b | overlap | union | jaccard |")
        lines.append("|---|---|---:|---:|---:|")
        for r in overlap.itertuples(index=False):
            lines.append(
                f"| {r.config_a} | {r.config_b} | {r.overlap_count} | "
                f"{r.union_count} | {r.jaccard_similarity:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_validation(
    target_hole_id: str,
    *,
    configs: Optional[list[str]] = None,
    result_dirs: Optional[list[Path]] = None,
    top_n: int = 10,
    courses_root: Path = COURSES_ROOT,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
    write_markdown: bool = True,
) -> dict:
    """Build all validation artifacts for one target hole and return the manifest.

    Exactly one of ``configs`` or ``result_dirs`` must be provided. Output goes to
    ``output_dir`` (default
    ``courses/_index/pointcloud_similarity/_validation/<sanitized>/``).
    """
    if bool(configs) == bool(result_dirs):
        raise ValueError("provide exactly one of `configs` or `result_dirs`.")

    if configs:
        dirs = [resolve_result_dir(c, courses_root) for c in configs]
    else:
        dirs = [Path(d) for d in result_dirs]  # type: ignore[arg-type]

    sanitized = sanitize_hole_id(target_hole_id)
    if output_dir is None:
        output_dir = (IndexPaths.for_root(courses_root).pointcloud_similarity_dir
                      / VALIDATION_DIRNAME / sanitized)
    output_dir = Path(output_dir)

    manifest_path = output_dir / VALIDATION_MANIFEST_FILENAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; pass overwrite=True (--overwrite) to replace it."
        )

    # Load each config's results and extract this target's top matches.
    top_by_config: dict[str, pd.DataFrame] = {}
    source_manifests: dict[str, dict] = {}
    resolved_dirs: dict[str, str] = {}
    for d in dirs:
        df, src_manifest, config_name = load_result_dir(d)
        top_by_config[config_name] = top_matches_for_target(df, target_hole_id, top_n, config_name)
        resolved_dirs[config_name] = str(d)
        if src_manifest is not None:
            source_manifests[config_name] = {
                k: src_manifest.get(k)
                for k in ("model_version", "config_name", "config_hash",
                          "created_at", "top_n", "total_written_rows")
            }

    overlap = config_overlap_summary(top_by_config, top_n)
    ranks = rank_comparison(top_by_config)

    # Write outputs.
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[str] = []
    for name in sorted(top_by_config):
        fname = f"{TOP_MATCHES_PREFIX}{name}.csv"
        top_by_config[name].to_csv(output_dir / fname, index=False)
        output_files.append(fname)
    overlap.to_csv(output_dir / OVERLAP_FILENAME, index=False)
    output_files.append(OVERLAP_FILENAME)
    ranks.to_csv(output_dir / RANK_COMPARISON_FILENAME, index=False)
    output_files.append(RANK_COMPARISON_FILENAME)

    if write_markdown:
        (output_dir / TOP_MATCHES_MD_FILENAME).write_text(
            render_markdown(target_hole_id, top_by_config, overlap), encoding="utf-8"
        )
        output_files.append(TOP_MATCHES_MD_FILENAME)

    manifest = {
        "target_hole_id": target_hole_id,
        "sanitized_target_hole_id": sanitized,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_n": top_n,
        "configs": sorted(top_by_config),
        "result_dirs": {name: resolved_dirs[name] for name in sorted(resolved_dirs)},
        "output_dir": str(output_dir),
        "output_files": [VALIDATION_MANIFEST_FILENAME, *output_files],
        "match_counts": {name: int(len(top_by_config[name])) for name in sorted(top_by_config)},
        "source_manifests": source_manifests,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    log.info("validation '%s': %d configs, output -> %s",
             target_hole_id, len(top_by_config), output_dir)
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.modeling.pointcloud.validate_similarity",
        description="Compare v2.5 point-cloud similarity rankings for one target "
                    "hole across scoring configs.",
    )
    parser.add_argument("--target-hole-id", required=True,
                        help="Target hole id, e.g. 'augusta_national:13'.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--configs", nargs="+",
                        help="Config names resolved under "
                             "courses/_index/pointcloud_similarity/<name>/.")
    source.add_argument("--result-dirs", nargs="+", type=Path,
                        help="Explicit batch result directories.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Top matches to compare per config (default: 10).")
    parser.add_argument("--courses-root", type=Path, default=COURSES_ROOT,
                        help="Root of the courses/ artifact tree.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing validation outputs.")
    parser.add_argument("--no-markdown", action="store_true",
                        help="Skip writing top_matches.md.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_validation(
        args.target_hole_id,
        configs=args.configs,
        result_dirs=args.result_dirs,
        top_n=args.top_n,
        courses_root=args.courses_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        write_markdown=not args.no_markdown,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
