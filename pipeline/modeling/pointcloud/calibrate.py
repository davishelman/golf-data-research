"""Calibration analysis for v2.5 point-cloud similarity scores.

Given a batch ``similarity_results.csv`` and the config that produced it, this
module decomposes each pair's ``total_score`` into its weighted components so a
human can answer the calibration questions:

* Do missing-surface penalties dominate the score?
* Do the yardage / elevation penalties dominate?
* Is elevation effectively double-counted (once via the Chamfer ``z_weight`` that
  amplifies vertical offsets on every surface, and again via the explicit
  tee->green elevation penalty)?
* Are the surface weights pulling their intended share?

It is pure analysis over existing result files — it never re-scores geometry and
never mutates inputs. Component contributions are reconstructed from the stored
per-surface scores and the config weights:

    contribution_<surface> = surface_weight[surface] * <surface>_score
    contribution_yardage   = yardage_penalty        (already absolute)
    contribution_elevation = elevation_penalty       (already absolute)
    total                  = sum(contributions)      (≈ stored total_score)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...logging_config import get_logger
from ...paths import COURSES_ROOT
from .config import PointCloudSimilarityConfig, load_config
from .export_similarity import RESULTS_FILENAME
from .schemas import KNOWN_SURFACES
from .validate_similarity import resolve_result_dir

log = get_logger("modeling.pointcloud.calibrate")

CALIBRATION_REPORT_FILENAME = "calibration_report.csv"
CALIBRATION_SUMMARY_FILENAME = "calibration_summary.json"

#: A component "dominates" a pair's score when its share exceeds this fraction.
DOMINANCE_THRESHOLD: float = 0.5


def decompose_components(
    results: pd.DataFrame, config: PointCloudSimilarityConfig
) -> pd.DataFrame:
    """Per-pair weighted component decomposition of ``total_score``.

    Returns a frame with one row per input pair: the id columns, each
    ``contrib_<component>`` (weighted surface contributions plus yardage and
    elevation), ``recomputed_total``, and ``share_<component>`` (contribution /
    recomputed_total, 0 when the total is 0).
    """
    out = pd.DataFrame({
        "target_hole_id": results["target_hole_id"].to_numpy(),
        "candidate_hole_id": results["candidate_hole_id"].to_numpy(),
        "total_score": results["total_score"].to_numpy(dtype="float64"),
    })

    contrib_cols: list[str] = []
    for surface in KNOWN_SURFACES:
        weight = config.surface_weights.get(surface, 0.0)
        score_col = f"{surface}_score"
        raw = (results[score_col].to_numpy(dtype="float64")
               if score_col in results.columns else np.zeros(len(results)))
        contrib = np.nan_to_num(raw, nan=0.0) * weight
        col = f"contrib_{surface}"
        out[col] = contrib
        contrib_cols.append(col)

    for component, src in (("yardage", "yardage_penalty"), ("elevation", "elevation_penalty")):
        vals = (results[src].to_numpy(dtype="float64")
                if src in results.columns else np.zeros(len(results)))
        col = f"contrib_{component}"
        out[col] = np.nan_to_num(vals, nan=0.0)
        contrib_cols.append(col)

    out["recomputed_total"] = out[contrib_cols].sum(axis=1)
    total = out["recomputed_total"].to_numpy()
    safe = np.where(total == 0.0, 1.0, total)
    for col in contrib_cols:
        out[f"share_{col.removeprefix('contrib_')}"] = np.where(
            total == 0.0, 0.0, out[col].to_numpy() / safe
        )
    return out


def summarize_calibration(
    decomposed: pd.DataFrame, config: PointCloudSimilarityConfig
) -> dict:
    """Aggregate calibration findings across all decomposed pairs.

    Reports mean component shares, the fraction of pairs where missing-surface
    penalties or the yardage/elevation penalties dominate, and a heuristic
    elevation double-count signal (Pearson correlation between the elevation
    penalty and the summed surface contributions — both rise with vertical
    dissimilarity, so a strong positive correlation flags redundancy).
    """
    n = len(decomposed)
    share_cols = [c for c in decomposed.columns if c.startswith("share_")]
    mean_shares = {c.removeprefix("share_"): float(decomposed[c].mean()) if n else 0.0
                   for c in share_cols}

    # Missing-surface contribution: per-surface contribution equal to that
    # surface's configured missing penalty * weight implies the surface was
    # missing on one side. We approximate "missing dominates" via the high-value
    # penalty contributions relative to total.
    surface_contribs = [f"contrib_{s}" for s in KNOWN_SURFACES if f"contrib_{s}" in decomposed.columns]
    penalty_contribs = decomposed[["contrib_yardage", "contrib_elevation"]].sum(axis=1)
    total = decomposed["recomputed_total"].replace(0.0, np.nan)

    missing_share = _missing_share(decomposed, config)
    missing_dominant = float((missing_share > DOMINANCE_THRESHOLD).mean()) if n else 0.0
    penalty_dominant = float(((penalty_contribs / total).fillna(0.0) > DOMINANCE_THRESHOLD).mean()) if n else 0.0

    elev = decomposed["contrib_elevation"].to_numpy()
    surf_sum = decomposed[surface_contribs].sum(axis=1).to_numpy() if surface_contribs else np.zeros(n)
    elevation_double_count_corr = _safe_corr(elev, surf_sum)

    return {
        "n_pairs": int(n),
        "config_name": config.config_name,
        "config_hash": config.config_hash,
        "mean_component_shares": mean_shares,
        "missing_surface_dominant_fraction": missing_dominant,
        "penalty_dominant_fraction": penalty_dominant,
        "elevation_double_count_correlation": elevation_double_count_corr,
        "surface_weights": dict(config.surface_weights),
        "distance_scaling": {
            "x_weight": config.distance_scaling.x_weight,
            "y_weight": config.distance_scaling.y_weight,
            "z_weight": config.distance_scaling.z_weight,
        },
        "flags": _calibration_flags(missing_dominant, penalty_dominant,
                                    elevation_double_count_corr, config),
    }


def _missing_share(decomposed: pd.DataFrame, config: PointCloudSimilarityConfig) -> pd.Series:
    """Estimated share of total from missing-surface penalties, per pair.

    A surface contribution that equals ``weight * missing_penalty`` (within a
    small tolerance) is treated as a missing-surface penalty rather than a real
    Chamfer distance.
    """
    total = decomposed["recomputed_total"].replace(0.0, np.nan)
    missing = pd.Series(0.0, index=decomposed.index)
    for surface in KNOWN_SURFACES:
        col = f"contrib_{surface}"
        if col not in decomposed.columns:
            continue
        weight = config.surface_weights.get(surface, 0.0)
        penalty = config.surface_missing_penalties.get(surface)
        if penalty is None or weight == 0.0:
            continue
        expected = weight * penalty
        is_missing = np.isclose(decomposed[col].to_numpy(), expected, rtol=1e-6, atol=1e-6)
        missing = missing + np.where(is_missing, decomposed[col].to_numpy(), 0.0)
    return (missing / total).fillna(0.0)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Pearson correlation, or ``None`` if undefined (constant input / n<2)."""
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _calibration_flags(
    missing_dominant: float, penalty_dominant: float,
    elevation_corr: Optional[float], config: PointCloudSimilarityConfig
) -> dict:
    """Boolean heuristics that suggest a config may need recalibration."""
    return {
        "missing_penalties_may_dominate": missing_dominant > 0.25,
        "penalties_may_dominate": penalty_dominant > 0.25,
        "elevation_possibly_double_counted": (
            config.distance_scaling.z_weight > 1.0
            and config.penalties.tee_to_green_elevation_weight > 0.0
            and (elevation_corr is not None and elevation_corr > 0.5)
        ),
    }


def run_calibration(
    config_path: Path,
    *,
    result_dir: Optional[Path] = None,
    courses_root: Path = COURSES_ROOT,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> dict:
    """Decompose + summarize a config's batch results; write report + summary.

    ``result_dir`` defaults to the config's batch output dir. Outputs are written
    next to the results (or to ``output_dir``). Returns the summary dict.
    """
    config = load_config(config_path)
    if result_dir is None:
        result_dir = resolve_result_dir(config.config_name, courses_root)
    result_dir = Path(result_dir)
    results_path = result_dir / RESULTS_FILENAME
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Build batch outputs first:\n"
            "    python -m pipeline.modeling.pointcloud.export_similarity "
            f"--config {config_path} --all"
        )

    results = pd.read_csv(results_path)
    decomposed = decompose_components(results, config)
    summary = summarize_calibration(decomposed, config)

    output_dir = Path(output_dir) if output_dir is not None else result_dir
    report_path = output_dir / CALIBRATION_REPORT_FILENAME
    summary_path = output_dir / CALIBRATION_SUMMARY_FILENAME
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_path} already exists; pass overwrite=True (--overwrite)."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    decomposed.to_csv(report_path, index=False)
    summary["created_at"] = datetime.now(timezone.utc).isoformat()
    summary["result_dir"] = str(result_dir)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    log.info("calibration '%s': %d pairs, missing_dominant=%.3f, penalty_dominant=%.3f",
             config.config_name, summary["n_pairs"],
             summary["missing_surface_dominant_fraction"],
             summary["penalty_dominant_fraction"])
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.modeling.pointcloud.calibrate",
        description="Decompose v2.5 similarity scores into weighted components "
                    "and surface calibration findings.",
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Config YAML that produced the batch results.")
    parser.add_argument("--result-dir", type=Path, default=None,
                        help="Batch result dir (default: the config's output dir).")
    parser.add_argument("--courses-root", type=Path, default=COURSES_ROOT,
                        help="Root of the courses/ artifact tree.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write the report/summary (default: result dir).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing calibration summary.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = run_calibration(
        args.config, result_dir=args.result_dir, courses_root=args.courses_root,
        output_dir=args.output_dir, overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
