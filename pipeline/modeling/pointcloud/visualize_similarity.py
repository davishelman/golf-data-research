"""Lightweight visual inspection reports for v2.5 point-cloud matches.

Renders a target hole next to its top-N similar candidates so a human can
eyeball whether the surface-aware Chamfer score is finding genuinely similar
holes. Each hole is drawn in its normalized tee-relative, green-aligned frame
(``x_lateral_m`` across, ``y_down_hole_m`` up the page), colored by surface, with
the per-pair score breakdown printed beside it.

Design notes
------------
* The data-extraction and text helpers are pure and unit-tested; rendering uses
  the non-interactive ``Agg`` backend and is exercised only as a smoke test (no
  brittle pixel snapshots).
* Geometry is read through the existing :class:`PointCloudArtifactLoader` seam,
  so this works against any loader without change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ...logging_config import get_logger  # noqa: E402
from ...paths import COURSES_ROOT, IndexPaths  # noqa: E402
from .export_similarity import CompactArtifactLoader, PointCloudArtifactLoader  # noqa: E402
from .schemas import KNOWN_SURFACES, SurfacePoint  # noqa: E402
from .validate_similarity import (  # noqa: E402
    VALIDATION_DIRNAME,
    load_result_dir,
    resolve_result_dir,
    sanitize_hole_id,
    top_matches_for_target,
)

log = get_logger("modeling.pointcloud.visualize")

#: Distinct colors per surface for the scatter plots.
SURFACE_COLORS: dict[str, str] = {
    "fairway": "#4caf50",
    "green": "#1b5e20",
    "bunker": "#e0c068",
    "water": "#2196f3",
    "tee": "#9c27b0",
}

VISUALS_DIRNAME = "visuals"
_SCORE_FIELDS = (
    "total_score", "fairway_score", "green_score", "bunker_score",
    "water_score", "tee_score", "yardage_penalty", "elevation_penalty",
    "missing_surface_penalty",
)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def surface_points_xy(points: Sequence[SurfacePoint], surface: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_lateral, y_down_hole)`` arrays for one surface's points."""
    xs = [p.x_lateral_m for p in points if p.surface == surface]
    ys = [p.y_down_hole_m for p in points if p.surface == surface]
    return np.asarray(xs, dtype="float64"), np.asarray(ys, dtype="float64")


def compute_bounds(points: Sequence[SurfacePoint], pad: float = 5.0) -> tuple[float, float, float, float]:
    """Padded ``(xmin, xmax, ymin, ymax)`` over all points (a unit box if empty)."""
    if not points:
        return (-pad, pad, -pad, pad)
    xs = np.array([p.x_lateral_m for p in points], dtype="float64")
    ys = np.array([p.y_down_hole_m for p in points], dtype="float64")
    return (float(xs.min()) - pad, float(xs.max()) + pad,
            float(ys.min()) - pad, float(ys.max()) + pad)


def score_breakdown_text(row: dict) -> str:
    """Multi-line score breakdown for a result row (``None``/NaN shown as '—')."""
    def fmt(key: str) -> str:
        v = row.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{float(v):.3f}"

    lines = [f"total: {fmt('total_score')}"]
    for surface in KNOWN_SURFACES:
        lines.append(f"{surface}: {fmt(surface + '_score')}")
    lines.append(f"yardage pen: {fmt('yardage_penalty')}")
    lines.append(f"elev pen: {fmt('elevation_penalty')}")
    lines.append(f"missing pen: {fmt('missing_surface_penalty')}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def plot_hole(ax, points: Sequence[SurfacePoint], title: str) -> None:
    """Scatter one hole's surfaces onto ``ax`` in its normalized frame."""
    for surface in KNOWN_SURFACES:
        xs, ys = surface_points_xy(points, surface)
        if xs.size:
            ax.scatter(xs, ys, s=4, c=SURFACE_COLORS[surface], label=surface, alpha=0.7)
    xmin, xmax, ymin, ymax = compute_bounds(points)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def build_comparison_figure(
    target_hole_id: str,
    target_points: Sequence[SurfacePoint],
    candidates: list[tuple[str, Sequence[SurfacePoint], dict]],
):
    """Build (but do not save) a target-vs-candidates comparison figure.

    ``candidates`` is ``[(candidate_hole_id, points, score_row), ...]`` already
    ordered by rank. Returns a Matplotlib ``Figure``.
    """
    n = len(candidates) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    row = axes[0]
    plot_hole(row[0], target_points, f"TARGET\n{target_hole_id}")
    row[0].legend(loc="upper right", fontsize=5, markerscale=1.5)

    for i, (cand_id, pts, score_row) in enumerate(candidates, start=1):
        rank = score_row.get("rank", i)
        plot_hole(row[i], pts, f"#{rank} {cand_id}")
        row[i].text(
            0.02, 0.98, score_breakdown_text(score_row),
            transform=row[i].transAxes, fontsize=5, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8),
        )
    fig.suptitle(f"v2.5 point-cloud similarity — {target_hole_id}", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Report runner
# --------------------------------------------------------------------------- #

def run_visual_report(
    target_hole_id: str,
    config_name: str,
    *,
    loader: Optional[PointCloudArtifactLoader] = None,
    courses_root: Path = COURSES_ROOT,
    result_dir: Optional[Path] = None,
    top_n: int = 6,
    output_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Render a target-vs-top-N comparison PNG for one hole + config.

    Reads the config's batch results, takes the target's top ``top_n`` matches,
    loads geometry through ``loader`` (default :class:`CompactArtifactLoader`),
    and writes a PNG. Returns the PNG path.
    """
    loader = loader or CompactArtifactLoader(courses_root=courses_root)
    result_dir = Path(result_dir) if result_dir else resolve_result_dir(config_name, courses_root)
    results, _manifest, resolved_name = load_result_dir(result_dir)
    top = top_matches_for_target(results, target_hole_id, top_n, resolved_name)

    if output_dir is None:
        output_dir = (IndexPaths.for_root(courses_root).pointcloud_similarity_dir
                      / VALIDATION_DIRNAME / sanitize_hole_id(target_hole_id) / VISUALS_DIRNAME)
    output_dir = Path(output_dir)
    png_path = output_dir / f"compare_{resolved_name}.png"
    if png_path.exists() and not overwrite:
        raise FileExistsError(f"{png_path} already exists; pass overwrite=True (--overwrite).")

    target_points = loader.load_points(target_hole_id)
    candidates: list[tuple[str, list[SurfacePoint], dict]] = []
    for r in top.itertuples(index=False):
        row = {f: getattr(r, f, None) for f in (*_SCORE_FIELDS, "rank")}
        candidates.append((r.candidate_hole_id, loader.load_points(r.candidate_hole_id), row))

    fig = build_comparison_figure(target_hole_id, target_points, candidates)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    log.info("visual report '%s' (%s): %d candidates -> %s",
             target_hole_id, resolved_name, len(candidates), png_path)
    return png_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.modeling.pointcloud.visualize_similarity",
        description="Render a target hole next to its top-N v2.5 similar holes.",
    )
    parser.add_argument("--target-hole-id", required=True,
                        help="Target hole id, e.g. 'augusta_national:13'.")
    parser.add_argument("--config-name", required=True,
                        help="Config name whose batch results to read (e.g. 'baseline').")
    parser.add_argument("--top-n", type=int, default=6,
                        help="Candidates to show (default: 6).")
    parser.add_argument("--courses-root", type=Path, default=COURSES_ROOT)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    png = run_visual_report(
        args.target_hole_id, args.config_name, courses_root=args.courses_root,
        result_dir=args.result_dir, top_n=args.top_n, output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(str(png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
