"""Side-by-side plan-view visualization of hole point clouds.

A lightweight visual-validation layer: load a hole's compact point file, plot it
in the tee-relative aligned frame (tee at origin, green toward +Y, x<0 left /
x>0 right), and compare several holes on a **shared x/y scale** so shape is
directly comparable.

This is point-cloud visualization, not imagery — it shows *where the labeled
surfaces are*, which is exactly what the similarity model sees.

matplotlib only; no seaborn. The module does **not** set a matplotlib backend, so
it works inline in notebooks and headless (Agg) in tests/CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from ..constants import LABEL_NAMES
from ..logging_config import get_logger
from ..paths import COURSES_ROOT, CoursePaths, HolePaths

log = get_logger("modeling.visual")

__all__ = [
    "load_compact_hole_points",
    "compact_points_to_df",
    "downsample_points",
    "compute_shared_limits",
    "plot_hole_points",
    "plot_hole_comparison",
    "save_hole_comparison",
    "LABEL_COLOR",
]

# Stable label color map (kept consistent across every plot).
LABEL_COLOR: dict[str, str] = {
    "unknown": "#cccccc",
    "tee": "#111111",
    "green": "#1b7837",
    "fairway": "#5fae5f",
    "rough_osm": "#8a9a5b",
    "rough_inferred": "#cdd3bf",
    "bunker": "#e8c969",
    "water": "#1e88e5",
    "trees": "#2e6b34",
    "cartpath": "#7b5e57",
    "sand": "#e6d2a3",
}

# Per-label point styling: rough/background is subtle; hazards/markers pop.
_BG = dict(s=2, alpha=0.18, zorder=1)
_MID = dict(s=4, alpha=0.55, zorder=2)
_HAZ = dict(s=7, alpha=0.9, zorder=4)
_MARK = dict(s=12, alpha=0.95, zorder=5)
_LABEL_STYLE: dict[str, dict] = {
    "rough_inferred": _BG,
    "rough_osm": dict(s=3, alpha=0.3, zorder=1),
    "unknown": _BG,
    "fairway": dict(s=4, alpha=0.6, zorder=2),
    "cartpath": _MID,
    "sand": _MID,
    "trees": _HAZ,
    "bunker": _HAZ,
    "water": _HAZ,
    "green": _MARK,
    "tee": _MARK,
}
# Draw order: background first so hazards/markers render on top.
_DRAW_ORDER: tuple[str, ...] = (
    "rough_inferred", "rough_osm", "unknown", "fairway", "sand",
    "cartpath", "trees", "bunker", "water", "green", "tee",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _split_hole_id(hole_id: str) -> tuple[str, int]:
    if "__" not in hole_id:
        raise ValueError(f"hole_id '{hole_id}' is not '<course_slug>__<NN>'.")
    slug, num = hole_id.rsplit("__", 1)
    return slug, int(num)


def compact_points_to_df(payload: dict) -> pd.DataFrame:
    """Convert a parsed ``hole_points_compact.json`` payload into a DataFrame.

    Columns: ``x, y, z, label_id, label``. ``x`` is lateral (x<0 left, x>0 right),
    ``y`` is downrange (tee->green), ``z`` is elevation relative to the tee.
    """
    pts = payload.get("points", [])
    label_map = {str(k): v for k, v in (payload.get("label_map") or {}).items()}
    df = pd.DataFrame(pts, columns=["x", "y", "z", "label_id"])
    if df.empty:
        df["label"] = pd.Series(dtype="object")
        return df
    df["label_id"] = df["label_id"].astype(int)
    df["label"] = df["label_id"].map(
        lambda i: label_map.get(str(i)) or LABEL_NAMES.get(int(i), "unknown")
    )
    return df


def load_compact_hole_points(courses_root, hole_id: str) -> pd.DataFrame:
    """Load a hole's compact point cloud as a DataFrame (see :func:`compact_points_to_df`)."""
    slug, num = _split_hole_id(hole_id)
    cp = CoursePaths.for_slug(slug, courses_root=Path(courses_root))
    path = HolePaths.for_hole(cp, num).hole_points_compact
    if not path.exists():
        raise FileNotFoundError(f"compact point file not found for {hole_id}: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return compact_points_to_df(payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def downsample_points(df: pd.DataFrame, max_points: Optional[int], seed: int = 0) -> pd.DataFrame:
    """Uniformly downsample to at most ``max_points`` rows (keeps label mix ≈ stable)."""
    if max_points and len(df) > max_points:
        return df.sample(n=max_points, random_state=seed).reset_index(drop=True)
    return df


def compute_shared_limits(
    point_dfs: Sequence[pd.DataFrame], margin: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shared (xlim, ylim) covering every hole, so side-by-side shape is fair."""
    xs: list[float] = []
    ys: list[float] = []
    for df in point_dfs:
        if len(df):
            xs += [float(df["x"].min()), float(df["x"].max())]
            ys += [float(df["y"].min()), float(df["y"].max())]
    if not xs:
        return (-1.0, 1.0), (-1.0, 1.0)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    mx = (xmax - xmin) * margin or 1.0
    my = (ymax - ymin) * margin or 1.0
    return (xmin - mx, xmax + mx), (ymin - my, ymax + my)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_hole_points(
    points_df: pd.DataFrame,
    ax=None,
    color_by: str = "label",
    title: Optional[str] = None,
    max_points: int = 50_000,
    norm: Optional[Normalize] = None,
):
    """Plot a single hole in 2D plan view (x = left/right, y = tee→green).

    ``color_by="label"`` colors points by surface (stable color map, subtle
    rough); ``color_by="elevation"`` colors by ``z`` relative to the tee. A legend
    / colorbar is added only when this function creates its own axes (``ax=None``);
    in a comparison grid the caller supplies one shared legend/colorbar.
    """
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(5, 6))

    df = downsample_points(points_df, max_points)

    if color_by == "elevation":
        sc = ax.scatter(df["x"], df["y"], c=df["z"], cmap="viridis", s=4,
                        alpha=0.85, norm=norm, edgecolors="none")
        if standalone:
            plt.colorbar(sc, ax=ax, label="z rel tee (m)")
    elif color_by == "label":
        for name in _DRAW_ORDER:
            sub = df[df["label"] == name]
            if len(sub) == 0:
                continue
            st = _LABEL_STYLE.get(name, _MID)
            ax.scatter(sub["x"], sub["y"], c=LABEL_COLOR.get(name, "#999999"),
                       s=st["s"], alpha=st["alpha"], zorder=st["zorder"],
                       label=name, edgecolors="none")
        if standalone:
            ax.legend(loc="best", fontsize=7, markerscale=2.0, framealpha=0.9)
    else:
        raise ValueError("color_by must be 'label' or 'elevation'")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)  left < 0 < right")
    ax.set_ylabel("y (m)  tee → green")
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_hole_comparison(
    courses_root,
    hole_ids: Sequence[str],
    titles: Optional[Sequence[str]] = None,
    color_by: str = "label",
    max_points: int = 50_000,
):
    """Plot several holes side by side on a **shared x/y scale**. Returns the Figure."""
    if not hole_ids:
        raise ValueError("hole_ids must be non-empty")
    dfs = [load_compact_hole_points(courses_root, h) for h in hole_ids]
    xlim, ylim = compute_shared_limits(dfs)
    n = len(dfs)

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 6.0), squeeze=False)
    axes = axes[0]

    norm = None
    if color_by == "elevation":
        zs = [df["z"] for df in dfs if len(df)]
        if zs:
            allz = pd.concat(zs)
            norm = Normalize(float(allz.min()), float(allz.max()))

    present: list[str] = []
    for i, (ax, df) in enumerate(zip(axes, dfs)):
        title = titles[i] if titles is not None else hole_ids[i]
        plot_hole_points(df, ax=ax, color_by=color_by, title=title,
                         max_points=max_points, norm=norm)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        for nm in df["label"].dropna().unique():
            if nm not in present:
                present.append(nm)

    if color_by == "label":
        handles = [Patch(facecolor=LABEL_COLOR.get(nm, "#999999"), edgecolor="none", label=nm)
                   for nm in _DRAW_ORDER if nm in present]
        if handles:
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(len(handles), 6), fontsize=8, frameon=False)
            fig.subplots_adjust(bottom=0.16)
    else:
        mappable = next((ax.collections[0] for ax in axes if ax.collections), None)
        if mappable is not None:
            fig.colorbar(mappable, ax=list(axes), label="z rel tee (m)", shrink=0.8)

    fig.suptitle(f"Hole comparison ({color_by})", y=0.99)
    return fig


def save_hole_comparison(
    courses_root,
    hole_ids: Sequence[str],
    output_path,
    titles: Optional[Sequence[str]] = None,
    color_by: str = "label",
    max_points: int = 50_000,
) -> Path:
    """Render :func:`plot_hole_comparison` and save it to ``output_path`` (PNG)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_hole_comparison(courses_root, hole_ids, titles=titles,
                               color_by=color_by, max_points=max_points)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("saved hole comparison -> %s", output_path)
    return output_path
