"""All plot rendering (per-hole + course synthesis).

Plotting is optional: the orchestrator wraps these calls in try/except so a
matplotlib/plotly failure never blocks data artifacts. Overlays are keyed by the
canonical layer names (fairways, greens, tees, bunkers, water, trees, cartpaths,
sand, rough_osm).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # headless backend; safe for batch runs
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from .raster.sampling import build_elevation_profile  # noqa: E402

SURFACE_MAX_CELLS = 250

# Canonical layer -> fill style for the plan view.
_PLAN_STYLE = [
    ("rough_osm", dict(color="#8a9a5b", alpha=0.30, edgecolor="none")),
    ("trees", dict(color="#2e6b34", alpha=0.30, edgecolor="none")),
    ("fairways", dict(color="#4caf50", alpha=0.55, edgecolor="none")),
    ("greens", dict(color="#2e7d32", alpha=0.85, edgecolor="black", linewidth=0.4)),
    ("tees", dict(color="#1b5e20", alpha=0.85, edgecolor="black", linewidth=0.4)),
    ("sand", dict(color="#efe2b3", alpha=0.6, edgecolor="#a07b00", linewidth=0.3)),
    ("bunkers", dict(color="#f4e1a1", alpha=0.95, edgecolor="#a07b00", linewidth=0.4)),
    ("water", dict(color="#1e88e5", alpha=0.7, edgecolor="navy", linewidth=0.4)),
    ("cartpaths", dict(color="#5d4037", linewidth=0.8)),
]


def profile_arrays(projected_dem_path: Path, hole_line: LineString, n: int = 200):
    return build_elevation_profile(projected_dem_path, hole_line, n)


def _summary_dict(summary: Any) -> dict:
    if isinstance(summary, dict):
        return summary
    if hasattr(summary, "to_dict"):
        return summary.to_dict()
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raster_extent(transform, width, height):
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return (left, right, bottom, top)


def _overlay_vector(ax, gdf, dst_crs, **kwargs):
    if gdf is None or len(gdf) == 0:
        return
    try:
        g = gdf.to_crs(dst_crs) if gdf.crs is not None else gdf
    except Exception:  # noqa: BLE001
        return
    try:
        g.plot(ax=ax, **kwargs)
    except Exception:  # noqa: BLE001
        return


def _draw_plan_view(ax, dem, transform, dem_crs, overlays, hole_line,
                    show_legend=True, show_colorbar=True, fig=None):
    extent = _raster_extent(transform, dem.shape[1], dem.shape[0])
    im = ax.imshow(
        dem, extent=extent, origin="upper", cmap="terrain",
        norm=Normalize(vmin=float(np.nanmin(dem)), vmax=float(np.nanmax(dem))),
        alpha=0.85,
    )
    if show_colorbar and fig is not None:
        fig.colorbar(im, ax=ax, shrink=0.75, label="Elev (m)")
    for layer, style in _PLAN_STYLE:
        _overlay_vector(ax, overlays.get(layer), dem_crs, **style)

    gpd.GeoSeries([hole_line], crs=dem_crs).plot(ax=ax, color="red", linewidth=1.8, label="Centerline")
    coords = list(hole_line.coords)
    ax.scatter([coords[0][0]], [coords[0][1]], c="white", edgecolor="black", s=50, zorder=5, label="Tee")
    ax.scatter([coords[-1][0]], [coords[-1][1]], c="yellow", edgecolor="black", s=70, marker="*", zorder=5, label="Green")
    if show_legend:
        ax.legend(loc="lower right", fontsize=7)


# ---------------------------------------------------------------------------
# Per-hole plots (rendered together)
# ---------------------------------------------------------------------------


def render_hole_plots(dem, slope_deg, transform, dem_crs, overlays, hole_line,
                      distances, elevations, summary, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    s = _summary_dict(summary)
    plot_elevation_heatmap(dem, transform, dem_crs, overlays, hole_line, plots_dir / "elevation_heatmap.png")
    plot_slope_heatmap(slope_deg, transform, dem_crs, overlays, hole_line, plots_dir / "slope_heatmap.png")
    plot_elevation_profile(distances, elevations, plots_dir / "elevation_profile.png")
    plot_hole_overview(dem, slope_deg, transform, dem_crs, overlays, hole_line,
                       distances, elevations, s, plots_dir / "overview.png")
    try:
        plot_3d_surface(dem, transform, hole_line, plots_dir / "3d_terrain.html")
    except Exception:  # noqa: BLE001 - plotly optional
        pass


def plot_elevation_heatmap(dem, transform, dem_crs, overlays, hole_line, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_plan_view(ax, dem, transform, dem_crs, overlays, hole_line, fig=fig)
    ax.set_title("Elevation heatmap")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_slope_heatmap(slope_deg, transform, dem_crs, overlays, hole_line, out_path):
    extent = _raster_extent(transform, slope_deg.shape[1], slope_deg.shape[0])
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = float(np.nanpercentile(slope_deg, 98)) if np.isfinite(np.nanmax(slope_deg)) else 30.0
    im = ax.imshow(slope_deg, extent=extent, origin="upper", cmap="magma",
                   norm=Normalize(vmin=0.0, vmax=max(vmax, 1.0)))
    plt.colorbar(im, ax=ax, shrink=0.85, label="Slope (deg)")
    for layer in ("fairways", "greens", "tees", "bunkers", "water"):
        _overlay_vector(ax, overlays.get(layer), dem_crs, facecolor="none", edgecolor="#222", linewidth=0.6)
    gpd.GeoSeries([hole_line], crs=dem_crs).plot(ax=ax, color="cyan", linewidth=2.0)
    ax.set_title("Slope heatmap")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_elevation_profile(distances, elevations, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    base = float(np.nanmin(elevations)) if np.any(np.isfinite(elevations)) else 0.0
    ax.plot(distances, elevations, color="darkgreen", linewidth=2.0)
    ax.fill_between(distances, elevations, base, color="darkgreen", alpha=0.2)
    ax.set_xlabel("Distance from tee (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Tee-to-green elevation profile")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_3d_surface(dem, transform, hole_line, out_path, max_cells=SURFACE_MAX_CELLS):
    import plotly.graph_objects as go  # lazy: only needed for 3D html

    h, w = dem.shape
    step_y = max(1, h // max_cells)
    step_x = max(1, w // max_cells)
    z = dem[::step_y, ::step_x]
    rows = np.arange(0, h, step_y)[: z.shape[0]]
    cols = np.arange(0, w, step_x)[: z.shape[1]]
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    surface = go.Surface(x=xs, y=ys, z=z, colorscale="Earth", colorbar=dict(title="Elev (m)"))
    line_x = [p[0] for p in hole_line.coords]
    line_y = [p[1] for p in hole_line.coords]
    fallback = float(np.nanmean(dem))
    line_z = []
    for x, y in zip(line_x, line_y):
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        if 0 <= row < h and 0 <= col < w and np.isfinite(dem[row, col]):
            line_z.append(float(dem[row, col]) + 1.0)
        else:
            line_z.append(fallback)
    line_trace = go.Scatter3d(x=line_x, y=line_y, z=line_z, mode="lines+markers",
                              line=dict(color="red", width=6), marker=dict(size=3, color="red"),
                              name="Hole centerline")
    fig = go.Figure(data=[surface, line_trace])
    fig.update_layout(title="3D Terrain Surface",
                      scene=dict(xaxis_title="Easting (m)", yaxis_title="Northing (m)",
                                 zaxis_title="Elevation (m)", aspectmode="data"),
                      margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def plot_hole_overview(dem, slope_deg, transform, dem_crs, overlays, hole_line,
                       distances, elevations, summary, out_path):
    s = _summary_dict(summary)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.4, 1.0])

    ax_plan = fig.add_subplot(gs[0, 0])
    _draw_plan_view(ax_plan, dem, transform, dem_crs, overlays, hole_line, fig=fig)
    ax_plan.set_title("Plan view")

    extent = _raster_extent(transform, slope_deg.shape[1], slope_deg.shape[0])
    ax_slope = fig.add_subplot(gs[0, 1])
    vmax = float(np.nanpercentile(slope_deg, 98)) if np.isfinite(np.nanmax(slope_deg)) else 30.0
    im2 = ax_slope.imshow(slope_deg, extent=extent, origin="upper", cmap="magma",
                          norm=Normalize(vmin=0.0, vmax=max(vmax, 1.0)))
    fig.colorbar(im2, ax=ax_slope, shrink=0.75, label="Slope (deg)")
    for layer in ("fairways", "greens", "bunkers", "water"):
        _overlay_vector(ax_slope, overlays.get(layer), dem_crs, facecolor="none", edgecolor="#222", linewidth=0.6)
    gpd.GeoSeries([hole_line], crs=dem_crs).plot(ax=ax_slope, color="cyan", linewidth=2.0)
    ax_slope.set_title("Slope")

    ax_prof = fig.add_subplot(gs[1, 0])
    base = float(np.nanmin(elevations)) if np.any(np.isfinite(elevations)) else 0.0
    ax_prof.plot(distances, elevations, color="darkgreen", linewidth=2.0)
    ax_prof.fill_between(distances, elevations, base, color="darkgreen", alpha=0.2)
    ax_prof.set_xlabel("Distance from tee (m)")
    ax_prof.set_ylabel("Elevation (m)")
    ax_prof.set_title("Tee -> Green elevation profile")
    ax_prof.grid(True, alpha=0.3)

    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis("off")

    def _fmt(v, suffix=""):
        return f"{v}{suffix}" if v is not None else "-"

    rows = [
        ("Course", s.get("course_name")),
        ("Hole", f"#{s.get('hole_number')}  {s.get('hole_name') or ''}"),
        ("Par / Handicap", f"{_fmt(s.get('par'))}  /  {_fmt(s.get('handicap'))}"),
        ("Length", f"{_fmt(s.get('hole_length_m'),' m')}   ({_fmt(s.get('hole_length_yd'),' yd')})"),
        ("Tee elev", _fmt(s.get("tee_elevation_m"), " m")),
        ("Green elev", _fmt(s.get("green_elevation_m"), " m")),
        ("Net change", _fmt(s.get("net_elevation_change_m"), " m")),
        ("Elev min/max/mean",
         f"{_fmt(s.get('min_elevation_m'))} / {_fmt(s.get('max_elevation_m'))} / {_fmt(s.get('mean_elevation_m'))} m"),
        ("Avg slope", f"{_fmt(s.get('avg_slope_deg'),'deg')}  ({_fmt(s.get('avg_slope_percent'),'%')})"),
        ("Max slope", f"{_fmt(s.get('max_slope_deg'),'deg')}  ({_fmt(s.get('max_slope_percent'),'%')})"),
        ("DEM", f"{s.get('dem_type')}  ({s.get('dem_source')})"),
    ]
    ax_stats.text(0.0, 1.02, f"{s.get('course_name', '')} - Hole {s.get('hole_number')}",
                  fontsize=13, fontweight="bold", transform=ax_stats.transAxes)
    y = 0.92
    for label, value in rows:
        ax_stats.text(0.02, y, f"{label}:", fontsize=10, fontweight="bold", transform=ax_stats.transAxes)
        ax_stats.text(0.42, y, str(value), fontsize=10, transform=ax_stats.transAxes)
        y -= 0.085

    fig.suptitle("Hole overview", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Course synthesis
# ---------------------------------------------------------------------------


def plot_course_overview(per_hole_data: list[dict], course_name: str, out_path: Path) -> None:
    data = sorted(per_hole_data, key=lambda d: _summary_dict(d["summary"])["hole_number"])
    n = len(data)
    if n == 0:
        return
    if n == 18:
        ncols, nrows = 6, 3
    elif n == 9:
        ncols, nrows = 3, 3
    else:
        ncols = min(6, n)
        nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.5))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for idx, d in enumerate(data):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        _draw_plan_view(ax, d["dem"], d["transform"], d["dem_crs"], d["overlays"],
                        d["hole_line"], show_legend=False, show_colorbar=False, fig=fig)
        s = _summary_dict(d["summary"])
        title = f"Hole {s['hole_number']} - {s.get('hole_name','') or ''}"
        subtitle = (f"Par {s.get('par','?')} . {s.get('hole_length_yd','?')} yd . "
                    f"d {s.get('net_elevation_change_m','?')} m . "
                    f"max slope {s.get('max_slope_deg','?')} deg")
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"{course_name} - Course Overview", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
