"""Stage 7d — read DEM arrays, sample points, and enumerate cell centers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from shapely.geometry import LineString


def read_dem_masked(path: Path):
    """Return (array float64, transform, crs) with nodata/non-finite -> NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr, transform, crs


def raster_resolution_m(transform) -> float:
    """Average absolute pixel size in CRS units (meters for a projected DEM)."""
    return (abs(transform.a) + abs(transform.e)) / 2.0


def sample_raster_at_points(dem_path: Path, points_xy: list[tuple[float, float]]) -> list[float]:
    """Sample elevations at (x, y) in the raster CRS; nodata/non-finite -> NaN."""
    with rasterio.open(dem_path) as src:
        samples = list(src.sample(points_xy))
        nodata = src.nodata
    out: list[float] = []
    for s in samples:
        v = float(s[0])
        if (nodata is not None and v == nodata) or not np.isfinite(v):
            out.append(float("nan"))
        else:
            out.append(v)
    return out


def build_elevation_profile(dem_path: Path, line: LineString, n_samples: int = 200):
    """Return (distances_m, elevations_m) along the centerline."""
    total = float(line.length)
    distances = np.linspace(0.0, total, n_samples)
    points = [line.interpolate(d) for d in distances]
    xy = [(p.x, p.y) for p in points]
    elevations = np.array(sample_raster_at_points(dem_path, xy), dtype="float64")
    return distances, elevations


def stride_for_resolution(transform, target_resolution_m: float) -> int:
    """Pixel stride so output spacing ≈ target resolution (never sub-pixel).

    If the DEM is coarser than the requested resolution, stride is 1 — we do not
    invent precision finer than the raster.
    """
    res = raster_resolution_m(transform)
    if res <= 0:
        return 1
    return max(1, int(round(target_resolution_m / res)))


def cell_center_points(dem: np.ndarray, transform, stride: int = 1):
    """Yield arrays (xs, ys, zs) of finite DEM cell centers at the given stride.

    Coordinates are in the DEM's (metric) CRS. Cell center for pixel (row, col):
        x = c + (col + 0.5) * a
        y = f + (row + 0.5) * e
    """
    h, w = dem.shape
    rows = np.arange(0, h, stride)
    cols = np.arange(0, w, stride)
    sub = dem[np.ix_(rows, cols)]
    finite = np.isfinite(sub)

    col_grid, row_grid = np.meshgrid(cols, rows)  # shape (len(rows), len(cols))
    xs = transform.c + (col_grid + 0.5) * transform.a
    ys = transform.f + (row_grid + 0.5) * transform.e

    return xs[finite], ys[finite], sub[finite]
