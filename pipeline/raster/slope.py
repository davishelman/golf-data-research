"""Stage 7c — slope from a metric DEM array (degrees and percent)."""

from __future__ import annotations

import numpy as np


def compute_slope(dem: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Return (slope_degrees, slope_percent) arrays matching ``dem`` shape.

    Pixel sizes come from the affine transform, so the DEM must already be in a
    metric CRS for the result to be physically meaningful.
    """
    px = abs(transform.a)
    py = abs(transform.e)
    dz_dy, dz_dx = np.gradient(dem, py, px)
    slope_rise = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    slope_percent = slope_rise * 100.0
    slope_deg = np.degrees(np.arctan(slope_rise))
    return slope_deg, slope_percent
