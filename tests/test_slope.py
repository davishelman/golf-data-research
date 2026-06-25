from __future__ import annotations

import math

import numpy as np
from rasterio.transform import Affine

from pipeline.raster.slope import compute_slope

# 1 m pixels, north-up.
_T = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def test_flat_raster_zero_slope():
    dem = np.full((20, 20), 50.0)
    slope_deg, slope_pct = compute_slope(dem, _T)
    assert np.allclose(slope_deg, 0.0)
    assert np.allclose(slope_pct, 0.0)


def test_ramp_raster_known_slope():
    # z increases 0.1 m per meter in +x  => 10% grade, atan(0.1) deg.
    cols = np.arange(40)
    dem = np.tile(cols * 0.1, (40, 1)).astype("float64")
    slope_deg, slope_pct = compute_slope(dem, _T)
    interior = slope_deg[1:-1, 1:-1]
    assert np.allclose(interior, math.degrees(math.atan(0.1)), atol=1e-6)
    assert np.allclose(slope_pct[1:-1, 1:-1], 10.0, atol=1e-6)
