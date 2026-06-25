from __future__ import annotations

import math

import numpy as np

from pipeline.features.transforms import (
    aligned_arrays,
    aligned_scalar,
    alignment_angle,
    relative_arrays,
    relative_scalar,
)


def test_tee_maps_to_origin():
    tee = (1234.5, 6789.0, 100.0)
    xr, yr, zr = relative_scalar(*tee, *tee)
    assert (xr, yr, zr) == (0.0, 0.0, 0.0)


def test_z_relative_is_absolute_minus_tee():
    z_abs, tee_z = 218.0, 227.0
    _, _, zr = relative_scalar(0, 0, z_abs, 0, 0, tee_z)
    assert zr == z_abs - tee_z


def test_green_aligned_y_positive_x_zero():
    tee = (10.0, 20.0)
    green = (40.0, 220.0)  # offset (30, 200)
    angle = alignment_angle(tee[0], tee[1], green[0], green[1])
    xr, yr, _ = relative_arrays(
        np.array([green[0]]), np.array([green[1]]), np.array([0.0]),
        tee[0], tee[1], 0.0,
    )
    xa, ya = aligned_arrays(xr, yr, angle)
    assert abs(float(xa[0])) < 1e-6
    length = math.hypot(30.0, 200.0)
    assert float(ya[0]) > 0
    assert abs(float(ya[0]) - length) < 1e-6


def test_aligned_scalar_matches_arrays():
    angle = alignment_angle(0, 0, 0, 100)  # already +Y
    xa, ya = aligned_scalar(5.0, 12.0, angle)
    assert abs(xa - 5.0) < 1e-9 and abs(ya - 12.0) < 1e-9
