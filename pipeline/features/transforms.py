"""Coordinate transforms: tee-relative translation and tee->green alignment.

Two coordinate systems are produced for every point:

* **tee-relative** — translate so the selected tee anchor is ``(0, 0, 0)``.
* **aligned**       — additionally rotate the XY plane so the tee->green vector
  points toward **+Y**. Invariant: the green maps to ``x_aligned ~= 0`` and
  ``y_aligned > 0``.

The rotation angle is ``atan2(dx, dy)`` (measured from +Y), and:

    x_aligned = x_rel * cos(angle) - y_rel * sin(angle)
    y_aligned = x_rel * sin(angle) + y_rel * cos(angle)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def alignment_angle(tee_x: float, tee_y: float, green_x: float, green_y: float) -> float:
    """Angle (radians) of the tee->green vector measured from the +Y axis."""
    dx = green_x - tee_x
    dy = green_y - tee_y
    return math.atan2(dx, dy)


def relative_scalar(x, y, z, tee_x, tee_y, tee_z) -> Tuple[float, float, float]:
    return (x - tee_x, y - tee_y, z - tee_z)


def aligned_scalar(x_rel, y_rel, angle) -> Tuple[float, float]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x_rel * cos_a - y_rel * sin_a, x_rel * sin_a + y_rel * cos_a)


def relative_arrays(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    tee_x: float, tee_y: float, tee_z: float,
):
    return (xs - tee_x, ys - tee_y, zs - tee_z)


def aligned_arrays(x_rel: np.ndarray, y_rel: np.ndarray, angle: float):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (x_rel * cos_a - y_rel * sin_a, x_rel * sin_a + y_rel * cos_a)
