"""Symmetric Chamfer distance between two 3D point sets.

Used by the v2.5 scorer to compare a single surface (fairway, green, ...) of two
holes after both have been normalized into tee-relative, green-aligned space.

Backend
-------
``scipy.spatial.cKDTree`` is used for the nearest-neighbor queries; scipy is
already a project dependency (it ships with the geo/raster stack). A small,
exact NumPy brute-force fallback is provided for completeness and is selected
automatically if scipy is unavailable — it is O(Na*Nb) in memory, so it is only
appropriate for the modest per-surface point budgets configured in v2.5
(hundreds–low-thousands of points). The KD-tree path is preferred and is what
runs in practice.
"""

from __future__ import annotations

import importlib.util
from typing import Optional, Sequence, Union

import numpy as np

_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None

PointsLike = Union[np.ndarray, Sequence[Sequence[float]]]


class EmptySurfaceError(ValueError):
    """Raised when exactly one of the two point sets is empty.

    The scorer catches this condition by checking emptiness *before* calling
    :func:`symmetric_chamfer_distance`, applying a configured missing-surface
    penalty instead. The exception exists so direct callers fail loudly rather
    than silently returning a meaningless distance.
    """


def _as_xyz(points: PointsLike) -> np.ndarray:
    """Coerce input to a contiguous ``(N, 3)`` float64 array."""
    arr = np.asarray(points, dtype="float64")
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"expected points of shape (N, 3) with x, y, z columns; got {arr.shape}"
        )
    return arr


def _scale(points: np.ndarray, x_weight: float, y_weight: float, z_weight: float) -> np.ndarray:
    """Multiply each axis by its weight (cheap anisotropic distance shaping)."""
    return points * np.array([x_weight, y_weight, z_weight], dtype="float64")


def _mean_nn_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over points in ``a`` of the Euclidean distance to the nearest in ``b``."""
    if _SCIPY_AVAILABLE:
        from scipy.spatial import cKDTree  # local import keeps module import light

        tree = cKDTree(b)
        dists, _ = tree.query(a, k=1)
        return float(np.mean(dists))
    # Exact NumPy fallback: pairwise distances, min over b, mean over a.
    diff = a[:, None, :] - b[None, :, :]
    dists = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    return float(np.mean(dists.min(axis=1)))


def symmetric_chamfer_distance(
    points_a: PointsLike,
    points_b: PointsLike,
    x_weight: float,
    y_weight: float,
    z_weight: float,
) -> Optional[float]:
    """Symmetric Chamfer distance between two scaled 3D point sets.

    Each coordinate is scaled before distances are taken
    (``x_scaled = x * x_weight``, etc.), then the result is::

        0.5 * (mean_{a in A} min_{b in B} ||a - b||
               + mean_{b in B} min_{a in A} ||b - a||)

    Behavior at the edges:

    * both sets empty            -> ``0.0`` (nothing to distinguish).
    * exactly one set empty      -> raises :class:`EmptySurfaceError`; the scorer
      handles this via a configured missing-surface penalty.
    * identical point sets       -> ``0.0``.

    Lower is more similar.
    """
    a = _scale(_as_xyz(points_a), x_weight, y_weight, z_weight)
    b = _scale(_as_xyz(points_b), x_weight, y_weight, z_weight)

    a_empty, b_empty = a.shape[0] == 0, b.shape[0] == 0
    if a_empty and b_empty:
        return 0.0
    if a_empty or b_empty:
        raise EmptySurfaceError(
            "one point set is empty and the other is not; "
            "let the scorer apply a missing-surface penalty instead"
        )

    return 0.5 * (_mean_nn_distance(a, b) + _mean_nn_distance(b, a))


__all__ = ["symmetric_chamfer_distance", "EmptySurfaceError"]
