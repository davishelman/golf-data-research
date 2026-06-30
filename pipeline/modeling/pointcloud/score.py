"""Surface-aware point-cloud scoring for v2.5 similarity.

Consumes *clean* inputs only — two :class:`HoleMetadata` and their already
normalized :class:`SurfacePoint` lists — and a validated
:class:`PointCloudSimilarityConfig`. It does not download OSM, read DEMs, or
parse course APIs; producing the normalized points is the upstream pipeline's
job (see :mod:`pipeline.modeling.pointcloud.export_similarity` for the loader
seam).

Scoring model
-------------
For every weighted surface ``s`` with weight ``w_s``:

* both holes have points -> ``score_s`` = surface Chamfer distance.
* exactly one has points  -> ``score_s`` = ``surface_missing_penalties[s]`` and
  that contribution is also tracked in ``missing_surface_penalty``.
* neither has points       -> surface skipped (component ``None``, no weight).

Then::

    total = sum_s (w_s * score_s)
            + yardage_penalty                # |Δyards| * yardage_weight
            + elevation_penalty              # |Δ(green-tee elev)| * elev_weight

Lower ``total_score`` means more similar.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

from ...logging_config import get_logger
from .candidate_filter import REASON_PASS
from .chamfer import symmetric_chamfer_distance
from .config import PointCloudSimilarityConfig
from .schemas import HoleMetadata, SimilarityResult, SurfacePoint

log = get_logger("modeling.pointcloud.score")


def group_points_by_surface(points: Iterable[SurfacePoint]) -> dict[str, np.ndarray]:
    """Group ``SurfacePoint``s into ``{surface: (N, 3) xyz array}``."""
    buckets: dict[str, list[list[float]]] = defaultdict(list)
    for p in points:
        buckets[p.surface].append([p.x_lateral_m, p.y_down_hole_m, p.z_relative_m])
    return {s: np.asarray(rows, dtype="float64") for s, rows in buckets.items()}


def _apply_budget(xyz: np.ndarray, budget: Optional[int]) -> np.ndarray:
    """Deterministically down-sample to ``budget`` points (even linspace pick).

    Matches the subsampling style used elsewhere in the pipeline so results are
    reproducible regardless of input ordering density.
    """
    if budget is None or budget <= 0 or xyz.shape[0] <= budget:
        return xyz
    sel = np.unique(np.linspace(0, xyz.shape[0] - 1, budget).astype(int))
    return xyz[sel]


def _elevation_delta(meta: HoleMetadata) -> Optional[float]:
    """Tee->green elevation change for a hole, or ``None`` if either is missing."""
    if meta.tee_elevation_m is None or meta.green_elevation_m is None:
        return None
    return float(meta.green_elevation_m) - float(meta.tee_elevation_m)


def score_pair(
    target: HoleMetadata,
    candidate: HoleMetadata,
    target_points: Iterable[SurfacePoint],
    candidate_points: Iterable[SurfacePoint],
    config: PointCloudSimilarityConfig,
    *,
    filter_reason: str = REASON_PASS,
) -> SimilarityResult:
    """Score one ``target`` against one ``candidate`` and return a result row.

    Assumes the pair has already passed candidate filtering; ``filter_reason`` is
    stored verbatim for provenance. Raw points are never copied into the result.
    """
    ds = config.distance_scaling
    target_by_surface = group_points_by_surface(target_points)
    cand_by_surface = group_points_by_surface(candidate_points)

    surface_scores: dict[str, Optional[float]] = {}
    weighted_total = 0.0
    missing_penalty_total = 0.0

    for surface in config.weighted_surfaces():
        weight = config.surface_weights[surface]
        a = _apply_budget(target_by_surface.get(surface, _empty()), config.point_budgets.get(surface))
        b = _apply_budget(cand_by_surface.get(surface, _empty()), config.point_budgets.get(surface))
        a_has, b_has = a.shape[0] > 0, b.shape[0] > 0

        if a_has and b_has:
            score = symmetric_chamfer_distance(
                a, b, ds.x_weight, ds.y_weight, ds.z_weight
            )
        elif a_has != b_has:
            score = config.surface_missing_penalties.get(surface, 0.0)
            missing_penalty_total += weight * score
        else:  # neither side has this surface — skip it entirely.
            surface_scores[surface] = None
            continue

        surface_scores[surface] = score
        weighted_total += weight * float(score)

    yardage_penalty = abs(float(target.yards) - float(candidate.yards)) * config.penalties.yardage_weight

    elevation_penalty = 0.0
    t_delta, c_delta = _elevation_delta(target), _elevation_delta(candidate)
    if t_delta is not None and c_delta is not None:
        elevation_penalty = abs(t_delta - c_delta) * config.penalties.tee_to_green_elevation_weight

    total_score = weighted_total + yardage_penalty + elevation_penalty

    return SimilarityResult(
        model_version=config.model_version,
        config_name=config.config_name,
        config_hash=config.config_hash,
        target_hole_id=target.hole_id,
        candidate_hole_id=candidate.hole_id,
        total_score=total_score,
        yardage_penalty=yardage_penalty,
        elevation_penalty=elevation_penalty,
        missing_surface_penalty=missing_penalty_total,
        filter_reason=filter_reason,
        fairway_score=surface_scores.get("fairway"),
        green_score=surface_scores.get("green"),
        bunker_score=surface_scores.get("bunker"),
        water_score=surface_scores.get("water"),
        tee_score=surface_scores.get("tee"),
        surface_scores=surface_scores,
    )


def _empty() -> np.ndarray:
    return np.empty((0, 3), dtype="float64")


__all__ = ["score_pair", "group_points_by_surface"]
