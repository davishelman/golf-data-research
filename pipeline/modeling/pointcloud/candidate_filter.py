"""Cheap pre-scoring candidate filtering for v2.5 point-cloud similarity.

Filtering is deliberately separate from scoring: it answers "is this candidate
even worth running an (expensive) Chamfer comparison against?" using only
:class:`~pipeline.modeling.pointcloud.schemas.HoleMetadata`. No geometry is
touched here.

Reasons are stable string codes so they can be stored and grouped downstream:

* ``PASS``                     — candidate is eligible for scoring.
* ``DIFFERENT_PAR``            — same-par required and pars differ.
* ``MISSING_REQUIRED_SURFACE`` — target or candidate lacks a required surface.
* ``YARDAGE_TOO_DIFFERENT``    — yardage gap exceeds the per-par window.
* ``NO_YARDAGE_WINDOW``        — no window configured for this par bucket.
"""

from __future__ import annotations

from .config import PointCloudSimilarityConfig
from .schemas import CandidateFilterResult, HoleMetadata

REASON_PASS = "PASS"
REASON_DIFFERENT_PAR = "DIFFERENT_PAR"
REASON_MISSING_REQUIRED_SURFACE = "MISSING_REQUIRED_SURFACE"
REASON_YARDAGE_TOO_DIFFERENT = "YARDAGE_TOO_DIFFERENT"
REASON_NO_YARDAGE_WINDOW = "NO_YARDAGE_WINDOW"


def filter_candidate(
    target: HoleMetadata,
    candidate: HoleMetadata,
    config: PointCloudSimilarityConfig,
) -> CandidateFilterResult:
    """Decide whether ``candidate`` is eligible to be scored against ``target``.

    Order of checks (first failure wins):

    1. If same par is required and pars differ -> ``DIFFERENT_PAR``.
    2. If any required surface is missing on target or candidate ->
       ``MISSING_REQUIRED_SURFACE``.
    3. Yardage window for the target's par; if none configured ->
       ``NO_YARDAGE_WINDOW``.
    4. If ``|candidate.yards - target.yards|`` exceeds the window ->
       ``YARDAGE_TOO_DIFFERENT``.
    5. Otherwise ``PASS``.
    """
    cf = config.candidate_filter

    if cf.require_same_par and int(target.par) != int(candidate.par):
        return CandidateFilterResult(False, REASON_DIFFERENT_PAR)

    for surface in config.required_surfaces:
        if not target.has_surface(surface) or not candidate.has_surface(surface):
            return CandidateFilterResult(False, REASON_MISSING_REQUIRED_SURFACE)

    window = cf.window_for_par(target.par)
    if window is None:
        return CandidateFilterResult(False, REASON_NO_YARDAGE_WINDOW)

    allowed = window.allowed_diff(target.yards)
    if abs(float(candidate.yards) - float(target.yards)) > allowed:
        return CandidateFilterResult(False, REASON_YARDAGE_TOO_DIFFERENT)

    return CandidateFilterResult(True, REASON_PASS)


__all__ = [
    "filter_candidate",
    "REASON_PASS",
    "REASON_DIFFERENT_PAR",
    "REASON_MISSING_REQUIRED_SURFACE",
    "REASON_YARDAGE_TOO_DIFFERENT",
    "REASON_NO_YARDAGE_WINDOW",
]
