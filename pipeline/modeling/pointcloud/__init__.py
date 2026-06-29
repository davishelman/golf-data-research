"""v2.5 surface-aware point-cloud hole similarity (additive to v2).

This package compares golf holes by their *normalized point clouds* — tee-
relative, green-aligned points grouped by surface (fairway, green, bunker,
water, tee) — using a symmetric Chamfer distance per surface, combined with
config-driven surface weights plus yardage/elevation penalties.

It is intentionally decoupled from the v2 feature-table similarity
(:mod:`pipeline.modeling.similarity`): different artifacts, its own id space
(``course_slug:hole_number``), and its own configs under
``configs/similarity/``. Nothing here mutates v2 outputs.

Separation of concerns:

* :mod:`.schemas`           — data contracts (metadata, points, results).
* :mod:`.config`            — YAML config model, validation, deterministic hash.
* :mod:`.candidate_filter`  — cheap par/yardage/surface gating (no geometry).
* :mod:`.chamfer`           — symmetric Chamfer distance primitive.
* :mod:`.score`             — surface-weighted scoring over clean inputs.
* :mod:`.export_similarity` — loader seam + CLI orchestration.
"""

from __future__ import annotations

#: v2.5 model family version tag (configs may pin a more specific value).
MODEL_VERSION: str = "v2_5_chamfer_v1"

from .candidate_filter import filter_candidate  # noqa: E402
from .config import PointCloudSimilarityConfig, load_config  # noqa: E402
from .schemas import (  # noqa: E402
    KNOWN_SURFACES,
    CandidateFilterResult,
    HoleMetadata,
    SimilarityResult,
    SurfacePoint,
    make_pc_hole_id,
    parse_pc_hole_id,
)
from .score import score_pair  # noqa: E402

__all__ = [
    "MODEL_VERSION",
    "KNOWN_SURFACES",
    "PointCloudSimilarityConfig",
    "load_config",
    "filter_candidate",
    "score_pair",
    "HoleMetadata",
    "SurfacePoint",
    "SimilarityResult",
    "CandidateFilterResult",
    "make_pc_hole_id",
    "parse_pc_hole_id",
]
