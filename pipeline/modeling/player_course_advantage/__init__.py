"""Player-course advantage — a downstream model over v2.5 similar-hole sets.

This package answers a *different* question from v2/v2.5 similarity. v2 and v2.5
say **"which holes look alike?"**. This layer says **"given those look-alike
holes, is player p likely to have an edge on upcoming course C?"** — by looking
up how p historically scored on holes similar to each of C's 18 holes.

It is strictly additive and read-only with respect to v2/v2.5: it *consumes*
v2.5 similarity outputs (target hole -> ranked similar holes + ``total_score``)
and a historical player-by-hole scoring table, and never mutates either. See
:doc:`the spec </docs/player_course_advantage>` (``docs/player_course_advantage.md``)
for the full problem statement, notation, and formulas.

Layout (the scorer and backtest remain deferred):

* :mod:`.schema` — historical hole-score input contract, the positive-is-good
  field-adjusted outcome convention, default parameters (``n``, ``W``, ``m``),
  and lightweight validation (:func:`.schema.validate_hole_score_history`).
* :mod:`.similar_holes` — reads v2.5 similarity result CSVs into normalized
  per-target-hole similar-hole sets (:func:`.similar_holes.load_similar_hole_sets`).
* :mod:`.scorer` — the recency-weighted player-hole and player-course advantage
  scorer (:func:`.scorer.score_player_holes`, :func:`.scorer.score_player_course`).

Nothing here depends on real PGA data, streamlit, or the geometry/DEM stack.
"""

from __future__ import annotations

#: Model family version tag for this downstream advantage layer. Bumped when the
#: input contract or aggregation semantics change in a breaking way.
MODEL_VERSION: str = "player_course_advantage_v0"

from .schema import (  # noqa: E402
    DEFAULT_PARAMS,
    KEY_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    AdvantageParams,
    SchemaError,
    ValidationReport,
    add_field_adjusted_advantage,
    field_adjusted_advantage,
    validate_hole_score_history,
)
from .similar_holes import (  # noqa: E402
    WEIGHT_METHODS,
    SimilarHoleLoaderError,
    add_similarity_weights,
    available_target_hole_numbers,
    load_similar_hole_sets,
    missing_target_hole_numbers,
    parse_v25_hole_id,
)
from .scorer import (  # noqa: E402
    AGGREGATIONS,
    HOLE_OUTPUT_COLUMNS,
    AdvantageScorerError,
    PlayerCourseAdvantage,
    PlayerHoleAdvantage,
    filter_history_for_prediction_window,
    recency_weight,
    score_player_course,
    score_player_holes,
)

__all__ = [
    "MODEL_VERSION",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "KEY_COLUMNS",
    "DEFAULT_PARAMS",
    "AdvantageParams",
    "SchemaError",
    "ValidationReport",
    "field_adjusted_advantage",
    "add_field_adjusted_advantage",
    "validate_hole_score_history",
    # similar-hole loader (#32)
    "WEIGHT_METHODS",
    "SimilarHoleLoaderError",
    "parse_v25_hole_id",
    "add_similarity_weights",
    "load_similar_hole_sets",
    "available_target_hole_numbers",
    "missing_target_hole_numbers",
    # advantage scorer (#33)
    "AGGREGATIONS",
    "HOLE_OUTPUT_COLUMNS",
    "AdvantageScorerError",
    "PlayerHoleAdvantage",
    "PlayerCourseAdvantage",
    "recency_weight",
    "filter_history_for_prediction_window",
    "score_player_holes",
    "score_player_course",
]
