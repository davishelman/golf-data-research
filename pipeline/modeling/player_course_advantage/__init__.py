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

Layout (this first PR ships only the spec + schema; the scorer is deferred):

* :mod:`.schema` — historical hole-score input contract, the positive-is-good
  field-adjusted outcome convention, default parameters (``n``, ``W``, ``m``),
  and lightweight validation (:func:`.schema.validate_hole_score_history`).

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
]
