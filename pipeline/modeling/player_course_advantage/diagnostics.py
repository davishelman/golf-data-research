"""Explanation / diagnostics outputs for the advantage scorer (issue #39).

The scorer (:mod:`.scorer`) answers *how much* edge a player has on a course;
this companion module answers *why* — which target holes, which similar holes,
and which seasons drove the number, plus where coverage was too thin to trust.

It is a **read-only companion**: it re-derives the same row-level contributions
the scorer aggregates (identical join and ``similarity_weight · recency_weight``
weighting), so its breakdowns reconcile exactly with
:func:`.scorer.score_player_holes` / :func:`.scorer.score_player_course`. It adds
**no** new modelling and keeps scorer internals free of presentation concerns.

Everything returned is a tidy :class:`pandas.DataFrame` or a plain dict, so a
notebook or a (future) Streamlit view can render it without importing this
module's internals. Nothing here imports streamlit or touches v2/v2.5 scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .schema import (
    DEFAULT_PARAMS,
    FIELD_ADJUSTED_COL,
    AdvantageParams,
    add_field_adjusted_advantage,
    validate_hole_score_history,
)
from .scorer import (
    filter_history_for_prediction_window,
    recency_weight,
    score_player_course,
)

#: Columns of the atomic per-occurrence contribution frame.
CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "target_hole_id",
    "target_hole_number",
    "candidate_hole_id",
    "candidate_course_slug",
    "year",
    "round",
    "tournament_id",
    "similarity_weight",
    "recency_weight",
    "row_weight",
    FIELD_ADJUSTED_COL,
    "contribution",
)


@dataclass(frozen=True)
class PlayerCourseExplanation:
    """A player's advantage broken down into reconciling pieces.

    Every frame is safe to serialize (``.to_dict(orient="records")``). The
    ``*_contributions`` frames reconcile with the scorer: within each covered
    target hole the ``weighted_contribution`` values of its similar holes sum to
    that hole's ``hole_advantage``, and the covered holes' advantages sum to
    ``course_summary["course_advantage"]`` under ``sum`` aggregation.
    """

    player_id: str
    player_name: Optional[str]
    target_course_slug: str
    config_name: str
    course_summary: dict
    #: Per target hole: advantage, coverage, and (sum-)contribution to the course.
    hole_contributions: pd.DataFrame
    #: Per (target hole, similar hole): weight share and weighted contribution.
    similar_hole_contributions: pd.DataFrame
    #: Occurrence + weight counts by (target hole, season).
    occurrence_year_counts: pd.DataFrame
    #: Target holes that were withheld, with their reason.
    low_coverage: pd.DataFrame

    def top_target_holes(self, k: int = 5) -> pd.DataFrame:
        """The ``k`` covered target holes with the largest |contribution|."""
        covered = self.hole_contributions[
            self.hole_contributions["course_contribution"].notna()
        ]
        return (
            covered.reindex(
                covered["course_contribution"].abs().sort_values(ascending=False).index
            )
            .head(k)
            .reset_index(drop=True)
        )

    def top_similar_holes(self, k: int = 5) -> pd.DataFrame:
        """The ``k`` (target, similar) pairs with the largest |weighted contribution|."""
        sc = self.similar_hole_contributions
        if sc.empty:
            return sc
        return (
            sc.reindex(sc["weighted_contribution"].abs().sort_values(ascending=False).index)
            .head(k)
            .reset_index(drop=True)
        )

    def to_records(self) -> dict:
        """Fully serializable form (dict of records + the course summary dict)."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "target_course_slug": self.target_course_slug,
            "config_name": self.config_name,
            "course_summary": self.course_summary,
            "hole_contributions": self.hole_contributions.to_dict("records"),
            "similar_hole_contributions": self.similar_hole_contributions.to_dict("records"),
            "occurrence_year_counts": self.occurrence_year_counts.to_dict("records"),
            "low_coverage": self.low_coverage.to_dict("records"),
        }


def _empty_contribution_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CONTRIBUTION_COLUMNS})


def contribution_rows(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    player_id: str,
    target_course_slug: str,
    predict_season: int,
    params: AdvantageParams = DEFAULT_PARAMS,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """One row per eligible occurrence on a similar hole — the atomic explainer.

    Applies exactly the scorer's eligibility (past-season window + optional
    same-course exclusion) and weighting (``row_weight = similarity_weight ·
    recency_weight``, ``contribution = row_weight · (field_avg - player_score)``).
    Summing ``contribution`` and ``row_weight`` per target hole reproduces that
    hole's advantage. Returns :data:`CONTRIBUTION_COLUMNS`; never mutates inputs.

    ``validate=False`` skips :func:`validate_hole_score_history` (used internally
    when the caller has already validated).
    """
    if validate:
        validate_hole_score_history(history)

    sim = similar_holes
    if "target_course_slug" in sim.columns:
        sim = sim[sim["target_course_slug"] == target_course_slug]

    player_all = history[history["player_id"] == player_id]
    eligible = filter_history_for_prediction_window(
        player_all, predict_season, params.lookback_years
    )
    if not params.include_current_course_history:
        eligible = eligible[eligible["course_slug"] != target_course_slug]

    if eligible.empty or sim.empty:
        return _empty_contribution_frame()

    eligible = add_field_adjusted_advantage(eligible)  # returns a copy
    eligible["recency_weight"] = recency_weight(
        pd.to_numeric(eligible["year"], errors="coerce"),
        predict_season,
        params.recency_decay,
    )

    merged = sim.merge(
        eligible, left_on="candidate_hole_id", right_on="hole_id_v25", how="inner"
    )
    if merged.empty:
        return _empty_contribution_frame()

    merged["row_weight"] = merged["similarity_weight"] * merged["recency_weight"]
    merged["contribution"] = merged["row_weight"] * merged[FIELD_ADJUSTED_COL]

    present = [c for c in CONTRIBUTION_COLUMNS if c in merged.columns]
    return (
        merged[present]
        .sort_values(["target_hole_number", "target_hole_id", "candidate_hole_id", "year"])
        .reset_index(drop=True)
    )


def _similar_hole_contributions(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the atomic rows to one row per (target hole, similar hole)."""
    cols = [
        "target_hole_id", "target_hole_number", "candidate_hole_id",
        "candidate_course_slug", "occurrences", "n_years", "weight_sum",
        "contribution_sum", "mean_outcome", "fractional_weight", "weighted_contribution",
    ]
    if rows.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    grouped = (
        rows.groupby(
            ["target_hole_id", "target_hole_number", "candidate_hole_id"], sort=False
        )
        .agg(
            candidate_course_slug=("candidate_course_slug", "first")
            if "candidate_course_slug" in rows.columns
            else ("candidate_hole_id", "first"),
            occurrences=("row_weight", "size"),
            n_years=("year", "nunique"),
            weight_sum=("row_weight", "sum"),
            contribution_sum=("contribution", "sum"),
        )
        .reset_index()
    )
    grouped["mean_outcome"] = grouped["contribution_sum"] / grouped["weight_sum"]

    # Per-target-hole totals give each similar hole its share of the hole advantage:
    # weighted_contribution sums (over similar holes) to the hole advantage.
    hole_weight = grouped.groupby("target_hole_id")["weight_sum"].transform("sum")
    grouped["fractional_weight"] = grouped["weight_sum"] / hole_weight
    grouped["weighted_contribution"] = grouped["contribution_sum"] / hole_weight
    return grouped[cols].sort_values(
        ["target_hole_number", "target_hole_id", "weighted_contribution"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _occurrence_year_counts(rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["target_hole_id", "target_hole_number", "year", "occurrences", "weight_sum"]
    if rows.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return (
        rows.groupby(["target_hole_id", "target_hole_number", "year"], sort=False)
        .agg(occurrences=("row_weight", "size"), weight_sum=("row_weight", "sum"))
        .reset_index()
        .sort_values(["target_hole_number", "target_hole_id", "year"])
        .reset_index(drop=True)
    )


def explain_player_course(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    player_id: str,
    target_course_slug: str,
    predict_season: int,
    config_name: str = "baseline",
    params: AdvantageParams = DEFAULT_PARAMS,
    aggregate: str = "sum",
) -> PlayerCourseExplanation:
    """Explain a player's course advantage as reconciling, serializable pieces.

    Runs the scorer (which validates ``history`` and raises
    :class:`~pipeline.modeling.player_course_advantage.schema.SchemaError` on a
    contract violation) for the authoritative per-hole advantages and course
    summary, then attaches the similar-hole / per-season contribution breakdown
    and the low-coverage reasons. Never mutates the inputs.
    """
    per_hole, summary = score_player_course(
        history, similar_holes, player_id, target_course_slug, predict_season,
        config_name=config_name, params=params, aggregate=aggregate,
    )
    # history already validated by the scorer above.
    rows = contribution_rows(
        history, similar_holes, player_id, target_course_slug, predict_season,
        params, validate=False,
    )

    hole_contributions = per_hole.copy()
    # A covered hole's contribution to the (sum) course score is its advantage;
    # withheld holes contribute nothing (NaN, not 0). Ranking is aggregate-agnostic.
    hole_contributions["course_contribution"] = hole_contributions["hole_advantage"]

    low_coverage = (
        per_hole.loc[
            per_hole["low_coverage"],
            ["target_hole_id", "target_hole_number", "raw_occurrences",
             "similar_holes_used", "reason"],
        ]
        .reset_index(drop=True)
    )

    return PlayerCourseExplanation(
        player_id=player_id,
        player_name=summary.get("player_name"),
        target_course_slug=target_course_slug,
        config_name=config_name,
        course_summary=summary,
        hole_contributions=hole_contributions,
        similar_hole_contributions=_similar_hole_contributions(rows),
        occurrence_year_counts=_occurrence_year_counts(rows),
        low_coverage=low_coverage,
    )


__all__ = [
    "CONTRIBUTION_COLUMNS",
    "PlayerCourseExplanation",
    "contribution_rows",
    "explain_player_course",
]
