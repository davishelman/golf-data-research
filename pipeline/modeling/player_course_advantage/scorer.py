"""Recency-weighted player-course advantage scorer (issue #33).

The first *real* scorer for this layer. It consumes two things the earlier PRs
already produce and never touches similarity scoring itself:

1. a **historical player-by-hole score table** validated by
   :func:`pipeline.modeling.player_course_advantage.schema.validate_hole_score_history`, and
2. **v2.5 similar-hole sets** produced by the #32 loader
   (:func:`pipeline.modeling.player_course_advantage.similar_holes.load_similar_hole_sets`).

For player ``p`` and an upcoming target hole ``h`` on course ``C``::

    advantage(p, h) = Σ_{s, o}  sim_weight(h, s) · recency_weight(y(o)) · outcome(p, s, o)
                      ────────────────────────────────────────────────────────────────────
                      Σ_{s, o}  sim_weight(h, s) · recency_weight(y(o))

where ``s`` ranges over the similar holes of ``h``, ``o`` over the requested
player's historical occurrences on those similar holes, and the **positive-is-good**
outcome is ``outcome = field_avg_score - player_score`` (beating the field is
positive). The 18 hole advantages aggregate to a course number (``sum`` by
default → expected strokes-vs-field over a round; ``mean`` also available).

**Leakage guard.** Only strictly-past seasons inside the lookback window are
eligible: ``predict_season - W <= y(o) < predict_season`` (see
:func:`filter_history_for_prediction_window`). Same-course prior history is
excluded unless :attr:`AdvantageParams.include_current_course_history` is set.

Pure, deterministic, Streamlit-free, and free of real-PGA-data dependencies.
This is still **v0** — the defaults are uncalibrated and the model needs the
retrospective backtest (#35) before its numbers mean much. Batch field ranking
(#34), the backtest (#35), and parameter sweeps (#36) are intentionally deferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .schema import (
    DEFAULT_PARAMS,
    FIELD_ADJUSTED_COL,
    AdvantageParams,
    add_field_adjusted_advantage,
    validate_hole_score_history,
)

#: Recognized course-aggregation modes.
AGGREGATIONS: tuple[str, ...] = ("sum", "mean")

# Coverage / withholding reasons (stable strings for downstream filtering).
REASON_NO_PLAYER_HISTORY = "no_player_history"
REASON_NO_ELIGIBLE_HISTORY = "no_eligible_history"
REASON_NO_SIMILAR_HOLES = "no_similar_holes"
REASON_BELOW_MIN_OCCURRENCES = "below_min_occurrences"
REASON_BELOW_MIN_HOLES_COVERED = "below_min_holes_covered"

#: Columns a similar-hole set must carry for the scorer to consume it.
_REQUIRED_SIMILAR_COLUMNS: tuple[str, ...] = (
    "target_hole_id",
    "target_hole_number",
    "candidate_hole_id",
    "similarity_weight",
)

# Weights below this are treated as "no usable coverage" (guards a 0/0 mean when,
# e.g., recency_decay=0 zeroes every eligible weight).
_MIN_USABLE_WEIGHT = 1e-12


class AdvantageScorerError(ValueError):
    """Raised when scorer inputs are structurally unusable (bad similar-hole set,
    unknown aggregation, …). History-contract violations raise
    :class:`~pipeline.modeling.player_course_advantage.schema.SchemaError` instead.
    """


# --------------------------------------------------------------------------- #
# Output records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlayerHoleAdvantage:
    """One target hole's advantage for a player (a row of the per-hole output)."""

    player_id: str
    player_name: Optional[str]
    target_course_slug: str
    target_hole_id: str
    target_hole_number: int
    hole_advantage: Optional[float]
    weighted_occurrences: float
    raw_occurrences: int
    similar_holes_used: int
    low_coverage: bool
    reason: Optional[str]


@dataclass(frozen=True)
class PlayerCourseAdvantage:
    """A player's aggregate advantage over an upcoming course (course summary)."""

    player_id: str
    player_name: Optional[str]
    target_course_slug: str
    config_name: str
    course_advantage: Optional[float]
    course_advantage_mean: Optional[float]
    holes_covered: int
    total_target_holes: int
    total_raw_occurrences: int
    total_weighted_occurrences: float
    low_coverage: bool
    reason: Optional[str]


#: Column order of the per-hole output frame (matches PlayerHoleAdvantage fields).
HOLE_OUTPUT_COLUMNS: tuple[str, ...] = tuple(
    f.name for f in PlayerHoleAdvantage.__dataclass_fields__.values()
)


# --------------------------------------------------------------------------- #
# Recency weighting
# --------------------------------------------------------------------------- #
def recency_weight(
    occurrence_year: int,
    predict_season: int,
    recency_decay: float,
) -> float:
    """``recency_decay ** age`` where ``age = (predict_season - 1) - occurrence_year``.

    Age is measured from the *immediately prior* season because the target season
    itself is never eligible (:func:`filter_history_for_prediction_window`), so the
    most recent admissible season (``predict_season - 1``) has age 0 and full
    weight ``1.0``. ``recency_decay = 1`` disables decay. Works element-wise on
    pandas/NumPy inputs as well as scalars.

    >>> recency_weight(2023, 2024, 0.85)   # prior season -> weight 1.0
    1.0
    >>> round(recency_weight(2021, 2024, 0.85), 4)   # age 2 -> 0.85 ** 2
    0.7225
    """
    age = (predict_season - 1) - occurrence_year
    return float(recency_decay) ** age


# --------------------------------------------------------------------------- #
# Prediction-window filtering (leakage guard)
# --------------------------------------------------------------------------- #
def filter_history_for_prediction_window(
    history: pd.DataFrame,
    predict_season: int,
    lookback_years: int,
) -> pd.DataFrame:
    """Keep only rows eligible to predict ``predict_season``: no leakage.

    Eligible years satisfy ``predict_season - lookback_years <= year < predict_season``
    — strictly past seasons within the window. The target season and any future
    season are dropped. Returns a fresh copy; never mutates ``history``.
    """
    if lookback_years <= 0:
        raise AdvantageScorerError(
            f"lookback_years must be positive, got {lookback_years}"
        )
    year = pd.to_numeric(history["year"], errors="coerce")
    lo = predict_season - lookback_years
    mask = (year >= lo) & (year < predict_season)
    return history.loc[mask].copy()


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _HoleScoring:
    """Internal bundle shared by the hole- and course-level entry points."""

    per_hole: pd.DataFrame
    total_target_holes: int
    player_name: Optional[str]
    n_player_raw: int
    n_eligible: int


def _check_similar_holes(similar_holes: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_SIMILAR_COLUMNS if c not in similar_holes.columns]
    if missing:
        raise AdvantageScorerError(
            f"similar-hole set missing required columns: {missing}; "
            "expected output of load_similar_hole_sets()"
        )


def _empty_hole_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in HOLE_OUTPUT_COLUMNS})


def _compute_hole_scores(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    player_id: str,
    target_course_slug: str,
    predict_season: int,
    params: AdvantageParams,
) -> _HoleScoring:
    """Shared implementation behind :func:`score_player_holes` / :func:`score_player_course`."""
    # Fail loudly on an invalid history table (raises SchemaError) before any work.
    validate_hole_score_history(history)
    _check_similar_holes(similar_holes)

    # Target-hole universe: the holes of C we have similar-hole sets for.
    sim = similar_holes
    if "target_course_slug" in sim.columns:
        sim = sim[sim["target_course_slug"] == target_course_slug]
    target_holes = (
        sim[["target_hole_id", "target_hole_number"]]
        .drop_duplicates()
        .assign(target_hole_number=lambda d: d["target_hole_number"].astype(int))
        .sort_values(["target_hole_number", "target_hole_id"])
        .reset_index(drop=True)
    )
    total_target_holes = int(len(target_holes))

    # This player's full history (for no_player_history vs no_eligible_history).
    player_all = history[history["player_id"] == player_id]
    player_name = _player_name(player_all)
    n_player_raw = int(len(player_all))

    # Eligibility filters: past-season window, then optional same-course exclusion.
    eligible = filter_history_for_prediction_window(
        player_all, predict_season, params.lookback_years
    )
    if not params.include_current_course_history:
        eligible = eligible[eligible["course_slug"] != target_course_slug]
    n_eligible = int(len(eligible))

    if total_target_holes == 0:
        return _HoleScoring(
            _empty_hole_frame(), 0, player_name, n_player_raw, n_eligible
        )

    # Positive-is-good outcome + per-row recency weight on the eligible rows.
    if n_eligible:
        eligible = add_field_adjusted_advantage(eligible)  # returns a copy
        eligible["_recency_weight"] = recency_weight(
            pd.to_numeric(eligible["year"], errors="coerce"),
            predict_season,
            params.recency_decay,
        )

    agg = _aggregate_by_target_hole(sim, eligible, target_holes)

    # Global fallbacks: if the player has no usable history the whole course is
    # low-coverage for a single, clear reason.
    global_reason: Optional[str] = None
    if n_player_raw == 0:
        global_reason = REASON_NO_PLAYER_HISTORY
    elif n_eligible == 0:
        global_reason = REASON_NO_ELIGIBLE_HISTORY

    rows = [
        _build_hole_record(
            r, player_id, player_name, target_course_slug, params, global_reason
        )
        for r in agg.itertuples(index=False)
    ]
    per_hole = pd.DataFrame(
        [asdict(r) for r in rows], columns=list(HOLE_OUTPUT_COLUMNS)
    )
    return _HoleScoring(
        per_hole, total_target_holes, player_name, n_player_raw, n_eligible
    )


def _aggregate_by_target_hole(
    sim: pd.DataFrame, eligible: pd.DataFrame, target_holes: pd.DataFrame
) -> pd.DataFrame:
    """Left-join weighted occurrences onto every target hole (0-filled when absent)."""
    if len(eligible):
        merged = sim.merge(
            eligible,
            left_on="candidate_hole_id",
            right_on="hole_id_v25",
            how="inner",
        )
    else:
        merged = sim.iloc[0:0].copy()

    if len(merged):
        merged = merged.assign(
            _row_weight=merged["similarity_weight"] * merged["_recency_weight"],
        )
        merged["_contrib"] = merged["_row_weight"] * merged[FIELD_ADJUSTED_COL]
        grouped = (
            merged.groupby("target_hole_id")
            .agg(
                weight_sum=("_row_weight", "sum"),
                contrib_sum=("_contrib", "sum"),
                raw_occurrences=("_row_weight", "size"),
                similar_holes_used=("candidate_hole_id", "nunique"),
            )
            .reset_index()
        )
    else:
        # Explicit dtypes so the left-join + fillna below stays numeric (avoids a
        # pandas object-dtype downcast warning when the player has no matches).
        grouped = pd.DataFrame({
            "target_hole_id": pd.Series(dtype="object"),
            "weight_sum": pd.Series(dtype="float"),
            "contrib_sum": pd.Series(dtype="float"),
            "raw_occurrences": pd.Series(dtype="int"),
            "similar_holes_used": pd.Series(dtype="int"),
        })

    out = target_holes.merge(grouped, on="target_hole_id", how="left")
    out["weight_sum"] = out["weight_sum"].fillna(0.0)
    out["contrib_sum"] = out["contrib_sum"].fillna(0.0)
    out["raw_occurrences"] = out["raw_occurrences"].fillna(0).astype(int)
    out["similar_holes_used"] = out["similar_holes_used"].fillna(0).astype(int)
    return out


def _build_hole_record(
    row,
    player_id: str,
    player_name: Optional[str],
    target_course_slug: str,
    params: AdvantageParams,
    global_reason: Optional[str],
) -> PlayerHoleAdvantage:
    """Turn one aggregated target-hole row into a coverage-gated output record."""
    raw = int(row.raw_occurrences)
    weight_sum = float(row.weight_sum)
    has_weight = weight_sum > _MIN_USABLE_WEIGHT
    low_coverage = (
        raw < params.min_occurrences_per_hole or not has_weight
    )

    if low_coverage:
        advantage = None
        reason = global_reason or REASON_BELOW_MIN_OCCURRENCES
    else:
        advantage = float(row.contrib_sum / weight_sum)
        reason = None

    return PlayerHoleAdvantage(
        player_id=player_id,
        player_name=player_name,
        target_course_slug=target_course_slug,
        target_hole_id=str(row.target_hole_id),
        target_hole_number=int(row.target_hole_number),
        hole_advantage=advantage,
        weighted_occurrences=weight_sum,
        raw_occurrences=raw,
        similar_holes_used=int(row.similar_holes_used),
        low_coverage=low_coverage,
        reason=reason,
    )


def _player_name(player_rows: pd.DataFrame) -> Optional[str]:
    if "player_name" not in player_rows.columns:
        return None
    names = player_rows["player_name"].dropna()
    return str(names.iloc[0]) if not names.empty else None


def score_player_holes(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    player_id: str,
    target_course_slug: str,
    predict_season: int,
    params: AdvantageParams = DEFAULT_PARAMS,
) -> pd.DataFrame:
    """Per-target-hole advantage for one player on one upcoming course.

    Validates ``history`` (raises
    :class:`~pipeline.modeling.player_course_advantage.schema.SchemaError` on a
    contract violation), joins the player's eligible occurrences to each target
    hole's similar holes (``hole_id_v25 == candidate_hole_id``), and returns one
    row per target hole with :data:`HOLE_OUTPUT_COLUMNS`.

    Low-coverage holes (fewer than ``params.min_occurrences_per_hole`` raw
    occurrences) get ``hole_advantage = NaN`` and a ``reason`` — never a fabricated
    0. Rows are sorted deterministically by ``(target_hole_number, target_hole_id)``.
    Never mutates the inputs.
    """
    result = _compute_hole_scores(
        history, similar_holes, player_id, target_course_slug, predict_season, params
    )
    return result.per_hole


def score_player_course(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    player_id: str,
    target_course_slug: str,
    predict_season: int,
    config_name: str = "baseline",
    params: AdvantageParams = DEFAULT_PARAMS,
    aggregate: str = "sum",
) -> tuple[pd.DataFrame, dict]:
    """Aggregate the per-hole advantages into one player-course number.

    Returns ``(per_hole_df, course_summary)`` where ``per_hole_df`` is exactly
    :func:`score_player_holes`' output and ``course_summary`` is
    :class:`PlayerCourseAdvantage` as a dict (coverage diagnostics included).

    ``aggregate`` selects the headline ``course_advantage``: ``"sum"`` (default —
    expected strokes-vs-field over the round) or ``"mean"``. ``course_advantage_mean``
    is always the mean of covered holes. The course score is **withheld**
    (``None``, ``low_coverage=True``, ``reason=below_min_holes_covered``) when fewer
    than ``params.min_holes_covered`` holes are covered. Never mutates the inputs.
    """
    if aggregate not in AGGREGATIONS:
        raise AdvantageScorerError(
            f"unknown aggregate {aggregate!r}; expected one of {list(AGGREGATIONS)}"
        )

    result = _compute_hole_scores(
        history, similar_holes, player_id, target_course_slug, predict_season, params
    )
    per_hole = result.per_hole

    covered = per_hole[~per_hole["low_coverage"]] if len(per_hole) else per_hole
    holes_covered = int(len(covered))
    total_raw = int(per_hole["raw_occurrences"].sum()) if len(per_hole) else 0
    total_weighted = (
        float(per_hole["weighted_occurrences"].sum()) if len(per_hole) else 0.0
    )

    course_mean: Optional[float] = None
    course_headline: Optional[float] = None
    low_coverage = True
    reason: Optional[str]

    if result.total_target_holes == 0:
        reason = REASON_NO_SIMILAR_HOLES
    elif result.n_player_raw == 0:
        reason = REASON_NO_PLAYER_HISTORY
    elif result.n_eligible == 0:
        reason = REASON_NO_ELIGIBLE_HISTORY
    elif holes_covered < params.min_holes_covered:
        reason = REASON_BELOW_MIN_HOLES_COVERED
    else:
        adv = covered["hole_advantage"].astype(float)
        course_sum = float(adv.sum())
        course_mean = float(adv.mean())
        course_headline = course_sum if aggregate == "sum" else course_mean
        low_coverage = False
        reason = None

    summary = PlayerCourseAdvantage(
        player_id=player_id,
        player_name=result.player_name,
        target_course_slug=target_course_slug,
        config_name=config_name,
        course_advantage=course_headline,
        course_advantage_mean=course_mean,
        holes_covered=holes_covered,
        total_target_holes=result.total_target_holes,
        total_raw_occurrences=total_raw,
        total_weighted_occurrences=total_weighted,
        low_coverage=low_coverage,
        reason=reason,
    )
    return per_hole, asdict(summary)


__all__ = [
    "AGGREGATIONS",
    "REASON_NO_PLAYER_HISTORY",
    "REASON_NO_ELIGIBLE_HISTORY",
    "REASON_NO_SIMILAR_HOLES",
    "REASON_BELOW_MIN_OCCURRENCES",
    "REASON_BELOW_MIN_HOLES_COVERED",
    "HOLE_OUTPUT_COLUMNS",
    "AdvantageScorerError",
    "PlayerHoleAdvantage",
    "PlayerCourseAdvantage",
    "recency_weight",
    "filter_history_for_prediction_window",
    "score_player_holes",
    "score_player_course",
]
