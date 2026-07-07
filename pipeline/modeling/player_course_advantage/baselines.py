"""Simple baseline scorers for comparison (issue #38).

The similar-hole advantage model only earns its keep if it beats dumb baselines.
This module provides several, each a **drop-in field ranker** with the same
signature and output shape as :func:`.batch.score_tournament_field`, so the #35
backtest can evaluate them on the exact same events and metrics
(:func:`compare_baselines`).

Baselines (all leakage-guarded by the same prediction window as the model —
``predict_season - W <= year < predict_season``):

* :func:`null_baseline` — field average / no signal (everyone equal). Reference.
* :func:`recent_form_baseline` — mean field-adjusted outcome over *all* recent
  eligible holes, ignoring hole similarity.
* :func:`course_history_baseline` — mean field-adjusted outcome on the target
  course itself in prior seasons.
* :func:`same_par_baseline` — mean field-adjusted outcome on holes sharing par
  with the target course's holes.
* :func:`season_average_baseline` — player's raw recent scoring average
  (``-mean(player_score)``), no field adjustment.

A v2 feature-similarity baseline is intentionally **not** implemented here (it
would couple this layer to v2 internals; see the issue's "if practical").

These use a deliberately **simpler, course-level coverage rule** than the
per-hole model: a player is covered when they have at least
``params.min_occurrences_per_hole`` eligible rows the baseline can use. Pure,
deterministic, Streamlit-free.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import pandas as pd

from .batch import FIELD_RANKING_COLUMNS, rank_field, score_tournament_field
from .batch import _field_players  # normalize list/DataFrame fields the same way
from .backtest import run_backtest
from .scorer import (
    REASON_BELOW_MIN_OCCURRENCES,
    REASON_NO_ELIGIBLE_HISTORY,
    REASON_NO_PLAYER_HISTORY,
    filter_history_for_prediction_window,
)
from .schema import (
    DEFAULT_PARAMS,
    FIELD_ADJUSTED_COL,
    AdvantageParams,
    add_field_adjusted_advantage,
    validate_hole_score_history,
)

Field = Union[pd.DataFrame, Sequence[str]]
Ranker = Callable[..., pd.DataFrame]


def _sim_for_course(similar_holes: pd.DataFrame, course: str) -> pd.DataFrame:
    if "target_course_slug" in similar_holes.columns:
        return similar_holes[similar_holes["target_course_slug"] == course]
    return similar_holes


def _total_target_holes(similar_holes: pd.DataFrame, course: str) -> int:
    sim = _sim_for_course(similar_holes, course)
    return int(sim["target_hole_id"].nunique()) if "target_hole_id" in sim.columns else 0


def _target_pars(history: pd.DataFrame, similar_holes: pd.DataFrame, course: str) -> set[int]:
    """Pars of the target course's holes, inferred from candidate-hole pars.

    v2.5 candidates share par with their target hole, so the target-course par set
    equals the pars (from ``history``) of the candidate holes.
    """
    sim = _sim_for_course(similar_holes, course)
    cands = set(sim["candidate_hole_id"].astype(str)) if "candidate_hole_id" in sim.columns else set()
    if not cands:
        return set()
    par_by_hole = (
        history.dropna(subset=["hole_id_v25", "par"])
        .drop_duplicates("hole_id_v25")
        .set_index("hole_id_v25")["par"]
    )
    return {
        int(par_by_hole.loc[c]) for c in cands if c in par_by_hole.index
    }


def _baseline_rank(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    field: Field,
    target_course_slug: str,
    predict_season: int,
    *,
    config_name: str,
    params: AdvantageParams,
    select: Callable[[pd.DataFrame], pd.DataFrame],
    score: Callable[[pd.DataFrame], float],
) -> pd.DataFrame:
    """Shared machinery: per-player scalar score + model-shaped ranking table.

    ``select`` narrows a player's field-adjusted eligible history to the rows the
    baseline uses; ``score`` maps those rows to a higher-is-better scalar.
    """
    validate_hole_score_history(history)
    player_ids, field_names = _field_players(field)
    total_target = _total_target_holes(similar_holes, target_course_slug)

    rows: list[dict] = []
    for pid in player_ids:
        player_all = history[history["player_id"] == pid]
        eligible = filter_history_for_prediction_window(
            player_all, predict_season, params.lookback_years
        )
        used = select(add_field_adjusted_advantage(eligible)) if len(eligible) else eligible
        raw = int(len(used))

        if len(player_all) == 0:
            advantage, low, reason = None, True, REASON_NO_PLAYER_HISTORY
        elif len(eligible) == 0:
            advantage, low, reason = None, True, REASON_NO_ELIGIBLE_HISTORY
        elif raw < params.min_occurrences_per_hole:
            advantage, low, reason = None, True, REASON_BELOW_MIN_OCCURRENCES
        else:
            advantage, low, reason = float(score(used)), False, None

        name = field_names.get(pid)
        if name is None and "player_name" in player_all.columns:
            names = player_all["player_name"].dropna()
            name = str(names.iloc[0]) if not names.empty else None

        rows.append({
            "player_id": pid,
            "player_name": name,
            "course_advantage": advantage,
            "course_advantage_mean": advantage,
            "holes_covered": total_target if not low else 0,
            "total_target_holes": total_target,
            "total_raw_occurrences": raw,
            "total_weighted_occurrences": float(raw),
            "low_coverage": low,
            "reason": reason,
            "config_name": config_name,
            "target_course_slug": target_course_slug,
        })

    return rank_field(pd.DataFrame(rows, columns=[c for c in FIELD_RANKING_COLUMNS if c != "rank"]))


def _mean_field_adjusted(rows: pd.DataFrame) -> float:
    return float(rows[FIELD_ADJUSTED_COL].mean())


# --------------------------------------------------------------------------- #
# The baselines (each a drop-in ranker)
# --------------------------------------------------------------------------- #
def null_baseline(
    history, similar_holes, field, target_course_slug, predict_season,
    *, config_name="baseline", params=DEFAULT_PARAMS, aggregate="sum",
):
    """Field average: every player with enough eligible history scores 0 (no signal)."""
    return _baseline_rank(
        history, similar_holes, field, target_course_slug, predict_season,
        config_name=config_name, params=params,
        select=lambda df: df, score=lambda df: 0.0,
    )


def recent_form_baseline(
    history, similar_holes, field, target_course_slug, predict_season,
    *, config_name="baseline", params=DEFAULT_PARAMS, aggregate="sum",
):
    """Mean field-adjusted outcome over all recent eligible holes (ignores similarity)."""
    return _baseline_rank(
        history, similar_holes, field, target_course_slug, predict_season,
        config_name=config_name, params=params,
        select=lambda df: df, score=_mean_field_adjusted,
    )


def course_history_baseline(
    history, similar_holes, field, target_course_slug, predict_season,
    *, config_name="baseline", params=DEFAULT_PARAMS, aggregate="sum",
):
    """Mean field-adjusted outcome on the target course itself in prior seasons."""
    return _baseline_rank(
        history, similar_holes, field, target_course_slug, predict_season,
        config_name=config_name, params=params,
        select=lambda df: df[df["course_slug"] == target_course_slug],
        score=_mean_field_adjusted,
    )


def same_par_baseline(
    history, similar_holes, field, target_course_slug, predict_season,
    *, config_name="baseline", params=DEFAULT_PARAMS, aggregate="sum",
):
    """Mean field-adjusted outcome on holes sharing par with the target course."""
    pars = _target_pars(history, similar_holes, target_course_slug)
    return _baseline_rank(
        history, similar_holes, field, target_course_slug, predict_season,
        config_name=config_name, params=params,
        select=lambda df: df[df["par"].isin(pars)] if pars else df.iloc[0:0],
        score=_mean_field_adjusted,
    )


def season_average_baseline(
    history, similar_holes, field, target_course_slug, predict_season,
    *, config_name="baseline", params=DEFAULT_PARAMS, aggregate="sum",
):
    """Player's raw recent scoring average, ``-mean(player_score)`` (higher = better)."""
    return _baseline_rank(
        history, similar_holes, field, target_course_slug, predict_season,
        config_name=config_name, params=params,
        select=lambda df: df,
        score=lambda df: -float(pd.to_numeric(df["player_score"], errors="coerce").mean()),
    )


#: Registry of the built-in baselines (name -> ranker).
BASELINES: dict[str, Ranker] = {
    "null": null_baseline,
    "recent_form": recent_form_baseline,
    "course_history": course_history_baseline,
    "same_par": same_par_baseline,
    "season_average": season_average_baseline,
}

#: The metrics surfaced in a side-by-side comparison table.
_COMPARISON_METRICS: tuple[str, ...] = (
    "spearman_pooled", "spearman_mean", "pearson_pooled", "coverage", "n_pairs",
)

MODEL_NAME = "similar_hole_model"


def compare_baselines(
    history: pd.DataFrame,
    similar_holes,
    results: pd.DataFrame,
    *,
    outcome_col: str,
    higher_is_better: bool = False,
    params: AdvantageParams = DEFAULT_PARAMS,
    config_name: str = "baseline",
    aggregate: str = "sum",
    top_k: Sequence[int] = (10, 20),
    extra_rankers: Optional[dict[str, Ranker]] = None,
    include_model: bool = True,
) -> pd.DataFrame:
    """Backtest the model and every baseline on the same events; one metrics row each.

    Runs :func:`.backtest.run_backtest` per ranker and stacks the pooled/mean
    metrics into a table sorted by ``spearman_pooled`` (best first). Includes the
    similar-hole model as ``"similar_hole_model"`` unless ``include_model=False``.
    """
    rankers: dict[str, Ranker] = {}
    if include_model:
        rankers[MODEL_NAME] = score_tournament_field
    rankers.update(BASELINES)
    if extra_rankers:
        rankers.update(extra_rankers)

    top_k = tuple(int(k) for k in top_k)
    rows: list[dict] = []
    for name, ranker in rankers.items():
        res = run_backtest(
            history, similar_holes, results, outcome_col=outcome_col,
            higher_is_better=higher_is_better, params=params, config_name=config_name,
            aggregate=aggregate, top_k=top_k, ranker=ranker,
        )
        row = {"ranker": name}
        row.update({m: res.summary.get(m) for m in _COMPARISON_METRICS})
        for k in top_k:
            row[f"hit_rate_{k}_mean"] = res.summary.get(f"hit_rate_{k}_mean")
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("spearman_pooled", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def model_beats_baselines(
    comparison: pd.DataFrame,
    metric: str = "spearman_pooled",
    model_name: str = MODEL_NAME,
) -> bool:
    """True iff the model's ``metric`` strictly exceeds every baseline's.

    Honest by construction: returns ``False`` when the model ties or trails, or
    when its metric is undefined (``NaN``).
    """
    if model_name not in set(comparison["ranker"]):
        return False
    model_val = comparison.loc[comparison["ranker"] == model_name, metric].iloc[0]
    if pd.isna(model_val):
        return False
    others = comparison.loc[comparison["ranker"] != model_name, metric].dropna()
    return bool((model_val > others).all()) if not others.empty else True


__all__ = [
    "BASELINES",
    "MODEL_NAME",
    "null_baseline",
    "recent_form_baseline",
    "course_history_baseline",
    "same_par_baseline",
    "season_average_baseline",
    "compare_baselines",
    "model_beats_baselines",
]
