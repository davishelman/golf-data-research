"""Retrospective backtest framework (issue #35).

Walk historical events forward and ask: did the player-course advantage score,
computed **only from history available before each event**, line up with what
actually happened? This module wires the merged scorer/batch layer to a set of
actual event outcomes and reports rank-correlation / hit-rate / lift metrics.

**Leakage is the whole ballgame.** Every event is scored with
``predict_season = <event season>``, and the scorer's
:func:`.scorer.filter_history_for_prediction_window` admits only
``predict_season - W <= year < predict_season`` — never the event season or
later. The backtest never pools all years and scores in-sample; each event
freezes its own window. A test asserts a planted future-season row cannot change
a prediction.

This is **v0 plumbing**: it computes honest metrics on whatever labels it is
given (synthetic in tests). It makes **no** claim of predictive validity on real
data — that requires the #46 data sourcing before the numbers mean anything.
Pure, deterministic, Streamlit-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .batch import score_tournament_field
from .schema import DEFAULT_PARAMS, AdvantageParams, validate_hole_score_history

SimilarHoles = Union[pd.DataFrame, Mapping[str, pd.DataFrame]]

#: Default k values for top-k hit-rate / lift.
DEFAULT_TOP_K: tuple[int, ...] = (10, 20)


class BacktestError(ValueError):
    """Raised when backtest inputs are structurally unusable."""


# --------------------------------------------------------------------------- #
# Pure metric helpers (independently testable)
# --------------------------------------------------------------------------- #
def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; ``NaN`` for <2 points or a constant input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson on average-ranks (no scipy needed)."""
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    return pearson_corr(x.rank().to_numpy(), y.rank().to_numpy())


def _top_indices(scores: np.ndarray, k: int) -> set[int]:
    """Indices of the ``k`` highest scores (stable, ties broken by position)."""
    k = min(k, len(scores))
    if k <= 0:
        return set()
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    return set(int(i) for i in order[:k])


def top_k_hit_rate(pred: Sequence[float], performance: Sequence[float], k: int) -> float:
    """Fraction of the model's top-``k`` that are in the actual top-``k``.

    Both inputs are *higher-is-better*. ``k`` is clamped to the field size;
    ``NaN`` for an empty field.
    """
    pred = np.asarray(pred, dtype=float)
    perf = np.asarray(performance, dtype=float)
    kk = min(k, len(pred))
    if kk <= 0:
        return float("nan")
    return len(_top_indices(pred, kk) & _top_indices(perf, kk)) / kk


def top_k_lift(pred: Sequence[float], performance: Sequence[float], k: int) -> float:
    """Mean performance of the model's top-``k`` minus the field-mean performance.

    Positive = the model's picks beat the field average. ``NaN`` for an empty field.
    """
    pred = np.asarray(pred, dtype=float)
    perf = np.asarray(performance, dtype=float)
    if len(pred) == 0:
        return float("nan")
    top = list(_top_indices(pred, min(k, len(pred))))
    if not top:
        return float("nan")
    return float(perf[top].mean() - perf.mean())


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestResult:
    """Backtest outputs: per (event, player) predictions, per-event and pooled metrics."""

    predictions: pd.DataFrame
    per_event: pd.DataFrame
    summary: dict

    def to_markdown(self) -> str:
        """A compact markdown report (pooled summary + per-event table)."""
        s = self.summary
        lines = [
            "# Player-course advantage backtest",
            "",
            f"- events: **{s.get('n_events', 0)}**  ·  scored pairs: "
            f"**{s.get('n_pairs', 0)}**  ·  coverage: **{s.get('coverage', float('nan')):.3f}**",
            f"- outcome: `{s.get('outcome_col')}`  ·  higher_is_better="
            f"{s.get('higher_is_better')}",
            f"- Spearman (mean/pooled): **{s.get('spearman_mean', float('nan')):.3f}** / "
            f"**{s.get('spearman_pooled', float('nan')):.3f}**",
            f"- Pearson (mean/pooled): **{s.get('pearson_mean', float('nan')):.3f}** / "
            f"**{s.get('pearson_pooled', float('nan')):.3f}**",
            "",
            "> v0 — synthetic unless a real, leakage-free label set is supplied.",
            "",
        ]
        if not self.per_event.empty:
            lines.append(_df_to_markdown(self.per_event))
        return "\n".join(lines)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no ``tabulate`` dependency)."""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(fmt(v) for v in rec) + " |"
        for rec in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def _sim_for(similar_holes: SimilarHoles, course: str) -> pd.DataFrame:
    if isinstance(similar_holes, pd.DataFrame):
        return similar_holes
    try:
        return similar_holes[course]
    except KeyError as exc:
        raise BacktestError(
            f"no similar-hole set provided for course {course!r}"
        ) from exc


def run_backtest(
    history: pd.DataFrame,
    similar_holes: SimilarHoles,
    results: pd.DataFrame,
    *,
    outcome_col: str,
    higher_is_better: bool = False,
    params: AdvantageParams = DEFAULT_PARAMS,
    config_name: str = "baseline",
    aggregate: str = "sum",
    top_k: Sequence[int] = DEFAULT_TOP_K,
    error_metrics: bool = False,
    ranker: Callable[..., pd.DataFrame] = score_tournament_field,
    event_col: str = "event_id",
    season_col: str = "predict_season",
    course_col: str = "target_course_slug",
) -> BacktestResult:
    """Walk each event in ``results`` and score its field from prior history only.

    ``results`` is the actual-outcomes table: one row per (event, player) with
    ``event_col``, ``season_col``, ``course_col``, ``player_id`` and ``outcome_col``.
    Each event is scored with ``predict_season = <its season>``; the scorer admits
    only strictly-past seasons within ``params.lookback_years`` — the leakage guard.

    ``outcome_col`` is the actual result to correlate; set ``higher_is_better`` so
    "performance" is normalized higher-is-better (e.g. ``finish_rank`` →
    ``higher_is_better=False``, ``strokes_gained`` → ``True``). ``ranker`` is the
    field-scoring callable (default the similar-hole model
    :func:`.batch.score_tournament_field`); pass a #38 baseline to evaluate it on
    the exact same events/metrics. Returns a :class:`BacktestResult`. ``history``
    is validated once. Never mutates inputs.
    """
    validate_hole_score_history(history)
    for col in (event_col, season_col, course_col, "player_id", outcome_col):
        if col not in results.columns:
            raise BacktestError(f"results missing required column {col!r}")

    top_k = tuple(int(k) for k in top_k)
    pred_frames: list[pd.DataFrame] = []
    event_rows: list[dict] = []

    # Deterministic event order: by season, then event id.
    event_order = (
        results[[event_col, season_col, course_col]]
        .drop_duplicates()
        .sort_values([season_col, event_col])
    )

    for ev in event_order.itertuples(index=False):
        event_id = getattr(ev, event_col)
        season = int(getattr(ev, season_col))
        course = getattr(ev, course_col)
        group = results[results[event_col] == event_id]
        field = group["player_id"].astype(str).tolist()

        ranking = ranker(
            history, _sim_for(similar_holes, course), field, course, season,
            config_name=config_name, params=params, aggregate=aggregate,
        )
        merged = ranking.merge(
            group[["player_id", outcome_col]], on="player_id", how="left"
        )
        merged.insert(0, "event_id", event_id)
        merged["predict_season"] = season
        sign = 1.0 if higher_is_better else -1.0
        merged["performance"] = sign * pd.to_numeric(merged[outcome_col], errors="coerce")
        merged["covered"] = (~merged["low_coverage"]) & merged["performance"].notna() & \
            merged["course_advantage"].notna()
        pred_frames.append(merged)

        event_rows.append(
            _event_metrics(
                event_id, season, course, merged, outcome_col, top_k, error_metrics
            )
        )

    predictions = (
        pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    )
    per_event = pd.DataFrame(event_rows)
    summary = _summarize(predictions, per_event, outcome_col, higher_is_better, top_k)
    return BacktestResult(predictions=predictions, per_event=per_event, summary=summary)


def _event_metrics(
    event_id, season, course, merged, outcome_col, top_k, error_metrics
) -> dict:
    cov = merged[merged["covered"]]
    adv = cov["course_advantage"].to_numpy(dtype=float)
    perf = cov["performance"].to_numpy(dtype=float)

    row = {
        "event_id": event_id,
        "predict_season": season,
        "target_course_slug": course,
        "n_field": int(len(merged)),
        "n_covered": int(len(cov)),
        "coverage": float(len(cov) / len(merged)) if len(merged) else float("nan"),
        "spearman": spearman_corr(adv, perf),
        "pearson": pearson_corr(adv, perf),
    }
    for k in top_k:
        row[f"hit_rate_{k}"] = top_k_hit_rate(adv, perf, k)
        row[f"lift_{k}"] = top_k_lift(adv, perf, k)
    if error_metrics:
        out = pd.to_numeric(cov[outcome_col], errors="coerce").to_numpy(dtype=float)
        diff = adv - out
        row["mae"] = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
        row["rmse"] = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else float("nan")
    return row


def _summarize(predictions, per_event, outcome_col, higher_is_better, top_k) -> dict:
    if predictions.empty:
        return {
            "n_events": 0, "n_pairs": 0, "coverage": float("nan"),
            "outcome_col": outcome_col, "higher_is_better": higher_is_better,
            "spearman_mean": float("nan"), "spearman_pooled": float("nan"),
            "pearson_mean": float("nan"), "pearson_pooled": float("nan"),
        }
    covered = predictions[predictions["covered"]]
    summary = {
        "n_events": int(per_event["event_id"].nunique()),
        "n_pairs": int(len(covered)),
        "coverage": float(len(covered) / len(predictions)),
        "outcome_col": outcome_col,
        "higher_is_better": higher_is_better,
        # mean of per-event coefficients (ignoring NaN events) ...
        "spearman_mean": float(per_event["spearman"].mean(skipna=True)),
        "pearson_mean": float(per_event["pearson"].mean(skipna=True)),
        # ... and pooled over all scored pairs.
        "spearman_pooled": spearman_corr(
            covered["course_advantage"], covered["performance"]
        ),
        "pearson_pooled": pearson_corr(
            covered["course_advantage"], covered["performance"]
        ),
    }
    for k in top_k:
        summary[f"hit_rate_{k}_mean"] = float(per_event[f"hit_rate_{k}"].mean(skipna=True))
        summary[f"lift_{k}_mean"] = float(per_event[f"lift_{k}"].mean(skipna=True))
    return summary


__all__ = [
    "DEFAULT_TOP_K",
    "BacktestError",
    "BacktestResult",
    "pearson_corr",
    "spearman_corr",
    "top_k_hit_rate",
    "top_k_lift",
    "run_backtest",
]
