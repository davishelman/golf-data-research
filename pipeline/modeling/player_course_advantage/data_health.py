"""Data coverage & quality health report (issue #62).

Answers a question that is *separate* from predictive performance: **are the
inputs complete enough for any metric to mean anything?** It quantifies schema
health, similarity coverage, and (optionally) backtest coverage so it can be run
*before* an expensive backtest and stop early on a 🔴-red input set.

Deliberately **non-crashing**: it never calls the raising validator — it *counts*
problems (missing ``field_avg_score``, duplicate grain, malformed ids, uncovered
candidate holes, low-coverage predictions) and returns numbers + warnings on
imperfect data. Pure pandas, Streamlit-free, no real-data dependency.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .schema import KEY_COLUMNS

# An "event" is one tournament occurrence; a player-event pair is one player in it.
_EVENT_KEYS = ("tournament_id", "year")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _nunique(df: pd.DataFrame, cols) -> int:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return 0
    return int(df[cols].drop_duplicates().shape[0])


def summarize_history_quality(history: pd.DataFrame) -> dict:
    """Schema-health + volume counts for a hole-score history (never raises)."""
    n = int(len(history))
    out = {
        "rows": n,
        "events": _nunique(history, _EVENT_KEYS),
        "seasons": int(history["year"].nunique()) if "year" in history else 0,
        "courses": int(history["course_slug"].nunique()) if "course_slug" in history else 0,
        "players": int(history["player_id"].nunique()) if "player_id" in history else 0,
        "player_event_pairs": _nunique(history, ("player_id", *_EVENT_KEYS)),
    }

    # Missing field_avg_score (absent column counts as fully missing).
    if "field_avg_score" not in history.columns:
        out["missing_field_avg_score_rate"] = 1.0 if n else 0.0
    else:
        miss = int(_num(history["field_avg_score"]).isna().sum())
        out["missing_field_avg_score_rate"] = (miss / n) if n else 0.0

    # Duplicate occurrences on the grain (pre-validation).
    key = [c for c in KEY_COLUMNS if c in history.columns]
    out["duplicate_grain_count"] = (
        int(history.duplicated(subset=key, keep=False).sum()) if key and n else 0
    )

    # Malformed / inconsistent hole ids vs course_slug + hole_number.
    out["invalid_hole_id_count"] = _invalid_hole_id_count(history)
    return out


def _invalid_hole_id_count(history: pd.DataFrame) -> int:
    need = ("hole_id_v25", "course_slug", "hole_number")
    if any(c not in history.columns for c in need):
        return 0
    hn = _num(history["hole_number"])
    usable = (
        history["hole_id_v25"].notna()
        & history["course_slug"].notna()
        & hn.notna()
    )
    if not usable.any():
        return 0
    expected = (
        history.loc[usable, "course_slug"].astype(str)
        + ":"
        + hn[usable].astype(int).astype(str)
    )
    actual = history.loc[usable, "hole_id_v25"].astype(str)
    return int((actual.to_numpy() != expected.to_numpy()).sum())


def summarize_similarity_coverage(
    similar_holes: pd.DataFrame, history: pd.DataFrame
) -> dict:
    """How well the similar-hole sets connect to available historical outcomes."""
    out = {
        "target_holes": 0,
        "target_holes_with_similar_sets": 0,
        "target_holes_missing_similar_sets": 0,
        "candidate_holes": 0,
        "candidate_holes_with_history": 0,
        "candidate_holes_missing_history": 0,
    }
    if "target_hole_id" in similar_holes.columns:
        targets = similar_holes["target_hole_id"].dropna().astype(str)
        out["target_holes"] = int(targets.nunique())
        # A target has a usable set only if it has ≥1 candidate row with a weight.
        w = similar_holes.get("similarity_weight")
        has_set = similar_holes.assign(_w=(w if w is not None else 1.0))
        good = has_set.dropna(subset=["target_hole_id"])
        if "candidate_hole_id" in good.columns:
            good = good[good["candidate_hole_id"].notna()]
        covered_targets = good["target_hole_id"].astype(str).nunique()
        out["target_holes_with_similar_sets"] = int(covered_targets)
        out["target_holes_missing_similar_sets"] = int(out["target_holes"] - covered_targets)

    if "candidate_hole_id" in similar_holes.columns:
        cands = similar_holes["candidate_hole_id"].dropna().astype(str)
        uniq = set(cands.unique())
        out["candidate_holes"] = int(len(uniq))
        hist_ids = (
            set(history["hole_id_v25"].dropna().astype(str).unique())
            if "hole_id_v25" in history.columns else set()
        )
        with_hist = uniq & hist_ids
        out["candidate_holes_with_history"] = int(len(with_hist))
        out["candidate_holes_missing_history"] = int(len(uniq - hist_ids))
    return out


def summarize_backtest_coverage(predictions: pd.DataFrame) -> dict:
    """Coverage of the (player, event) pairs a backtest actually scored."""
    n = int(len(predictions))
    covered_mask = (
        predictions["covered"].fillna(False)
        if "covered" in predictions.columns else pd.Series([], dtype=bool)
    )
    covered = int(covered_mask.sum())
    out = {
        "player_event_pairs": n,
        "player_event_pairs_scored": covered,
        "coverage": (covered / n) if n else float("nan"),
    }

    if "event_id" in predictions.columns and n:
        per_event = covered_mask.groupby(predictions["event_id"]).sum()
        out["events"] = int(predictions["event_id"].nunique())
        out["event_coverage"] = float((per_event >= 2).mean()) if len(per_event) else float("nan")
    else:
        out["events"] = 0
        out["event_coverage"] = float("nan")

    if "holes_covered" in predictions.columns and n:
        hc = _num(predictions["holes_covered"])
        out["holes_covered_mean"] = float(hc.mean())
        out["holes_covered_distribution"] = {
            "min": float(hc.min()), "median": float(hc.median()), "max": float(hc.max()),
        }
    else:
        out["holes_covered_mean"] = float("nan")
        out["holes_covered_distribution"] = {}

    out["low_coverage_rate"] = (
        float((~covered_mask).mean()) if n else float("nan")
    )
    out["low_coverage_reason_counts"] = _reason_counts(predictions, covered_mask)
    out["coverage_by_course"] = _coverage_by(predictions, "target_course_slug", covered_mask)
    out["coverage_by_season"] = _coverage_by(predictions, "predict_season", covered_mask)
    out["coverage_by_player"] = _coverage_by(predictions, "player_id", covered_mask)
    return out


def _reason_counts(predictions: pd.DataFrame, covered_mask: pd.Series) -> dict:
    if "reason" not in predictions.columns or predictions.empty:
        return {}
    reasons = predictions.loc[~covered_mask, "reason"].dropna()
    return {str(k): int(v) for k, v in reasons.value_counts().items()}


def _coverage_by(predictions: pd.DataFrame, col: str, covered_mask: pd.Series) -> dict:
    if col not in predictions.columns or predictions.empty:
        return {}
    grouped = covered_mask.groupby(predictions[col]).mean()
    return {str(k): float(v) for k, v in grouped.items()}


def build_data_health_report(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
    *,
    data_source: str = "synthetic",
) -> dict:
    """Assemble the full health report (history + similarity + optional backtest).

    ``predictions`` (from ``run_backtest``) is optional so this can run *before* a
    backtest. Returns a dict with a ``warnings`` list flagging anything that would
    make evaluation untrustworthy. ``data_source`` is echoed so downstream reports
    never mislabel synthetic runs as real.
    """
    hist_q = summarize_history_quality(history)
    sim_c = summarize_similarity_coverage(similar_holes, history)
    bt_c = summarize_backtest_coverage(predictions) if predictions is not None else None

    warnings: list[str] = []
    if hist_q["missing_field_avg_score_rate"] > 0:
        warnings.append(
            f"{hist_q['missing_field_avg_score_rate']:.1%} of history rows lack "
            "field_avg_score — the field-adjusted outcome is uncomputable for those."
        )
    if hist_q["duplicate_grain_count"] > 0:
        warnings.append(
            f"{hist_q['duplicate_grain_count']} duplicate rows on the occurrence "
            "grain — validation would reject this history."
        )
    if hist_q["invalid_hole_id_count"] > 0:
        warnings.append(
            f"{hist_q['invalid_hole_id_count']} rows have a hole_id_v25 inconsistent "
            "with course_slug/hole_number."
        )
    if sim_c["candidate_holes_missing_history"] > 0:
        warnings.append(
            f"{sim_c['candidate_holes_missing_history']} candidate holes have no "
            "historical outcomes — they contribute nothing to any advantage."
        )
    if sim_c["target_holes_missing_similar_sets"] > 0:
        warnings.append(
            f"{sim_c['target_holes_missing_similar_sets']} target holes have no "
            "usable similar-hole set."
        )
    if bt_c is not None:
        if bt_c["coverage"] == bt_c["coverage"] and bt_c["coverage"] < 0.5:  # not NaN
            warnings.append(
                f"backtest coverage is {bt_c['coverage']:.1%} — metrics computed on "
                "so few scored pairs are unreliable."
            )

    return {
        "data_source": data_source,
        "history_quality": hist_q,
        "similarity_coverage": sim_c,
        "backtest_coverage": bt_c,
        "warnings": warnings,
    }


def data_health_to_frames(report: dict) -> dict[str, pd.DataFrame]:
    """Flatten a report into small tables (mirrors the issue's CSV outputs)."""
    frames: dict[str, pd.DataFrame] = {}
    summary = {**report["history_quality"], **report["similarity_coverage"]}
    if report.get("backtest_coverage"):
        bt = report["backtest_coverage"]
        summary.update({k: v for k, v in bt.items() if not isinstance(v, dict)})
        frames["low_coverage_reasons"] = _dict_frame(
            bt.get("low_coverage_reason_counts", {}), "reason", "count")
        frames["coverage_by_course"] = _dict_frame(bt.get("coverage_by_course", {}), "course", "coverage")
        frames["coverage_by_player"] = _dict_frame(bt.get("coverage_by_player", {}), "player_id", "coverage")
    frames["data_quality_summary"] = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in summary.items()]
    )
    return frames


def _dict_frame(d: dict, key: str, val: str) -> pd.DataFrame:
    return pd.DataFrame([{key: k, val: v} for k, v in d.items()])


__all__ = [
    "summarize_history_quality",
    "summarize_similarity_coverage",
    "summarize_backtest_coverage",
    "build_data_health_report",
    "data_health_to_frames",
]
