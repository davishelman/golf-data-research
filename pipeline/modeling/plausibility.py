"""Golf-plausibility layer on top of the raw similarity engine.

The raw nearest-neighbor engine (``similarity.py`` / ``export_similarity.py``)
answers *"which holes are numerically closest in feature space?"*. That is the
right primitive, but — as the similarity sanity check showed — its raw output can
include matches a golfer would reject: a par 5 for a par-4 query, a 320-yd hole
for a 450-yd one, a water hole for a dry one, or "17 other holes on the same
course".

This module wraps raw similarity rows with **golfer-readable flags, a 0–1
plausibility score, and a presentable filter**, without changing the engine. The
raw v1 / v2 / mode tables are untouched; this is an additive presentation layer.

Design
------
* :func:`match_plausibility_flags` — facts about one (query, candidate) pair.
* :func:`plausibility_score` — turn flags into a 0–1 score via simple penalties.
* :func:`explain_plausibility` — a human-readable reason string.
* :func:`add_plausibility_to_similarity` — annotate a whole raw table.
* :func:`filter_presentable_matches` — keep only golfer-presentable rows.
* :func:`presented_similarity_table` — the convenience pipeline: annotate →
  filter → re-rank → top-N per query.

Everything uses only pandas + stdlib (no sklearn), and degrades gracefully when
optional feature columns are missing (it uses whatever is available and says so
via the reasons / score).
"""

from __future__ import annotations

import math
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Water-pressure columns, in preference order (water_pct first, zones augment it).
WATER_COLS: tuple[str, ...] = (
    "water_pct", "drive_zone_water_pct", "approach_zone_water_pct",
    "green_complex_water_pct",
)
#: Combined hazard-pressure columns (bunker / water / trees). Sand is excluded on
#: purpose: ``sand_pct`` (natural/waste sand) is almost never tagged; real golf
#: bunkers are ``bunker_pct``.
HAZARD_COLS: tuple[str, ...] = ("bunker_pct", "water_pct", "trees_pct")
#: Hole shape / size columns used for the geometry mismatch.
GEOMETRY_COLS: tuple[str, ...] = (
    "hole_width_m", "hole_depth_m", "green_y_m", "dogleg_score",
    "fairway_centerline_shift",
)

#: Score penalties (start at 1.0 and subtract). ``same_course`` is only applied
#: when the caller is excluding same-course matches (``same_course_violation``).
PENALTIES: dict[str, float] = {
    "par_mismatch": 0.40,
    "length_mismatch": 0.30,
    "same_course": 0.20,
    "water_mismatch": 0.15,
    "hazard_mismatch": 0.15,
    "geometry_mismatch": 0.15,
}

#: Bad-flag keys that count toward ``presentable_bad_flag_count``.
_BAD_FLAGS: tuple[str, ...] = (
    "par_mismatch", "length_mismatch", "water_mismatch", "hazard_mismatch",
    "geometry_mismatch", "same_course_violation",
)

#: Preferred output column order for the presented table.
OUTPUT_COLUMNS: tuple[str, ...] = (
    "query_hole_id", "similar_hole_id", "similarity_mode", "raw_rank", "raw_distance",
    "query_course_slug", "similar_course_slug", "query_hole_number", "similar_hole_number",
    "query_length_m", "similar_length_m", "length_diff_m", "length_diff_pct",
    "same_par", "same_course", "par_mismatch", "length_mismatch",
    "water_mismatch", "hazard_mismatch", "geometry_mismatch",
    "presentable_bad_flag_count", "plausibility_score", "is_presentable",
    "plausibility_reasons", "presented_rank",
)


# --------------------------------------------------------------------------- #
# Small row helpers (work for both pandas Series and plain dicts)
# --------------------------------------------------------------------------- #

def _num(row, col: str) -> Optional[float]:
    """Numeric value of ``col`` in ``row`` (Series or dict), or None if absent/NaN."""
    if row is None:
        return None
    v = row.get(col) if hasattr(row, "get") else None
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _str(row, col: str):
    v = row.get(col) if hasattr(row, "get") else None
    return v


def _max_abs_diff(q, c, cols) -> Optional[float]:
    diffs = [abs(a - b) for col in cols
             if (a := _num(q, col)) is not None and (b := _num(c, col)) is not None]
    return max(diffs) if diffs else None


def _mean_abs_diff(q, c, cols) -> Optional[float]:
    diffs = [abs(a - b) for col in cols
             if (a := _num(q, col)) is not None and (b := _num(c, col)) is not None]
    return (sum(diffs) / len(diffs)) if diffs else None


def _sym_rel_mismatch(q, c, cols) -> Optional[float]:
    """Mean symmetric relative difference |a-b|/(|a|+|b|) over available columns.

    Bounded in [0, 1] per feature, robust to near-zero values (both ≈ 0 → 0), and
    treats opposite-signed features (e.g. left vs right centerline shift) as a
    large mismatch. Independent of any global scaler, so it works per-pair.
    """
    ratios = []
    for col in cols:
        a, b = _num(q, col), _num(c, col)
        if a is None or b is None:
            continue
        denom = abs(a) + abs(b)
        ratios.append(0.0 if denom < 1e-9 else abs(a - b) / denom)
    return (sum(ratios) / len(ratios)) if ratios else None


# --------------------------------------------------------------------------- #
# Core: flags, score, explanation
# --------------------------------------------------------------------------- #

def match_plausibility_flags(
    query_row,
    candidate_row,
    *,
    max_length_diff_m: float = 55.0,
    max_length_diff_pct: float = 0.15,
    water_mismatch_threshold: float = 0.08,
    hazard_mismatch_threshold: float = 0.12,
    geometry_mismatch_threshold: float = 0.45,
) -> dict:
    """Golf-plausibility facts for one (query, candidate) pair.

    ``query_row`` / ``candidate_row`` may be pandas Series or dicts. Only
    available columns are used; missing optional columns simply don't contribute
    (the corresponding mismatch is ``False`` because it cannot be judged).

    Returns a dict with ``same_par``, ``same_course``, ``length_diff_m``,
    ``length_diff_pct``, the boolean ``*_mismatch`` flags, and the raw mismatch
    magnitudes (``water_diff`` / ``hazard_diff`` / ``geometry_mismatch_score``).
    Note: ``same_course`` is reported as a fact; whether it *counts against* a
    match is decided by the caller (see ``same_course_violation`` in
    :func:`add_plausibility_to_similarity`).
    """
    qpar, cpar = _num(query_row, "par"), _num(candidate_row, "par")
    same_par = qpar is not None and cpar is not None and qpar == cpar

    qslug, cslug = _str(query_row, "course_slug"), _str(candidate_row, "course_slug")
    same_course = qslug is not None and cslug is not None and qslug == cslug

    qlen, clen = _num(query_row, "hole_length_m"), _num(candidate_row, "hole_length_m")
    if qlen is not None and clen is not None:
        length_diff_m = abs(qlen - clen)
        length_diff_pct = (length_diff_m / qlen) if qlen else float("nan")
    else:
        length_diff_m = float("nan")
        length_diff_pct = float("nan")
    length_mismatch = bool(
        (not math.isnan(length_diff_m) and length_diff_m > max_length_diff_m)
        or (not math.isnan(length_diff_pct) and length_diff_pct > max_length_diff_pct)
    )

    water_diff = _max_abs_diff(query_row, candidate_row, WATER_COLS)
    hazard_diff = _mean_abs_diff(query_row, candidate_row, HAZARD_COLS)
    geom_score = _sym_rel_mismatch(query_row, candidate_row, GEOMETRY_COLS)

    return {
        "same_par": bool(same_par),
        "same_course": bool(same_course),
        "length_diff_m": round(length_diff_m, 2) if not math.isnan(length_diff_m) else float("nan"),
        "length_diff_pct": round(length_diff_pct, 4) if not math.isnan(length_diff_pct) else float("nan"),
        "par_mismatch": not same_par,
        "length_mismatch": length_mismatch,
        "water_mismatch": bool(water_diff is not None and water_diff > water_mismatch_threshold),
        "hazard_mismatch": bool(hazard_diff is not None and hazard_diff > hazard_mismatch_threshold),
        "geometry_mismatch": bool(geom_score is not None and geom_score > geometry_mismatch_threshold),
        "water_diff": round(water_diff, 4) if water_diff is not None else float("nan"),
        "hazard_diff": round(hazard_diff, 4) if hazard_diff is not None else float("nan"),
        "geometry_mismatch_score": round(geom_score, 4) if geom_score is not None else float("nan"),
    }


def plausibility_score(flags: dict) -> float:
    """Turn flags into a 0–1 score (1.0 = perfectly plausible).

    Subtracts the configured penalty for each active bad flag and clamps to
    [0, 1]. The same-course penalty applies only when ``flags`` carries a truthy
    ``same_course_violation`` (set by the caller when excluding same-course
    matches), so a same-course pair isn't penalized when same-course is allowed.
    """
    score = 1.0
    if flags.get("par_mismatch"):
        score -= PENALTIES["par_mismatch"]
    if flags.get("length_mismatch"):
        score -= PENALTIES["length_mismatch"]
    if flags.get("same_course_violation"):
        score -= PENALTIES["same_course"]
    if flags.get("water_mismatch"):
        score -= PENALTIES["water_mismatch"]
    if flags.get("hazard_mismatch"):
        score -= PENALTIES["hazard_mismatch"]
    if flags.get("geometry_mismatch"):
        score -= PENALTIES["geometry_mismatch"]
    return float(max(0.0, min(1.0, score)))


def explain_plausibility(flags: dict) -> str:
    """Readable, semicolon-separated reasons a match is implausible ("" if clean)."""
    reasons: list[str] = []
    same_course_bad = flags.get("same_course_violation", flags.get("same_course", False))
    if same_course_bad:
        reasons.append("same course")
    if flags.get("par_mismatch"):
        reasons.append("different par")
    if flags.get("length_mismatch"):
        ld = flags.get("length_diff_m")
        reasons.append(f"length diff {ld:.1f}m"
                       if isinstance(ld, (int, float)) and not math.isnan(ld)
                       else "length mismatch")
    if flags.get("water_mismatch"):
        reasons.append("water profile mismatch")
    if flags.get("hazard_mismatch"):
        reasons.append("hazard profile mismatch")
    if flags.get("geometry_mismatch"):
        reasons.append("geometry mismatch")
    return "; ".join(reasons)


# --------------------------------------------------------------------------- #
# Table-level enrichment + filtering
# --------------------------------------------------------------------------- #

def _missing_record() -> dict:
    """Conservative annotation when a hole id is absent from the feature table."""
    return {
        "same_par": False, "same_course": False,
        "length_diff_m": float("nan"), "length_diff_pct": float("nan"),
        "par_mismatch": True, "length_mismatch": True,
        "water_mismatch": False, "hazard_mismatch": False, "geometry_mismatch": False,
        "presentable_bad_flag_count": 2, "plausibility_score": 0.0,
        "is_presentable": False, "plausibility_reasons": "missing feature row",
    }


def add_plausibility_to_similarity(
    similarity_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    query_id_col: str = "query_hole_id",
    candidate_id_col: str = "similar_hole_id",
    require_same_par: bool = True,
    exclude_same_course: bool = True,
    max_length_diff_m: float = 55.0,
    max_length_diff_pct: float = 0.15,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Annotate a raw similarity table with plausibility flags / score / reasons.

    Adds: ``same_par``, ``same_course``, ``length_diff_m``, ``length_diff_pct``,
    ``par_mismatch``, ``length_mismatch``, ``water_mismatch``, ``hazard_mismatch``,
    ``geometry_mismatch``, ``presentable_bad_flag_count``, ``plausibility_score``,
    ``is_presentable`` and ``plausibility_reasons``. Raw columns are preserved.

    ``is_presentable`` reflects the hard golf gates (same par if required,
    cross-course if required, length within guard) plus ``min_score`` when given.
    """
    out = similarity_df.copy().reset_index(drop=True)
    by_id = features_df.set_index("hole_id").to_dict("index")

    records = []
    for q, c in zip(out[query_id_col], out[candidate_id_col]):
        qr, cr = by_id.get(q), by_id.get(c)
        if qr is None or cr is None:
            records.append(_missing_record())
            continue
        flags = match_plausibility_flags(
            qr, cr, max_length_diff_m=max_length_diff_m,
            max_length_diff_pct=max_length_diff_pct)
        flags["same_course_violation"] = bool(flags["same_course"] and exclude_same_course)
        score = plausibility_score(flags)
        bad = sum(bool(flags.get(k)) for k in _BAD_FLAGS)
        gates = (
            (not require_same_par or flags["same_par"])
            and (not exclude_same_course or not flags["same_course"])
            and (not flags["length_mismatch"])
        )
        records.append({
            "same_par": flags["same_par"], "same_course": flags["same_course"],
            "length_diff_m": flags["length_diff_m"], "length_diff_pct": flags["length_diff_pct"],
            "par_mismatch": flags["par_mismatch"], "length_mismatch": flags["length_mismatch"],
            "water_mismatch": flags["water_mismatch"], "hazard_mismatch": flags["hazard_mismatch"],
            "geometry_mismatch": flags["geometry_mismatch"],
            "presentable_bad_flag_count": int(bad),
            "plausibility_score": round(score, 3),
            "is_presentable": bool(gates and (min_score is None or score >= min_score)),
            # Clean rows read as "ok" (not blank/NaN) for tidy display + CSV output.
            "plausibility_reasons": explain_plausibility(flags) or "ok",
        })

    enriched = pd.DataFrame.from_records(records, index=out.index)
    # Don't clobber any same-named raw columns; prefer the freshly computed ones.
    keep_raw = [c for c in out.columns if c not in enriched.columns]
    return pd.concat([out[keep_raw], enriched], axis=1)


def filter_presentable_matches(
    enriched_df: pd.DataFrame,
    *,
    min_score: float = 0.75,
    require_same_par: bool = True,
    exclude_same_course: bool = True,
    max_bad_flags: int = 0,
) -> pd.DataFrame:
    """Keep only golfer-presentable rows from an enriched table.

    A row passes when its score ≥ ``min_score``, it has ≤ ``max_bad_flags`` bad
    flags, and (optionally) it is same-par and cross-course. Operates purely on
    the annotation columns, so it can be re-applied with different thresholds.
    """
    df = enriched_df
    mask = df["plausibility_score"] >= min_score
    if require_same_par and "same_par" in df.columns:
        mask &= df["same_par"].astype(bool)
    if exclude_same_course and "same_course" in df.columns:
        mask &= ~df["same_course"].astype(bool)
    if "presentable_bad_flag_count" in df.columns:
        mask &= df["presentable_bad_flag_count"] <= max_bad_flags
    return df[mask].copy()


def presented_similarity_table(
    raw_similarity_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    n_neighbors: int = 10,
    require_same_par: bool = True,
    exclude_same_course: bool = True,
    min_score: float = 0.75,
) -> pd.DataFrame:
    """Annotate → filter → re-rank a raw similarity table into a presentable one.

    Returns at most ``n_neighbors`` rows per query, ranked by plausibility score
    (then raw distance), with a fresh ``presented_rank``. Queries with fewer
    presentable matches simply return fewer rows — nothing is silently faked; the
    row count makes shortfalls visible.
    """
    enriched = add_plausibility_to_similarity(
        raw_similarity_df, features_df,
        require_same_par=require_same_par, exclude_same_course=exclude_same_course,
        min_score=min_score)

    feat = features_df.set_index("hole_id")

    def _fill(col: str, id_col: str, src: str, rnd: bool = False):
        if col not in enriched.columns and src in feat.columns:
            mapped = enriched[id_col].map(feat[src])
            enriched[col] = mapped.round(2) if rnd else mapped

    _fill("query_course_slug", "query_hole_id", "course_slug")
    _fill("similar_course_slug", "similar_hole_id", "course_slug")
    _fill("query_hole_number", "query_hole_id", "hole_number")
    _fill("similar_hole_number", "similar_hole_id", "hole_number")
    _fill("query_length_m", "query_hole_id", "hole_length_m", rnd=True)
    _fill("similar_length_m", "similar_hole_id", "hole_length_m", rnd=True)
    if "similarity_mode" not in enriched.columns:
        enriched["similarity_mode"] = "overall_v2"
    enriched = enriched.rename(columns={"rank": "raw_rank", "distance": "raw_distance"})

    pres = enriched[enriched["is_presentable"]].copy()
    sort_cols, asc = ["plausibility_score"], [False]
    if "raw_distance" in pres.columns:
        sort_cols.append("raw_distance")
        asc.append(True)
    pres = pres.sort_values(["query_hole_id"] + sort_cols, ascending=[True] + asc)
    pres["presented_rank"] = pres.groupby("query_hole_id").cumcount() + 1
    pres = pres[pres["presented_rank"] <= n_neighbors].reset_index(drop=True)

    ordered = [c for c in OUTPUT_COLUMNS if c in pres.columns]
    extra = [c for c in pres.columns if c not in ordered]
    return pres[ordered + extra]


__all__ = [
    "match_plausibility_flags",
    "plausibility_score",
    "explain_plausibility",
    "add_plausibility_to_similarity",
    "filter_presentable_matches",
    "presented_similarity_table",
    "PENALTIES",
    "OUTPUT_COLUMNS",
]
