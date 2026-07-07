"""Input contract for the player-course advantage layer.

This module defines the *historical hole-score* table (issue #31) that the future
advantage scorer will consume, plus the small pure helpers needed to state and
check that contract:

* column groups (:data:`REQUIRED_COLUMNS`, :data:`OPTIONAL_COLUMNS`) and the
  occurrence-grain key (:data:`KEY_COLUMNS`),
* the **positive-is-good** field-adjusted outcome convention
  (:func:`field_adjusted_advantage` / :func:`add_field_adjusted_advantage`),
* experimental default parameters ``n`` / ``W`` / ``m`` (:data:`DEFAULT_PARAMS`),
* lightweight validation (:func:`validate_hole_score_history`).

Deliberately pure: only ``pandas`` (already a project dep) and stdlib. No
streamlit, no geometry, no network, and **no dependence on real PGA data** — the
scorer that turns this table into advantages is intentionally *not* implemented
here (see the package spec).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Column contract
# --------------------------------------------------------------------------- #
#: Columns that MUST be present to compute a field-adjusted advantage and to join
#: each occurrence to a v2.5 similar-hole set. ``field_avg_score`` is required so
#: the positive-is-good outcome is always computable without a fallback.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "player_id",
    "tournament_id",
    "year",
    "round",
    "hole_number",
    "course_slug",
    "hole_id_v25",       # join key into v2.5 similar-hole sets ("slug:hole_number")
    "par",
    "player_score",
    "field_avg_score",   # field mean strokes on this hole/round -> difficulty adjust
)

#: Columns that enrich the table but are not required. ``hole_id_v2`` lets a row
#: also join the v2 feature/compact space; the score-to-par and strokes-gained
#: columns are documented *fallback* outcomes (see the spec).
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "player_name",
    "tournament_name",
    "course_name",
    "yardage",
    "hole_id_v2",                 # v2 feature-table id ("slug__NN")
    "player_score_to_par",        # fallback outcome
    "field_score_to_par_avg",
    "player_strokes_gained_hole", # if a real SG-on-hole is available
    "field_adjusted_score",       # cached field_avg_score - player_score
    "made_cut",
)

#: The occurrence grain: one row per player / tournament / year / round / hole.
#: Duplicate rows on this key are a hard error (double-counting a scoring event).
KEY_COLUMNS: tuple[str, ...] = (
    "player_id",
    "tournament_id",
    "year",
    "round",
    "hole_number",
)

#: The canonical positive-is-good outcome column name.
FIELD_ADJUSTED_COL: str = "field_adjusted_score"

# ID shapes. v2 uses "slug__NN" (zero-padded, distinct id space); v2.5 uses
# "slug:hole_number" (see pipeline.modeling.pointcloud.schemas). Kept as local
# regexes so this package stays decoupled from the geometry stack.
_V2_ID_RE = re.compile(r"^[a-z0-9_]+__\d{2}$")
_V25_ID_RE = re.compile(r"^.+:\d+$")

# Plausible value ranges for range checks. Deliberately loose — these catch
# obviously malformed rows (round 0, hole 19, par 7), not soft outliers.
_YEAR_MIN, _YEAR_MAX = 1900, 2100
_ROUND_MIN, _ROUND_MAX = 1, 8        # 1-4 regulation; extra allows playoffs/odd formats
_HOLE_MIN, _HOLE_MAX = 1, 18
_PAR_MIN, _PAR_MAX = 3, 6


# --------------------------------------------------------------------------- #
# Default parameters (EXPERIMENTAL)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AdvantageParams:
    """Tunable knobs for the (future) advantage scorer.

    These defaults are **experimental placeholders** chosen from the sketch, not
    calibrated on data. They live here so the spec, tests, and eventual scorer
    share one source of truth.

    Attributes
    ----------
    n:
        ``n`` — number of top similar holes to pull from v2.5 per target hole
        (by ascending ``total_score`` / rank).
    lookback_years:
        ``W`` — inclusive lookback window in years of history to include,
        counted back from the season being predicted.
    recency_decay:
        ``m`` — per-year recency decay base. An occurrence ``age`` seasons old
        gets weight ``m ** age`` (age 0 = most recent season -> weight 1.0).
    include_current_course_history:
        Whether prior-year occurrences *on course C itself* may enter a hole's
        similar-hole history. Default ``False`` to keep the model about
        transferable hole shape rather than course-specific memory; also avoids
        a subtle leakage foot-gun (see the spec).
    min_occurrences_per_hole:
        Minimum weighted occurrences behind a target-hole advantage before it is
        emitted rather than marked low-coverage.
    min_holes_covered:
        Minimum of the 18 target holes that must have an advantage before a
        player-course score is emitted.
    """

    n: int = 10
    lookback_years: int = 5
    recency_decay: float = 0.85
    include_current_course_history: bool = False
    min_occurrences_per_hole: int = 3
    min_holes_covered: int = 12


#: Shared experimental defaults (``n=10``, ``W=5``, ``m=0.85``).
DEFAULT_PARAMS = AdvantageParams()


# --------------------------------------------------------------------------- #
# Positive-is-good outcome
# --------------------------------------------------------------------------- #
def field_adjusted_advantage(
    field_avg_score: "pd.Series | float",
    player_score: "pd.Series | float",
) -> "pd.Series | float":
    """The primary positive-is-good outcome: ``field_avg_score - player_score``.

    A player who beats the field average on that hole/round (a *lower* score than
    the field) yields a *positive* number, so larger = better. This adjusts each
    occurrence for hole difficulty and field/day/weather conditions, which is why
    it is preferred over raw score or score-to-par. Works element-wise on pandas
    Series or on plain scalars.
    """
    return field_avg_score - player_score


def add_field_adjusted_advantage(
    df: pd.DataFrame, *, overwrite: bool = False
) -> pd.DataFrame:
    """Return a copy of ``df`` with the :data:`FIELD_ADJUSTED_COL` column filled.

    Computes ``field_avg_score - player_score``. If the column already exists it
    is left as-is unless ``overwrite=True``. Requires ``field_avg_score`` and
    ``player_score`` to be present.
    """
    missing = {"field_avg_score", "player_score"} - set(df.columns)
    if missing:
        raise SchemaError(
            f"cannot compute {FIELD_ADJUSTED_COL!r}; missing columns: "
            f"{sorted(missing)}"
        )
    out = df.copy()
    if FIELD_ADJUSTED_COL not in out.columns or overwrite:
        out[FIELD_ADJUSTED_COL] = field_adjusted_advantage(
            out["field_avg_score"], out["player_score"]
        )
    return out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class SchemaError(ValueError):
    """Raised when a historical hole-score table violates the input contract.

    Carries a human-readable message; :attr:`errors` holds the individual
    problems found so callers can surface them all at once.
    """

    def __init__(self, message: str, errors: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or [message]


@dataclass(frozen=True)
class ValidationReport:
    """Summary returned by :func:`validate_hole_score_history` on success."""

    row_count: int
    player_count: int
    course_count: int
    year_min: Optional[int]
    year_max: Optional[int]
    has_field_adjusted: bool
    optional_columns_present: tuple[str, ...] = field(default_factory=tuple)


def validate_hole_score_history(
    df: pd.DataFrame,
    *,
    check_id_formats: bool = True,
    require_field_adjusted_consistency: bool = True,
) -> ValidationReport:
    """Validate a historical hole-score table against the input contract.

    Checks, accumulating *all* problems before failing:

    1. required columns present (:data:`REQUIRED_COLUMNS`),
    2. no nulls in the key columns (:data:`KEY_COLUMNS`),
    3. no duplicate rows on the occurrence grain,
    4. ``year`` / ``round`` / ``hole_number`` / ``par`` within plausible ranges,
    5. ``player_score`` positive,
    6. (optional) ``hole_id_v25`` / ``hole_id_v2`` match their id shapes,
    7. (optional) a supplied ``field_adjusted_score`` equals
       ``field_avg_score - player_score`` within tolerance.

    Returns a :class:`ValidationReport` on success; raises :class:`SchemaError`
    (with every problem in :attr:`SchemaError.errors`) on failure. Does not
    mutate ``df``.
    """
    errors: list[str] = []

    # 1. Required columns.
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        # Without the key/value columns the remaining checks are meaningless.
        raise SchemaError(_join(errors), errors)

    # 2. Null keys.
    key_nulls = {c: int(df[c].isna().sum()) for c in KEY_COLUMNS}
    bad_keys = {c: n for c, n in key_nulls.items() if n}
    if bad_keys:
        errors.append(f"null values in key columns: {bad_keys}")

    # 3. Duplicate occurrences.
    dup_mask = df.duplicated(subset=list(KEY_COLUMNS), keep=False)
    if dup_mask.any():
        dup_keys = (
            df.loc[dup_mask, list(KEY_COLUMNS)]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        errors.append(
            f"{int(dup_mask.sum())} duplicate occurrence rows on grain "
            f"{list(KEY_COLUMNS)}; e.g. {dup_keys}"
        )

    # 4. Range checks.
    errors += _range_errors(df, "year", _YEAR_MIN, _YEAR_MAX)
    errors += _range_errors(df, "round", _ROUND_MIN, _ROUND_MAX)
    errors += _range_errors(df, "hole_number", _HOLE_MIN, _HOLE_MAX)
    errors += _range_errors(df, "par", _PAR_MIN, _PAR_MAX)

    # 5. Positive player score.
    bad_score = _numeric(df["player_score"]) < 1
    if bad_score.any():
        errors.append(f"{int(bad_score.sum())} rows with player_score < 1")

    # 6. ID shapes.
    if check_id_formats:
        errors += _id_format_errors(df, "hole_id_v25", _V25_ID_RE, "slug:hole_number")
        if "hole_id_v2" in df.columns:
            errors += _id_format_errors(df, "hole_id_v2", _V2_ID_RE, "slug__NN")

    # 7. Field-adjusted consistency (only if the cached column is supplied).
    if require_field_adjusted_consistency and FIELD_ADJUSTED_COL in df.columns:
        expected = field_adjusted_advantage(df["field_avg_score"], df["player_score"])
        diff = (_numeric(df[FIELD_ADJUSTED_COL]) - _numeric(expected)).abs()
        bad = diff > 1e-6
        if bad.any():
            errors.append(
                f"{int(bad.sum())} rows where {FIELD_ADJUSTED_COL} != "
                "field_avg_score - player_score (positive-is-good convention)"
            )

    if errors:
        raise SchemaError(_join(errors), errors)

    years = _numeric(df["year"]).dropna()
    return ValidationReport(
        row_count=int(len(df)),
        player_count=int(df["player_id"].nunique()),
        course_count=int(df["course_slug"].nunique()),
        year_min=int(years.min()) if not years.empty else None,
        year_max=int(years.max()) if not years.empty else None,
        has_field_adjusted=FIELD_ADJUSTED_COL in df.columns,
        optional_columns_present=tuple(
            c for c in OPTIONAL_COLUMNS if c in df.columns
        ),
    )


# --------------------------------------------------------------------------- #
# Small internals
# --------------------------------------------------------------------------- #
def _numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric (non-numeric -> NaN) without mutating the input."""
    return pd.to_numeric(s, errors="coerce")


def _range_errors(df: pd.DataFrame, col: str, lo: int, hi: int) -> list[str]:
    """Report non-numeric or out-of-``[lo, hi]`` values in ``col``."""
    vals = _numeric(df[col])
    nan_from_bad = vals.isna() & df[col].notna()
    out_of_range = vals.notna() & ((vals < lo) | (vals > hi))
    bad = nan_from_bad | out_of_range
    if bad.any():
        sample = sorted(set(df.loc[bad, col].tolist()))[:5]
        return [f"{int(bad.sum())} rows with {col} outside [{lo}, {hi}]: e.g. {sample}"]
    return []


def _id_format_errors(
    df: pd.DataFrame, col: str, pattern: re.Pattern[str], shape: str
) -> list[str]:
    """Report values in ``col`` that don't match the ``pattern`` id shape."""
    non_null = df[col].dropna().astype(str)
    bad = non_null[~non_null.map(lambda v: bool(pattern.match(v)))]
    if not bad.empty:
        sample = sorted(set(bad.tolist()))[:5]
        return [f"{len(bad)} rows with malformed {col} (expected '{shape}'): e.g. {sample}"]
    return []


def _join(errors: list[str]) -> str:
    return "invalid hole-score history: " + "; ".join(errors)


__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "KEY_COLUMNS",
    "FIELD_ADJUSTED_COL",
    "AdvantageParams",
    "DEFAULT_PARAMS",
    "SchemaError",
    "ValidationReport",
    "field_adjusted_advantage",
    "add_field_adjusted_advantage",
    "validate_hole_score_history",
]
