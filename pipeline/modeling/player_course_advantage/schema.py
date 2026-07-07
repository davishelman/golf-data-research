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

#: The occurrence grain: one row per player / tournament / year / round /
#: course / hole. ``course_slug`` is part of the key so multi-course events (and
#: hole identity generally) are explicit. Duplicate rows on this key are a hard
#: error (double-counting a scoring event).
KEY_COLUMNS: tuple[str, ...] = (
    "player_id",
    "tournament_id",
    "year",
    "round",
    "course_slug",
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
        ``W`` — lookback window in years. Only strictly-past seasons are
        eligible: ``predict_season - W <= y(o) < predict_season``.
    recency_decay:
        ``m`` — per-year recency decay base. Since only past seasons are eligible
        (``y(o) < predict_season``), season-level age is measured from the
        immediately prior season: ``age = (predict_season - 1) - y(o)``, so the
        prior season (age 0) keeps weight ``m ** 0 = 1.0``. ``m = 1`` disables
        decay. (A future event-date implementation can use a stricter date cutoff
        to admit same-season prior starts without leakage.)
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
    2. no nulls in *any* required column (key columns are called out separately),
    3. no duplicate rows on the occurrence grain (:data:`KEY_COLUMNS`),
    4. ``year`` / ``round`` / ``hole_number`` / ``par`` within plausible ranges,
    5. ``player_score`` and ``field_avg_score`` numeric and ``>= 1``,
    6. (optional) ``hole_id_v25`` / ``hole_id_v2`` match their id shapes *and*
       are consistent with ``course_slug`` / ``hole_number``,
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

    # 2. Nulls in required columns (key columns reported separately for clarity).
    required_nulls = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    bad_required = {c: n for c, n in required_nulls.items() if n}
    if bad_required:
        key_bad = {c: n for c, n in bad_required.items() if c in KEY_COLUMNS}
        non_key_bad = {c: n for c, n in bad_required.items() if c not in KEY_COLUMNS}
        if key_bad:
            errors.append(f"null values in key columns: {key_bad}")
        if non_key_bad:
            errors.append(f"null values in required columns: {non_key_bad}")

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

    # 5. Score columns: numeric, non-null, >= 1. (Nulls are also caught in step 2;
    #    this guards against non-numeric values silently coercing to NaN and
    #    slipping past the >= 1 check.)
    errors += _numeric_min_errors(df, "player_score", 1)
    errors += _numeric_min_errors(df, "field_avg_score", 1)

    # 6. ID shape + consistency with course_slug / hole_number.
    if check_id_formats:
        errors += _id_format_errors(df, "hole_id_v25", _V25_ID_RE, "slug:hole_number")
        errors += _id_consistency_errors(
            df, "hole_id_v25", lambda s, n: f"{s}:{n}", "course_slug:hole_number")
        if "hole_id_v2" in df.columns:
            errors += _id_format_errors(df, "hole_id_v2", _V2_ID_RE, "slug__NN")
            errors += _id_consistency_errors(
                df, "hole_id_v2", lambda s, n: f"{s}__{n:02d}", "course_slug__NN")

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


def _numeric_min_errors(df: pd.DataFrame, col: str, minimum: float) -> list[str]:
    """Report non-numeric, null, or ``< minimum`` values in a score column.

    Reports non-numeric values explicitly so they cannot silently coerce to NaN
    and slip past the ``>= minimum`` bound. (Nulls in required score columns are
    also flagged by the required-null check; this keeps the message specific.)
    """
    raw = df[col]
    vals = _numeric(raw)
    errors: list[str] = []

    non_numeric = raw.notna() & vals.isna()
    if non_numeric.any():
        sample = sorted({str(v) for v in raw[non_numeric].tolist()})[:5]
        errors.append(f"{int(non_numeric.sum())} rows with non-numeric {col}: e.g. {sample}")

    if raw.isna().any():
        errors.append(f"{int(raw.isna().sum())} rows with null {col}")

    below = vals.notna() & (vals < minimum)
    if below.any():
        errors.append(f"{int(below.sum())} rows with {col} < {minimum}")

    return errors


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


def _id_consistency_errors(df: pd.DataFrame, col, build, shape: str) -> list[str]:
    """Report rows where ``col`` != the id ``build(course_slug, hole_number)``.

    Only checks rows where ``col``, ``course_slug``, and a numeric ``hole_number``
    are all present, so it composes with (rather than duplicates) the null and
    range checks. Catches a mismatched course slug *or* hole number.
    """
    needed = [col, "course_slug", "hole_number"]
    if any(c not in df.columns for c in needed):
        return []
    hn = _numeric(df["hole_number"])
    usable = df[col].notna() & df["course_slug"].notna() & hn.notna()
    if not usable.any():
        return []
    actual = df.loc[usable, col].astype(str)
    expected = [
        build(str(s), int(n))
        for s, n in zip(df.loc[usable, "course_slug"], hn[usable])
    ]
    mism = [(a, e) for a, e in zip(actual.tolist(), expected) if a != e]
    if mism:
        return [
            f"{len(mism)} rows where {col} != '{shape}' built from "
            f"course_slug/hole_number: e.g. {mism[:5]}"
        ]
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
