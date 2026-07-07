"""Schema tests for the player-course advantage input contract (issue #31).

Fake data only — no real PGA data, no network, no v2/v2.5 artifacts. Exercises
``validate_hole_score_history`` and the positive-is-good outcome helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    DEFAULT_PARAMS,
    KEY_COLUMNS,
    REQUIRED_COLUMNS,
    AdvantageParams,
    SchemaError,
    ValidationReport,
    add_field_adjusted_advantage,
    field_adjusted_advantage,
    validate_hole_score_history,
)


def _fake_row(**overrides) -> dict:
    """One valid occurrence row; override any field for negative cases."""
    row = {
        "player_id": "p1",
        "player_name": "Tester McTestface",
        "tournament_id": "t1",
        "tournament_name": "Fake Open",
        "course_slug": "augusta_national",
        "course_name": "Augusta National",
        "year": 2023,
        "round": 1,
        "hole_number": 13,
        "hole_id_v2": "augusta_national__13",
        "hole_id_v25": "augusta_national:13",
        "par": 5,
        "yardage": 545,
        "player_score": 4,
        "player_score_to_par": -1,
        "field_avg_score": 4.7,
        "field_score_to_par_avg": -0.3,
    }
    row.update(overrides)
    return row


def _fake_history(n_players: int = 2, n_holes: int = 3) -> pd.DataFrame:
    """A small valid dataframe spanning a couple players / holes / years."""
    rows = []
    for pi in range(n_players):
        for hi in range(n_holes):
            for yr in (2021, 2022, 2023):
                rows.append(
                    _fake_row(
                        player_id=f"p{pi}",
                        hole_number=hi + 1,
                        hole_id_v2=f"augusta_national__{hi + 1:02d}",
                        hole_id_v25=f"augusta_national:{hi + 1}",
                        year=yr,
                        player_score=4 + (hi % 2),
                    )
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_valid_fake_dataframe_passes():
    df = _fake_history()
    report = validate_hole_score_history(df)
    assert isinstance(report, ValidationReport)
    assert report.row_count == len(df)
    assert report.player_count == 2
    assert report.course_count == 1
    assert report.year_min == 2021 and report.year_max == 2023


def test_single_row_minimal_required_only_passes():
    # Only the required columns present (no optional enrichment) still validates.
    df = pd.DataFrame([{c: _fake_row()[c] for c in REQUIRED_COLUMNS}])
    report = validate_hole_score_history(df)
    assert report.row_count == 1
    assert report.optional_columns_present == ()


# --------------------------------------------------------------------------- #
# Missing required columns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["player_id", "hole_id_v25", "field_avg_score", "par"])
def test_missing_required_column_fails_clearly(missing):
    df = _fake_history().drop(columns=[missing])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "missing required columns" in str(exc.value)
    assert missing in str(exc.value)


def test_field_avg_score_is_required():
    # Required so the positive-is-good outcome is always computable.
    assert "field_avg_score" in REQUIRED_COLUMNS


# --------------------------------------------------------------------------- #
# Duplicate occurrence rows
# --------------------------------------------------------------------------- #
def test_duplicate_occurrence_rows_fail_clearly():
    df = _fake_history()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(dup)
    msg = str(exc.value)
    assert "duplicate occurrence rows" in msg
    # The grain that was violated is named in the error.
    for key in KEY_COLUMNS:
        assert key in msg


def test_same_hole_different_round_is_not_a_duplicate():
    a = _fake_row(round=1)
    b = _fake_row(round=2)  # same player/tournament/year/hole, different round
    report = validate_hole_score_history(pd.DataFrame([a, b]))
    assert report.row_count == 2


# --------------------------------------------------------------------------- #
# Range checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "col,bad_value",
    [("hole_number", 19), ("hole_number", 0), ("round", 0), ("par", 7), ("year", 1800)],
)
def test_out_of_range_values_fail(col, bad_value):
    df = pd.DataFrame([_fake_row(), _fake_row(**{col: bad_value})])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert col in str(exc.value)


def test_non_positive_player_score_fails():
    df = pd.DataFrame([_fake_row(), _fake_row(player_score=0)])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "player_score" in str(exc.value)


def test_null_key_fails():
    df = pd.DataFrame([_fake_row(), _fake_row(round=2, player_id=None)])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "null values in key columns" in str(exc.value)
    assert "player_id" in str(exc.value)


def test_null_required_non_key_column_fails():
    # `par` is required but not part of the occurrence key -> must still fail.
    df = pd.DataFrame([_fake_row(), _fake_row(round=2, par=None)])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "null values in required columns" in str(exc.value)
    assert "par" in str(exc.value)


def test_non_numeric_player_score_fails():
    # A non-numeric player_score must be rejected, not coerced to NaN and passed.
    df = pd.DataFrame([_fake_row(player_score="abc")])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "non-numeric player_score" in str(exc.value)


def test_non_numeric_field_avg_score_fails():
    df = pd.DataFrame([_fake_row(field_avg_score="oops")])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "non-numeric field_avg_score" in str(exc.value)


def test_null_field_avg_score_fails():
    df = pd.DataFrame([_fake_row(field_avg_score=None)])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "field_avg_score" in str(exc.value)


def test_below_minimum_field_avg_score_fails():
    df = pd.DataFrame([_fake_row(field_avg_score=0.0)])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "field_avg_score < 1" in str(exc.value)


def test_multiple_problems_are_all_reported():
    # A row that is both out-of-range and has a bad score -> both surfaced.
    df = pd.DataFrame([_fake_row(), _fake_row(hole_number=99, player_score=-1)])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert len(exc.value.errors) >= 2


# --------------------------------------------------------------------------- #
# ID formats: v2 and v2.5
# --------------------------------------------------------------------------- #
def test_valid_v2_and_v25_id_formats_pass():
    # Ids must be internally consistent with course_slug / hole_number.
    df = pd.DataFrame([
        _fake_row(course_slug="augusta_national", hole_number=1,
                  hole_id_v2="augusta_national__01", hole_id_v25="augusta_national:1"),
        _fake_row(course_slug="pebble_beach", hole_number=18,
                  hole_id_v2="pebble_beach__18", hole_id_v25="pebble_beach:18"),
    ])
    report = validate_hole_score_history(df)
    assert report.row_count == 2
    assert "hole_id_v2" in report.optional_columns_present


def test_malformed_v25_id_fails():
    df = pd.DataFrame([_fake_row(hole_id_v25="augusta_national-13")])  # dash, not colon
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "hole_id_v25" in str(exc.value)


def test_malformed_v2_id_fails():
    df = pd.DataFrame([_fake_row(hole_id_v2="augusta_national:13")])  # v2.5 shape in v2 col
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "hole_id_v2" in str(exc.value)


def test_id_format_check_can_be_disabled():
    df = pd.DataFrame([_fake_row(hole_id_v25="anything-goes")])
    # Should not raise for the id shape when the check is turned off.
    validate_hole_score_history(df, check_id_formats=False)


def test_mismatched_v25_hole_number_fails():
    # Well-formed id, but hole_number says 13 -> inconsistent.
    df = pd.DataFrame([_fake_row(hole_number=13, hole_id_v25="augusta_national:7")])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "hole_id_v25" in str(exc.value)


def test_mismatched_v25_course_slug_fails():
    df = pd.DataFrame([_fake_row(course_slug="augusta_national",
                                 hole_number=13, hole_id_v25="pebble_beach:13")])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "hole_id_v25" in str(exc.value)


def test_mismatched_v2_hole_number_fails():
    df = pd.DataFrame([_fake_row(hole_number=13, hole_id_v2="augusta_national__07")])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "hole_id_v2" in str(exc.value)


def test_mismatched_v2_course_slug_fails():
    df = pd.DataFrame([_fake_row(course_slug="augusta_national",
                                 hole_number=13, hole_id_v2="pebble_beach__13")])
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "hole_id_v2" in str(exc.value)


# --------------------------------------------------------------------------- #
# Positive-is-good field-adjusted outcome
# --------------------------------------------------------------------------- #
def test_field_adjusted_advantage_positive_when_beating_field():
    # Player shot 4, field averaged 4.7 -> +0.7 advantage (positive is good).
    assert field_adjusted_advantage(4.7, 4.0) == pytest.approx(0.7)
    # Worse than field -> negative.
    assert field_adjusted_advantage(4.0, 5.0) == pytest.approx(-1.0)


def test_add_field_adjusted_advantage_computes_column():
    df = _fake_history()
    out = add_field_adjusted_advantage(df)
    assert "field_adjusted_score" in out.columns
    expected = out["field_avg_score"] - out["player_score"]
    assert (out["field_adjusted_score"] == expected).all()
    # Original frame untouched.
    assert "field_adjusted_score" not in df.columns


def test_consistent_cached_field_adjusted_passes():
    df = add_field_adjusted_advantage(_fake_history())
    report = validate_hole_score_history(df)
    assert report.has_field_adjusted


def test_inconsistent_cached_field_adjusted_fails():
    df = _fake_history()
    df["field_adjusted_score"] = 99.0  # wrong: != field_avg - player_score
    with pytest.raises(SchemaError) as exc:
        validate_hole_score_history(df)
    assert "field_adjusted_score" in str(exc.value)


# --------------------------------------------------------------------------- #
# Experimental defaults are wired and stable
# --------------------------------------------------------------------------- #
def test_course_slug_in_key_columns():
    # Occurrence grain must include course_slug (protects multi-course events).
    assert "course_slug" in KEY_COLUMNS
    assert KEY_COLUMNS == (
        "player_id", "tournament_id", "year", "round", "course_slug", "hole_number",
    )


def test_default_params_values():
    assert isinstance(DEFAULT_PARAMS, AdvantageParams)
    assert DEFAULT_PARAMS.n == 10
    assert DEFAULT_PARAMS.lookback_years == 5
    assert DEFAULT_PARAMS.recency_decay == pytest.approx(0.85)
    assert DEFAULT_PARAMS.include_current_course_history is False
