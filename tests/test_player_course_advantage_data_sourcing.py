"""Guards the #46 data-sourcing contract: the committed synthetic sample fixture
must satisfy the exact schema real data will be normalized into.

The fixture is tiny and obviously fake (see the data-sourcing plan doc). This is
not real data — it pins the normalization *target* so the contract can't silently
drift.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.modeling.player_course_advantage import (
    REQUIRED_COLUMNS,
    validate_hole_score_history,
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures" / "player_course_advantage" / "sample_hole_scores.csv"
)


def test_sample_fixture_exists():
    assert FIXTURE.exists(), "synthetic sample hole-scores fixture is missing"


def test_sample_fixture_passes_schema_validation():
    df = pd.read_csv(FIXTURE)
    # All required columns present, and the full contract holds (ids consistent,
    # field_adjusted_score == field_avg_score - player_score, no dup grain, ...).
    report = validate_hole_score_history(df)
    assert set(REQUIRED_COLUMNS).issubset(df.columns)
    assert report.row_count == len(df)
    assert report.player_count == 3
    assert report.has_field_adjusted is True


def test_sample_fixture_is_small_and_synthetic():
    df = pd.read_csv(FIXTURE)
    # A committed fixture must stay tiny — real data is never committed (#46).
    assert len(df) <= 50
    # Player ids are the synthetic placeholders, not real tour identifiers.
    assert set(df["player_id"]) <= {"p001", "p002", "p003"}
