"""Tests for the real-data ingestion adapter (issue #70).

Tiny synthetic fixtures only — no real data. Verifies normalization produces a
schema-valid canonical history, field_avg_score handling, and clear failures.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    validate_hole_score_history,
)
from pipeline.modeling.player_course_advantage.ingestion import (
    IngestionError,
    normalize_and_validate,
    normalize_hole_scores,
)
from pipeline.modeling.player_course_advantage.schema import SchemaError

ALIASES = {"augusta national golf club": "augusta_national"}


def _raw(with_field_avg=False, with_par=True, n_holes=3, players=("Alice", "Bob"),
         course="Augusta National Golf Club"):
    rows = []
    # Two players so a computed field average is a real mean of >1 score.
    scores = {"Alice": 4, "Bob": 6}
    for pid in players:
        for h in range(1, n_holes + 1):
            row = {
                "source_event_id": "E1", "event_name": "Synthetic Open", "season": 2023,
                "round": 1, "course_name": course, "hole_number": h,
                "player_id": pid, "player_name": pid, "score": scores[pid],
            }
            if with_par:
                row["par"] = 4
            if with_field_avg:
                row["field_avg_score"] = 5.0
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_normalizes_and_passes_schema():
    canonical = normalize_hole_scores(_raw(), aliases=ALIASES, known_slugs={"augusta_national"})
    # validates without raising
    report = validate_hole_score_history(canonical)
    assert report.player_count == 2
    assert set(canonical["course_slug"]) == {"augusta_national"}
    assert (canonical["hole_id_v25"] == "augusta_national:" + canonical["hole_number"].astype(str)).all()


def test_field_avg_computed_when_missing():
    canonical = normalize_hole_scores(_raw(with_field_avg=False), aliases=ALIASES)
    # Alice 4, Bob 6 on each hole -> field mean 5.0.
    assert (canonical["field_avg_score"] == 5.0).all()


def test_existing_field_avg_preserved():
    canonical = normalize_hole_scores(_raw(with_field_avg=True), aliases=ALIASES)
    assert (canonical["field_avg_score"] == 5.0).all()


def test_par_filled_from_partial():
    raw = _raw(with_par=True)
    raw.loc[0, "par"] = None  # one missing par; same hole has par elsewhere
    canonical = normalize_hole_scores(raw, aliases=ALIASES)
    assert canonical["par"].notna().all()


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #
def test_missing_required_column_errors():
    raw = _raw().drop(columns=["score"])
    with pytest.raises(IngestionError) as exc:
        normalize_hole_scores(raw, aliases=ALIASES)
    assert "score" in str(exc.value)


def test_unmapped_course_errors():
    raw = _raw(course="Totally Unknown Course")
    with pytest.raises(IngestionError) as exc:
        normalize_hole_scores(raw, aliases=ALIASES, known_slugs={"augusta_national"})
    assert "unmapped course" in str(exc.value).lower()


def test_out_of_range_hole_errors():
    raw = _raw()
    raw.loc[0, "hole_number"] = 25
    with pytest.raises(IngestionError):
        normalize_hole_scores(raw, aliases=ALIASES)


def test_par_missing_everywhere_errors():
    raw = _raw(with_par=False)
    with pytest.raises(IngestionError) as exc:
        normalize_hole_scores(raw, aliases=ALIASES)
    assert "par" in str(exc.value).lower()


def test_invalid_score_fails_validation():
    raw = _raw()
    raw["score"] = 0  # < 1 -> schema rejects
    with pytest.raises(SchemaError):
        normalize_and_validate(raw, aliases=ALIASES)


def test_duplicate_grain_fails_validation():
    raw = _raw()
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError):
        normalize_and_validate(dup, aliases=ALIASES)


def test_normalize_and_validate_returns_reports():
    canonical, report, mapping = normalize_and_validate(
        _raw(), aliases=ALIASES, known_slugs={"augusta_national"})
    assert report.row_count == len(canonical)
    assert mapping["total_rows"] == len(canonical)
    assert mapping["unmapped_course_rows"] == 0


def test_does_not_mutate_input():
    raw = _raw()
    before = raw.copy()
    normalize_hole_scores(raw, aliases=ALIASES)
    pd.testing.assert_frame_equal(raw, before)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_cli():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "normalize_player_hole_scores.py"
    spec = importlib.util.spec_from_file_location("normalize_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_writes_to_tmpdir(tmp_path):
    cli = _load_cli()
    raw_csv = tmp_path / "raw.csv"
    _raw().to_csv(raw_csv, index=False)
    aliases_csv = tmp_path / "aliases.csv"
    pd.DataFrame([{"course_name": "Augusta National Golf Club",
                   "course_slug": "augusta_national"}]).to_csv(aliases_csv, index=False)
    out_csv = tmp_path / "out" / "normalized_history.csv"

    rc = cli.main(["--input", str(raw_csv), "--output", str(out_csv),
                   "--course-aliases", str(aliases_csv)])
    assert rc == 0
    assert out_csv.exists()
    # canonical output re-validates
    from pipeline.modeling.player_course_advantage import validate_hole_score_history
    validate_hole_score_history(pd.read_csv(out_csv))


def test_cli_fails_on_unmapped_course(tmp_path, capsys):
    cli = _load_cli()
    raw_csv = tmp_path / "raw.csv"
    _raw(course="Unknown Muni").to_csv(raw_csv, index=False)
    aliases_csv = tmp_path / "aliases.csv"
    pd.DataFrame([{"course_name": "Augusta National Golf Club",
                   "course_slug": "augusta_national"}]).to_csv(aliases_csv, index=False)
    rc = cli.main(["--input", str(raw_csv), "--output", str(tmp_path / "o.csv"),
                   "--course-aliases", str(aliases_csv)])
    assert rc == 2
    assert not (tmp_path / "o.csv").exists()
