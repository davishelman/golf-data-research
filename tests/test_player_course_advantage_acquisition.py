"""Tests for annual-course data acquisition / import (issue #81). Synthetic only."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage.acquisition import (
    find_course_raw_file,
    import_annual_courses,
    resolve_sources,
)

SUPPORTED = ["augusta_national", "pebble_beach_golf_links"]


def _targets():
    rows = []
    for slug in SUPPORTED:
        rows.append(_target_row(slug, True))
    rows.append(_target_row("geom_only_course", False))
    return pd.DataFrame(rows)


def _target_row(slug, supported):
    return {
        "course_slug": slug, "course_name": slug.title(), "event_name": "", "tour": "PGA Tour",
        "annual_status": "recurring", "supported_in_repo": supported,
        "has_v25_geometry": True, "has_similarity_outputs": supported, "expected_holes": 9,
        "source_priority": "byo_csv", "real_data_status": "missing" if supported else "unsupported",
        "notes": "",
    }


def _raw_csv(course_slug, n_holes=9, players=("Alice", "Bob")):
    rows = []
    scores = {"Alice": 4, "Bob": 5}
    for pid in players:
        for h in range(1, n_holes + 1):
            rows.append({
                "source_event_id": "E1", "event_name": "Syn Open", "season": 2023, "round": 1,
                "course_slug": course_slug, "course_name": course_slug.title(),
                "hole_number": h, "player_id": pid, "player_name": pid,
                "score": scores[pid], "par": 4,
            })
    return pd.DataFrame(rows)


def _write_raw(raw_dir, slug, **kw):
    raw_dir.mkdir(parents=True, exist_ok=True)
    _raw_csv(slug, **kw).to_csv(raw_dir / f"{slug}.csv", index=False)


def _outcomes(course_slug):
    return pd.DataFrame({
        "event_id": ["E1"] * 2, "event_name": ["Syn Open"] * 2, "season": [2023] * 2,
        "course_slug": [course_slug] * 2, "player_id": ["Alice", "Bob"],
        "player_name": ["Alice", "Bob"], "finish_position": [1, 2],
        "final_score": [-5, -3], "strokes_to_field": [-2.0, -1.0],
    })


# --------------------------------------------------------------------------- #
# Sources / discovery
# --------------------------------------------------------------------------- #
def test_resolve_sources_no_secrets():
    src = resolve_sources(env={})  # no creds
    assert src["byo_csv"] is True
    assert src["data_golf_api"] is False
    assert src["shotlink_export_path"] is None


def test_find_course_raw_file(tmp_path):
    _write_raw(tmp_path / "raw", "augusta_national")
    assert find_course_raw_file(tmp_path / "raw", "augusta_national") is not None
    assert find_course_raw_file(tmp_path / "raw", "nope") is None


# --------------------------------------------------------------------------- #
# Import classification
# --------------------------------------------------------------------------- #
def test_ready_when_history_and_outcomes(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "augusta_national")
    res = import_annual_courses(_targets(), raw, outcomes=_outcomes("augusta_national"))
    st = res.status.set_index("course_slug")
    assert st.loc["augusta_national", "status"] == "ready"


def test_partial_when_outcomes_missing_blocks_backtest(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "augusta_national")
    res = import_annual_courses(_targets(), raw, outcomes=None)  # no outcomes
    row = res.status.set_index("course_slug").loc["augusta_national"]
    assert row["status"] == "partial"
    assert "outcomes" in row["reason"].lower()


def test_missing_when_no_raw(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "augusta_national")  # pebble has no raw
    res = import_annual_courses(_targets(), raw, outcomes=_outcomes("augusta_national"))
    st = res.status.set_index("course_slug")
    assert st.loc["pebble_beach_golf_links", "status"] == "missing"


def test_unsupported_course_flagged(tmp_path):
    res = import_annual_courses(_targets(), tmp_path / "raw")
    assert res.status.set_index("course_slug").loc["geom_only_course", "status"] == "unsupported"


def test_partial_on_normalization_failure(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)
    bad = _raw_csv("augusta_national").drop(columns=["score"])  # missing required col
    bad.to_csv(raw / "augusta_national.csv", index=False)
    res = import_annual_courses(_targets(), raw, outcomes=_outcomes("augusta_national"))
    row = res.status.set_index("course_slug").loc["augusta_national"]
    assert row["status"] == "partial"
    assert "failed" in row["reason"].lower()


# --------------------------------------------------------------------------- #
# Combined output + validation + private writes
# --------------------------------------------------------------------------- #
def test_combined_history_and_validation(tmp_path):
    from pipeline.modeling.player_course_advantage import validate_hole_score_history
    raw = tmp_path / "raw"
    _write_raw(raw, "augusta_national")
    _write_raw(raw, "pebble_beach_golf_links")
    res = import_annual_courses(_targets(), raw, outcomes=pd.concat(
        [_outcomes("augusta_national"), _outcomes("pebble_beach_golf_links")], ignore_index=True))
    # combined spans both courses and re-validates
    assert set(res.combined_history["course_slug"]) == set(SUPPORTED)
    validate_hole_score_history(res.combined_history)


def test_writes_only_under_out_root(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "augusta_national")
    out = tmp_path / "private"
    import_annual_courses(_targets(), raw, outcomes=_outcomes("augusta_national"), out_root=out)
    assert (out / "coverage" / "annual_course_data_status.csv").exists()
    assert (out / "normalized" / "augusta_national_history.csv").exists()
    assert (out / "logs" / "import_report.md").exists()
    assert all(p.is_relative_to(tmp_path) for p in out.rglob("*"))


def test_does_not_crash_on_empty_raw_dir(tmp_path):
    (tmp_path / "raw").mkdir()
    res = import_annual_courses(_targets(), tmp_path / "raw")  # nothing available, no outcomes
    assert set(res.status["status"]) <= {"missing", "unsupported"}
    assert res.combined_history.empty
