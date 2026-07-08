"""Tests for the annual course target manifest (issue #80)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage.course_targets import (
    COURSE_TARGET_COLUMNS,
    CourseTargetError,
    build_targets_from_assets,
    load_course_targets,
    supported_targets,
    targets_by_status,
)

REPO_EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "data" / "player_course_advantage" / "templates" / "annual_course_targets.example.csv"
)


def _manifest(tmp_path, rows):
    p = tmp_path / "targets.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _row(slug, supported=True, status="missing"):
    return {
        "course_slug": slug, "course_name": slug.title(), "event_name": "", "tour": "PGA Tour",
        "annual_status": "recurring", "supported_in_repo": supported,
        "has_v25_geometry": True, "has_similarity_outputs": supported,
        "expected_holes": 18, "source_priority": "byo_csv", "real_data_status": status, "notes": "",
    }


# --------------------------------------------------------------------------- #
# Load / validate
# --------------------------------------------------------------------------- #
def test_load_ok(tmp_path):
    p = _manifest(tmp_path, [_row("augusta_national"), _row("aronimink_golf_club", False, "unsupported")])
    df = load_course_targets(p)
    assert set(COURSE_TARGET_COLUMNS).issubset(df.columns)
    assert df["supported_in_repo"].dtype == bool


def test_missing_column_raises(tmp_path):
    df = pd.DataFrame([_row("x")]).drop(columns=["real_data_status"])
    p = tmp_path / "m.csv"; df.to_csv(p, index=False)
    with pytest.raises(CourseTargetError):
        load_course_targets(p)


def test_duplicate_slug_raises(tmp_path):
    p = _manifest(tmp_path, [_row("dup"), _row("dup")])
    with pytest.raises(CourseTargetError):
        load_course_targets(p)


def test_bad_status_raises(tmp_path):
    p = _manifest(tmp_path, [_row("x", status="totally_made_up")])
    with pytest.raises(CourseTargetError):
        load_course_targets(p)


def test_supported_and_status_helpers(tmp_path):
    p = _manifest(tmp_path, [
        _row("a", True, "missing"), _row("b", True, "ready"),
        _row("c", False, "unsupported"),
    ])
    df = load_course_targets(p)
    assert list(supported_targets(df)["course_slug"]) == ["a", "b"]
    assert targets_by_status(df)["unsupported"] == 1


# --------------------------------------------------------------------------- #
# Build from assets (tiny fake repo tree)
# --------------------------------------------------------------------------- #
def test_build_from_assets(tmp_path):
    courses = tmp_path / "courses"
    for slug, holes in [("augusta_national", 18), ("geom_only_course", 18)]:
        d = courses / slug
        d.mkdir(parents=True)
        (d / "course_summary.json").write_text(
            json.dumps({"course": slug.replace("_", " ").title(),
                        "holes": list(range(holes))}), encoding="utf-8")
    # similarity only covers augusta_national
    sim = tmp_path / "sim.csv"
    pd.DataFrame({"target_hole_id": [f"augusta_national:{h}" for h in range(1, 19)]}).to_csv(sim, index=False)

    df = build_targets_from_assets(courses, sim)
    by = df.set_index("course_slug")
    assert bool(by.loc["augusta_national", "supported_in_repo"]) is True
    assert by.loc["augusta_national", "real_data_status"] == "missing"
    assert bool(by.loc["geom_only_course", "supported_in_repo"]) is False
    assert by.loc["geom_only_course", "real_data_status"] == "unsupported"


# --------------------------------------------------------------------------- #
# The committed example manifest is valid + reflects the real universe
# --------------------------------------------------------------------------- #
def test_committed_example_manifest_is_valid():
    df = load_course_targets(REPO_EXAMPLE)
    assert len(df) >= 30
    assert df["supported_in_repo"].sum() >= 20      # a real, non-trivial universe
    # every unsupported course is flagged unsupported, every supported has similarity
    assert (df.loc[~df["supported_in_repo"], "real_data_status"] == "unsupported").all()
    assert df.loc[df["supported_in_repo"], "has_similarity_outputs"].all()
