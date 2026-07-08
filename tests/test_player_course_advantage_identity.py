"""Tests for course/hole identity mapping diagnostics (issue #71).

Tiny synthetic fixtures only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage.identity import (
    IdentityError,
    build_mapping_report,
    hole_id_v25,
    load_course_aliases,
    resolve_course_slug,
    slugify,
)

KNOWN = {"augusta_national", "pebble_beach_golf_links"}


def _aliases_csv(tmp_path, rows):
    p = tmp_path / "aliases.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def test_slugify():
    assert slugify("Augusta National Golf Club") == "augusta_national_golf_club"


def test_resolve_via_alias_and_known_slug():
    aliases = {"augusta national golf club": "augusta_national"}
    assert resolve_course_slug("Augusta National Golf Club", aliases, known_slugs=KNOWN) == "augusta_national"
    # already-canonical slug passes straight through
    assert resolve_course_slug("augusta_national", None, known_slugs=KNOWN) == "augusta_national"


def test_resolve_unmapped_returns_none():
    # No alias, slugify won't match a known slug -> unmapped.
    assert resolve_course_slug("Some Random Muni", {}, known_slugs=KNOWN) is None


def test_hole_id_v25_format():
    assert hole_id_v25("augusta_national", 13) == "augusta_national:13"


# --------------------------------------------------------------------------- #
# Alias loading
# --------------------------------------------------------------------------- #
def test_load_aliases_ok(tmp_path):
    p = _aliases_csv(tmp_path, [
        {"course_name": "Augusta National Golf Club", "course_slug": "augusta_national"},
    ])
    aliases = load_course_aliases(p)
    assert aliases["augusta national golf club"] == "augusta_national"


def test_load_aliases_ambiguous_raises(tmp_path):
    p = _aliases_csv(tmp_path, [
        {"course_name": "The Open Course", "course_slug": "course_a"},
        {"course_name": "The Open Course", "course_slug": "course_b"},
    ])
    with pytest.raises(IdentityError):
        load_course_aliases(p)


def test_load_aliases_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"name": ["x"], "slug": ["y"]}).to_csv(p, index=False)
    with pytest.raises(IdentityError):
        load_course_aliases(p)


# --------------------------------------------------------------------------- #
# Mapping report
# --------------------------------------------------------------------------- #
def _raw(rows):
    return pd.DataFrame(rows)


def test_clean_mapping_report():
    aliases = {"augusta national golf club": "augusta_national"}
    raw = _raw([
        {"source_event_id": "E1", "season": 2023, "round": 1,
         "course_name": "Augusta National Golf Club", "hole_number": h}
        for h in range(1, 4)
    ])
    rep = build_mapping_report(raw, aliases, known_slugs=KNOWN)
    assert rep["total_rows"] == 3
    assert rep["mapped_rows"] == 3
    assert rep["unmapped_course_rows"] == 0
    assert rep["unmapped_hole_rows"] == 0


def test_unmapped_course_reported():
    raw = _raw([
        {"source_event_id": "E1", "season": 2023, "round": 1,
         "course_name": "Unknown Links", "hole_number": 1},
        {"source_event_id": "E1", "season": 2023, "round": 1,
         "course_name": "Unknown Links", "hole_number": 2},
    ])
    rep = build_mapping_report(raw, {}, known_slugs=KNOWN)
    assert rep["unmapped_course_rows"] == 2
    assert "Unknown Links" in rep["most_frequent_unmapped_course_labels"]


def test_unmapped_hole_reported():
    aliases = {"augusta national golf club": "augusta_national"}
    known_hole_ids = {f"augusta_national:{h}" for h in range(1, 10)}  # only holes 1..9
    raw = _raw([
        {"source_event_id": "E1", "season": 2023, "round": 1,
         "course_name": "Augusta National Golf Club", "hole_number": 5},
        {"source_event_id": "E1", "season": 2023, "round": 1,
         "course_name": "Augusta National Golf Club", "hole_number": 14},  # not in universe
    ])
    rep = build_mapping_report(raw, aliases, known_slugs=KNOWN, known_hole_ids=known_hole_ids)
    assert rep["unmapped_hole_rows"] == 1


def test_duplicate_source_identifier_counted():
    aliases = {"augusta national golf club": "augusta_national"}
    row = {"source_event_id": "E1", "season": 2023, "round": 1,
           "course_name": "Augusta National Golf Club", "hole_number": 1}
    rep = build_mapping_report(_raw([row, row]), aliases, known_slugs=KNOWN,
                               source_id_cols=("source_event_id", "season", "round",
                                               "course_name", "hole_number"))
    assert rep["duplicate_source_identifier_count"] == 2
