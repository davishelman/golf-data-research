"""Offline tests for the golf-plausibility layer (no network, no sklearn needed)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from pipeline.modeling.plausibility import (
    add_plausibility_to_similarity,
    explain_plausibility,
    filter_presentable_matches,
    match_plausibility_flags,
    plausibility_score,
    presented_similarity_table,
)


def hole(par=4, length=400.0, course="a", number=1, water=0.0, bunker=0.01,
         trees=0.10, width=60.0, depth=400.0, green_y=400.0, dogleg=0.05,
         shift=0.0, **extra):
    d = {
        "par": par, "hole_length_m": length, "course_slug": course,
        "hole_number": number, "water_pct": water, "bunker_pct": bunker,
        "trees_pct": trees, "hole_width_m": width, "hole_depth_m": depth,
        "green_y_m": green_y, "dogleg_score": dogleg, "fairway_centerline_shift": shift,
    }
    d.update(extra)
    return d


# --------------------------------------------------------------------------- #
# 1-6: pair-level flags / score / explanation
# --------------------------------------------------------------------------- #

def test_good_match_high_score_and_clean():
    f = match_plausibility_flags(hole(par=4, length=410, course="a"),
                                 hole(par=4, length=415, course="b"))
    assert not f["par_mismatch"] and not f["length_mismatch"]
    assert not f["water_mismatch"] and not f["hazard_mismatch"]
    assert plausibility_score(f) >= 0.9
    assert explain_plausibility(f) == ""


def test_different_par_flagged_and_penalized():
    f = match_plausibility_flags(hole(par=4, course="a"), hole(par=5, course="b"))
    assert f["par_mismatch"] is True
    assert plausibility_score(f) == pytest.approx(0.60)  # 1.0 - 0.40
    assert "different par" in explain_plausibility(f)


def test_large_length_mismatch_flagged():
    f = match_plausibility_flags(hole(par=4, length=400, course="a"),
                                 hole(par=4, length=520, course="b"))
    assert f["length_mismatch"] is True
    assert f["length_diff_m"] == pytest.approx(120.0)
    assert "length diff 120.0m" in explain_plausibility(f)


def test_same_course_penalized_only_when_excluded():
    f = match_plausibility_flags(hole(course="a"), hole(course="a", length=405))
    assert f["same_course"] is True
    assert plausibility_score(f) == 1.0  # no violation key -> no penalty

    f2 = dict(f, same_course_violation=True)
    assert plausibility_score(f2) == pytest.approx(0.80)
    assert "same course" in explain_plausibility(f2)

    feats = pd.DataFrame([
        {"hole_id": "a__01", **hole(course="a", number=1)},
        {"hole_id": "a__02", **hole(course="a", number=2, length=405)},
    ])
    sim = pd.DataFrame({"query_hole_id": ["a__01"], "similar_hole_id": ["a__02"]})
    on = add_plausibility_to_similarity(sim, feats, exclude_same_course=True)
    off = add_plausibility_to_similarity(sim, feats, exclude_same_course=False)
    assert on["plausibility_score"].iloc[0] < off["plausibility_score"].iloc[0]
    assert on["presentable_bad_flag_count"].iloc[0] >= 1
    assert off["presentable_bad_flag_count"].iloc[0] == 0


def test_water_mismatch_flagged():
    f = match_plausibility_flags(hole(par=4, water=0.0, course="a"),
                                 hole(par=4, water=0.30, course="b"))
    assert f["water_mismatch"] is True
    assert "water profile mismatch" in explain_plausibility(f)


def test_missing_optional_columns_do_not_crash():
    q = {"par": 4, "hole_length_m": 400, "course_slug": "a"}   # no hazard/geometry cols
    c = {"par": 4, "hole_length_m": 410, "course_slug": "b"}
    f = match_plausibility_flags(q, c)
    assert f["water_mismatch"] is False
    assert f["hazard_mismatch"] is False
    assert f["geometry_mismatch"] is False
    assert math.isnan(f["water_diff"]) and math.isnan(f["geometry_mismatch_score"])
    assert 0.0 <= plausibility_score(f) <= 1.0
    # a hole id missing from the feature table is annotated, not crashed
    feats = pd.DataFrame([{"hole_id": "a__01", **hole(course="a")}])
    sim = pd.DataFrame({"query_hole_id": ["a__01"], "similar_hole_id": ["ghost__99"]})
    enriched = add_plausibility_to_similarity(sim, feats)
    assert enriched["is_presentable"].iloc[0] is False or not enriched["is_presentable"].iloc[0]
    assert enriched["plausibility_reasons"].iloc[0] == "missing feature row"


# --------------------------------------------------------------------------- #
# 7-8: table-level pipeline
# --------------------------------------------------------------------------- #

def _features_two_courses():
    rows = [{"hole_id": "a__01", **hole(par=4, length=410, course="a", number=1)}]
    for i in range(1, 8):
        rows.append({"hole_id": f"b__{i:02d}",
                     **hole(par=4, length=410 + i, course="b", number=i)})
    return pd.DataFrame(rows)


def test_presented_table_returns_reranked_top_n_per_query():
    feats = _features_two_courses()
    raw = pd.DataFrame({
        "query_hole_id": ["a__01"] * 7,
        "similar_hole_id": [f"b__{i:02d}" for i in range(1, 8)],
        "rank": list(range(1, 8)),
        "distance": [0.5 * i for i in range(1, 8)],
    })
    pres = presented_similarity_table(raw, feats, n_neighbors=3)
    sub = pres[pres["query_hole_id"] == "a__01"].reset_index(drop=True)
    assert len(sub) == 3                              # top-N enforced
    assert list(sub["presented_rank"]) == [1, 2, 3]   # re-ranked
    assert sub["is_presentable"].all()
    # all clean good matches -> tie-broken by raw distance ascending
    assert list(sub["similar_hole_id"]) == ["b__01", "b__02", "b__03"]


def test_presented_output_includes_required_columns():
    feats = pd.DataFrame([
        {"hole_id": "a__01", **hole(course="a", number=1)},
        {"hole_id": "b__01", **hole(course="b", number=1, length=405)},
    ])
    raw = pd.DataFrame({"query_hole_id": ["a__01"], "similar_hole_id": ["b__01"],
                        "rank": [1], "distance": [0.3]})
    pres = presented_similarity_table(raw, feats, n_neighbors=5)
    required = {
        "query_hole_id", "similar_hole_id", "similarity_mode", "raw_rank", "raw_distance",
        "query_course_slug", "similar_course_slug", "query_hole_number", "similar_hole_number",
        "query_length_m", "similar_length_m", "length_diff_m", "length_diff_pct",
        "same_par", "same_course", "par_mismatch", "length_mismatch", "water_mismatch",
        "hazard_mismatch", "geometry_mismatch", "presentable_bad_flag_count",
        "plausibility_score", "is_presentable", "plausibility_reasons", "presented_rank",
    }
    assert required <= set(pres.columns)


def test_filter_presentable_matches_applies_all_gates():
    df = pd.DataFrame({
        "plausibility_score": [1.0, 0.60, 0.90, 0.80],
        "same_par": [True, False, True, True],
        "same_course": [False, False, True, False],
        "presentable_bad_flag_count": [0, 1, 1, 0],
    })
    out = filter_presentable_matches(df, min_score=0.75, require_same_par=True,
                                     exclude_same_course=True, max_bad_flags=0)
    assert len(out) == 2                       # rows 0 and 3 pass
    assert list(out.index) == [0, 3]
    # allowing same course keeps the same-course row too (if it clears other gates)
    out2 = filter_presentable_matches(df, min_score=0.75, require_same_par=True,
                                      exclude_same_course=False, max_bad_flags=1)
    assert 2 in out2.index


def test_clean_presentable_row_reads_ok_not_blank():
    feats = pd.DataFrame([
        {"hole_id": "a__01", **hole(par=4, length=410, course="a", number=1)},
        {"hole_id": "b__01", **hole(par=4, length=412, course="b", number=1)},
    ])
    raw = pd.DataFrame({"query_hole_id": ["a__01"], "similar_hole_id": ["b__01"],
                        "rank": [1], "distance": [0.2]})
    pres = presented_similarity_table(raw, feats, n_neighbors=5)
    row = pres.iloc[0]
    assert row["plausibility_reasons"] == "ok"        # not "" / NaN
    assert row["presentable_bad_flag_count"] == 0
    assert bool(row["is_presentable"]) is True
