"""Offline tests for the demo app's UI-free logic (no streamlit, no real artifact)."""

from __future__ import annotations

import pandas as pd

from pipeline.modeling import demo_utils as du


def _features():
    rows = []
    for slug in ("aaa", "bbb"):
        for n in (1, 2):
            rows.append({
                "hole_id": f"{slug}__{n:02d}", "course_slug": slug,
                "course_name": slug.upper(), "hole_number": n, "par": 4,
                "hole_length_m": 400.0 + n, "hole_length_yd": round((400.0 + n) * 1.09361),
                "bunker_pct": 0.01, "water_pct": 0.0, "trees_pct": 0.1,
                "z_range": 10.0, "dogleg_score": 0.05,
            })
    return pd.DataFrame(rows)


def _presented(feats):
    # cross-course presented rows for aaa__01
    return pd.DataFrame({
        "query_hole_id": ["aaa__01", "aaa__01"],
        "similar_hole_id": ["bbb__01", "bbb__02"],
        "presented_rank": [1, 2], "plausibility_score": [1.0, 0.85],
        "plausibility_reasons": ["ok", "water profile mismatch"],
        "length_diff_m": [1.0, 2.0], "same_course": [False, False],
    })


def _mode(feats):
    return pd.DataFrame({
        "query_hole_id": ["aaa__01"] * 3,
        "similar_hole_id": ["aaa__02", "bbb__01", "bbb__02"],
        "rank": [1, 2, 3], "distance": [0.1, 0.2, 0.3],
        "similarity_mode": ["hazard"] * 3, "length_diff_m": [1.0, 1.0, 2.0],
        "same_course": [True, False, False],
    })


def _art():
    feats = _features()
    return {
        "features": feats,
        "presented_similarity": {"overall_v2": _presented(feats)},
        "similarity_modes": {"hazard": _mode(feats), "overall_v2": _mode(feats)},
        "similarity_v2": None,
        "compact_dir": None,
        "label": "test artifact",
    }


def test_discover_artifact_root(tmp_path):
    (tmp_path / "data").mkdir()
    _features().to_parquet(tmp_path / "data" / "hole_features.parquet")
    found = du.discover_artifact_root((str(tmp_path), "nope"))
    assert found == tmp_path
    assert du.discover_artifact_root(("nope1", "nope2")) is None


def test_hole_label():
    row = {"hole_number": 1, "par": 4, "hole_length_yd": 448.2}
    assert du.hole_label(row) == "Hole 1 — Par 4 — 448 yd"
    assert du.hole_label({"hole_number": 7, "par": 3}) == "Hole 7 — Par 3"


def test_course_and_hole_selectors():
    feats = _features()
    assert du.course_slugs(feats) == ["aaa", "bbb"]
    holes = du.holes_for_course(feats, "aaa")
    assert list(holes["hole_number"]) == [1, 2]


def test_get_presented_prefers_artifact():
    table, source = du.get_presented(_art())
    assert source == "artifact"
    assert "ok" in set(table["plausibility_reasons"])


def test_get_presented_computes_live_when_absent():
    feats = _features()
    raw_v2 = pd.DataFrame({
        "query_hole_id": ["aaa__01"], "similar_hole_id": ["bbb__01"],
        "rank": [1], "distance": [0.2],
    })
    art = {"features": feats, "presented_similarity": {}, "similarity_v2": raw_v2}
    table, source = du.get_presented(art)
    assert source == "computed live"
    assert "plausibility_score" in table.columns


def test_presented_view_enriched_and_clean():
    table, kind, source = du.similarity_results(
        _art(), "Presented plays-like", "aaa__01", 10, show_same_course=False)
    assert kind == "presented" and source == "artifact"
    assert {"similar_par", "similar_length_yd", "similar_course_slug"} <= set(table.columns)
    assert not table["plausibility_reasons"].isna().any()
    assert (~table["same_course"]).all()                 # cross-course
    cols = du.display_columns(table, kind, show_raw_cols=False)
    assert "presented_rank" in cols and "plausibility_reasons" in cols


def test_facet_view_uses_mode_and_can_filter_same_course():
    art = _art()
    full, _, _ = du.similarity_results(art, "Facet: hazard", "aaa__01", 10, show_same_course=True)
    no_sc, kind, _ = du.similarity_results(art, "Facet: hazard", "aaa__01", 10, show_same_course=False)
    assert kind == "raw_mode"
    assert (full["same_course"]).any()                   # same-course present when allowed
    assert not no_sc["same_course"].any()                # filtered out when off
    cols = du.display_columns(no_sc, kind, show_raw_cols=False)
    assert "rank" in cols and "distance" in cols


def test_compare_features_groups_and_difference():
    cmp = du.compare_features(_features(), "aaa__01", "bbb__01")
    assert set(cmp["group"]) <= {"Basic", "Hazards", "Terrain", "Shape"}
    assert "difference" in cmp.columns
    par_row = cmp[cmp["feature"] == "par"].iloc[0]
    assert par_row["difference"] == 0.0


def test_dataset_summary():
    s = du.dataset_summary(_art())
    assert s["courses"] == 2 and s["holes"] == 4
    assert s["presented_similarity_rows"] == 2
    assert "hazard" in s["similarity_modes"]
    assert s["compact_point_clouds"] == 0
