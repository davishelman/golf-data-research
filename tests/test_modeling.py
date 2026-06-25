"""Offline tests for the hole-similarity modeling layer (no network, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.modeling.hole_feature_builder import (
    HolePoints,
    build_one_hole_features,
    green_y_value,
    label_features,
    left_right_features,
    strategic_features,
    zone_features,
)
from pipeline.modeling.similarity import (
    build_feature_matrix,
    cluster_kmeans,
    feature_columns,
    feature_summary,
    nearest_neighbor_table,
    similar_holes,
)


def mk(specs) -> HolePoints:
    """specs: list of (x, y, z, label, count)."""
    xs, ys, zs, ls = [], [], [], []
    for x, y, z, lab, cnt in specs:
        xs += [x] * cnt; ys += [y] * cnt; zs += [z] * cnt; ls += [lab] * cnt
    return HolePoints(np.array(xs, float), np.array(ys, float),
                      np.array(zs, float), np.array(ls, dtype=object))


# ---------------------------------------------------------------------------
# Label composition + rough collapsing
# ---------------------------------------------------------------------------


def test_label_percentages_and_rough_collapse():
    p = mk([(0, 0, 0, "fairway", 60), (0, 0, 0, "rough_osm", 10),
            (0, 0, 0, "rough_inferred", 30)])
    lf = label_features(p)
    assert lf["fairway_pct"] == pytest.approx(0.6)
    assert lf["rough_osm_pct"] == pytest.approx(0.1)
    assert lf["rough_inferred_pct"] == pytest.approx(0.3)
    # combined rough = rough_osm + rough_inferred, originals preserved
    assert lf["rough_pct"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Green anchor
# ---------------------------------------------------------------------------


def test_green_y_from_green_points():
    p = mk([(0, 348, 0, "green", 1), (0, 350, 0, "green", 1), (0, 352, 0, "green", 1),
            (0, 10, 0, "fairway", 50)])
    assert green_y_value(p) == pytest.approx(350.0)


def test_green_y_fallback_when_no_green():
    p = mk([(0, float(i), 0, "fairway", 1) for i in range(101)])
    assert green_y_value(p) > 90.0  # ~98th percentile


# ---------------------------------------------------------------------------
# Zone splitting
# ---------------------------------------------------------------------------


def _zoned_hole():
    # green_y will be passed explicitly = 500 so zones are disjoint:
    #   tee 0-75 | drive 175-300 | approach 325-500 | green_complex 425-500
    return mk([
        (0, 30, 0, "fairway", 10), (0, 30, 0, "trees", 10),          # tee zone
        (0, 200, 0, "fairway", 10),                                   # drive
        (-5, 200, 0, "bunker", 5), (5, 200, 0, "water", 5),          # drive L/R
        (0, 400, 0, "fairway", 10),                                   # approach only
        (-5, 460, 0, "bunker", 5), (5, 460, 0, "water", 5),          # green complex
    ])


def test_zone_splitting():
    p = _zoned_hole()
    zf = zone_features(p, green_y=500.0)
    assert zf["tee_zone_fairway_pct"] == pytest.approx(0.5)
    assert zf["tee_zone_trees_pct"] == pytest.approx(0.5)
    assert zf["drive_zone_fairway_pct"] == pytest.approx(0.5)
    assert zf["drive_zone_bunker_pct"] == pytest.approx(0.25)
    assert zf["drive_zone_water_pct"] == pytest.approx(0.25)
    # approach (325..500) = 10 fairway + 5 bunker + 5 water = 20
    assert zf["approach_zone_fairway_pct"] == pytest.approx(0.5)
    assert zf["green_complex_bunker_pct"] == pytest.approx(0.5)
    assert zf["green_complex_water_pct"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Left/right pressure
# ---------------------------------------------------------------------------


def test_left_right_pressure():
    p = _zoned_hole()
    lr = left_right_features(p, green_y=500.0)
    # drive zone: bunker on left, water on right (5 each of 20)
    assert lr["drive_bunker_left_pct"] == pytest.approx(0.25)
    assert lr["drive_bunker_right_pct"] == pytest.approx(0.0)
    assert lr["drive_water_right_pct"] == pytest.approx(0.25)
    assert lr["drive_water_left_pct"] == pytest.approx(0.0)
    # approach zone: green-complex bunker left, water right (5 each of 20)
    assert lr["approach_bunker_left_pct"] == pytest.approx(0.25)
    assert lr["approach_water_right_pct"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Strategic + full assembly
# ---------------------------------------------------------------------------


def test_strategic_features_keys():
    p = _zoned_hole()
    sf = strategic_features(p, green_y=500.0)
    for k in ("dogleg_score", "fairway_centerline_shift",
              "fairway_width_drive_zone", "fairway_width_approach_zone",
              "green_complex_bunker_pct", "green_complex_water_pct",
              "green_complex_trees_pct"):
        assert k in sf


def test_build_one_hole_row():
    p = _zoned_hole()
    ids = {"course_slug": "x", "hole_number": 1, "hole_id": "x__01"}
    terrain = {"net_elevation_change_m": -4.2}
    row = build_one_hole_features(p, ids, terrain)
    assert row["hole_id"] == "x__01"
    assert row["point_count"] == p.n
    assert row["tee_to_green_elevation_change"] == pytest.approx(-4.2)
    # spans all feature families
    for k in ("hole_width_m", "z_mean", "fairway_pct",
              "drive_zone_fairway_pct", "drive_bunker_left_pct", "dogleg_score"):
        assert k in row


# ---------------------------------------------------------------------------
# Similarity / clustering / nearest neighbors
# ---------------------------------------------------------------------------


def _feature_df(n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "hole_id": [f"c__{i:02d}" for i in range(n)],
        "course_slug": ["c"] * n,
        "course_name": ["C"] * n,
        "hole_number": list(range(1, n + 1)),
        "par": [4] * n,
        "hole_length_m": rng.uniform(300, 450, n),
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
        "feat_c": [np.nan] + list(rng.normal(size=n - 1)),  # exercise imputation
    })


def test_feature_columns_excludes_identifiers():
    df = _feature_df()
    cols = feature_columns(df)
    assert "hole_id" not in cols and "course_slug" not in cols and "hole_number" not in cols
    assert {"feat_a", "feat_b", "feat_c", "par", "hole_length_m"}.issubset(cols)


def test_matrix_shape_and_no_nans():
    df = _feature_df()
    cols = feature_columns(df)
    X, imp, scaler = build_feature_matrix(df, cols)
    assert X.shape == (len(df), len(cols))
    assert np.isfinite(X).all()  # imputation removed the NaN


def test_nearest_neighbor_table_shape_and_no_self():
    df = _feature_df(8)
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    k = 3
    table = nearest_neighbor_table(df, X, k=k)
    assert len(table) == len(df) * k
    assert set(table.columns) == {
        "query_hole_id", "query_course_slug", "query_hole_number",
        "similar_hole_id", "similar_course_slug", "similar_hole_number",
        "distance", "rank",
    }
    assert (table["query_hole_id"] != table["similar_hole_id"]).all()
    assert sorted(table[table["query_hole_id"] == "c__00"]["rank"]) == [1, 2, 3]


def test_similar_holes_and_kmeans():
    df = _feature_df(10)
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    labels = cluster_kmeans(X, k=3)
    assert len(labels) == len(df)
    top = similar_holes(df, X, "c__00", k=5)
    assert len(top) == 5
    assert (top["query_hole_id"] == "c__00").all()


def _multi_course_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "hole_id": [f"{c}__{i:02d}" for c in ("a", "b") for i in range(1, 5)],
        "course_slug": ["a"] * 4 + ["b"] * 4,
        "course_name": ["A"] * 4 + ["B"] * 4,
        "hole_number": [1, 2, 3, 4, 1, 2, 3, 4],
        "par": [4] * 8,
        "feat_a": rng.normal(size=8),
        "feat_b": rng.normal(size=8),
    })


def test_nearest_neighbor_exclude_same_course():
    df = _multi_course_df()
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    table = nearest_neighbor_table(df, X, k=3, exclude_same_course=True)
    assert (table["query_course_slug"] != table["similar_course_slug"]).all()
    # default (no exclusion) can include same-course neighbors
    plain = nearest_neighbor_table(df, X, k=3)
    assert len(plain) == len(df) * 3


def test_similar_holes_exclude_same_course():
    df = _multi_course_df()
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    top = similar_holes(df, X, "a__01", k=4, exclude_same_course=True)
    assert (top["similar_course_slug"] != "a").all()
    assert len(top) <= 4


def test_feature_summary_reports_missing():
    df = _feature_df()  # feat_c has one NaN
    fs = feature_summary(df)
    assert {"column", "dtype", "n_missing", "pct_missing"}.issubset(fs.columns)
    assert int(fs.loc[fs["column"] == "feat_c", "n_missing"].iloc[0]) == 1
    # identifiers are never reported as features
    assert "hole_id" not in set(fs["column"])


def _mixed_par_df():
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "hole_id": [f"a__{i:02d}" for i in range(1, 5)] + [f"b__{i:02d}" for i in range(1, 5)],
        "course_slug": ["a"] * 4 + ["b"] * 4,
        "course_name": ["A"] * 4 + ["B"] * 4,
        "hole_number": [1, 2, 3, 4, 1, 2, 3, 4],
        "par": [3, 4, 4, 5, 3, 4, 5, 4],
        "feat_a": rng.normal(size=8),
        "feat_b": rng.normal(size=8),
    })


def test_similar_holes_same_par():
    df = _mixed_par_df()
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    par_of = df.set_index("hole_id")["par"]
    # a__01 is par 3; only par-3 holes may be returned (here: b__01)
    top = similar_holes(df, X, "a__01", k=5, same_par=True)
    assert (top["similar_hole_id"].map(par_of) == 3).all()
    assert "a__01" not in set(top["similar_hole_id"])


def test_same_par_and_exclude_same_course_together():
    df = _mixed_par_df()
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    par_of = df.set_index("hole_id")["par"]
    # a__02 is par 4; cross-course + same-par => only par-4 holes on course b
    top = similar_holes(df, X, "a__02", k=5, same_par=True, exclude_same_course=True)
    assert (top["similar_course_slug"] == "b").all()
    assert (top["similar_hole_id"].map(par_of) == 4).all()


def test_nearest_neighbor_table_same_par():
    df = _mixed_par_df()
    cols = feature_columns(df)
    X, *_ = build_feature_matrix(df, cols)
    par_of = df.set_index("hole_id")["par"]
    t = nearest_neighbor_table(df, X, k=3, same_par=True)
    q_par = t["query_hole_id"].map(par_of)
    s_par = t["similar_hole_id"].map(par_of)
    assert (q_par.to_numpy() == s_par.to_numpy()).all()
