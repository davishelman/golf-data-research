"""Tests for the v2.5 -> similar-hole-set loader (issue #32).

Fake CSVs written to temp dirs only — no dependence on real ``courses/`` outputs,
no network, no v2.5 scorer. Covers both supported layouts, filtering, weighting,
determinism, tolerance to missing optional columns, and error paths.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    WEIGHT_METHODS,
    SimilarHoleLoaderError,
    add_similarity_weights,
    available_target_hole_numbers,
    load_similar_hole_sets,
    missing_target_hole_numbers,
    parse_v25_hole_id,
)

RESULTS_FILE = "similarity_results.csv"


# --------------------------------------------------------------------------- #
# Fake data helpers
# --------------------------------------------------------------------------- #
def _results_frame(
    course: str = "augusta_national",
    n_holes: int = 18,
    n_candidates: int = 12,
    *,
    with_components: bool = True,
    candidate_course: str = "riviera",
) -> pd.DataFrame:
    """A fake v2.5 results frame: ``n_holes`` targets, ``n_candidates`` each."""
    rows = []
    for h in range(1, n_holes + 1):
        for c in range(1, n_candidates + 1):
            row = {
                "model_version": "v2_5_chamfer_v1",
                "config_name": "baseline",
                "config_hash": "deadbeef",
                "target_hole_id": f"{course}:{h}",
                "candidate_hole_id": f"{candidate_course}:{c}",
                "rank": c,
                "total_score": float(c),  # ascending -> candidate c is c-th best
                "filter_reason": "PASS",
            }
            if with_components:
                row.update({
                    "fairway_score": 1.0 * c, "green_score": 0.5 * c,
                    "bunker_score": 0.1 * c, "water_score": 0.0, "tee_score": 0.2 * c,
                    "yardage_penalty": 0.3, "elevation_penalty": 0.1,
                    "missing_surface_penalty": 0.0,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _write_results(
    root: Path, df: pd.DataFrame, *, config: str = "baseline", layout: str = "local"
) -> Path:
    """Write ``df`` as a v2.5 results CSV in the given layout; return ``root``."""
    base = root / "data" / "pointcloud_similarity" if layout == "bundle" else root / "pointcloud_similarity"
    cfg_dir = base / config
    cfg_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg_dir / RESULTS_FILE, index=False)
    return root


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #
def test_load_from_local_index_layout(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=2, n_candidates=5), layout="local")
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=3)
    assert len(out) == 2 * 3
    assert set(out["target_course_slug"]) == {"augusta_national"}


def test_load_from_artifact_bundle_layout(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=2, n_candidates=5), layout="bundle")
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=3)
    assert len(out) == 2 * 3


# --------------------------------------------------------------------------- #
# Filtering, sizing, coverage
# --------------------------------------------------------------------------- #
def test_filters_to_target_course(tmp_path):
    combined = pd.concat([
        _results_frame("augusta_national", n_holes=2, n_candidates=4),
        _results_frame("pebble_beach", n_holes=2, n_candidates=4),
    ], ignore_index=True)
    _write_results(tmp_path, combined)
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=4)
    assert set(out["target_course_slug"]) == {"augusta_national"}
    assert not out["target_hole_id"].str.startswith("pebble_beach").any()


def test_returns_18_times_top_n_rows(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=18, n_candidates=12))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=10)
    assert len(out) == 18 * 10
    assert available_target_hole_numbers(out) == list(range(1, 19))
    assert missing_target_hole_numbers(out) == []


def test_partial_coverage_does_not_crash(tmp_path):
    # Only holes 1..3 present -> return them; report the rest as missing.
    _write_results(tmp_path, _results_frame(n_holes=3, n_candidates=6))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=5)
    assert len(out) == 3 * 5
    assert available_target_hole_numbers(out) == [1, 2, 3]
    assert missing_target_hole_numbers(out) == list(range(4, 19))


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_no_rows_for_requested_course_raises(tmp_path):
    _write_results(tmp_path, _results_frame("augusta_national", n_holes=2))
    with pytest.raises(SimilarHoleLoaderError) as exc:
        load_similar_hole_sets(tmp_path, "pebble_beach")
    assert "pebble_beach" in str(exc.value)


def test_missing_results_dir_raises(tmp_path):
    with pytest.raises(SimilarHoleLoaderError) as exc:
        load_similar_hole_sets(tmp_path, "augusta_national")
    assert "no v2.5 point-cloud similarity results" in str(exc.value)


def test_missing_config_file_raises(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=1), config="baseline")
    with pytest.raises(SimilarHoleLoaderError) as exc:
        load_similar_hole_sets(tmp_path, "augusta_national", config_name="hazard_heavy")
    assert "hazard_heavy" in str(exc.value)


def test_missing_required_column_raises(tmp_path):
    df = _results_frame(n_holes=1, n_candidates=3).drop(columns=["total_score"])
    _write_results(tmp_path, df)
    with pytest.raises(SimilarHoleLoaderError) as exc:
        load_similar_hole_sets(tmp_path, "augusta_national")
    assert "total_score" in str(exc.value)


def test_non_positive_top_n_raises(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=1))
    with pytest.raises(SimilarHoleLoaderError):
        load_similar_hole_sets(tmp_path, "augusta_national", top_n=0)


def test_unknown_weight_method_raises(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=1))
    with pytest.raises(SimilarHoleLoaderError) as exc:
        load_similar_hole_sets(tmp_path, "augusta_national", weight_method="magic")
    assert "magic" in str(exc.value)


def test_malformed_candidate_hole_id_raises(tmp_path):
    df = _results_frame(n_holes=1, n_candidates=3)
    df.loc[0, "candidate_hole_id"] = "no_colon_here"
    _write_results(tmp_path, df)
    with pytest.raises(SimilarHoleLoaderError) as exc:
        load_similar_hole_sets(tmp_path, "augusta_national")
    assert "candidate_hole_id" in str(exc.value)


# --------------------------------------------------------------------------- #
# parse_v25_hole_id
# --------------------------------------------------------------------------- #
def test_parse_v25_hole_id_ok():
    assert parse_v25_hole_id("augusta_national:13") == ("augusta_national", 13)


@pytest.mark.parametrize("bad", ["augusta_national", "augusta_national:", ":13", "a:b"])
def test_parse_v25_hole_id_bad_fails(bad):
    with pytest.raises(SimilarHoleLoaderError):
        parse_v25_hole_id(bad)


# --------------------------------------------------------------------------- #
# Weighting: each method normalizes per target hole
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", list(WEIGHT_METHODS))
def test_weights_sum_to_one_per_target(tmp_path, method):
    _write_results(tmp_path, _results_frame(n_holes=4, n_candidates=6))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=5, weight_method=method)
    sums = out.groupby("target_hole_id")["similarity_weight"].sum()
    assert (sums - 1.0).abs().max() < 1e-9
    assert set(out["weight_method"]) == {method}
    # All four methods are exercised across the parametrization.
    assert method in ("rank_decay", "inverse_score", "softmax_score", "uniform")


def test_uniform_weights_are_equal(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=1, n_candidates=5))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=5, weight_method="uniform")
    assert out["similarity_weight"].round(9).nunique() == 1
    assert out["similarity_weight"].iloc[0] == pytest.approx(1 / 5)


def test_rank_decay_is_monotonic_decreasing(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=1, n_candidates=5))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=5, weight_method="rank_decay")
    w = out.sort_values("rank")["similarity_weight"].tolist()
    assert all(a > b for a, b in zip(w, w[1:]))  # rank 1 heaviest


def test_add_similarity_weights_does_not_mutate_input():
    df = pd.DataFrame({
        "target_hole_id": ["c:1", "c:1", "c:1"],
        "rank": [1, 2, 3],
        "total_score": [1.0, 2.0, 3.0],
    })
    before = df.copy()
    add_similarity_weights(df, method="rank_decay")
    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------- #
# Determinism + component columns
# --------------------------------------------------------------------------- #
def test_output_sort_is_deterministic(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=3, n_candidates=6))
    a = load_similar_hole_sets(tmp_path, "augusta_national", top_n=4)
    b = load_similar_hole_sets(tmp_path, "augusta_national", top_n=4)
    pd.testing.assert_frame_equal(a, b)
    # Rows are ordered exactly by (target_hole_number, rank, candidate_hole_id).
    expected = a.sort_values(
        ["target_hole_number", "rank", "candidate_hole_id"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, expected)


def test_component_columns_preserved_when_present(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=1, n_candidates=4, with_components=True))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=3)
    for col in ("fairway_score", "green_score", "bunker_score", "water_score",
                "tee_score", "yardage_penalty", "elevation_penalty",
                "missing_surface_penalty"):
        assert col in out.columns


def test_missing_component_columns_do_not_break(tmp_path):
    _write_results(tmp_path, _results_frame(n_holes=2, n_candidates=4, with_components=False))
    out = load_similar_hole_sets(tmp_path, "augusta_national", top_n=3)
    assert len(out) == 2 * 3
    assert "fairway_score" not in out.columns
    assert "similarity_weight" in out.columns  # weighting still works
