"""Tests for the parameter sweep runner (issue #36).

Fake data only. Covers grid enumeration, provider caching, the metrics table,
coverage/pairs-guarded recommendation, the train/validation split + in-sample
warning, export, and no-mutation.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage import (
    AdvantageParams,
    SweepGrid,
    export_sweep,
    iter_param_sets,
    run_sweep,
)

TARGET_COURSE = "augusta_national"


# --------------------------------------------------------------------------- #
# Fake data
# --------------------------------------------------------------------------- #
def make_sim(n_holes=3):
    rows = [{
        "target_course_slug": TARGET_COURSE, "target_hole_number": h,
        "target_hole_id": f"{TARGET_COURSE}:{h}",
        "candidate_course_slug": "other", "candidate_hole_number": h,
        "candidate_hole_id": f"other:{h}",
        "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
        "weight_method": "manual", "config_name": "baseline",
    } for h in range(1, n_holes + 1)]
    return pd.DataFrame(rows)


def make_history(player_outcomes, years=(2022, 2023), n_holes=3):
    rows = []
    tid = 0
    for pid, outcome in player_outcomes.items():
        for yr in years:
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "tournament_id": f"{yr}-T{tid}", "year": yr,
                    "round": 1, "hole_number": h, "course_slug": "other",
                    "hole_id_v25": f"other:{h}", "par": 4,
                    "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    return pd.DataFrame(rows)


def _results(seasons):
    """One event per season; finish order pA<pB<pC (pA best)."""
    frames = []
    for season in seasons:
        frames.append(pd.DataFrame({
            "event_id": [f"E{season}"] * 3, "predict_season": [season] * 3,
            "target_course_slug": [TARGET_COURSE] * 3,
            "player_id": ["pA", "pB", "pC"], "finish_rank": [1, 2, 3],
        }))
    return pd.concat(frames, ignore_index=True)


def _provider_factory():
    calls = {"n": 0}
    sim = make_sim(3)

    def provider(config_name, top_n, weight_method):
        calls["n"] += 1
        return sim

    return provider, calls


# --------------------------------------------------------------------------- #
# Grid enumeration
# --------------------------------------------------------------------------- #
def test_grid_cartesian_product():
    grid = SweepGrid(recency_decay=(0.7, 0.85), min_holes_covered=(1, 12))
    combos = list(iter_param_sets(grid))
    assert len(combos) == 4 == len(grid)
    assert {c["recency_decay"] for c in combos} == {0.7, 0.85}


# --------------------------------------------------------------------------- #
# Table + provider caching
# --------------------------------------------------------------------------- #
def test_run_sweep_table_and_provider_cache():
    provider, calls = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    # 2 param sets that share one (config, top_n, weight) triple -> 1 provider call.
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(1, 99))
    res = run_sweep(hist, results, provider, grid=grid,
                    outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    assert len(res.table) == 2
    for col in ("top_n", "recency_decay", "min_holes_covered",
                "spearman_pooled", "coverage", "n_pairs"):
        assert col in res.table.columns
    assert calls["n"] == 1  # memoized


# --------------------------------------------------------------------------- #
# Recommendation with coverage/pairs guards
# --------------------------------------------------------------------------- #
def test_recommend_skips_low_coverage_rows():
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    # min_holes_covered=99 withholds everyone (coverage 0); =1 keeps them.
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(1, 99))
    res = run_sweep(hist, results, provider, grid=grid,
                    outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    best = res.recommend(min_coverage=0.5, min_pairs=1)
    assert best is not None
    assert int(best["min_holes_covered"]) == 1
    assert res.recommended_params(min_coverage=0.5).min_holes_covered == 1


def test_recommend_returns_none_when_nothing_eligible():
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(99,))
    res = run_sweep(hist, results, provider, grid=grid,
                    outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    assert res.recommend(min_coverage=0.5) is None
    assert res.recommended_params(min_coverage=0.5) is None


# --------------------------------------------------------------------------- #
# Train / validation split + in-sample warning
# --------------------------------------------------------------------------- #
def test_validation_split_adds_val_columns_and_selects_on_it():
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1}, years=(2022, 2023))
    results = _results([2023, 2024])
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(1,))
    res = run_sweep(
        hist, results, provider, grid=grid, outcome_col="finish_rank",
        higher_is_better=False, top_k=(1,), validation_seasons=[2024],
    )
    assert res.has_validation
    assert "val_spearman_pooled" in res.table.columns
    # recommend prefers the validation metric when present.
    best = res.recommend(prefer_validation=True, min_coverage=0.0)
    assert best is not None
    # No in-sample warning when a split is provided.
    assert not any("in-sample" in w for w in res.warnings)


def test_in_sample_warning_without_split():
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(1,))
    res = run_sweep(hist, results, provider, grid=grid,
                    outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    assert not res.has_validation
    assert any("in-sample" in w for w in res.warnings)


# --------------------------------------------------------------------------- #
# Determinism, export, no mutation
# --------------------------------------------------------------------------- #
def test_run_sweep_deterministic():
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    grid = SweepGrid(min_occurrences_per_hole=(1,), recency_decay=(0.7, 0.85))
    a = run_sweep(hist, results, _provider_factory()[0], grid=grid,
                  outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    b = run_sweep(hist, results, _provider_factory()[0], grid=grid,
                  outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    pd.testing.assert_frame_equal(a.table, b.table)


def test_export_sweep(tmp_path):
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(1,))
    res = run_sweep(hist, results, provider, grid=grid,
                    outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    paths = export_sweep(res, tmp_path / "sweep")
    assert paths["table"].exists() and paths["manifest"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["n_param_sets"] == len(res.table)
    assert "warnings" in manifest


def test_run_sweep_does_not_mutate_inputs():
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    hist_b, results_b = hist.copy(), results.copy()
    grid = SweepGrid(min_occurrences_per_hole=(1,))
    run_sweep(hist, results, provider, grid=grid,
              outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    pd.testing.assert_frame_equal(hist, hist_b)
    pd.testing.assert_frame_equal(results, results_b)


def test_to_markdown_report():
    provider, _ = _provider_factory()
    hist = make_history({"pA": 3, "pB": 2, "pC": 1})
    results = _results([2024])
    grid = SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=(1,))
    res = run_sweep(hist, results, provider, grid=grid,
                    outcome_col="finish_rank", higher_is_better=False, top_k=(1,))
    md = res.to_markdown()
    assert isinstance(md, str) and "sweep" in md.lower()
