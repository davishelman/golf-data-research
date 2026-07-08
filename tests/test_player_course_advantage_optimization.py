"""Tests for the validation-split parameter optimizer (issue #73). Synthetic only."""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage.optimization import (
    OptimizationError,
    export_optimization,
    optimize_parameters,
)
from pipeline.modeling.player_course_advantage.sweep import SweepGrid

TARGET = "augusta_national"
PLAYERS = {"pA": 3.0, "pB": 2.0, "pC": 1.5, "pD": 1.0}


def _sim(n_holes=9):
    return pd.DataFrame([{
        "target_course_slug": TARGET, "target_hole_number": h,
        "target_hole_id": f"{TARGET}:{h}", "candidate_course_slug": "src",
        "candidate_hole_number": h, "candidate_hole_id": f"src:{h}",
        "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
    } for h in range(1, n_holes + 1)])


def _history(n_holes=9, years=range(2018, 2023)):
    rows, tid = [], 0
    for pid, outcome in PLAYERS.items():
        for yr in years:
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "tournament_id": f"{yr}-{tid}", "year": yr, "round": 1,
                    "hole_number": h, "course_slug": "src", "hole_id_v25": f"src:{h}",
                    "par": 4, "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    return pd.DataFrame(rows)


def _results(seasons=(2021, 2022, 2023)):
    order = sorted(PLAYERS, key=lambda p: PLAYERS[p], reverse=True)  # best first
    finish = {p: r for r, p in enumerate(order, start=1)}
    return pd.concat([
        pd.DataFrame({
            "event_id": [f"E{s}"] * len(PLAYERS), "predict_season": [s] * len(PLAYERS),
            "target_course_slug": [TARGET] * len(PLAYERS),
            "player_id": list(PLAYERS), "finish_rank": [finish[p] for p in PLAYERS],
        }) for s in seasons
    ], ignore_index=True)


def _provider():
    sim = _sim()
    return lambda c, n, w: sim


def _grid(min_holes=(9,)):
    return SweepGrid(min_occurrences_per_hole=(1,), min_holes_covered=min_holes,
                     recency_decay=(0.7, 1.0))


# --------------------------------------------------------------------------- #
# Split hygiene
# --------------------------------------------------------------------------- #
def test_overlapping_splits_raise():
    with pytest.raises(OptimizationError):
        optimize_parameters(_history(), _results(), _provider(), _grid(),
                            train_seasons=[2021, 2022], validation_seasons=[2022],
                            test_seasons=[2023])


def test_missing_validation_raises():
    with pytest.raises(OptimizationError):
        optimize_parameters(_history(), _results(), _provider(), _grid(),
                            train_seasons=[2021], validation_seasons=[])


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_recommends_on_validation_and_reports_test():
    res = optimize_parameters(
        _history(), _results(), _provider(), _grid(),
        train_seasons=[2021], validation_seasons=[2022], test_seasons=[2023],
        data_source="synthetic", allow_synthetic_recommendation=True,
    )
    assert res.recommended_params is not None
    # recommendation is the top validation-ranked row's params
    top = res.rankings.iloc[0]
    assert res.recommended_params["recency_decay"] == top["recency_decay"]
    # held-out test metrics reported separately
    assert res.test_metrics is not None and "spearman_mean" in res.test_metrics


def test_synthetic_recommendation_withheld_by_default():
    res = optimize_parameters(
        _history(), _results(), _provider(), _grid(),
        train_seasons=[2021], validation_seasons=[2022],
        data_source="synthetic",  # allow flag not set
    )
    assert res.recommended_params is None
    assert any("withheld" in w for w in res.warnings)


def test_low_coverage_configs_rank_below_valid():
    res = optimize_parameters(
        _history(), _results(), _provider(), _grid(min_holes=(9, 999)),
        train_seasons=[2021], validation_seasons=[2022],
        data_source="synthetic", allow_synthetic_recommendation=True,
    )
    # the 999-min-holes config is uncovered (NaN validation metric) -> ranked last
    valid_metric = "val_spearman_mean"
    last = res.rankings.iloc[-1]
    assert last["min_holes_covered"] == 999
    assert pd.isna(last[valid_metric])
    # recommended uses the covered config
    assert res.recommended_params["min_holes_covered"] == 9


def test_export_writes_files(tmp_path):
    res = optimize_parameters(
        _history(), _results(), _provider(), _grid(),
        train_seasons=[2021], validation_seasons=[2022], test_seasons=[2023],
        data_source="synthetic", allow_synthetic_recommendation=True,
    )
    paths = export_optimization(res, tmp_path / "opt")
    for key in ("optimization_summary", "parameter_rankings", "validation_metrics",
                "recommended_params", "report", "test_metrics"):
        assert paths[key].exists()
