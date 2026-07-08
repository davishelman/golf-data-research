"""Tests for baseline lift & ablation analysis (issue #60).

Deterministic synthetic comparison / sweep tables — verifies wins are strict,
NaNs are handled, and the ablation summary picks the influential knob.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.modeling.player_course_advantage import (
    ablation_effects,
    baseline_lift_summary,
    baseline_lift_table,
    rank_ablation,
)

MODEL = "similar_hole_model"


def _comparison(model, null=0.1, recent=0.5, course=None):
    return pd.DataFrame({
        "ranker": [MODEL, "null", "recent_form", "course_history"],
        "spearman_pooled": [model, null, recent, course],
        "coverage": [1.0, 1.0, 1.0, 0.0],
        "n_pairs": [10, 10, 10, 0],
    })


# --------------------------------------------------------------------------- #
# Baseline lift
# --------------------------------------------------------------------------- #
def test_model_beats_all_baselines():
    comp = _comparison(model=0.9, null=0.1, recent=0.5, course=0.3)
    tbl = baseline_lift_table(comp)
    assert (tbl["model_wins"]).all()
    assert tbl.loc[tbl["baseline"] == "null", "lift"].iloc[0] == 0.8

    s = baseline_lift_summary(comp)
    assert s["wins_vs_baselines_count"] == 3
    assert s["beats_all_baselines"] is True
    assert s["model_rank"] == 1


def test_model_loses_to_a_baseline():
    comp = _comparison(model=0.3, null=0.1, recent=0.5, course=0.2)
    s = baseline_lift_summary(comp)
    assert s["beats_all_baselines"] is False
    assert s["wins_vs_baselines_count"] == 2  # beats null & course, not recent_form
    assert s["model_rank"] == 2               # recent_form ranks above the model


def test_ties_are_not_wins():
    comp = _comparison(model=0.5, null=0.1, recent=0.5, course=0.2)
    tbl = baseline_lift_table(comp)
    assert not tbl.loc[tbl["baseline"] == "recent_form", "model_wins"].iloc[0]
    assert baseline_lift_summary(comp)["beats_all_baselines"] is False


def test_nan_baseline_excluded_from_comparable():
    # course_history is NaN (low coverage) -> not comparable, not a win/loss.
    comp = _comparison(model=0.9, null=0.1, recent=0.5, course=np.nan)
    s = baseline_lift_summary(comp)
    assert s["n_baselines_comparable"] == 2   # null + recent_form
    assert s["wins_vs_baselines_count"] == 2
    assert s["beats_all_baselines"] is True    # beats all *comparable* baselines


def test_nan_model_metric_is_safe():
    comp = _comparison(model=np.nan, null=0.1, recent=0.5, course=0.2)
    s = baseline_lift_summary(comp)
    assert s["wins_vs_baselines_count"] == 0
    assert s["beats_all_baselines"] is False


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
def _sweep_table():
    return pd.DataFrame({
        "recency_decay": [0.7, 0.7, 1.0, 1.0],
        "lookback_years": [3, 5, 3, 5],
        "spearman_pooled": [0.4, 0.6, 0.5, 0.9],
        "coverage": [1.0, 1.0, 1.0, 1.0],
        "n_pairs": [10, 10, 10, 10],
    })


def test_rank_ablation_orders_by_metric():
    ranked = rank_ablation(_sweep_table())
    assert ranked["spearman_pooled"].tolist() == [0.9, 0.6, 0.5, 0.4]
    assert ranked.iloc[0]["ablation_rank"] == 1


def test_ablation_effects_picks_influential_knob():
    eff = ablation_effects(_sweep_table(),
                           param_cols=["recency_decay", "lookback_years"])
    lb = eff[eff["param"] == "lookback_years"].iloc[0]
    rd = eff[eff["param"] == "recency_decay"].iloc[0]
    # lookback spread (0.75-0.45=0.30) > recency spread (0.70-0.50=0.20).
    assert lb["spread"] > rd["spread"]
    assert lb["best_value"] == 5


def test_ablation_effects_skips_unvaried_knobs():
    tbl = _sweep_table()
    tbl["aggregate"] = "sum"  # constant -> should not appear
    eff = ablation_effects(tbl, param_cols=["aggregate", "lookback_years"])
    assert "aggregate" not in set(eff["param"])
