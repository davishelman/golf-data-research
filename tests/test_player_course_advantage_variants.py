"""Tests for shrinkage / baseline-ensemble variants (issue #74)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.modeling.player_course_advantage.variants import (
    best_variant_by_validation,
    blend,
    compare_variants,
    default_variants,
    render_variant_summary,
    select_blend_alpha,
    shrink_by_count,
    shrink_toward_zero,
)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_shrink_toward_zero():
    s = pd.Series([2.0, -4.0])
    assert list(shrink_toward_zero(s, 0.5)) == [1.0, -2.0]


def test_shrink_by_count_reduces_low_coverage_extremes():
    adv = pd.Series([3.0, 3.0])          # same raw advantage
    counts = pd.Series([1.0, 100.0])     # thin vs deep coverage
    out = shrink_by_count(adv, counts, k=5.0)
    assert abs(out.iloc[0]) < abs(out.iloc[1])          # low count shrunk more
    assert out.iloc[1] == pytest.approx(3.0 * 100 / 105)  # high count barely touched
    assert out.iloc[1] > 2.8


def test_blend():
    a = pd.Series([2.0]); b = pd.Series([0.0])
    assert blend(a, b, 0.25).iloc[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Comparison + selection
# --------------------------------------------------------------------------- #
def _predictions():
    # Validation (2023): advantage perfectly ranks performance -> similar_only best.
    # Test (2024): different pattern (used only for reporting).
    val = pd.DataFrame({
        "course_advantage": [1.0, 2.0, 3.0, 4.0],
        "performance": [1.0, 2.0, 3.0, 4.0],
        "recent_form_advantage": [4.0, 3.0, 2.0, 1.0],  # anti-correlated noise
        "holes_covered": [9, 9, 9, 9],
        "total_raw_occurrences": [20, 20, 20, 20],
        "total_weighted_occurrences": [20.0, 20.0, 20.0, 20.0],
        "covered": [True] * 4,
        "predict_season": [2023] * 4,
    })
    test = val.copy(); test["predict_season"] = 2024
    return pd.concat([val, test], ignore_index=True)


def test_compare_variants_plain_model_wins():
    preds = _predictions()
    variants = default_variants(baseline_cols=["recent_form_advantage"])
    comp = compare_variants(preds, variants, validation_seasons=[2023], test_seasons=[2024])
    assert "validation_metric" in comp.columns and "test_metric" in comp.columns
    assert best_variant_by_validation(comp) == "similar_only"
    # blending in the anti-correlated baseline does not beat the plain model
    # (its metric is lower or degenerate/NaN).
    blend_row = comp[comp["variant"] == "blend_recent_form_advantage"].iloc[0]
    sim_row = comp[comp["variant"] == "similar_only"].iloc[0]
    assert pd.isna(blend_row["validation_metric"]) or \
        sim_row["validation_metric"] >= blend_row["validation_metric"]


def test_render_says_keep_simple_model_when_variants_lose():
    comp = compare_variants(_predictions(),
                            default_variants(baseline_cols=["recent_form_advantage"]),
                            validation_seasons=[2023], test_seasons=[2024])
    md = render_variant_summary(comp, data_source="synthetic")
    assert "plain model" in md.lower() and "keep the simple model" in md.lower()


def test_variant_can_win_when_it_helps():
    # A blend that de-noises: model is imperfect, baseline matches performance,
    # so the 0.5 blend ranks better than the model alone.
    val = pd.DataFrame({
        "course_advantage": [2.0, 1.0, 3.0, 4.0],       # imperfect (spearman ~0.8)
        "performance": [1.0, 2.0, 3.0, 4.0],
        "good_baseline": [1.0, 2.0, 3.0, 4.0],          # baseline matches performance
        "holes_covered": [9, 9, 9, 9],
        "total_raw_occurrences": [20] * 4,
        "total_weighted_occurrences": [20.0] * 4,
        "covered": [True] * 4,
        "predict_season": [2023] * 4,
    })
    comp = compare_variants(val, default_variants(baseline_cols=["good_baseline"]),
                            validation_seasons=[2023])
    assert best_variant_by_validation(comp) == "blend_good_baseline"


# --------------------------------------------------------------------------- #
# Validation-only blend selection
# --------------------------------------------------------------------------- #
def test_select_blend_alpha_uses_validation_only():
    # Validation favours pure model (alpha=1); test favours baseline (alpha=0).
    val = pd.DataFrame({
        "course_advantage": [1, 2, 3, 4], "performance": [1, 2, 3, 4],
        "base": [4, 3, 2, 1], "covered": [True] * 4, "predict_season": [2023] * 4,
    })
    test = pd.DataFrame({
        "course_advantage": [1, 2, 3, 4], "performance": [4, 3, 2, 1],
        "base": [4, 3, 2, 1], "covered": [True] * 4, "predict_season": [2024] * 4,
    })
    preds = pd.concat([val, test], ignore_index=True)
    res = select_blend_alpha(preds, "base", validation_seasons=[2023],
                             alphas=(0.0, 0.5, 1.0))
    assert res["best_alpha"] == 1.0   # chosen on validation, ignoring test
