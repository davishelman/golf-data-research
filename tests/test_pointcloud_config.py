"""Tests for v2.5 point-cloud config loading, validation, and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.modeling.pointcloud.config import (
    ConfigError,
    config_from_dict,
    load_config,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASELINE = _REPO_ROOT / "configs" / "similarity" / "pointcloud_chamfer_v1.yaml"

_ALL_CONFIGS = sorted((_REPO_ROOT / "configs" / "similarity").glob("pointcloud_chamfer_*.yaml"))


def _valid_payload() -> dict:
    """A minimal, valid config payload (mirrors the baseline YAML)."""
    return {
        "model_name": "pointcloud_chamfer",
        "model_version": "v2_5_chamfer_v1",
        "config_name": "unit_test",
        "candidate_filter": {
            "require_same_par": True,
            "yardage_windows": {
                "par_3": {"absolute_yards": 25, "percentage": 0.12},
                "par_4": {"absolute_yards": 40, "percentage": 0.10},
                "par_5": {"absolute_yards": 60, "percentage": 0.10},
            },
        },
        "required_surfaces": ["tee", "green", "fairway"],
        "surface_weights": {
            "fairway": 0.30, "green": 0.25, "bunker": 0.25, "water": 0.15, "tee": 0.05,
        },
        "surface_missing_penalties": {
            "fairway": 999.0, "green": 999.0, "tee": 999.0, "bunker": 40.0, "water": 50.0,
        },
        "point_budgets": {
            "fairway": 1500, "green": 600, "bunker": 800, "water": 800, "tee": 200,
        },
        "distance_scaling": {"x_weight": 1.0, "y_weight": 1.0, "z_weight": 2.0},
        "penalties": {
            "par_mismatch": 999.0, "yardage_weight": 0.15,
            "tee_to_green_elevation_weight": 0.10,
        },
    }


# --- 1. loads successfully ------------------------------------------------- #

def test_baseline_config_loads():
    config = load_config(_BASELINE)
    assert config.model_name == "pointcloud_chamfer"
    assert config.model_version == "v2_5_chamfer_v1"
    assert config.config_name == "baseline"
    assert config.weighted_surfaces() == ("fairway", "green", "bunker", "water", "tee")


@pytest.mark.parametrize("path", _ALL_CONFIGS, ids=lambda p: p.stem)
def test_all_shipped_configs_load_and_validate(path):
    config = load_config(path)
    assert config.config_name
    assert abs(sum(config.surface_weights.values()) - 1.0) < 1e-6


# --- 2. surface weights must sum to 1.0 ------------------------------------ #

def test_surface_weights_must_sum_to_one():
    payload = _valid_payload()
    payload["surface_weights"]["fairway"] = 0.50  # now sums to 1.20
    with pytest.raises(ConfigError, match="sum to 1.0"):
        config_from_dict(payload)


def test_unknown_weighted_surface_rejected():
    payload = _valid_payload()
    payload["surface_weights"] = {"fairway": 0.5, "putting_green": 0.5}
    with pytest.raises(ConfigError, match="unknown surface"):
        config_from_dict(payload)


def test_unknown_required_surface_rejected():
    payload = _valid_payload()
    payload["required_surfaces"] = ["tee", "driving_range"]
    with pytest.raises(ConfigError, match="unknown surface"):
        config_from_dict(payload)


def test_missing_yardage_bucket_rejected():
    payload = _valid_payload()
    del payload["candidate_filter"]["yardage_windows"]["par_5"]
    with pytest.raises(ConfigError, match="par_5"):
        config_from_dict(payload)


def test_missing_model_version_rejected():
    payload = _valid_payload()
    payload["model_version"] = ""
    with pytest.raises(ConfigError, match="model_version"):
        config_from_dict(payload)


def test_missing_config_name_rejected():
    payload = _valid_payload()
    payload["config_name"] = ""
    with pytest.raises(ConfigError, match="config_name"):
        config_from_dict(payload)


# --- 10. config hash is deterministic -------------------------------------- #

def test_config_hash_is_deterministic():
    a = config_from_dict(_valid_payload())
    b = config_from_dict(_valid_payload())
    assert a.config_hash == b.config_hash
    assert len(a.config_hash) == 64  # sha-256 hex digest


def test_config_hash_changes_with_weights():
    base = config_from_dict(_valid_payload())
    changed_payload = _valid_payload()
    changed_payload["surface_weights"] = {
        "fairway": 0.25, "green": 0.30, "bunker": 0.25, "water": 0.15, "tee": 0.05,
    }
    changed = config_from_dict(changed_payload)
    assert base.config_hash != changed.config_hash


def test_baseline_file_hash_is_stable_across_loads():
    assert load_config(_BASELINE).config_hash == load_config(_BASELINE).config_hash
