"""Smoke tests for the Streamlit demo app's v2.5 integration (#27).

These exercise the app end-to-end with Streamlit's headless ``AppTest`` harness
*and* the v2.5 demo data layer against a synthetic local-index tree, so they run
without a downloaded artifact and without a browser.

The full-app render is skipped when streamlit isn't installed or no artifact is
discoverable in the environment (e.g. minimal CI), so the suite stays green
everywhere; the data-layer integration test always runs.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.modeling.pointcloud import demo as pcdemo
from pipeline.modeling.pointcloud.export_similarity import (
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    SIMILARITY_RESULTS_COLUMNS,
)


# --------------------------------------------------------------------------- #
# Data-layer integration (always runs; no streamlit, no artifact needed)
# --------------------------------------------------------------------------- #

def _row(target, candidate, rank, score, config):
    base = {c: None for c in SIMILARITY_RESULTS_COLUMNS}
    base.update({
        "model_version": "v2_5_chamfer_v1", "config_name": config, "config_hash": "h",
        "target_hole_id": target, "candidate_hole_id": candidate, "rank": rank,
        "total_score": score, "fairway_score": score * 0.5, "green_score": score * 0.3,
        "bunker_score": score * 0.2, "water_score": None, "tee_score": 0.5,
        "yardage_penalty": 0.1, "elevation_penalty": 0.0, "missing_surface_penalty": 0.0,
        "filter_reason": "PASS",
    })
    return base


def _local_index(tmp_path):
    """A courses/_index-style tree with two configs sharing a target hole."""
    target = "augusta_national:13"
    idx = tmp_path / "_index" / "pointcloud_similarity"
    for cfg, rows in {
        "baseline": [
            _row(target, "quail_hollow_club:15", 1, 30.8, "baseline"),
            _row(target, "doral:2", 2, 31.7, "baseline"),
            _row(target, "tpc_southwind:16", 3, 39.6, "baseline"),
        ],
        "hazard_heavy": [
            _row(target, "tpc_southwind:16", 1, 38.0, "hazard_heavy"),
            _row(target, "doral:2", 2, 39.3, "hazard_heavy"),
            _row(target, "quail_hollow_club:15", 3, 41.1, "hazard_heavy"),
        ],
    }.items():
        d = idx / cfg
        d.mkdir(parents=True)
        pd.DataFrame(rows, columns=list(SIMILARITY_RESULTS_COLUMNS)).to_csv(
            d / RESULTS_FILENAME, index=False)
        (d / MANIFEST_FILENAME).write_text(json.dumps({"config_name": cfg}), encoding="utf-8")
    return tmp_path / "_index", target


def test_demo_layer_resolves_local_index_and_compares_configs(tmp_path):
    index_root, target = _local_index(tmp_path)

    assert pcdemo.list_pointcloud_configs(index_root) == ["baseline", "hazard_heavy"]
    results = pcdemo.load_pointcloud_results(index_root)

    # Top matches for the hole render as display columns.
    top = pcdemo.top_matches_for_hole(results["baseline"], target, top_n=5)
    assert list(top.columns) == list(pcdemo.DISPLAY_COLUMNS)
    assert top.iloc[0]["candidate_hole_id"] == "quail_hollow_club:15"

    # Config comparison surfaces #1-per-preset, shared matches, and overlap.
    cmp = pcdemo.compare_configs_for_hole(results, target, top_n=5)
    assert cmp["best_per_config"]["baseline"][0] == "quail_hollow_club:15"
    assert cmp["best_per_config"]["hazard_heavy"][0] == "tpc_southwind:16"
    # All three candidates are shared across both presets.
    assert set(cmp["shared_candidates"]) == {"quail_hollow_club:15", "doral:2", "tpc_southwind:16"}
    assert not cmp["overlap"].empty
    assert not cmp["rank_comparison"].empty


def test_compare_configs_handles_missing_hole(tmp_path):
    index_root, _ = _local_index(tmp_path)
    results = pcdemo.load_pointcloud_results(index_root)
    cmp = pcdemo.compare_configs_for_hole(results, "no_such:1", top_n=5)
    assert all(v is None for v in cmp["best_per_config"].values())
    assert cmp["shared_candidates"] == []
    assert cmp["rank_comparison"].empty


# --------------------------------------------------------------------------- #
# Full-app render (skipped when streamlit / artifact unavailable)
# --------------------------------------------------------------------------- #

def test_app_renders_without_exception():
    pytest.importorskip("streamlit", reason="streamlit not installed")
    from pipeline.modeling import demo_utils as du

    if du.discover_artifact_root() is None:
        pytest.skip("no artifact discoverable in this environment")

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"

    headers = " ".join(s.value for s in at.subheader)
    assert "Selected hole" in headers
    # v2 section is always present; v2.5 section present whenever results exist.
    assert "v2" in headers
