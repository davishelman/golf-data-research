"""Offline tests for the dual-source artifact loader.

Synthetic temp dirs only -- no real ``courses/`` data and no network. Exercises
both recognized layouts (HF artifact root vs local index folder), auto-detect,
clear errors when files are missing, and optional-metadata handling.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.modeling.artifact_loader import (
    _classify_root,
    load_modeling_artifacts,
    resolve_artifact_root,
)


def _features_df() -> pd.DataFrame:
    return pd.DataFrame({
        "hole_id": ["c__01", "c__02"],
        "course_slug": ["c", "c"],
        "course_name": ["C", "C"],
        "hole_number": [1, 2],
        "par": [4, 3],
        "hole_length_m": [400.0, 160.0],
        "fairway_pct": [0.4, 0.3],
    })


def _make_hf(root: Path) -> Path:
    """Build a minimal Hugging Face artifact layout: tables under data/."""
    (root / "data").mkdir(parents=True)
    df = _features_df()
    df.to_parquet(root / "data" / "hole_features.parquet")
    (df.assign(kmeans_cluster=0, agg_cluster=0, pca_1=0.0, pca_2=0.0)
       [["hole_id", "kmeans_cluster", "agg_cluster", "pca_1", "pca_2"]]
       .to_parquet(root / "data" / "hole_clusters.parquet"))
    pd.DataFrame({
        "query_hole_id": ["c__01"], "similar_hole_id": ["c__02"],
        "rank": [1], "distance": [0.5],
    }).to_csv(root / "data" / "hole_similarity_v2.csv", index=False)
    return root


def _make_local(root: Path) -> Path:
    """Build a minimal local index layout: tables in the root itself."""
    root.mkdir(parents=True)
    _features_df().to_parquet(root / "hole_features.parquet")
    return root


# --------------------------------------------------------------------------- #
# Classification + resolution
# --------------------------------------------------------------------------- #

def test_detects_hf_artifact_root(tmp_path):
    root = _make_hf(tmp_path / "artifact")
    assert _classify_root(root) == "hf_artifact"
    assert resolve_artifact_root(root) == root


def test_detects_local_index_root(tmp_path):
    root = _make_local(tmp_path / "_index")
    assert _classify_root(root) == "local_index"
    assert resolve_artifact_root(root) == root


def test_missing_root_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_artifact_root(tmp_path / "does_not_exist")
    assert "hole_features.parquet" in str(exc.value)


def test_autodetect_with_nothing_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_artifact_root(None)


def test_autodetect_prefers_local_index(tmp_path, monkeypatch):
    _make_local(tmp_path / "courses" / "_index")
    monkeypatch.chdir(tmp_path)
    art = load_modeling_artifacts(None)
    assert art["source_kind"] == "local_index"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def test_load_hf_artifact_tables_and_metadata(tmp_path):
    root = _make_hf(tmp_path / "artifact")
    (root / "metadata").mkdir()
    (root / "metadata" / "schema.json").write_text(json.dumps({"v": 1}), encoding="utf-8")
    (root / "dataset_manifest.json").write_text(json.dumps({"tier": "lite"}), encoding="utf-8")

    art = load_modeling_artifacts(root)
    assert art["source_kind"] == "hf_artifact"
    assert len(art["features"]) == 2
    assert art["clusters"] is not None
    assert art["similarity_v2"] is not None and len(art["similarity_v2"]) == 1
    assert art["similarity_v1"] is None              # not written -> skipped
    assert art["schema"] == {"v": 1}                 # optional metadata loaded
    assert art["manifest"] == {"tier": "lite"}
    assert art["feature_dictionary"] is None         # optional metadata absent -> None
    assert art["compact_dir"] is None                # no point_clouds dir present
    assert "Hugging Face" in art["label"]


def test_load_local_index(tmp_path):
    root = _make_local(tmp_path / "_index")
    art = load_modeling_artifacts(root)
    assert art["source_kind"] == "local_index"
    assert len(art["features"]) == 2
    assert art["similarity_v2"] is None              # absent in this minimal layout
    assert art["courses_root"] == tmp_path           # parent of _index
    assert "local" in art["label"]


def test_hf_compact_dir_detected(tmp_path):
    root = _make_hf(tmp_path / "artifact")
    compact = root / "point_clouds" / "compact"
    compact.mkdir(parents=True)
    (compact / "c__01.json").write_text(
        json.dumps({"points": [], "label_map": {}}), encoding="utf-8")
    art = load_modeling_artifacts(root)
    assert art["compact_dir"] == compact
