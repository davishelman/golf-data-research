"""Offline tests for the Hugging Face artifact exporter.

All tests build a synthetic ``courses_root`` in a temp dir — no real ``courses/``
data, no network. Only pandas + pyarrow are needed (the exporter avoids the geo /
sklearn stack on purpose).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.modeling import hf_export
from pipeline.modeling.hf_export import (
    _is_secret,
    _safe_copy,
    _verify_no_secrets,
    build_feature_dictionary,
    build_hf_artifact,
    build_schema,
    curated_hole_ids,
)
from pipeline.paths import IndexPaths


# --------------------------------------------------------------------------- #
# Synthetic fixture builder
# --------------------------------------------------------------------------- #

# (hole_id, course_slug, hole_number, par, length_m, water_pct, z_range)
_HOLES = [
    ("course_a__01", "course_a", 1, 4, 410.0, 0.02, 5.0),
    ("course_a__02", "course_a", 2, 3, 160.0, 0.05, 3.0),
    ("course_a__03", "course_a", 3, 5, 520.0, 0.03, 8.0),
    ("course_b__01", "course_b", 1, 4, 395.0, 0.40, 4.0),   # water-heavy
    ("course_b__02", "course_b", 2, 4, 430.0, 0.01, 22.0),  # terrain-heavy
    ("course_b__03", "course_b", 3, 3, 175.0, 0.06, 2.0),
]


def _feature_df() -> pd.DataFrame:
    rows = []
    for hid, slug, num, par, length, water, zr in _HOLES:
        rows.append({
            "course_slug": slug, "hole_number": num, "hole_id": hid,
            "course_name": slug.replace("_", " ").title(), "par": par,
            "hole_length_m": length, "hole_length_yd": round(length * 1.09361, 1),
            "green_y_m": length, "point_count": 1000 + num, "valid_point_count": 1000 + num,
            "water_pct": water, "z_range": zr,
            # a couple of columns from each engineered family to exercise the classifier
            "fairway_pct": 0.4, "rough_pct": 0.5,
            "drive_zone_fairway_pct": 0.6, "drive_zone_mean_z": 1.2,
            "drive_bunker_left_pct": 0.0, "dogleg_score": 0.05,
            "tee_to_green_elevation_change": zr / 2,
        })
    return pd.DataFrame(rows)


def _write_compact(courses_root: Path, slug: str, num: int, hid: str) -> None:
    feat = (courses_root / slug / "holes" / f"hole_{num:02d}" / "features")
    feat.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0", "hole_id": hid, "course_slug": slug,
        "hole_number": num, "coordinate_system": "tee_relative_aligned_meters",
        "label_map": {"0": "unknown", "3": "fairway"},
        "points": [[0.0, 1.0, 0.5, 3], [0.1, 2.0, 0.6, 3]],
    }
    (feat / "hole_points_compact.json").write_text(json.dumps(payload), encoding="utf-8")
    # a tiny per-hole point parquet too, for full-tier --include-point-parquet
    pd.DataFrame({"hole_id": [hid, hid], "label_id": [3, 3],
                  "x_aligned_m": [0.0, 0.1]}).to_parquet(feat / "hole_points.parquet")


def make_courses(tmp_path: Path, *, optional: bool = True, compact: bool = True) -> Path:
    """Create a synthetic courses_root; return it."""
    courses_root = tmp_path / "courses"
    index = IndexPaths.for_root(courses_root)
    index.ensure()

    df = _feature_df()
    df.to_parquet(index.hole_features_parquet)  # required input

    if optional:
        df.to_csv(index.hole_features_csv, index=False)
        df[["hole_id", "course_slug", "hole_number", "par", "hole_length_m"]].to_parquet(
            index.all_holes_parquet)
        df[["hole_id", "course_slug", "hole_number"]].assign(
            kmeans_cluster=0, agg_cluster=0, pca_1=0.0, pca_2=0.0
        ).to_parquet(index.hole_clusters_parquet)
        # v2 similarity so curated selection can pull "neighbors"
        pd.DataFrame({
            "query_hole_id": ["course_a__01", "course_a__01"],
            "similar_hole_id": ["course_b__01", "course_a__03"],
            "rank": [1, 2], "distance": [0.5, 0.9],
            "query_course_slug": ["course_a", "course_a"],
            "similar_course_slug": ["course_b", "course_a"],
        }).to_csv(index.hole_similarity_v2_csv, index=False)
        df.assign(rank=1, distance=0.1, similar_hole_id="course_a__02").to_csv(
            index.hole_similarity_examples_csv, index=False)
        manifest = {"schema_version": "1.0.0", "courses": [
            {"course_slug": "course_a", "status": "processed", "dem_type": "USGS1m"},
            {"course_slug": "course_b", "status": "processed", "dem_type": "COP30"},
            {"course_slug": "course_c", "status": "skipped", "dem_type": None},
            {"course_slug": "course_d", "status": "failed", "dem_type": None},
        ]}
        index.all_courses_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        # a visual check PNG
        index.visual_checks.mkdir(parents=True, exist_ok=True)
        (index.visual_checks / "sample.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    if compact:
        for hid, slug, num, *_ in _HOLES:
            _write_compact(courses_root, slug, num, hid)

    return courses_root


# --------------------------------------------------------------------------- #
# Secret guard
# --------------------------------------------------------------------------- #

def test_is_secret_flags_env_and_keys():
    assert _is_secret(Path(".env"))
    assert _is_secret(Path(".env.local"))
    assert _is_secret(Path("private.pem"))
    assert _is_secret(Path("id_rsa.key"))
    assert _is_secret(Path("credentials.json"))
    assert not _is_secret(Path("hole_features.parquet"))
    assert not _is_secret(Path("README.md"))


def test_safe_copy_refuses_env(tmp_path: Path):
    src = tmp_path / ".env"
    src.write_text("OPENTOPOGRAPHY_API_KEY=secret", encoding="utf-8")
    with pytest.raises(RuntimeError, match="secret"):
        _safe_copy(src, tmp_path / "out" / ".env")
    assert not (tmp_path / "out" / ".env").exists()


def test_verify_no_secrets(tmp_path: Path):
    (tmp_path / "data.parquet").write_bytes(b"x")
    _verify_no_secrets(tmp_path)  # clean -> no raise
    (tmp_path / ".env").write_text("k=v", encoding="utf-8")
    with pytest.raises(RuntimeError, match="secret-like"):
        _verify_no_secrets(tmp_path)


def test_export_does_not_sweep_in_env(tmp_path: Path):
    """A stray .env in the index dir must never reach the artifact."""
    courses_root = make_courses(tmp_path)
    (IndexPaths.for_root(courses_root).root / ".env").write_text(
        "OPENTOPOGRAPHY_API_KEY=secret", encoding="utf-8")
    out = tmp_path / "art"
    build_hf_artifact("lite", out, courses_root=courses_root)
    assert not list(out.rglob(".env"))  # allow-list copy never picked it up


# --------------------------------------------------------------------------- #
# Manifest / schema / feature dictionary
# --------------------------------------------------------------------------- #

def test_manifest_creation(tmp_path: Path):
    courses_root = make_courses(tmp_path)
    out = tmp_path / "art"
    build_hf_artifact("lite", out, courses_root=courses_root)

    manifest = json.loads((out / "dataset_manifest.json").read_text())
    assert manifest["project"] == "GolfDataScience"
    assert manifest["tier"] == "lite"
    assert manifest["holes_processed"] == 6
    assert manifest["courses_processed"] == 2
    assert manifest["courses_skipped"] == 1
    assert manifest["courses_failed"] == 1
    assert manifest["point_count_total"] == int(_feature_df()["point_count"].sum())
    assert manifest["labels"]["3"] == "fairway"
    assert manifest["artifacts"] and all("path" in a for a in manifest["artifacts"])
    # the manifest never lists itself
    assert all(a["path"] != "dataset_manifest.json" for a in manifest["artifacts"])


def test_schema_and_feature_dictionary(tmp_path: Path):
    df = _feature_df()
    schema = build_schema(df)
    assert "hole_features" in schema["tables"]
    assert "point_cloud_compact_json" in schema
    assert schema["point_cloud_compact_json"]["points_row"].startswith("[x_aligned_m")

    fd = build_feature_dictionary(df)
    # every column resolves to a group and a non-empty description
    assert fd["n_columns"] == len(df.columns)
    for name, info in fd["columns"].items():
        assert info["group"]
        assert info["description"]
    # identifiers and engineered families are recognised (no everything-"other")
    groups = {info["group"] for info in fd["columns"].values()}
    assert {"identifier", "label_composition", "zone_composition"} <= groups


# --------------------------------------------------------------------------- #
# Tier behavior
# --------------------------------------------------------------------------- #

def test_lite_copies_expected_files(tmp_path: Path):
    courses_root = make_courses(tmp_path)
    out = tmp_path / "art"
    summary = build_hf_artifact("lite", out, courses_root=courses_root)

    for rel in [
        "data/hole_features.parquet", "data/hole_features.csv",
        "data/hole_clusters.parquet", "data/hole_similarity_v2.csv",
        "data/hole_similarity_examples.csv", "data/all_holes.parquet",
        "data/all_courses_manifest.json",
        "metadata/label_map.json", "metadata/schema.json",
        "metadata/feature_dictionary.json", "metadata/provenance.json",
        "README.md", "dataset_card.md", "dataset_manifest.json",
    ]:
        assert (out / rel).exists(), f"missing {rel}"

    # lite selects a curated subset, not all six holes
    compacts = list((out / "point_clouds" / "compact").glob("*.json"))
    assert 0 < len(compacts) < len(_HOLES)
    # README carries HF front matter; dataset_card.md does not
    assert (out / "README.md").read_text().startswith("---")
    assert not (out / "dataset_card.md").read_text().startswith("---")
    assert summary["missing_optional_inputs"] == []


def test_output_paths_created(tmp_path: Path):
    courses_root = make_courses(tmp_path)
    out = tmp_path / "nested" / "art"
    build_hf_artifact("lite", out, courses_root=courses_root)
    assert (out / "data").is_dir()
    assert (out / "metadata").is_dir()
    assert (out / "point_clouds" / "compact").is_dir()


def test_full_tier_discovers_all_compacts(tmp_path: Path):
    courses_root = make_courses(tmp_path)
    out = tmp_path / "art_full"
    summary = build_hf_artifact("full", out, courses_root=courses_root,
                                include_point_parquet=True)

    compacts = sorted(p.stem for p in (out / "point_clouds" / "compact").glob("*.json"))
    assert compacts == sorted(h[0] for h in _HOLES)
    assert summary["compact_point_clouds"] == len(_HOLES)
    # parquet point files copied when requested
    pqs = list((out / "point_clouds" / "parquet").glob("*.parquet"))
    assert len(pqs) == len(_HOLES)


def test_curated_selection_is_representative(tmp_path: Path):
    courses_root = make_courses(tmp_path)
    df = _feature_df()
    ids = curated_hole_ids(
        df, IndexPaths.for_root(courses_root).hole_similarity_v2_csv,
        anchor_hole_id="course_a__01", n_neighbors=4)
    assert ids[0] == "course_a__01"          # anchor first
    assert "course_b__01" in ids             # v2 neighbor + water-heavy
    assert "course_b__02" in ids             # terrain-heavy (max z_range)
    assert len(ids) == len(set(ids))         # de-duplicated


# --------------------------------------------------------------------------- #
# Graceful degradation / errors
# --------------------------------------------------------------------------- #

def test_missing_optional_files_handled(tmp_path: Path):
    courses_root = make_courses(tmp_path, optional=False)  # only hole_features.parquet
    out = tmp_path / "art"
    summary = build_hf_artifact("lite", out, courses_root=courses_root)

    # required output still present; optional ones recorded as missing, no crash
    assert (out / "data" / "hole_features.parquet").exists()
    assert (out / "dataset_manifest.json").exists()
    assert "data/hole_clusters.parquet" in summary["missing_optional_inputs"]
    assert "data/all_holes.parquet" in summary["missing_optional_inputs"]


def test_missing_feature_table_raises(tmp_path: Path):
    courses_root = tmp_path / "courses"
    IndexPaths.for_root(courses_root).ensure()
    with pytest.raises(FileNotFoundError, match="hole_features"):
        build_hf_artifact("lite", tmp_path / "art", courses_root=courses_root)


def test_invalid_tier_raises(tmp_path: Path):
    courses_root = make_courses(tmp_path)
    with pytest.raises(ValueError, match="tier"):
        build_hf_artifact("medium", tmp_path / "art", courses_root=courses_root)
