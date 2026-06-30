"""Tests for the v2.5 config sweep runner."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from pipeline.modeling.pointcloud.export_similarity import RESULTS_FILENAME
from pipeline.modeling.pointcloud.schemas import (
    HoleMetadata,
    SurfacePoint,
    make_pc_hole_id,
)
from pipeline.modeling.pointcloud.sweep import (
    STATUS_FAILED,
    STATUS_RAN,
    STATUS_SKIPPED,
    SWEEP_MANIFEST_FILENAME,
    run_sweep,
)
from tests.test_pointcloud_config import _valid_payload


class FakeLoader:
    def __init__(self, metadata, points):
        self._metadata = metadata
        self._points = points

    def load_metadata(self):
        return dict(self._metadata)

    def load_points(self, hole_id):
        return list(self._points.get(hole_id, []))


def _all_surface_points(hole_id, *, dx=0.0):
    pts = []
    for surface in ("fairway", "green", "bunker", "water", "tee"):
        for i in range(3):
            pts.append(SurfacePoint(hole_id, surface, float(i) + dx, float(i), 0.0))
    return pts


def _loader():
    a = make_pc_hole_id("aaa_course", 1)
    b = make_pc_hole_id("bbb_course", 1)
    meta = {
        a: HoleMetadata(a, "aaa_course", 1, 4, 440.0, has_tee=True, has_green=True,
                        has_fairway=True, has_bunker=True, has_water=True),
        b: HoleMetadata(b, "bbb_course", 1, 4, 445.0, has_tee=True, has_green=True,
                        has_fairway=True, has_bunker=True, has_water=True),
    }
    pts = {a: _all_surface_points(a), b: _all_surface_points(b, dx=1.0)}
    return FakeLoader(meta, pts)


def _write_config(path, name):
    payload = _valid_payload()
    payload["config_name"] = name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_sweep_runs_multiple_configs(tmp_path):
    c1 = _write_config(tmp_path / "c1.yaml", "alpha")
    c2 = _write_config(tmp_path / "c2.yaml", "beta")
    out_root = tmp_path / "out"

    manifest = run_sweep([c1, c2], loader=_loader(), top_n=5,
                         output_root=out_root, sweep_manifest_dir=out_root)

    assert manifest["ok"] is True
    assert manifest["n_ran"] == 2
    assert manifest["n_failed"] == 0
    # Per-config outputs written.
    assert (out_root / "alpha" / RESULTS_FILENAME).exists()
    assert (out_root / "beta" / RESULTS_FILENAME).exists()
    # Sweep manifest persisted.
    on_disk = json.loads((out_root / SWEEP_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert {e["config_name"] for e in manifest["configs"]} == {"alpha", "beta"}


def test_sweep_skips_existing_unless_overwrite(tmp_path):
    c1 = _write_config(tmp_path / "c1.yaml", "alpha")
    out_root = tmp_path / "out"
    loader = _loader()

    run_sweep([c1], loader=loader, top_n=5, output_root=out_root, sweep_manifest_dir=out_root)
    # Second run without overwrite -> skipped.
    again = run_sweep([c1], loader=loader, top_n=5, output_root=out_root,
                      sweep_manifest_dir=out_root)
    assert again["configs"][0]["status"] == STATUS_SKIPPED
    assert again["n_skipped"] == 1

    # With overwrite -> ran again.
    forced = run_sweep([c1], loader=loader, top_n=5, output_root=out_root,
                       sweep_manifest_dir=out_root, overwrite=True)
    assert forced["configs"][0]["status"] == STATUS_RAN


def test_sweep_records_failure_and_continues(tmp_path):
    good = _write_config(tmp_path / "good.yaml", "alpha")
    missing = tmp_path / "does_not_exist.yaml"
    out_root = tmp_path / "out"

    manifest = run_sweep([missing, good], loader=_loader(), top_n=5,
                         output_root=out_root, sweep_manifest_dir=out_root)

    assert manifest["ok"] is False
    assert manifest["n_failed"] == 1
    assert manifest["n_ran"] == 1
    failed = [e for e in manifest["configs"] if e["status"] == STATUS_FAILED][0]
    assert failed["error_type"] == "FileNotFoundError"
    assert "error" in failed
    # The good config still ran despite the earlier failure.
    assert (out_root / "alpha" / RESULTS_FILENAME).exists()


def test_sweep_manifest_counts_are_consistent(tmp_path):
    c1 = _write_config(tmp_path / "c1.yaml", "alpha")
    c2 = _write_config(tmp_path / "c2.yaml", "beta")
    out_root = tmp_path / "out"
    manifest = run_sweep([c1, c2], loader=_loader(), top_n=5,
                         output_root=out_root, sweep_manifest_dir=out_root)
    ran = [e for e in manifest["configs"] if e["status"] == STATUS_RAN]
    for e in ran:
        results = pd.read_csv(tmp_path / "out" / e["config_name"] / RESULTS_FILENAME)
        assert e["total_written_rows"] == len(results)
