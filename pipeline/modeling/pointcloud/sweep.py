"""Reproducible config sweep runner for v2.5 point-cloud similarity.

Runs the batch export for several configs in one command, writing each config's
outputs to its own directory and a top-level ``sweep_manifest.json`` recording
per-config status (ran / skipped / failed), row counts, and timings.

Design choices:

* **Skip existing** — a config whose ``similarity_results.csv`` already exists is
  skipped unless ``overwrite`` is set, so re-running a sweep is cheap and
  idempotent.
* **Continue on failure, report clearly** — one config raising does not abort the
  sweep; the error is captured (type + message) into that config's entry and the
  sweep continues. Failures are surfaced in the manifest and the return value
  (``ok=False``) rather than swallowed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...logging_config import get_logger
from ...paths import COURSES_ROOT, IndexPaths
from .config import load_config
from .export_similarity import (
    CompactArtifactLoader,
    PointCloudArtifactLoader,
    RESULTS_FILENAME,
    default_output_dir,
    run_batch_export,
)

log = get_logger("modeling.pointcloud.sweep")

SWEEP_MANIFEST_FILENAME = "sweep_manifest.json"

STATUS_RAN = "ran"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


def run_sweep(
    config_paths: list[Path],
    *,
    loader: Optional[PointCloudArtifactLoader] = None,
    courses_root: Path = COURSES_ROOT,
    top_n: int = 25,
    include_self: bool = False,
    overwrite: bool = False,
    output_root: Optional[Path] = None,
    sweep_manifest_dir: Optional[Path] = None,
) -> dict:
    """Run a batch export for each config; return a sweep manifest dict.

    Each config's outputs go to ``output_root/<config_name>/`` (default
    ``courses/_index/pointcloud_similarity/<config_name>/``). Existing results are
    skipped unless ``overwrite``. A config that raises is recorded as ``failed``
    and the sweep continues.
    """
    loader = loader or CompactArtifactLoader(courses_root=courses_root)
    started = datetime.now(timezone.utc)

    entries: list[dict] = []
    for config_path in config_paths:
        config_path = Path(config_path)
        entry: dict = {"config_path": str(config_path)}
        try:
            config = load_config(config_path)
        except Exception as exc:  # noqa: BLE001 - recorded + surfaced, not swallowed
            entry.update(status=STATUS_FAILED, error_type=type(exc).__name__, error=str(exc))
            entries.append(entry)
            log.error("sweep: failed to load %s: %s", config_path, exc)
            continue

        entry["config_name"] = config.config_name
        out_dir = (Path(output_root) / config.config_name if output_root
                   else default_output_dir(config, courses_root))
        entry["output_dir"] = str(out_dir)

        if (out_dir / RESULTS_FILENAME).exists() and not overwrite:
            entry["status"] = STATUS_SKIPPED
            entries.append(entry)
            log.info("sweep: skipping '%s' (results exist; use overwrite)", config.config_name)
            continue

        t0 = time.perf_counter()
        try:
            manifest = run_batch_export(
                config, loader, output_dir=out_dir, config_path=config_path,
                top_n=top_n, include_self=include_self, overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - recorded + surfaced, not swallowed
            entry.update(status=STATUS_FAILED, error_type=type(exc).__name__, error=str(exc),
                         elapsed_seconds=round(time.perf_counter() - t0, 3))
            entries.append(entry)
            log.error("sweep: config '%s' failed: %s", config.config_name, exc)
            continue

        entry.update(
            status=STATUS_RAN,
            elapsed_seconds=round(time.perf_counter() - t0, 3),
            total_targets=manifest["total_targets"],
            total_scored_pairs=manifest["total_scored_pairs"],
            total_written_rows=manifest["total_written_rows"],
            config_hash=manifest["config_hash"],
        )
        entries.append(entry)

    ran = [e for e in entries if e["status"] == STATUS_RAN]
    skipped = [e for e in entries if e["status"] == STATUS_SKIPPED]
    failed = [e for e in entries if e["status"] == STATUS_FAILED]

    sweep_manifest = {
        "created_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "top_n": top_n,
        "include_self": include_self,
        "overwrite": overwrite,
        "n_configs": len(config_paths),
        "n_ran": len(ran),
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "ok": len(failed) == 0,
        "configs": entries,
    }

    manifest_dir = Path(sweep_manifest_dir) if sweep_manifest_dir else (
        Path(output_root) if output_root else IndexPaths.for_root(courses_root).pointcloud_similarity_dir
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / SWEEP_MANIFEST_FILENAME).write_text(
        json.dumps(sweep_manifest, indent=2) + "\n", encoding="utf-8"
    )

    log.info("sweep complete: %d ran, %d skipped, %d failed (ok=%s)",
             len(ran), len(skipped), len(failed), sweep_manifest["ok"])
    return sweep_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline.modeling.pointcloud.sweep",
        description="Run v2.5 batch similarity export across multiple configs.",
    )
    parser.add_argument("--configs", nargs="+", required=True, type=Path,
                        help="Config YAML paths to sweep.")
    parser.add_argument("--top-n", type=int, default=25,
                        help="Neighbors to keep per target (default: 25).")
    parser.add_argument("--courses-root", type=Path, default=COURSES_ROOT)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Root for per-config output dirs (default: "
                             "courses/_index/pointcloud_similarity/).")
    parser.add_argument("--include-self", action="store_true",
                        help="Include each hole's self-comparison.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run configs even if their results already exist.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_sweep(
        args.configs, courses_root=args.courses_root, top_n=args.top_n,
        include_self=args.include_self, overwrite=args.overwrite,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
