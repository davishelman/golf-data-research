"""Add v2.5 point-cloud similarity outputs to a Hugging Face artifact folder.

This is additive to :mod:`pipeline.modeling.hf_export`: it does not change the v2
artifact contents, and the v2.5 results live in their own subtree
(``data/pointcloud_similarity/<config_name>/``) so the two model families stay
clearly separated. Only IDs + scores are exported — no raw point-cloud geometry.

It reuses the v2 exporter's secret guard (:func:`_safe_copy` /
:func:`_verify_no_secrets`) so an ``.env`` (or any secret-like file) can never be
swept into the bundle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ...logging_config import get_logger
from ...paths import COURSES_ROOT, IndexPaths
from ..hf_export import _safe_copy, _verify_no_secrets
from . import MODEL_VERSION
from .export_similarity import (
    FILTER_SUMMARY_FILENAME,
    MANIFEST_FILENAME,
    RESULTS_FILENAME,
    SIMILARITY_RESULTS_COLUMNS,
)
from .validate_similarity import VALIDATION_DIRNAME

log = get_logger("modeling.pointcloud.artifact_export")

#: Relative subtree (under the artifact root) for v2.5 outputs.
ARTIFACT_SUBPATH = "data/pointcloud_similarity"

#: Files copied per config (results + provenance; never geometry).
_PER_CONFIG_FILES = (RESULTS_FILENAME, MANIFEST_FILENAME, FILTER_SUMMARY_FILENAME)

_METADATA_FILENAME = "pointcloud_similarity.json"


def discover_config_dirs(courses_root: Path = COURSES_ROOT) -> list[Path]:
    """Batch-output config dirs (those containing ``similarity_results.csv``).

    Excludes the ``_validation`` working area. Sorted by name for determinism.
    """
    root = IndexPaths.for_root(courses_root).pointcloud_similarity_dir
    if not root.exists():
        return []
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and d.name != VALIDATION_DIRNAME and (d / RESULTS_FILENAME).exists()
    )


def add_pointcloud_similarity_to_artifact(
    artifact_root: Path,
    *,
    courses_root: Path = COURSES_ROOT,
    config_names: Optional[list[str]] = None,
) -> dict:
    """Copy selected v2.5 batch outputs into an existing artifact folder.

    For each chosen config, copies ``similarity_results.csv``, ``manifest.json``
    and ``filter_summary.csv`` into ``data/pointcloud_similarity/<config_name>/``
    and writes a top-level ``metadata/pointcloud_similarity.json`` index. Returns
    a summary dict. Raises if no eligible config outputs are found.
    """
    artifact_root = Path(artifact_root)
    available = discover_config_dirs(courses_root)
    if config_names is not None:
        wanted = set(config_names)
        available = [d for d in available if d.name in wanted]
    if not available:
        raise FileNotFoundError(
            "no v2.5 batch outputs found to export. Build them first:\n"
            "    python -m pipeline.modeling.pointcloud.export_similarity "
            "--config <cfg> --all"
        )

    dest_root = artifact_root / ARTIFACT_SUBPATH
    dest_root.mkdir(parents=True, exist_ok=True)

    configs_meta: list[dict] = []
    for cfg_dir in available:
        dest = dest_root / cfg_dir.name
        copied: list[str] = []
        for fname in _PER_CONFIG_FILES:
            src = cfg_dir / fname
            if src.exists():
                _safe_copy(src, dest / fname)
                copied.append(fname)
            else:
                log.warning("v2.5 export: %s missing for config '%s'", fname, cfg_dir.name)

        manifest = _load_manifest(cfg_dir / MANIFEST_FILENAME)
        configs_meta.append({
            "config_name": cfg_dir.name,
            "files": copied,
            "model_version": manifest.get("model_version"),
            "config_hash": manifest.get("config_hash"),
            "total_written_rows": manifest.get("total_written_rows"),
            "top_n": manifest.get("top_n"),
        })

    meta_dir = artifact_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "description": (
            "v2.5 surface-aware point-cloud Chamfer similarity rankings. Additive "
            "to and separate from the v2 feature-table similarity (data/"
            "hole_similarity_v2.csv); these use the 'course_slug:hole_number' id "
            "space and store IDs + scores only (no geometry)."
        ),
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subpath": ARTIFACT_SUBPATH,
        "results_columns": list(SIMILARITY_RESULTS_COLUMNS),
        "configs": configs_meta,
    }
    (meta_dir / _METADATA_FILENAME).write_text(json.dumps(index, indent=2) + "\n",
                                               encoding="utf-8")

    # Defensive: ensure nothing secret-like slipped into the v2.5 subtree.
    _verify_no_secrets(dest_root)

    summary = {
        "artifact_root": str(artifact_root),
        "subpath": ARTIFACT_SUBPATH,
        "configs_exported": [c["config_name"] for c in configs_meta],
        "n_configs": len(configs_meta),
    }
    log.info("v2.5 artifact export: %d configs -> %s",
             len(configs_meta), dest_root)
    return summary


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "add_pointcloud_similarity_to_artifact",
    "discover_config_dirs",
    "ARTIFACT_SUBPATH",
]
