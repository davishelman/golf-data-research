"""Backward-compatible CSV export.

Delegates to ``exports.collect_hole_rows`` (which reads the new per-hole
``stats/terrain_summary.json`` artifacts) and writes a flat ``all_holes.csv``.
"""

from __future__ import annotations

from pathlib import Path

from . import exports
from .paths import COURSES_ROOT, IndexPaths


def export_holes_csv(
    courses_root: Path = COURSES_ROOT,
    output_path: Path | None = None,
) -> Path:
    """Compile every processed hole into a single flat CSV. Returns its path."""
    out = output_path or IndexPaths.for_root(courses_root).all_holes_csv
    rows = exports.collect_hole_rows(courses_root)
    if not rows:
        print(f"[export] no processed holes found under {courses_root}")
        out.parent.mkdir(parents=True, exist_ok=True)
    return exports.write_all_holes_csv(rows, out)
