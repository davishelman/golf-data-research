"""Hole-similarity modeling subpackage.

Decoupled from the geo pipeline: depends only on the light data-science stack
(pandas, numpy, duckdb, pyarrow, scikit-learn; UMAP optional). It reads the
aggregate / per-hole point clouds produced by the pipeline and builds:

  hole_features.parquet/csv         (one feature row per hole)
  hole_clusters.parquet/csv         (cluster labels + 2D embedding per hole)
  hole_similarity_examples.csv      (top-K nearest holes per hole)

Run with:  python -m pipeline.modeling {features|similarity|all}
"""

from __future__ import annotations

# Columns that identify a hole — never scaled or treated as model features.
ID_COLUMNS: tuple[str, ...] = (
    "hole_id", "course_slug", "course_name", "hole_number",
)

# Canonical point labels (must match pipeline.constants.LABEL_IDS keys).
POINT_LABELS: tuple[str, ...] = (
    "unknown", "tee", "green", "fairway", "rough_osm", "bunker",
    "water", "trees", "cartpath", "sand", "rough_inferred",
)
ROUGH_LABELS: tuple[str, ...] = ("rough_osm", "rough_inferred")

__all__ = ["ID_COLUMNS", "POINT_LABELS", "ROUGH_LABELS"]
