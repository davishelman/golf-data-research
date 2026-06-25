"""Run clustering + nearest-neighbor search over hole features and write outputs.

Produces:
  courses/_index/hole_clusters.parquet / .csv   (ids + cluster labels + 2D embeddings)
  courses/_index/hole_similarity_examples.csv    (top-K similar holes per hole)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_config import get_logger
from ..paths import COURSES_ROOT, IndexPaths
from .similarity import (
    build_feature_matrix,
    cluster_agglomerative,
    cluster_kmeans,
    feature_columns,
    nearest_neighbor_table,
    run_pca,
    run_umap,
)

log = get_logger("modeling.export")

_CLUSTER_ID_COLS = ("hole_id", "course_slug", "course_name", "hole_number",
                    "par", "hole_length_m")


def build_hole_similarity(
    courses_root: Path = COURSES_ROOT,
    n_clusters: int = 8,
    n_neighbors: int = 10,
    pca_components: int = 10,
    use_umap: bool = True,
) -> dict[str, str]:
    index = IndexPaths.for_root(courses_root)
    if not index.hole_features_parquet.exists():
        raise FileNotFoundError(
            f"{index.hole_features_parquet} not found. Build features first:\n"
            "    python -m pipeline.modeling features"
        )

    df = pd.read_parquet(index.hole_features_parquet)
    if len(df) < 2:
        raise ValueError("Need at least 2 holes to compute similarity/clusters.")

    cols = feature_columns(df)
    log.info("modeling %d holes on %d numeric features", len(df), len(cols))
    X, _imputer, _scaler = build_feature_matrix(df, cols)

    # Clusters on the full scaled feature space.
    kmeans = cluster_kmeans(X, k=n_clusters)
    agg = cluster_agglomerative(X, k=n_clusters)

    # 2D embeddings for visualization.
    pca2, _ = run_pca(X, n_components=2)
    umap2 = run_umap(X) if use_umap else None
    # (pca_components kept for callers who want a richer PCA; 2D used for plots.)
    if pca_components and pca_components > 2:
        run_pca(X, n_components=pca_components)  # validates it fits; not persisted

    keep = [c for c in _CLUSTER_ID_COLS if c in df.columns]
    clusters = df[keep].copy()
    clusters["kmeans_cluster"] = kmeans
    clusters["agg_cluster"] = agg
    clusters["pca_1"] = pca2[:, 0]
    clusters["pca_2"] = pca2[:, 1] if pca2.shape[1] > 1 else 0.0
    if umap2 is not None:
        clusters["umap_1"] = umap2[:, 0]
        clusters["umap_2"] = umap2[:, 1]

    index.ensure()
    clusters.to_parquet(index.hole_clusters_parquet, index=False)
    clusters.to_csv(index.hole_clusters_csv, index=False)

    examples = nearest_neighbor_table(df, X, k=n_neighbors)
    examples.to_csv(index.hole_similarity_examples_csv, index=False)

    written = {
        "hole_clusters_parquet": str(index.hole_clusters_parquet),
        "hole_clusters_csv": str(index.hole_clusters_csv),
        "hole_similarity_examples_csv": str(index.hole_similarity_examples_csv),
    }
    log.info("similarity outputs written: %s", list(written))
    log.info("clusters: kmeans=%d agg=%d | similarity rows=%d",
             len(set(kmeans)), len(set(agg)), len(examples))
    return written
