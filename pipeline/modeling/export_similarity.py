"""Run clustering + nearest-neighbor search over hole features and write outputs.

Produces:
  courses/_index/hole_clusters.parquet / .csv     (ids + cluster labels + 2D embeddings)
  courses/_index/hole_similarity_examples.csv      (v1: top-K similar holes per hole)
  courses/_index/hole_similarity_v2.csv            (v2: length-aware, cross-course,
                                                    same-par, length-guarded)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..logging_config import get_logger
from ..paths import COURSES_ROOT, IndexPaths
from .similarity import (
    GOLF_MODES,
    build_feature_matrix,
    cluster_agglomerative,
    cluster_kmeans,
    feature_columns,
    feature_columns_for_mode,
    nearest_neighbor_table,
    resolve_mode,
    run_pca,
    run_umap,
)

log = get_logger("modeling.export")

_CLUSTER_ID_COLS = ("hole_id", "course_slug", "course_name", "hole_number",
                    "par", "hole_length_m")

V2_MODE = "cross_course_same_par_length_guarded"

# Column order for per-mode similarity files.
_MODE_CSV_COLUMNS = (
    "similarity_mode", "query_hole_id", "query_course_slug", "query_hole_number",
    "similar_hole_id", "similar_course_slug", "similar_hole_number",
    "rank", "distance", "query_length_m", "similar_length_m", "length_diff_m",
    "same_par", "same_course",
)


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

    # v1 similarity examples (unweighted, no filters) — preserved exactly.
    examples = nearest_neighbor_table(df, X, k=n_neighbors)
    examples.to_csv(index.hole_similarity_examples_csv, index=False)

    # v2 similarity examples (length-aware, cross-course, same-par, length-guarded).
    v2 = _build_v2_table(df, cols, n_neighbors)
    v2.to_csv(index.hole_similarity_v2_csv, index=False)

    written = {
        "hole_clusters_parquet": str(index.hole_clusters_parquet),
        "hole_clusters_csv": str(index.hole_clusters_csv),
        "hole_similarity_examples_csv": str(index.hole_similarity_examples_csv),
        "hole_similarity_v2_csv": str(index.hole_similarity_v2_csv),
    }
    log.info("similarity outputs written: %s", list(written))
    log.info("clusters: kmeans=%d agg=%d | v1 rows=%d | v2 rows=%d",
             len(set(kmeans)), len(set(agg)), len(examples), len(v2))
    return written


def _build_v2_table(df: pd.DataFrame, cols: list[str], n_neighbors: int) -> pd.DataFrame:
    """Length-aware v2 neighbor table with length-diff and provenance columns."""
    cfg = resolve_mode(V2_MODE)
    Xw, _, _ = build_feature_matrix(df, cols, feature_weights=cfg["feature_weights"])
    table = nearest_neighbor_table(
        df, Xw, k=n_neighbors,
        exclude_same_course=cfg["exclude_same_course"], same_par=cfg["same_par"],
        max_length_diff_m=cfg["max_length_diff_m"],
        max_length_diff_pct=cfg["max_length_diff_pct"],
    )
    len_of = df.set_index("hole_id")["hole_length_m"]
    if table.empty:
        return pd.DataFrame(columns=[
            "query_hole_id", "similar_hole_id", "rank", "distance",
            "query_length_m", "similar_length_m", "length_diff_m",
            "same_par", "same_course", "similarity_mode",
        ])
    table["query_length_m"] = table["query_hole_id"].map(len_of).round(2)
    table["similar_length_m"] = table["similar_hole_id"].map(len_of).round(2)
    table["length_diff_m"] = (table["query_length_m"] - table["similar_length_m"]).abs().round(2)
    table["same_par"] = bool(cfg["same_par"])
    table["same_course"] = table["query_course_slug"] == table["similar_course_slug"]
    table["similarity_mode"] = V2_MODE
    return table[[
        "query_hole_id", "similar_hole_id", "rank", "distance",
        "query_length_m", "similar_length_m", "length_diff_m",
        "same_par", "same_course", "similarity_mode",
    ]]


def _enrich_mode(table: pd.DataFrame, df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Add length / par / course provenance columns for a per-mode neighbor table."""
    if table.empty:
        return pd.DataFrame(columns=list(_MODE_CSV_COLUMNS))
    len_of = df.set_index("hole_id")["hole_length_m"]
    par_of = df.set_index("hole_id")["par"]
    t = table.copy()
    t["query_length_m"] = t["query_hole_id"].map(len_of).round(2)
    t["similar_length_m"] = t["similar_hole_id"].map(len_of).round(2)
    t["length_diff_m"] = (t["query_length_m"] - t["similar_length_m"]).abs().round(2)
    t["same_par"] = (t["query_hole_id"].map(par_of).to_numpy()
                     == t["similar_hole_id"].map(par_of).to_numpy())
    t["same_course"] = t["query_course_slug"] == t["similar_course_slug"]
    t["similarity_mode"] = mode
    return t[list(_MODE_CSV_COLUMNS)]


def build_similarity_modes(
    courses_root: Path = COURSES_ROOT,
    modes: tuple[str, ...] = GOLF_MODES,
    n_neighbors: int = 10,
    exclude_same_course: bool | None = None,
) -> dict[str, str]:
    """Write one ``similarity_modes/<mode>.csv`` per golf mode.

    Each mode uses its own feature subset, weights, and default filters. Pass
    ``exclude_same_course`` to override every mode's default (e.g. force
    cross-course). Existing v1/v2 outputs are untouched.
    """
    index = IndexPaths.for_root(courses_root)
    if not index.hole_features_parquet.exists():
        raise FileNotFoundError(
            f"{index.hole_features_parquet} not found. Build features first:\n"
            "    python -m pipeline.modeling features"
        )
    df = pd.read_parquet(index.hole_features_parquet)
    if len(df) < 2:
        raise ValueError("Need at least 2 holes to compute similarity modes.")

    out_dir = index.similarity_modes_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for mode in modes:
        cfg = resolve_mode(mode)
        cols = feature_columns_for_mode(df, mode)
        Xm, _, _ = build_feature_matrix(df, cols, feature_weights=cfg["feature_weights"])
        esc = cfg["exclude_same_course"] if exclude_same_course is None else exclude_same_course
        table = nearest_neighbor_table(
            df, Xm, k=n_neighbors, exclude_same_course=esc, same_par=cfg["same_par"],
            max_length_diff_m=cfg["max_length_diff_m"],
            max_length_diff_pct=cfg["max_length_diff_pct"],
        )
        enriched = _enrich_mode(table, df, mode)
        path = out_dir / f"{mode}.csv"
        enriched.to_csv(path, index=False)
        written[mode] = str(path)
        log.info("mode '%s' (%d features): %d rows -> %s",
                 mode, len(cols), len(enriched), path)
    return written


def build_presented_similarity(
    courses_root: Path = COURSES_ROOT,
    *,
    n_neighbors: int = 10,
    require_same_par: bool = True,
    exclude_same_course: bool = True,
    min_score: float = 0.75,
    source_csv: Path | None = None,
    name: str = "overall_v2",
) -> dict[str, str]:
    """Write a golfer-presentable similarity table to ``presented_similarity/``.

    Wraps a raw similarity table (default: ``hole_similarity_v2.csv``) with the
    golf-plausibility layer (:mod:`pipeline.modeling.plausibility`): flags, a 0–1
    score, readable reasons, then keeps the best ``n_neighbors`` same-par,
    cross-course, length-/plausibility-filtered matches per hole. The raw v1/v2/
    mode outputs are read-only and untouched.
    """
    from .plausibility import presented_similarity_table

    index = IndexPaths.for_root(courses_root)
    if not index.hole_features_parquet.exists():
        raise FileNotFoundError(
            f"{index.hole_features_parquet} not found. Build features first:\n"
            "    python -m pipeline.modeling features"
        )
    df = pd.read_parquet(index.hole_features_parquet)

    src = Path(source_csv) if source_csv is not None else index.hole_similarity_v2_csv
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Build the raw similarity tables first:\n"
            "    python -m pipeline.modeling similarity"
        )
    raw = pd.read_csv(src)

    table = presented_similarity_table(
        raw, df, n_neighbors=n_neighbors,
        require_same_par=require_same_par, exclude_same_course=exclude_same_course,
        min_score=min_score)

    out_dir = index.presented_similarity_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    table.to_csv(path, index=False)

    n_queries = int(table["query_hole_id"].nunique()) if not table.empty else 0
    total_queries = int(raw["query_hole_id"].nunique()) if "query_hole_id" in raw else 0
    log.info("presented '%s': %d rows, %d/%d queries (min_score=%.2f, same_par=%s, "
             "cross_course=%s) -> %s", name, len(table), n_queries, total_queries,
             min_score, require_same_par, exclude_same_course, path)
    if total_queries and n_queries < total_queries:
        log.info("  %d queries had no presentable match at this threshold",
                 total_queries - n_queries)
    return {name: str(path)}
