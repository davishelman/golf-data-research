"""Similarity + clustering primitives for hole feature rows.

Pure, composable functions over a feature DataFrame. Identifiers (see
``pipeline.modeling.ID_COLUMNS``) and any non-numeric columns are never scaled or
fed to the model. Optional dependencies (UMAP) degrade gracefully; required ones
(scikit-learn) raise a clear install message.

Missing-value policy
--------------------
Engineered features are intentionally ``NaN`` when undefined for a hole (e.g. a
par-3 has no drive zone, so ``fairway_width_drive_zone`` is NaN). ``build_feature_matrix``
makes the handling explicit: each feature column is **median-imputed**, then
standardized. All-missing columns are kept (imputed to 0) so the matrix shape
always matches the selected columns, and constant columns scale to 0 (no NaNs
leak into the model).
"""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from . import ID_COLUMNS

log = get_logger("modeling.similarity")


def _require(module: str, pip_name: str):
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            f"'{module}' is required for this step. Install it with:\n"
            f"    pip install {pip_name}"
        ) from exc


def umap_available() -> bool:
    return importlib.util.find_spec("umap") is not None


# ---------------------------------------------------------------------------
# Feature selection + inspection + scaling
# ---------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric, model-eligible columns (excludes identifiers + non-numeric)."""
    drop = set(ID_COLUMNS)
    return [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]


def feature_summary(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Per-feature inspection table: dtype + missing counts, sorted by missingness.

    Handy for notebooks to see *which* engineered features are sparse (and why),
    before they are median-imputed for modeling.
    """
    cols = cols if cols is not None else feature_columns(df)
    n = len(df)
    rows = []
    for c in cols:
        miss = int(df[c].isna().sum())
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "n_missing": miss,
            "pct_missing": round(miss / n, 4) if n else 0.0,
        })
    return (pd.DataFrame(rows)
            .sort_values("pct_missing", ascending=False)
            .reset_index(drop=True))


def build_feature_matrix(df: pd.DataFrame, cols: list[str]):
    """Median-impute then standardize. Returns ``(X, imputer, scaler)``.

    See the module docstring for the missing-value policy. Identifiers are not in
    ``cols`` (use :func:`feature_columns`) and are therefore never scaled.
    """
    sk_impute = _require("sklearn.impute", "scikit-learn")
    sk_pre = _require("sklearn.preprocessing", "scikit-learn")
    X = df[cols].to_numpy(dtype="float64")
    imputer = sk_impute.SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imputer.fit_transform(X)
    scaler = sk_pre.StandardScaler()
    Xs = scaler.fit_transform(Xi)
    return Xs, imputer, scaler


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------


def run_pca(X: np.ndarray, n_components: int = 2):
    sk_dec = _require("sklearn.decomposition", "scikit-learn")
    n = max(1, min(n_components, X.shape[1], X.shape[0]))
    pca = sk_dec.PCA(n_components=n, random_state=0)
    return pca.fit_transform(X), pca


def run_umap(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15,
             min_dist: float = 0.1, seed: int = 0) -> Optional[np.ndarray]:
    """UMAP embedding, or None if umap-learn is not installed."""
    if not umap_available():
        log.info("umap-learn not installed; skipping UMAP embedding "
                 "(install with: pip install umap-learn)")
        return None
    import umap  # type: ignore
    n_neighbors = min(n_neighbors, max(2, X.shape[0] - 1))
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(X)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_kmeans(X: np.ndarray, k: int = 8, seed: int = 0) -> np.ndarray:
    sk_cluster = _require("sklearn.cluster", "scikit-learn")
    k = max(1, min(k, X.shape[0]))
    return sk_cluster.KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def cluster_agglomerative(X: np.ndarray, k: int = 8) -> np.ndarray:
    sk_cluster = _require("sklearn.cluster", "scikit-learn")
    k = max(1, min(k, X.shape[0]))
    return sk_cluster.AgglomerativeClustering(n_clusters=k).fit_predict(X)


# ---------------------------------------------------------------------------
# Nearest-neighbor lookup
# ---------------------------------------------------------------------------


def _par_equal(a, b) -> bool:
    """True only when both pars are present and equal."""
    return bool(pd.notna(a) and pd.notna(b) and a == b)


def _pars(df: pd.DataFrame, same_par: bool):
    if not same_par:
        return None
    if "par" not in df.columns:
        raise ValueError("same_par=True requires a 'par' column in the feature table.")
    return df["par"].to_numpy()


def nearest_neighbor_table(
    df: pd.DataFrame,
    X: np.ndarray,
    k: int = 10,
    exclude_same_course: bool = False,
    same_par: bool = False,
) -> pd.DataFrame:
    """For each hole, the K most similar OTHER holes (Euclidean in scaled space).

    Optional filters (combinable):
      * ``exclude_same_course`` — skip neighbors on the query's course.
      * ``same_par`` — only return neighbors with the same par as the query.

    Returns long-format rows:
        query_hole_id, query_course_slug, query_hole_number,
        similar_hole_id, similar_course_slug, similar_hole_number, distance, rank
    """
    sk_nn = _require("sklearn.neighbors", "scikit-learn")
    n = X.shape[0]
    # Over-fetch when filtering so we can still return K passing neighbors.
    k_query = n if (exclude_same_course or same_par) else min(k + 1, n)
    nn = sk_nn.NearestNeighbors(n_neighbors=k_query, metric="euclidean").fit(X)
    dist, idx = nn.kneighbors(X)

    hid = df["hole_id"].to_numpy()
    slug = df["course_slug"].to_numpy()
    hnum = df["hole_number"].to_numpy()
    par = _pars(df, same_par)

    rows: list[dict] = []
    for i in range(n):
        rank = 0
        for j, d in zip(idx[i], dist[i]):
            if j == i:
                continue
            if exclude_same_course and slug[j] == slug[i]:
                continue
            if par is not None and not _par_equal(par[j], par[i]):
                continue
            rank += 1
            rows.append({
                "query_hole_id": hid[i],
                "query_course_slug": slug[i],
                "query_hole_number": int(hnum[i]),
                "similar_hole_id": hid[j],
                "similar_course_slug": slug[j],
                "similar_hole_number": int(hnum[j]),
                "distance": float(d),
                "rank": rank,
            })
            if rank >= k:
                break
    return pd.DataFrame(rows)


def similar_holes(
    df: pd.DataFrame,
    X: np.ndarray,
    hole_id: str,
    k: int = 10,
    exclude_same_course: bool = False,
    same_par: bool = False,
) -> pd.DataFrame:
    """The K most similar holes to a single ``hole_id``.

    Supports the same ``exclude_same_course`` and ``same_par`` filters as
    :func:`nearest_neighbor_table`.
    """
    sk_nn = _require("sklearn.neighbors", "scikit-learn")
    if hole_id not in set(df["hole_id"]):
        raise KeyError(f"hole_id '{hole_id}' not found in feature table.")
    n = X.shape[0]
    qpos = int(np.flatnonzero(df["hole_id"].to_numpy() == hole_id)[0])
    nn = sk_nn.NearestNeighbors(n_neighbors=n, metric="euclidean").fit(X)
    dist, idx = nn.kneighbors(X[qpos : qpos + 1])

    slug = df["course_slug"].to_numpy()
    hid = df["hole_id"].to_numpy()
    hnum = df["hole_number"].to_numpy()
    par = _pars(df, same_par)
    q_slug = slug[qpos]

    rows: list[dict] = []
    rank = 0
    for j, d in zip(idx[0], dist[0]):
        if j == qpos:
            continue
        if exclude_same_course and slug[j] == q_slug:
            continue
        if par is not None and not _par_equal(par[j], par[qpos]):
            continue
        rank += 1
        rows.append({
            "query_hole_id": hole_id,
            "query_course_slug": q_slug,
            "query_hole_number": int(hnum[qpos]),
            "similar_hole_id": hid[j],
            "similar_course_slug": slug[j],
            "similar_hole_number": int(hnum[j]),
            "distance": float(d),
            "rank": rank,
        })
        if rank >= k:
            break
    return pd.DataFrame(rows)
