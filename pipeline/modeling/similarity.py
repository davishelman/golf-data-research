"""Similarity + clustering primitives for hole feature rows.

Pure, composable functions over a feature DataFrame. Identifiers (see
``pipeline.modeling.ID_COLUMNS``) and any non-numeric columns are never scaled or
fed to the model. Optional dependencies (UMAP) degrade gracefully; required ones
(scikit-learn) raise a clear install message.

v1 / v2 / golf modes
--------------------
* **v1** (defaults everywhere): median-impute → ``StandardScaler`` → unweighted
  Euclidean nearest neighbors. Backward compatible.
* **v2 (length-aware)**: optional **feature weighting** (after standardization)
  and an optional **length guard** that drops candidates whose length differs
  from the query by more than a threshold.
* **Golf modes** (``SIMILARITY_MODES`` / ``GOLF_MODES``): each mode selects a
  *subset* of features (or all), with weights + default filters, to answer a
  specific golf question — "similar off the tee", "similar approach", "similar
  green complex", "similar hazards", "similar terrain", "similar shot shape", and
  the balanced ``overall_v2``.

Missing-value policy
--------------------
Each feature column is **median-imputed**, then standardized. All-missing columns
are kept (imputed to 0); constant columns scale to 0 (no NaNs leak in).
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
# v2 presets: feature weights + length-guard defaults
# ---------------------------------------------------------------------------

# Length-aware weighting. Weights scale each standardized feature column, so a
# weight of w multiplies that feature's contribution to Euclidean distance by w
# (and to squared distance by w^2). ``hole_length_yd`` is forced to 0 so length
# is not double-counted when both metres and yards are present as features.
LENGTH_AWARE_WEIGHTS: dict[str, float] = {
    "hole_length_m": 4.0,
    "hole_length_yd": 0.0,
    "hole_depth_m": 2.0,
    "green_y_m": 3.0,
    "par": 2.0,
    "fairway_width_drive_zone": 1.5,
    "fairway_width_approach_zone": 1.5,
    "tee_to_green_elevation_change": 1.5,
    "rough_pct": 0.5,
    "rough_inferred_pct": 0.5,
}

# Recommended length-guard defaults (a 400 m hole allows ~48 m, not ~100 m).
DEFAULT_MAX_LENGTH_DIFF_M: float = 35.0
DEFAULT_MAX_LENGTH_DIFF_PCT: float = 0.12


# ---------------------------------------------------------------------------
# Golf-mode feature subsets
# ---------------------------------------------------------------------------

_OFF_THE_TEE_COLS = (
    "hole_length_m", "par",
    "drive_zone_fairway_pct", "drive_zone_trees_pct",
    "drive_zone_bunker_pct", "drive_zone_water_pct",
    "drive_trees_left_pct", "drive_trees_right_pct",
    "drive_bunker_left_pct", "drive_bunker_right_pct",
    "drive_water_left_pct", "drive_water_right_pct",
    "fairway_width_drive_zone", "dogleg_score", "fairway_centerline_shift",
)
_APPROACH_COLS = (
    "approach_zone_fairway_pct", "approach_zone_trees_pct",
    "approach_zone_bunker_pct", "approach_zone_water_pct",
    "approach_trees_left_pct", "approach_trees_right_pct",
    "approach_bunker_left_pct", "approach_bunker_right_pct",
    "approach_water_left_pct", "approach_water_right_pct",
    "fairway_width_approach_zone", "tee_to_green_elevation_change",
    "green_relative_elevation",
)
_GREEN_COMPLEX_COLS = (
    "green_pct", "green_complex_bunker_pct", "green_complex_water_pct",
    "green_complex_trees_pct", "green_complex_mean_z", "green_complex_z_range",
    "green_relative_elevation", "tee_to_green_elevation_change",
)
_HAZARD_COLS = (
    "bunker_pct", "water_pct", "trees_pct", "sand_pct",
    "drive_zone_bunker_pct", "drive_zone_water_pct", "drive_zone_trees_pct",
    "approach_zone_bunker_pct", "approach_zone_water_pct", "approach_zone_trees_pct",
    "green_complex_bunker_pct", "green_complex_water_pct", "green_complex_trees_pct",
    "drive_trees_left_pct", "drive_trees_right_pct",
    "drive_bunker_left_pct", "drive_bunker_right_pct",
    "drive_water_left_pct", "drive_water_right_pct",
    "approach_trees_left_pct", "approach_trees_right_pct",
    "approach_bunker_left_pct", "approach_bunker_right_pct",
    "approach_water_left_pct", "approach_water_right_pct",
)
_TERRAIN_COLS = (
    "z_min", "z_max", "z_mean", "z_std", "z_range", "z_p10", "z_p50", "z_p90",
    "tee_to_green_elevation_change", "green_relative_elevation",
    "tee_zone_mean_z", "tee_zone_z_range", "drive_zone_mean_z", "drive_zone_z_range",
    "approach_zone_mean_z", "approach_zone_z_range",
    "green_complex_mean_z", "green_complex_z_range",
)
_SHOT_SHAPE_COLS = (
    "dogleg_score", "fairway_centerline_shift", "hole_width_m", "hole_depth_m",
    "fairway_width_drive_zone", "fairway_width_approach_zone",
    "x_min", "x_max", "y_min", "y_max",
)


def _mode(feature_columns, feature_weights, description, *, same_par=False,
          exclude_same_course=False, max_length_diff_m=None, max_length_diff_pct=None):
    return {
        "feature_columns": feature_columns,
        "feature_weights": feature_weights,
        "description": description,
        "same_par": same_par,
        "exclude_same_course": exclude_same_course,
        "max_length_diff_m": max_length_diff_m,
        "max_length_diff_pct": max_length_diff_pct,
    }


# Each mode is a plain dict (backward compatible with v1/v2 dict access).
# ``feature_columns=None`` means "use all numeric features".
SIMILARITY_MODES: dict[str, dict] = {
    # --- legacy v1 / v2 modes (unchanged behavior) ---
    "v1": _mode(None, None, "v1: all features, unweighted, no filters."),
    "length_weighted": _mode(
        None, LENGTH_AWARE_WEIGHTS,
        "All features with length-aware weights; no hard length filter."),
    "same_par_length_guarded": _mode(
        None, LENGTH_AWARE_WEIGHTS,
        "Length-aware weights, same par, length guard.",
        same_par=True, max_length_diff_m=DEFAULT_MAX_LENGTH_DIFF_M,
        max_length_diff_pct=DEFAULT_MAX_LENGTH_DIFF_PCT),
    "cross_course_same_par_length_guarded": _mode(
        None, LENGTH_AWARE_WEIGHTS,
        "Cross-course, same par, length-guarded, length-aware weights (v2 export).",
        same_par=True, exclude_same_course=True,
        max_length_diff_m=DEFAULT_MAX_LENGTH_DIFF_M,
        max_length_diff_pct=DEFAULT_MAX_LENGTH_DIFF_PCT),

    # --- golf-specific modes ---
    "overall_v2": _mode(
        None, LENGTH_AWARE_WEIGHTS,
        "Balanced overall similarity: all features, length-aware weights, same "
        "par, length guard (cross-course optional).",
        same_par=True, max_length_diff_m=DEFAULT_MAX_LENGTH_DIFF_M,
        max_length_diff_pct=DEFAULT_MAX_LENGTH_DIFF_PCT),
    "off_the_tee": _mode(
        _OFF_THE_TEE_COLS, {},
        "How the hole plays from the tee through the drive zone: length, "
        "drive-zone fairway/hazards, left/right pressure, width, dogleg/shape."),
    "approach": _mode(
        _APPROACH_COLS, {},
        "The approach-shot portion: approach-zone fairway/hazards, left/right "
        "pressure, approach width, and elevation into the green."),
    "green_complex": _mode(
        _GREEN_COMPLEX_COLS, {},
        "Green surroundings and defenses: green share, green-complex hazards, "
        "and green-complex / tee-to-green elevation."),
    "hazard": _mode(
        _HAZARD_COLS, {},
        "Penalty/defense pattern: overall + per-zone bunker/water/trees/sand and "
        "left/right hazard pressure."),
    "terrain": _mode(
        _TERRAIN_COLS, {},
        "Elevation/terrain profile: z statistics, tee-to-green change, and "
        "per-zone mean_z / z_range."),
    "shot_shape": _mode(
        _SHOT_SHAPE_COLS, {},
        "Hole bend / corridor geometry: dogleg, centerline shift, width/depth, "
        "fairway widths, and the spatial bounding box."),
}

# The domain-specific golf modes, in a stable order (used by the mode exporter).
GOLF_MODES: tuple[str, ...] = (
    "overall_v2", "off_the_tee", "approach", "green_complex",
    "hazard", "terrain", "shot_shape",
)


def resolve_mode(mode: str) -> dict:
    """Return a copy of a named mode's config dict."""
    if mode not in SIMILARITY_MODES:
        raise KeyError(f"unknown similarity mode '{mode}'. "
                       f"Choices: {sorted(SIMILARITY_MODES)}")
    return dict(SIMILARITY_MODES[mode])


def mode_description(mode: str) -> str:
    return resolve_mode(mode).get("description", "")


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


def available_columns(df: pd.DataFrame, desired) -> list[str]:
    """Subset of ``desired`` that exists, is numeric, and is not an identifier."""
    drop = set(ID_COLUMNS)
    return [
        c for c in desired
        if c in df.columns and c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]


def missing_mode_columns(df: pd.DataFrame, mode: str) -> list[str]:
    """Desired columns for ``mode`` that are absent from ``df`` ([] for 'all features')."""
    desired = resolve_mode(mode)["feature_columns"]
    if desired is None:
        return []
    have = set(available_columns(df, desired))
    return [c for c in desired if c not in have]


def feature_columns_for_mode(df: pd.DataFrame, mode: str) -> list[str]:
    """Usable feature columns for ``mode``.

    ``None`` feature set -> all numeric features. A subset mode returns its
    available columns; if a subset mode has **no** usable columns it raises a
    clear error (the mode is unusable for this table). Missing-but-not-fatal
    columns are available via :func:`missing_mode_columns`.
    """
    desired = resolve_mode(mode)["feature_columns"]
    if desired is None:
        return feature_columns(df)
    present = available_columns(df, desired)
    if not present:
        raise ValueError(
            f"similarity mode '{mode}' has no usable feature columns in this "
            f"table (wanted: {list(desired)})."
        )
    missing = [c for c in desired if c not in set(present)]
    if missing:
        log.warning("mode '%s' missing %d/%d columns: %s",
                    mode, len(missing), len(desired), missing)
    return present


def feature_summary(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Per-feature inspection table: dtype + missing counts, sorted by missingness."""
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


def build_feature_matrix(
    df: pd.DataFrame,
    cols: list[str],
    feature_weights: Optional[dict[str, float]] = None,
):
    """Median-impute, standardize, and (optionally) weight. Returns ``(X, imputer, scaler)``.

    ``feature_weights`` maps a column name to a multiplier applied **after**
    standardization (unlisted columns default to weight 1.0). The raw feature
    table is never mutated. With ``feature_weights=None`` the output is v1.
    """
    sk_impute = _require("sklearn.impute", "scikit-learn")
    sk_pre = _require("sklearn.preprocessing", "scikit-learn")
    X = df[cols].to_numpy(dtype="float64")
    imputer = sk_impute.SimpleImputer(strategy="median", keep_empty_features=True)
    Xi = imputer.fit_transform(X)
    scaler = sk_pre.StandardScaler()
    Xs = scaler.fit_transform(Xi)
    if feature_weights:
        w = np.array([float(feature_weights.get(c, 1.0)) for c in cols], dtype="float64")
        Xs = Xs * w
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
# Length guard + par helpers
# ---------------------------------------------------------------------------


def _allowed_length_diff(query_len: float, max_m: Optional[float],
                         max_pct: Optional[float]) -> Optional[float]:
    """Permitted absolute length difference; the more permissive threshold wins."""
    vals: list[float] = []
    if max_m is not None:
        vals.append(float(max_m))
    if max_pct is not None and pd.notna(query_len):
        vals.append(float(query_len) * float(max_pct))
    return max(vals) if vals else None


def _lengths(df: pd.DataFrame, active: bool) -> Optional[np.ndarray]:
    if not active:
        return None
    if "hole_length_m" not in df.columns:
        raise ValueError("length guard requires a 'hole_length_m' column.")
    return df["hole_length_m"].to_numpy()


def _length_ok(lengths, i: int, j: int, max_m, max_pct) -> bool:
    if lengths is None:
        return True
    Li, Lj = lengths[i], lengths[j]
    if pd.isna(Li) or pd.isna(Lj):
        return True
    allowed = _allowed_length_diff(Li, max_m, max_pct)
    return allowed is None or abs(Li - Lj) <= allowed


def _par_equal(a, b) -> bool:
    return bool(pd.notna(a) and pd.notna(b) and a == b)


def _pars(df: pd.DataFrame, same_par: bool):
    if not same_par:
        return None
    if "par" not in df.columns:
        raise ValueError("same_par=True requires a 'par' column in the feature table.")
    return df["par"].to_numpy()


# ---------------------------------------------------------------------------
# Nearest-neighbor lookup
# ---------------------------------------------------------------------------


def nearest_neighbor_table(
    df: pd.DataFrame,
    X: np.ndarray,
    k: int = 10,
    exclude_same_course: bool = False,
    same_par: bool = False,
    max_length_diff_m: Optional[float] = None,
    max_length_diff_pct: Optional[float] = None,
) -> pd.DataFrame:
    """For each hole, the K most similar OTHER holes (Euclidean in scaled space).

    Optional filters (all combinable; defaults reproduce v1):
      * ``exclude_same_course`` — skip neighbors on the query's course.
      * ``same_par`` — only neighbors with the same par as the query.
      * ``max_length_diff_m`` / ``max_length_diff_pct`` — length guard.
    """
    sk_nn = _require("sklearn.neighbors", "scikit-learn")
    n = X.shape[0]
    length_guard = max_length_diff_m is not None or max_length_diff_pct is not None
    over_fetch = exclude_same_course or same_par or length_guard
    k_query = n if over_fetch else min(k + 1, n)
    nn = sk_nn.NearestNeighbors(n_neighbors=k_query, metric="euclidean").fit(X)
    dist, idx = nn.kneighbors(X)

    hid = df["hole_id"].to_numpy()
    slug = df["course_slug"].to_numpy()
    hnum = df["hole_number"].to_numpy()
    par = _pars(df, same_par)
    lengths = _lengths(df, length_guard)

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
            if not _length_ok(lengths, i, j, max_length_diff_m, max_length_diff_pct):
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
    max_length_diff_m: Optional[float] = None,
    max_length_diff_pct: Optional[float] = None,
) -> pd.DataFrame:
    """The K most similar holes to a single ``hole_id`` (same filters as the table)."""
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
    length_guard = max_length_diff_m is not None or max_length_diff_pct is not None
    lengths = _lengths(df, length_guard)
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
        if not _length_ok(lengths, qpos, j, max_length_diff_m, max_length_diff_pct):
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


# ---------------------------------------------------------------------------
# Mode convenience wrappers (build the mode's matrix + apply the mode's filters)
# ---------------------------------------------------------------------------


def matrix_for_mode(df: pd.DataFrame, mode: str = "v1"):
    """Build the standardized (and weighted) matrix for ``mode``.

    Returns ``(X, cols, cfg)`` where ``cols`` is the mode's resolved feature list.
    """
    cfg = resolve_mode(mode)
    cols = feature_columns_for_mode(df, mode)
    X, _, _ = build_feature_matrix(df, cols, feature_weights=cfg["feature_weights"])
    return X, cols, cfg


def nearest_neighbor_table_mode(
    df: pd.DataFrame, cols: Optional[list[str]], mode: str = "v1", k: int = 10,
    exclude_same_course: Optional[bool] = None,
) -> pd.DataFrame:
    """Neighbor table for a named mode (``cols`` retained for API compatibility;
    the mode's own feature subset is used). ``exclude_same_course`` overrides the
    mode default when provided."""
    X, _cols, cfg = matrix_for_mode(df, mode)
    esc = cfg["exclude_same_course"] if exclude_same_course is None else exclude_same_course
    return nearest_neighbor_table(
        df, X, k=k, exclude_same_course=esc, same_par=cfg["same_par"],
        max_length_diff_m=cfg["max_length_diff_m"],
        max_length_diff_pct=cfg["max_length_diff_pct"],
    )


def similar_holes_mode(
    df: pd.DataFrame, cols: Optional[list[str]], hole_id: str, mode: str = "v1",
    k: int = 10, exclude_same_course: Optional[bool] = None,
) -> pd.DataFrame:
    """K most similar holes to ``hole_id`` under a named mode.

    ``cols`` is retained for backward compatibility; the mode determines the
    actual feature subset. ``exclude_same_course`` overrides the mode default.
    """
    X, _cols, cfg = matrix_for_mode(df, mode)
    esc = cfg["exclude_same_course"] if exclude_same_course is None else exclude_same_course
    return similar_holes(
        df, X, hole_id, k=k, exclude_same_course=esc, same_par=cfg["same_par"],
        max_length_diff_m=cfg["max_length_diff_m"],
        max_length_diff_pct=cfg["max_length_diff_pct"],
    )
