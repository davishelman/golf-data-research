"""Demo data layer for surfacing v2.5 point-cloud similarity in an app.

Pure, Streamlit-free functions that read the v2.5 outputs an artifact bundle
ships under ``data/pointcloud_similarity/<config_name>/`` (see
:mod:`pipeline.modeling.pointcloud.artifact_export`) and return small tables a UI
can render. Keeping this logic here — mirroring how ``pipeline.modeling.demo_utils``
backs the existing Streamlit app — means the v2.5 view is unit-testable without
importing streamlit and without breaking the v2 views.

Wiring into the Streamlit ``app.py`` is a thin addition, e.g.::

    from pipeline.modeling.pointcloud import demo as pcdemo
    configs = pcdemo.list_pointcloud_configs(root)
    if configs:
        cfg = st.selectbox("v2.5 config", configs)
        results = pcdemo.load_pointcloud_results(root)[cfg]
        st.dataframe(pcdemo.top_matches_for_hole(results, query_id, top_n))
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ...logging_config import get_logger
from .export_similarity import RESULTS_FILENAME
from .validate_similarity import TOP_MATCHES_COLUMNS, top_matches_for_target

log = get_logger("modeling.pointcloud.demo")

PathLike = Union[str, Path]

#: Where the artifact bundle stores v2.5 results (matches artifact_export).
POINTCLOUD_SUBPATH = "data/pointcloud_similarity"

#: Display columns for a UI table (a friendly subset of the full results).
DISPLAY_COLUMNS: tuple[str, ...] = (
    "rank", "candidate_hole_id", "total_score",
    "fairway_score", "green_score", "bunker_score", "water_score", "tee_score",
    "yardage_penalty", "elevation_penalty", "missing_surface_penalty",
)


def pointcloud_dir(artifact_root: PathLike) -> Path:
    """The ``data/pointcloud_similarity`` dir under an artifact root."""
    return Path(artifact_root) / POINTCLOUD_SUBPATH


def list_pointcloud_configs(artifact_root: PathLike) -> list[str]:
    """Config names available in the artifact (sorted), or ``[]`` if none."""
    root = pointcloud_dir(artifact_root)
    if not root.exists():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / RESULTS_FILENAME).exists()
    )


def load_pointcloud_results(artifact_root: PathLike) -> dict[str, pd.DataFrame]:
    """Load ``{config_name: results_df}`` for every v2.5 config in the artifact."""
    root = pointcloud_dir(artifact_root)
    out: dict[str, pd.DataFrame] = {}
    for name in list_pointcloud_configs(artifact_root):
        out[name] = pd.read_csv(root / name / RESULTS_FILENAME)
    return out


def top_matches_for_hole(
    results: pd.DataFrame, target_hole_id: str, top_n: int = 10,
    *, config_name: str = "",
) -> pd.DataFrame:
    """The best ``top_n`` matches for ``target_hole_id`` (display columns).

    Thin wrapper over :func:`validate_similarity.top_matches_for_target` that
    selects the UI-friendly :data:`DISPLAY_COLUMNS`. Returns an empty (typed)
    frame when the hole is absent.
    """
    full = top_matches_for_target(results, target_hole_id, top_n, config_name)
    if full.empty:
        return pd.DataFrame(columns=list(DISPLAY_COLUMNS))
    return full[[c for c in DISPLAY_COLUMNS if c in full.columns]].reset_index(drop=True)


def available_target_holes(results: pd.DataFrame) -> list[str]:
    """Sorted unique target hole ids present in a results table."""
    if "target_hole_id" not in results.columns:
        return []
    return sorted(results["target_hole_id"].astype(str).unique())


def pointcloud_summary(artifact_root: PathLike) -> dict:
    """Headline counts for a dataset-summary panel."""
    results = load_pointcloud_results(artifact_root)
    return {
        "configs": list(results),
        "n_configs": len(results),
        "rows_by_config": {name: int(len(df)) for name, df in results.items()},
        "targets_by_config": {
            name: int(df["target_hole_id"].nunique()) if "target_hole_id" in df else 0
            for name, df in results.items()
        },
    }


__all__ = [
    "POINTCLOUD_SUBPATH",
    "DISPLAY_COLUMNS",
    "pointcloud_dir",
    "list_pointcloud_configs",
    "load_pointcloud_results",
    "top_matches_for_hole",
    "available_target_holes",
    "pointcloud_summary",
]
