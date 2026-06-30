"""Demo data layer for surfacing v2.5 point-cloud similarity in an app.

Pure, Streamlit-free functions that read the v2.5 batch outputs and return small
tables a UI can render. Keeping this logic here — mirroring how
``pipeline.modeling.demo_utils`` backs the existing Streamlit app — means the
v2.5 view is unit-testable without importing streamlit and without breaking the
v2 views.

Two layouts are supported transparently (see :func:`resolve_pointcloud_dir`):

* **Hugging Face bundle** — ``<root>/data/pointcloud_similarity/<config>/`` (what
  :mod:`pipeline.modeling.pointcloud.artifact_export` writes).
* **Local index** — ``<root>/pointcloud_similarity/<config>/`` (i.e. point the
  root at ``courses/_index`` to read the batch outputs in place, no export step).

Wiring into the Streamlit ``app.py`` is a thin addition, e.g.::

    from pipeline.modeling.pointcloud import demo as pcdemo
    root = pcdemo.discover_pointcloud_root([artifact_root, "courses/_index"])
    if root is not None:
        configs = pcdemo.list_pointcloud_configs(root)
        cfg = st.selectbox("v2.5 config", configs)
        results = pcdemo.load_pointcloud_results(root)[cfg]
        st.dataframe(pcdemo.top_matches_for_hole(results, query_pc_id, top_n))
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from ...logging_config import get_logger
from .export_similarity import RESULTS_FILENAME
from .schemas import parse_pc_hole_id
from .validate_similarity import VALIDATION_DIRNAME, top_matches_for_target

log = get_logger("modeling.pointcloud.demo")

PathLike = Union[str, Path]

#: Where an HF bundle stores v2.5 results (matches artifact_export.ARTIFACT_SUBPATH).
POINTCLOUD_SUBPATH = "data/pointcloud_similarity"

#: Relative locations checked under a root, most-specific (HF bundle) first.
_CANDIDATE_SUBPATHS: tuple[str, ...] = ("data/pointcloud_similarity", "pointcloud_similarity")

#: Display columns for a UI table (a friendly subset of the full results).
DISPLAY_COLUMNS: tuple[str, ...] = (
    "rank", "candidate_hole_id", "total_score",
    "fairway_score", "green_score", "bunker_score", "water_score", "tee_score",
    "yardage_penalty", "elevation_penalty", "missing_surface_penalty",
)


# --------------------------------------------------------------------------- #
# Location resolution
# --------------------------------------------------------------------------- #

def pointcloud_dir(artifact_root: PathLike) -> Path:
    """The HF-bundle ``data/pointcloud_similarity`` dir under an artifact root.

    Kept for backward compatibility; prefer :func:`resolve_pointcloud_dir`, which
    also recognizes the local-index layout.
    """
    return Path(artifact_root) / POINTCLOUD_SUBPATH


def _has_configs(directory: Path) -> bool:
    return directory.is_dir() and any(
        d.is_dir() and d.name != VALIDATION_DIRNAME and (d / RESULTS_FILENAME).exists()
        for d in directory.iterdir()
    )


def resolve_pointcloud_dir(root: PathLike) -> Optional[Path]:
    """First v2.5 results dir under ``root`` that actually contains configs.

    Checks ``<root>/data/pointcloud_similarity`` then ``<root>/pointcloud_similarity``;
    also accepts ``root`` itself already being a ``pointcloud_similarity`` dir.
    Returns ``None`` if no config outputs are found.
    """
    root = Path(root)
    if _has_configs(root):
        return root
    for sub in _CANDIDATE_SUBPATHS:
        cand = root / sub
        if _has_configs(cand):
            return cand
    return None


def discover_pointcloud_root(roots: Iterable[PathLike]) -> Optional[Path]:
    """First resolvable v2.5 results dir across several candidate roots."""
    for r in roots:
        if r is None:
            continue
        resolved = resolve_pointcloud_dir(r)
        if resolved is not None:
            return resolved
    return None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def list_pointcloud_configs(root: PathLike) -> list[str]:
    """Config names available under ``root`` (sorted), or ``[]`` if none.

    ``root`` may be an artifact root, a local index, or a resolved
    ``pointcloud_similarity`` dir.
    """
    resolved = resolve_pointcloud_dir(root)
    if resolved is None:
        return []
    return sorted(
        d.name for d in resolved.iterdir()
        if d.is_dir() and d.name != VALIDATION_DIRNAME and (d / RESULTS_FILENAME).exists()
    )


def load_pointcloud_results(root: PathLike) -> dict[str, pd.DataFrame]:
    """Load ``{config_name: results_df}`` for every v2.5 config under ``root``."""
    resolved = resolve_pointcloud_dir(root)
    if resolved is None:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for name in list_pointcloud_configs(root):
        out[name] = pd.read_csv(resolved / name / RESULTS_FILENAME)
    return out


# --------------------------------------------------------------------------- #
# Per-hole views
# --------------------------------------------------------------------------- #

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


def feature_id_for_pc_hole(pc_hole_id: str) -> str:
    """Map a v2.5 id (``slug:hole_number``) to the v2 feature/compact id (``slug__NN``).

    The v2 tables and compact point clouds use a zero-padded, double-underscore id
    (``augusta_national__01``); v2.5 uses ``augusta_national:1``. This lets the UI
    cross-reference a v2.5 match back to v2 metadata / point-cloud visuals.
    """
    slug, number = parse_pc_hole_id(pc_hole_id)
    return f"{slug}__{number:02d}"


def pointcloud_summary(root: PathLike) -> dict:
    """Headline counts for a dataset-summary panel."""
    results = load_pointcloud_results(root)
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
    "resolve_pointcloud_dir",
    "discover_pointcloud_root",
    "list_pointcloud_configs",
    "load_pointcloud_results",
    "top_matches_for_hole",
    "available_target_holes",
    "feature_id_for_pc_hole",
    "pointcloud_summary",
]
