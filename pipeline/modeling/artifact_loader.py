"""Load modeling artifacts from local pipeline output *or* a downloaded HF artifact.

The notebook (and any analysis code) should not care whether the feature tables
came from a full local pipeline run (``courses/_index/``) or from a Hugging Face
artifact folder downloaded with ``hf download ... --local-dir``. This module
resolves whichever source is available and loads the same set of tables from it.

Two recognized layouts
----------------------
* **Local index** — the pipeline's ``courses/_index/`` folder: tables live
  directly in the root (``hole_features.parquet`` is right there). The matching
  per-hole compact point clouds live under the sibling ``courses/<slug>/holes/``
  tree, so visual comparisons have full coverage.
* **Hugging Face artifact** — a downloaded artifact root: tables live under
  ``data/`` (``data/hole_features.parquet``), metadata under ``metadata/``, the
  manifest at ``dataset_manifest.json``, and any shipped compact point clouds
  under ``point_clouds/compact/<hole_id>.json`` (the *lite* tier ships only a
  curated subset).

Only the light stack (pandas + stdlib) is used, so this stays importable and
unit-testable without the geo / sklearn toolchain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from ..logging_config import get_logger

log = get_logger("modeling.artifact_loader")

PathLike = Union[str, Path]

# Candidate locations searched when no explicit root is given. Both repo-root-
# relative and notebook-relative ("../") forms are listed so auto-detect works
# whether you run from the repo root or from notebooks/.
_LOCAL_INDEX_CANDIDATES: tuple[str, ...] = ("courses/_index", "../courses/_index")
_ARTIFACT_CANDIDATES: tuple[str, ...] = (
    "golf-data-research-artifacts",
    "../golf-data-research-artifacts",
    "hf_artifact_lite",
    "../hf_artifact_lite",
)

#: Hint printed when nothing can be found, with the exact download command.
DOWNLOAD_HINT: str = (
    "No modeling artifacts found.\n"
    "Either build them locally (full pipeline):\n"
    "    python -m pipeline.modeling all\n"
    "or download the Hugging Face artifact and point the root at it:\n"
    "    hf download davishelman/golf-data-research-artifacts --repo-type dataset \\\n"
    "        --local-dir golf-data-research-artifacts\n"
    "    # then: load_modeling_artifacts('golf-data-research-artifacts')"
)


# --------------------------------------------------------------------------- #
# Classification + resolution
# --------------------------------------------------------------------------- #

def _classify_root(root: Path) -> Optional[str]:
    """Return ``'hf_artifact'``, ``'local_index'``, or ``None`` for ``root``."""
    if (root / "data" / "hole_features.parquet").exists():
        return "hf_artifact"
    if (root / "hole_features.parquet").exists():
        return "local_index"
    return None


def resolve_artifact_root(preferred_root: PathLike | None = None) -> Path:
    """Return a usable artifact root, raising a clear error if none is found.

    If ``preferred_root`` is given it must look like one of the two recognized
    layouts (contain ``data/hole_features.parquet`` or ``hole_features.parquet``);
    otherwise a ``FileNotFoundError`` with download instructions is raised. With
    no argument, local ``courses/_index`` is preferred, then common downloaded
    artifact folders.
    """
    if preferred_root is not None:
        root = Path(preferred_root)
        if _classify_root(root) is None:
            raise FileNotFoundError(
                f"{root} does not look like a modeling artifact root (expected "
                f"data/hole_features.parquet or hole_features.parquet).\n\n"
                + DOWNLOAD_HINT
            )
        return root

    for cand in _LOCAL_INDEX_CANDIDATES + _ARTIFACT_CANDIDATES:
        root = Path(cand)
        if _classify_root(root) is not None:
            return root
    raise FileNotFoundError(DOWNLOAD_HINT)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _maybe_read(path: Path, reader: Callable[[Path], object]) -> Optional[object]:
    """Read ``path`` with ``reader`` if it exists; warn + return None otherwise."""
    if not path.exists():
        return None
    try:
        return reader(path)
    except Exception as exc:  # noqa: BLE001 - any read error is non-fatal here
        log.warning("could not read %s: %s", path, exc)
        return None


def _read_json(path: Path) -> Optional[dict]:
    return _maybe_read(path, lambda p: json.loads(p.read_text(encoding="utf-8")))  # type: ignore[return-value]


def load_modeling_artifacts(root: PathLike | None = None) -> dict:
    """Load the modeling tables (and optional metadata) from a resolved root.

    Returns a dict with:

    * ``source_kind`` — ``'local_index'`` or ``'hf_artifact'``
    * ``root`` / ``data_dir`` — resolved root and where the tables live
    * ``label`` — human-readable source description (print this in the notebook)
    * ``features`` — required ``hole_features`` table (DataFrame)
    * ``clusters`` / ``similarity_v1`` / ``similarity_v2`` — DataFrames or None
    * ``manifest`` / ``schema`` / ``feature_dictionary`` — dicts or None
    * ``similarity_modes`` — ``{mode_name: DataFrame}`` from
      ``data/similarity_modes/*.csv`` (``{}`` if none shipped)
    * ``similarity_modes_dir`` — that folder's Path, or None
    * ``presented_similarity`` — ``{name: DataFrame}`` from
      ``data/presented_similarity/*.csv`` (golfer-facing; ``{}`` if none shipped)
    * ``presented_similarity_dir`` — that folder's Path, or None
    * ``compact_dir`` — HF compact point-cloud dir, or None for local
    * ``courses_root`` — local ``courses/`` tree (for visual checks), or None
    """
    resolved = resolve_artifact_root(root)
    kind = _classify_root(resolved)
    data_dir = resolved / "data" if kind == "hf_artifact" else resolved

    features = pd.read_parquet(data_dir / "hole_features.parquet")
    clusters = _maybe_read(data_dir / "hole_clusters.parquet", pd.read_parquet)
    similarity_v1 = _maybe_read(data_dir / "hole_similarity_examples.csv", pd.read_csv)
    similarity_v2 = _maybe_read(data_dir / "hole_similarity_v2.csv", pd.read_csv)

    manifest = _read_json(resolved / "dataset_manifest.json")
    schema = _read_json(resolved / "metadata" / "schema.json")
    feature_dictionary = _read_json(resolved / "metadata" / "feature_dictionary.json")

    # Optional per-mode similarity CSVs (data/similarity_modes/<mode>.csv). The
    # path is the same relative to data_dir for both layouts. Absent -> {} / None.
    modes_dir = data_dir / "similarity_modes"
    similarity_modes: dict[str, object] = {}
    if modes_dir.exists():
        for csv in sorted(modes_dir.glob("*.csv")):
            dfm = _maybe_read(csv, pd.read_csv)
            if dfm is not None:
                similarity_modes[csv.stem] = dfm
    similarity_modes_dir = modes_dir if modes_dir.exists() else None

    # Optional presented (golf-plausibility-filtered) similarity CSVs. Same shape
    # as similarity_modes; absent -> {} / None.
    presented_dir = data_dir / "presented_similarity"
    presented_similarity: dict[str, object] = {}
    if presented_dir.exists():
        for csv in sorted(presented_dir.glob("*.csv")):
            dfp = _maybe_read(csv, pd.read_csv)
            if dfp is not None:
                presented_similarity[csv.stem] = dfp
    presented_similarity_dir = presented_dir if presented_dir.exists() else None

    if kind == "hf_artifact":
        compact = resolved / "point_clouds" / "compact"
        compact_dir = compact if compact.exists() else None
        courses_root = None
        label = f"Hugging Face artifact at {resolved}"
    else:
        compact_dir = None
        courses_root = resolved.parent  # courses/_index -> courses/
        label = f"local {resolved.as_posix()}"

    return {
        "source_kind": kind,
        "root": resolved,
        "data_dir": data_dir,
        "label": label,
        "features": features,
        "clusters": clusters,
        "similarity_v1": similarity_v1,
        "similarity_v2": similarity_v2,
        "manifest": manifest,
        "schema": schema,
        "feature_dictionary": feature_dictionary,
        "similarity_modes": similarity_modes,          # {mode_name: DataFrame}, may be {}
        "similarity_modes_dir": similarity_modes_dir,  # Path or None
        "presented_similarity": presented_similarity,          # {name: DataFrame}, may be {}
        "presented_similarity_dir": presented_similarity_dir,  # Path or None
        "compact_dir": compact_dir,
        "courses_root": courses_root,
    }


__all__ = [
    "resolve_artifact_root",
    "load_modeling_artifacts",
    "DOWNLOAD_HINT",
]
