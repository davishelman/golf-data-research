"""Artifact export/load for player-course advantage outputs (issue #44).

Persists a scoring/ranking run in a small, self-describing layout so notebooks,
Streamlit, and demos can consume results without rerunning the model::

    data/player_course_advantage/<run_id>/
        player_rankings.csv       # one row per player (batch ranking)
        player_hole_details.csv   # one row per (player, target hole)
        diagnostics.csv           # per (player, target hole, similar hole) contributions
        backtest_summary.csv      # optional — written only when #35 provides it
        parameters.json           # AdvantageParams + aggregate mode
        manifest.json             # run metadata: ids, counts, model version, timestamps

Only **ids, scores, and diagnostics** are written — never raw point-cloud
geometry (enforced by :func:`_reject_geometry`). Generated runs live under
``data/player_course_advantage/`` and are gitignored; the only committed
artifacts are the tiny synthetic fixtures a test builds in a temp dir.

Pure IO + light orchestration over the merged scorer / batch / diagnostics
layers; Streamlit-free and free of real-data dependencies.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from .batch import score_tournament_field
from .diagnostics import explain_player_course
from .schema import DEFAULT_PARAMS, AdvantageParams

PathLike = Union[str, Path]

#: Default root for generated runs (gitignored).
ARTIFACT_SUBDIR = "data/player_course_advantage"

RANKINGS_FILE = "player_rankings.csv"
HOLE_DETAILS_FILE = "player_hole_details.csv"
DIAGNOSTICS_FILE = "diagnostics.csv"
BACKTEST_SUMMARY_FILE = "backtest_summary.csv"
PARAMETERS_FILE = "parameters.json"
MANIFEST_FILE = "manifest.json"

# Column-name fragments that would indicate raw geometry sneaking into an export.
_GEOMETRY_MARKERS: tuple[str, ...] = (
    "point_cloud", "pointcloud", "vertices", "xyz", "mesh", "dem_", "raster",
)


class ArtifactExportError(ValueError):
    """Raised when a run cannot be exported or loaded (bad frame, missing file)."""


@dataclass(frozen=True)
class AdvantageRun:
    """An exported (or freshly loaded) advantage run: metadata + its frames."""

    run_dir: Path
    manifest: dict
    parameters: dict
    rankings: pd.DataFrame
    hole_details: pd.DataFrame
    diagnostics: pd.DataFrame
    backtest_summary: Optional[pd.DataFrame] = None


def default_run_root() -> Path:
    """The conventional (gitignored) root for generated runs."""
    return Path(ARTIFACT_SUBDIR)


def make_run_id(prefix: str = "run") -> str:
    """A sortable, unique run id: ``<prefix>-<UTC yyyymmddThhmmss>-<short hex>``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def _reject_geometry(name: str, df: pd.DataFrame) -> None:
    bad = [
        c for c in df.columns
        if any(marker in str(c).lower() for marker in _GEOMETRY_MARKERS)
    ]
    if bad:
        raise ArtifactExportError(
            f"refusing to export {name}: looks like raw geometry columns {bad}; "
            "artifacts hold ids/scores/diagnostics only"
        )


# --------------------------------------------------------------------------- #
# Assemble a field's outputs from the merged layers
# --------------------------------------------------------------------------- #
def assemble_field_outputs(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    field: Union[pd.DataFrame, Sequence[str]],
    target_course_slug: str,
    predict_season: int,
    config_name: str = "baseline",
    params: AdvantageParams = DEFAULT_PARAMS,
    aggregate: str = "sum",
) -> dict[str, pd.DataFrame]:
    """Run ranking + per-player hole details + diagnostics for a whole field.

    Returns ``{"rankings", "hole_details", "diagnostics"}``. ``hole_details`` is
    the per-(player, target hole) advantage/coverage detail; ``diagnostics`` is the
    per-(player, target hole, similar hole) contribution breakdown from #39. Rows
    follow the ranking order (then hole number). Never mutates the inputs.
    """
    rankings = score_tournament_field(
        history, similar_holes, field, target_course_slug, predict_season,
        config_name=config_name, params=params, aggregate=aggregate,
    )

    hole_frames: list[pd.DataFrame] = []
    diag_frames: list[pd.DataFrame] = []
    for pid in rankings["player_id"].tolist():
        expl = explain_player_course(
            history, similar_holes, pid, target_course_slug, predict_season,
            config_name=config_name, params=params, aggregate=aggregate,
        )
        hole_frames.append(expl.hole_contributions)  # carries player_id already
        sc = expl.similar_hole_contributions.copy()
        sc.insert(0, "player_id", pid)
        diag_frames.append(sc)

    hole_details = (
        pd.concat(hole_frames, ignore_index=True) if hole_frames else pd.DataFrame()
    )
    diagnostics = (
        pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    )
    return {"rankings": rankings, "hole_details": hole_details, "diagnostics": diagnostics}


# --------------------------------------------------------------------------- #
# Export / load
# --------------------------------------------------------------------------- #
def export_advantage_run(
    rankings: pd.DataFrame,
    *,
    target_course_slug: str,
    config_name: str,
    predict_season: int,
    params: AdvantageParams,
    hole_details: Optional[pd.DataFrame] = None,
    diagnostics: Optional[pd.DataFrame] = None,
    backtest_summary: Optional[pd.DataFrame] = None,
    aggregate: str = "sum",
    root: PathLike = ARTIFACT_SUBDIR,
    run_id: Optional[str] = None,
    input_counts: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> AdvantageRun:
    """Write a run to ``<root>/<run_id>/`` and return the resulting :class:`AdvantageRun`.

    ``rankings`` is required; ``hole_details`` / ``diagnostics`` / ``backtest_summary``
    are written only when supplied (``backtest_summary`` lands whenever #35 hands it
    over — the layout is ready for it now). Refuses frames that carry raw geometry
    columns. Reproducibility metadata (model version, timestamp, params, input/output
    row counts) goes into ``manifest.json`` + ``parameters.json``.
    """
    from . import MODEL_VERSION  # lazy: avoid a package-import cycle

    run_id = run_id or make_run_id()
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, int] = {}

    def _write(name: str, df: Optional[pd.DataFrame]) -> None:
        if df is None:
            return
        _reject_geometry(name, df)
        df.to_csv(run_dir / name, index=False)
        files[name] = int(len(df))

    _write(RANKINGS_FILE, rankings)
    _write(HOLE_DETAILS_FILE, hole_details)
    _write(DIAGNOSTICS_FILE, diagnostics)
    _write(BACKTEST_SUMMARY_FILE, backtest_summary)

    parameters = {"aggregate": aggregate, **asdict(params)}
    (run_dir / PARAMETERS_FILE).write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "kind": "player_course_advantage.run",
        "target_course_slug": target_course_slug,
        "config_name": config_name,
        "predict_season": predict_season,
        "aggregate": aggregate,
        "files": files,
        "input_counts": input_counts or {},
        "output_counts": {
            "n_players": int(len(rankings)),
            "n_covered": int((~rankings["low_coverage"]).sum()) if len(rankings) else 0,
        },
        "has_backtest": BACKTEST_SUMMARY_FILE in files,
        "parameters_file": PARAMETERS_FILE,
    }
    if extra_meta:
        manifest["extra"] = extra_meta
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AdvantageRun(
        run_dir=run_dir,
        manifest=manifest,
        parameters=parameters,
        rankings=rankings,
        hole_details=hole_details if hole_details is not None else pd.DataFrame(),
        diagnostics=diagnostics if diagnostics is not None else pd.DataFrame(),
        backtest_summary=backtest_summary,
    )


def load_advantage_run(run_dir: PathLike) -> AdvantageRun:
    """Load a run written by :func:`export_advantage_run`.

    Raises :class:`ArtifactExportError` if the directory or its manifest/rankings
    are missing. Optional files (hole details, diagnostics, backtest summary) load
    as empty / ``None`` when absent.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise ArtifactExportError(f"no {MANIFEST_FILE} in {run_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params_path = run_dir / PARAMETERS_FILE
    parameters = (
        json.loads(params_path.read_text(encoding="utf-8"))
        if params_path.exists() else {}
    )

    def _read(name: str) -> Optional[pd.DataFrame]:
        path = run_dir / name
        return pd.read_csv(path) if path.exists() else None

    rankings = _read(RANKINGS_FILE)
    if rankings is None:
        raise ArtifactExportError(f"no {RANKINGS_FILE} in {run_dir}")

    hole_details = _read(HOLE_DETAILS_FILE)
    diagnostics = _read(DIAGNOSTICS_FILE)
    return AdvantageRun(
        run_dir=run_dir,
        manifest=manifest,
        parameters=parameters,
        rankings=rankings,
        hole_details=hole_details if hole_details is not None else pd.DataFrame(),
        diagnostics=diagnostics if diagnostics is not None else pd.DataFrame(),
        backtest_summary=_read(BACKTEST_SUMMARY_FILE),
    )


__all__ = [
    "ARTIFACT_SUBDIR",
    "RANKINGS_FILE",
    "HOLE_DETAILS_FILE",
    "DIAGNOSTICS_FILE",
    "BACKTEST_SUMMARY_FILE",
    "PARAMETERS_FILE",
    "MANIFEST_FILE",
    "ArtifactExportError",
    "AdvantageRun",
    "default_run_root",
    "make_run_id",
    "assemble_field_outputs",
    "export_advantage_run",
    "load_advantage_run",
]
