"""Annual course target manifest (issue #80).

Defines and loads the *target universe* for real player-course advantage
evaluation: which recurring/annual courses the repo can actually evaluate (has
``course_slug`` geometry **and** v2.5 similarity outputs), and each course's
real-data status. A committed example manifest lives at
``data/player_course_advantage/templates/annual_course_targets.example.csv``.

This module reads/validates the manifest and can (re)build it from repo assets.
It acquires **no** real data. Pure pandas, Streamlit-free.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, Path]

#: Required columns of an annual course target manifest.
COURSE_TARGET_COLUMNS: tuple[str, ...] = (
    "course_slug", "course_name", "event_name", "tour", "annual_status",
    "supported_in_repo", "has_v25_geometry", "has_similarity_outputs",
    "expected_holes", "source_priority", "real_data_status", "notes",
)

#: Recognized real-data status values.
REAL_DATA_STATUSES: tuple[str, ...] = ("missing", "partial", "ready", "unsupported")

_BOOL_COLS = ("supported_in_repo", "has_v25_geometry", "has_similarity_outputs")


class CourseTargetError(ValueError):
    """Raised when a course target manifest is malformed."""


def _coerce_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "yes")) \
        if s.dtype == object else s.astype(bool)


def load_course_targets(path: PathLike) -> pd.DataFrame:
    """Load + validate a manifest CSV. Raises :class:`CourseTargetError` on problems."""
    df = pd.read_csv(path)
    missing = [c for c in COURSE_TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise CourseTargetError(f"manifest missing required columns: {missing}")
    if df["course_slug"].duplicated().any():
        dups = df.loc[df["course_slug"].duplicated(), "course_slug"].tolist()
        raise CourseTargetError(f"duplicate course_slug rows: {sorted(set(dups))}")
    for c in _BOOL_COLS:
        df[c] = _coerce_bool(df[c])
    bad = set(df["real_data_status"].astype(str)) - set(REAL_DATA_STATUSES)
    if bad:
        raise CourseTargetError(
            f"unrecognized real_data_status values {sorted(bad)}; "
            f"expected {list(REAL_DATA_STATUSES)}")
    return df


def supported_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Rows the repo can actually evaluate (``supported_in_repo`` is true)."""
    return df[df["supported_in_repo"]].reset_index(drop=True)


def targets_by_status(df: pd.DataFrame) -> dict[str, int]:
    """Count of courses by ``real_data_status``."""
    return {str(k): int(v) for k, v in df["real_data_status"].value_counts().items()}


def build_targets_from_assets(
    courses_root: PathLike,
    similarity_results_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """(Re)build the manifest from repo assets — geometry dirs + v2.5 similarity.

    A course is ``supported_in_repo`` only when it has both a geometry directory
    and v2.5 similarity target holes. ``real_data_status`` defaults to
    ``unsupported`` (no similarity) or ``missing`` (supported, no private data yet).
    Local convenience — the committed example manifest is the source of truth.
    """
    courses_root = Path(courses_root)
    sim_holes: dict[str, int] = {}
    if similarity_results_path and Path(similarity_results_path).exists():
        s = pd.read_csv(similarity_results_path, usecols=["target_hole_id"])["target_hole_id"].astype(str)
        slug = s.str.rsplit(":", n=1).str[0]
        hole = s.str.rsplit(":", n=1).str[1]
        sim_holes = (pd.DataFrame({"slug": slug, "hole": hole})
                     .drop_duplicates().groupby("slug")["hole"].nunique().to_dict())

    rows = []
    for d in sorted(glob.glob(str(courses_root / "*"))):
        cs = os.path.basename(d)
        if not os.path.isdir(d) or cs.startswith("_"):
            continue
        name, expected = cs.replace("_", " ").title(), 18
        summ = Path(d) / "course_summary.json"
        if summ.exists():
            try:
                j = json.loads(summ.read_text(encoding="utf-8"))
                name = j.get("course", name)
                expected = len(j.get("holes", [])) or int(j.get("holes_processed", 18))
            except (ValueError, OSError):
                pass
        sh = int(sim_holes.get(cs, 0))
        has_sim = cs in sim_holes
        status = "unsupported" if not has_sim else "missing"
        note = "" if has_sim else "no v2.5 similarity outputs (geometry only)"
        if has_sim and sh < expected:
            note = f"partial similarity coverage: {sh}/{expected} holes"
        rows.append({
            "course_slug": cs, "course_name": name, "event_name": "", "tour": "PGA Tour",
            "annual_status": "unknown", "supported_in_repo": has_sim,
            "has_v25_geometry": True, "has_similarity_outputs": has_sim,
            "expected_holes": expected, "similarity_holes": sh,
            "source_priority": "byo_csv", "real_data_status": status, "notes": note,
        })
    return pd.DataFrame(rows)


__all__ = [
    "COURSE_TARGET_COLUMNS",
    "REAL_DATA_STATUSES",
    "CourseTargetError",
    "load_course_targets",
    "supported_targets",
    "targets_by_status",
    "build_targets_from_assets",
]
