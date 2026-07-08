"""Annual course data acquisition / import tools (issue #81).

Scales the single-file ingestion adapter (#70) to a whole annual-course
workflow: read the target manifest (#80), find which supported courses have
private raw per-hole scores available, normalize + validate each, and write
canonical per-course and combined histories — **only to private/gitignored
paths**. Missing data produces a status report, never a crash.

Source modes (priority): (1) bring-your-own CSV directory, (2) paid/API export
if credentials are present, (3) manual scorecard CSV, (4) audit-only. Secrets are
**never** hardcoded — only env var *presence* is checked
(``DATA_GOLF_API_KEY``, ``SHOTLINK_EXPORT_PATH``, ``PGA_SCORECARD_RAW_DIR``).
No real data is bundled; no predictive claims. Pure/Streamlit-free.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .course_targets import supported_targets
from .identity import IdentityError
from .ingestion import IngestionError, normalize_and_validate
from .schema import SchemaError

PathLike = Union[str, Path]

SOURCE_MODES: tuple[str, ...] = ("byo_csv", "api_export", "manual_scorecard", "audit")
DATA_STATUSES: tuple[str, ...] = ("ready", "partial", "missing", "unsupported")

_STATUS_COLUMNS = ("course_slug", "supported", "status", "raw_file",
                   "has_outcomes", "history_rows", "reason")


class AcquisitionError(ValueError):
    """Raised on a structurally unusable acquisition request."""


@dataclass(frozen=True)
class AcquisitionResult:
    """Per-course status, the combined canonical history, and a report."""

    status: pd.DataFrame
    combined_history: pd.DataFrame
    per_course_history: dict
    report_md: str


# --------------------------------------------------------------------------- #
# Source discovery (no secrets echoed)
# --------------------------------------------------------------------------- #
def resolve_sources(env: Optional[dict] = None) -> dict:
    """Report which acquisition sources are *configured* (booleans/paths only)."""
    env = env if env is not None else os.environ
    return {
        "byo_csv": True,  # always available (a directory you point at)
        "data_golf_api": bool(env.get("DATA_GOLF_API_KEY")),
        "shotlink_export_path": env.get("SHOTLINK_EXPORT_PATH") or None,
        "pga_scorecard_raw_dir": env.get("PGA_SCORECARD_RAW_DIR") or None,
    }


def find_course_raw_file(raw_dir: PathLike, course_slug: str) -> Optional[Path]:
    """Find a private raw per-hole CSV for a course (``<slug>.csv`` / ``<slug>*.csv``)."""
    raw_dir = Path(raw_dir)
    exact = raw_dir / f"{course_slug}.csv"
    if exact.exists():
        return exact
    for pat in (f"{course_slug}.csv", f"{course_slug}_*.csv", f"{course_slug}/*.csv"):
        hits = sorted(glob.glob(str(raw_dir / pat)))
        if hits:
            return Path(hits[0])
    return None


def _outcomes_frame(outcomes: Union[pd.DataFrame, PathLike, None]) -> Optional[pd.DataFrame]:
    if outcomes is None:
        return None
    if isinstance(outcomes, pd.DataFrame):
        return outcomes
    p = Path(outcomes)
    return pd.read_csv(p) if p.exists() else None


def _has_outcomes(outcomes_df: Optional[pd.DataFrame], course_slug: str) -> bool:
    if outcomes_df is None or "course_slug" not in outcomes_df.columns:
        return False
    return bool((outcomes_df["course_slug"].astype(str) == course_slug).any())


# --------------------------------------------------------------------------- #
# Per-course import
# --------------------------------------------------------------------------- #
def import_course_history(
    raw_path: PathLike,
    course_slug: str,
    *,
    aliases: Optional[dict] = None,
    known_slugs=None,
) -> tuple[pd.DataFrame, object, dict]:
    """Normalize + validate one course's raw CSV (raises on bad data)."""
    raw = pd.read_csv(raw_path)
    return normalize_and_validate(raw, aliases=aliases, known_slugs=known_slugs)


def import_annual_courses(
    targets: pd.DataFrame,
    raw_dir: PathLike,
    *,
    aliases: Optional[dict] = None,
    outcomes: Union[pd.DataFrame, PathLike, None] = None,
    out_root: Optional[PathLike] = None,
    source_mode: str = "byo_csv",
) -> AcquisitionResult:
    """Import every supported annual course with available raw data.

    Classifies each course ``ready`` (normalized + outcomes present) / ``partial``
    (normalized but outcomes missing, or a mapping/validation failure) / ``missing``
    (no raw file) / ``unsupported``. Writes canonical per-course + combined history,
    a status CSV, and a report **only** under ``out_root`` (a private/gitignored
    path). Never mutates inputs; never fetches or commits data.
    """
    if source_mode not in SOURCE_MODES:
        raise AcquisitionError(f"unknown source_mode {source_mode!r}; expected {list(SOURCE_MODES)}")

    outcomes_df = _outcomes_frame(outcomes)
    known = set(supported_targets(targets)["course_slug"])
    rows: list[dict] = []
    per_course: dict[str, pd.DataFrame] = {}

    for r in targets.itertuples(index=False):
        slug = getattr(r, "course_slug")
        supported = bool(getattr(r, "supported_in_repo"))
        rec = {"course_slug": slug, "supported": supported, "status": "unsupported",
               "raw_file": None, "has_outcomes": False, "history_rows": 0, "reason": ""}
        if not supported:
            rec["reason"] = "not supported in repo (no v2.5 similarity)"
            rows.append(rec)
            continue

        raw = find_course_raw_file(raw_dir, slug)
        rec["raw_file"] = str(raw) if raw else None
        if raw is None:
            rec.update(status="missing", reason="no raw hole-score file found")
            rows.append(rec)
            continue

        try:
            canonical, _report, _mapping = import_course_history(
                raw, slug, aliases=aliases, known_slugs=known)
        except (IngestionError, IdentityError, SchemaError) as exc:
            rec.update(status="partial", reason=f"normalization/validation failed: {exc}")
            rows.append(rec)
            continue

        per_course[slug] = canonical
        rec["history_rows"] = int(len(canonical))
        has_out = _has_outcomes(outcomes_df, slug)
        rec["has_outcomes"] = has_out
        if has_out:
            rec["status"] = "ready"
        else:
            rec.update(status="partial",
                       reason="event outcomes missing — real backtest blocked")
        rows.append(rec)

    status = pd.DataFrame(rows, columns=list(_STATUS_COLUMNS))
    combined = (pd.concat(per_course.values(), ignore_index=True)
                if per_course else pd.DataFrame())
    report_md = _report_md(status, source_mode, outcomes_df is not None)

    if out_root is not None:
        _write_outputs(out_root, status, combined, per_course, report_md)
    return AcquisitionResult(status=status, combined_history=combined,
                             per_course_history=per_course, report_md=report_md)


def _write_outputs(out_root, status, combined, per_course, report_md) -> None:
    root = Path(out_root)
    (root / "coverage").mkdir(parents=True, exist_ok=True)
    (root / "normalized").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    status.to_csv(root / "coverage" / "annual_course_data_status.csv", index=False)
    for slug, df in per_course.items():
        df.to_csv(root / "normalized" / f"{slug}_history.csv", index=False)
    if not combined.empty:
        combined.to_csv(
            root / "normalized" / "all_supported_annual_courses_history.csv", index=False)
    (root / "logs" / "import_report.md").write_text(report_md, encoding="utf-8")


def _report_md(status: pd.DataFrame, source_mode: str, has_outcomes: bool) -> str:
    counts = {s: int((status["status"] == s).sum()) for s in DATA_STATUSES}
    lines = [
        "# Annual course data import report", "",
        f"- source mode: `{source_mode}`  ·  event outcomes provided: {has_outcomes}",
        f"- ready: **{counts['ready']}**  ·  partial: {counts['partial']}  ·  "
        f"missing: {counts['missing']}  ·  unsupported: {counts['unsupported']}",
        "",
        "> Private/synthetic import — no predictive claims. Courses are `ready` only "
        "with normalized history **and** event outcomes.",
        "",
    ]
    blocked = status[(status["status"] == "partial") &
                     status["reason"].str.contains("outcomes", na=False)]
    if not blocked.empty:
        lines.append("## Backtest blocked (missing event outcomes)")
        lines += [f"- {s}" for s in blocked["course_slug"].tolist()]
        lines.append("")
    miss = status[status["status"] == "missing"]
    if not miss.empty:
        lines.append("## Missing raw scores")
        lines += [f"- {s}" for s in miss["course_slug"].tolist()]
    return "\n".join(lines)


__all__ = [
    "SOURCE_MODES",
    "DATA_STATUSES",
    "AcquisitionError",
    "AcquisitionResult",
    "resolve_sources",
    "find_course_raw_file",
    "import_course_history",
    "import_annual_courses",
]
