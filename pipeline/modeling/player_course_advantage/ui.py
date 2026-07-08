"""Streamlit-free view helpers for the player-course advantage UI (issue #47).

All the logic behind the Streamlit "Player-course advantage" section lives here as
pure functions, exactly like :mod:`pipeline.modeling.pointcloud.demo` backs the
v2.5 view — so the app stays a thin rendering shell and this is unit-testable
**without importing streamlit**. It formats already-computed outputs (rankings,
per-hole details, diagnostics) for display; it adds no modelling.

Two data sources, matching the #40 plan:

* **artifacts** — a run dir written by :func:`.artifact_export.export_advantage_run`
  (:func:`discover_advantage_runs`, :func:`load_ranking_view`), and
* **synthetic demo** — a tiny in-memory fake so the view is inspectable with no
  real data (:func:`synthetic_demo_view`). No real hole-score data is bundled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

from .artifact_export import MANIFEST_FILE, load_advantage_run
from .schema import AdvantageParams
from .artifact_export import assemble_field_outputs

PathLike = Union[str, Path]

#: Columns shown in the ranking table (in order), when present.
RANKING_DISPLAY_COLUMNS: tuple[str, ...] = (
    "rank", "player_id", "player_name", "course_advantage", "holes_covered", "coverage",
)


def coverage_label(row) -> str:
    """A human coverage badge for a ranking row.

    Withheld/low-coverage players are labelled with their reason (never a number);
    covered players show ``holes_covered / total``.
    """
    if bool(row.get("low_coverage")):
        reason = row.get("reason") or "low coverage"
        return f"⚠️ withheld ({reason})"
    total = row.get("total_target_holes")
    covered = row.get("holes_covered")
    return f"✓ {int(covered)}/{int(total)}" if pd.notna(total) else "✓ covered"


def format_ranking_for_display(rankings: pd.DataFrame) -> pd.DataFrame:
    """Round scores and add a text ``coverage`` badge; select display columns."""
    if rankings.empty:
        return pd.DataFrame(columns=[c for c in RANKING_DISPLAY_COLUMNS])
    disp = rankings.copy()
    disp["coverage"] = disp.apply(coverage_label, axis=1)
    if "course_advantage" in disp.columns:
        disp["course_advantage"] = pd.to_numeric(
            disp["course_advantage"], errors="coerce"
        ).round(3)
    cols = [c for c in RANKING_DISPLAY_COLUMNS if c in disp.columns]
    return disp[cols].reset_index(drop=True)


def player_hole_detail(
    hole_details: pd.DataFrame, diagnostics: pd.DataFrame, player_id: str, top_n: int = 10
) -> dict:
    """Per-hole advantage + top contributing similar holes for one player."""
    per_hole = pd.DataFrame()
    if not hole_details.empty and "player_id" in hole_details.columns:
        sub = hole_details[hole_details["player_id"] == player_id]
        keep = [c for c in ("target_hole_id", "hole_advantage", "raw_occurrences",
                            "low_coverage", "reason") if c in sub.columns]
        per_hole = sub[keep].reset_index(drop=True)
        if "hole_advantage" in per_hole.columns:
            # A fully low-coverage player's advantage column is all-None (object
            # dtype); pandas>=2.3 raises on Series.round of object, so coerce first.
            per_hole["hole_advantage"] = pd.to_numeric(
                per_hole["hole_advantage"], errors="coerce"
            ).round(3)

    top_similar = pd.DataFrame()
    if not diagnostics.empty and "player_id" in diagnostics.columns:
        d = diagnostics[diagnostics["player_id"] == player_id].copy()
        if "weighted_contribution" in d.columns and not d.empty:
            d = d.reindex(
                d["weighted_contribution"].abs().sort_values(ascending=False).index
            ).head(top_n)
            keep = [c for c in ("target_hole_id", "candidate_hole_id",
                                "weighted_contribution") if c in d.columns]
            top_similar = d[keep].reset_index(drop=True)
            # Round only the numeric column (the id columns are strings).
            top_similar["weighted_contribution"] = pd.to_numeric(
                top_similar["weighted_contribution"], errors="coerce"
            ).round(3)

    return {"per_hole": per_hole, "top_similar": top_similar}


def discover_advantage_runs(roots: Iterable[PathLike]) -> list[Path]:
    """Find advantage run dirs (a ``manifest.json`` of the run kind) under ``roots``.

    Looks at ``<root>/data/player_course_advantage/*``, ``<root>/player_course_advantage/*``,
    and ``<root>/*`` — so it works against an artifact root or the gitignored
    ``data/player_course_advantage/`` directory. Returns sorted, de-duplicated dirs.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root is None:
            continue
        base = Path(root)
        for pattern_base in (base / "data" / "player_course_advantage", base / "player_course_advantage", base):
            if not pattern_base.is_dir():
                continue
            for child in sorted(pattern_base.iterdir()):
                manifest = child / MANIFEST_FILE
                if not manifest.is_file():
                    continue
                try:
                    kind = json.loads(manifest.read_text(encoding="utf-8")).get("kind", "")
                except (ValueError, OSError):
                    continue
                if kind == "player_course_advantage.run" and child.resolve() not in seen:
                    seen.add(child.resolve())
                    found.append(child)
    return found


def load_ranking_view(run_dir: PathLike) -> dict:
    """Load a persisted run and format it for display.

    Returns ``{manifest, data_source, ranking_display, hole_details, diagnostics}``.
    """
    run = load_advantage_run(run_dir)
    data_source = str(run.manifest.get("extra", {}).get("data_source", "artifact"))
    return {
        "manifest": run.manifest,
        "data_source": data_source,
        "ranking_display": format_ranking_for_display(run.rankings),
        "hole_details": run.hole_details,
        "diagnostics": run.diagnostics,
    }


# --------------------------------------------------------------------------- #
# Synthetic demo (no real data) — lets the view render offline for review
# --------------------------------------------------------------------------- #
def _synthetic_inputs(n_holes: int = 9):
    """A tiny, clearly-fake similar-hole set + history + field."""
    sim = pd.DataFrame([
        {
            "target_course_slug": "augusta_national", "target_hole_number": h,
            "target_hole_id": f"augusta_national:{h}",
            "candidate_course_slug": "demo_src", "candidate_hole_number": h,
            "candidate_hole_id": f"demo_src:{h}", "rank": 1, "total_score": 1.0,
            "similarity_weight": 1.0, "weight_method": "manual", "config_name": "baseline",
        }
        for h in range(1, n_holes + 1)
    ])
    players = {"demo_ace": 2.0, "demo_good": 1.0, "demo_mid": 0.5, "demo_low": 0.0}
    rows, tid = [], 0
    for pid, outcome in players.items():
        for yr in range(2021, 2024):
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "player_name": pid.upper(),
                    "tournament_id": f"{yr}-{tid}", "year": yr, "round": 1,
                    "hole_number": h, "course_slug": "demo_src",
                    "hole_id_v25": f"demo_src:{h}", "par": 4,
                    "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    history = pd.DataFrame(rows)
    return sim, history, list(players), n_holes


def synthetic_demo_view(predict_season: int = 2024) -> dict:
    """A ready-to-render view backed by synthetic data (``data_source='SYNTHETIC'``)."""
    sim, history, field, n_holes = _synthetic_inputs()
    params = AdvantageParams(min_holes_covered=n_holes)
    outs = assemble_field_outputs(
        history, sim, field, "augusta_national", predict_season, params=params
    )
    return {
        "manifest": {"run_id": "synthetic-demo"},
        "data_source": "SYNTHETIC",
        "ranking_display": format_ranking_for_display(outs["rankings"]),
        "hole_details": outs["hole_details"],
        "diagnostics": outs["diagnostics"],
    }


__all__ = [
    "RANKING_DISPLAY_COLUMNS",
    "coverage_label",
    "format_ranking_for_display",
    "player_hole_detail",
    "discover_advantage_runs",
    "load_ranking_view",
    "synthetic_demo_view",
]
