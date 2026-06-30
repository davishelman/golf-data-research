"""Streamlit-free helpers for the Golf Hole Similarity Explorer demo app (``app.py``).

All UI-independent logic lives here so it can be unit-tested without importing
streamlit: artifact discovery, picking the right similarity table for a view,
formatting hole labels, building the side-by-side feature comparison, and the
dataset summary. The app is a thin Streamlit wrapper around these.

Uses only the light stack (pandas + existing project modules); no geopandas, no
network, and never writes files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .artifact_loader import load_modeling_artifacts
from .plausibility import presented_similarity_table

# Artifact folders auto-detected by the app (courses/_index is intentionally NOT
# here — this demo is about the downloaded artifact).
DEFAULT_CANDIDATES: tuple[str, ...] = (
    "golf-data-research-artifacts",
    "hf_artifact_lite",
    "../golf-data-research-artifacts",
    "../hf_artifact_lite",
)

DOWNLOAD_CMD = (
    "hf download davishelman/golf-data-research-artifacts "
    "--repo-type dataset --local-dir golf-data-research-artifacts"
)

#: Sidebar view label -> (kind, mode). ``presented`` uses the plausibility table;
#: ``raw_mode`` uses a raw similarity_modes/<mode>.csv facet.
VIEW_OPTIONS: dict[str, tuple[str, str]] = {
    "Presented plays-like": ("presented", "overall_v2"),
    "Raw overall_v2": ("raw_mode", "overall_v2"),
    "Facet: off the tee": ("raw_mode", "off_the_tee"),
    "Facet: approach": ("raw_mode", "approach"),
    "Facet: green complex": ("raw_mode", "green_complex"),
    "Facet: hazard": ("raw_mode", "hazard"),
    "Facet: terrain": ("raw_mode", "terrain"),
    "Facet: shot shape": ("raw_mode", "shot_shape"),
}

#: Feature comparison groups (only available columns are shown).
COMPARE_GROUPS: dict[str, tuple[str, ...]] = {
    "Basic": ("par", "hole_length_yd", "hole_length_m", "hole_width_m",
              "hole_depth_m", "green_y_m"),
    "Hazards": ("bunker_pct", "water_pct", "trees_pct", "rough_pct",
                "rough_inferred_pct"),
    "Terrain": ("z_range", "z_mean", "tee_to_green_elevation_change",
                "green_relative_elevation"),
    "Shape": ("dogleg_score", "fairway_centerline_shift",
              "fairway_width_drive_zone", "fairway_width_approach_zone"),
}

#: Columns shown in the selected-hole summary card (only those present).
HOLE_SUMMARY_COLS: tuple[str, ...] = (
    "course_slug", "hole_number", "par", "hole_length_yd", "hole_length_m",
    "trees_pct", "bunker_pct", "water_pct", "tee_to_green_elevation_change",
    "green_relative_elevation", "dogleg_score", "fairway_centerline_shift",
)


# --------------------------------------------------------------------------- #
# Discovery + loading
# --------------------------------------------------------------------------- #

def discover_artifact_root(candidates: tuple[str, ...] = DEFAULT_CANDIDATES) -> Optional[Path]:
    """First candidate folder that looks like an artifact (or local index), else None."""
    for cand in candidates:
        p = Path(cand)
        if (p / "data" / "hole_features.parquet").exists() or (p / "hole_features.parquet").exists():
            return p
    return None


def load_artifact(root) -> dict:
    """Load the artifact dict (raises FileNotFoundError if ``root`` is not valid)."""
    return load_modeling_artifacts(root)


def get_presented(art: dict, n_neighbors: int = 15) -> tuple[Optional[pd.DataFrame], str]:
    """Presented table: from the artifact if shipped, else computed live from v2.

    Returns ``(table_or_None, source)`` where source is ``"artifact"``,
    ``"computed live"``, or ``"unavailable"``.
    """
    shipped = art.get("presented_similarity", {}).get("overall_v2")
    if shipped is not None:
        return shipped, "artifact"
    v2 = art.get("similarity_v2")
    if v2 is None or art.get("features") is None:
        return None, "unavailable"
    live = presented_similarity_table(v2, art["features"], n_neighbors=n_neighbors)
    return live, "computed live"


# --------------------------------------------------------------------------- #
# Selectors / labels
# --------------------------------------------------------------------------- #

def course_slugs(features: pd.DataFrame) -> list[str]:
    return sorted(features["course_slug"].dropna().unique().tolist())


def holes_for_course(features: pd.DataFrame, slug: str) -> pd.DataFrame:
    sub = features[features["course_slug"] == slug]
    return sub.sort_values("hole_number").reset_index(drop=True)


def hole_label(row) -> str:
    """e.g. 'Hole 1 — Par 4 — 448 yd' (yardage omitted if absent)."""
    num = int(row["hole_number"])
    par = int(row["par"]) if pd.notna(row.get("par")) else "?"
    label = f"Hole {num} — Par {par}"
    yd = row.get("hole_length_yd")
    if yd is not None and pd.notna(yd):
        label += f" — {round(float(yd))} yd"
    return label


# --------------------------------------------------------------------------- #
# Similarity results for a view
# --------------------------------------------------------------------------- #

def _enrich_similar(d: pd.DataFrame, fi: pd.DataFrame, *, presented: bool) -> pd.DataFrame:
    """Add similar_par / similar_length_yd / course / number; fill 'ok' reasons."""
    if d.empty:
        return d
    d = d.copy()
    d["similar_par"] = d["similar_hole_id"].map(fi["par"]).astype("Int64")
    if "hole_length_yd" in fi.columns:
        d["similar_length_yd"] = d["similar_hole_id"].map(fi["hole_length_yd"]).round(0)
    if "similar_course_slug" not in d.columns:
        d["similar_course_slug"] = d["similar_hole_id"].map(fi["course_slug"])
    if "similar_hole_number" not in d.columns:
        d["similar_hole_number"] = d["similar_hole_id"].map(fi["hole_number"]).astype("Int64")
    if presented and "plausibility_reasons" in d.columns:
        d["plausibility_reasons"] = (d["plausibility_reasons"].fillna("ok")
                                     .replace("", "ok"))
    return d


def similarity_results(art: dict, view_label: str, hole_id: str, n: int,
                       show_same_course: bool) -> tuple[Optional[pd.DataFrame], str, str]:
    """Return ``(table, kind, source)`` for the selected view.

    ``kind`` is ``"presented"`` or ``"raw_mode"``; ``source`` is where the data
    came from. ``table`` is None if the view's data is unavailable.
    """
    kind, mode = VIEW_OPTIONS[view_label]
    feats = art["features"]
    fi = feats.set_index("hole_id")

    if kind == "presented":
        pres, source = get_presented(art)
        if pres is None:
            return None, kind, source
        d = pres[pres["query_hole_id"] == hole_id].copy()
        if not show_same_course and "same_course" in d.columns:
            d = d[~d["same_course"].astype(bool)]
        sort_col = "presented_rank" if "presented_rank" in d.columns else "plausibility_score"
        d = d.sort_values(sort_col).head(n)
        return _enrich_similar(d, fi, presented=True), kind, source

    modes = art.get("similarity_modes", {})
    raw = modes.get(mode)
    if raw is None:
        return None, kind, "unavailable"
    d = raw[raw["query_hole_id"] == hole_id].copy()
    if not show_same_course and "same_course" in d.columns:
        d = d[~d["same_course"].astype(bool)]
    sort_col = "rank" if "rank" in d.columns else "distance"
    d = d.sort_values(sort_col).head(n)
    return _enrich_similar(d, fi, presented=False), kind, "artifact"


PRESENTED_DISPLAY_COLS = (
    "presented_rank", "similar_hole_id", "similar_course_slug", "similar_hole_number",
    "similar_par", "similar_length_yd", "length_diff_m", "plausibility_score",
    "plausibility_reasons",
)
RAW_DISPLAY_COLS = (
    "rank", "similar_hole_id", "similar_course_slug", "similar_hole_number",
    "similar_par", "similar_length_yd", "distance", "length_diff_m", "similarity_mode",
)


def display_columns(table: pd.DataFrame, kind: str, show_raw_cols: bool) -> list[str]:
    """Column subset to show for a results table, in a friendly order."""
    base = PRESENTED_DISPLAY_COLS if kind == "presented" else RAW_DISPLAY_COLS
    cols = [c for c in base if c in table.columns]
    if show_raw_cols:
        cols += [c for c in table.columns if c not in cols]
    return cols


# --------------------------------------------------------------------------- #
# Feature comparison + dataset summary
# --------------------------------------------------------------------------- #

def compare_features(features: pd.DataFrame, query_id: str, match_id: str) -> pd.DataFrame:
    """Grouped query-vs-match feature table with a numeric ``difference`` column."""
    fi = features.set_index("hole_id")
    rows = []
    for group, cols in COMPARE_GROUPS.items():
        for c in cols:
            if c not in fi.columns:
                continue
            qv, mv = fi.loc[query_id, c], fi.loc[match_id, c]
            try:
                diff = round(float(qv) - float(mv), 3)
            except (TypeError, ValueError):
                diff = None
            rows.append({"group": group, "feature": c, "query": qv,
                         "match": mv, "difference": diff})
    return pd.DataFrame(rows, columns=["group", "feature", "query", "match", "difference"])


def dataset_summary(art: dict) -> dict:
    """Headline counts for the 'Dataset summary' panel (real loaded values)."""
    from .visual_compare import available_compact_ids
    feats = art["features"]
    pres, _ = get_presented(art)
    compact = art.get("compact_dir")
    n_compact = len(available_compact_ids(compact)) if compact is not None else 0
    return {
        "courses": int(feats["course_slug"].nunique()),
        "holes": int(feats["hole_id"].nunique()),
        "feature_columns": int(feats.shape[1]),
        "presented_similarity_rows": int(len(pres)) if pres is not None else 0,
        "similarity_modes": sorted(art.get("similarity_modes", {}).keys()),
        "compact_point_clouds": n_compact,
    }


__all__ = [
    "DEFAULT_CANDIDATES", "DOWNLOAD_CMD", "VIEW_OPTIONS", "COMPARE_GROUPS",
    "HOLE_SUMMARY_COLS", "discover_artifact_root", "load_artifact", "get_presented",
    "course_slugs", "holes_for_course", "hole_label", "similarity_results",
    "display_columns", "compare_features", "dataset_summary",
    "PRESENTED_DISPLAY_COLS", "RAW_DISPLAY_COLS",
]
