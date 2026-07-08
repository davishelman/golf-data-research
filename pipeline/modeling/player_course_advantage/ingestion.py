"""Bring-your-own per-hole score ingestion adapter (issue #70).

Turns a *private* raw historical per-hole score export into the canonical
player-course advantage history schema (:mod:`.schema`), so a local user can go:

    private raw CSV → normalize → map course/hole ids → compute field_avg_score
    → validate_hole_score_history → write canonical history CSV (private path)

It never bundles, downloads, or commits real data — it only *transforms* what the
user supplies, writing to a caller-chosen (gitignored/private) path. Pure pandas,
Streamlit-free. Course/hole identity resolution lives in :mod:`.identity`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from .identity import build_mapping_report, hole_id_v2, hole_id_v25, resolve_course_slug
from .schema import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    validate_hole_score_history,
)

#: Canonical field → default source column name for a bring-your-own CSV.
DEFAULT_SOURCE_COLUMNS: dict[str, str] = {
    "player_id": "player_id",
    "player_name": "player_name",
    "tournament_id": "source_event_id",
    "tournament_name": "event_name",
    "year": "season",
    "round": "round",
    "course_slug": "course_slug",      # optional; else derived from course_name
    "course_name": "course_name",
    "hole_number": "hole_number",
    "par": "par",                      # optional in raw; required in output
    "player_score": "score",
    "field_avg_score": "field_avg_score",  # optional; computed if absent
}

#: Source fields that must be present (course identity may come from either
#: ``course_slug`` or ``course_name``).
_REQUIRED_SOURCE_FIELDS = (
    "tournament_id", "year", "round", "hole_number", "player_id", "player_score",
)

#: Grain used to compute a missing ``field_avg_score``.
_FIELD_AVG_GRAIN = ("tournament_id", "year", "course_slug", "round", "hole_number")


class IngestionError(ValueError):
    """Raised when a raw export cannot be safely normalized (bad columns, unmapped
    courses, out-of-range holes, un-fillable par)."""


def compute_field_avg_scores(df: pd.DataFrame) -> pd.Series:
    """Field mean strokes per event/season/course/round/hole (``field_avg_score``)."""
    grp = [c for c in _FIELD_AVG_GRAIN if c in df.columns]
    return df.groupby(grp)["player_score"].transform("mean")


def normalize_hole_scores(
    raw: pd.DataFrame,
    *,
    aliases: Optional[dict[str, str]] = None,
    known_slugs: Optional[Iterable[str]] = None,
    column_map: Optional[dict[str, str]] = None,
    compute_field_avg: bool = True,
) -> pd.DataFrame:
    """Normalize a raw per-hole export into a canonical hole-score history frame.

    Maps source columns (``column_map`` overrides :data:`DEFAULT_SOURCE_COLUMNS`),
    resolves ``course_slug`` (from a ``course_slug`` column or ``course_name`` via
    ``aliases`` / :func:`.identity.resolve_course_slug`), derives ``hole_id_v25``
    /``hole_id_v2``, fills ``par`` per hole where possible, and computes
    ``field_avg_score`` when absent. Raises :class:`IngestionError` with a clear
    message on missing required columns, unmapped courses, out-of-range holes, or
    par that cannot be filled. Does **not** mutate ``raw``.
    """
    cmap = {**DEFAULT_SOURCE_COLUMNS, **(column_map or {})}
    src = {canon: cmap[canon] for canon in cmap}

    # Required source fields present?
    missing = [
        canon for canon in _REQUIRED_SOURCE_FIELDS if src[canon] not in raw.columns
    ]
    if missing:
        raise IngestionError(
            f"raw export missing required source columns "
            f"{[src[c] for c in missing]} (for canonical fields {missing})"
        )
    has_slug_col = src["course_slug"] in raw.columns
    has_name_col = src["course_name"] in raw.columns
    if not has_slug_col and not has_name_col:
        raise IngestionError(
            f"raw export needs either '{src['course_slug']}' or "
            f"'{src['course_name']}' to identify the course."
        )

    out = pd.DataFrame(index=raw.index)
    out["player_id"] = raw[src["player_id"]].astype(str)
    out["tournament_id"] = raw[src["tournament_id"]].astype(str)
    out["year"] = pd.to_numeric(raw[src["year"]], errors="coerce").astype("Int64")
    out["round"] = pd.to_numeric(raw[src["round"]], errors="coerce").astype("Int64")
    hole_num = pd.to_numeric(raw[src["hole_number"]], errors="coerce")
    out["hole_number"] = hole_num.astype("Int64")
    out["player_score"] = pd.to_numeric(raw[src["player_score"]], errors="coerce")

    # --- course_slug ---
    out["course_slug"] = _resolve_course_slugs(
        raw, src, has_slug_col, has_name_col, aliases, known_slugs
    )

    # --- hole range ---
    bad_hole = hole_num.isna() | (hole_num < 1) | (hole_num > 18)
    if bad_hole.any():
        sample = sorted(set(raw.loc[bad_hole, src["hole_number"]].astype(str)))[:5]
        raise IngestionError(
            f"{int(bad_hole.sum())} rows have a hole_number outside 1..18: e.g. {sample}"
        )

    # --- ids ---
    out["hole_id_v25"] = [
        hole_id_v25(s, h) for s, h in zip(out["course_slug"], out["hole_number"])
    ]
    out["hole_id_v2"] = [
        hole_id_v2(s, h) for s, h in zip(out["course_slug"], out["hole_number"])
    ]

    # --- par (required by the schema): use given, fill per-hole, else error ---
    out["par"] = _resolve_par(raw, src, out)

    # --- optional enrichers ---
    for canon in ("player_name", "tournament_name", "course_name"):
        if src[canon] in raw.columns:
            out[canon] = raw[src[canon]]

    # --- field_avg_score: given or computed at event/season/course/round/hole ---
    fa_col = src["field_avg_score"]
    if fa_col in raw.columns and raw[fa_col].notna().all():
        out["field_avg_score"] = pd.to_numeric(raw[fa_col], errors="coerce")
    elif compute_field_avg:
        out["field_avg_score"] = compute_field_avg_scores(out)
    elif fa_col in raw.columns:
        out["field_avg_score"] = pd.to_numeric(raw[fa_col], errors="coerce")
    else:
        raise IngestionError(
            "field_avg_score is absent and compute_field_avg=False; "
            "either supply it or allow computation."
        )

    return _order_columns(out)


def _resolve_course_slugs(raw, src, has_slug_col, has_name_col, aliases, known_slugs):
    slugs = []
    unresolved: dict[str, int] = {}
    for idx in raw.index:
        slug = None
        if has_slug_col:
            v = raw.at[idx, src["course_slug"]]
            slug = None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v).strip()
        if not slug and has_name_col:
            slug = resolve_course_slug(
                raw.at[idx, src["course_name"]], aliases, known_slugs=known_slugs
            )
        if not slug:
            label = str(raw.at[idx, src["course_name"]]) if has_name_col else "<none>"
            unresolved[label] = unresolved.get(label, 0) + 1
        slugs.append(slug)
    if unresolved:
        top = dict(sorted(unresolved.items(), key=lambda kv: -kv[1])[:5])
        raise IngestionError(
            f"{sum(unresolved.values())} rows have an unmapped course. Add aliases "
            f"for these labels (course_name → course_slug): {top}"
        )
    return slugs


def _resolve_par(raw, src, out) -> pd.Series:
    par_col = src["par"]
    if par_col in raw.columns:
        par = pd.to_numeric(raw[par_col], errors="coerce")
    else:
        par = pd.Series([pd.NA] * len(out), index=out.index, dtype="Float64")
    # Fill missing par from any known par for the same (course_slug, hole_number).
    filler = pd.DataFrame({
        "course_slug": out["course_slug"], "hole_number": out["hole_number"], "par": par,
    })
    per_hole = filler.dropna(subset=["par"]).groupby(
        ["course_slug", "hole_number"]
    )["par"].first()
    filled = filler.apply(
        lambda r: r["par"] if pd.notna(r["par"])
        else per_hole.get((r["course_slug"], r["hole_number"]), pd.NA),
        axis=1,
    )
    if filled.isna().any():
        holes = (
            filler.loc[filled.isna(), ["course_slug", "hole_number"]]
            .drop_duplicates().head(5).to_dict("records")
        )
        raise IngestionError(
            "par is required but missing for some holes with no par anywhere in the "
            f"input: e.g. {holes}. Provide a par column or fill these holes."
        )
    return pd.to_numeric(filled, errors="coerce")


def _order_columns(out: pd.DataFrame) -> pd.DataFrame:
    ordered = list(REQUIRED_COLUMNS) + [
        c for c in OPTIONAL_COLUMNS if c in out.columns and c not in REQUIRED_COLUMNS
    ]
    ordered = [c for c in ordered if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    return out[ordered + extra].reset_index(drop=True)


def normalize_and_validate(
    raw: pd.DataFrame,
    *,
    aliases: Optional[dict[str, str]] = None,
    known_slugs: Optional[Iterable[str]] = None,
    known_hole_ids: Optional[Iterable[str]] = None,
    column_map: Optional[dict[str, str]] = None,
    compute_field_avg: bool = True,
) -> tuple[pd.DataFrame, "object", dict]:
    """Normalize, then run :func:`validate_hole_score_history`.

    Returns ``(canonical_df, validation_report, mapping_report)``. The mapping
    report is built from the *raw* labels (before normalization) for traceability.
    Raises :class:`IngestionError` (bad mapping/par) or
    :class:`~pipeline.modeling.player_course_advantage.schema.SchemaError`
    (duplicate grain, invalid scores, id inconsistency).
    """
    cmap = {**DEFAULT_SOURCE_COLUMNS, **(column_map or {})}
    mapping_report = build_mapping_report(
        raw,
        aliases,
        course_col=cmap["course_name"],
        hole_col=cmap["hole_number"],
        source_id_cols=[c for c in (cmap["tournament_id"], cmap["year"],
                                    cmap["round"], cmap["course_name"],
                                    cmap["hole_number"], cmap["player_id"])
                        if c in raw.columns],
        known_slugs=known_slugs,
        known_hole_ids=known_hole_ids,
    )
    canonical = normalize_hole_scores(
        raw, aliases=aliases, known_slugs=known_slugs,
        column_map=column_map, compute_field_avg=compute_field_avg,
    )
    # Astype cleanups so the schema's numeric checks see plain ints/floats.
    for c in ("year", "round", "hole_number", "par"):
        canonical[c] = pd.to_numeric(canonical[c], errors="coerce")
    report = validate_hole_score_history(canonical)
    return canonical, report, mapping_report


__all__ = [
    "DEFAULT_SOURCE_COLUMNS",
    "IngestionError",
    "compute_field_avg_scores",
    "normalize_hole_scores",
    "normalize_and_validate",
]
