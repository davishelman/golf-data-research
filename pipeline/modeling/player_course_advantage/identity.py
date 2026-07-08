"""Course / hole identity mapping + diagnostics (issue #71).

Confirms that imported real scoring rows join correctly to the v2.5 hole
universe. It resolves messy source course labels (e.g. "Augusta National Golf
Club") to the project's canonical ``course_slug`` (``augusta_national``) via an
alias table, builds the ``hole_id_v25`` join key, and reports what could **not**
be mapped so a user can fix aliases before running any evaluation.

Pure pandas, Streamlit-free, no real-data dependency. It resolves *identity* only
— it does not read raw geometry or modify v2/v2.5 scoring. Source identifiers are
preserved on every row for traceability.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Union

import pandas as pd

PathLike = Union[str, Path]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class IdentityError(ValueError):
    """Raised when course aliases are unusable (e.g. ambiguous, missing columns)."""


def slugify(label: str) -> str:
    """A best-effort ``course_slug`` from a free-text label.

    Lowercases, replaces runs of non-alphanumerics with ``_``, trims edges.
    ``"Augusta National Golf Club"`` → ``"augusta_national_golf_club"``. This is a
    *fallback* only — the canonical slugs (``augusta_national``) usually need an
    alias, which is why unmapped rows are reported rather than guessed.
    """
    return _SLUG_RE.sub("_", str(label).strip().lower()).strip("_")


def _norm(label) -> str:
    return str(label).strip().lower()


def load_course_aliases(path: PathLike) -> dict[str, str]:
    """Load a ``course_name,course_slug`` alias CSV into ``{normalized_label: slug}``.

    Raises :class:`IdentityError` if the file lacks the two columns or maps the
    same label to two different slugs (an ambiguous alias).
    """
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "course_name" not in cols or "course_slug" not in cols:
        raise IdentityError(
            "course-aliases CSV must have 'course_name' and 'course_slug' columns; "
            f"got {list(df.columns)}"
        )
    aliases: dict[str, str] = {}
    conflicts: dict[str, set] = {}
    for name, slug in zip(df[cols["course_name"]], df[cols["course_slug"]]):
        key = _norm(name)
        slug = str(slug).strip()
        if key in aliases and aliases[key] != slug:
            conflicts.setdefault(key, {aliases[key]}).add(slug)
        aliases[key] = slug
    if conflicts:
        raise IdentityError(
            f"ambiguous course aliases (same label → multiple slugs): "
            f"{ {k: sorted(v) for k, v in conflicts.items()} }"
        )
    return aliases


def resolve_course_slug(
    label,
    aliases: Optional[dict[str, str]] = None,
    *,
    known_slugs: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Resolve a course label to a ``course_slug`` (or ``None`` if unmappable).

    Order: exact ``known_slugs`` match → alias table → ``slugify`` fallback. When
    ``known_slugs`` is given, a resolved slug outside that set returns ``None`` (it
    is *unmapped* relative to the v2.5 universe).
    """
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    known = set(known_slugs) if known_slugs is not None else None
    raw = str(label).strip()

    if known is not None and raw in known:
        return raw
    slug = (aliases or {}).get(_norm(label))
    if slug is None:
        slug = raw if (known is not None and raw in known) else slugify(label)
    if known is not None and slug not in known:
        return None
    return slug


def hole_id_v25(course_slug: str, hole_number) -> str:
    """The v2.5 join id ``"{course_slug}:{hole_number}"`` (mirrors the schema)."""
    return f"{course_slug}:{int(hole_number)}"


def hole_id_v2(course_slug: str, hole_number) -> str:
    """The v2 feature id ``"{course_slug}__NN"`` (zero-padded)."""
    return f"{course_slug}__{int(hole_number):02d}"


def build_mapping_report(
    df: pd.DataFrame,
    aliases: Optional[dict[str, str]] = None,
    *,
    course_col: str = "course_name",
    hole_col: str = "hole_number",
    source_id_cols: Iterable[str] = ("source_event_id", "season", "round"),
    known_slugs: Optional[Iterable[str]] = None,
    known_hole_ids: Optional[Iterable[str]] = None,
) -> dict:
    """Diagnose how well ``df``'s course/hole labels map into the v2.5 universe.

    Returns the required report fields (``total_rows``, ``mapped_rows``,
    ``unmapped_course_rows``, ``unmapped_hole_rows``,
    ``duplicate_source_identifier_count``, ``ambiguous_course_label_count``,
    ``most_frequent_unmapped_course_labels``). ``known_slugs`` /
    ``known_hole_ids`` (e.g. from the loaded v2.5 similar-hole sets) enable real
    unmapped detection; without them only structurally-missing values are flagged.
    """
    n = int(len(df))
    known_hids = set(known_hole_ids) if known_hole_ids is not None else None

    slugs = df[course_col].map(
        lambda v: resolve_course_slug(v, aliases, known_slugs=known_slugs)
    ) if course_col in df.columns else pd.Series([None] * n, index=df.index)
    hole_num = (
        pd.to_numeric(df[hole_col], errors="coerce") if hole_col in df.columns
        else pd.Series([float("nan")] * n, index=df.index)
    )

    unmapped_course = slugs.isna()
    # A hole is unmapped if its number is out of range, or (when a known hole-id
    # universe is supplied) the resolved hole_id_v25 is absent from it.
    bad_hole = hole_num.isna() | (hole_num < 1) | (hole_num > 18)
    if known_hids is not None:
        resolved_hid = [
            hole_id_v25(s, h) if (s is not None and pd.notna(h)) else None
            for s, h in zip(slugs, hole_num)
        ]
        bad_hole = bad_hole | pd.Series(
            [(hid is None) or (hid not in known_hids) for hid in resolved_hid],
            index=df.index,
        )
    unmapped_hole = bad_hole & ~unmapped_course

    src_cols = [c for c in source_id_cols if c in df.columns]
    dup_count = int(df.duplicated(subset=src_cols, keep=False).sum()) if src_cols else 0

    ambiguous = _ambiguous_label_count(df, slugs, course_col)
    top_unmapped = (
        df.loc[unmapped_course, course_col].astype(str).value_counts().head(5).to_dict()
        if course_col in df.columns else {}
    )
    mapped = int((~unmapped_course & ~unmapped_hole).sum())
    return {
        "total_rows": n,
        "mapped_rows": mapped,
        "unmapped_course_rows": int(unmapped_course.sum()),
        "unmapped_hole_rows": int(unmapped_hole.sum()),
        "duplicate_source_identifier_count": dup_count,
        "ambiguous_course_label_count": ambiguous,
        "most_frequent_unmapped_course_labels": {str(k): int(v) for k, v in top_unmapped.items()},
    }


def _ambiguous_label_count(df: pd.DataFrame, slugs: pd.Series, course_col: str) -> int:
    """Distinct source labels that (after normalization) resolve to >1 slug."""
    if course_col not in df.columns:
        return 0
    tmp = pd.DataFrame({"label": df[course_col].map(_norm), "slug": slugs})
    tmp = tmp.dropna(subset=["slug"])
    per_label = tmp.groupby("label")["slug"].nunique()
    return int((per_label > 1).sum())


__all__ = [
    "IdentityError",
    "slugify",
    "load_course_aliases",
    "resolve_course_slug",
    "hole_id_v25",
    "hole_id_v2",
    "build_mapping_report",
]
