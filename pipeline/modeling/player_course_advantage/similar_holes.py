"""Similar-hole set loader (issue #32).

Converts existing **v2.5** point-cloud similarity result CSVs into normalized,
per-target-hole *similar-hole sets* for the downstream player-course advantage
model. This module only *reads* v2.5 outputs and reshapes/weights them — it does
**not** score similarity, does not touch v2.5 scoring logic, and stores no raw
point-cloud geometry.

Path resolution is delegated to the existing
:mod:`pipeline.modeling.pointcloud.demo` helpers, so both supported layouts work
transparently:

* **Local index** — ``<root>/pointcloud_similarity/<config>/similarity_results.csv``
  (point ``root`` at ``courses/_index``).
* **Artifact bundle** — ``<root>/data/pointcloud_similarity/<config>/similarity_results.csv``.

The output is a tidy frame with one row per (target hole, candidate hole) pair,
carrying a ``similarity_weight`` that sums to 1.0 within each target hole under
every supported weighting method. The (future) advantage scorer consumes these
weights; it is intentionally **not** implemented here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

from ..pointcloud.demo import list_pointcloud_configs, resolve_pointcloud_dir
from ..pointcloud.export_similarity import RESULTS_FILENAME
from ..pointcloud.schemas import parse_pc_hole_id

PathLike = Union[str, Path]

#: Columns a v2.5 result CSV must have for this loader to work.
REQUIRED_RESULT_COLUMNS: tuple[str, ...] = (
    "target_hole_id",
    "candidate_hole_id",
    "rank",
    "total_score",
)

#: v2.5 per-surface / penalty component columns preserved *when present*. The
#: loader never requires these, so it tolerates older / smaller result CSVs.
COMPONENT_COLUMNS: tuple[str, ...] = (
    "fairway_score",
    "green_score",
    "bunker_score",
    "water_score",
    "tee_score",
    "yardage_penalty",
    "elevation_penalty",
    "missing_surface_penalty",
)

#: Stable leading columns of the loader output (component columns follow).
CORE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "target_course_slug",
    "target_hole_number",
    "target_hole_id",
    "candidate_course_slug",
    "candidate_hole_number",
    "candidate_hole_id",
    "rank",
    "total_score",
    "similarity_weight",
    "weight_method",
    "config_name",
)

#: Recognized similarity-weighting methods.
WEIGHT_METHODS: tuple[str, ...] = (
    "rank_decay",
    "inverse_score",
    "softmax_score",
    "uniform",
)

#: Tolerance for the per-target "weights sum to 1" invariant.
_WEIGHT_SUM_TOL = 1e-6


class SimilarHoleLoaderError(ValueError):
    """Raised when v2.5 results are missing or malformed for the loader."""


# --------------------------------------------------------------------------- #
# ID parsing
# --------------------------------------------------------------------------- #
def parse_v25_hole_id(hole_id: str) -> tuple[str, int]:
    """Split a v2.5 hole id ``"slug:number"`` into ``(course_slug, hole_number)``.

    Thin wrapper over :func:`pipeline.modeling.pointcloud.schemas.parse_pc_hole_id`
    (single source of truth for the v2.5 id contract), re-raised as a
    :class:`SimilarHoleLoaderError` with the offending value for clearer loader
    diagnostics.

    >>> parse_v25_hole_id("augusta_national:13")
    ('augusta_national', 13)
    """
    try:
        return parse_pc_hole_id(str(hole_id))
    except ValueError as exc:
        raise SimilarHoleLoaderError(
            f"could not parse v2.5 hole id {hole_id!r} (expected 'slug:number'): {exc}"
        ) from exc


def _split_hole_ids(ids: pd.Series, which: str) -> tuple[pd.Series, pd.Series]:
    """Parse a column of v2.5 ids into parallel (slug, number) Series.

    ``which`` names the column for error messages (e.g. ``"target_hole_id"``).
    """
    slugs: list[str] = []
    numbers: list[int] = []
    for value in ids:
        try:
            slug, number = parse_v25_hole_id(value)
        except SimilarHoleLoaderError as exc:
            raise SimilarHoleLoaderError(f"malformed {which}: {exc}") from exc
        slugs.append(slug)
        numbers.append(number)
    return (
        pd.Series(slugs, index=ids.index, dtype=object),
        pd.Series(numbers, index=ids.index, dtype=int),
    )


# --------------------------------------------------------------------------- #
# Weighting
# --------------------------------------------------------------------------- #
def add_similarity_weights(
    df: pd.DataFrame,
    method: str = "rank_decay",
    rank_decay: float = 0.8,
    softmax_temperature: float = 1.0,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Return a copy of ``df`` with normalized ``similarity_weight`` + ``weight_method``.

    Weights are computed per row from the chosen ``method`` and then normalized so
    they **sum to 1.0 within each ``target_hole_id``**:

    * ``rank_decay``    — ``rank_decay ** (rank - 1)`` (default; robust to
      ``total_score`` scale differences across pars / configs).
    * ``inverse_score`` — ``1 / (total_score + epsilon)``.
    * ``softmax_score`` — softmax over ``-total_score / softmax_temperature``.
    * ``uniform``       — equal weight (a baseline).

    Requires ``target_hole_id`` plus the column each method needs (``rank`` for
    ``rank_decay``; ``total_score`` for the score-based methods). Does not mutate
    ``df``.
    """
    if method not in WEIGHT_METHODS:
        raise SimilarHoleLoaderError(
            f"unknown weight_method {method!r}; expected one of {list(WEIGHT_METHODS)}"
        )

    out = df.copy()
    if out.empty:
        out["similarity_weight"] = pd.Series(dtype=float)
        out["weight_method"] = pd.Series(dtype=object)
        return out

    if "target_hole_id" not in out.columns:
        raise SimilarHoleLoaderError("cannot weight: missing 'target_hole_id' column")

    groups = out["target_hole_id"]
    raw = _raw_weights(out, groups, method, rank_decay, softmax_temperature, epsilon)

    denom = raw.groupby(groups).transform("sum")
    if (denom.abs() < _WEIGHT_SUM_TOL).any():
        raise SimilarHoleLoaderError(
            f"weight normalization failed for method {method!r}: a target hole has "
            "a zero total weight (check total_score / rank values)"
        )
    out["similarity_weight"] = (raw / denom).astype(float)
    out["weight_method"] = method
    return out


def _raw_weights(
    df: pd.DataFrame,
    groups: pd.Series,
    method: str,
    rank_decay: float,
    softmax_temperature: float,
    epsilon: float,
) -> pd.Series:
    """Per-row *unnormalized* weights for ``method`` (normalization is caller's job)."""
    if method == "uniform":
        return pd.Series(1.0, index=df.index)
    if method == "rank_decay":
        return float(rank_decay) ** (df["rank"].astype(float) - 1.0)
    if method == "inverse_score":
        return 1.0 / (df["total_score"].astype(float) + float(epsilon))
    # softmax_score: exp(-score / T), stabilized by subtracting the per-target max.
    z = -df["total_score"].astype(float) / float(softmax_temperature)
    z_max = z.groupby(groups).transform("max")
    return pd.Series(np.exp(z - z_max), index=df.index)


def _verify_weights_normalized(df: pd.DataFrame) -> None:
    """Assert per-target ``similarity_weight`` sums are 1.0 (defensive invariant)."""
    sums = df.groupby("target_hole_id")["similarity_weight"].sum()
    bad = sums[(sums - 1.0).abs() > _WEIGHT_SUM_TOL]
    if not bad.empty:
        raise SimilarHoleLoaderError(
            f"internal error: per-target weights do not sum to 1.0: "
            f"{bad.round(6).to_dict()}"
        )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _resolve_results_path(root: PathLike, config_name: str) -> Path:
    """Locate ``<...>/<config_name>/similarity_results.csv`` under ``root``.

    Reuses :func:`pointcloud.demo.resolve_pointcloud_dir` so both the local-index
    and artifact-bundle layouts are handled. Raises a clear
    :class:`SimilarHoleLoaderError` when the results dir or config file is absent.
    """
    resolved = resolve_pointcloud_dir(root)
    if resolved is None:
        raise SimilarHoleLoaderError(
            f"no v2.5 point-cloud similarity results found under {str(root)!r}; "
            "expected a 'pointcloud_similarity' dir (local index) or "
            "'data/pointcloud_similarity' (artifact bundle)"
        )
    path = resolved / config_name / RESULTS_FILENAME
    if not path.exists():
        available = list_pointcloud_configs(root)
        raise SimilarHoleLoaderError(
            f"no results file for config {config_name!r} at {path}; "
            f"available configs: {available}"
        )
    return path


def load_similar_hole_sets(
    root: PathLike,
    target_course_slug: str,
    config_name: str = "baseline",
    top_n: int = 10,
    weight_method: str = "rank_decay",
    rank_decay: float = 0.8,
    softmax_temperature: float = 1.0,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Load normalized similar-hole sets for one course from v2.5 results.

    For every target hole on ``target_course_slug`` present in the ``config_name``
    v2.5 results, keeps its best ``top_n`` candidates (by ``total_score`` asc, ties
    broken by ``candidate_hole_id``), re-ranks them ``1..N``, and attaches a
    ``similarity_weight`` normalized to sum to 1.0 within each target hole.

    Returns a tidy frame with :data:`CORE_OUTPUT_COLUMNS` followed by whichever of
    :data:`COMPONENT_COLUMNS` the source CSV provided. For a full 18-hole course
    with enough candidates this is up to ``18 * top_n`` rows. Rows are sorted
    deterministically by ``(target_hole_number, rank, candidate_hole_id)``.

    Diagnose partial coverage with :func:`available_target_hole_numbers` /
    :func:`missing_target_hole_numbers`.

    Raises :class:`SimilarHoleLoaderError` when the results dir/file is missing,
    required columns are absent, ids are malformed, or no rows exist for the
    requested course/config; raises on a non-positive ``top_n`` or unknown
    ``weight_method``.
    """
    if top_n <= 0:
        raise SimilarHoleLoaderError(f"top_n must be positive, got {top_n}")
    if weight_method not in WEIGHT_METHODS:
        raise SimilarHoleLoaderError(
            f"unknown weight_method {weight_method!r}; "
            f"expected one of {list(WEIGHT_METHODS)}"
        )

    path = _resolve_results_path(root, config_name)
    raw = pd.read_csv(path)  # a fresh frame; never mutate a caller's df

    missing = [c for c in REQUIRED_RESULT_COLUMNS if c not in raw.columns]
    if missing:
        raise SimilarHoleLoaderError(
            f"v2.5 results at {path} missing required columns: {missing}"
        )

    # Numeric / positivity gates on the source data.
    total_score = pd.to_numeric(raw["total_score"], errors="coerce")
    if total_score.isna().any():
        raise SimilarHoleLoaderError(
            f"{int(total_score.isna().sum())} rows have non-numeric total_score in {path}"
        )
    rank = pd.to_numeric(raw["rank"], errors="coerce")
    if rank.isna().any() or (rank <= 0).any():
        raise SimilarHoleLoaderError(
            f"rank must be numeric and positive in {path}"
        )

    # Split target ids and filter to the requested course before doing more work.
    target_slug, target_hole = _split_hole_ids(raw["target_hole_id"], "target_hole_id")
    on_course = target_slug == target_course_slug
    if not on_course.any():
        available = sorted(target_slug.unique())
        raise SimilarHoleLoaderError(
            f"no target holes for course {target_course_slug!r} in config "
            f"{config_name!r} at {path}; available course slugs: {available}"
        )

    sub = raw.loc[on_course].copy()
    sub["target_course_slug"] = target_slug[on_course]
    sub["target_hole_number"] = target_hole[on_course]
    cand_slug, cand_hole = _split_hole_ids(sub["candidate_hole_id"], "candidate_hole_id")
    sub["candidate_course_slug"] = cand_slug
    sub["candidate_hole_number"] = cand_hole
    sub["total_score"] = total_score[on_course]

    # Per-target top_n selection + clean contiguous re-rank (mirrors the v2.5
    # demo's determinism: total_score asc, candidate_hole_id asc).
    sub = sub.sort_values(["target_hole_number", "total_score", "candidate_hole_id"])
    sub = sub.groupby("target_hole_id", sort=False).head(top_n).copy()
    sub["rank"] = sub.groupby("target_hole_id", sort=False).cumcount() + 1
    sub["config_name"] = config_name

    weighted = add_similarity_weights(
        sub, weight_method, rank_decay, softmax_temperature, epsilon
    )
    _verify_weights_normalized(weighted)

    present_components = [c for c in COMPONENT_COLUMNS if c in weighted.columns]
    ordered = list(CORE_OUTPUT_COLUMNS) + present_components
    out = weighted[ordered].sort_values(
        ["target_hole_number", "rank", "candidate_hole_id"]
    ).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- #
# Coverage diagnostics
# --------------------------------------------------------------------------- #
def available_target_hole_numbers(df: pd.DataFrame) -> list[int]:
    """Sorted distinct ``target_hole_number`` values present in a loader frame."""
    if "target_hole_number" not in df.columns or df.empty:
        return []
    return sorted(int(n) for n in df["target_hole_number"].unique())


def missing_target_hole_numbers(
    df: pd.DataFrame, expected_holes: Iterable[int] = range(1, 19)
) -> list[int]:
    """Expected hole numbers with no rows in ``df`` (default: a full 1..18 course)."""
    present = set(available_target_hole_numbers(df))
    return [h for h in expected_holes if h not in present]


__all__ = [
    "REQUIRED_RESULT_COLUMNS",
    "COMPONENT_COLUMNS",
    "CORE_OUTPUT_COLUMNS",
    "WEIGHT_METHODS",
    "SimilarHoleLoaderError",
    "parse_v25_hole_id",
    "add_similarity_weights",
    "load_similar_hole_sets",
    "available_target_hole_numbers",
    "missing_target_hole_numbers",
]
