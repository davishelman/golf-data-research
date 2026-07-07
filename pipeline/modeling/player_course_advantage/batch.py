"""Batch tournament-field ranking (issue #34).

Scores every player in an upcoming tournament field for one course with the
merged advantage scorer (:func:`.scorer.score_player_course`) and returns a
single deterministic ranking table. This is a thin, pure orchestration layer:

* it adds **no** new modelling — one ``score_player_course`` call per player,
* players with little or no eligible similar-hole history are ranked last and
  flagged (``low_coverage`` / ``reason``) rather than dropped or faked, and
* it is Streamlit-free and free of real-data dependencies.

Runnable two ways: the Python API (:func:`score_tournament_field`) or a small CLI
(``python -m pipeline.modeling.player_course_advantage.batch``). Per-player *why*
(top contributing holes) is available from the #39 diagnostics module
(:func:`.diagnostics.explain_player_course`); this module deliberately keeps the
ranking table flat and serializable.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

from .schema import DEFAULT_PARAMS, AdvantageParams
from .scorer import AGGREGATIONS, AdvantageScorerError, score_player_course
from .similar_holes import load_similar_hole_sets

PathLike = Union[str, Path]

#: Columns (and order) of the ranking table.
FIELD_RANKING_COLUMNS: tuple[str, ...] = (
    "rank",
    "player_id",
    "player_name",
    "course_advantage",
    "course_advantage_mean",
    "holes_covered",
    "total_target_holes",
    "total_raw_occurrences",
    "total_weighted_occurrences",
    "low_coverage",
    "reason",
    "config_name",
    "target_course_slug",
)

#: Default filenames written by :func:`export_field_ranking`.
RANKING_FILENAME = "player_rankings.csv"
MANIFEST_FILENAME = "manifest.json"


def _field_players(
    field: Union[pd.DataFrame, Sequence[str]],
) -> tuple[list[str], dict[str, Optional[str]]]:
    """Normalize ``field`` to ``(player_ids, {player_id: player_name})``.

    ``field`` may be a list/sequence of player ids, or a DataFrame with a
    ``player_id`` column (and optionally ``player_name``). Order is preserved but
    duplicates are dropped (first wins) so a player is scored once.
    """
    names: dict[str, Optional[str]] = {}
    ids: list[str] = []

    if isinstance(field, pd.DataFrame):
        if "player_id" not in field.columns:
            raise AdvantageScorerError("field DataFrame must have a 'player_id' column")
        has_name = "player_name" in field.columns
        for row in field.itertuples(index=False):
            pid = str(getattr(row, "player_id"))
            if pid in names:
                continue
            ids.append(pid)
            nm = getattr(row, "player_name") if has_name else None
            names[pid] = None if nm is None or pd.isna(nm) else str(nm)
    else:
        for pid in field:
            pid = str(pid)
            if pid in names:
                continue
            ids.append(pid)
            names[pid] = None

    return ids, names


def rank_field(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach a deterministic 1-based ``rank`` to per-player course summaries.

    Ordering: valid (covered) players first, then descending ``course_advantage``,
    then higher ``holes_covered``, then ``player_id`` ascending as the final
    tie-breaker. Withheld players sort to the bottom but still receive a rank.
    """
    ordered = rows.sort_values(
        by=["low_coverage", "course_advantage", "holes_covered", "player_id"],
        ascending=[True, False, False, True],
        na_position="last",
        kind="mergesort",  # stable, so the key list fully determines the order
    ).reset_index(drop=True)
    ordered.insert(0, "rank", range(1, len(ordered) + 1))
    return ordered[list(FIELD_RANKING_COLUMNS)]


def score_tournament_field(
    history: pd.DataFrame,
    similar_holes: pd.DataFrame,
    field: Union[pd.DataFrame, Sequence[str]],
    target_course_slug: str,
    predict_season: int,
    config_name: str = "baseline",
    params: AdvantageParams = DEFAULT_PARAMS,
    aggregate: str = "sum",
) -> pd.DataFrame:
    """Rank an entire tournament field for one course.

    Runs :func:`score_player_course` for every player in ``field`` (a list of
    player ids or a DataFrame with ``player_id`` / optional ``player_name``) and
    returns one deterministically ordered row per player with
    :data:`FIELD_RANKING_COLUMNS`. ``history`` is validated once by the scorer
    (raises
    :class:`~pipeline.modeling.player_course_advantage.schema.SchemaError` on a
    contract violation). Never mutates the inputs.
    """
    if aggregate not in AGGREGATIONS:
        raise AdvantageScorerError(
            f"unknown aggregate {aggregate!r}; expected one of {list(AGGREGATIONS)}"
        )
    player_ids, field_names = _field_players(field)
    if not player_ids:
        raise AdvantageScorerError("field is empty; nothing to rank")

    summaries: list[dict] = []
    for pid in player_ids:
        _, summary = score_player_course(
            history, similar_holes, pid, target_course_slug, predict_season,
            config_name=config_name, params=params, aggregate=aggregate,
        )
        # Prefer a name the field supplied when history didn't carry one.
        if summary.get("player_name") is None and field_names.get(pid) is not None:
            summary["player_name"] = field_names[pid]
        summaries.append(summary)

    rows = pd.DataFrame(summaries)
    return rank_field(rows)


# --------------------------------------------------------------------------- #
# Optional export (writes a manifest alongside the ranking CSV)
# --------------------------------------------------------------------------- #
def export_field_ranking(
    ranking: pd.DataFrame,
    out_dir: PathLike,
    *,
    target_course_slug: str,
    config_name: str,
    predict_season: int,
    params: AdvantageParams,
    history_rows: Optional[int] = None,
) -> dict[str, Path]:
    """Write ``ranking`` to ``out_dir`` as a CSV plus a small ``manifest.json``.

    Returns ``{"ranking": <csv path>, "manifest": <json path>}``. Generated output
    lives under ``data/player_course_advantage/`` by convention and is gitignored;
    only tiny synthetic fixtures created inside tests are ever committed. (The full
    multi-file artifact layout is #44's job — this is the minimal manifest #34 asks
    for when exporting.)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranking_path = out / RANKING_FILENAME
    manifest_path = out / MANIFEST_FILENAME

    ranking.to_csv(ranking_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "player_course_advantage.field_ranking",
        "target_course_slug": target_course_slug,
        "config_name": config_name,
        "predict_season": predict_season,
        "params": {
            "n": params.n,
            "lookback_years": params.lookback_years,
            "recency_decay": params.recency_decay,
            "include_current_course_history": params.include_current_course_history,
            "min_occurrences_per_hole": params.min_occurrences_per_hole,
            "min_holes_covered": params.min_holes_covered,
        },
        "n_players": int(len(ranking)),
        "n_covered": int((~ranking["low_coverage"]).sum()) if len(ranking) else 0,
        "history_rows": int(history_rows) if history_rows is not None else None,
        "ranking_file": RANKING_FILENAME,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"ranking": ranking_path, "manifest": manifest_path}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="player_course_advantage.batch",
        description="Rank a tournament field for a course using the "
        "player-course advantage model.",
    )
    p.add_argument("--root", required=True,
                   help="v2.5 results root (local index or artifact bundle)")
    p.add_argument("--history", required=True, help="hole-score history CSV")
    p.add_argument("--course", required=True, help="target course slug")
    p.add_argument("--predict-season", type=int, required=True)
    p.add_argument("--config", default="baseline", help="v2.5 config name")
    p.add_argument("--field", default=None,
                   help="optional CSV with player_id[,player_name]; "
                        "defaults to every player in the history")
    p.add_argument("--out", default=None, help="optional output dir (writes CSV + manifest)")
    p.add_argument("--aggregate", default="sum", choices=list(AGGREGATIONS))
    p.add_argument("--top-n", type=int, default=DEFAULT_PARAMS.n)
    p.add_argument("--lookback-years", type=int, default=DEFAULT_PARAMS.lookback_years)
    p.add_argument("--recency-decay", type=float, default=DEFAULT_PARAMS.recency_decay)
    p.add_argument("--min-occurrences", type=int,
                   default=DEFAULT_PARAMS.min_occurrences_per_hole)
    p.add_argument("--min-holes", type=int, default=DEFAULT_PARAMS.min_holes_covered)
    p.add_argument("--include-current-course-history", action="store_true")
    return p


def _field_from_args(field_arg: Optional[str], history: pd.DataFrame):
    if field_arg is not None:
        return pd.read_csv(field_arg)
    # Default field: every player in the history, in first-seen order.
    return list(dict.fromkeys(history["player_id"].astype(str).tolist()))


def main(argv: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """CLI entry point. Returns the ranking DataFrame (and prints it)."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    params = AdvantageParams(
        n=args.top_n,
        lookback_years=args.lookback_years,
        recency_decay=args.recency_decay,
        include_current_course_history=args.include_current_course_history,
        min_occurrences_per_hole=args.min_occurrences,
        min_holes_covered=args.min_holes,
    )
    history = pd.read_csv(args.history)
    similar_holes = load_similar_hole_sets(
        args.root, args.course, config_name=args.config, top_n=args.top_n
    )
    field = _field_from_args(args.field, history)

    ranking = score_tournament_field(
        history, similar_holes, field, args.course, args.predict_season,
        config_name=args.config, params=params, aggregate=args.aggregate,
    )
    if args.out:
        paths = export_field_ranking(
            ranking, args.out, target_course_slug=args.course,
            config_name=args.config, predict_season=args.predict_season,
            params=params, history_rows=len(history),
        )
        print(f"wrote {paths['ranking']} and {paths['manifest']}")
    print(ranking.to_string(index=False))
    return ranking


__all__ = [
    "FIELD_RANKING_COLUMNS",
    "RANKING_FILENAME",
    "MANIFEST_FILENAME",
    "score_tournament_field",
    "rank_field",
    "export_field_ranking",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
