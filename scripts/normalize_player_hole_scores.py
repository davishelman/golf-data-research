"""CLI: normalize a private raw per-hole score CSV into canonical history (#70).

    python scripts/normalize_player_hole_scores.py \
        --input path/to/private/raw.csv \
        --output data/player_course_advantage/private/normalized_history.csv \
        --course-aliases path/to/private/course_aliases.csv

Writes ONLY to the caller-specified ``--output`` path (keep it under a
gitignored/private directory). Prints validation + mapping/data-health summaries,
and fails clearly on missing columns, unmapped courses/holes, duplicate grain, or
invalid scores. Never commits or downloads data.
"""

from __future__ import annotations

import argparse
import os
import sys

_p = os.path.abspath(os.getcwd())
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "pipeline")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from pipeline.modeling.player_course_advantage.identity import (  # noqa: E402
    IdentityError, load_course_aliases,
)
from pipeline.modeling.player_course_advantage.ingestion import (  # noqa: E402
    IngestionError, normalize_and_validate,
)
from pipeline.modeling.player_course_advantage.schema import SchemaError  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Normalize a raw per-hole score CSV "
                                             "into canonical player-course advantage history.")
    ap.add_argument("--input", required=True, help="private raw per-hole score CSV")
    ap.add_argument("--output", required=True, help="output canonical history CSV (private path)")
    ap.add_argument("--course-aliases", default=None, help="course_name,course_slug alias CSV")
    ap.add_argument("--no-compute-field-avg", action="store_true",
                    help="do not compute field_avg_score when missing")
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.input)
    aliases = None
    if args.course_aliases:
        try:
            aliases = load_course_aliases(args.course_aliases)
        except IdentityError as exc:
            print(f"ERROR (aliases): {exc}", file=sys.stderr)
            return 2

    known_slugs = set(aliases.values()) if aliases else None
    try:
        canonical, report, mapping = normalize_and_validate(
            raw, aliases=aliases, known_slugs=known_slugs,
            compute_field_avg=not args.no_compute_field_avg,
        )
    except (IngestionError, IdentityError) as exc:
        print(f"ERROR (ingestion): {exc}", file=sys.stderr)
        return 2
    except SchemaError as exc:
        print("ERROR (validation): normalized history failed the schema contract:",
              file=sys.stderr)
        for e in exc.errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    canonical.to_csv(args.output, index=False)

    print("=== mapping / data-health summary ===")
    for k, v in mapping.items():
        print(f"  {k}: {v}")
    print("=== validation summary ===")
    print(f"  rows: {report.row_count}  players: {report.player_count}  "
          f"courses: {report.course_count}  years: {report.year_min}..{report.year_max}")
    print(f"wrote canonical history -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
