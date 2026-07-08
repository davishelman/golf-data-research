"""CLI: import + normalize annual-course per-hole scores (#81).

    python scripts/import_annual_course_hole_scores.py \
        --manifest data/player_course_advantage/templates/annual_course_targets.example.csv \
        --raw-dir data/player_course_advantage/private/raw \
        --aliases data/player_course_advantage/private/course_aliases/aliases.csv \
        --outcomes data/player_course_advantage/private/raw/event_outcomes.csv \
        --output data/player_course_advantage/private

Normalizes every supported course's private raw CSV to the canonical schema,
validates it, and writes per-course + combined history, a status CSV, and a report
under --output (a private/gitignored path). Courses without event outcomes are
imported but flagged as backtest-blocked. Fetches/commits nothing.
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

from pipeline.modeling.player_course_advantage.acquisition import import_annual_courses  # noqa: E402
from pipeline.modeling.player_course_advantage.course_targets import load_course_targets  # noqa: E402
from pipeline.modeling.player_course_advantage.identity import (  # noqa: E402
    IdentityError, load_course_aliases,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import + normalize annual-course per-hole scores.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--outcomes", default=None)
    ap.add_argument("--output", required=True, help="private output root (gitignored)")
    ap.add_argument("--source-mode", default="byo_csv")
    args = ap.parse_args(argv)

    targets = load_course_targets(args.manifest)
    aliases = None
    if args.aliases:
        try:
            aliases = load_course_aliases(args.aliases)
        except IdentityError as exc:
            print(f"ERROR (aliases): {exc}", file=sys.stderr)
            return 2

    result = import_annual_courses(
        targets, args.raw_dir, aliases=aliases, outcomes=args.outcomes,
        out_root=args.output, source_mode=args.source_mode)

    counts = result.status["status"].value_counts().to_dict()
    print("=== import status ===")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"combined history rows: {len(result.combined_history)}")
    print(f"wrote normalized/coverage/logs under {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
