"""CLI: audit annual-course data availability (#81, source/status report).

    python scripts/acquire_annual_course_hole_scores.py \
        --manifest data/player_course_advantage/templates/annual_course_targets.example.csv \
        --raw-dir data/player_course_advantage/private/raw \
        --outcomes data/player_course_advantage/private/raw/event_outcomes.csv \
        --output data/player_course_advantage/private/coverage

Reports which acquisition sources are configured (env presence only — no secrets)
and classifies each supported course ready/partial/missing/unsupported. Writes the
status CSV to --output only; does NOT normalize or fetch anything.
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

from pathlib import Path  # noqa: E402

from pipeline.modeling.player_course_advantage.acquisition import (  # noqa: E402
    import_annual_courses, resolve_sources,
)
from pipeline.modeling.player_course_advantage.course_targets import load_course_targets  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit annual-course data availability.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--outcomes", default=None)
    ap.add_argument("--output", required=True, help="dir for the status CSV (private)")
    args = ap.parse_args(argv)

    print("=== acquisition sources (configured?) ===")
    for k, v in resolve_sources().items():
        print(f"  {k}: {v}")

    targets = load_course_targets(args.manifest)
    result = import_annual_courses(targets, args.raw_dir, outcomes=args.outcomes, out_root=None)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "annual_course_data_status.csv"
    result.status.to_csv(status_path, index=False)

    counts = result.status["status"].value_counts().to_dict()
    print("=== status ===")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"wrote {status_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
