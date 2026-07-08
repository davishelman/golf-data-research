"""CLI: batch real-analysis across all covered annual courses (#82).

    python scripts/run_annual_course_real_analysis.py \
        --manifest data/player_course_advantage/templates/annual_course_targets.example.csv \
        --history data/player_course_advantage/private/normalized/all_supported_annual_courses_history.csv \
        --outcomes data/player_course_advantage/private/raw/event_outcomes.csv \
        --similar-holes courses/_index \
        --output data/player_course_advantage/analysis_runs/annual_2026 \
        --data-source real --run-sweep

Evaluates every supported course that has validated history + event outcomes, and
writes per-course bundles plus aggregate reports under --output only. Sources no
data; makes no predictive claim on synthetic data.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

_p = os.path.abspath(os.getcwd())
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "pipeline")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

from pipeline.modeling.player_course_advantage.annual_analysis import run_annual_analysis  # noqa: E402
from pipeline.modeling.player_course_advantage.course_targets import load_course_targets  # noqa: E402
from pipeline.modeling.player_course_advantage.similar_holes import (  # noqa: E402
    SimilarHoleLoaderError, load_similar_hole_sets,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Batch real-analysis across annual courses.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--history", required=True, help="combined canonical history CSV")
    ap.add_argument("--outcomes", required=True, help="event outcomes CSV")
    ap.add_argument("--similar-holes", required=True, help="v2.5 results root")
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--output", default=None, help="output dir (default: annual_<timestamp>)")
    ap.add_argument("--data-source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--outcome-col", default="finish_position")
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--run-sweep", action="store_true")
    args = ap.parse_args(argv)

    out = args.output or f"data/player_course_advantage/analysis_runs/annual_{datetime.now():%Y%m%d_%H%M%S}"
    targets = load_course_targets(args.manifest)
    history = pd.read_csv(args.history)
    outcomes = pd.read_csv(args.outcomes)

    def provider(course_slug):
        try:
            return load_similar_hole_sets(args.similar_holes, course_slug, config_name=args.config)
        except SimilarHoleLoaderError:
            return None

    manifest = run_annual_analysis(
        targets, history, outcomes, provider, out,
        data_source=args.data_source, config_name=args.config,
        outcome_col=args.outcome_col, higher_is_better=args.higher_is_better,
        min_coverage=args.min_coverage, do_sweep=args.run_sweep)

    print("=== annual analysis ===")
    print("  status:", manifest["status_counts"])
    print("  evaluated:", manifest["evaluated_courses"])
    print(f"wrote aggregate + per-course bundles -> {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
