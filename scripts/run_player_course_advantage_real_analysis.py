"""CLI: run the player-course advantage analysis bundle (#72).

    python scripts/run_player_course_advantage_real_analysis.py \
        --history data/player_course_advantage/private/normalized_history.csv \
        --similar-holes courses/_index \
        --results path/to/private/event_outcomes.csv \
        --output data/player_course_advantage/analysis_runs/<timestamp> \
        --data-source real --run-sweep --run-benchmarks

Consumes a validated canonical history, v2.5 similar-hole output, and an actual
event-outcomes table; writes the bundle to --output only. Labels data_source=real
only when you pass it. Sources no data itself.
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

from pipeline.modeling.player_course_advantage.real_analysis import run_real_analysis  # noqa: E402
from pipeline.modeling.player_course_advantage.similar_holes import load_similar_hole_sets  # noqa: E402
from pipeline.modeling.player_course_advantage.schema import SchemaError  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the player-course advantage analysis bundle.")
    ap.add_argument("--history", required=True, help="canonical normalized history CSV")
    ap.add_argument("--similar-holes", required=True,
                    help="v2.5 similar-hole results root (local index or artifact bundle)")
    ap.add_argument("--course", required=True, help="target course slug for the similar-hole load")
    ap.add_argument("--config", default="baseline", help="v2.5 config preset")
    ap.add_argument("--results", required=True, help="actual event-outcomes CSV")
    ap.add_argument("--output", required=True, help="output directory (gitignored/private)")
    ap.add_argument("--data-source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--outcome-col", default="finish_rank")
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--run-sweep", action="store_true")
    ap.add_argument("--run-benchmarks", action="store_true")
    args = ap.parse_args(argv)

    history = pd.read_csv(args.history)
    results = pd.read_csv(args.results)
    similar = load_similar_hole_sets(args.similar_holes, args.course, config_name=args.config)

    try:
        manifest = run_real_analysis(
            history, similar, results, args.output,
            data_source=args.data_source, config_name=args.config,
            outcome_col=args.outcome_col, higher_is_better=args.higher_is_better,
            min_coverage=args.min_coverage, do_sweep=args.run_sweep,
            do_benchmarks=args.run_benchmarks,
        )
    except SchemaError as exc:
        print("ERROR: history failed validation; predictive evaluation refused:", file=sys.stderr)
        for e in exc.errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    print(f"analysis_type={manifest['analysis_type']} data_source={manifest['data_source']}")
    if manifest.get("stopped_early"):
        print(f"STOPPED EARLY: {manifest.get('stop_reason')}")
    else:
        print(f"spearman_pooled={manifest.get('spearman_pooled')} "
              f"coverage={manifest.get('coverage')} "
              f"beats_all_baselines={manifest.get('beats_all_baselines')}")
    print(f"wrote bundle -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
