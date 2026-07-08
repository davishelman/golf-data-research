"""CLI: validation-split parameter optimizer for player-course advantage (#73).

    python scripts/optimize_player_course_advantage_params.py \
        --history data/player_course_advantage/private/normalized_history.csv \
        --similar-holes courses/_index --course augusta_national \
        --results path/to/private/event_outcomes.csv \
        --train 2019 2020 2021 --validation 2022 --test 2023 \
        --output data/player_course_advantage/analysis_runs/<timestamp>/opt \
        --data-source real

Tunes on validation seasons only; held-out test seasons are never used for
selection. Refuses to recommend from synthetic data unless --allow-synthetic.
Writes tables/recommendation to --output only.
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

from pipeline.modeling.player_course_advantage.optimization import (  # noqa: E402
    OptimizationError, export_optimization, optimize_parameters,
)
from pipeline.modeling.player_course_advantage.similar_holes import load_similar_hole_sets  # noqa: E402
from pipeline.modeling.player_course_advantage.sweep import SweepGrid  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Optimize player-course advantage params on a validation split.")
    ap.add_argument("--history", required=True)
    ap.add_argument("--similar-holes", required=True)
    ap.add_argument("--course", required=True)
    ap.add_argument("--config", nargs="+", default=["baseline"], help="v2.5 config presets to sweep")
    ap.add_argument("--results", required=True)
    ap.add_argument("--train", type=int, nargs="+", required=True)
    ap.add_argument("--validation", type=int, nargs="+", required=True)
    ap.add_argument("--test", type=int, nargs="*", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-source", choices=["real", "synthetic"], default="synthetic")
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--primary-metric", default="spearman_mean")
    ap.add_argument("--outcome-col", default="finish_rank")
    ap.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--top-n", type=int, nargs="+", default=[10])
    ap.add_argument("--recency-decay", type=float, nargs="+", default=[0.85, 1.0])
    ap.add_argument("--lookback-years", type=int, nargs="+", default=[5])
    args = ap.parse_args(argv)

    history = pd.read_csv(args.history)
    results = pd.read_csv(args.results)

    def provider(config_name, top_n, weight_method):
        return load_similar_hole_sets(args.similar_holes, args.course,
                                      config_name=config_name, top_n=top_n,
                                      weight_method=weight_method)

    grid = SweepGrid(top_n=tuple(args.top_n), config_name=tuple(args.config),
                     recency_decay=tuple(args.recency_decay),
                     lookback_years=tuple(args.lookback_years))
    try:
        result = optimize_parameters(
            history, results, provider, grid,
            train_seasons=args.train, validation_seasons=args.validation,
            test_seasons=args.test, outcome_col=args.outcome_col,
            higher_is_better=args.higher_is_better, primary_metric=args.primary_metric,
            data_source=args.data_source, allow_synthetic_recommendation=args.allow_synthetic,
        )
    except OptimizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    paths = export_optimization(result, args.output)
    print(result.to_markdown())
    print(f"\nwrote optimization outputs -> {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
