"""Runtime/scalability benchmark runner for player-course advantage (#63).

Synthetic data only — no real/generated artifacts. Prints a concise table and,
optionally, writes it to a user-specified path (CSV or JSON). Not part of the
normal unit-test suite (a tiny smoke test lives in
``tests/test_player_course_advantage_benchmarks.py``).

Usage:

    python scripts/benchmark_player_course_advantage.py \
        --field-sizes 10 50 150 --event-counts 1 5 20 --out bench.csv
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on path whether run from root or scripts/.
_p = os.path.abspath(os.getcwd())
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "pipeline")):
    _p = os.path.dirname(_p)
if _p not in sys.path:
    sys.path.insert(0, _p)

from pipeline.modeling.player_course_advantage.benchmarks import (  # noqa: E402
    export_benchmarks,
    run_benchmarks,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark the player-course advantage pipeline.")
    ap.add_argument("--field-sizes", type=int, nargs="+", default=[10, 50, 150])
    ap.add_argument("--event-counts", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--track-memory", action="store_true", help="record peak memory (slower)")
    ap.add_argument("--out", default=None, help="optional output path (.csv or .json)")
    args = ap.parse_args(argv)

    df = run_benchmarks(args.field_sizes, args.event_counts, track_memory=args.track_memory)
    with_cols = [c for c in df.columns if c in (
        "component", "field_size", "n_events", "seconds",
        "rows_per_second", "players_per_second")]
    print(df[with_cols].to_string(index=False))
    if args.out:
        path = export_benchmarks(df, args.out)
        print(f"\nwrote {path}")
    return df


if __name__ == "__main__":  # pragma: no cover
    main()
