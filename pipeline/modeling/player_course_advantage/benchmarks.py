"""Runtime & scalability benchmarks (issue #63).

Times the player-course advantage pipeline on **synthetic** data of varying size
so we can tell whether it scales to realistic tournament fields, backtests, and
sweeps. It measures the shipped functions — it adds no modelling.

Reports wall-clock, throughput (rows/sec, players/sec), and (optional) peak
memory. Synthetic only; deterministic in *shape* (same columns every run) so
results are comparable across commits. Benchmark outputs are never committed —
the CLI writes only to a user-supplied path. Streamlit-free.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import pandas as pd

from .artifact_export import assemble_field_outputs, export_advantage_run
from .backtest import run_backtest
from .batch import score_tournament_field
from .diagnostics import explain_player_course
from .schema import AdvantageParams
from .scorer import score_player_course

PathLike = Union[str, Path]

BENCHMARK_COLUMNS: tuple[str, ...] = (
    "component", "field_size", "n_events", "rows_processed",
    "seconds", "rows_per_second", "players_per_second", "peak_kb",
)


def timed(fn: Callable, *args, track_memory: bool = False, **kwargs):
    """Return ``(result, seconds, peak_kb)`` for one call (peak_kb 0 if untracked)."""
    if track_memory:
        tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    seconds = time.perf_counter() - start
    peak_kb = 0.0
    if track_memory:
        peak_kb = tracemalloc.get_traced_memory()[1] / 1024.0
        tracemalloc.stop()
    return result, seconds, peak_kb


def synthetic_inputs(n_players: int, n_events: int, n_holes: int = 9):
    """Build a synthetic (sim, history, field, results) set of the requested size."""
    sim = pd.DataFrame([{
        "target_course_slug": "augusta_national", "target_hole_number": h,
        "target_hole_id": f"augusta_national:{h}", "candidate_course_slug": "src",
        "candidate_hole_number": h, "candidate_hole_id": f"src:{h}",
        "rank": 1, "total_score": 1.0, "similarity_weight": 1.0,
    } for h in range(1, n_holes + 1)])

    players = [f"p{i}" for i in range(n_players)]
    rows, tid = [], 0
    for i, pid in enumerate(players):
        outcome = 1.0 + (i % 5) * 0.3
        for yr in range(2021, 2024):
            for h in range(1, n_holes + 1):
                rows.append({
                    "player_id": pid, "tournament_id": f"{yr}-{tid}", "year": yr, "round": 1,
                    "hole_number": h, "course_slug": "src", "hole_id_v25": f"src:{h}",
                    "par": 4, "player_score": 4 - outcome, "field_avg_score": 4,
                })
                tid += 1
    history = pd.DataFrame(rows)

    results = pd.concat([
        pd.DataFrame({
            "event_id": [f"E{e}"] * n_players, "predict_season": [2024] * n_players,
            "target_course_slug": ["augusta_national"] * n_players,
            "player_id": players, "finish_rank": list(range(1, n_players + 1)),
        })
        for e in range(n_events)
    ], ignore_index=True) if n_events else pd.DataFrame()
    return sim, history, players, results


def _params(n_holes: int = 9) -> AdvantageParams:
    return AdvantageParams(min_occurrences_per_hole=1, min_holes_covered=n_holes)


def benchmark_components(
    field_size: int, n_events: int = 1, *, track_memory: bool = False,
    export_root: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Time each pipeline stage once for a given field size / event count."""
    n_holes = 9
    sim, history, field, results = synthetic_inputs(field_size, n_events, n_holes)
    params = _params(n_holes)
    rows_hist = int(len(history))
    records: list[dict] = []

    def _record(component, seconds, peak_kb, rows, players):
        records.append({
            "component": component, "field_size": field_size, "n_events": n_events,
            "rows_processed": rows, "seconds": seconds,
            "rows_per_second": (rows / seconds) if seconds > 0 else float("nan"),
            "players_per_second": (players / seconds) if seconds > 0 and players else float("nan"),
            "peak_kb": peak_kb,
        })

    _, s, m = timed(score_player_course, history, sim, field[0], "augusta_national", 2024,
                    params=params, track_memory=track_memory)
    _record("score_player_course", s, m, rows_hist, 1)

    ranking, s, m = timed(score_tournament_field, history, sim, field, "augusta_national",
                          2024, params=params, track_memory=track_memory)
    _record("score_tournament_field", s, m, rows_hist, field_size)

    _, s, m = timed(explain_player_course, history, sim, field[0], "augusta_national", 2024,
                    params=params, track_memory=track_memory)
    _record("explain_player_course", s, m, rows_hist, 1)

    if len(results):
        _, s, m = timed(run_backtest, history, sim, results, outcome_col="finish_rank",
                        higher_is_better=False, params=params, top_k=(10,),
                        track_memory=track_memory)
        _record("run_backtest", s, m, rows_hist, field_size * n_events)

    if export_root is not None:
        outs, s, m = timed(assemble_field_outputs, history, sim, field, "augusta_national",
                           2024, params=params, track_memory=track_memory)
        _record("assemble_field_outputs", s, m, rows_hist, field_size)
        run, s, m = timed(export_advantage_run, outs["rankings"],
                          hole_details=outs["hole_details"], diagnostics=outs["diagnostics"],
                          target_course_slug="augusta_national", config_name="baseline",
                          predict_season=2024, params=params, root=export_root,
                          track_memory=track_memory)
        size = sum(f.stat().st_size for f in Path(run.run_dir).glob("*") if f.is_file())
        rec = {"component": "export_advantage_run", "field_size": field_size,
               "n_events": n_events, "rows_processed": rows_hist, "seconds": s,
               "rows_per_second": (rows_hist / s) if s > 0 else float("nan"),
               "players_per_second": (field_size / s) if s > 0 else float("nan"),
               "peak_kb": m, "artifact_bytes": size}
        records.append(rec)

    return pd.DataFrame(records)


def run_benchmarks(
    field_sizes: Sequence[int] = (10, 50, 150),
    event_counts: Sequence[int] = (1, 5, 20),
    *,
    track_memory: bool = False,
) -> pd.DataFrame:
    """Full sweep: field-size scaling (1 event) + backtest-size scaling (fixed field)."""
    frames = [benchmark_components(fs, 1, track_memory=track_memory) for fs in field_sizes]
    fixed = min(field_sizes) if field_sizes else 10
    frames += [benchmark_components(fixed, ec, track_memory=track_memory) for ec in event_counts]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=list(BENCHMARK_COLUMNS))


def export_benchmarks(df: pd.DataFrame, out_path: PathLike) -> Path:
    """Write the benchmark table to ``out_path`` (``.json`` → JSON, else CSV)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        df.to_csv(out, index=False)
    return out


__all__ = [
    "BENCHMARK_COLUMNS",
    "timed",
    "synthetic_inputs",
    "benchmark_components",
    "run_benchmarks",
    "export_benchmarks",
]
