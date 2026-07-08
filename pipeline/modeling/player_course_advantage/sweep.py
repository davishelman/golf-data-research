"""Reproducible parameter sweep around the backtest (issue #36).

Grid-search the model's knobs and see which actually move the backtest metrics,
then surface a sane recommended default — without kidding ourselves about
overfitting the sweep itself.

Swept knobs fall in two places:

* **loader-side** — ``top_n`` (similar holes per target hole), ``weight_method``
  (similarity weighting), and the v2.5 ``config_name`` preset. These change the
  *similar-hole set*, so the sweep asks a caller-supplied
  ``similar_holes_provider(config_name, top_n, weight_method)`` for the right set
  (cached, so each unique triple loads once).
* **scorer-side** — ``lookback_years`` (W), ``recency_decay`` (m),
  ``min_occurrences_per_hole`` / ``min_holes_covered`` coverage thresholds, and the
  course ``aggregate`` mode. These go straight into :class:`AdvantageParams`.

**Honest selection.** Picking the best row on the same events you evaluate is
in-sample tuning. Pass ``validation_seasons`` to hold those seasons out: metrics
are computed separately on train and validation, :meth:`SweepResult.recommend`
selects on validation when available, and a warning fires when it isn't. Each
individual backtest is still leakage-free per event (the scorer's window guard).

Pure, deterministic, Streamlit-free. Real sweep outputs are gitignored; only
tiny synthetic fixtures used by tests are ever committed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Union

import pandas as pd

from .backtest import SimilarHoles, run_backtest
from .schema import DEFAULT_PARAMS, AdvantageParams

PathLike = Union[str, Path]
#: ``(config_name, top_n, weight_method) -> similar-hole set(s)``.
SimilarHolesProvider = Callable[[str, int, str], SimilarHoles]

_PARAM_COLUMNS: tuple[str, ...] = (
    "top_n", "weight_method", "config_name", "lookback_years", "recency_decay",
    "min_occurrences_per_hole", "min_holes_covered", "aggregate",
)


@dataclass(frozen=True)
class SweepGrid:
    """The values to cross for each knob (defaults = a single point at the model defaults)."""

    top_n: Sequence[int] = (DEFAULT_PARAMS.n,)
    weight_method: Sequence[str] = ("rank_decay",)
    config_name: Sequence[str] = ("baseline",)
    lookback_years: Sequence[int] = (DEFAULT_PARAMS.lookback_years,)
    recency_decay: Sequence[float] = (DEFAULT_PARAMS.recency_decay,)
    min_occurrences_per_hole: Sequence[int] = (DEFAULT_PARAMS.min_occurrences_per_hole,)
    min_holes_covered: Sequence[int] = (DEFAULT_PARAMS.min_holes_covered,)
    aggregate: Sequence[str] = ("sum",)

    def __iter__(self) -> Iterator[dict]:
        for combo in product(
            self.top_n, self.weight_method, self.config_name, self.lookback_years,
            self.recency_decay, self.min_occurrences_per_hole, self.min_holes_covered,
            self.aggregate,
        ):
            yield dict(zip(_PARAM_COLUMNS, combo))

    def __len__(self) -> int:
        n = 1
        for seq in (
            self.top_n, self.weight_method, self.config_name, self.lookback_years,
            self.recency_decay, self.min_occurrences_per_hole, self.min_holes_covered,
            self.aggregate,
        ):
            n *= len(tuple(seq))
        return n


def iter_param_sets(grid: SweepGrid) -> Iterator[dict]:
    """Yield every parameter combination in the grid (deterministic order)."""
    return iter(grid)


def _params_for(row: dict) -> AdvantageParams:
    return AdvantageParams(
        n=int(row["top_n"]),
        lookback_years=int(row["lookback_years"]),
        recency_decay=float(row["recency_decay"]),
        min_occurrences_per_hole=int(row["min_occurrences_per_hole"]),
        min_holes_covered=int(row["min_holes_covered"]),
    )


@dataclass(frozen=True)
class SweepResult:
    """The sweep table plus selection helpers and honesty warnings."""

    table: pd.DataFrame
    metric_keys: tuple[str, ...]
    has_validation: bool
    warnings: list[str] = field(default_factory=list)

    def recommend(
        self,
        metric: str = "spearman_pooled",
        *,
        min_coverage: float = 0.5,
        min_pairs: int = 1,
        prefer_validation: bool = True,
    ) -> Optional[pd.Series]:
        """The best-scoring eligible row (or ``None`` if none clear the guards).

        Selects on the validation split when present (and ``prefer_validation``),
        else on train. Eligibility requires coverage ``>= min_coverage``, at least
        ``min_pairs`` scored pairs, and a non-NaN metric — so a spuriously high
        correlation from a near-empty field can't win.
        """
        use_val = prefer_validation and self.has_validation
        prefix = "val_" if use_val else ""
        m, cov, pairs = f"{prefix}{metric}", f"{prefix}coverage", f"{prefix}n_pairs"
        df = self.table
        elig = df[
            (df[cov] >= min_coverage) & (df[pairs] >= min_pairs) & df[m].notna()
        ]
        if elig.empty:
            return None
        return elig.loc[elig[m].idxmax()]

    def recommended_params(self, **kwargs) -> Optional[AdvantageParams]:
        """:class:`AdvantageParams` for :meth:`recommend` (``None`` if nothing eligible)."""
        best = self.recommend(**kwargs)
        return None if best is None else _params_for(best.to_dict())

    def to_markdown(self, metric: str = "spearman_pooled") -> str:
        lines = ["# Player-course advantage parameter sweep", ""]
        best = self.recommend(metric=metric)
        if best is not None:
            picks = ", ".join(f"{c}={best[c]}" for c in _PARAM_COLUMNS)
            lines.append(f"**Recommended ({metric}):** {picks}")
        else:
            lines.append("**Recommended:** none cleared the coverage/pairs guards.")
        lines.append("")
        for w in self.warnings:
            lines.append(f"> ⚠️ {w}")
        return "\n".join(lines)


def run_sweep(
    history: pd.DataFrame,
    results: pd.DataFrame,
    similar_holes_provider: SimilarHolesProvider,
    *,
    grid: SweepGrid,
    outcome_col: str,
    higher_is_better: bool = False,
    top_k: Sequence[int] = (10, 20),
    validation_seasons: Optional[Sequence[int]] = None,
    season_col: str = "predict_season",
) -> SweepResult:
    """Backtest every grid point and return a compact metrics table.

    Splits ``results`` into train / validation by ``season_col`` when
    ``validation_seasons`` is given; metrics for each split get a ``val_`` prefix.
    ``similar_holes_provider`` is memoized on ``(config_name, top_n, weight_method)``.
    Never mutates inputs.
    """
    val_set = set(int(s) for s in validation_seasons) if validation_seasons else set()
    has_validation = bool(val_set)
    if has_validation:
        seasons = pd.to_numeric(results[season_col], errors="coerce")
        train_results = results[~seasons.isin(val_set)]
        valid_results = results[seasons.isin(val_set)]
    else:
        train_results, valid_results = results, None

    metric_keys = ("spearman_pooled", "spearman_mean", "pearson_pooled",
                   "coverage", "n_pairs")
    cache: dict[tuple, SimilarHoles] = {}

    def _sim(config_name: str, top_n: int, weight_method: str) -> SimilarHoles:
        key = (config_name, top_n, weight_method)
        if key not in cache:
            cache[key] = similar_holes_provider(config_name, top_n, weight_method)
        return cache[key]

    def _metrics(res_summary: dict) -> dict:
        return {k: res_summary.get(k) for k in metric_keys}

    rows: list[dict] = []
    for ps in grid:
        sim = _sim(ps["config_name"], ps["top_n"], ps["weight_method"])
        params = _params_for(ps)
        train = run_backtest(
            history, sim, train_results, outcome_col=outcome_col,
            higher_is_better=higher_is_better, params=params,
            config_name=ps["config_name"], aggregate=ps["aggregate"],
            top_k=top_k,
        )
        row = dict(ps)
        row.update(_metrics(train.summary))
        if has_validation and len(valid_results):
            val = run_backtest(
                history, sim, valid_results, outcome_col=outcome_col,
                higher_is_better=higher_is_better, params=params,
                config_name=ps["config_name"], aggregate=ps["aggregate"],
                top_k=top_k,
            )
            row.update({f"val_{k}": v for k, v in _metrics(val.summary).items()})
        rows.append(row)

    table = pd.DataFrame(rows)
    warnings = _sweep_warnings(table, has_validation)
    return SweepResult(
        table=table, metric_keys=metric_keys, has_validation=has_validation,
        warnings=warnings,
    )


def _sweep_warnings(table: pd.DataFrame, has_validation: bool) -> list[str]:
    warnings: list[str] = []
    if not has_validation:
        warnings.append(
            "selection is in-sample: pass validation_seasons to hold out seasons "
            "for honest tuning."
        )
    if "coverage" in table.columns and (table["coverage"] < 0.3).any():
        n = int((table["coverage"] < 0.3).sum())
        warnings.append(f"{n} parameter set(s) have coverage < 0.30 — metrics unstable.")
    if "n_pairs" in table.columns and (table["n_pairs"] < 5).any():
        n = int((table["n_pairs"] < 5).sum())
        warnings.append(
            f"{n} parameter set(s) scored < 5 pairs — high overfit risk, treat "
            "their metrics as noise."
        )
    return warnings


# --------------------------------------------------------------------------- #
# Optional export
# --------------------------------------------------------------------------- #
def export_sweep(
    result: SweepResult, out_dir: PathLike, *, extra_meta: Optional[dict] = None
) -> dict[str, Path]:
    """Write the sweep table + a manifest to ``out_dir`` (gitignored by convention)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "sweep_summary.csv"
    manifest_path = out / "manifest.json"
    result.table.to_csv(table_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "player_course_advantage.parameter_sweep",
        "n_param_sets": int(len(result.table)),
        "has_validation": result.has_validation,
        "warnings": result.warnings,
        "table_file": "sweep_summary.csv",
    }
    if extra_meta:
        manifest["extra"] = extra_meta
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"table": table_path, "manifest": manifest_path}


__all__ = [
    "SweepGrid",
    "SweepResult",
    "SimilarHolesProvider",
    "iter_param_sets",
    "run_sweep",
    "export_sweep",
]
