# Player-course advantage — metrics contract & acceptance gates (#58)

The official contract for judging the player-course advantage model. It fixes
*which* numbers we report, what each means, and the qualitative bar the model
must clear before anyone claims it works. It deliberately sets **no final numeric
thresholds** — those cannot be honestly chosen until real, leakage-free
historical player-by-hole data exists (blocked, see
[data sourcing](player_course_advantage_data_sourcing.md), #46). Everything below
that isn't marked *real-data-only* can be exercised on synthetic data to validate
the *plumbing*, never to claim predictive validity.

Metric names match what the code emits: `run_backtest` →
`BacktestResult.summary` / `.per_event`; `compare_baselines` → one row per ranker;
`run_sweep` → `SweepResult.table`.

---

## 1. Predictive ranking quality

Does a higher predicted `course_advantage` line up with better realized results?
"performance" is the outcome normalized higher-is-better (e.g. `-finish_rank`).

| Metric | Meaning | Formula (pseudo) | Higher better? | Inputs | Limitations |
|--------|---------|------------------|:--:|--------|-------------|
| `spearman_pooled` | Rank correlation over **all** scored (player, event) pairs pooled | `spearman(advantage, performance)` across all covered pairs | ✅ | predictions (advantage, performance, covered) | Pooling mixes events of different difficulty/field size; dominated by large events |
| `spearman_mean` | Mean of **per-event** Spearman | `mean_e spearman_e` | ✅ | per-event predictions | Small/degenerate events give noisy per-event values; a big gap vs pooled signals instability |
| `pearson_pooled` | Linear correlation of advantage vs performance, pooled | `pearson(advantage, performance)` | ✅ | predictions | Assumes linearity; sensitive to outliers/scale |
| `top_10_hit_rate` | Fraction of the model's top-10 picks that are actually in the top-10 performers | `|pred_top10 ∩ actual_top10| / min(10, n)` | ✅ | per-event advantage + performance | Clamped to field size; ties arbitrary; ignores near-misses |
| `top_20_hit_rate` | Same at k=20 | as above, k=20 | ✅ | " | " |
| `top_10_lift` | Mean performance of the model's top-10 minus the field mean | `mean(perf[pred_top10]) − mean(perf)` | ✅ | " | In performance units (event-relative), not comparable across outcome definitions |
| `top_20_lift` | Same at k=20 | as above, k=20 | ✅ | " | " |

Correlations are computed SciPy-free (Spearman = Pearson on ranks). All are
`NaN` when a field has <2 covered players or zero variance — such events are
excluded from means, not counted as 0.

---

## 2. Baseline lift

A model that can't beat trivial baselines is not useful. Baselines
(`baselines.py`): `null` (field-average), `recent_form`, `course_history`,
`same_par`, `season_average`. Lift is measured on the **primary metric**
(`spearman_pooled` by default) via `compare_baselines`.

| Metric | Meaning | Formula | Higher better? | Inputs | Limitations |
|--------|---------|---------|:--:|--------|-------------|
| `spearman_lift_vs_recent_form` | Model Spearman minus recent-form baseline's | `model − recent_form` | ✅ | baseline comparison table | A positive lift on synthetic data proves nothing |
| `spearman_lift_vs_course_history` | vs course-history baseline | `model − course_history` | ✅ | " | Course-history baseline is often low-coverage → NaN |
| `spearman_lift_vs_same_par` | vs same-par baseline | `model − same_par` | ✅ | " | " |
| `top_k_lift_vs_null` | Top-k lift improvement over the null (field-average) ranker | `model_top_k_lift − null_top_k_lift` | ✅ | " | Null lift is ~0 by construction; mostly a sanity check |
| `wins_vs_baselines_count` | How many baselines the model strictly beats on the primary metric | `Σ_b [model > baseline_b]` | ✅ | " | Ties are **not** wins; NaN baselines excluded from the denominator |

**Contract:** the model is **not** considered useful unless it beats these simple
baselines on **real, leakage-free** data. On synthetic data it frequently ties
(`model_beats_baselines` → `False`) — that is expected and must be reported as-is.

---

## 3. Coverage / reliability

Are inputs complete enough for any metric to mean something? (Detailed report:
#62 `data_health.py`.)

| Metric | Meaning | Higher better? | Inputs | Notes |
|--------|---------|:--:|--------|-------|
| `coverage` | Scored (covered) pairs / total (player, event) pairs | ✅ | predictions | Low coverage makes correlations unreliable regardless of value |
| `event_coverage` | Fraction of events with ≥2 covered players (a metric is computable) | ✅ | per-event | Events below this contribute no correlation |
| `holes_covered_mean` | Mean covered target holes per scored player-course | ✅ | rankings | Near `total_target_holes` = healthy |
| `low_coverage_rate` | Share of player-courses withheld (`low_coverage=True`) | ❌ | rankings | High rate → thin history |
| `no_history_rate` | Share withheld with reason `no_player_history` / `no_eligible_history` | ❌ | rankings | Distinguishes "no data" from "thin data" |
| `below_min_occurrences_rate` | Share of holes withheld for `below_min_occurrences` | ❌ | hole details | Sensitive to `min_occurrences_per_hole` |

---

## 4. Stability

Does the model work broadly, or only on cherry-picked cases?

| Metric | Meaning | Better | Inputs | Limitations |
|--------|---------|:--:|--------|-------------|
| metric variance **across seasons** | Std/spread of `spearman` over predict seasons | lower | per-event (has `predict_season`) | Needs several seasons; synthetic has few |
| metric variance **across courses** | Std/spread of `spearman` over `target_course_slug` | lower | per-event | Needs multiple courses |
| **rank stability across v2.5 configs** | Rank-correlation of player rankings between v2.5 presets (baseline vs hazard_heavy…) | higher | rankings per config | Some divergence is expected (presets are opinionated lenses) |
| **parameter sensitivity around defaults** | How much the primary metric moves for small changes in `n`, `W`, `m` near defaults | lower (smooth) | `run_sweep` table | High sensitivity = fragile/overfit-prone defaults |

---

## 5. Operational health

Is it fast/small enough to run for real fields, backtests, and sweeps? (Report:
#63 benchmarks.)

| Metric | Meaning | Better | Notes |
|--------|---------|:--:|-------|
| `batch_scoring_runtime` | Wall-clock for `score_tournament_field` on a full field | lower | report by field size 10/50/150 |
| `backtest_runtime` | Wall-clock for `run_backtest` | lower | report by 1/5/20 events |
| `sweep_runtime` | Wall-clock for `run_sweep` | lower | scales with grid × events |
| `artifact_size` | Bytes of an exported run | lower | ids/scores only, never geometry |
| `rows_processed` | History rows consumed | — | throughput denominator |
| `players_scored_per_second` | Field throughput | higher | primary scalability number |

---

## 6. Acceptance gates (qualitative only)

**No final numeric thresholds** until real data exists. Judge holistically:

- 🔴 **Red** — the model **cannot** beat the `null`/`recent_form` baselines on real
  data (`wins_vs_baselines_count` low, lifts ≤ 0), **or** coverage is too low for
  metrics to be trustworthy (`coverage`/`event_coverage` low, `low_coverage_rate`
  high). Do not ship or claim value.
- 🟡 **Yellow** — mixed: beats some baselines but not others, or positive overall
  but **unstable** across seasons/courses, or high parameter sensitivity. Promising,
  not proven — keep iterating.
- 🟢 **Green** — **consistent** lift over baselines (positive `spearman_lift_*`,
  high `wins_vs_baselines_count`) with acceptable coverage **and** stability across
  seasons/courses/configs, on real leakage-free data.

The first time real thresholds are set, they must be fixed on a **held-out**
split (see the sweep's `validation_seasons`), not tuned in-sample.

---

## 7. What can be tested synthetically vs needs real data

**Synthetic-testable (plumbing only — no validity claim):**
- that every metric **computes** and has the right shape/monotonic behavior
  (e.g. a perfectly-ordered synthetic field → `spearman_pooled ≈ 1`),
- baseline-lift and `wins_vs_baselines_count` arithmetic,
- coverage/health counts on deliberately imperfect fixtures,
- calibration/monotonicity direction on constructed monotonic vs scrambled data,
- runtime/scalability trends.

**Requires real historical player-by-hole data (blocked, #46):**
- any claim about **actual** predictive ranking quality,
- real baseline lift and whether the model genuinely beats simple alternatives,
- real calibration (do bigger advantages mean better outcomes?),
- real stability across seasons/courses, and any 🟢 green judgement.

---

## 8. Recommended first real-data evaluation checklist

1. Source and normalize real per-hole scores into
   `validate_hole_score_history` (per the [sourcing plan](player_course_advantage_data_sourcing.md)).
2. Run the **data-health report** (#62) first — if coverage is 🔴 red, stop.
3. Build v2.5 similar-hole sets for the target courses (leakage-free vs each event).
4. `run_backtest` walking seasons forward; then `compare_baselines`.
5. Compute baseline lift + `wins_vs_baselines_count` (§2). **If the model doesn't
   beat `null`/`recent_form`, report 🔴 and stop claiming value.**
6. Calibration/reliability (#61) and stability across seasons/courses/configs (§4).
7. Only then fix numeric gate thresholds, on a **held-out** validation split.
8. Record everything in the evaluation report (#59) with
   `data_source = real`.

Until step 1 is possible, every number this project produces is **synthetic** and
carries **no** predictive-validity claim.
