# Player-course advantage (v0 spec)

**Status:** spec + building blocks. Shipped so far: the spec, the package
(`pipeline/modeling/player_course_advantage/`), the historical hole-score
**input schema + validation** (#30, #31), the **similar-hole set loader**
(#32, [§13](#13-similar-hole-set-loader-issue-32)), and the **recency-weighted
advantage scorer** (#33, [§14](#14-advantage-scorer-issue-33)). The **backtest**
is still **intentionally deferred** (see [Deferred](#whats-deferred)).

> **Experimental.** Every default below (`n`, `W`, `m`, coverage thresholds) is a
> placeholder from the sketch, **not** calibrated on data. Treat numbers as
> starting points, not recommendations.

---

## 1. Problem statement

Given an **upcoming course `C`** (18 holes) and a **player `p`**, estimate how
much of an edge `p` is likely to have on `C` — *before* `p` has necessarily
played `C` — by transferring `p`'s historical scoring from **holes that look
like** each of `C`'s holes.

"Looks like" comes entirely from the already-merged **v2.5** point-cloud
similarity model. This layer does **not** define or change any similarity; it
*consumes* v2.5 similar-hole sets and a historical player-by-hole scoring table
and produces per-hole and per-course advantage numbers.

This is a **different question** from v2/v2.5:

| Layer | Question | Output |
|-------|----------|--------|
| v2 | Which holes share engineered features? | feature-vector neighbors |
| v2.5 | Which holes share surface geometry? | Chamfer `total_score` neighbors |
| **player-course advantage (this)** | Given those neighbors, where does **player p** have an edge on course C? | per-hole + per-course advantage |

## 2. Inputs

1. **v2.5 similar-hole sets** — for each target hole `h` on `C`, the top-`n`
   similar historical holes with their v2.5 `total_score` (lower = more similar)
   and `rank`. Produced by
   `pipeline.modeling.pointcloud.export_similarity`. Read-only here.
2. **Historical hole-score history** — one row per player/tournament/year/round/
   hole occurrence. Contract defined in
   [§9](#9-historical-hole-score-input-schema-issue-31) and enforced by
   `pipeline.modeling.player_course_advantage.schema.validate_hole_score_history`.
3. **Parameters** — `AdvantageParams` (`n`, `W`, `m`, coverage thresholds).

## 3. Outputs

- **Player-hole advantage** `A(p, h)` — one number per target hole `h` on `C`
  (positive = `p` tends to beat the field on holes like `h`), plus a coverage
  count (how much weighted history backed it).
- **Player-course advantage** `A(p, C)` — one number per player per course,
  aggregating the 18 hole advantages, plus a coverage summary
  (`holes_covered / 18`).

Both are emitted as tidy tables; a score is withheld (marked low-coverage)
rather than fabricated when history is too thin — see [§8](#8-missing-data--coverage-policy).

## 4. Notation

For a target hole `h` on upcoming course `C`:

- **`Sim(h)`** — the top-`n` similar historical holes from v2.5.
- **`s`** — one similar hole in `Sim(h)`.
- **`o`** — one historical player occurrence on similar hole `s` (a specific
  player/tournament/year/round/hole row).
- **`y(o)`** — the season/year of occurrence `o`.
- **`outcome(p, s, o)`** — `p`'s normalized, positive-is-good performance on
  occurrence `o` (see [§6](#6-positive-is-good-outcome-convention)).
- **`sim_weight(h, s)`** — a weight derived from how similar `s` is to `h`
  ([§5](#5-similarity-weighting)).
- **`recency_weight(y)`** — a weight that fades older seasons ([§7](#7-recency-weighting)).

## 5. Similarity weighting

Each similar hole `s` contributes in proportion to how similar it is to `h`.
v2.5 gives `total_score` (lower = more similar) and `rank` (1 = closest).
Options, in rough order of preference:

1. **Rank-decay** *(default)* — `sim_weight = r ** (rank - 1)` with `r = 0.8`.
   Robust to `total_score` scale differences across configs/pars.
2. **Inverse score** — `sim_weight = 1 / (total_score + eps)`. Uses the raw
   distance but is sensitive to its scale.
3. **Softmax over `-total_score`** — smooth, temperature-controlled.
4. **Uniform** — `sim_weight = 1` for all of `Sim(h)`; a baseline/ablation.

The scorer will expose the choice; rank-decay is the v0 default because it does
not assume `total_score` is comparable across pars or presets.

## 6. Positive-is-good outcome convention

The primary outcome is **field-adjusted hole advantage**:

```
outcome(p, s, o) = field_avg_score_on_hole_round(o) - player_score_on_hole_round(p, o)
```

- **Positive = good.** Beating the field average (a *lower* stroke count than the
  field) yields a *positive* number, so "bigger is better" throughout.
- Adjusts each occurrence for **hole difficulty** and **field / day / weather /
  setup** conditions, because the field played the same hole the same round.
- Implemented as `schema.field_adjusted_advantage(field_avg_score, player_score)`.

**Fallback outcomes** (documented, not preferred):

- `player_score_to_par` — adjusts for par but *not* difficulty or conditions.
- raw `player_score` — no adjustment; least preferred.
- `player_strokes_gained_hole` — if a real per-hole SG is available it is a fine
  substitute, but it is not assumed to exist.

The scorer defaults to field-adjusted and only uses a fallback when
`field_avg_score` is absent (which the schema currently disallows in required
columns, so fallbacks are an explicit opt-in).

## 7. Recency weighting

Older seasons matter less. Because only strictly-past seasons are eligible
(`y(o) < predict_season`, see [§10](#10-lookahead-leakage)), season-level age is
measured from the **immediately prior** season:

```
age = (predict_season - 1) - y(o)
recency_weight(y(o)) = m ** age        # prior season (age 0) -> weight 1.0
```

Eligible years are `predict_season - W <= y(o) < predict_season`
(i.e. `0 <= age < W`). `m = 1` disables decay (flat window); smaller `m` fades the
past faster. A future **event-date** implementation can apply a stricter *date*
cutoff so same-season prior starts can be admitted up to the event date without
leakage; the season-level rule here is the conservative default.

## 8. Player-hole and player-course formulas

**Player-hole advantage** — similarity- and recency-weighted mean of the
outcomes over every occurrence of every similar hole:

```
                Σ_s∈Sim(h) Σ_o  sim_weight(h,s) · recency_weight(y(o)) · outcome(p,s,o)
A(p, h)  =      ─────────────────────────────────────────────────────────────────────
                Σ_s∈Sim(h) Σ_o  sim_weight(h,s) · recency_weight(y(o))
```

A weighted mean (normalized by the weight sum) keeps `A(p, h)` on the same
positive-is-good stroke scale as the outcome, regardless of how many occurrences
back it.

**Player-course advantage** — aggregate the 18 hole advantages:

```
A(p, C)  =  aggregate over h=1..18 of  A(p, h)
```

- **Default aggregate: sum** over covered holes → a full-round strokes-vs-field
  edge (natural units for an 18-hole course).
- **Mean** is available when coverage varies a lot between players and a
  per-hole average is fairer to compare.

The choice and per-hole coverage weighting will be scorer parameters.

## 9. Parameters: `n`, `W`, `m` (and coverage)

| Symbol | Meaning | Default | Notes |
|--------|---------|---------|-------|
| `n` | similar holes per target hole (from v2.5) | **10** | top-`n` by ascending `total_score` |
| `W` | lookback window (years) | **5** | inclusive; `0 <= age < W` |
| `m` | recency decay base | **0.85** | weight `= m ** age`; `1` = no decay |
| — | `include_current_course_history` | **False** | see [§10](#10-lookahead-leakage) |
| — | `min_occurrences_per_hole` | **3** | else hole marked low-coverage |
| — | `min_holes_covered` | **12** | of 18, else no course score |

All live in `AdvantageParams` / `DEFAULT_PARAMS` so the spec, tests, and future
scorer share one source of truth. **Experimental** — tune on real data later.

## 10. Lookahead leakage

⚠️ **This model is trivially easy to leak future information into. Guard it.**

- **Never** include occurrences from the season being predicted (or later) in
  `A(p, C)` for that season. Only `y(o) < predict_season` (and within `W`) are
  eligible. A backtest that lets `age = 0` be the target season is measuring the
  answer, not predicting it.
- **Current-course prior years** (`include_current_course_history`) default to
  **off**. Including them mixes "transferable hole shape" with "course-specific
  memory of `C`," which both muddies the interpretation and is the most common
  path to accidentally training on near-identical setups. Turn on only
  deliberately, and never for the target season.
- **v2.5 similarity itself must be leakage-free** relative to the prediction: use
  a similar-hole set built from data available before `predict_season`. This
  layer assumes the v2.5 inputs it is handed are already time-valid; it cannot
  detect leakage baked into upstream similarity.
- Backtests must **freeze the window** per predicted season and walk it forward,
  never pooling all years and scoring in-sample.

## 11. Missing-data / coverage policy

- A **target hole** with fewer than `min_occurrences_per_hole` weighted
  occurrences across its similar-hole set is marked **low-coverage** and excluded
  from the course aggregate (not counted as advantage 0 — that would bias toward
  the mean).
- A **player-course** score with fewer than `min_holes_covered` covered holes is
  **withheld** (emitted as null with a coverage note) rather than reported.
- Missing **optional** columns are fine. Missing **required** columns fail
  validation up front ([§9 schema](#9-historical-hole-score-input-schema-issue-31)).
- Coverage counts travel with every score so downstream consumers can filter.

## 12. How this differs from v2 / v2.5 similarity generation

- **It generates no similarity.** v2 (`pipeline.modeling.similarity`) and v2.5
  (`pipeline.modeling.pointcloud`) decide which holes are alike; this layer only
  *reads* v2.5's answer.
- **It is player-aware.** v2/v2.5 are player-agnostic geometry/feature models;
  advantage is defined per `(player, course)`.
- **It needs history that v2/v2.5 never touch** — a player-by-hole scoring table
  ([§9 schema](#9-historical-hole-score-input-schema-issue-31)).
- **Its own id space usage.** It joins history to v2.5 via `hole_id_v25`
  (`slug:hole_number`) and optionally to v2 via `hole_id_v2` (`slug__NN`); it
  does not redefine either id scheme.
- **Read-only and additive.** Nothing here mutates v2 or v2.5 artifacts.

---

## 9. Historical hole-score input schema (issue #31)

Defined and enforced in
`pipeline/modeling/player_course_advantage/schema.py`.

**Grain:** exactly one row per **player / tournament / year / round / course /
hole** occurrence. Duplicate rows on that key are a hard validation error (they
would double-count a scoring event).

**Occurrence key** (`KEY_COLUMNS`): `player_id`, `tournament_id`, `year`,
`round`, `course_slug`, `hole_number`. `course_slug` is part of the key so
multi-course events (e.g. rotating venues) and hole identity are explicit.

### Required columns (`REQUIRED_COLUMNS`)

| Column | Meaning |
|--------|---------|
| `player_id` | stable player identifier |
| `tournament_id` | stable event identifier |
| `year` | season/year of the occurrence |
| `round` | round number (1–4 regulation; up to 8 tolerated) |
| `hole_number` | 1–18 |
| `course_slug` | course identifier (e.g. `augusta_national`) |
| `hole_id_v25` | v2.5 join id, `slug:hole_number` (e.g. `augusta_national:13`) |
| `par` | 3–6 |
| `player_score` | strokes the player took on the hole (≥ 1) |
| `field_avg_score` | field mean strokes on that hole/round (difficulty adjust) |

`field_avg_score` is **required** so the positive-is-good outcome is always
computable without falling back to a weaker metric.

### Optional columns (`OPTIONAL_COLUMNS`)

`player_name`, `tournament_name`, `course_name`, `yardage`, `hole_id_v2`
(v2 id `slug__NN`), `player_score_to_par`, `field_score_to_par_avg`,
`player_strokes_gained_hole`, `field_adjusted_score` (cached
`field_avg_score - player_score`), `made_cut`.

### ID mapping (v2 ↔ v2.5)

- **v2.5**: `hole_id_v25 = "{course_slug}:{hole_number}"` — e.g.
  `augusta_national:13`. This is the join key into v2.5 similar-hole sets.
- **v2**: `hole_id_v2 = "{course_slug}__{hole_number:02d}"` — e.g.
  `augusta_national__13`. Optional; lets a row also reach the v2 feature/compact
  space. Same slug + hole number, different id shape (mirrors
  `pointcloud.demo.feature_id_for_pc_hole`).

### Validation — `validate_hole_score_history(df)`

Lightweight, pure-pandas, no real-data dependency. Accumulates **all** problems,
then raises `SchemaError` (with `.errors`) or returns a `ValidationReport`.
Checks: required columns present; **no nulls in any required column** (key columns
reported separately); no duplicate occurrences; plausible
`year`/`round`/`hole_number`/`par` ranges; `player_score` and `field_avg_score`
**numeric, non-null, and ≥ 1** (non-numeric values are reported, not silently
coerced); `hole_id_v25` (and `hole_id_v2` if present) match their id **shape** and
are **consistent** with `course_slug`/`hole_number`
(`hole_id_v25 == f"{course_slug}:{hole_number}"`,
`hole_id_v2 == f"{course_slug}__{hole_number:02d}"`); and — when a
`field_adjusted_score` column is supplied — that it equals
`field_avg_score - player_score`.

## 13. Similar-hole set loader (issue #32)

Implemented in `pipeline/modeling/player_course_advantage/similar_holes.py`. It
turns the existing **v2.5** similarity result CSVs into the `Sim(h)` sets this
model consumes — it does **not** generate similarity or touch v2.5 scoring.

```python
load_similar_hole_sets(root, target_course_slug, config_name="baseline",
                       top_n=10, weight_method="rank_decay",
                       rank_decay=0.8, softmax_temperature=1.0, epsilon=1e-9)
```

- **Inputs.** Reads `<config>/similarity_results.csv` under either supported
  layout (local index `<root>/pointcloud_similarity/…` or artifact bundle
  `<root>/data/pointcloud_similarity/…`), reusing
  `pipeline.modeling.pointcloud.demo` for path resolution.
- **Output.** A tidy frame, one row per (target hole, candidate hole):
  `target_course_slug, target_hole_number, target_hole_id,
  candidate_course_slug, candidate_hole_number, candidate_hole_id, rank,
  total_score, similarity_weight, weight_method, config_name`, followed by any
  v2.5 component columns present (`fairway_score` … `missing_surface_penalty`).
  Up to `18 * top_n` rows for a full course; sorted deterministically by
  `(target_hole_number, rank, candidate_hole_id)`.
- **Weighting** ([§5](#5-similarity-weighting) options) via
  `add_similarity_weights`: `rank_decay` (default), `inverse_score`,
  `softmax_score`, `uniform` — each **normalized to sum to 1.0 within every
  target hole**.
- **Validation.** Requires `target_hole_id`, `candidate_hole_id`, `rank`,
  `total_score`; checks numeric `total_score`, positive `rank`, parseable
  `slug:number` ids, positive `top_n`, and a recognized `weight_method`; raises a
  clear `SimilarHoleLoaderError` when the results dir/file is missing or the
  requested course has no rows.
- **Coverage.** `available_target_hole_numbers` /
  `missing_target_hole_numbers` diagnose partial courses without crashing.

Tolerant to older/smaller CSVs: missing optional component columns are simply
omitted. It never stores raw point-cloud geometry.

## 14. Advantage scorer (issue #33)

Implemented in `pipeline/modeling/player_course_advantage/scorer.py`. It is the
first real scorer: it consumes the validated hole-score history ([§9](#9-historical-hole-score-input-schema-issue-31))
and the #32 similar-hole sets ([§13](#13-similar-hole-set-loader-issue-32)) and
emits per-hole and per-course advantages. It generates **no** similarity.

```python
per_hole = score_player_holes(history, similar_holes, player_id,
                              target_course_slug, predict_season, params=DEFAULT_PARAMS)
per_hole, summary = score_player_course(history, similar_holes, player_id,
                              target_course_slug, predict_season,
                              config_name="baseline", params=DEFAULT_PARAMS,
                              aggregate="sum")
```

- **Model.** For a target hole `h`, `advantage(p, h)` is the weighted mean of
  the [§6](#6-positive-is-good-outcome-convention) outcome over the player's
  occurrences on `h`'s similar holes, with per-row weight
  `similarity_weight(h, s) · recency_weight(y(o))` (exactly [§8](#8-player-hole-and-player-course-formulas)).
  History joins to similar holes on `hole_id_v25 == candidate_hole_id`. The
  course number aggregates the covered holes — **`sum` by default** (expected
  strokes-vs-field over a round), `mean` optional; `course_advantage_mean` is
  always reported.
- **Leakage rule** ([§7](#7-recency-weighting), [§10](#10-lookahead-leakage)).
  Eligible history is only

  ```
  predict_season - W <= year < predict_season
  ```

  (never the target season or later), with `age = (predict_season - 1) - year`
  so the prior season has recency weight `1.0`. Same-course prior history is
  excluded unless `include_current_course_history=True`.
- **Coverage** ([§11](#11-missing-data--coverage-policy)). A target hole below
  `min_occurrences_per_hole` raw occurrences is marked low-coverage with
  `hole_advantage = NaN` (never a fabricated 0) and a `reason`; a player-course
  score with fewer than `min_holes_covered` covered holes is withheld. Reasons:
  `no_player_history`, `no_eligible_history`, `no_similar_holes`,
  `below_min_occurrences`, `below_min_holes_covered`. Coverage diagnostics
  (`raw_occurrences`, `weighted_occurrences`, `holes_covered`, …) travel with
  every score.
- **Validation.** Calls `validate_hole_score_history` first (raises `SchemaError`
  on a contract violation); uses a present, validated `field_adjusted_score` or
  computes it from `field_avg_score - player_score`.
- Pure, deterministic, Streamlit-free, and free of real-data dependencies.

Still **v0**: defaults are uncalibrated and the numbers are not trustworthy until
the retrospective **backtest (#35)** has walked them forward against held-out
seasons.

## What's deferred

Intentionally **not** implemented yet (guards issue scope):

- **Batch tournament-field ranking** (#34) — scoring a whole field at once.
- **Backtesting** / walk-forward evaluation (#35).
- **Parameter sweeps** (#36) — tuning `n` / `W` / `m` / coverage on real results.
- Any **real PGA data** sourcing/scraping — planned and blocker-documented in
  [`player_course_advantage_data_sourcing.md`](player_course_advantage_data_sourcing.md) (#46).
- Wiring into the Streamlit demo or the HF artifact — planned in
  [`player_course_advantage_ui_plan.md`](player_course_advantage_ui_plan.md) (#40);
  ranking view implementation is #47.

## Open questions

- Best `sim_weight` form and default `r` — needs backtesting against held-out
  seasons.
- Course aggregate: **sum** vs **coverage-weighted mean** as the headline number.
- Whether `player_strokes_gained_hole`, when present, should *replace* the
  field-adjusted outcome or be blended with it.
- Whether same-par is enough or par-5s/short par-4s need separate handling in
  aggregation (v2.5 candidates are same-par by construction, so `Sim(h)` already
  shares par with `h`).
