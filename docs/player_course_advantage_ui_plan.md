# Player-course advantage — Streamlit UI integration plan (#40)

**Status: plan.** The modeling layer is complete and stable (scorer #33, batch
ranking #34, diagnostics #39, backtest #35, baselines #38, sweep #36, artifacts
#44) — so the UI can now be built. This doc scopes *how*, so the implementation
(#47, ranking view) stays small and the existing v2/v2.5 app is untouched. It is
a plan, **not** an implementation.

## 1. Where it lives

The demo is a single `app.py` (Golf Hole Similarity Explorer) with a sidebar
(artifact / query hole / similarity view / v2.5 preset) and numbered subheaders
1–6 for the v2 and v2.5 similarity views. The advantage view is **additive**:

- Add a **top-level mode switch** (`st.tabs(["Hole similarity", "Player-course advantage"])`,
  or a sidebar radio) so the existing similarity explorer renders unchanged in its
  own tab.
- All new logic goes through a thin `pipeline.modeling.player_course_advantage`
  import — **no model logic in `app.py`**, mirroring how the current app defers to
  `demo_utils` / `pointcloud.demo`.
- The similarity UI keeps working with **zero behavioral change** (a hard
  requirement: the advantage tab must never break tab 1).

## 2. UI sections (advantage tab)

| Section | Control / display | Backing call (already shipped) |
|---------|-------------------|--------------------------------|
| Target course | `selectbox` over course slugs (reuse the existing course list) | — |
| v2.5 config | `selectbox` over `list_pointcloud_configs` | `pointcloud.demo` |
| Parameters | `slider`s for `n` (top_n), `W` (lookback_years), `m` (recency_decay); coverage thresholds | `AdvantageParams` |
| Field | select "all players in history" **or** upload a small `player_id[,player_name]` CSV | `score_tournament_field(field=…)` |
| Ranking table | sortable table: rank, player, `course_advantage`, coverage, low-coverage badge | `score_tournament_field` |
| Coverage / confidence | badge/'⚠️ low coverage' per player; `holes_covered / total` | ranking `low_coverage`, `holes_covered`, `reason` |
| Per-hole contribution | expander per player: per-hole advantage + top similar holes | `explain_player_course` (`hole_contributions`, `top_similar_holes`) |
| Baseline comparison | small table: model vs baselines + honest verdict | `compare_baselines`, `model_beats_baselines` |
| (optional) Backtest summary | read-only metrics table from a precomputed run | `load_advantage_run().backtest_summary` |

## 3. Data loading path

The scorer needs a **historical hole-score table** — which the repo does **not**
have (blocked on #46). So the UI supports two modes, with an honest empty state:

1. **Precomputed artifacts (preferred).** Point at a run dir under
   `data/player_course_advantage/<run_id>/` and load with
   `load_advantage_run(run_dir)` → `player_rankings.csv`, `player_hole_details.csv`,
   `diagnostics.csv`, optional `backtest_summary.csv`, `manifest.json`. This is fast,
   Streamlit-cache-friendly (`@st.cache_data`), and runs **no heavy scoring in the
   browser session**.
2. **On-the-fly (demo).** If a user supplies a history CSV (or the synthetic sample
   fixture), load similar holes with `load_similar_hole_sets` and call
   `score_tournament_field` / `explain_player_course` live. Guarded to small inputs.

**No interactive backtests or sweeps** — those are batch/offline jobs; the UI only
*displays* their precomputed `backtest_summary.csv` / sweep table.

## 4. Required scorer outputs (all already provided)

The UI needs nothing new from the model layer:

- **Ranking** — `score_tournament_field` → `FIELD_RANKING_COLUMNS` (rank, player,
  `course_advantage`, `course_advantage_mean`, `holes_covered`, coverage counts,
  `low_coverage`, `reason`).
- **Explanation** — `explain_player_course` → `hole_contributions`,
  `similar_hole_contributions`, `top_target_holes`, `top_similar_holes`,
  `occurrence_year_counts`, `low_coverage` (all serializable).
- **Baselines** — `compare_baselines`, `model_beats_baselines`.
- **Artifacts** — `load_advantage_run` for the persisted layout.

If any gap appears during #47, it is fixed in the model layer (with tests), not
patched in `app.py`.

## 5. Empty states & graceful degradation

- **No history / no artifacts** → info banner: "No player-course advantage data
  loaded. Provide a run directory or upload a history CSV. (No real PGA data is
  bundled — see the data-sourcing plan.)" The similarity tab stays fully usable.
- **Low-coverage players** → shown but clearly badged; a player-course score below
  `min_holes_covered` shows "withheld — low coverage" with its `reason`, never a
  fabricated number.
- **Missing v2.5 config for a course** → disable the advantage tab controls for
  that course with an explanatory note.
- Every screen states whether the data is **REAL or SYNTHETIC**.

## 6. Caveats to surface in the UI

- Numbers are **experimental / uncalibrated**; on synthetic data the model often
  ties the baselines (`model_beats_baselines` → False). The UI must not imply
  proven edge.
- Defaults (`n`, `W`, `m`, coverage) are placeholders; the parameter sliders are
  for exploration, not a claim that any setting is optimal.

## 7. Testing plan

- Extend `tests/test_app_smoke.py` (or a new `test_app_advantage_smoke.py`) to
  import the advantage-tab render helpers with a tiny synthetic run and assert no
  exception — **no network, no real data**, matching the existing smoke pattern.
- Keep all rendering logic in small, importable functions (e.g. a
  `pca_view.py`/`demo`-style module) so it is unit-testable **without** importing
  streamlit, exactly like `pointcloud.demo` backs the current v2.5 view.
- `@st.cache_data` on the artifact/loader calls; verify cache keys include the run
  dir + params.

## 8. Implementation checklist for #47 (ranking view)

1. Add a Streamlit-free `..._view` helper module (course/config/param selection →
   ranking DataFrame + coverage), unit-tested with synthetic data.
2. Add the advantage tab to `app.py` (mode switch) wiring the helper, with the
   empty state and REAL/SYNTHETIC banner.
3. Ranking table + per-player coverage badges + per-hole contribution expander.
4. Optional baseline-comparison table.
5. Smoke test; confirm the similarity tab is unchanged.

Deferred beyond #47 (future issues): interactive parameter re-scoring at scale,
backtest/sweep dashboards, and real-data wiring (gated on #46).
