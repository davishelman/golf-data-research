# Annual course real-data plan (#80)

The target universe for real player-course advantage evaluation: every recurring
PGA course the repo can actually evaluate today, what data each still needs, and
where private data must live. **No real data is acquired or committed here** —
this is the planning foundation (#80) for the acquisition (#81) and batch
analysis (#82) that follow.

## Supported universe (from repo assets)

A course is **supported** only if the repo has both course geometry
(`courses/<slug>/`) *and* v2.5 similarity outputs
(`courses/_index/pointcloud_similarity/`). By that bar:

- **30 supported** courses (geometry + v2.5 similarity) — ready to evaluate once
  real per-hole scores + event outcomes are supplied.
- **12 unsupported** courses (geometry only, no similarity yet) — cannot be
  evaluated until v2.5 similarity is generated.

The full, machine-readable list is the committed example manifest:
[`data/player_course_advantage/templates/annual_course_targets.example.csv`](../data/player_course_advantage/templates/annual_course_targets.example.csv).
Regenerate it from assets with
`build_targets_from_assets("courses", "courses/_index/pointcloud_similarity/baseline/similarity_results.csv")`.

Some supported courses have **partial** similarity coverage (fewer than 18 target
holes, e.g. `renaissance_club`); the manifest's `similarity_holes` /
`expected_holes` / `notes` columns capture this — expect lower coverage there.

## Manifest columns

`course_slug, course_name, event_name, tour, annual_status, supported_in_repo,
has_v25_geometry, has_similarity_outputs, expected_holes, similarity_holes,
source_priority, real_data_status, notes`

`real_data_status` is one of: **missing** (supported, no private data yet),
**partial** (some data), **ready** (validated data present), **unsupported** (no
similarity). Fill `event_name` / `tour` / `annual_status` per course as you curate
targets — the generated example leaves them for you.

## Private data layout (gitignored — never commit)

Place private/real data under these paths (all ignored by `.gitignore`):

```
data/player_course_advantage/private/raw/            # your raw per-hole exports
data/player_course_advantage/private/normalized/     # canonical validated history
data/player_course_advantage/private/course_aliases/ # course_name → course_slug maps
data/player_course_advantage/private/coverage/       # data-status manifests
data/player_course_advantage/private/logs/           # import/run logs
data/player_course_advantage/analysis_runs/          # generated analysis bundles
```

Only files under `data/player_course_advantage/templates/` are committed
(docs/examples, no real data).

## Per-course readiness

Before a course can be evaluated it needs, in addition to `supported_in_repo`:
1. real per-hole scores normalized to the canonical schema (`real_data_status` →
   `partial`/`ready`), and
2. real event outcomes for the backtest (finish rank / strokes-gained / etc.).

Missing either → the course is reported but predictive evaluation is skipped.

## Exact next commands (once private data is placed locally)

```bash
# 1) normalize one course's raw scores (see the ingestion doc)
python scripts/normalize_player_hole_scores.py \
    --input data/player_course_advantage/private/raw/augusta_national.csv \
    --output data/player_course_advantage/private/normalized/augusta_national_history.csv \
    --course-aliases data/player_course_advantage/private/course_aliases/aliases.csv

# 2) run analysis for that course
python scripts/run_player_course_advantage_real_analysis.py \
    --history data/player_course_advantage/private/normalized/augusta_national_history.csv \
    --similar-holes courses/_index --course augusta_national \
    --results data/player_course_advantage/private/raw/augusta_national_outcomes.csv \
    --output data/player_course_advantage/analysis_runs/augusta_2026 --data-source real
```

## Annual acquisition / import (#81)

Templates (committed): `templates/raw_hole_scores_template.csv`,
`templates/event_outcomes_template.csv`, `templates/course_aliases_template.csv`.

Source modes (priority): **byo_csv** (a directory of `<course_slug>.csv` raw
exports), **api_export** (paid/API — only if creds present; `DATA_GOLF_API_KEY`,
`SHOTLINK_EXPORT_PATH`, `PGA_SCORECARD_RAW_DIR` — secrets are never stored),
**manual_scorecard**, and **audit** (report only). Missing creds/data → a status
report, never a crash.

**Audit availability:**

```bash
python scripts/acquire_annual_course_hole_scores.py \
    --manifest data/player_course_advantage/templates/annual_course_targets.example.csv \
    --raw-dir data/player_course_advantage/private/raw \
    --outcomes data/player_course_advantage/private/raw/event_outcomes.csv \
    --output data/player_course_advantage/private/coverage
```

**Import + normalize + validate all supported courses:**

```bash
python scripts/import_annual_course_hole_scores.py \
    --manifest data/player_course_advantage/templates/annual_course_targets.example.csv \
    --raw-dir data/player_course_advantage/private/raw \
    --aliases data/player_course_advantage/private/course_aliases/aliases.csv \
    --outcomes data/player_course_advantage/private/raw/event_outcomes.csv \
    --output data/player_course_advantage/private
```

Each supported course is classified **ready** (normalized history + event
outcomes), **partial** (normalized but outcomes missing → backtest blocked, or a
mapping/validation failure), **missing** (no raw file), or **unsupported**.
Private outputs (gitignored): `private/coverage/annual_course_data_status.csv`,
`private/normalized/<slug>_history.csv`,
`private/normalized/all_supported_annual_courses_history.csv`,
`private/logs/import_report.md`.

## Batch annual analysis (#82)

Once histories + outcomes are imported, evaluate **every covered course at once**:

```bash
python scripts/run_annual_course_real_analysis.py \
    --manifest data/player_course_advantage/templates/annual_course_targets.example.csv \
    --history data/player_course_advantage/private/normalized/all_supported_annual_courses_history.csv \
    --outcomes data/player_course_advantage/private/raw/event_outcomes.csv \
    --similar-holes courses/_index \
    --output data/player_course_advantage/analysis_runs/annual_2026 \
    --data-source real --run-sweep
```

Per supported course with data it runs the full pipeline (validate → data-health →
backtest → baselines → optional sweep → calibration → error analysis) via the #72
runner, then aggregates into `analysis_runs/annual_<ts>/`:
`annual_analysis_manifest.json`, `annual_course_status.csv`,
`annual_data_health_summary.csv`, `annual_model_metrics.csv`,
`annual_baseline_comparison.csv`, `annual_optimization_summary.csv` (if `--run-sweep`),
`annual_calibration_summary.csv`, `annual_error_analysis_summary.csv`,
`annual_insight_report.md`, `per_course/<slug>/…`, `missing_outputs.json`.

Courses are classified `evaluated` / `no_event_outcomes` / `no_similarity` /
`low_coverage` / `validation_failed` / `unsupported`. The aggregate report answers:
which courses have enough data, which lack scores/outcomes, where the model beats
or loses to baselines, whether higher advantage tracks better outcomes, and the
failure modes. On synthetic data it makes **no** predictive claim and reports
`Insufficient real data for reliable optimization.`
