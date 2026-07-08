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

Acquisition tooling that scales this to the whole annual universe is #81; the
batch runner across all covered courses is #82.
