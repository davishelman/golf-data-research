# v2.5 — Surface-Aware Point-Cloud Hole Similarity

An **additive** similarity model that compares golf holes by their *normalized
point clouds* — tee-relative, green-aligned points grouped by surface — rather
than by the engineered feature vector used in v2. It lives entirely under
`pipeline/modeling/pointcloud/` and never mutates the v2 pipeline or its outputs.

It depends only on the light stack (**pandas, numpy, scipy, pyyaml**; matplotlib
for the optional visual report). It does **not** import geopandas/rasterio.

## Why a separate model (v2.5 vs v2)

| | **v2** (`pipeline.modeling.similarity`) | **v2.5** (`pipeline.modeling.pointcloud`) |
|---|---|---|
| Unit of comparison | one ~90-dim engineered feature row per hole | the hole's labeled point cloud, per surface |
| Distance | weighted Euclidean in standardized feature space | per-surface symmetric **Chamfer** distance |
| Id space | `course_slug__NN` (e.g. `augusta_national__01`) | `course_slug:hole_number` (e.g. `augusta_national:13`) |
| Config | Python dicts (`SIMILARITY_MODES`) | YAML files under `configs/similarity/` |
| Outputs | `courses/_index/hole_similarity_*.csv` | `courses/_index/pointcloud_similarity/<config>/` |

v2.5 is kept separate on purpose: it has its own artifacts, its own id space, and
its own configs, so it can evolve without any risk to the committed v2 model.
Both can be shipped side by side in the Hugging Face artifact.

## Architecture

```
HoleMetadata + SurfacePoint   (schemas.py)        ← clean data contracts
        │
        ▼
candidate_filter.py           cheap par/yardage/surface gating (no geometry)
        │   passes
        ▼
chamfer.py                    symmetric per-surface Chamfer distance
        │
        ▼
score.py                      surface-weighted score + yardage/elevation penalties
        │
        ▼
export_similarity.py          single-target + batch export (results = IDs + scores)
        │
        ├── validate_similarity.py   cross-config comparison reports
        ├── calibrate.py             score-component decomposition
        ├── visualize_similarity.py  target-vs-top-N visual PNG
        ├── sweep.py                 reproducible multi-config runner
        ├── surface_loader.py        dedicated per-surface artifact loader
        ├── artifact_export.py       add v2.5 outputs to the HF bundle
        └── demo.py                  Streamlit-free demo data layer
```

### Data contracts (`schemas.py`)

* **`HoleMetadata`** — `hole_id, course_slug, hole_number, par, yards`, the
  `has_<surface>` flags, and optional `course_name`, `tee_elevation_m`,
  `green_elevation_m`. Enough for candidate filtering and penalties; **no
  geometry**.
* **`SurfacePoint`** — one normalized point: `hole_id, surface, x_lateral_m,
  y_down_hole_m, z_relative_m, point_weight`. Coordinate frame matches the
  existing compact clouds (tee at origin, +Y toward green, z relative to tee).
* **`SimilarityResult`** — IDs + scores only (target, candidate, total, per-surface
  components, yardage/elevation/missing penalties, filter reason). **Never stores
  point-cloud geometry.**

### Candidate filter (`candidate_filter.py`)

Cheap gating before any expensive Chamfer work, returning a stable reason code:
`PASS`, `DIFFERENT_PAR`, `MISSING_REQUIRED_SURFACE`, `YARDAGE_TOO_DIFFERENT`,
`NO_YARDAGE_WINDOW`. The per-par yardage window is the **more permissive** of an
absolute-yards and a percentage threshold.

### Scoring (`score.py`)

For each weighted surface: both holes have points → surface Chamfer distance; one
side missing → the configured missing-surface penalty; neither → skipped. Then:

```
total_score = Σ surface_weight·surface_score
            + |Δyards|·yardage_weight
            + |Δ(green−tee elevation)|·elevation_weight
```

**Lower `total_score` = more similar.** Surface weights are **config-driven** —
never hardcoded in the scorer.

## Config (`config.py`, `configs/similarity/*.yaml`)

A `PointCloudSimilarityConfig` loads from YAML and is validated at load time:
surface weights must sum to 1.0; required/weighted/penalty surfaces must be known;
yardage windows must exist for par_3/par_4/par_5; `model_version` and
`config_name` are required. Each config carries a deterministic `config_hash`
(SHA-256 of its normalized contents) that is stamped onto every result row for
traceability.

Shipped configs:

| Config | Emphasis |
|---|---|
| `pointcloud_chamfer_v1` (`baseline`) | balanced surface weights |
| `pointcloud_chamfer_hazard_heavy` | bunker + water |
| `pointcloud_chamfer_green_heavy` | green complex + elevation |
| `pointcloud_chamfer_fairway_heavy` | fairway corridor |
| `pointcloud_chamfer_v1_calibrated` | recalibrated baseline (see Calibration) |

## Key commands

```powershell
# Single target: rank the field against one hole
python -m pipeline.modeling.pointcloud.export_similarity `
    --config configs/similarity/pointcloud_chamfer_v1.yaml `
    --target-hole-id augusta_national:13 --top-n 10

# Batch: every hole, written to courses/_index/pointcloud_similarity/<config>/
python -m pipeline.modeling.pointcloud.export_similarity `
    --config configs/similarity/pointcloud_chamfer_v1.yaml --all --top-n 25

# Sweep several configs in one reproducible run (skips existing unless --overwrite)
python -m pipeline.modeling.pointcloud.sweep `
    --configs configs/similarity/pointcloud_chamfer_v1.yaml `
              configs/similarity/pointcloud_chamfer_hazard_heavy.yaml `
              configs/similarity/pointcloud_chamfer_green_heavy.yaml `
              configs/similarity/pointcloud_chamfer_fairway_heavy.yaml --top-n 25

# Validation: compare a hole's rankings across configs
python -m pipeline.modeling.pointcloud.validate_similarity `
    --target-hole-id augusta_national:13 `
    --configs baseline hazard_heavy green_heavy fairway_heavy --top-n 10

# Calibration: decompose scores into weighted components
python -m pipeline.modeling.pointcloud.calibrate `
    --config configs/similarity/pointcloud_chamfer_v1.yaml

# Visual: target vs top-N (PNG)
python -m pipeline.modeling.pointcloud.visualize_similarity `
    --target-hole-id augusta_national:13 --config-name baseline --top-n 6
```

## Outputs (under `courses/_index/pointcloud_similarity/`)

```
<config_name>/similarity_results.csv     IDs + scores per (target, candidate), ranked
<config_name>/filter_summary.csv         filter_reason → count
<config_name>/manifest.json              config hash, totals, filter-reason counts, provenance
_validation/<sanitized_hole>/            per-config top_matches, overlap, rank_comparison, manifest
sweep_manifest.json                      per-config status (ran/skipped/failed) for a sweep
```

`similarity_results.csv` columns: `model_version, config_name, config_hash,
target_hole_id, candidate_hole_id, rank, total_score, fairway_score, green_score,
bunker_score, water_score, tee_score, yardage_penalty, elevation_penalty,
missing_surface_penalty, filter_reason`.

These generated artifacts live under the gitignored `courses/` tree; they are not
committed and are exported to the HF bundle only intentionally (see below).

### Interpreting a result row

A low `total_score` with small per-surface components and near-zero penalties is a
genuine plays-alike. A row whose total is dominated by `missing_surface_penalty`
means the candidate is *missing a hazard the target has* (e.g. no water) rather
than being geometrically different — use the validation/calibration reports to
tell these apart.

## Calibration findings

The `calibrate` module decomposes every scored pair into weighted components and
flags two structural issues, addressed in `pointcloud_chamfer_v1_calibrated.yaml`
(a **new** config — the baseline is unchanged):

1. **Elevation double-count** — the Chamfer `z_weight` (2.0) already amplifies
   vertical offsets on every surface, and the explicit tee→green elevation penalty
   adds elevation again. The calibrated config lowers `z_weight` to 1.5 and the
   elevation penalty weight to 0.05.
2. **Missing-hazard penalties can dominate** otherwise-close pairs. The calibrated
   config softens bunker (40→25) and water (50→30) missing penalties so one
   missing hazard no longer swamps real surface similarity.

## Artifact + demo integration

* `artifact_export.add_pointcloud_similarity_to_artifact()` copies selected config
  outputs into `data/pointcloud_similarity/<config>/` of an HF bundle and writes
  `metadata/pointcloud_similarity.json`. It reuses v2's secret guard (an `.env`
  can never be swept in) and exports **IDs + scores only**.
* `demo.py` is a Streamlit-free data layer (`list_pointcloud_configs`,
  `load_pointcloud_results`, `top_matches_for_hole`, …) so a v2.5 view can be added
  to `app.py` in a few lines without touching the existing v2 views.

## Extending the loader

Geometry is read through the `PointCloudArtifactLoader` protocol. Two loaders ship:
`CompactArtifactLoader` (default; reads the existing `hole_points_compact.json` +
`hole_features.parquet`) and `SurfacePointArtifactLoader` (a dedicated per-surface
points + metadata artifact, schema documented in `surface_loader.py`). A new store
means one new loader — **scoring is untouched**.

## Limitations & next steps

* **Cost** — batch is O(N²) candidate checks; a full 540-hole run is ~30 min per
  config (most pairs rejected cheaply by the filter before any Chamfer work).
  Next: precompute per-surface KD-trees / cache, or restrict candidates by course
  metadata up front.
* **Chamfer is unweighted** — `point_weight` is carried on `SurfacePoint` but not
  yet used; density-aware or class-balanced Chamfer is a natural extension.
* **Tee/green elevation** is derived (tee from the compact origin, green = tee +
  `tee_to_green_elevation_change`); a dedicated absolute green elevation would be
  cleaner.
* **No learned weights** — surface weights are hand-set per config; the calibration
  report is the groundwork for fitting them against human plays-alike judgments.
```
