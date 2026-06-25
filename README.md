# GolfDataScience

An enterprise-grade geospatial pipeline that ingests golf courses (lat/lon +
metadata) and produces, per hole: terrain statistics, clipped/reprojected DEM
rasters, and a **tee-relative, tee→green-aligned, labeled 3D point cloud**
(fairway / rough / bunker / water / green / tee / trees / cartpath / sand), plus
course manifests, quality reports, and aggregate CSV/Parquet/DuckDB exports.

## Pipeline stages

```
config -> osm(fetch -> boundary -> detect/validate holes -> layers -> assignment)
       -> raster(course DEM -> clip -> reproject -> slope/sample)
       -> features(anchors -> labels -> transforms -> point cloud)
       -> terrain(stats)
       -> storage(json/geojson/jsonl/parquet/duckdb) + plotting
```

Dirty courses (not exactly the expected hole count) are **skipped with an
explicit `quality_report.json`**, never silently dropped. One failing hole or
course never crashes the batch.

## Package layout

```
pipeline/
├── cli.py / __main__.py        # CLI + entry point
├── orchestrator.py             # staged run_course(course, options)
├── config.py                   # CourseConfig + loaders
├── constants.py                # labels, priorities, OSM tag rules, schema version
├── schemas.py                  # dataclasses: HoleIdentity, HoleAnchors, TerrainSummary, FeaturePoint, RunOptions
├── quality.py                  # QualityReport / QualityIssue
├── logging_config.py           # structured logging
├── geometry.py                 # shapely helpers (ensure_linestring, nearest_hole)
├── paths.py                    # CoursePaths / HolePaths / IndexPaths
├── plotting.py                 # matplotlib/plotly (optional; never blocks data)
├── exports.py / export_csv.py  # aggregate index (csv/parquet/duckdb) + legacy CSV
├── osm/        fetch, boundary, holes, layers, assignment
├── raster/     dem, clip, slope, sampling
├── features/   anchors, labels, transforms, point_cloud
├── terrain/    stats
└── storage/    json_io, parquet_io, duckdb_writer
tests/          unit + offline synthetic integration tests
```

## Setup

```powershell
pip install geopandas osmnx numpy pandas matplotlib plotly rasterio requests pyproj shapely python-dotenv pyarrow duckdb pytest
```

Create a `.env` at the repo root with your OpenTopography key:

```
OPENTOPOGRAPHY_API_KEY=your_key_here
```

(`pyarrow`/`duckdb`/`plotly` are optional — the pipeline degrades gracefully if
they're missing; Parquet/DuckDB/3D-HTML are simply skipped.)

## Running

```powershell
python -m pipeline --all                       # process every course
python -m pipeline -c augusta_national          # one course
python -m pipeline -c augusta_national -c pebble_beach
python -m pipeline --list                       # list configured slugs
```

### Useful flags

| Flag | Effect |
|---|---|
| `--refetch-osm` | Force OSM re-fetch (ignores cached `source/`). |
| `--redownload-dem` | Force DEM re-download. |
| `--rebuild-points` | Regenerate point clouds even if present. |
| `--strict-18` (default) / `--allow-dirty` | Skip vs. process courses with != expected holes. |
| `--skip-plots` / `--only-plots` | Headless data only / re-render plots only. |
| `--export-csv` | Build/refresh aggregate CSV + manifest index. |
| `--export-parquet` | Also build Parquet + DuckDB aggregate exports. |
| `--point-resolution <m>` | Target point spacing (default 1.0; never finer than the DEM). |
| `--max-points <n>` | Per-hole point guardrail (default 250000). |
| `--log-level DEBUG` | Verbosity. |

## Artifacts per course

```
courses/<slug>/
├── course_manifest.json        # status, holes[], crs, dem provenance, quality flags
├── quality_report.json         # machine-readable issues (errors/warnings/info)
├── boundary_selection.json     # how the course boundary was chosen
├── course_summary.json         # legacy roll-up (backward compatible)
├── course_overview.png
├── source/<layer>.geojson      # canonical OSM layers (unclipped) — full-course truth
└── holes/hole_XX/
    ├── vectors/<layer>.geojson  # assigned + clipped per-hole layers + assignment.json
    ├── dem/dem_clipped_projected.tif
    ├── stats/terrain_summary.json + anchors.json
    ├── features/
    │   ├── hole_points.jsonl            # one labeled point per line (streamed)
    │   ├── hole_points_compact.json     # [x_aligned, y_aligned, z_rel, label_id]
    │   ├── hole_points.parquet          # columnar (if pyarrow present)
    │   └── label_map.json
    └── plots/                  # heatmap, slope, profile, overview, 3d_terrain.html
```

Aggregate index (`courses/_index/`): `all_holes.csv`, `all_holes.parquet`,
`all_hole_points.parquet`, `all_courses_manifest.json`, `golf.duckdb`.

## The 3D point representation

Each in-hole DEM cell center becomes a point with:

- **absolute** `x_abs_m, y_abs_m, z_abs_m` (UTM meters),
- **tee-relative** `x_rel_m, y_rel_m, z_rel_m` (selected tee anchor = origin;
  `z_rel = z_abs - tee_elevation`),
- **aligned** `x_aligned_m, y_aligned_m` (rotated so tee→green points +Y;
  invariant: green is `x≈0, y>0`),
- a `label` + `label_id`, `source`, and `confidence`.

Labeling is deterministic by priority: green > tee > bunker > water > fairway >
cartpath > sand > trees > rough_osm > rough_inferred > unknown. Untagged in-hole
area becomes `rough_inferred` (flagged as inferred, not OSM truth).

DuckDB example:

```python
import duckdb
con = duckdb.connect("courses/_index/golf.duckdb")
con.sql("SELECT label, count(*) FROM hole_points GROUP BY label ORDER BY 2 DESC")
```

## Data quality notes (OSM realities)

- **Tees** aren't labeled "pro/championship" in OSM; the selected tee is the tee
  feature nearest the centerline start (method + confidence recorded in
  `anchors.json`).
- **Rough** is often unmapped → `rough_inferred` vs explicit `rough_osm`.
- **Trees** come from `natural=wood` / `landuse=forest` / `tree_row` / `tree`;
  treated as a best-effort layer, not guaranteed truth.
- Raster elevation is a **DEM** (ground), not a DSM (no canopy height).

## Hole similarity (modeling)

A decoupled data-science layer (`pipeline/modeling/`) turns the point clouds into
one interpretable feature row per hole, then clusters holes and finds similar
holes across courses. It needs only the light stack (pandas/numpy/duckdb/pyarrow/
scikit-learn; UMAP optional) — no geopandas.

```powershell
python -m pipeline.modeling features      # -> courses/_index/hole_features.parquet/.csv
python -m pipeline.modeling similarity    # -> hole_clusters.* + hole_similarity_examples.csv
python -m pipeline.modeling all           # both
# (equivalent: python -m pipeline --build-hole-features / --build-hole-similarity)
```

Outputs land in `courses/_index/`: `hole_features.parquet/csv`,
`hole_clusters.parquet/csv`, `hole_similarity_examples.csv`. Explore them in
`notebooks/hole_similarity_research.ipynb`. Full reference + feature formulas:
[`docs/hole_similarity.md`](docs/hole_similarity.md).

## Testing

```powershell
python -m pytest tests/ -q                    # full suite (needs the geo stack)
python -m pytest tests/test_modeling.py -q    # modeling only (light stack)
```

The suite is fully offline (synthetic course + synthetic raster; OSM and
OpenTopography are monkeypatched) and covers ref parsing, hole dedup/validation,
feature assignment, label priority, coordinate transforms, slope, the source
cache round-trip, an end-to-end run, and the modeling layer (features, zones,
rough collapsing, left/right pressure, similarity + nearest-neighbor).
```
