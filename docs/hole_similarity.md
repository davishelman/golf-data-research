# Hole Similarity Workflow

A decoupled data-science layer (`pipeline/modeling/`) that turns the per-hole 3D
point clouds into **one interpretable feature row per hole**, then clusters holes
and finds the most similar holes across all courses.

It depends only on the light stack — **pandas, numpy, duckdb, pyarrow,
scikit-learn** (UMAP optional). It does **not** import geopandas/rasterio, so you
can run the modeling phase without the geo toolchain.

## Quick start

```powershell
# 1. Build the feature table (reads the pipeline's aggregate/per-hole parquet)
python -m pipeline.modeling features

# 2. Cluster + nearest-neighbor search
python -m pipeline.modeling similarity --clusters 8 --neighbors 10

# both at once
python -m pipeline.modeling all
```

Equivalent flags exist on the main CLI (full env):

```powershell
python -m pipeline --build-hole-features
python -m pipeline --build-hole-similarity --clusters 8 --neighbors 10
```

Then open `notebooks/hole_similarity_research.ipynb`.

## Inputs

| Source | Used for |
|---|---|
| `courses/_index/all_holes.parquet` | identifiers + terrain stats (par, length, net elevation change) |
| `courses/<slug>/holes/hole_XX/features/hole_points.parquet` | per-hole points (preferred) |
| `courses/_index/all_hole_points.parquet` | fallback point source (via DuckDB) |

## Outputs (in `courses/_index/`)

| File | Contents |
|---|---|
| `hole_features.parquet` / `.csv` | one feature row per hole (540 × ~90) |
| `hole_clusters.parquet` / `.csv` | ids + `kmeans_cluster`, `agg_cluster`, `pca_1/2` (+ `umap_1/2` if UMAP installed) |
| `hole_similarity_examples.csv` | **v1** top-K nearest holes per hole (unweighted, no filters) |
| `hole_similarity_v2.csv` | **v2** length-aware top-K (cross-course, same-par, length-guarded) |

`hole_similarity_examples.csv` (v1) columns: `query_hole_id, query_course_slug,
query_hole_number, similar_hole_id, similar_course_slug, similar_hole_number,
distance, rank`.

`hole_similarity_v2.csv` columns: `query_hole_id, similar_hole_id, rank, distance,
query_length_m, similar_length_m, length_diff_m, same_par, same_course,
similarity_mode`. v1 is preserved unchanged; v2 is an additional file.

> These artifacts (and the per-hole point clouds) are **not committed** to git —
> they're distributed via a Hugging Face Dataset. Build an upload folder with
> `python -m pipeline.modeling hf-export --tier {lite,full}`; see
> [`docs/huggingface_artifact.md`](huggingface_artifact.md) for build/upload/
> download steps and how to run this notebook against downloaded data.

## Coordinate frame

From the pipeline's aligned point cloud:

- `x` = `x_aligned_m` — lateral; **x < 0 is LEFT, x > 0 is RIGHT**
- `y` = `y_aligned_m` — downrange distance from tee toward green (tee at `0`)
- `z` = `z_rel_m` — elevation relative to tee (tee elevation `0`)

Because every hole shares this frame, closeness in feature space means the holes
*play* alike.

## Feature families (≈90 columns)

All `*_pct` values are fractions in `[0,1]`; a feature is `NaN` when undefined for
a hole (e.g. a par-3 has no drive zone). The similarity step median-imputes NaNs.

### Identifiers (never scaled)
`course_slug, hole_number, hole_id, course_name, par, hole_length_m, hole_length_yd`

### Geometry
`x_min, x_max, y_min, y_max, hole_width_m (=x_max−x_min), hole_depth_m (=y_max−y_min),
point_count, valid_point_count, green_y_m`

### Elevation (relative to tee)
`z_min, z_max, z_mean, z_std, z_range, z_p10, z_p50, z_p90,
green_relative_elevation (mean z of green points), tee_to_green_elevation_change
(authoritative terrain stat; falls back to green_relative_elevation)`

### Label composition
For every label: `<label>_pct`. **Rough is collapsed** into `rough_pct =
rough_osm_pct + rough_inferred_pct`, while the originals are preserved. Labels:
`tee, green, fairway, rough_osm, rough_inferred, rough, bunker, water, trees,
cartpath, sand, unknown`.

### Zones (Y-distance from tee)
| Zone | Definition |
|---|---|
| `tee_zone` | 0–75 m |
| `drive_zone` | 175–300 m |
| `approach_zone` | final 175 m before green: `[green_y−175, green_y)` |
| `green_complex` | final 75 m before green: `[green_y−75, green_y)` |

`green_y` = mean Y of green-labeled points (≥3), else the 98th-percentile playable
Y. For each zone: `<zone>_{fairway,rough,trees,bunker,water,sand,cartpath}_pct`
(rough combined), `<zone>_mean_z`, `<zone>_z_range`.

### Left/right pressure
For `drive` and `approach` zones, and hazards `{trees, bunker, water}`:
`<zone>_<hazard>_left_pct`, `<zone>_<hazard>_right_pct`. Defined as the fraction
of zone points that are on that side **and** that hazard, so `left+right` equals
the hazard's share of the zone.

### Strategic shape
- `dogleg_score` = max |fairway centerline x| across 12 Y-bins, ÷ hole length.
  (Aligned tee→green line is x=0, so a bend pushes the centerline off-axis.)
- `fairway_centerline_shift` = mean fairway x in approach − mean fairway x in
  drive (signed lateral move).
- `fairway_width_drive_zone`, `fairway_width_approach_zone` = `p95(x) − p5(x)` of
  fairway points in that zone (robust width).
- `green_complex_{bunker,water,trees}_pct` = hazard share of the green complex.

## What each artifact means

- **`hole_features.parquet/csv`** — the model's input: one row per hole, ~90
  numeric features + identifiers. Human-readable; safe to load anywhere.
- **`hole_clusters.parquet/csv`** — each hole's `kmeans_cluster` and `agg_cluster`
  assignment plus 2D `pca_1/pca_2` (and `umap_1/umap_2` if UMAP is installed) for
  plotting. Two clustering methods are provided so you can sanity-check stability.
- **`hole_similarity_examples.csv`** — the precomputed top-K nearest holes per
  hole (the `exclude_same_course=False` view). Use it for quick lookups; use the
  notebook's live helper for cross-course-only queries.

## Modeling (`similarity.py`)

1. `feature_columns(df)` — numeric, non-identifier columns.
2. `feature_summary(df)` — per-feature dtype + missingness table (for inspection
   before imputation).
3. `build_feature_matrix` — median `SimpleImputer` (NaN → column median;
   all-missing → 0) → `StandardScaler`. Identifiers are not in the column list,
   so they are never scaled or modeled.
4. `run_pca` (2D for plots / N-D optional), `run_umap` (optional).
5. `cluster_kmeans`, `cluster_agglomerative` (default k=8).
6. `nearest_neighbor_table(df, X, k, exclude_same_course=False, same_par=False,
   max_length_diff_m=None, max_length_diff_pct=None)` and the single-hole
   `similar_holes(...)` — `NearestNeighbors` (Euclidean) over the scaled matrix;
   exclude self, and optionally exclude same-course holes, restrict to the query's
   par, and/or apply a **length guard** (drop candidates whose `hole_length_m`
   differs from the query by more than `max(max_length_diff_m,
   query_len * max_length_diff_pct)`). All filters combine; ranked 1..K.
7. Modes: `similar_holes_mode(df, cols, hole_id, mode)` /
   `nearest_neighbor_table_mode(df, cols, mode)` build a (weighted) matrix and
   apply a named mode from `SIMILARITY_MODES` in one call.

## v1 vs v2 similarity

**v1 (unrestricted).** Standardize all features, then take unweighted Euclidean
nearest neighbors. `hole_length_m` is just one feature among ~86, so strategic
shape, hazards, and elevation can outvote a large length gap — a ~400 m hole can
rank a ~300 m hole highly. v1 is preserved as the default of every function and as
`hole_similarity_examples.csv`.

**Why length matters more than a single standardized feature implies.** Two holes
that play alike are first of all *the same kind of shot sequence*, which is
dominated by length (driver-wedge vs driver-mid-iron vs three-shot). A 100 m
difference changes the whole strategy even if hazards look similar, so length
deserves more weight than one standardized column.

**v2 (length-aware).** Two mechanisms, both off by default:
- **Feature weighting** (`build_feature_matrix(df, cols, feature_weights=...)`):
  weights multiply standardized columns *after* scaling. `LENGTH_AWARE_WEIGHTS`
  up-weights `hole_length_m`, `green_y_m`, `hole_depth_m`, `par`, fairway widths,
  and elevation change; `hole_length_yd` is weighted **0** so length isn't
  double-counted (both metres and yards are present as features).
- **Length guard** (the `max_length_diff_*` params above): a hard filter so the
  main mode never returns a hole far off in length.

**Named modes** (`SIMILARITY_MODES`): `v1` (defaults), `length_weighted` (weights,
no hard filter), `same_par_length_guarded`, and
`cross_course_same_par_length_guarded` (the v2 export mode).

### Recommended presentation defaults

```python
similar_holes(df, X, hole_id,
              same_par=True, exclude_same_course=True,
              max_length_diff_m=35, max_length_diff_pct=0.12)   # with length-weighted X
# equivalently:
similar_holes_mode(df, cols, hole_id, mode="cross_course_same_par_length_guarded")
```

## Using the notebook

`notebooks/hole_similarity_research.ipynb` is the presentation surface. It loads
the three artifacts, validates them (missing values, PCA explained variance),
profiles + names the clusters, plots PCA/cluster/hazard charts, and provides:

```python
show_similar(hole_id, n=10, exclude_same_course=False, same_par=False)  # nearest holes (live)
compare_holes(query_hole_id, match_hole_id)                             # transposed feature diff
```

### Data source: local pipeline output or a downloaded HF artifact

The notebook loads through `pipeline.modeling.artifact_loader`, so it runs against
**either** your local `courses/_index/` (full pipeline run) **or** a downloaded
Hugging Face artifact folder. The first code cell selects the source:

```python
from pathlib import Path
ARTIFACT_ROOT = None   # auto-detect: local courses/_index, then artifact folders
# ARTIFACT_ROOT = Path("..") / "courses" / "_index"        # force local
# ARTIFACT_ROOT = Path("golf-data-research-artifacts")     # force a downloaded artifact
```

The tabular/cluster/similarity sections reproduce from any source. Visual
side-by-sides need each hole's compact point cloud — all holes locally or in a
*full*-tier artifact, but only the curated subset in a *lite* artifact (the
notebook's `viz_compare` helper skips missing holes with a message). Download +
load details: [`docs/huggingface_artifact.md`](huggingface_artifact.md).

Run (needs `matplotlib`):

```powershell
python -m pipeline.modeling all          # (re)build the artifacts first
jupyter notebook notebooks/hole_similarity_research.ipynb
# or execute headlessly:
python -m nbconvert --to notebook --execute --inplace notebooks/hole_similarity_research.ipynb
```

## Visual validation

Numbers can say two holes are close while they look nothing alike, so
`pipeline/modeling/visual_compare.py` plots holes side by side in the same
tee-relative aligned frame (tee at origin, green upward, **shared x/y scale**) to
sanity-check the model.

```python
from pipeline.modeling.visual_compare import plot_hole_comparison, save_hole_comparison
plot_hole_comparison("courses", ["augusta_national__01", "tpc_deere_run__13"], color_by="label")
```

Or from the CLI (saves a PNG under `courses/_index/visual_checks/`):

```powershell
python -m pipeline.modeling visual-check --hole-id augusta_national__01 --same-par --exclude-same-course --n 4
# colored by elevation instead of surface:
python -m pipeline.modeling visual-check --hole-id augusta_national__01 --color-by elevation
```

**Why it matters:** it confirms the engineered features capture real shape/hazard
similarity rather than spurious numeric closeness — the fastest way to catch a
bad match. Only generate a couple of checks at a time (don't batch hundreds).

**What the colors mean** (stable map; rough is intentionally subtle so hazards
read clearly):

| color | label | | color | label |
|---|---|---|---|---|
| dark green dot | tee | | gold | bunker |
| medium green | green | | blue | water |
| light green | fairway | | deep green | trees |
| olive (faint) | rough_osm | | brown | cartpath |
| pale grey-green (faint) | rough_inferred | | pale sand | sand |

`color_by="elevation"` instead colors points by `z` relative to the tee.

## Known limitations (v1)

- **Point-cloud visualization, not imagery** — the visual checks show *labeled
  surfaces* (what the model sees), not satellite/turf imagery; a feature missing
  from OSM is missing from the plot.
- **Background dominance** — points span the wide hole corridor, so `rough_pct`
  reflects background area (not penal rough) and can wash out subtle differences.
- **Same-course bias** — holes from one course share style, terrain, and OSM
  tagging, so top neighbors skew same-course. The notebook quantifies this and
  offers `exclude_same_course=True`.
- **OSM tagging inconsistency** — which features exist depends on how each course
  was mapped (some lack `rough_osm`, `cartpath`, etc.).
- **PCA is only a 2D projection** — used for plotting, not for distances;
  clustering and nearest-neighbors run on the full standardized space.
- **Engineered, not learned** — these are hand-built features encoding our
  assumptions, not a learned embedding of the raw point clouds.

## Recommended next steps

- Hazard-weighted distance (down-weight background, up-weight bunkers/water/trees).
- A cross-course-only similarity export next to the default CSV.
- Zone-emphasis similarity variants ("approach-similar" vs "off-the-tee-similar").
- UMAP embedding (optional dependency) for non-linear structure.
- Eventually: learned hole embeddings directly from the point clouds.

## Missing optional dependencies

- **pyarrow / scikit-learn** missing → a clear `ImportError` with the `pip install`
  command (these are required).
- **umap-learn** missing → UMAP is skipped with a log line (PCA still produced).

## Tests

`tests/test_modeling.py` covers (offline, synthetic): label percentages, rough
collapsing, zone splitting, left/right pressure, strategic feature presence, full
row assembly, matrix shape/imputation, nearest-neighbor shape (no self), and
clustering. Run:

```powershell
python -m pytest tests/test_modeling.py -q
```
