"""Build a self-contained artifact folder for upload to a Hugging Face Dataset.

The git repo holds *code* (pipeline, modeling, notebook, docs, tests). The large
*generated data* (feature tables, similarity tables, and the 54.7M-point labeled
point clouds) is too big for GitHub, so it lives on a Hugging Face Dataset
instead. This module assembles that data — plus a manifest, schema, feature
dictionary, and dataset card — into a clean folder you can upload as-is.

Two tiers (see :func:`build_hf_artifact`):

* ``lite`` — the aggregate modeling tables + a small *curated* set of point
  clouds and one or two visual checks. Small enough to review quickly and to
  reproduce the notebook's tabular outputs.
* ``full`` — everything in ``lite`` plus the compact point cloud for **every**
  processed hole (and, behind explicit flags, the per-hole point Parquet files
  and/or the ~1 GB ``all_hole_points.parquet``). The complete data product.

Nothing here uploads anything or touches git. It only writes a local folder and
prints a size summary; you review and upload manually.

Design notes
------------
* Files are copied from an explicit allow-list — the repo tree is never globbed,
  so secrets (``.env``) cannot be swept in. A defensive secret guard
  (:func:`_safe_copy` / :func:`_verify_no_secrets`) backs that up.
* Only the light stack is needed (pandas + stdlib); no geopandas / sklearn, so
  this stays importable and unit-testable against a synthetic ``courses_root``.
* Optional source files (clusters, similarity, manifest, visual checks) are
  skipped with a warning when absent rather than crashing — only
  ``hole_features.parquet`` is strictly required.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..constants import LABEL_MAP_JSON, LABEL_PRIORITY, SCHEMA_VERSION
from ..logging_config import get_logger
from ..paths import COURSES_ROOT, CoursePaths, HolePaths, IndexPaths

log = get_logger("modeling.hf_export")

# --------------------------------------------------------------------------- #
# Project-level constants
# --------------------------------------------------------------------------- #

ARTIFACT_VERSION = "0.1.0"
SOURCE_REPO = "https://github.com/davishelman/golf-data-research"
# Recommended Hugging Face dataset repo (see docs/huggingface_artifact.md).
HF_DATASET_REPO = "davishelman/golf-data-research-artifacts"
ARTIFACT_TIERS = ("lite", "full")

#: Default curated point-cloud anchor for the lite tier.
DEFAULT_ANCHOR_HOLE = "augusta_national__01"

COORDINATE_FRAME = {
    "x": "tee-relative aligned meters, lateral (x<0 LEFT of tee->green line, x>0 RIGHT)",
    "y": "tee-relative aligned meters, downrange from tee (0) toward green (y>0)",
    "z": "relative elevation in meters from the selected tee anchor (tee = 0)",
    "alignment": "each hole rotated so the tee->green axis points +Y; green is x~0, y>0",
}

# Files that must never end up in a published artifact, even by accident.
_SECRET_EXACT = {"credentials.json", "secrets.json", "service_account.json"}
_SECRET_SUFFIXES = (".pem", ".key", ".pfx", ".p12")

# Threshold above which we record file size but skip the (slow) sha256.
_SHA256_MAX_BYTES = 50 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Secret / safety guards
# --------------------------------------------------------------------------- #

def _is_secret(path: Path) -> bool:
    """True for files that look like secrets and must never be exported."""
    name = path.name.lower()
    if name.startswith(".env"):
        return True
    if name in _SECRET_EXACT:
        return True
    return any(name.endswith(suf) for suf in _SECRET_SUFFIXES)


def _safe_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` -> ``dst`` unless ``src`` looks like a secret."""
    if _is_secret(src):
        raise RuntimeError(f"refusing to export secret-like file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _verify_no_secrets(root: Path) -> None:
    """Raise if any secret-like file slipped into the built artifact."""
    offenders = [p for p in root.rglob("*") if p.is_file() and _is_secret(p)]
    if offenders:
        raise RuntimeError(
            "secret-like files found in artifact output: "
            + ", ".join(str(p) for p in offenders)
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    """Hex sha256 for files under the size threshold, else ``None``."""
    if path.stat().st_size > _SHA256_MAX_BYTES:
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _compact_src(courses_root: Path, course_slug: str, hole_number: int) -> Path:
    cp = CoursePaths.for_slug(course_slug, courses_root)
    return HolePaths.for_hole(cp, hole_number).hole_points_compact


def _parquet_src(courses_root: Path, course_slug: str, hole_number: int) -> Path:
    cp = CoursePaths.for_slug(course_slug, courses_root)
    return HolePaths.for_hole(cp, hole_number).hole_points_parquet


# --------------------------------------------------------------------------- #
# Curated hole selection (lite tier)
# --------------------------------------------------------------------------- #

def curated_hole_ids(
    df: pd.DataFrame,
    similarity_v2_csv: Path | None = None,
    anchor_hole_id: str = DEFAULT_ANCHOR_HOLE,
    n_neighbors: int = 4,
) -> list[str]:
    """Pick a small, representative set of hole ids for the lite artifact.

    Deterministic selection: the anchor hole, its top length-aware (v2) neighbors
    (if the v2 table is available), plus one par-3, one water-heavy hole, and one
    terrain-heavy hole. Duplicates are removed while preserving order.
    """
    ids: list[str] = []
    have = set(df["hole_id"])

    if anchor_hole_id in have:
        ids.append(anchor_hole_id)
    elif len(df):  # fall back to a stable first row for synthetic/partial data
        ids.append(str(df.sort_values("hole_id").iloc[0]["hole_id"]))

    if similarity_v2_csv is not None and Path(similarity_v2_csv).exists() and ids:
        try:
            v2 = pd.read_csv(similarity_v2_csv)
            nbrs = (v2[v2["query_hole_id"] == ids[0]]
                    .sort_values("rank")["similar_hole_id"].astype(str).tolist())
            ids.extend(nbrs[:n_neighbors])
        except (KeyError, ValueError, OSError) as exc:  # pragma: no cover - defensive
            log.warning("could not read v2 neighbors from %s: %s", similarity_v2_csv, exc)

    def _pick(mask_or_series, *, by: str, largest: bool) -> str | None:
        sub = df[mask_or_series] if mask_or_series is not None else df
        sub = sub.dropna(subset=[by]) if by in sub.columns else sub.iloc[0:0]
        if sub.empty:
            return None
        row = sub.loc[sub[by].idxmax()] if largest else sub.loc[sub[by].idxmin()]
        return str(row["hole_id"])

    if "par" in df.columns and "hole_length_m" in df.columns:
        par3 = _pick(df["par"] == 3, by="hole_length_m", largest=False)
        if par3:
            ids.append(par3)
    if "water_pct" in df.columns:
        wet = _pick(None, by="water_pct", largest=True)
        if wet:
            ids.append(wet)
    if "z_range" in df.columns:
        hilly = _pick(None, by="z_range", largest=True)
        if hilly:
            ids.append(hilly)

    seen: set[str] = set()
    ordered = [h for h in ids if h in have and not (h in seen or seen.add(h))]
    return ordered


# --------------------------------------------------------------------------- #
# Feature dictionary (built from the actual feature columns present)
# --------------------------------------------------------------------------- #

_IDENTIFIER_COLS = (
    "course_slug", "hole_number", "hole_id", "course_name",
    "par", "hole_length_m", "hole_length_yd",
)
_ZONE_LABELS = {
    "tee_zone": "tee zone (0-75 m from tee)",
    "drive_zone": "drive zone (175-300 m from tee)",
    "approach_zone": "approach zone (final 175 m before the green)",
    "green_complex": "green complex (final 75 m before the green)",
}
_ZONE_SURFACES = ("fairway", "rough", "trees", "bunker", "water", "sand", "cartpath")

_GEOMETRY_DESC = {
    "x_min": "Leftmost aligned x of any hole point (m).",
    "x_max": "Rightmost aligned x of any hole point (m).",
    "y_min": "Nearest downrange y of any hole point (m).",
    "y_max": "Farthest downrange y of any hole point (m).",
    "hole_width_m": "Lateral extent x_max - x_min (m).",
    "hole_depth_m": "Downrange extent y_max - y_min (m).",
    "green_y_m": "Downrange y of the green centroid (m); zone anchor.",
    "point_count": "Number of labeled points sampled for the hole.",
    "valid_point_count": "Points with a finite elevation used in stats.",
}
_ELEVATION_DESC = {
    "z_min": "Minimum point elevation relative to the tee (m).",
    "z_max": "Maximum point elevation relative to the tee (m).",
    "z_mean": "Mean point elevation relative to the tee (m).",
    "z_std": "Std. dev. of point elevation relative to the tee (m).",
    "z_range": "z_max - z_min; vertical relief of the hole (m).",
    "z_p10": "10th-percentile relative elevation (m).",
    "z_p50": "Median relative elevation (m).",
    "z_p90": "90th-percentile relative elevation (m).",
    "green_relative_elevation": "Mean relative elevation of green points (m).",
    "tee_to_green_elevation_change": (
        "Authoritative tee->green elevation change (m); falls back to "
        "green_relative_elevation when unavailable."
    ),
}
_STRATEGIC_DESC = {
    "dogleg_score": (
        "Max |fairway centerline x| across 12 downrange bins, divided by hole "
        "length; 0 = straight, higher = sharper dogleg."
    ),
    "fairway_centerline_shift": (
        "Mean fairway x in the approach zone minus mean fairway x in the drive "
        "zone (signed lateral move of the corridor, m)."
    ),
    "fairway_width_drive_zone": "Robust fairway width p95(x)-p5(x) in the drive zone (m).",
    "fairway_width_approach_zone": "Robust fairway width p95(x)-p5(x) in the approach zone (m).",
}

_PRESSURE_RE = re.compile(r"^(drive|approach)_(trees|bunker|water)_(left|right)_pct$")
_ZONE_RE = re.compile(
    r"^(tee_zone|drive_zone|approach_zone|green_complex)_"
    r"(fairway|rough|trees|bunker|water|sand|cartpath)_pct$"
)
_ZONE_Z_RE = re.compile(
    r"^(tee_zone|drive_zone|approach_zone|green_complex)_(mean_z|z_range)$"
)
_LABEL_PCT_RE = re.compile(r"^([a-z_]+)_pct$")


def _classify_feature(name: str) -> dict[str, str]:
    """Return ``{group, unit, description}`` for one feature column."""
    if name in _IDENTIFIER_COLS:
        return {"group": "identifier", "unit": "", "description": "Hole / course identifier or key attribute (never modeled)."}
    if name in _STRATEGIC_DESC:
        unit = "score" if name == "dogleg_score" else "m"
        return {"group": "strategic_shape", "unit": unit, "description": _STRATEGIC_DESC[name]}

    m = _PRESSURE_RE.match(name)
    if m:
        zone, hazard, side = m.groups()
        return {
            "group": "left_right_pressure", "unit": "fraction[0,1]",
            "description": (
                f"Fraction of {zone}-zone points that are {hazard} on the "
                f"{side} side (x{'<0' if side == 'left' else '>0'}); "
                f"left+right = the hazard's share of the zone."
            ),
        }
    m = _ZONE_RE.match(name)
    if m:
        zone, surface = m.groups()
        return {
            "group": "zone_composition", "unit": "fraction[0,1]",
            "description": f"Fraction of {_ZONE_LABELS[zone]} points labeled {surface}.",
        }
    m = _ZONE_Z_RE.match(name)
    if m:
        zone, stat = m.groups()
        what = "mean relative elevation" if stat == "mean_z" else "relative elevation range"
        return {"group": "zone_elevation", "unit": "m",
                "description": f"{what.capitalize()} of points in the {_ZONE_LABELS[zone]} (m)."}

    if name in _ELEVATION_DESC:
        return {"group": "elevation", "unit": "m", "description": _ELEVATION_DESC[name]}
    if name in _GEOMETRY_DESC:
        unit = "count" if name.endswith("count") else "m"
        return {"group": "geometry", "unit": unit, "description": _GEOMETRY_DESC[name]}

    m = _LABEL_PCT_RE.match(name)
    if m:
        label = m.group(1)
        note = ""
        if label == "rough":
            note = " (rough_osm + rough_inferred combined)."
        return {"group": "label_composition", "unit": "fraction[0,1]",
                "description": f"Fraction of all hole points labeled {label}{note or '.'}"}

    return {"group": "other", "unit": "", "description": "Engineered hole feature."}


def build_feature_dictionary(df: pd.DataFrame) -> dict:
    """Describe every column of the hole-feature table, grouped by family."""
    columns = {}
    group_counts: dict[str, int] = {}
    for name in df.columns:
        info = _classify_feature(str(name))
        info["dtype"] = str(df[name].dtype)
        columns[str(name)] = info
        group_counts[info["group"]] = group_counts.get(info["group"], 0) + 1
    return {
        "description": (
            "One row per golf hole. *_pct columns are fractions in [0,1]; a value "
            "is NaN when undefined for a hole (e.g. a par-3 has no drive zone). "
            "The similarity step median-imputes NaNs and standardizes features; "
            "identifier columns are never scaled or modeled."
        ),
        "n_columns": len(columns),
        "group_counts": dict(sorted(group_counts.items())),
        "columns": columns,
    }


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def build_schema(df: pd.DataFrame) -> dict:
    """Describe the column layout of every tabular / point artifact."""
    feature_cols = {str(c): str(df[c].dtype) for c in df.columns}
    return {
        "description": "Column/format reference for the GolfDataScience artifact.",
        "tables": {
            "hole_features": {
                "path": "data/hole_features.parquet",
                "also": "data/hole_features.csv",
                "row_unit": "one golf hole",
                "n_rows": int(len(df)),
                "n_columns": len(feature_cols),
                "columns": feature_cols,
                "see_also": "metadata/feature_dictionary.json for per-column descriptions",
            },
            "all_holes": {
                "path": "data/all_holes.parquet",
                "row_unit": "one golf hole",
                "description": (
                    "Pipeline roll-up of identifiers + terrain stats (par, lengths, "
                    "tee/green elevation, slope, DEM provenance, quality flags). "
                    "Upstream of hole_features."
                ),
            },
            "hole_clusters": {
                "path": "data/hole_clusters.parquet",
                "row_unit": "one golf hole",
                "columns": {
                    "hole_id": "string", "course_slug": "string", "course_name": "string",
                    "hole_number": "int", "par": "int", "hole_length_m": "double",
                    "kmeans_cluster": "int", "agg_cluster": "int",
                    "pca_1": "double", "pca_2": "double",
                    "umap_1": "double (optional)", "umap_2": "double (optional)",
                },
            },
            "hole_similarity_examples": {
                "path": "data/hole_similarity_examples.csv",
                "row_unit": "one (query hole, neighbor) pair",
                "description": "v1 unweighted nearest neighbors (no filters).",
                "columns": {
                    "query_hole_id": "string", "query_course_slug": "string",
                    "query_hole_number": "int", "similar_hole_id": "string",
                    "similar_course_slug": "string", "similar_hole_number": "int",
                    "distance": "double", "rank": "int",
                },
            },
            "hole_similarity_v2": {
                "path": "data/hole_similarity_v2.csv",
                "row_unit": "one (query hole, neighbor) pair",
                "description": (
                    "v2 length-aware nearest neighbors (cross-course, same-par, "
                    "length-guarded)."
                ),
                "columns": {
                    "query_hole_id": "string", "similar_hole_id": "string",
                    "rank": "int", "distance": "double",
                    "query_length_m": "double", "similar_length_m": "double",
                    "length_diff_m": "double", "same_par": "bool",
                    "same_course": "bool", "similarity_mode": "string",
                },
            },
        },
        "point_cloud_parquet": {
            "path": "point_clouds/parquet/<course_slug>__<hole_number>.parquet",
            "row_unit": "one labeled DEM-cell point",
            "columns": {
                "hole_id": "string", "point_id": "int",
                "x_abs_m": "double", "y_abs_m": "double", "z_abs_m": "double",
                "x_rel_m": "double", "y_rel_m": "double", "z_rel_m": "double",
                "x_aligned_m": "double", "y_aligned_m": "double",
                "label": "string", "label_id": "int",
                "source": "string", "confidence": "double",
            },
            "note": "Present only in the full tier with --include-point-parquet.",
        },
        "point_cloud_compact_json": {
            "path": "point_clouds/compact/<course_slug>__<hole_number>.json",
            "structure": {
                "schema_version": "string",
                "hole_id": "string", "course_slug": "string", "hole_number": "int",
                "coordinate_system": "tee_relative_aligned_meters",
                "origin": "{type, x_abs_m, y_abs_m, z_abs_m} (selected tee anchor)",
                "alignment": "{enabled, axis, rotation_degrees}",
                "label_map": "{id: label_name}",
                "points": "array of [x_aligned_m, y_aligned_m, z_rel_m, label_id]",
            },
            "points_row": "[x_aligned_m, y_aligned_m, z_rel_m, label_id]",
        },
    }


# --------------------------------------------------------------------------- #
# Provenance + manifest
# --------------------------------------------------------------------------- #

def _course_status_counts(manifest: dict | None) -> dict[str, int]:
    counts = {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "other": 0}
    if not manifest:
        return counts
    for c in manifest.get("courses", []):
        counts["total"] += 1
        status = c.get("status", "other")
        counts[status if status in counts else "other"] += 1
    return counts


def _dem_products_by_course(manifest: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not manifest:
        return out
    for c in manifest.get("courses", []):
        dem = c.get("dem_type")
        if dem:
            out[dem] = out.get(dem, 0) + 1
    return out


def build_provenance(manifest: dict | None) -> dict:
    """Describe how the data was produced and its source licenses."""
    labeling_order = sorted(LABEL_PRIORITY, key=LABEL_PRIORITY.get, reverse=True)
    return {
        "generated_by": "pipeline.modeling.hf_export",
        "generated_at_utc": _utc_now(),
        "source_repo": SOURCE_REPO,
        "pipeline_schema_version": SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "data_sources": {
            "course_geometry": {
                "name": "OpenStreetMap",
                "accessed_via": "osmnx / Overpass API",
                "license": "Open Database License (ODbL) 1.0",
                "attribution": "(c) OpenStreetMap contributors",
                "notes": (
                    "Surface polygons (fairway/green/tee/bunker/water/trees/...) "
                    "depend on how each course was mapped; tagging is inconsistent "
                    "and incomplete in places."
                ),
            },
            "elevation": {
                "name": "OpenTopography global/US DEMs",
                "products": _dem_products_by_course(manifest) or {"USGS1m": None, "COP30": None},
                "license": (
                    "USGS 3DEP (USGS1m): public domain (US Gov). "
                    "Copernicus GLO-30 (COP30): free and open per ESA/Copernicus terms."
                ),
                "notes": "Bare-earth DEM (ground), not a DSM; no canopy/building height.",
            },
        },
        "coordinate_frame": COORDINATE_FRAME,
        "processing": {
            "tee_relative": "Points are translated so the selected tee anchor is the origin.",
            "alignment": "Each hole is rotated so the tee->green axis points +Y.",
            "labeling_priority_high_to_low": labeling_order,
            "rough_inferred": (
                "Untagged in-hole area becomes rough_inferred (flagged inferred, "
                "not OSM truth); rough_pct = rough_osm_pct + rough_inferred_pct."
            ),
            "tee_selection": (
                "Best-effort: OSM tees are not labeled championship/pro, so the tee "
                "nearest the centerline start is selected (method + confidence recorded "
                "per hole in the pipeline)."
            ),
            "point_resolution_m_default": 1.0,
        },
        "courses": _course_status_counts(manifest),
    }


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Dataset card (README.md + dataset_card.md)
# --------------------------------------------------------------------------- #

def _size_category(point_total: int | None) -> str:
    n = point_total or 0
    buckets = [
        (1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
        (1_000_000, "100K<n<1M"), (10_000_000, "1M<n<10M"),
        (100_000_000, "10M<n<100M"), (1_000_000_000, "100M<n<1B"),
    ]
    for ceiling, label in buckets:
        if n < ceiling:
            return label
    return "1B<n<10B"


def _dataset_card_body(stats: dict) -> str:
    labels_md = "\n".join(
        f"| {i} | `{name}` |" for i, name in sorted(LABEL_MAP_JSON.items(), key=lambda kv: int(kv[0]))
    )
    return f"""# Golf Hole Point Clouds & Engineered Features

Derived geospatial data from the
[`golf-data-research`]({SOURCE_REPO}) project: per-hole **tee-relative,
tee->green-aligned, labeled 3D point clouds** for PGA/championship-style golf
courses, plus an engineered one-row-per-hole feature table, clustering, and
hole-to-hole similarity rankings.

- **Courses processed:** {stats['courses_processed']} (of {stats['courses_total']} configured)
- **Holes:** {stats['holes_processed']}
- **Labeled points (total):** {stats['point_count_total']:,}
- **Artifact tier:** `{stats['tier']}`
- **Version:** {ARTIFACT_VERSION}

> **Not official course data.** Everything here is *derived from OpenStreetMap and
> public DEMs*. It is a best-effort reconstruction for analytics/ML, **not**
> official PGA Tour / course-architect data, and should not be treated as ground
> truth for course geometry, hazards, or yardages.

## What this dataset contains

```
data/                      aggregate modeling tables (parquet/csv) + manifests
point_clouds/compact/      per-hole compact JSON point clouds
point_clouds/parquet/      per-hole columnar point clouds (full tier, optional)
metadata/                  label_map, schema, feature_dictionary, provenance
dataset_manifest.json      machine-readable inventory + counts + checksums
```

| File | Rows | What it is |
|---|---|---|
| `data/hole_features.parquet` / `.csv` | one per hole | ~90 engineered features per hole (the model input) |
| `data/all_holes.parquet` | one per hole | pipeline roll-up: identifiers + terrain stats |
| `data/hole_clusters.parquet` | one per hole | KMeans/agglomerative clusters + PCA(/UMAP) 2-D coords |
| `data/hole_similarity_examples.csv` | query×neighbor | v1 unweighted nearest holes |
| `data/hole_similarity_v2.csv` | query×neighbor | v2 length-aware nearest holes |
| `point_clouds/compact/*.json` | array of points | `[x_aligned_m, y_aligned_m, z_rel_m, label_id]` per point |

## How the data was generated

A staged pipeline (OSM -> DEM -> per-hole labeled point cloud -> engineered
features -> similarity) documented in the source repo. Per hole: OSM surface
polygons are clipped to the hole, a DEM is clipped/reprojected, DEM cell centers
become labeled points, then points are translated to a tee-relative frame and
rotated so the tee->green axis points +Y.

## Coordinate frame

- `x` — {COORDINATE_FRAME['x']}
- `y` — {COORDINATE_FRAME['y']}
- `z` — {COORDINATE_FRAME['z']}

Because every hole shares this frame, closeness in feature space means the holes
*play* alike.

## Label definitions

| id | label |
|---|---|
{labels_md}

`rough` in the feature table is `rough_osm + rough_inferred` combined.

## Intended uses

- Hole-to-hole similarity / "find holes that play like this one".
- Clustering and exploratory analysis of course design.
- Teaching/portfolio examples of a geospatial -> ML feature pipeline.
- A labeled 3D point-cloud benchmark for golf-hole shape/terrain.

## Limitations & known data-quality issues

- **OSM tagging is inconsistent/incomplete** — which surfaces exist depends on how
  each course was mapped (some lack `rough_osm`, `cartpath`, etc.).
- **Inferred rough / background dominance** — untagged in-hole area is
  `rough_inferred`; points span the whole corridor, so `rough_pct` reflects
  background area, not penal rough.
- **DEM is bare-earth ground elevation**, not a canopy/building DSM — tree and
  structure *heights* are not represented.
- **Tee selection is best-effort** — OSM tees are not labeled championship/pro;
  the tee nearest the centerline start is used.
- **Skipped/failed courses are excluded** — only courses that processed cleanly
  to the expected hole count are present.
- **Visual checks are point-cloud renderings**, not satellite/turf imagery.
- **Similarity scores are engineered/model-derived**, not ground truth.

## Load it with pandas

```python
import pandas as pd, json

features = pd.read_parquet("data/hole_features.parquet")
sim = pd.read_csv("data/hole_similarity_v2.csv")

with open("point_clouds/compact/{stats['anchor_example']}.json") as fh:
    hole = json.load(fh)
pts = hole["points"]  # [[x_aligned_m, y_aligned_m, z_rel_m, label_id], ...]
```

From the [Hugging Face Hub]({"https://huggingface.co/datasets/" + HF_DATASET_REPO}):

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("{HF_DATASET_REPO}", "data/hole_features.parquet", repo_type="dataset")
features = pd.read_parquet(path)
```

## Use it with the code

Clone [`golf-data-research`]({SOURCE_REPO}), drop these files under
`courses/_index/` (and the compact JSONs under their course/hole paths), and run
`notebooks/hole_similarity_research.ipynb`. See `docs/huggingface_artifact.md` in
the repo for the exact mapping and a download helper.

## License & citation

- **Course geometry:** OpenStreetMap, (c) OpenStreetMap contributors, ODbL 1.0.
- **Elevation:** USGS 3DEP (public domain) and Copernicus GLO-30 (free/open).
- Derived artifacts are released for research/educational use; **retain OSM
  attribution and ODbL share-alike obligations** for any redistribution of the
  geometry-derived parts.

```
GolfDataScience — derived golf-hole point clouds & features.
Source: {SOURCE_REPO}  |  Data: OpenStreetMap (ODbL) + OpenTopography DEMs.
```

## Privacy / safety

No personal data. The dataset describes public golf-course terrain and surfaces
only; it contains no people, no PII, and no private location traces.
"""


def _frontmatter(stats: dict) -> str:
    return (
        "---\n"
        "license: other\n"
        "license_name: osm-odbl-1.0-plus-dem-terms\n"
        "language:\n  - en\n"
        "pretty_name: Golf Hole Point Clouds & Engineered Features\n"
        "tags:\n"
        "  - golf\n  - geospatial\n  - point-cloud\n  - golf-course\n"
        "  - openstreetmap\n  - digital-elevation-model\n  - sports-analytics\n"
        f"size_categories:\n  - {_size_category(stats['point_count_total'])}\n"
        "---\n\n"
    )


# --------------------------------------------------------------------------- #
# Size accounting + manifest assembly
# --------------------------------------------------------------------------- #

def _iter_files(root: Path) -> Iterable[Path]:
    return (p for p in root.rglob("*") if p.is_file())


def _section_of(rel: Path) -> str:
    parts = rel.parts
    if len(parts) == 1:
        return "root"
    if parts[0] == "point_clouds":
        return "point_clouds"
    return parts[0]


def _size_summary(root: Path) -> dict:
    by_section: dict[str, dict[str, int]] = {}
    total_bytes = 0
    total_files = 0
    for p in _iter_files(root):
        rel = p.relative_to(root)
        section = _section_of(rel)
        size = p.stat().st_size
        slot = by_section.setdefault(section, {"files": 0, "bytes": 0})
        slot["files"] += 1
        slot["bytes"] += size
        total_bytes += size
        total_files += 1
    return {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "total_human": _human_bytes(total_bytes),
        "by_section": {
            k: {**v, "human": _human_bytes(v["bytes"])}
            for k, v in sorted(by_section.items())
        },
    }


def _artifacts_list(root: Path, descriptions: dict[str, str]) -> list[dict]:
    """One record per file in data/ + metadata/ + root (point_clouds summarized)."""
    records = []
    for p in sorted(_iter_files(root)):
        rel = p.relative_to(root)
        if _section_of(rel) == "point_clouds":
            continue
        rel_posix = rel.as_posix()
        if rel_posix == "dataset_manifest.json":
            continue  # the manifest does not describe itself
        records.append({
            "path": rel_posix,
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
            "description": descriptions.get(rel_posix, descriptions.get(rel.name, "")),
        })
    return records


_ARTIFACT_DESCRIPTIONS = {
    "data/hole_features.parquet": "Engineered per-hole feature table (model input).",
    "data/hole_features.csv": "CSV mirror of hole_features.parquet.",
    "data/hole_clusters.parquet": "Per-hole cluster assignments + PCA/UMAP 2-D coords.",
    "data/hole_similarity_v2.csv": "v2 length-aware nearest-hole rankings.",
    "data/hole_similarity_examples.csv": "v1 unweighted nearest-hole rankings.",
    "data/all_holes.parquet": "Pipeline roll-up: identifiers + terrain stats per hole.",
    "data/all_courses_manifest.json": "Per-course processing status + DEM provenance.",
    "metadata/label_map.json": "Surface label id -> name map.",
    "metadata/schema.json": "Column/format reference for every artifact.",
    "metadata/feature_dictionary.json": "Per-column descriptions for the feature table.",
    "metadata/provenance.json": "Data sources, licenses, and processing notes.",
    "README.md": "Hugging Face dataset card.",
    "dataset_card.md": "Dataset card (body, no YAML front matter).",
}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def build_hf_artifact(
    tier: str = "lite",
    output_dir: Path | str | None = None,
    *,
    courses_root: Path = COURSES_ROOT,
    include_point_parquet: bool = False,
    include_all_points: bool = False,
    anchor_hole_id: str = DEFAULT_ANCHOR_HOLE,
    n_curated_neighbors: int = 4,
    max_visual_checks: int | None = None,
) -> dict:
    """Assemble a Hugging Face upload folder for the given ``tier``.

    Parameters
    ----------
    tier:
        ``"lite"`` (aggregate tables + curated point clouds) or ``"full"``
        (all compact point clouds; heavy point Parquet behind explicit flags).
    output_dir:
        Destination folder; defaults to ``hf_artifact_<tier>``.
    courses_root:
        Root of the pipeline outputs (``.../courses``); its ``_index`` holds the
        aggregate tables.
    include_point_parquet:
        Full tier only — also copy each hole's columnar ``hole_points.parquet``.
    include_all_points:
        Full tier only — also copy the aggregate ``all_hole_points.parquet`` (~1 GB).
    anchor_hole_id, n_curated_neighbors:
        Control the lite tier's curated point-cloud selection.
    max_visual_checks:
        Cap on visual-check PNGs to copy (default 2 for lite, all for full).

    Returns a summary dict (also printed): output path, counts, size summary,
    and any missing optional inputs.
    """
    if tier not in ARTIFACT_TIERS:
        raise ValueError(f"tier must be one of {ARTIFACT_TIERS}, got {tier!r}")

    courses_root = Path(courses_root)
    index = IndexPaths.for_root(courses_root)
    if not index.hole_features_parquet.exists():
        raise FileNotFoundError(
            f"{index.hole_features_parquet} not found. Build the feature table first:\n"
            "    python -m pipeline.modeling features"
        )

    out = Path(output_dir) if output_dir is not None else Path(f"hf_artifact_{tier}")
    if out.exists() and any(out.iterdir()):
        log.warning("output dir %s already exists and is not empty; files may be overwritten", out)
    data_dir = out / "data"
    meta_dir = out / "metadata"
    pc_compact = out / "point_clouds" / "compact"
    for d in (out, data_dir, meta_dir, pc_compact):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index.hole_features_parquet)
    if df.empty:
        raise ValueError("hole_features table is empty; nothing to export.")

    missing: list[str] = []

    # --- 1. aggregate data tables (allow-list copy) ------------------------- #
    data_specs: list[tuple[Path, str]] = [
        (index.hole_features_parquet, "hole_features.parquet"),
        (index.hole_features_csv, "hole_features.csv"),
        (index.hole_clusters_parquet, "hole_clusters.parquet"),
        (index.hole_similarity_v2_csv, "hole_similarity_v2.csv"),
        (index.hole_similarity_examples_csv, "hole_similarity_examples.csv"),
        (index.all_holes_parquet, "all_holes.parquet"),
        (index.all_courses_manifest, "all_courses_manifest.json"),
    ]
    for src, name in data_specs:
        if src.exists():
            _safe_copy(src, data_dir / name)
        else:
            missing.append(f"data/{name}")
            log.warning("optional input missing, skipping: %s", src)

    # --- 2. visual checks (a couple of sample PNGs) ------------------------- #
    cap = (2 if tier == "lite" else None) if max_visual_checks is None else max_visual_checks
    vc_count = 0
    if index.visual_checks.exists():
        pngs = sorted(index.visual_checks.glob("*.png"))
        if cap is not None:
            pngs = pngs[:cap]
        for png in pngs:
            _safe_copy(png, data_dir / "visual_checks" / png.name)
            vc_count += 1
    if vc_count == 0:
        missing.append("data/visual_checks/*.png")

    # --- 3. point clouds ---------------------------------------------------- #
    hole_lookup = {
        str(r.hole_id): (str(r.course_slug), int(r.hole_number))
        for r in df.itertuples(index=False)
    }
    if tier == "lite":
        selected = curated_hole_ids(
            df, index.hole_similarity_v2_csv, anchor_hole_id, n_curated_neighbors)
    else:
        selected = [str(h) for h in df["hole_id"]]

    compact_written = 0
    for hid in selected:
        slug, num = hole_lookup[hid]
        src = _compact_src(courses_root, slug, num)
        if src.exists():
            _safe_copy(src, pc_compact / f"{hid}.json")
            compact_written += 1
        else:
            log.warning("compact point cloud missing for %s: %s", hid, src)
    if compact_written == 0:
        missing.append("point_clouds/compact/*.json")

    parquet_written = 0
    if tier == "full" and include_point_parquet:
        pc_parquet = out / "point_clouds" / "parquet"
        pc_parquet.mkdir(parents=True, exist_ok=True)
        for hid in selected:
            slug, num = hole_lookup[hid]
            src = _parquet_src(courses_root, slug, num)
            if src.exists():
                _safe_copy(src, pc_parquet / f"{hid}.parquet")
                parquet_written += 1
            else:
                log.warning("point parquet missing for %s: %s", hid, src)

    all_points_included = False
    if tier == "full" and include_all_points:
        if index.all_hole_points_parquet.exists():
            _safe_copy(index.all_hole_points_parquet,
                       out / "point_clouds" / "all_hole_points.parquet")
            all_points_included = True
        else:
            missing.append("point_clouds/all_hole_points.parquet")
            log.warning("all_hole_points.parquet not found: %s", index.all_hole_points_parquet)

    # --- 4. metadata -------------------------------------------------------- #
    manifest_json = _load_json(index.all_courses_manifest)
    (meta_dir / "label_map.json").write_text(
        json.dumps(LABEL_MAP_JSON, indent=2), encoding="utf-8")
    (meta_dir / "schema.json").write_text(
        json.dumps(build_schema(df), indent=2), encoding="utf-8")
    (meta_dir / "feature_dictionary.json").write_text(
        json.dumps(build_feature_dictionary(df), indent=2), encoding="utf-8")
    (meta_dir / "provenance.json").write_text(
        json.dumps(build_provenance(manifest_json), indent=2), encoding="utf-8")

    # --- 5. headline stats (for cards + manifest) --------------------------- #
    course_counts = _course_status_counts(manifest_json)
    point_total = int(df["point_count"].sum()) if "point_count" in df.columns else None
    anchor_example = selected[0] if selected else (str(df.iloc[0]["hole_id"]))
    stats = {
        "tier": tier,
        "courses_total": course_counts["total"] or int(df["course_slug"].nunique()),
        "courses_processed": (
            course_counts["processed"] or int(df["course_slug"].nunique())),
        "holes_processed": int(df["hole_id"].nunique()),
        "point_count_total": point_total or 0,
        "anchor_example": anchor_example,
    }

    # --- 6. dataset card (README.md + dataset_card.md) ---------------------- #
    body = _dataset_card_body(stats)
    (out / "README.md").write_text(_frontmatter(stats) + body, encoding="utf-8")
    (out / "dataset_card.md").write_text(body, encoding="utf-8")

    # --- 7. size summary + manifest ----------------------------------------- #
    # Computed before the manifest is written, so it covers every *other* file;
    # the returned/logged totals below are recomputed once the manifest exists.
    size = _size_summary(out)
    size_for_manifest = {**size, "note": "byte totals exclude dataset_manifest.json itself"}
    manifest = {
        "project": "GolfDataScience",
        "version": ARTIFACT_VERSION,
        "tier": tier,
        "created_at_utc": _utc_now(),
        "source_repo": SOURCE_REPO,
        "hf_dataset_repo": HF_DATASET_REPO,
        "pipeline_schema_version": SCHEMA_VERSION,
        "courses_total": stats["courses_total"],
        "courses_processed": stats["courses_processed"],
        "courses_skipped": course_counts["skipped"],
        "courses_failed": course_counts["failed"],
        "holes_processed": stats["holes_processed"],
        "point_count_total": point_total,
        "coordinate_frame": COORDINATE_FRAME,
        "labels": LABEL_MAP_JSON,
        "point_clouds": {
            "compact_count": compact_written,
            "parquet_count": parquet_written,
            "all_hole_points_included": all_points_included,
            "selection": "all processed holes" if tier == "full" else "curated subset",
        },
        "visual_checks_included": vc_count,
        "missing_optional_inputs": missing,
        "size": size_for_manifest,
        "artifacts": _artifacts_list(out, _ARTIFACT_DESCRIPTIONS),
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    # --- 8. final safety sweep --------------------------------------------- #
    _verify_no_secrets(out)

    # Recompute now that the manifest exists, so the user sees the true total.
    size = _size_summary(out)

    summary = {
        "output_dir": str(out),
        "tier": tier,
        "courses_processed": stats["courses_processed"],
        "holes_processed": stats["holes_processed"],
        "point_count_total": point_total,
        "compact_point_clouds": compact_written,
        "point_parquet_files": parquet_written,
        "all_hole_points_included": all_points_included,
        "visual_checks": vc_count,
        "missing_optional_inputs": missing,
        "total_files": size["total_files"],
        "total_bytes": size["total_bytes"],
        "total_human": size["total_human"],
        "size_by_section": size["by_section"],
    }
    log.info("HF artifact (%s) -> %s | %s across %d files",
             tier, out, size["total_human"], size["total_files"])
    for section, info in size["by_section"].items():
        log.info("  %-14s %8s  (%d files)", section, info["human"], info["files"])
    if missing:
        log.info("  missing optional inputs: %s", ", ".join(missing))
    return summary
