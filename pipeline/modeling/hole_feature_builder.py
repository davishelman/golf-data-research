"""Build one interpretable feature row per hole from the labeled point clouds.

Coordinate frame (from the pipeline's aligned point cloud):
  * x = ``x_aligned_m``  -> lateral; x < 0 is LEFT, x > 0 is RIGHT
  * y = ``y_aligned_m``  -> downrange distance from tee toward the green (tee at 0)
  * z = ``z_rel_m``      -> elevation relative to the tee (tee elevation = 0)

Identifiers + terrain stats are read from ``courses/_index/all_holes.parquet``.
Per-hole points are read from each hole's ``features/hole_points.parquet``
(efficient natural partition); if a hole's file is missing we fall back to the
aggregate ``all_hole_points.parquet`` via DuckDB.

All percentage features are fractions in [0, 1]. A feature is ``NaN`` when it is
undefined for a hole (e.g. an empty zone) — the similarity step imputes those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..logging_config import get_logger
from ..paths import COURSES_ROOT, CoursePaths, HolePaths, IndexPaths
from . import POINT_LABELS, ROUGH_LABELS

log = get_logger("modeling.features")

# --- modeling constants ---------------------------------------------------

# Labels reported per-zone (rough is the COMBINED rough class).
ZONE_LABELS: tuple[str, ...] = (
    "fairway", "rough", "trees", "bunker", "water", "sand", "cartpath",
)
# Hazards measured for left/right pressure.
PRESSURE_LABELS: tuple[str, ...] = ("trees", "bunker", "water")

TEE_ZONE = (0.0, 75.0)
DRIVE_ZONE = (175.0, 300.0)
APPROACH_DEPTH = 175.0      # final 175 m before the green
GREEN_COMPLEX_DEPTH = 75.0  # final 75 m before the green

_MIN_GREEN_POINTS = 3       # need this many green points to trust a green anchor
_MIN_FAIRWAY_FOR_SHAPE = 20  # min fairway points before computing shape features
_DOGLEG_BINS = 12


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "pyarrow is required to read/write Parquet. Install it with:\n"
            "    pip install pyarrow"
        ) from exc


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _pct(mask: np.ndarray, total: int) -> float:
    return _safe_div(int(np.count_nonzero(mask)), total)


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


# ---------------------------------------------------------------------------
# Per-hole point container
# ---------------------------------------------------------------------------


@dataclass
class HolePoints:
    x: np.ndarray            # x_aligned_m (left/right)
    y: np.ndarray            # y_aligned_m (downrange)
    z: np.ndarray            # z_rel_m (elevation vs tee)
    label: np.ndarray        # str labels

    @property
    def n(self) -> int:
        return int(self.x.size)

    def zone_mask(self, lo: float, hi: float) -> np.ndarray:
        return (self.y >= lo) & (self.y < hi)


# ---------------------------------------------------------------------------
# Feature groups (pure functions — each returns a flat dict)
# ---------------------------------------------------------------------------


def green_y_value(p: HolePoints) -> float:
    """Downrange Y of the green (its mean Y), else the 98th-pctile playable Y."""
    green = p.y[p.label == "green"]
    if green.size >= _MIN_GREEN_POINTS:
        return float(np.nanmean(green))
    if p.n:
        return float(np.nanpercentile(p.y, 98))
    return float("nan")


def geometry_features(p: HolePoints) -> dict[str, float]:
    if p.n == 0:
        return {k: float("nan") for k in
                ("x_min", "x_max", "y_min", "y_max", "hole_width_m", "hole_depth_m")} | \
               {"point_count": 0, "valid_point_count": 0}
    finite = np.isfinite(p.x) & np.isfinite(p.y) & np.isfinite(p.z)
    return {
        "x_min": float(np.nanmin(p.x)),
        "x_max": float(np.nanmax(p.x)),
        "y_min": float(np.nanmin(p.y)),
        "y_max": float(np.nanmax(p.y)),
        "hole_width_m": float(np.nanmax(p.x) - np.nanmin(p.x)),
        "hole_depth_m": float(np.nanmax(p.y) - np.nanmin(p.y)),
        "point_count": int(p.n),
        "valid_point_count": int(np.count_nonzero(finite)),
    }


def elevation_features(p: HolePoints) -> dict[str, float]:
    z = p.z[np.isfinite(p.z)]
    if z.size == 0:
        keys = ("z_min", "z_max", "z_mean", "z_std", "z_range",
                "z_p10", "z_p50", "z_p90", "green_relative_elevation")
        return {k: float("nan") for k in keys}
    green_z = p.z[p.label == "green"]
    return {
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "z_range": float(np.max(z) - np.min(z)),
        "z_p10": float(np.percentile(z, 10)),
        "z_p50": float(np.percentile(z, 50)),
        "z_p90": float(np.percentile(z, 90)),
        # Green elevation relative to the tee (tee z == 0 by construction).
        "green_relative_elevation": float(np.nanmean(green_z)) if green_z.size else float("nan"),
    }


def _combined_label_mask(label: np.ndarray, lab: str) -> np.ndarray:
    """Mask for a label; ``rough`` collapses rough_osm + rough_inferred."""
    if lab == "rough":
        return np.isin(label, ROUGH_LABELS)
    return label == lab


def label_features(p: HolePoints) -> dict[str, float]:
    total = p.n
    out: dict[str, float] = {}
    for lab in POINT_LABELS:
        out[f"{lab}_pct"] = _pct(p.label == lab, total)
    # Combined rough preserves the split (rough_osm_pct / rough_inferred_pct above).
    out["rough_pct"] = out.get("rough_osm_pct", float("nan"))
    out["rough_pct"] = _pct(np.isin(p.label, ROUGH_LABELS), total)
    return out


def _zone_label_block(p: HolePoints, lo: float, hi: float, prefix: str) -> dict[str, float]:
    mask = p.zone_mask(lo, hi)
    n = int(np.count_nonzero(mask))
    block: dict[str, float] = {}
    sub_label = p.label[mask]
    sub_z = p.z[mask]
    for lab in ZONE_LABELS:
        block[f"{prefix}_{lab}_pct"] = _pct(_combined_label_mask(sub_label, lab), n) if n else float("nan")
    zfin = sub_z[np.isfinite(sub_z)]
    block[f"{prefix}_mean_z"] = float(np.mean(zfin)) if zfin.size else float("nan")
    block[f"{prefix}_z_range"] = float(np.max(zfin) - np.min(zfin)) if zfin.size else float("nan")
    return block


def zone_features(p: HolePoints, green_y: float) -> dict[str, float]:
    zones = {
        "tee_zone": TEE_ZONE,
        "drive_zone": DRIVE_ZONE,
        "approach_zone": (max(0.0, green_y - APPROACH_DEPTH), green_y),
        "green_complex": (max(0.0, green_y - GREEN_COMPLEX_DEPTH), green_y),
    }
    out: dict[str, float] = {}
    for name, (lo, hi) in zones.items():
        out.update(_zone_label_block(p, lo, hi, name))
    return out


def left_right_features(p: HolePoints, green_y: float) -> dict[str, float]:
    """Left/right hazard pressure in the drive and approach zones.

    Each ``*_left_pct`` / ``*_right_pct`` is the fraction of points in that zone
    that are both on that side (x<0 / x>0) and that hazard label. So left+right
    for a label equals that label's share of the zone.
    """
    zones = {
        "drive": DRIVE_ZONE,
        "approach": (max(0.0, green_y - APPROACH_DEPTH), green_y),
    }
    out: dict[str, float] = {}
    for zname, (lo, hi) in zones.items():
        mask = p.zone_mask(lo, hi)
        n = int(np.count_nonzero(mask))
        x = p.x[mask]
        lab = p.label[mask]
        for hz in PRESSURE_LABELS:
            is_hz = _combined_label_mask(lab, hz)
            out[f"{zname}_{hz}_left_pct"] = _pct(is_hz & (x < 0), n) if n else float("nan")
            out[f"{zname}_{hz}_right_pct"] = _pct(is_hz & (x > 0), n) if n else float("nan")
    return out


def _fairway_centerline(p: HolePoints, green_y: float) -> Optional[np.ndarray]:
    """Mean fairway x in evenly spaced y-bins from tee to green (the centerline)."""
    fmask = p.label == "fairway"
    if int(np.count_nonzero(fmask)) < _MIN_FAIRWAY_FOR_SHAPE or not np.isfinite(green_y) or green_y <= 0:
        return None
    fy = p.y[fmask]
    fx = p.x[fmask]
    edges = np.linspace(0.0, green_y, _DOGLEG_BINS + 1)
    centers = []
    for i in range(_DOGLEG_BINS):
        b = (fy >= edges[i]) & (fy < edges[i + 1])
        if np.count_nonzero(b) >= 5:
            centers.append(np.mean(fx[b]))
    return np.asarray(centers) if centers else None


def _fairway_width(p: HolePoints, lo: float, hi: float) -> float:
    fmask = (p.label == "fairway") & p.zone_mask(lo, hi)
    fx = p.x[fmask]
    if fx.size < 5:
        return float("nan")
    return float(np.percentile(fx, 95) - np.percentile(fx, 5))


def _fairway_mean_x(p: HolePoints, lo: float, hi: float) -> float:
    fx = p.x[(p.label == "fairway") & p.zone_mask(lo, hi)]
    return float(np.mean(fx)) if fx.size else float("nan")


def strategic_features(p: HolePoints, green_y: float) -> dict[str, float]:
    approach = (max(0.0, green_y - APPROACH_DEPTH), green_y)
    gc = (max(0.0, green_y - GREEN_COMPLEX_DEPTH), green_y)

    # Dogleg: max absolute lateral excursion of the fairway centerline from the
    # straight tee->green line (x=0), normalized by hole length.
    centerline = _fairway_centerline(p, green_y)
    if centerline is not None and np.isfinite(green_y) and green_y > 0:
        dogleg_score = float(np.nanmax(np.abs(centerline)) / green_y)
    else:
        dogleg_score = float("nan")

    # Net lateral shift of the fairway from the drive zone to the approach zone.
    shift = _fairway_mean_x(p, *approach) - _fairway_mean_x(p, *DRIVE_ZONE)

    gc_mask = p.zone_mask(*gc)
    n_gc = int(np.count_nonzero(gc_mask))
    gc_label = p.label[gc_mask]

    return {
        "dogleg_score": dogleg_score,
        "fairway_centerline_shift": float(shift),
        "fairway_width_drive_zone": _fairway_width(p, *DRIVE_ZONE),
        "fairway_width_approach_zone": _fairway_width(p, *approach),
        "green_complex_bunker_pct": _pct(gc_label == "bunker", n_gc) if n_gc else float("nan"),
        "green_complex_water_pct": _pct(gc_label == "water", n_gc) if n_gc else float("nan"),
        "green_complex_trees_pct": _pct(gc_label == "trees", n_gc) if n_gc else float("nan"),
    }


def build_one_hole_features(p: HolePoints, identifiers: dict, terrain: dict) -> dict:
    """Assemble the full feature row for a single hole."""
    green_y = green_y_value(p)
    row: dict = dict(identifiers)
    row["green_y_m"] = green_y
    row.update(geometry_features(p))
    row.update(elevation_features(p))
    row.update(label_features(p))
    row.update(zone_features(p, green_y))
    row.update(left_right_features(p, green_y))
    row.update(strategic_features(p, green_y))

    # Prefer authoritative terrain stat for tee->green change; fall back to points.
    net = terrain.get("net_elevation_change_m")
    row["tee_to_green_elevation_change"] = (
        _nan(net) if net is not None and np.isfinite(_nan(net))
        else row.get("green_relative_elevation", float("nan"))
    )
    return row


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_hole_index(index: IndexPaths) -> pd.DataFrame:
    if not index.all_holes_parquet.exists():
        raise FileNotFoundError(
            f"{index.all_holes_parquet} not found. Build aggregate outputs first:\n"
            "    python -m pipeline --export-parquet"
        )
    df = pd.read_parquet(index.all_holes_parquet)
    if df.empty:
        raise ValueError("all_holes.parquet is empty — no processed holes to model.")
    return df


def _read_points_per_hole(hp: HolePaths) -> Optional[HolePoints]:
    pq = hp.hole_points_parquet
    if not pq.exists():
        return None
    df = pd.read_parquet(pq, columns=["x_aligned_m", "y_aligned_m", "z_rel_m", "label"])
    return _to_points(df)


def _read_points_from_aggregate(index: IndexPaths, hole_id: str) -> Optional[HolePoints]:
    if not index.all_hole_points_parquet.exists():
        return None
    import duckdb
    con = duckdb.connect()
    try:
        df = con.execute(
            "SELECT x_aligned_m, y_aligned_m, z_rel_m, label "
            "FROM read_parquet(?) WHERE hole_id = ?",
            [str(index.all_hole_points_parquet), hole_id],
        ).fetch_df()
    finally:
        con.close()
    return _to_points(df) if not df.empty else None


def _to_points(df: pd.DataFrame) -> HolePoints:
    return HolePoints(
        x=df["x_aligned_m"].to_numpy(dtype="float64"),
        y=df["y_aligned_m"].to_numpy(dtype="float64"),
        z=df["z_rel_m"].to_numpy(dtype="float64"),
        label=df["label"].astype(str).to_numpy(),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_hole_features(courses_root: Path = COURSES_ROOT) -> Path:
    """Build per-hole features and write hole_features.parquet + .csv."""
    _require_pyarrow()
    index = IndexPaths.for_root(courses_root)
    index.ensure()
    holes = _load_hole_index(index)
    log.info("building features for %d holes", len(holes))

    id_cols = ["course_slug", "hole_number", "hole_id", "course_name",
               "par", "hole_length_m", "hole_length_yd"]
    rows: list[dict] = []
    missing_points = 0

    for _, h in holes.iterrows():
        slug = str(h["course_slug"])
        hole_number = int(h["hole_number"])
        hp = HolePaths.for_hole(CoursePaths.for_slug(slug, courses_root=courses_root), hole_number)
        pts = _read_points_per_hole(hp)
        if pts is None:
            pts = _read_points_from_aggregate(index, str(h["hole_id"]))
        if pts is None or pts.n == 0:
            missing_points += 1
            log.warning("no points for %s; skipping", h["hole_id"])
            continue

        identifiers = {c: h[c] for c in id_cols if c in holes.columns}
        terrain = {"net_elevation_change_m": h.get("net_elevation_change_m")}
        rows.append(build_one_hole_features(pts, identifiers, terrain))

    if not rows:
        raise RuntimeError("No hole features were built (no point data found).")

    features = pd.DataFrame(rows)
    # Stable column order: identifiers first, then features sorted.
    front = [c for c in id_cols if c in features.columns] + ["green_y_m"]
    rest = sorted(c for c in features.columns if c not in front)
    features = features[front + rest]

    features.to_parquet(index.hole_features_parquet, index=False)
    features.to_csv(index.hole_features_csv, index=False)
    log.info("wrote %d hole feature rows -> %s (%d holes had no points)",
             len(features), index.hole_features_parquet, missing_points)
    return index.hole_features_parquet
