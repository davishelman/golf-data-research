"""Synthetic course + raster fixtures for tests (no OSM / OpenTopography).

Geometry is built in a real UTM zone (EPSG:32617) near a chosen lat/lon so that
boundary scoring and CRS transforms behave like production, but every value is
deterministic and offline.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon, box

from pipeline.osm.fetch import OsmSource

CRS = "EPSG:32617"
ANCHOR_LAT = 33.50
ANCHOR_LON = -82.02

# Elevation model: z = BASE + SLOPE_PER_M * (northing - N0). Known slope.
BASE_ELEV = 100.0
SLOPE_PER_M = 0.1  # 10% grade -> ~5.71 degrees


def utm_origin() -> tuple[float, float]:
    t = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    e0, n0 = t.transform(ANCHOR_LON, ANCHOR_LAT)
    return float(e0), float(n0)


def _rect(cx, cy, hw, hh) -> Polygon:
    return box(cx - hw, cy - hh, cx + hw, cy + hh)


def hole_layout(e0: float, n0: float):
    """Return list of dicts describing 18 holes: number, tee_xy, green_xy."""
    holes = []
    for j in range(3):
        for i in range(6):
            n = j * 6 + i + 1
            x = e0 + 200 + i * 300
            y0 = n0 + 200 + j * 450
            holes.append({
                "hole_number": n,
                "tee_xy": (x, y0),
                "green_xy": (x, y0 + 300),
            })
    return holes


def build_course_features(e0: float, n0: float) -> gpd.GeoDataFrame:
    """Build an OSM-like projected GeoDataFrame for a clean 18-hole course."""
    cols = ["leisure", "golf", "natural", "landuse", "highway", "water",
            "ref", "name", "par", "handicap", "id", "element"]
    records: list[dict] = []
    geoms: list = []

    def add(geom, **kw):
        row = {c: None for c in cols}
        row.update(kw)
        records.append(row)
        geoms.append(geom)

    # Course boundary.
    boundary = box(e0 + 50, n0 + 50, e0 + 1850, n0 + 1550)
    add(boundary, leisure="golf_course", id=999, element="relation")

    for h in hole_layout(e0, n0):
        (tx, ty), (gx, gy) = h["tee_xy"], h["green_xy"]
        ref = str(h["hole_number"])
        # Centerline.
        add(LineString([(tx, ty), (gx, gy)]), golf="hole", ref=ref,
            name=f"Hole {h['hole_number']}", par=4, handicap=h["hole_number"])
        # Tee box just behind the start.
        add(_rect(tx, ty - 5, 6, 6), golf="tee")
        # Green around the end.
        add(_rect(gx, gy, 12, 12), golf="green")
        # Fairway corridor along the centerline.
        add(box(tx - 18, ty + 10, tx + 18, gy - 15), golf="fairway")
        # Greenside bunker just right of the green.
        add(_rect(gx + 20, gy - 10, 7, 7), golf="bunker")

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=CRS)
    return gdf


def make_osm_source(e0: float, n0: float) -> OsmSource:
    feats = build_course_features(e0, n0)
    return OsmSource(features=feats, crs=CRS, osm_id_col="id", element_col="element")


def write_ramp_dem(path: Path, e0: float, n0: float, res: float = 5.0) -> Path:
    """Write a north-rising ramp DEM (EPSG:32617) covering the whole course."""
    minx, miny = e0, n0
    maxx, maxy = e0 + 1900, n0 + 1600
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    transform = from_origin(minx, maxy, res, res)

    rows = np.arange(height)
    ys = maxy - (rows + 0.5) * res  # northing of each row center
    z_col = BASE_ELEV + SLOPE_PER_M * (ys - n0)
    dem = np.repeat(z_col[:, None], width, axis=1).astype("float32")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=CRS, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(dem, 1)
    return path
