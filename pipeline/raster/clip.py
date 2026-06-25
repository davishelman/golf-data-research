"""Stage 7b — clip the course DEM to a hole buffer and reproject to metric CRS."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import mapping

from ..logging_config import get_logger

log = get_logger("raster.clip")


def clip_dem_to_buffer(raw_dem_path: Path, buffer_geom_4326, out_path: Path) -> Path:
    """Mask the DEM to a hole buffer (geometry supplied in EPSG:4326)."""
    with rasterio.open(raw_dem_path) as src:
        if src.crs is None:
            raise RuntimeError(f"DEM at {raw_dem_path} has no CRS.")
        buffer_gs = gpd.GeoSeries([buffer_geom_4326], crs="EPSG:4326").to_crs(src.crs)
        geoms = [mapping(g) for g in buffer_gs.geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True)
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1], width=out_image.shape[2], transform=out_transform,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_image)
    return out_path


def reproject_raster_to_crs(src_path: Path, dst_crs, dst_path: Path) -> Path:
    """Reproject a raster to ``dst_crs`` (passthrough copy if already matching)."""
    with rasterio.open(src_path) as src:
        if src.crs and src.crs.to_string() == str(dst_crs):
            data = src.read()
            profile = src.profile.copy()
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(data)
            return dst_path

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=dst_crs, transform=transform, width=width, height=height)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    return dst_path
