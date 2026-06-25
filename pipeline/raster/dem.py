"""Stage 7a — DEM acquisition from OpenTopography.

One DEM is downloaded per course covering all hole buffers (+ pad). Re-runs reuse
the file unless forced. ``dem_source`` records whether it was local or freshly
downloaded.
"""

from __future__ import annotations

import os
from pathlib import Path

from pyproj import Transformer

from ..logging_config import get_logger

log = get_logger("raster.dem")

OPENTOPO_GLOBALDEM_URL = "https://portal.opentopography.org/API/globaldem"
OPENTOPO_USGSDEM_URL = "https://portal.opentopography.org/API/usgsdem"
DEM_OUTPUT_FORMAT = "GTiff"
USGS_CANONICAL = ("USGS1m", "USGS10m", "USGS30m")


class DemDownloadError(RuntimeError):
    """Raised when OpenTopography returns no usable DEM for the bbox."""


def normalize_dem_type(dem_type: str) -> str:
    """USGS dataset names are case-sensitive."""
    if dem_type.lower().startswith("usgs"):
        for canonical in USGS_CANONICAL:
            if dem_type.lower() == canonical.lower():
                return canonical
    return dem_type


def download_course_dem(
    bounds_projected: tuple[float, float, float, float],
    src_crs,
    dem_path: Path,
    dem_type: str,
    pad_meters: float = 200.0,
    force: bool = False,
) -> tuple[Path, str]:
    """Download a DEM covering ``bounds_projected`` (+ pad).

    Returns ``(dem_path, dem_source)`` where dem_source is "local" or
    "OpenTopography". Raises ``DemDownloadError`` on HTTP/content failure.
    """
    if dem_path.exists() and not force:
        log.info("using existing DEM: %s", dem_path)
        return dem_path, "local"

    import requests  # lazy: tests never hit the network

    api_key = os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        raise DemDownloadError(
            "No DEM found and OPENTOPOGRAPHY_API_KEY is not set. Either set the "
            f"env var or place a DEM GeoTIFF at: {dem_path}"
        )

    minx, miny, maxx, maxy = bounds_projected
    minx -= pad_meters
    miny -= pad_meters
    maxx += pad_meters
    maxy += pad_meters

    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(minx, miny)
    east, north = transformer.transform(maxx, maxy)

    normalized = normalize_dem_type(dem_type)
    is_usgs = normalized.lower().startswith("usgs")
    if is_usgs:
        endpoint = OPENTOPO_USGSDEM_URL
        params = {"datasetName": normalized}
    else:
        endpoint = OPENTOPO_GLOBALDEM_URL
        params = {"demtype": normalized}
    params.update({
        "south": south, "north": north, "west": west, "east": east,
        "outputFormat": DEM_OUTPUT_FORMAT, "API_Key": api_key,
    })

    log.info("requesting OpenTopography (%s) dem_type=%s bbox=(%.5f,%.5f,%.5f,%.5f)",
             endpoint.rsplit("/", 1)[-1], normalized, west, south, east, north)
    resp = requests.get(endpoint, params=params, timeout=300)
    if resp.status_code != 200 or not resp.content:
        raise DemDownloadError(
            f"OpenTopography DEM download failed: HTTP {resp.status_code}. "
            f"Body (truncated): {resp.text[:300]}"
        )
    if resp.content[:2] not in (b"II", b"MM"):
        raise DemDownloadError(
            f"OpenTopography returned non-GeoTIFF (first bytes: {resp.content[:32]!r})."
        )
    dem_path.parent.mkdir(parents=True, exist_ok=True)
    dem_path.write_bytes(resp.content)
    log.info("saved DEM -> %s (%.0f KB)", dem_path, dem_path.stat().st_size / 1024)
    return dem_path, "OpenTopography"
