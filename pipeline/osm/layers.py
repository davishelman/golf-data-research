"""Stage 4 — split OSM features into canonical, normalized vector layers.

Each feature is assigned to exactly one canonical layer using FEATURE_TAG_RULES,
resolved in LAYER_RESOLUTION_ORDER so that, e.g., a bunker tagged ``natural=sand``
resolves to ``bunkers`` rather than ``sand``.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd

from ..constants import (
    FEATURE_LAYERS,
    FEATURE_TAG_RULES,
    LAYER_COURSE_BOUNDARY,
    LAYER_HOLE_CENTERLINES,
    LAYER_RESOLUTION_ORDER,
)
from ..logging_config import get_logger
from .boundary import BoundarySelection
from .fetch import OsmSource

log = get_logger("osm.layers")


def _predicate_mask(gdf: gpd.GeoDataFrame, predicate: dict) -> pd.Series:
    mask = pd.Series(True, index=gdf.index)
    for col, val in predicate.items():
        if col not in gdf.columns:
            return pd.Series(False, index=gdf.index)
        if val is True:
            mask &= gdf[col].notna()
        else:
            mask &= gdf[col] == val
    return mask


def _layer_mask(gdf: gpd.GeoDataFrame, layer: str) -> pd.Series:
    mask = pd.Series(False, index=gdf.index)
    for predicate in FEATURE_TAG_RULES.get(layer, []):
        mask |= _predicate_mask(gdf, predicate)
    return mask


def build_feature_layers(
    source: OsmSource,
    boundary: BoundarySelection,
) -> dict[str, gpd.GeoDataFrame]:
    """Return {canonical_layer: GeoDataFrame} for all feature layers present.

    Only features intersecting the selected boundary are kept (this filters out
    noisy off-course ``highway``/``landuse`` objects pulled by the broad query).
    """
    feats = source.features
    inside = feats[feats.intersects(boundary.geometry)].copy()
    log.info("features inside boundary: %d", len(inside))

    # Assign each row to its first-matching canonical layer (resolution order).
    assigned = pd.Series(pd.NA, index=inside.index, dtype="object")
    for layer in LAYER_RESOLUTION_ORDER:
        mask = _layer_mask(inside, layer) & assigned.isna()
        assigned[mask] = layer
    inside = inside.assign(canonical_layer=assigned)

    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer in FEATURE_LAYERS:
        sub = inside[inside["canonical_layer"] == layer].copy()
        if not sub.empty:
            sub["canonical_layer"] = layer
            layers[layer] = sub.reset_index(drop=True)
        else:
            layers[layer] = _empty_like(inside)
    counts = {k: len(v) for k, v in layers.items() if len(v)}
    log.info("layer counts: %s", counts)
    return layers


def _empty_like(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.iloc[0:0].copy()


# ---------------------------------------------------------------------------
# Source persistence (for reproducibility + caching)
# ---------------------------------------------------------------------------


_PROPERTY_WHITELIST = (
    "hole_number", "canonical_layer", "golf", "natural", "landuse", "highway",
    "water", "leisure", "ref", "name", "description", "par", "handicap",
)


def _slim(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep geometry + a stable, GeoJSON-friendly subset of attribute columns."""
    if gdf is None or gdf.empty:
        return gdf
    keep = [c for c in _PROPERTY_WHITELIST if c in gdf.columns]
    out = gdf[keep + ["geometry"]].copy()
    # Coerce list-valued cells (osmnx can return lists) to strings for GeoJSON.
    for c in keep:
        out[c] = out[c].apply(lambda v: ";".join(map(str, v)) if isinstance(v, list) else v)
    return out


def save_source_layers(
    paths,  # CoursePaths
    layers: dict[str, gpd.GeoDataFrame],
    main_holes: gpd.GeoDataFrame,
    boundary: BoundarySelection,
) -> None:
    from .. import storage  # local import avoids cycle at module load
    paths.ensure()
    storage.json_io.save_geojson(_slim(main_holes), paths.source_layer(LAYER_HOLE_CENTERLINES))
    storage.json_io.save_geojson(boundary.boundary[["geometry"]],
                                 paths.source_layer(LAYER_COURSE_BOUNDARY))
    for name, gdf in layers.items():
        storage.json_io.save_geojson(_slim(gdf), paths.source_layer(name))


def source_layers_exist(paths) -> bool:
    return paths.source_layer(LAYER_HOLE_CENTERLINES).exists()


def load_source_layers(paths) -> tuple[dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reload persisted source layers. Returns (feature_layers, main_holes, boundary)."""
    main_holes = gpd.read_file(paths.source_layer(LAYER_HOLE_CENTERLINES))
    boundary = gpd.read_file(paths.source_layer(LAYER_COURSE_BOUNDARY))
    layers: dict[str, gpd.GeoDataFrame] = {}
    for name in FEATURE_LAYERS:
        p = paths.source_layer(name)
        if p.exists():
            try:
                layers[name] = gpd.read_file(p)
            except Exception:  # noqa: BLE001
                layers[name] = gpd.GeoDataFrame(geometry=[], crs=main_holes.crs)
        else:
            layers[name] = gpd.GeoDataFrame(geometry=[], crs=main_holes.crs)
    return layers, main_holes, boundary


def get_layer_crs(layers: dict[str, gpd.GeoDataFrame], main_holes: gpd.GeoDataFrame):
    if main_holes.crs is not None:
        return main_holes.crs
    for gdf in layers.values():
        if gdf is not None and gdf.crs is not None:
            return gdf.crs
    return None
