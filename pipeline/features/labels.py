"""Point labeling by deterministic polygon-priority classification.

Each point is tested against the per-hole clipped feature layers in priority
order (green > tee > bunker > water > fairway > cartpath > sand > trees >
rough_osm). The first hit wins. Points inside the hole buffer that match nothing
become ``rough_inferred`` (if enabled) or ``unknown``.

Uses vectorized spatial joins (GeoPandas spatial index) — never a naive
point x polygon double loop.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np

from ..constants import (
    LABEL_IDS,
    LABEL_PRIORITY,
    LAYER_TO_LABEL,
    SOURCE_INFERRED,
    SOURCE_OSM_PREFIX,
    SOURCE_UNKNOWN,
)
from ..logging_config import get_logger

log = get_logger("features.labels")

_FEATURE_CONFIDENCE = 0.9
_INFERRED_CONFIDENCE = 0.4
_UNKNOWN_CONFIDENCE = 0.2


def _priority_ordered_layers(layers: dict[str, gpd.GeoDataFrame]) -> list[tuple[str, str]]:
    """Return [(layer_name, label)] sorted by label priority (highest first)."""
    present = [
        (layer, LAYER_TO_LABEL[layer])
        for layer in layers
        if layer in LAYER_TO_LABEL and layers[layer] is not None and not layers[layer].empty
    ]
    present.sort(key=lambda lp: LABEL_PRIORITY.get(lp[1], 99))
    return present


def classify_points(
    points_gdf: gpd.GeoDataFrame,
    layers: dict[str, gpd.GeoDataFrame],
    infer_rough: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify points against feature layers.

    Returns (labels, label_ids, sources, confidences) aligned to ``points_gdf``
    row order.
    """
    n = len(points_gdf)
    labels = np.array([None] * n, dtype=object)
    sources = np.array([None] * n, dtype=object)

    # Positional index per row so we can write back into the numpy arrays.
    pts = points_gdf.reset_index(drop=True).copy()
    pts["_pos"] = np.arange(n)

    for layer, label in _priority_ordered_layers(layers):
        remaining = pts[[labels[p] is None for p in pts["_pos"]]]
        if remaining.empty:
            break
        gdf = layers[layer][["geometry"]].copy()
        try:
            joined = gpd.sjoin(remaining, gdf, how="inner", predicate="intersects")
        except Exception as exc:  # noqa: BLE001
            log.warning("sjoin failed for layer %s (%s); skipping", layer, exc)
            continue
        hit_positions = joined["_pos"].unique()
        for p in hit_positions:
            labels[p] = label
            sources[p] = f"{SOURCE_OSM_PREFIX}{label}"

    # Fill unlabeled points.
    default_label = "rough_inferred" if infer_rough else "unknown"
    default_source = SOURCE_INFERRED if infer_rough else SOURCE_UNKNOWN
    confidences = np.empty(n, dtype="float64")
    label_ids = np.empty(n, dtype="int64")
    for i in range(n):
        if labels[i] is None:
            labels[i] = default_label
            sources[i] = default_source
            confidences[i] = _INFERRED_CONFIDENCE if infer_rough else _UNKNOWN_CONFIDENCE
        else:
            confidences[i] = _FEATURE_CONFIDENCE
        label_ids[i] = LABEL_IDS.get(labels[i], LABEL_IDS["unknown"])

    return labels, label_ids, sources, confidences
