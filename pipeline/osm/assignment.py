"""Stage 5 — assign features to a specific hole and clip to its corridor.

Assignment uses the strongest available evidence:
  1. OSM ``ref`` (exact hole ownership) for hole-owned layers,
  2. else nearest hole centerline,
  3. geometric overlap for shared layers (water, trees, cartpaths).

Each kept feature records ``assigned_hole`` / ``assignment_method`` /
``assignment_confidence``. Owned features that belong to another hole are dropped
so neighbors don't leak across overlapping buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import geopandas as gpd

from ..constants import HOLE_OWNED_LAYERS
from ..geometry import nearest_hole_number
from ..logging_config import get_logger
from .holes import parse_ref_tokens

log = get_logger("osm.assignment")

_CONF = {"ref": 1.0, "nearest_centerline": 0.6, "shared_overlap": 0.5}


@dataclass
class AssignmentSummary:
    counts: dict[str, int] = field(default_factory=dict)   # layer -> kept count
    methods: dict[str, int] = field(default_factory=dict)  # method -> count

    def record(self, layer: str, method: str, n: int) -> None:
        self.counts[layer] = self.counts.get(layer, 0) + n
        if n:
            self.methods[method] = self.methods.get(method, 0) + n

    def to_dict(self) -> dict[str, Any]:
        return {"counts": self.counts, "methods": self.methods}


def _decide(
    row, hole_number: int, hole_lines: dict[int, Any], owned: bool
) -> Optional[str]:
    """Return assignment_method if the feature belongs to this hole, else None."""
    if not owned:
        return "shared_overlap"
    ref_val = row["ref"] if "ref" in row.index else None
    tokens = parse_ref_tokens(ref_val)
    if tokens is not None:
        return "ref" if hole_number in tokens else None
    nearest = nearest_hole_number(row.geometry, hole_lines)
    return "nearest_centerline" if nearest == hole_number else None


def assign_layer_to_hole(
    layer_name: str,
    layer_gdf: Optional[gpd.GeoDataFrame],
    hole_number: int,
    buffer_geom,
    hole_lines: dict[int, Any],
    summary: Optional[AssignmentSummary] = None,
) -> gpd.GeoDataFrame:
    """Assign + clip one canonical layer to one hole."""
    owned = layer_name in HOLE_OWNED_LAYERS
    crs = layer_gdf.crs if layer_gdf is not None else None
    if layer_gdf is None or layer_gdf.empty:
        return _empty(crs)

    # Only consider features that touch this hole's corridor at all.
    candidates = layer_gdf[layer_gdf.intersects(buffer_geom)]
    if candidates.empty:
        return _empty(crs)

    keep_methods: list[tuple[int, str]] = []
    for idx, row in candidates.iterrows():
        method = _decide(row, hole_number, hole_lines, owned)
        if method is not None:
            keep_methods.append((idx, method))

    if not keep_methods:
        return _empty(crs)

    idxs = [i for i, _ in keep_methods]
    kept = candidates.loc[idxs].copy()
    kept["assigned_hole"] = hole_number
    kept["assignment_method"] = [m for _, m in keep_methods]
    kept["assignment_confidence"] = [_CONF.get(m, 0.5) for _, m in keep_methods]

    # Clip geometry to the hole buffer.
    kept["geometry"] = kept.geometry.intersection(buffer_geom)
    kept = kept[~kept.geometry.is_empty & kept.geometry.notna()].copy()

    if summary is not None:
        for _, m in keep_methods:
            summary.record(layer_name, m, 1)

    return kept.reset_index(drop=True)


def _empty(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"assigned_hole": [], "assignment_method": [], "assignment_confidence": []},
        geometry=[], crs=crs,
    )
