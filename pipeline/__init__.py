"""End-to-end golf course terrain + 3D point-cloud pipeline.

Staged architecture (see ``orchestrator.run_course``):

  config -> osm (fetch/boundary/holes/layers/assignment)
         -> raster (dem/clip/slope/sampling)
         -> features (anchors/labels/transforms/point_cloud)
         -> terrain (stats)
         -> storage (json/parquet/duckdb) + plotting

Primary deliverable: per-hole tee-relative, tee->green-aligned, labeled 3D point
clouds with quality metadata, alongside terrain statistics and manifests.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["orchestrator", "cli", "config", "schemas", "constants"]
