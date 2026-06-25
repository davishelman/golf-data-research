"""Terrain analysis subpackage: statistics over a per-hole metric DEM."""

from __future__ import annotations

from .stats import build_terrain_summary  # noqa: F401

__all__ = ["build_terrain_summary"]
