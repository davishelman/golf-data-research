"""JSON / GeoJSON / JSONL writers and readers.

All writers create parent directories and write atomically enough for a batch
pipeline. JSONL is streamed so per-hole point clouds never sit fully in memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd

from ..logging_config import get_logger

log = get_logger("storage.json")

_EMPTY_FC = '{"type": "FeatureCollection", "features": []}'


def save_json(obj: Any, path: Path, *, indent: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    return path


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_geojson(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    """Write a GeoDataFrame to GeoJSON; empty/None -> empty FeatureCollection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf is None or len(gdf) == 0:
        path.write_text(_EMPTY_FC, encoding="utf-8")
        return path
    # Drop helper/object columns that GeoJSON can't serialize cleanly.
    safe = gdf.copy()
    for col in list(safe.columns):
        if col == "geometry":
            continue
        if safe[col].apply(lambda v: isinstance(v, (list, dict, set))).any():
            safe[col] = safe[col].apply(
                lambda v: ";".join(map(str, v)) if isinstance(v, (list, set)) else v
            )
    try:
        safe.to_file(path, driver="GeoJSON")
    except Exception as exc:  # noqa: BLE001
        log.warning("GeoJSON write failed for %s (%s); writing empty FC", path.name, exc)
        path.write_text(_EMPTY_FC, encoding="utf-8")
    return path


class JsonlWriter:
    """Streaming JSON-lines writer usable as a context manager.

    with JsonlWriter(path) as w:
        for rec in records:
            w.write(rec)
    """

    def __init__(self, path: Path):
        self.path = path
        self._fh = None
        self.count = 0

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        return self

    def write(self, record: dict) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(record, ensure_ascii=False))
        self._fh.write("\n")
        self.count += 1

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
