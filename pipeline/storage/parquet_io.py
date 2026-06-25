"""Parquet writers for hole points and tabular roll-ups.

pyarrow is optional. If it's unavailable, the functions degrade gracefully:
``parquet_available()`` returns False and writers become no-ops that log a warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..logging_config import get_logger

log = get_logger("storage.parquet")

try:  # pragma: no cover - import guard
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _PA = True
except Exception:  # noqa: BLE001
    pa = None  # type: ignore
    pq = None  # type: ignore
    _PA = False


# Explicit schema for point parquet keeps types stable across courses.
_POINT_COLUMNS = (
    "hole_id", "point_id", "x_abs_m", "y_abs_m", "z_abs_m",
    "x_rel_m", "y_rel_m", "z_rel_m", "x_aligned_m", "y_aligned_m",
    "label", "label_id", "source", "confidence",
)


def parquet_available() -> bool:
    return _PA


def write_points_parquet(records: Iterable[dict], path: Path) -> Path | None:
    """Write point JSONL-style records to Parquet (columnar)."""
    if not _PA:
        log.warning("pyarrow unavailable; skipping parquet write for %s", path.name)
        return None
    cols: dict[str, list] = {c: [] for c in _POINT_COLUMNS}
    n = 0
    for rec in records:
        for c in _POINT_COLUMNS:
            cols[c].append(rec.get(c))
        n += 1
    if n == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(cols)
    pq.write_table(table, path)
    return path


def write_rows_parquet(rows: Sequence[dict], path: Path) -> Path | None:
    """Write a list of flat dict rows to Parquet (union of keys)."""
    if not _PA:
        log.warning("pyarrow unavailable; skipping parquet write for %s", path.name)
        return None
    if not rows:
        return None
    columns: list[str] = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    data = {c: [r.get(c) for r in rows] for c in columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(data), path)
    return path


def concat_parquet(sources: Sequence[Path], dest: Path) -> Path | None:
    """Concatenate per-hole point parquet files into one aggregate file."""
    if not _PA:
        log.warning("pyarrow unavailable; skipping parquet concat for %s", dest.name)
        return None
    existing = [p for p in sources if p.exists()]
    if not existing:
        return None
    tables = []
    for p in existing:
        try:
            tables.append(pq.read_table(p))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping unreadable parquet %s (%s)", p, exc)
    if not tables:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(tables, promote_options="default"), dest)
    return dest
