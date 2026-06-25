"""Optional DuckDB writer: build a local analytics database from artifacts.

duckdb is optional. ``duckdb_available()`` reports availability; ``build_database``
creates tables/views over the aggregate Parquet/CSV files produced by exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..logging_config import get_logger

log = get_logger("storage.duckdb")

try:  # pragma: no cover - import guard
    import duckdb  # type: ignore
    _DUCK = True
except Exception:  # noqa: BLE001
    duckdb = None  # type: ignore
    _DUCK = False


def duckdb_available() -> bool:
    return _DUCK


def build_database(
    db_path: Path,
    holes_parquet: Optional[Path] = None,
    points_parquet: Optional[Path] = None,
) -> Optional[Path]:
    """(Re)build a DuckDB database with views over the aggregate Parquet files."""
    if not _DUCK:
        log.warning("duckdb unavailable; skipping database build")
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        if holes_parquet and holes_parquet.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW holes AS "
                f"SELECT * FROM read_parquet({_sql_literal(holes_parquet)})"
            )
            con.execute(
                """
                CREATE OR REPLACE VIEW hole_terrain_stats AS
                SELECT hole_id, tee_elevation_m, green_elevation_m,
                       net_elevation_change_m, min_elevation_m, max_elevation_m,
                       mean_elevation_m, avg_slope_deg, max_slope_deg,
                       avg_slope_percent, max_slope_percent
                FROM holes
                """
            )
        if points_parquet and points_parquet.exists():
            con.execute(
                f"CREATE OR REPLACE VIEW hole_points AS "
                f"SELECT * FROM read_parquet({_sql_literal(points_parquet)})"
            )
        log.info("duckdb built at %s", db_path)
    finally:
        con.close()
    return db_path


def _sql_literal(path: Path) -> str:
    """A single-quoted SQL string literal for a filesystem path.

    DuckDB does not allow bound parameters inside ``CREATE VIEW``, so the path is
    embedded directly (forward slashes, single quotes escaped).
    """
    s = str(path).replace("\\", "/").replace("'", "''")
    return f"'{s}'"
