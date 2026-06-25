"""Storage/export subpackage: JSON/GeoJSON/JSONL, Parquet, DuckDB.

``parquet_io`` and ``duckdb_writer`` guard their optional 3rd-party imports
internally (``parquet_available()`` / ``duckdb_available()``), so importing them
here is always safe even when pyarrow/duckdb are absent.
"""

from __future__ import annotations

from . import json_io  # noqa: F401
from . import parquet_io  # noqa: F401
from . import duckdb_writer  # noqa: F401

__all__ = ["json_io", "parquet_io", "duckdb_writer"]
