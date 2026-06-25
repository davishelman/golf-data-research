"""Centralized logging configuration.

Provides a single ``get_logger`` so every stage logs consistently. Logging is
idempotent (safe to call configure multiple times) and never writes to stdout in
a way that interferes with machine-readable artifacts — all logs go to stderr.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: int | str | None = None) -> None:
    """Configure the root logger once. Honors the ``GOLF_LOG_LEVEL`` env var."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("GOLF_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger("golf")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``golf`` root."""
    configure_logging()
    if not name.startswith("golf"):
        name = f"golf.{name}"
    return logging.getLogger(name)
