#!/usr/bin/env python
"""Build a Hugging Face upload folder from the generated pipeline artifacts.

Thin CLI wrapper around :func:`pipeline.modeling.hf_export.build_hf_artifact`.
Equivalent to ``python -m pipeline.modeling hf-export``; provided for the
``python scripts/build_hf_artifact.py`` invocation style.

Examples
--------
    python scripts/build_hf_artifact.py --tier lite --output hf_artifact_lite
    python scripts/build_hf_artifact.py --tier full --output hf_artifact_full \
        --include-point-parquet

Nothing is uploaded and git is untouched; this only writes a local folder and
prints a size summary for you to review before uploading manually.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a loose script (python scripts/build_hf_artifact.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.logging_config import configure_logging  # noqa: E402
from pipeline.modeling.hf_export import build_hf_artifact  # noqa: E402
from pipeline.paths import COURSES_ROOT  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", choices=["lite", "full"], default="lite",
                   help="Artifact tier (default: lite).")
    p.add_argument("--output", default=None,
                   help="Output folder (default: hf_artifact_<tier>).")
    p.add_argument("--courses-root", type=Path, default=COURSES_ROOT,
                   help=f"Root of course outputs (default: {COURSES_ROOT}).")
    p.add_argument("--include-point-parquet", action="store_true",
                   help="Full tier: also copy per-hole point Parquet files.")
    p.add_argument("--include-all-points", action="store_true",
                   help="Full tier: also copy the ~1 GB all_hole_points.parquet.")
    p.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR.")
    args = p.parse_args(argv)

    configure_logging(args.log_level)
    summary = build_hf_artifact(
        args.tier, args.output, courses_root=args.courses_root,
        include_point_parquet=args.include_point_parquet,
        include_all_points=args.include_all_points,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
