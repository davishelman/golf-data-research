"""Path conventions for course/hole/index artifacts.

Layout per course (rooted at COURSES_ROOT / <slug>):
  course_manifest.json
  quality_report.json
  boundary_selection.json
  source/<layer>.geojson + preview.png
  dem/course_dem_raw.tif
  holes/hole_XX/
    vectors/<layer>.geojson
    dem/dem_clipped.tif, dem_clipped_projected.tif
    stats/terrain_summary.json
    features/hole_points.jsonl, hole_points_compact.json, label_map.json, hole_points.parquet
    plots/<plot>.png + 3d_terrain.html + overview.png
  course_summary.json   (legacy roll-up; kept for backward compat)

Aggregate index (rooted at COURSES_ROOT / _index):
  all_holes.csv, all_holes.parquet, all_hole_points.parquet,
  all_courses_manifest.json, golf.duckdb
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT: Path = Path(__file__).resolve().parent.parent
COURSES_ROOT: Path = REPO_ROOT / "courses"
CONFIG_ROOT: Path = REPO_ROOT / "config"
DEFAULT_COURSES_JSON: Path = CONFIG_ROOT / "courses.json"


# ---------------------------------------------------------------------------
# Aggregate index paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexPaths:
    root: Path

    @classmethod
    def for_root(cls, courses_root: Path = COURSES_ROOT) -> "IndexPaths":
        return cls(root=courses_root / "_index")

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def all_holes_csv(self) -> Path:
        return self.root / "all_holes.csv"

    @property
    def all_holes_parquet(self) -> Path:
        return self.root / "all_holes.parquet"

    @property
    def all_hole_points_parquet(self) -> Path:
        return self.root / "all_hole_points.parquet"

    @property
    def all_courses_manifest(self) -> Path:
        return self.root / "all_courses_manifest.json"

    @property
    def duckdb(self) -> Path:
        return self.root / "golf.duckdb"

    # --- modeling / similarity outputs ---
    @property
    def hole_features_parquet(self) -> Path:
        return self.root / "hole_features.parquet"

    @property
    def hole_features_csv(self) -> Path:
        return self.root / "hole_features.csv"

    @property
    def hole_clusters_parquet(self) -> Path:
        return self.root / "hole_clusters.parquet"

    @property
    def hole_clusters_csv(self) -> Path:
        return self.root / "hole_clusters.csv"

    @property
    def hole_similarity_examples_csv(self) -> Path:
        return self.root / "hole_similarity_examples.csv"


# Legacy constant kept for backward compatibility with older callers.
ALL_HOLES_CSV: Path = IndexPaths.for_root().all_holes_csv


@dataclass(frozen=True)
class CoursePaths:
    slug: str
    root: Path

    @classmethod
    def for_slug(cls, slug: str, courses_root: Path = COURSES_ROOT) -> "CoursePaths":
        return cls(slug=slug, root=courses_root / slug)

    # Directories ----------------------------------------------------------
    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def dem(self) -> Path:
        return self.root / "dem"

    @property
    def holes(self) -> Path:
        return self.root / "holes"

    def ensure(self) -> None:
        for p in (self.source, self.dem, self.holes):
            p.mkdir(parents=True, exist_ok=True)

    # Files ----------------------------------------------------------------
    def source_layer(self, name: str) -> Path:
        return self.source / f"{name}.geojson"

    def source_preview(self) -> Path:
        return self.source / "preview.png"

    def course_dem(self) -> Path:
        return self.dem / "course_dem_raw.tif"

    @property
    def manifest(self) -> Path:
        return self.root / "course_manifest.json"

    @property
    def quality_report(self) -> Path:
        return self.root / "quality_report.json"

    @property
    def boundary_selection(self) -> Path:
        return self.root / "boundary_selection.json"

    @property
    def course_summary(self) -> Path:  # legacy roll-up
        return self.root / "course_summary.json"

    @property
    def course_overview(self) -> Path:
        return self.root / "course_overview.png"


@dataclass(frozen=True)
class HolePaths:
    hole_number: int
    root: Path

    @classmethod
    def for_hole(cls, course_paths: CoursePaths, hole_number: int) -> "HolePaths":
        return cls(
            hole_number=hole_number,
            root=course_paths.holes / f"hole_{hole_number:02d}",
        )

    @property
    def vectors(self) -> Path:
        return self.root / "vectors"

    @property
    def dem(self) -> Path:
        return self.root / "dem"

    @property
    def stats(self) -> Path:
        return self.root / "stats"

    @property
    def features(self) -> Path:
        return self.root / "features"

    @property
    def plots(self) -> Path:
        return self.root / "plots"

    def ensure(self) -> None:
        for p in (self.vectors, self.dem, self.stats, self.features, self.plots):
            p.mkdir(parents=True, exist_ok=True)

    # Files ----------------------------------------------------------------
    @property
    def terrain_summary(self) -> Path:
        return self.stats / "terrain_summary.json"

    @property
    def anchors(self) -> Path:
        return self.stats / "anchors.json"

    @property
    def hole_points_jsonl(self) -> Path:
        return self.features / "hole_points.jsonl"

    @property
    def hole_points_compact(self) -> Path:
        return self.features / "hole_points_compact.json"

    @property
    def hole_points_parquet(self) -> Path:
        return self.features / "hole_points.parquet"

    @property
    def label_map(self) -> Path:
        return self.features / "label_map.json"

    @property
    def assignment_report(self) -> Path:
        return self.vectors / "assignment.json"

    @property
    def clipped_dem(self) -> Path:
        return self.dem / "dem_clipped.tif"

    @property
    def projected_dem(self) -> Path:
        return self.dem / "dem_clipped_projected.tif"

    def relpath_from_course(self, course_paths: CoursePaths, p: Path) -> str:
        """POSIX relative path of ``p`` from the course root (for manifests)."""
        return p.relative_to(course_paths.root).as_posix()
