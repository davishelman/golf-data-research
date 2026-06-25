"""Project-wide constants: labels, priorities, OSM tag rules, schema versions.

This module is dependency-free (stdlib only) so every other module can import
it without risk of cycles.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Canonical feature labels
# ---------------------------------------------------------------------------
# label name -> integer id. The integer ids are stable and used in compact
# point arrays, Parquet, and DuckDB. Do not renumber existing ids; only append.

LABEL_IDS: dict[str, int] = {
    "unknown": 0,
    "tee": 1,
    "green": 2,
    "fairway": 3,
    "rough_osm": 4,
    "bunker": 5,
    "water": 6,
    "trees": 7,
    "cartpath": 8,
    "sand": 9,
    "rough_inferred": 10,
}

# Reverse map (id -> label) for label_map.json emission and decoding.
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_IDS.items()}

# label_map.json is keyed by string ids per the spec's JSON examples.
LABEL_MAP_JSON: dict[str, str] = {str(i): name for i, name in LABEL_NAMES.items()}

# ---------------------------------------------------------------------------
# Point-labeling priority (highest wins when a point intersects many polygons)
# ---------------------------------------------------------------------------
# Lower number = higher priority. Bunkers/water override fairway/rough; green/tee
# are special surfaces; explicitly-mapped rough beats inferred rough.

LABEL_PRIORITY: dict[str, int] = {
    "green": 1,
    "tee": 2,
    "bunker": 3,
    "water": 4,
    "fairway": 5,
    "cartpath": 6,
    "sand": 7,
    "trees": 8,
    "rough_osm": 9,
    "rough_inferred": 10,
    "unknown": 11,
}

# ---------------------------------------------------------------------------
# Canonical vector layers
# ---------------------------------------------------------------------------
# These are the per-course layer names persisted under source/ and per-hole
# under holes/hole_XX/vectors/.

LAYER_HOLE_CENTERLINES = "hole_centerlines"
LAYER_COURSE_BOUNDARY = "course_boundary"

# Feature layers that are "owned" by a specific hole. Neighboring instances of
# these must NOT leak into the wrong hole — they are assigned by ref, else by
# nearest centerline.
HOLE_OWNED_LAYERS: tuple[str, ...] = (
    "tees", "greens", "fairways", "bunkers", "sand", "rough_osm",
)

# Feature layers that legitimately span multiple holes (a pond bordering two
# fairways, a wood behind several greens, a cartpath threading the course).
# Assigned by geometric overlap, not exclusively owned.
SHARED_LAYERS: tuple[str, ...] = (
    "water", "trees", "cartpaths",
)

# All feature layers (excludes centerlines + boundary which are structural).
FEATURE_LAYERS: tuple[str, ...] = HOLE_OWNED_LAYERS + SHARED_LAYERS

# Map a canonical *layer* name to the canonical *label* a point gets when it
# falls inside that layer's geometry.
LAYER_TO_LABEL: dict[str, str] = {
    "tees": "tee",
    "greens": "green",
    "fairways": "fairway",
    "bunkers": "bunker",
    "sand": "sand",
    "rough_osm": "rough_osm",
    "water": "water",
    "trees": "trees",
    "cartpaths": "cartpath",
}

# ---------------------------------------------------------------------------
# OSM tag rules
# ---------------------------------------------------------------------------
# The Overpass query. Broad enough to capture trees/cartpaths; noisy tags
# (highway/landuse) are filtered to the course boundary downstream.

OSM_TAGS: dict = {
    "leisure": "golf_course",
    "golf": [
        "hole", "fairway", "green", "tee", "bunker",
        "water_hazard", "lateral_water_hazard", "rough", "cartpath",
    ],
    "natural": ["water", "sand", "wood", "tree", "tree_row"],
    "water": True,
    "landuse": ["grass", "forest", "recreation_ground"],
    "highway": ["path", "service", "track"],
}

# Canonical layer -> list of OSM tag predicates (any match assigns the feature
# to that layer). Each predicate is a dict of {column: value}; value True means
# "column present/non-null". Earlier layers in FEATURE_LAYERS win on conflict.
FEATURE_TAG_RULES: dict[str, list[dict]] = {
    "tees": [{"golf": "tee"}],
    "greens": [{"golf": "green"}],
    "fairways": [{"golf": "fairway"}],
    "rough_osm": [{"golf": "rough"}],
    "bunkers": [{"golf": "bunker"}],
    "water": [
        {"golf": "water_hazard"},
        {"golf": "lateral_water_hazard"},
        {"natural": "water"},
        {"water": True},
    ],
    "trees": [
        {"natural": "wood"},
        {"landuse": "forest"},
        {"natural": "tree_row"},
        {"natural": "tree"},
    ],
    "cartpaths": [
        {"golf": "cartpath"},
        {"highway": "path"},
        {"highway": "service"},
        {"highway": "track"},
    ],
    "sand": [{"natural": "sand"}],
}

# Order layers are tested in when a single feature matches multiple rules.
# (A bunker tagged natural=sand should resolve to bunker, etc.)
LAYER_RESOLUTION_ORDER: tuple[str, ...] = (
    "greens", "tees", "bunkers", "water", "fairways",
    "cartpaths", "sand", "trees", "rough_osm",
)

# ---------------------------------------------------------------------------
# Feature provenance strings (for the point "source" field)
# ---------------------------------------------------------------------------
SOURCE_OSM_PREFIX = "osm:golf="
SOURCE_INFERRED = "inferred:background"
SOURCE_UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
METERS_TO_YARDS: float = 1.09361
