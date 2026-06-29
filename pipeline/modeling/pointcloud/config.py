"""Config model + YAML loader + validation for v2.5 point-cloud similarity.

A single ``PointCloudSimilarityConfig`` drives candidate filtering, surface
weighting, point budgets, distance scaling, and penalties. Everything that
shapes a score lives here (and in the YAML), never hardcoded in the scorer.

The config is validated at load time (:func:`load_config`) and carries a
deterministic ``config_hash`` derived from its normalized contents, so a stored
``SimilarityResult`` can be traced back to the exact weights that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import yaml

from .schemas import KNOWN_SURFACES

PathLike = Union[str, Path]

#: Tolerance when checking that surface weights sum to 1.0.
WEIGHT_SUM_TOLERANCE: float = 1e-6

#: Par buckets that must each have a yardage window.
_REQUIRED_PAR_BUCKETS: tuple[str, ...] = ("par_3", "par_4", "par_5")


class ConfigError(ValueError):
    """Raised when a config fails validation (bad weights, surfaces, windows)."""


@dataclass(frozen=True)
class YardageWindow:
    """Per-par yardage tolerance: the more permissive of abs / pct wins."""

    absolute_yards: float
    percentage: float

    def allowed_diff(self, target_yards: float) -> float:
        """Max permitted ``|candidate.yards - target.yards|`` for ``target_yards``."""
        return max(float(self.absolute_yards), float(target_yards) * float(self.percentage))


@dataclass(frozen=True)
class CandidateFilterConfig:
    """Cheap pre-scoring gates: same-par requirement + per-par yardage windows."""

    require_same_par: bool
    yardage_windows: dict[str, YardageWindow]

    def window_for_par(self, par: int) -> YardageWindow | None:
        """Yardage window for a par value (3/4/5), or ``None`` if unbucketed."""
        return self.yardage_windows.get(f"par_{int(par)}")


@dataclass(frozen=True)
class DistanceScaling:
    """Per-axis multipliers applied before Chamfer distance is computed."""

    x_weight: float = 1.0
    y_weight: float = 1.0
    z_weight: float = 2.0


@dataclass(frozen=True)
class PenaltyConfig:
    """Scalar penalty knobs applied on top of the weighted surface score."""

    par_mismatch: float = 999.0
    yardage_weight: float = 0.15
    tee_to_green_elevation_weight: float = 0.10


@dataclass(frozen=True)
class PointCloudSimilarityConfig:
    """Fully-resolved, validated v2.5 similarity configuration."""

    model_name: str
    model_version: str
    config_name: str
    candidate_filter: CandidateFilterConfig
    required_surfaces: tuple[str, ...]
    surface_weights: dict[str, float]
    surface_missing_penalties: dict[str, float]
    point_budgets: dict[str, int]
    distance_scaling: DistanceScaling
    penalties: PenaltyConfig
    #: Deterministic hash of the normalized config contents (set at load time).
    config_hash: str = ""
    #: Source path, for diagnostics (not part of the hash).
    source_path: str = field(default="", compare=False)

    def weighted_surfaces(self) -> tuple[str, ...]:
        """Surfaces that carry weight, in canonical order."""
        return tuple(s for s in KNOWN_SURFACES if s in self.surface_weights)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _coerce_yardage_windows(raw: dict) -> dict[str, YardageWindow]:
    windows: dict[str, YardageWindow] = {}
    for bucket, spec in (raw or {}).items():
        if not isinstance(spec, dict):
            raise ConfigError(f"yardage_windows.{bucket} must be a mapping, got {spec!r}")
        try:
            windows[bucket] = YardageWindow(
                absolute_yards=float(spec["absolute_yards"]),
                percentage=float(spec["percentage"]),
            )
        except KeyError as exc:
            raise ConfigError(
                f"yardage_windows.{bucket} missing key {exc.args[0]!r} "
                "(need 'absolute_yards' and 'percentage')"
            ) from exc
    return windows


def _config_hash(payload: dict) -> str:
    """SHA-256 over canonical JSON of the *input* payload (excludes hash/path)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_from_dict(payload: dict, *, source_path: str = "") -> PointCloudSimilarityConfig:
    """Build and validate a config from an already-parsed mapping.

    The ``config_hash`` is computed from ``payload`` before any defaults are
    folded in, so two byte-identical YAML files always hash the same.
    """
    cf_raw = payload.get("candidate_filter") or {}
    candidate_filter = CandidateFilterConfig(
        require_same_par=bool(cf_raw.get("require_same_par", True)),
        yardage_windows=_coerce_yardage_windows(cf_raw.get("yardage_windows", {})),
    )

    ds_raw = payload.get("distance_scaling") or {}
    distance_scaling = DistanceScaling(
        x_weight=float(ds_raw.get("x_weight", 1.0)),
        y_weight=float(ds_raw.get("y_weight", 1.0)),
        z_weight=float(ds_raw.get("z_weight", 2.0)),
    )

    pen_raw = payload.get("penalties") or {}
    penalties = PenaltyConfig(
        par_mismatch=float(pen_raw.get("par_mismatch", 999.0)),
        yardage_weight=float(pen_raw.get("yardage_weight", 0.15)),
        tee_to_green_elevation_weight=float(
            pen_raw.get("tee_to_green_elevation_weight", 0.10)
        ),
    )

    config = PointCloudSimilarityConfig(
        model_name=str(payload.get("model_name", "")),
        model_version=str(payload.get("model_version", "")),
        config_name=str(payload.get("config_name", "")),
        candidate_filter=candidate_filter,
        required_surfaces=tuple(payload.get("required_surfaces", []) or []),
        surface_weights={k: float(v) for k, v in (payload.get("surface_weights") or {}).items()},
        surface_missing_penalties={
            k: float(v) for k, v in (payload.get("surface_missing_penalties") or {}).items()
        },
        point_budgets={k: int(v) for k, v in (payload.get("point_budgets") or {}).items()},
        distance_scaling=distance_scaling,
        penalties=penalties,
        config_hash=_config_hash(payload),
        source_path=source_path,
    )
    validate_config(config)
    return config


def load_config(path: PathLike) -> PointCloudSimilarityConfig:
    """Load, validate, and hash a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"{path} did not parse to a mapping (got {type(payload).__name__})")
    return config_from_dict(payload, source_path=str(path))


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_config(config: PointCloudSimilarityConfig) -> None:
    """Validate a config in place, raising :class:`ConfigError` on any problem.

    Rules:
      * ``model_version`` and ``config_name`` are required (non-empty).
      * every weighted surface must be a known surface.
      * surface weights must sum to 1.0 within :data:`WEIGHT_SUM_TOLERANCE`.
      * every required surface must be a known surface.
      * a yardage window must exist for ``par_3``, ``par_4`` and ``par_5``.
    """
    if not config.model_version:
        raise ConfigError("model_version is required")
    if not config.config_name:
        raise ConfigError("config_name is required")

    if not config.surface_weights:
        raise ConfigError("surface_weights must define at least one surface")

    unknown_weighted = [s for s in config.surface_weights if s not in KNOWN_SURFACES]
    if unknown_weighted:
        raise ConfigError(
            f"surface_weights contains unknown surface(s) {unknown_weighted}; "
            f"known surfaces: {list(KNOWN_SURFACES)}"
        )

    weight_sum = sum(config.surface_weights.values())
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ConfigError(
            f"surface_weights must sum to 1.0 (got {weight_sum:.6f}); "
            f"weights: {config.surface_weights}"
        )

    unknown_required = [s for s in config.required_surfaces if s not in KNOWN_SURFACES]
    if unknown_required:
        raise ConfigError(
            f"required_surfaces contains unknown surface(s) {unknown_required}; "
            f"known surfaces: {list(KNOWN_SURFACES)}"
        )

    unknown_missing = [s for s in config.surface_missing_penalties if s not in KNOWN_SURFACES]
    if unknown_missing:
        raise ConfigError(
            f"surface_missing_penalties contains unknown surface(s) {unknown_missing}; "
            f"known surfaces: {list(KNOWN_SURFACES)}"
        )

    missing_buckets = [b for b in _REQUIRED_PAR_BUCKETS
                       if b not in config.candidate_filter.yardage_windows]
    if missing_buckets:
        raise ConfigError(
            f"candidate_filter.yardage_windows missing required bucket(s) {missing_buckets}; "
            f"need windows for {list(_REQUIRED_PAR_BUCKETS)}"
        )


def config_to_dict(config: PointCloudSimilarityConfig) -> dict:
    """Plain-dict view of a config (handy for logging / serialization)."""
    return asdict(config)


__all__ = [
    "ConfigError",
    "YardageWindow",
    "CandidateFilterConfig",
    "DistanceScaling",
    "PenaltyConfig",
    "PointCloudSimilarityConfig",
    "WEIGHT_SUM_TOLERANCE",
    "load_config",
    "config_from_dict",
    "validate_config",
    "config_to_dict",
]
