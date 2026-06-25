"""Quality reporting: machine-readable issues persisted alongside artifacts.

Every course run produces a ``quality_report.json``. Dirty courses (e.g. != 18
holes detected) are skipped but always explained here rather than vanishing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .constants import SCHEMA_VERSION


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# Canonical issue codes (use these constants rather than free strings so the
# code set stays greppable and stable for downstream dashboards).
CODE_EXPECTED_HOLES_MISMATCH = "EXPECTED_HOLES_MISMATCH"
CODE_NO_COURSE_BOUNDARY = "NO_COURSE_BOUNDARY"
CODE_NO_HOLE_FEATURES = "NO_HOLE_FEATURES"
CODE_DUPLICATE_HOLE_REFS = "DUPLICATE_HOLE_REFS"
CODE_MISSING_LAYER = "MISSING_LAYER"
CODE_DEM_DOWNLOAD_FAILED = "DEM_DOWNLOAD_FAILED"
CODE_MISSING_DEM_VALUES = "MISSING_DEM_VALUES"
CODE_LOW_DEM_COVERAGE = "LOW_DEM_COVERAGE"
CODE_TEE_ELEVATION_NAN = "TEE_ELEVATION_NAN"
CODE_GREEN_ELEVATION_NAN = "GREEN_ELEVATION_NAN"
CODE_SLOPE_OUTLIER = "SLOPE_OUTLIER"
CODE_COARSE_DEM_RESOLUTION = "COARSE_DEM_RESOLUTION"
CODE_POINT_LIMIT_REACHED = "POINT_LIMIT_REACHED"
CODE_NO_TEE_FEATURE = "NO_TEE_FEATURE"
CODE_NO_GREEN_FEATURE = "NO_GREEN_FEATURE"
CODE_HOLE_PROCESSING_FAILED = "HOLE_PROCESSING_FAILED"
CODE_PLOTTING_FAILED = "PLOTTING_FAILED"


@dataclass
class QualityIssue:
    severity: Severity
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    hole_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
        }
        if self.hole_id is not None:
            d["hole_id"] = self.hole_id
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class QualityReport:
    course_slug: str
    status: str = "processed"  # "processed" | "skipped" | "failed"
    issues: list[QualityIssue] = field(default_factory=list)

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        hole_id: str | None = None,
    ) -> QualityIssue:
        issue = QualityIssue(severity, code, message, details or {}, hole_id)
        self.issues.append(issue)
        return issue

    def error(self, code: str, message: str, **kw) -> QualityIssue:
        return self.add(Severity.ERROR, code, message, **kw)

    def warning(self, code: str, message: str, **kw) -> QualityIssue:
        return self.add(Severity.WARNING, code, message, **kw)

    def info(self, code: str, message: str, **kw) -> QualityIssue:
        return self.add(Severity.INFO, code, message, **kw)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def flags(self) -> list[str]:
        """Distinct issue codes (used as compact quality_flags lists)."""
        seen: list[str] = []
        for i in self.issues:
            if i.code not in seen:
                seen.append(i.code)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "course_slug": self.course_slug,
            "status": self.status,
            "issues": [i.to_dict() for i in self.issues],
        }
