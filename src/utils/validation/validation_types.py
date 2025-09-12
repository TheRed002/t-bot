"""Common validation types and utilities."""

from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.types import ValidationLevel


class ValidationCategory(Enum):
    """Validation category types."""

    SCHEMA = "schema"
    BUSINESS = "business"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    REGULATORY = "regulatory"
    INTEGRITY = "integrity"
    RANGE = "range"
    FORMAT = "format"
    CONSISTENCY = "consistency"


class ValidationSeverity(Enum):
    """Validation severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityDimension(Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


def _get_utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class ValidationIssue:
    """Standardized validation issue record."""

    field: str
    value: Any
    expected: Any
    message: str
    level: ValidationLevel
    timestamp: datetime = dataclass_field(default_factory=_get_utc_now)
    source: str = "Validator"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)
    category: ValidationCategory = ValidationCategory.SCHEMA

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field": self.field,
            "value": self.value,
            "expected": self.expected,
            "message": self.message,
            "level": self.level.value
            if isinstance(self.level, ValidationLevel)
            else str(self.level),
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            "category": self.category.value,
        }


@dataclass
class QualityScore:
    """Data quality score breakdown."""

    overall: float = 0.0
    completeness: float = 0.0
    accuracy: float = 0.0
    consistency: float = 0.0
    timeliness: float = 0.0
    validity: float = 0.0
    uniqueness: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "completeness": self.completeness,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "timeliness": self.timeliness,
            "validity": self.validity,
            "uniqueness": self.uniqueness,
        }
