"""
State module types and data structures.

This module contains shared types used by both controllers and services
to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

from src.core.types import ExecutionResult, OrderRequest


class ValidationResult(Enum):
    """Validation result enumeration."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationCheck:
    """Individual validation check result."""

    check_name: str = ""
    result: ValidationResult = ValidationResult.PASSED
    score: Decimal = Decimal("100.0")
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "low"  # low, medium, high, critical


@dataclass
class PreTradeValidation:
    """Pre-trade validation results."""

    validation_id: str = field(default_factory=lambda: str(uuid4()))
    order_request: OrderRequest | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Overall results
    overall_result: ValidationResult = ValidationResult.PASSED
    overall_score: Decimal = Decimal("100.0")

    # Individual checks
    checks: list[ValidationCheck] = field(default_factory=list)

    # Risk assessment
    risk_level: str = "low"  # low, medium, high, critical
    risk_score: Decimal = Decimal("0.0")

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Processing time
    validation_time_ms: Decimal = Decimal("0.0")


@dataclass
class PostTradeAnalysis:
    """Post-trade analysis results."""

    analysis_id: str = field(default_factory=lambda: str(uuid4()))
    trade_id: str = ""
    execution_result: ExecutionResult | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Quality scores
    execution_quality_score: Decimal = Decimal("100.0")
    timing_quality_score: Decimal = Decimal("100.0")
    price_quality_score: Decimal = Decimal("100.0")
    overall_quality_score: Decimal = Decimal("100.0")

    # Performance metrics
    slippage_bps: Decimal = Decimal("0.0")
    execution_time_seconds: Decimal = Decimal("0.0")
    fill_rate: Decimal = Decimal("100.0")

    # Market impact analysis
    market_impact_bps: Decimal = Decimal("0.0")
    temporary_impact_bps: Decimal = Decimal("0.0")
    permanent_impact_bps: Decimal = Decimal("0.0")

    # Benchmark comparison
    benchmark_scores: dict[str, Decimal] = field(default_factory=dict)

    # Issues and recommendations
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Processing time
    analysis_time_ms: Decimal = Decimal("0.0")
