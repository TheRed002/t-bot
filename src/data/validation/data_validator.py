"""
DataValidator - Enterprise Data Quality and Validation System

This module implements comprehensive data validation and quality monitoring
for financial data, featuring real-time validation, quality scoring,
anomaly detection, and regulatory compliance checks.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002: Database models, queries, and connections
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.base import BaseComponent
from src.core.config import Config
from src.core.types import MarketData
from src.utils.decorators import time_execution


class ValidationSeverity(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Validation category types."""

    SCHEMA = "schema"
    BUSINESS = "business"
    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    REGULATORY = "regulatory"
    INTEGRITY = "integrity"


class QualityDimension(Enum):
    """Data quality dimensions."""

    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class ValidationIssue:
    """Validation issue details."""

    category: ValidationCategory
    severity: ValidationSeverity
    dimension: QualityDimension
    message: str
    field: str | None = None
    value: Any | None = None
    expected: Any | None = None
    rule_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


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


class ValidationRule(BaseModel):
    """Validation rule definition."""

    name: str = Field(..., min_length=1, max_length=100)
    category: ValidationCategory
    severity: ValidationSeverity
    dimension: QualityDimension
    description: str = Field(..., min_length=1)
    enabled: bool = True
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=True)


class MarketDataValidationResult(BaseModel):
    """Market data validation result."""

    symbol: str
    is_valid: bool
    quality_score: QualityScore
    issues: list[ValidationIssue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    validation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DataValidator(BaseComponent):
    """
    Comprehensive data validator for financial data quality assurance.

    Features:
    - Schema validation for data structure integrity
    - Business rule validation for domain-specific logic
    - Statistical validation for outlier detection
    - Temporal validation for time-series consistency
    - Regulatory compliance checks
    - Real-time quality scoring
    """

    def __init__(self, config: Config):
        """Initialize the data validator."""
        super().__init__()
        self.config = config

        # Validation rules registry
        self._rules: dict[str, ValidationRule] = {}

        # Configuration
        self._setup_configuration()

        # Initialize validation rules
        self._initialize_rules()

        # Statistics for adaptive thresholds
        self._price_stats: dict[str, dict[str, float]] = {}
        self._volume_stats: dict[str, dict[str, float]] = {}

        self._initialized = False

    def _setup_configuration(self) -> None:
        """Setup validator configuration."""
        validator_config = getattr(self.config, "data_validator", {})

        self.validation_config = {
            "enable_schema_validation": validator_config.get("enable_schema", True),
            "enable_business_validation": validator_config.get("enable_business", True),
            "enable_statistical_validation": validator_config.get("enable_statistical", True),
            "enable_temporal_validation": validator_config.get("enable_temporal", True),
            "enable_regulatory_validation": validator_config.get("enable_regulatory", True),
            # Thresholds
            "price_change_threshold": validator_config.get("price_change_threshold", 0.20),  # 20%
            "volume_spike_threshold": validator_config.get(
                "volume_spike_threshold", 5.0
            ),  # 5x normal
            "staleness_threshold_minutes": validator_config.get("staleness_threshold", 60),
            "decimal_precision": validator_config.get("decimal_precision", 8),
            # Quality scoring weights
            "quality_weights": validator_config.get(
                "quality_weights",
                {
                    "completeness": 0.25,
                    "accuracy": 0.25,
                    "consistency": 0.20,
                    "timeliness": 0.15,
                    "validity": 0.10,
                    "uniqueness": 0.05,
                },
            ),
        }

    def _initialize_rules(self) -> None:
        """Initialize validation rules."""
        # Schema validation rules
        self._add_rule(
            ValidationRule(
                name="required_fields",
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.COMPLETENESS,
                description="Check for required fields in market data",
                parameters={"required_fields": ["symbol", "price", "timestamp"]},
            )
        )

        self._add_rule(
            ValidationRule(
                name="field_types",
                category=ValidationCategory.SCHEMA,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.VALIDITY,
                description="Validate field data types",
            )
        )

        # Business validation rules
        self._add_rule(
            ValidationRule(
                name="positive_price",
                category=ValidationCategory.BUSINESS,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.VALIDITY,
                description="Price must be positive",
            )
        )

        self._add_rule(
            ValidationRule(
                name="positive_volume",
                category=ValidationCategory.BUSINESS,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.VALIDITY,
                description="Volume must be non-negative",
            )
        )

        self._add_rule(
            ValidationRule(
                name="bid_ask_spread",
                category=ValidationCategory.BUSINESS,
                severity=ValidationSeverity.WARNING,
                dimension=QualityDimension.CONSISTENCY,
                description="Bid price should be less than ask price",
            )
        )

        # Statistical validation rules
        self._add_rule(
            ValidationRule(
                name="price_outlier",
                category=ValidationCategory.STATISTICAL,
                severity=ValidationSeverity.WARNING,
                dimension=QualityDimension.ACCURACY,
                description="Detect price outliers using statistical methods",
                parameters={"z_score_threshold": 3.0},
            )
        )

        self._add_rule(
            ValidationRule(
                name="volume_spike",
                category=ValidationCategory.STATISTICAL,
                severity=ValidationSeverity.WARNING,
                dimension=QualityDimension.ACCURACY,
                description="Detect unusual volume spikes",
                parameters={"spike_multiplier": 5.0},
            )
        )

        # Temporal validation rules
        self._add_rule(
            ValidationRule(
                name="timestamp_freshness",
                category=ValidationCategory.TEMPORAL,
                severity=ValidationSeverity.WARNING,
                dimension=QualityDimension.TIMELINESS,
                description="Check data freshness",
                parameters={"max_age_minutes": 60},
            )
        )

        self._add_rule(
            ValidationRule(
                name="future_timestamp",
                category=ValidationCategory.TEMPORAL,
                severity=ValidationSeverity.ERROR,
                dimension=QualityDimension.VALIDITY,
                description="Timestamp should not be in the future",
                parameters={"tolerance_minutes": 5},
            )
        )

        # Regulatory validation rules
        self._add_rule(
            ValidationRule(
                name="decimal_precision",
                category=ValidationCategory.REGULATORY,
                severity=ValidationSeverity.WARNING,
                dimension=QualityDimension.ACCURACY,
                description="Check decimal precision for financial compliance",
                parameters={"max_decimal_places": 8},
            )
        )

    def _add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule to registry."""
        self._rules[rule.name] = rule

    @time_execution
    async def validate_market_data(
        self, data: MarketData | list[MarketData], include_statistical: bool = True
    ) -> MarketDataValidationResult | list[MarketDataValidationResult]:
        """
        Validate market data with comprehensive checks.

        Args:
            data: Market data to validate
            include_statistical: Whether to include statistical validation

        Returns:
            Validation result(s)
        """
        try:
            if isinstance(data, list):
                return await self._validate_batch(data, include_statistical)
            else:
                return await self._validate_single(data, include_statistical)

        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            raise

    async def _validate_single(
        self, data: MarketData, include_statistical: bool
    ) -> MarketDataValidationResult:
        """Validate single market data record."""
        issues = []

        # Schema validation
        if self.validation_config["enable_schema_validation"]:
            schema_issues = await self._validate_schema(data)
            issues.extend(schema_issues)

        # Business validation
        if self.validation_config["enable_business_validation"]:
            business_issues = await self._validate_business_rules(data)
            issues.extend(business_issues)

        # Statistical validation
        if include_statistical and self.validation_config["enable_statistical_validation"]:
            statistical_issues = await self._validate_statistical(data)
            issues.extend(statistical_issues)

        # Temporal validation
        if self.validation_config["enable_temporal_validation"]:
            temporal_issues = await self._validate_temporal(data)
            issues.extend(temporal_issues)

        # Regulatory validation
        if self.validation_config["enable_regulatory_validation"]:
            regulatory_issues = await self._validate_regulatory(data)
            issues.extend(regulatory_issues)

        # Calculate quality score
        quality_score = self._calculate_quality_score(data, issues)

        # Determine overall validity
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        is_valid = len(critical_issues) == 0 and len(error_issues) == 0

        return MarketDataValidationResult(
            symbol=data.symbol,
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            metadata={
                "total_issues": len(issues),
                "critical_issues": len(critical_issues),
                "error_issues": len(error_issues),
                "warning_issues": len(
                    [i for i in issues if i.severity == ValidationSeverity.WARNING]
                ),
            },
        )

    async def _validate_batch(
        self, data_list: list[MarketData], include_statistical: bool
    ) -> list[MarketDataValidationResult]:
        """Validate batch of market data records."""
        # Update statistics for statistical validation
        if include_statistical:
            await self._update_statistics(data_list)

        # Validate each record
        results = []
        for data in data_list:
            result = await self._validate_single(data, include_statistical)
            results.append(result)

        return results

    async def _validate_schema(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data schema and structure."""
        issues = []

        rule = self._rules.get("required_fields")
        if rule and rule.enabled:
            required_fields = rule.parameters.get("required_fields", [])

            for field in required_fields:
                value = getattr(data, field, None)
                if value is None:
                    issues.append(
                        ValidationIssue(
                            category=rule.category,
                            severity=rule.severity,
                            dimension=rule.dimension,
                            message=f"Required field '{field}' is missing",
                            field=field,
                            rule_name=rule.name,
                        )
                    )

        # Validate field types
        rule = self._rules.get("field_types")
        if rule and rule.enabled:
            # Check price is numeric
            if data.price is not None:
                try:
                    float(data.price)
                except (ValueError, TypeError):
                    issues.append(
                        ValidationIssue(
                            category=rule.category,
                            severity=rule.severity,
                            dimension=rule.dimension,
                            message="Price must be numeric",
                            field="price",
                            value=data.price,
                            rule_name=rule.name,
                        )
                    )

            # Check volume is numeric
            if data.volume is not None:
                try:
                    float(data.volume)
                except (ValueError, TypeError):
                    issues.append(
                        ValidationIssue(
                            category=rule.category,
                            severity=rule.severity,
                            dimension=rule.dimension,
                            message="Volume must be numeric",
                            field="volume",
                            value=data.volume,
                            rule_name=rule.name,
                        )
                    )

        return issues

    async def _validate_business_rules(self, data: MarketData) -> list[ValidationIssue]:
        """Validate business logic rules."""
        issues = []

        # Positive price check
        rule = self._rules.get("positive_price")
        if rule and rule.enabled and data.price is not None:
            try:
                price_value = float(data.price)
                if price_value <= 0:
                    issues.append(
                        ValidationIssue(
                            category=rule.category,
                            severity=rule.severity,
                            dimension=rule.dimension,
                            message="Price must be positive",
                            field="price",
                            value=price_value,
                            expected="positive number",
                            rule_name=rule.name,
                        )
                    )
            except (ValueError, TypeError):
                pass  # Type validation handled in schema validation

        # Positive volume check
        rule = self._rules.get("positive_volume")
        if rule and rule.enabled and data.volume is not None:
            try:
                volume_value = float(data.volume)
                if volume_value < 0:
                    issues.append(
                        ValidationIssue(
                            category=rule.category,
                            severity=rule.severity,
                            dimension=rule.dimension,
                            message="Volume must be non-negative",
                            field="volume",
                            value=volume_value,
                            expected="non-negative number",
                            rule_name=rule.name,
                        )
                    )
            except (ValueError, TypeError):
                pass

        # Bid-ask spread check
        rule = self._rules.get("bid_ask_spread")
        if rule and rule.enabled and data.bid is not None and data.ask is not None:
            try:
                bid_value = float(data.bid)
                ask_value = float(data.ask)
                if bid_value >= ask_value:
                    issues.append(
                        ValidationIssue(
                            category=rule.category,
                            severity=rule.severity,
                            dimension=rule.dimension,
                            message="Bid price should be less than ask price",
                            field="bid_ask",
                            value=f"bid: {bid_value}, ask: {ask_value}",
                            rule_name=rule.name,
                        )
                    )
            except (ValueError, TypeError):
                pass

        return issues

    async def _validate_statistical(self, data: MarketData) -> list[ValidationIssue]:
        """Validate using statistical methods."""
        issues = []

        # Price outlier detection
        rule = self._rules.get("price_outlier")
        if rule and rule.enabled and data.price is not None and data.symbol in self._price_stats:
            try:
                price_value = float(data.price)
                stats = self._price_stats[data.symbol]

                if "mean" in stats and "std" in stats and stats["std"] > 0:
                    z_score = abs(price_value - stats["mean"]) / stats["std"]
                    threshold = rule.parameters.get("z_score_threshold", 3.0)

                    if z_score > threshold:
                        issues.append(
                            ValidationIssue(
                                category=rule.category,
                                severity=rule.severity,
                                dimension=rule.dimension,
                                message=f"Price is statistical outlier (z-score: {z_score:.2f})",
                                field="price",
                                value=price_value,
                                rule_name=rule.name,
                                metadata={"z_score": z_score, "threshold": threshold},
                            )
                        )
            except (ValueError, TypeError):
                pass

        # Volume spike detection
        rule = self._rules.get("volume_spike")
        if rule and rule.enabled and data.volume is not None and data.symbol in self._volume_stats:
            try:
                volume_value = float(data.volume)
                stats = self._volume_stats[data.symbol]

                if "mean" in stats and stats["mean"] > 0:
                    multiplier = volume_value / stats["mean"]
                    threshold = rule.parameters.get("spike_multiplier", 5.0)

                    if multiplier > threshold:
                        issues.append(
                            ValidationIssue(
                                category=rule.category,
                                severity=rule.severity,
                                dimension=rule.dimension,
                                message=f"Volume spike detected ({multiplier:.1f}x normal)",
                                field="volume",
                                value=volume_value,
                                rule_name=rule.name,
                                metadata={"multiplier": multiplier, "threshold": threshold},
                            )
                        )
            except (ValueError, TypeError):
                pass

        return issues

    async def _validate_temporal(self, data: MarketData) -> list[ValidationIssue]:
        """Validate temporal aspects of data."""
        issues = []

        if data.timestamp is None:
            return issues

        current_time = datetime.now(timezone.utc)

        # Future timestamp check
        rule = self._rules.get("future_timestamp")
        if rule and rule.enabled:
            tolerance_minutes = rule.parameters.get("tolerance_minutes", 5)
            tolerance = timedelta(minutes=tolerance_minutes)

            if data.timestamp > current_time + tolerance:
                issues.append(
                    ValidationIssue(
                        category=rule.category,
                        severity=rule.severity,
                        dimension=rule.dimension,
                        message="Timestamp is in the future",
                        field="timestamp",
                        value=data.timestamp.isoformat(),
                        rule_name=rule.name,
                    )
                )

        # Freshness check
        rule = self._rules.get("timestamp_freshness")
        if rule and rule.enabled:
            max_age_minutes = rule.parameters.get("max_age_minutes", 60)
            max_age = timedelta(minutes=max_age_minutes)

            age = current_time - data.timestamp
            if age > max_age:
                issues.append(
                    ValidationIssue(
                        category=rule.category,
                        severity=rule.severity,
                        dimension=rule.dimension,
                        message=f"Data is stale (age: {age.total_seconds() / 60:.1f} minutes)",
                        field="timestamp",
                        value=data.timestamp.isoformat(),
                        rule_name=rule.name,
                        metadata={"age_minutes": age.total_seconds() / 60},
                    )
                )

        return issues

    async def _validate_regulatory(self, data: MarketData) -> list[ValidationIssue]:
        """Validate regulatory compliance."""
        issues = []

        # Decimal precision check
        rule = self._rules.get("decimal_precision")
        if rule and rule.enabled:
            max_places = rule.parameters.get("max_decimal_places", 8)

            for field_name in ["price", "bid", "ask", "high_price", "low_price", "open_price"]:
                value = getattr(data, field_name, None)
                if value is not None:
                    try:
                        # Convert to Decimal for precise decimal place counting
                        decimal_value = Decimal(str(value))
                        _, digits, exponent = decimal_value.as_tuple()

                        # Count decimal places
                        decimal_places = -exponent if exponent < 0 else 0

                        if decimal_places > max_places:
                            issues.append(
                                ValidationIssue(
                                    category=rule.category,
                                    severity=rule.severity,
                                    dimension=rule.dimension,
                                    message=f"Too many decimal places in {field_name} ({decimal_places} > {max_places})",
                                    field=field_name,
                                    value=value,
                                    rule_name=rule.name,
                                    metadata={
                                        "decimal_places": decimal_places,
                                        "max_allowed": max_places,
                                    },
                                )
                            )
                    except (ValueError, TypeError, InvalidOperation):
                        pass

        return issues

    def _calculate_quality_score(
        self, data: MarketData, issues: list[ValidationIssue]
    ) -> QualityScore:
        """Calculate data quality score based on validation results."""
        # Initialize scores
        scores = {
            QualityDimension.COMPLETENESS: 1.0,
            QualityDimension.ACCURACY: 1.0,
            QualityDimension.CONSISTENCY: 1.0,
            QualityDimension.TIMELINESS: 1.0,
            QualityDimension.VALIDITY: 1.0,
            QualityDimension.UNIQUENESS: 1.0,
        }

        # Deduct points for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                scores[issue.dimension] *= 0.0  # Critical issues make dimension 0
            elif issue.severity == ValidationSeverity.ERROR:
                scores[issue.dimension] *= 0.3  # Severe penalty
            elif issue.severity == ValidationSeverity.WARNING:
                scores[issue.dimension] *= 0.7  # Moderate penalty
            else:  # INFO
                scores[issue.dimension] *= 0.9  # Minor penalty

        # Calculate completeness score based on field availability
        required_fields = ["symbol", "price", "timestamp"]
        optional_fields = ["volume", "bid", "ask", "high_price", "low_price", "open_price"]

        required_present = sum(
            1 for field in required_fields if getattr(data, field, None) is not None
        )
        optional_present = sum(
            1 for field in optional_fields if getattr(data, field, None) is not None
        )

        completeness_base = required_present / len(required_fields)
        completeness_bonus = (optional_present / len(optional_fields)) * 0.2
        completeness_score = min(1.0, completeness_base + completeness_bonus)

        scores[QualityDimension.COMPLETENESS] *= completeness_score

        # Calculate weighted overall score
        weights = self.validation_config["quality_weights"]
        overall_score = sum(
            scores[dimension] * weights.get(dimension.value, 0.0) for dimension in QualityDimension
        )

        return QualityScore(
            overall=overall_score,
            completeness=scores[QualityDimension.COMPLETENESS],
            accuracy=scores[QualityDimension.ACCURACY],
            consistency=scores[QualityDimension.CONSISTENCY],
            timeliness=scores[QualityDimension.TIMELINESS],
            validity=scores[QualityDimension.VALIDITY],
            uniqueness=scores[QualityDimension.UNIQUENESS],
        )

    async def _update_statistics(self, data_list: list[MarketData]) -> None:
        """Update statistical baselines for validation."""
        # Group data by symbol
        symbol_groups = {}
        for data in data_list:
            if data.symbol not in symbol_groups:
                symbol_groups[data.symbol] = []
            symbol_groups[data.symbol].append(data)

        # Update statistics for each symbol
        for symbol, symbol_data in symbol_groups.items():
            # Price statistics
            prices = [float(d.price) for d in symbol_data if d.price is not None]
            if prices:
                if symbol not in self._price_stats:
                    self._price_stats[symbol] = {}

                self._price_stats[symbol].update(
                    {
                        "mean": statistics.mean(prices),
                        "std": statistics.stdev(prices) if len(prices) > 1 else 0.0,
                        "min": min(prices),
                        "max": max(prices),
                        "count": len(prices),
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

            # Volume statistics
            volumes = [float(d.volume) for d in symbol_data if d.volume is not None]
            if volumes:
                if symbol not in self._volume_stats:
                    self._volume_stats[symbol] = {}

                self._volume_stats[symbol].update(
                    {
                        "mean": statistics.mean(volumes),
                        "std": statistics.stdev(volumes) if len(volumes) > 1 else 0.0,
                        "min": min(volumes),
                        "max": max(volumes),
                        "count": len(volumes),
                        "updated_at": datetime.now(timezone.utc),
                    }
                )

    async def add_custom_rule(self, rule: ValidationRule) -> bool:
        """Add custom validation rule."""
        try:
            self._add_rule(rule)
            self.logger.info(f"Added custom validation rule: {rule.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add custom rule: {e}")
            return False

    async def disable_rule(self, rule_name: str) -> bool:
        """Disable validation rule."""
        if rule_name in self._rules:
            self._rules[rule_name].enabled = False
            self.logger.info(f"Disabled validation rule: {rule_name}")
            return True
        return False

    async def enable_rule(self, rule_name: str) -> bool:
        """Enable validation rule."""
        if rule_name in self._rules:
            self._rules[rule_name].enabled = True
            self.logger.info(f"Enabled validation rule: {rule_name}")
            return True
        return False

    async def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_rules": len(self._rules),
            "enabled_rules": len([r for r in self._rules.values() if r.enabled]),
            "rules_by_category": {
                category.value: len([r for r in self._rules.values() if r.category == category])
                for category in ValidationCategory
            },
            "price_stats_symbols": len(self._price_stats),
            "volume_stats_symbols": len(self._volume_stats),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform validator health check."""
        return {
            "status": "healthy",
            "initialized": self._initialized,
            "validation_stats": await self.get_validation_stats(),
            "configuration": {
                "schema_validation": self.validation_config["enable_schema_validation"],
                "business_validation": self.validation_config["enable_business_validation"],
                "statistical_validation": self.validation_config["enable_statistical_validation"],
                "temporal_validation": self.validation_config["enable_temporal_validation"],
                "regulatory_validation": self.validation_config["enable_regulatory_validation"],
            },
        }

    async def cleanup(self) -> None:
        """Cleanup validator resources."""
        try:
            self._price_stats.clear()
            self._volume_stats.clear()
            self._rules.clear()

            self.logger.info("DataValidator cleanup completed")

        except Exception as e:
            self.logger.error(f"DataValidator cleanup error: {e}")
