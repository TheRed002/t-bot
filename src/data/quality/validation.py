"""
Real-time Data Validation System

This module provides comprehensive validation for incoming market data, including:
- Schema validation for data structure
- Range checks and business rule validation
- Statistical outlier detection
- Data freshness monitoring
- Cross-source consistency checks

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.base import BaseComponent
from src.core.config import Config

# Import from P-001 core components
from src.core.types import MarketData, Signal, ValidationLevel

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.recovery_scenarios import DataFeedInterruptionRecovery

# Import from P-007A utilities
from src.utils.decorators import time_execution

# ValidationLevel and ValidationResult are now imported from core.types


@dataclass
class ValidationIssue:
    """Data validation issue record"""

    field: str
    value: Any
    expected: Any
    message: str
    level: ValidationLevel
    timestamp: datetime
    source: str
    metadata: dict[str, Any]


class DataValidator(BaseComponent):
    """
    Comprehensive data validation system for market data quality assurance.

    This class provides real-time validation for all incoming market data,
    ensuring data quality for ML models and trading strategies.
    """

    def __init__(self, config):
        """
        Initialize the data validator with configuration.

        Args:
            config: Application configuration or config dict
        """

        # Handle both Config object and dict
        super().__init__()  # Initialize BaseComponent
        if isinstance(config, dict):
            # Create a minimal Config object for testing
            self.config = type("Config", (), config)()
            for key, value in config.items():
                setattr(self.config, key, value)
        else:
            self.config = config

        # Initialize error handling components only if config is a Config object
        if isinstance(config, Config):
            self.error_handler = ErrorHandler(config)
            self.recovery_scenario = DataFeedInterruptionRecovery(config)
            self.pattern_analytics = ErrorPatternAnalytics(config)
        else:
            # For testing with dict config, skip error handling components
            self.error_handler = None
            self.recovery_scenario = None
            self.pattern_analytics = None

        # Validation thresholds
        self.price_change_threshold = getattr(
            config, "price_change_threshold", 0.5
        )  # 50% max change
        self.volume_change_threshold = getattr(
            config, "volume_change_threshold", 10.0
        )  # 1000% max change
        self.outlier_std_threshold = getattr(
            config, "outlier_std_threshold", 3.0
        )  # 3 standard deviations
        self.max_data_age_seconds = getattr(config, "max_data_age_seconds", 60)  # 1 minute max age

        # Statistical tracking for outlier detection
        self.price_history: dict[str, list[float]] = {}
        self.volume_history: dict[str, list[float]] = {}
        self.max_history_size = getattr(config, "max_history_size", 1000)

        # Cross-source consistency tracking
        self.source_data: dict[str, dict[str, Any]] = {}
        self.consistency_threshold = getattr(
            config, "consistency_threshold", 0.01
        )  # 1% max difference

        self.logger.info("DataValidator initialized", config=config)

    @time_execution
    async def validate_market_data(self, data: MarketData) -> tuple[bool, list[ValidationIssue]]:
        """
        Validate market data for quality and consistency.

        Args:
            data: Market data to validate

        Returns:
            Tuple of (is_valid, validation_issues)
        """
        issues = []

        try:
            # Handle None data
            if data is None:
                issue = ValidationIssue(
                    field="data",
                    value=None,
                    expected="valid_market_data",
                    message="Market data is None",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata={"error_type": "NoneData"},
                )
                return False, [issue]

            # Schema validation
            schema_issues = await self._validate_schema(data)
            issues.extend(schema_issues)

            # Range checks
            range_issues = await self._validate_ranges(data)
            issues.extend(range_issues)

            # Business rule validation
            business_issues = await self._validate_business_rules(data)
            issues.extend(business_issues)

            # Statistical outlier detection
            outlier_issues = await self._detect_outliers(data)
            issues.extend(outlier_issues)

            # Data freshness check
            freshness_issues = await self._validate_freshness(data)
            issues.extend(freshness_issues)

            # Update statistical tracking
            await self._update_statistics(data)

            # Determine overall validation result
            critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
            high_issues = [i for i in issues if i.level == ValidationLevel.HIGH]

            is_valid = len(critical_issues) == 0 and len(high_issues) == 0

            if issues:
                self.logger.warning(
                    "Market data validation issues detected",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    issue_count=len(issues),
                    critical_count=len(critical_issues),
                    high_count=len(high_issues),
                )
            else:
                self.logger.debug(
                    "Market data validation passed", symbol=data.symbol if data else "unknown"
                )

            return is_valid, issues

        except Exception as e:
            # Use ErrorHandler for comprehensive error management
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_market_data",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={"data_type": "market_data", "validation_stage": "main_validation"},
                )

                # Handle the error through the error handling framework
                await self.error_handler.handle_error(e, error_context)

            # Add error event to pattern analytics
            if self.pattern_analytics and error_context:
                self.pattern_analytics.add_error_event(error_context.__dict__)

            # Create validation issue for the error
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            issue = ValidationIssue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Validation system error: {e!s}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata=metadata,
            )
            return False, [issue]

    @time_execution
    async def validate_signal(self, signal: Signal) -> tuple[bool, list[ValidationIssue]]:
        """
        Validate trading signal for quality and consistency.

        Args:
            signal: Trading signal to validate

        Returns:
            Tuple of (is_valid, validation_issues)
        """
        issues = []

        try:
            # Basic signal validation
            if not signal.direction:
                issues.append(
                    ValidationIssue(
                        field="direction",
                        value=signal.direction,
                        expected="valid_direction",
                        message="Signal direction is required",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Confidence validation
            if not (0.0 <= signal.confidence <= 1.0):
                issues.append(
                    ValidationIssue(
                        field="confidence",
                        value=signal.confidence,
                        expected="0.0_to_1.0",
                        message="Signal confidence must be between 0 and 1",
                        level=ValidationLevel.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Timestamp validation
            if signal.timestamp > datetime.now(timezone.utc) + timedelta(seconds=60):
                issues.append(
                    ValidationIssue(
                        field="timestamp",
                        value=signal.timestamp,
                        expected="current_or_past_time",
                        message="Signal timestamp is in the future",
                        level=ValidationLevel.MEDIUM,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Symbol validation
            if not signal.symbol:
                issues.append(
                    ValidationIssue(
                        field="symbol",
                        value=signal.symbol,
                        expected="valid_symbol",
                        message="Signal symbol is required",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Determine overall validation result
            critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
            high_issues = [i for i in issues if i.level == ValidationLevel.HIGH]

            is_valid = len(critical_issues) == 0 and len(high_issues) == 0

            if issues:
                self.logger.warning(
                    "Signal validation issues detected",
                    symbol=signal.symbol if signal else "unknown",
                    issue_count=len(issues),
                    critical_count=len(critical_issues),
                    high_count=len(high_issues),
                )
            else:
                self.logger.debug(
                    "Signal validation passed", symbol=signal.symbol if signal else "unknown"
                )

            return is_valid, issues

        except Exception as e:
            # Use ErrorHandler for comprehensive error management
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_signal",
                    symbol=signal.symbol if signal else "unknown",
                    details={"data_type": "signal", "validation_stage": "signal_validation"},
                )

                # Handle the error through the error handling framework
                await self.error_handler.handle_error(e, error_context)

            # Add error event to pattern analytics
            if self.pattern_analytics and error_context:
                self.pattern_analytics.add_error_event(error_context.__dict__)

            # Create validation issue for the error
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            issue = ValidationIssue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Signal validation system error: {e!s}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata=metadata,
            )
            return False, [issue]

    async def validate_cross_source_consistency(
        self, primary_data: MarketData, secondary_data: MarketData
    ) -> tuple[bool, list[ValidationIssue]]:
        """
        Public method to validate consistency between different data sources.

        Args:
            primary_data: Primary source market data
            secondary_data: Secondary source market data

        Returns:
            Tuple of (is_consistent, validation_issues)
        """
        return await self._validate_cross_source_consistency(primary_data, secondary_data)

    async def _validate_schema(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data schema and structure"""
        issues = []

        try:
            # Required fields validation
            required_fields = ["symbol", "price", "volume", "timestamp"]
            for field in required_fields:
                if not hasattr(data, field) or getattr(data, field) is None:
                    issues.append(
                        ValidationIssue(
                            field=field,
                            value=getattr(data, field, None),
                            expected="non_null_value",
                            message=f"Required field {field} is missing or null",
                            level=ValidationLevel.CRITICAL,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={},
                        )
                    )

            # Data type validation
            if not isinstance(data.symbol, str):
                issues.append(
                    ValidationIssue(
                        field="symbol",
                        value=type(data.symbol),
                        expected="str",
                        message="Symbol must be a string",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            if not isinstance(data.price, Decimal):
                issues.append(
                    ValidationIssue(
                        field="price",
                        value=type(data.price),
                        expected="Decimal",
                        message="Price must be a Decimal",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            if not isinstance(data.timestamp, datetime):
                issues.append(
                    ValidationIssue(
                        field="timestamp",
                        value=type(data.timestamp),
                        expected="datetime",
                        message="Timestamp must be a datetime",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            return issues

        except Exception as e:
            # Use ErrorHandler for schema validation errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_schema",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={"validation_stage": "schema_validation", "data_type": "market_data"},
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            # Return critical issue for schema validation failure
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            return [
                ValidationIssue(
                    field="schema_validation",
                    value="exception",
                    expected="successful_schema_validation",
                    message=f"Schema validation failed: {e!s}",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata=metadata,
                )
            ]

    async def _validate_ranges(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data ranges and bounds"""
        issues = []

        try:
            # Price range validation
            if data.price <= 0:
                issues.append(
                    ValidationIssue(
                        field="price",
                        value=float(data.price),
                        expected="positive_value",
                        message="Price must be positive",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Volume range validation
            if data.volume < 0:
                issues.append(
                    ValidationIssue(
                        field="volume",
                        value=float(data.volume),
                        expected="non_negative_value",
                        message="Volume must be non-negative",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Price change validation (if previous price exists)
            if data.symbol in self.price_history and len(self.price_history[data.symbol]) > 0:
                prev_price = Decimal(str(self.price_history[data.symbol][-1]))
                price_change = abs(data.price - prev_price) / prev_price

                if price_change > self.price_change_threshold:
                    issues.append(
                        ValidationIssue(
                            field="price_change",
                            value=price_change,
                            expected=f"<={self.price_change_threshold}",
                            message=f"Price change {price_change:.2%} exceeds threshold {self.price_change_threshold:.2%}",
                            level=ValidationLevel.HIGH,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={
                                "previous_price": prev_price,
                                "current_price": float(data.price),
                                "change_percentage": price_change,
                            },
                        )
                    )

            # Volume change validation (if previous volume exists)
            if data.symbol in self.volume_history and len(self.volume_history[data.symbol]) > 0:
                prev_volume = Decimal(str(self.volume_history[data.symbol][-1]))
                volume_change = abs(data.volume - prev_volume) / prev_volume

                if volume_change > self.volume_change_threshold:
                    issues.append(
                        ValidationIssue(
                            field="volume_change",
                            value=volume_change,
                            expected=f"<={self.volume_change_threshold}",
                            message=f"Volume change {volume_change:.2%} exceeds threshold {self.volume_change_threshold:.2%}",
                            level=ValidationLevel.HIGH,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={
                                "previous_volume": prev_volume,
                                "current_volume": float(data.volume),
                                "change_percentage": volume_change,
                            },
                        )
                    )

            return issues

        except Exception as e:
            # Use ErrorHandler for range validation errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_ranges",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={"validation_stage": "range_validation", "data_type": "market_data"},
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            # Return critical issue for range validation failure
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            return [
                ValidationIssue(
                    field="range_validation",
                    value="exception",
                    expected="successful_range_validation",
                    message=f"Range validation failed: {e!s}",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata=metadata,
                )
            ]

    async def _validate_business_rules(self, data: MarketData) -> list[ValidationIssue]:
        """Validate business rules and trading logic"""
        issues = []

        try:
            # Symbol format validation
            if not data.symbol or len(data.symbol) < 3:
                issues.append(
                    ValidationIssue(
                        field="symbol",
                        value=data.symbol,
                        expected="valid_symbol_format",
                        message="Symbol must be at least 3 characters",
                        level=ValidationLevel.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Price precision validation (basic check)
            if data.price and len(str(data.price).split(".")[-1]) > 8:
                issues.append(
                    ValidationIssue(
                        field="price",
                        value=float(data.price),
                        expected="reasonable_precision",
                        message="Price precision exceeds 8 decimal places",
                        level=ValidationLevel.MEDIUM,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Volume precision validation
            if data.volume and len(str(data.volume).split(".")[-1]) > 8:
                issues.append(
                    ValidationIssue(
                        field="volume",
                        value=float(data.volume),
                        expected="reasonable_precision",
                        message="Volume precision exceeds 8 decimal places",
                        level=ValidationLevel.MEDIUM,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={},
                    )
                )

            # Bid/Ask spread validation
            if (
                hasattr(data, "bid")
                and hasattr(data, "ask")
                and data.bid is not None
                and data.ask is not None
            ):
                if data.bid >= data.ask:
                    issues.append(
                        ValidationIssue(
                            field="bid_ask_spread",
                            value={"bid": float(data.bid), "ask": float(data.ask)},
                            expected="bid < ask",
                            message=f"Invalid bid/ask spread: bid ({data.bid}) >= ask ({data.ask})",
                            level=ValidationLevel.CRITICAL,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={"bid": float(data.bid), "ask": float(data.ask)},
                        )
                    )

                # Check for extremely wide spreads (>5%)
                spread_pct = ((data.ask - data.bid) / data.ask) * 100
                if spread_pct > 5.0:
                    issues.append(
                        ValidationIssue(
                            field="bid_ask_spread",
                            value={"spread_percentage": float(spread_pct)},
                            expected="spread < 5%",
                            message=f"Abnormally wide bid/ask spread: {spread_pct:.2f}%",
                            level=ValidationLevel.HIGH,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={"spread_percentage": float(spread_pct)},
                        )
                    )

            return issues

        except Exception as e:
            # Use ErrorHandler for business rule validation errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_business_rules",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={
                        "validation_stage": "business_rules_validation",
                        "data_type": "market_data",
                    },
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            # Return critical issue for business rule validation failure
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            return [
                ValidationIssue(
                    field="business_rules_validation",
                    value="exception",
                    expected="successful_business_rules_validation",
                    message=f"Business rules validation failed: {e!s}",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata=metadata,
                )
            ]

    async def _detect_outliers(self, data: MarketData) -> list[ValidationIssue]:
        """Detect statistical outliers in the data"""
        issues: list[ValidationIssue] = []

        try:
            if data.symbol not in self.price_history or len(self.price_history[data.symbol]) < 10:
                return issues  # Need sufficient history for outlier detection

            price_history = self.price_history[data.symbol]
            current_price = float(data.price)

            # Calculate statistics
            mean_price = statistics.mean(price_history)
            std_price = statistics.stdev(price_history) if len(price_history) > 1 else 0

            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price

                if z_score > self.outlier_std_threshold:
                    issues.append(
                        ValidationIssue(
                            field="price_outlier",
                            value=current_price,
                            expected=f"within_{self.outlier_std_threshold}σ",
                            message=f"Price {current_price} is an outlier (z-score: {z_score:.2f})",
                            level=ValidationLevel.HIGH,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={
                                "z_score": z_score,
                                "mean_price": mean_price,
                                "std_price": std_price,
                                "threshold": self.outlier_std_threshold,
                            },
                        )
                    )

            # Volume outlier detection
            if data.symbol in self.volume_history and len(self.volume_history[data.symbol]) >= 10:
                volume_history = self.volume_history[data.symbol]
                current_volume = float(data.volume)

                mean_volume = statistics.mean(volume_history)
                std_volume = statistics.stdev(volume_history) if len(volume_history) > 1 else 0

                if std_volume > 0:
                    z_score = abs(current_volume - mean_volume) / std_volume

                    if z_score > self.outlier_std_threshold:
                        issues.append(
                            ValidationIssue(
                                field="volume",
                                value=current_volume,
                                expected=f"within_{self.outlier_std_threshold}σ",
                                message=f"Volume {current_volume} is an outlier (z-score: {z_score:.2f})",
                                level=ValidationLevel.HIGH,
                                timestamp=datetime.now(timezone.utc),
                                source="DataValidator",
                                metadata={
                                    "z_score": z_score,
                                    "mean_volume": mean_volume,
                                    "std_volume": std_volume,
                                    "threshold": self.outlier_std_threshold,
                                },
                            )
                        )

            return issues

        except Exception as e:
            # Use ErrorHandler for outlier detection errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="detect_outliers",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={"validation_stage": "outlier_detection", "data_type": "market_data"},
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            # Return critical issue for outlier detection failure
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            return [
                ValidationIssue(
                    field="outlier_detection",
                    value="exception",
                    expected="successful_outlier_detection",
                    message=f"Outlier detection failed: {e!s}",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata=metadata,
                )
            ]

    async def _validate_freshness(self, data: MarketData) -> list[ValidationIssue]:
        """Validate data freshness and timeliness"""
        issues = []

        try:
            current_time = datetime.now(timezone.utc)
            data_age = (current_time - data.timestamp).total_seconds()

            if data_age > self.max_data_age_seconds:
                issues.append(
                    ValidationIssue(
                        field="data_freshness",
                        value=data.timestamp,
                        expected=f"within_{self.max_data_age_seconds}s",
                        message=f"Data is {data_age:.1f}s old, exceeds {self.max_data_age_seconds}s threshold",
                        level=ValidationLevel.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "data_age_seconds": data_age,
                            "max_age_seconds": self.max_data_age_seconds,
                            "current_time": current_time.isoformat(),
                            "data_time": data.timestamp.isoformat(),
                        },
                    )
                )

            # Check for future timestamps
            if data.timestamp > current_time + timedelta(seconds=5):
                issues.append(
                    ValidationIssue(
                        field="future_timestamp",
                        value=data.timestamp,
                        expected="current_or_past_time",
                        message="Data timestamp is in the future",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "data_time": data.timestamp.isoformat(),
                            "current_time": current_time.isoformat(),
                            "time_difference": (data.timestamp - current_time).total_seconds(),
                        },
                    )
                )

            return issues

        except Exception as e:
            # Use ErrorHandler for freshness validation errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_freshness",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={
                        "validation_stage": "freshness_validation",
                        "data_type": "market_data",
                    },
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            # Return critical issue for freshness validation failure
            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            return [
                ValidationIssue(
                    field="freshness_validation",
                    value="exception",
                    expected="successful_freshness_validation",
                    message=f"Freshness validation failed: {e!s}",
                    level=ValidationLevel.CRITICAL,
                    timestamp=datetime.now(timezone.utc),
                    source="DataValidator",
                    metadata=metadata,
                )
            ]

    async def _update_statistics(self, data: MarketData) -> None:
        """Update statistical tracking for outlier detection"""
        try:
            # Update price history
            if data.symbol not in self.price_history:
                self.price_history[data.symbol] = []

            self.price_history[data.symbol].append(float(data.price))

            # Maintain history size
            if len(self.price_history[data.symbol]) > self.max_history_size:
                self.price_history[data.symbol] = self.price_history[data.symbol][
                    -self.max_history_size :
                ]

            # Update volume history
            if data.symbol not in self.volume_history:
                self.volume_history[data.symbol] = []

            self.volume_history[data.symbol].append(float(data.volume))

            # Maintain history size
            if len(self.volume_history[data.symbol]) > self.max_history_size:
                self.volume_history[data.symbol] = self.volume_history[data.symbol][
                    -self.max_history_size :
                ]

        except Exception as e:
            # Use ErrorHandler for statistics update errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="update_statistics",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={"operation": "statistics_update", "data_type": "market_data"},
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to update statistics for {data.symbol}: {e!s}")

    async def _validate_cross_source_consistency(
        self, primary_data: MarketData, secondary_data: MarketData
    ) -> tuple[bool, list[ValidationIssue]]:
        """Validate consistency between different data sources"""
        issues: list[ValidationIssue] = []

        try:
            if not primary_data or not secondary_data:
                return True, issues

            # Symbol mismatch check
            if primary_data.symbol != secondary_data.symbol:
                issues.append(
                    ValidationIssue(
                        field="symbol_mismatch",
                        value={"primary": primary_data.symbol, "secondary": secondary_data.symbol},
                        expected="matching_symbols",
                        message=f"Symbol mismatch: primary ({primary_data.symbol}) != secondary ({secondary_data.symbol})",
                        level=ValidationLevel.CRITICAL,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "primary_symbol": primary_data.symbol,
                            "secondary_symbol": secondary_data.symbol,
                        },
                    )
                )

            # Price consistency check
            price_diff = abs(primary_data.price - secondary_data.price) / primary_data.price

            if price_diff > self.consistency_threshold:
                issues.append(
                    ValidationIssue(
                        field="price_consistency",
                        value=price_diff,
                        expected=f"<={self.consistency_threshold}",
                        message=f"Price difference {price_diff:.2%} exceeds consistency threshold {self.consistency_threshold:.2%}",
                        level=ValidationLevel.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        source="DataValidator",
                        metadata={
                            "primary_price": float(primary_data.price),
                            "secondary_price": float(secondary_data.price),
                            "difference_percentage": price_diff,
                            "threshold": self.consistency_threshold,
                        },
                    )
                )

            # Volume consistency check
            if primary_data.volume > 0 and secondary_data.volume > 0:
                volume_diff = abs(primary_data.volume - secondary_data.volume) / primary_data.volume

                if volume_diff > self.consistency_threshold:
                    issues.append(
                        ValidationIssue(
                            field="volume_consistency",
                            value=volume_diff,
                            expected=f"<={self.consistency_threshold}",
                            message=f"Volume difference {volume_diff:.2%} exceeds consistency threshold {self.consistency_threshold:.2%}",
                            level=ValidationLevel.HIGH,
                            timestamp=datetime.now(timezone.utc),
                            source="DataValidator",
                            metadata={
                                "primary_volume": float(primary_data.volume),
                                "secondary_volume": float(secondary_data.volume),
                                "difference_percentage": volume_diff,
                                "threshold": self.consistency_threshold,
                            },
                        )
                    )

            is_consistent = len(issues) == 0

            if is_consistent:
                self.logger.debug("Cross-source consistency validated", symbol=primary_data.symbol)
            else:
                self.logger.warning(
                    "Cross-source consistency issues detected",
                    symbol=primary_data.symbol,
                    issue_count=len(issues),
                )

            return is_consistent, issues

        except Exception as e:
            # Use ErrorHandler for cross-source validation errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="validate_cross_source_consistency",
                    symbol=(
                        primary_data.symbol if primary_data and primary_data.symbol else "unknown"
                    ),
                    details={
                        "validation_stage": "cross_source_consistency",
                        "data_type": "market_data",
                    },
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            metadata = {
                "error_type": type(e).__name__,
            }
            if error_context:
                metadata["error_id"] = error_context.error_id
                metadata["severity"] = error_context.severity.value

            issue = ValidationIssue(
                field="cross_source_validation",
                value="exception",
                expected="successful_cross_source_validation",
                message=f"Cross-source validation system error: {e!s}",
                level=ValidationLevel.CRITICAL,
                timestamp=datetime.now(timezone.utc),
                source="DataValidator",
                metadata=metadata,
            )
            return False, [issue]

    async def get_validation_summary(self) -> dict[str, Any]:
        """Get validation statistics and summary"""
        try:
            # Get error pattern summary from analytics
            pattern_summary = (
                self.pattern_analytics.get_pattern_summary() if self.pattern_analytics else {}
            )
            correlation_summary = (
                self.pattern_analytics.get_correlation_summary() if self.pattern_analytics else {}
            )
            trend_summary = (
                self.pattern_analytics.get_trend_summary() if self.pattern_analytics else {}
            )

            # Get circuit breaker status
            circuit_breaker_status = (
                self.error_handler.get_circuit_breaker_status() if self.error_handler else {}
            )

            return {
                "price_history_size": sum(len(history) for history in self.price_history.values()),
                "volume_history_size": sum(
                    len(history) for history in self.volume_history.values()
                ),
                "validation_config": {
                    "symbols_tracked": len(self.price_history),
                    "max_history_size": self.max_history_size,
                    "price_change_threshold": self.price_change_threshold,
                    "volume_change_threshold": self.volume_change_threshold,
                    "outlier_std_threshold": self.outlier_std_threshold,
                    "max_data_age_seconds": self.max_data_age_seconds,
                    "consistency_threshold": self.consistency_threshold,
                },
                "validation_stats": {
                    "symbols_tracked": len(self.price_history),
                    "max_history_size": self.max_history_size,
                    "price_change_threshold": self.price_change_threshold,
                    "volume_change_threshold": self.volume_change_threshold,
                    "outlier_std_threshold": self.outlier_std_threshold,
                    "max_data_age_seconds": self.max_data_age_seconds,
                    "consistency_threshold": self.consistency_threshold,
                },
                "error_patterns": pattern_summary,
                "error_correlations": correlation_summary,
                "error_trends": trend_summary,
                "circuit_breaker_status": circuit_breaker_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # Use ErrorHandler for summary generation errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="get_validation_summary",
                    details={"operation": "summary_generation"},
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to generate validation summary: {e!s}")
            result = {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if error_context:
                result["error_id"] = error_context.error_id
            return result

    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """
        Validate symbol format according to common trading conventions.

        Args:
            symbol: Trading symbol to validate

        Returns:
            True if symbol format is valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False

        # Must be at least 6 characters for trading pairs (e.g., BTCUSD)
        if len(symbol) < 6:
            return False

        # Must contain only alphanumeric characters and allowed separators
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        if not all(c in allowed_chars for c in symbol.upper()):
            return False

        # Must not start or end with separator
        if symbol.startswith("-") or symbol.endswith("-"):
            return False

        # Must not have consecutive separators
        if "--" in symbol:
            return False

        return True

    async def cleanup(self) -> None:
        """Cleanup validation resources"""
        try:
            # Clear history data
            self.price_history.clear()
            self.volume_history.clear()
            self.source_data.clear()

            self.logger.info("DataValidator cleanup completed")

        except Exception as e:
            # Use ErrorHandler for cleanup errors
            error_context = None
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="cleanup",
                    details={"operation": "cleanup"},
                )

                await self.error_handler.handle_error(e, error_context)
                if self.pattern_analytics:
                    self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Error during DataValidator cleanup: {e!s}")
