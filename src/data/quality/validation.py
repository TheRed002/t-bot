"""
Real-time Data Validation System

This module provides comprehensive validation for incoming market data using
consolidated validation utilities to eliminate code duplication.

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility functions and decorators
"""

import statistics
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext
from typing import Any

from src.core import BaseComponent, Config, MarketData, Signal, ValidationLevel

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.recovery_scenarios import DataFeedInterruptionRecovery

# Import from P-007A utilities
from src.utils.decorators import time_execution

# Use consolidated validation utilities
from src.utils.validation.market_data_validation import (
    MarketDataValidationUtils,
    MarketDataValidator,
)
from src.utils.validation.validation_types import ValidationCategory, ValidationIssue


class DataValidator(BaseComponent):
    """
    Comprehensive data validation system for market data quality assurance.

    This class provides real-time validation for all incoming market data,
    using consolidated validation utilities to ensure consistency and reduce duplication.
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
            # Use the dict for easier access to values
            config_source = config
        else:
            self.config = config
            config_source = config

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

        # Initialize consolidated market data validator
        self._market_data_validator = MarketDataValidator(
            enable_precision_validation=True,
            enable_consistency_validation=True,
            enable_timestamp_validation=True,
            max_decimal_places=config_source.get("decimal_precision", 8) if isinstance(config_source, dict) else getattr(config_source, "decimal_precision", 8),
            max_future_seconds=config_source.get("max_future_seconds", 300) if isinstance(config_source, dict) else getattr(config_source, "max_future_seconds", 300),
            max_age_seconds=config_source.get("max_data_age_seconds", 3600) if isinstance(config_source, dict) else getattr(config_source, "max_data_age_seconds", 3600),
        )

        # Statistical tracking for outlier detection
        self.price_history: dict[str, list[float]] = {}
        self.volume_history: dict[str, list[float]] = {}
        self.max_history_size = config_source.get("max_history_size", 1000) if isinstance(config_source, dict) else getattr(config_source, "max_history_size", 1000)
        self.outlier_std_threshold = config_source.get("outlier_std_threshold", 3.0) if isinstance(config_source, dict) else getattr(config_source, "outlier_std_threshold", 3.0)

        # Validation thresholds
        self.price_change_threshold = config_source.get("price_change_threshold", 0.5) if isinstance(config_source, dict) else getattr(config_source, "price_change_threshold", 0.5)
        self.volume_change_threshold = config_source.get("volume_change_threshold", 10.0) if isinstance(config_source, dict) else getattr(config_source, "volume_change_threshold", 10.0)
        self.max_data_age_seconds = config_source.get("max_data_age_seconds", 3600) if isinstance(config_source, dict) else getattr(config_source, "max_data_age_seconds", 3600)

        # Cross-source consistency tracking
        self.source_data: dict[str, dict[str, Any]] = {}
        self.consistency_threshold = config_source.get("consistency_threshold", 0.01) if isinstance(config_source, dict) else getattr(config_source, "consistency_threshold", 0.01)

        self.logger.info("DataValidator initialized with consolidated utilities", config=config)

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
                issue = MarketDataValidationUtils.create_validation_issue(
                    field="data",
                    value=None,
                    expected="valid_market_data",
                    message="Market data is None",
                    level=ValidationLevel.CRITICAL,
                    category=ValidationCategory.SCHEMA,
                    source="DataValidator",
                    metadata={"error_type": "NoneData"},
                )
                return False, [issue]

            # Use consolidated validator for basic validation
            basic_valid = self._market_data_validator.validate_market_data_record(data)
            if not basic_valid:
                # Convert validation errors to ValidationIssue objects
                for error in self._market_data_validator.get_validation_errors():
                    # Try to extract field name from error message
                    field_name = "market_data"  # default

                    # Check for bid/ask spread error
                    if "Bid price" in error and "must be less than ask price" in error:
                        field_name = "bid_ask_spread"
                    elif "bid/ask spread" in error.lower():
                        field_name = "bid_ask_spread"
                    # Check for timestamp errors
                    elif "timestamp is too far in the future" in error:
                        field_name = "future_timestamp"
                    elif "timestamp is too old" in error:
                        field_name = "data_freshness"
                    elif "timestamp" in error.lower() and "future" in error.lower():
                        field_name = "future_timestamp"
                    elif "timestamp" in error.lower() and "old" in error.lower():
                        field_name = "data_freshness"
                    # Error format: '[VALID_000] close 0 below minimum 1E-8 (Category: validation)'
                    # Need to find the field name after the bracket prefix
                    elif "] " in error:
                        # Split at bracket end and get next part
                        after_bracket = error.split("] ", 1)[1]
                        if " " in after_bracket:
                            parts = after_bracket.split(" ")
                            potential_field = parts[0]
                            # Check if it's a known field name
                            if potential_field in ["close", "open", "high", "low", "bid_price", "ask_price", "volume", "price"]:
                                field_name = potential_field

                    issue = MarketDataValidationUtils.create_validation_issue(
                        field=field_name,
                        value=data,
                        expected="valid_market_data",
                        message=error,
                        level=ValidationLevel.HIGH,
                        category=ValidationCategory.SCHEMA,
                        source="ConsolidatedValidator",
                    )
                    issues.append(issue)

            # Statistical outlier detection (custom logic)
            outlier_issues = await self._detect_outliers(data)
            issues.extend(outlier_issues)

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

            issue = MarketDataValidationUtils.create_validation_issue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Validation system error: {e!s}",
                level=ValidationLevel.CRITICAL,
                category=ValidationCategory.INTEGRITY,
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
                    MarketDataValidationUtils.create_validation_issue(
                        field="direction",
                        value=signal.direction,
                        expected="valid_direction",
                        message="Signal direction is required",
                        level=ValidationLevel.CRITICAL,
                        category=ValidationCategory.SCHEMA,
                        source="DataValidator",
                    )
                )

            # Strength validation
            if not (0.0 <= signal.strength <= 1.0):
                issues.append(
                    MarketDataValidationUtils.create_validation_issue(
                        field="confidence",
                        value=signal.strength,
                        expected="0.0_to_1.0",
                        message="Signal confidence must be between 0 and 1",
                        level=ValidationLevel.HIGH,
                        category=ValidationCategory.RANGE,
                        source="DataValidator",
                    )
                )

            # Timestamp validation
            if signal.timestamp > datetime.now(timezone.utc) + timedelta(seconds=60):
                issues.append(
                    MarketDataValidationUtils.create_validation_issue(
                        field="timestamp",
                        value=signal.timestamp,
                        expected="current_or_past_time",
                        message="Signal timestamp is in the future",
                        level=ValidationLevel.MEDIUM,
                        category=ValidationCategory.TEMPORAL,
                        source="DataValidator",
                    )
                )

            # Symbol validation
            if not signal.symbol:
                issues.append(
                    MarketDataValidationUtils.create_validation_issue(
                        field="symbol",
                        value=signal.symbol,
                        expected="valid_symbol",
                        message="Signal symbol is required",
                        level=ValidationLevel.CRITICAL,
                        category=ValidationCategory.SCHEMA,
                        source="DataValidator",
                    )
                )

            # Determine overall validation result
            critical_issues = [i for i in issues if i.level == ValidationLevel.CRITICAL]
            high_issues = [i for i in issues if i.level == ValidationLevel.HIGH]

            is_valid = len(critical_issues) == 0 and len(high_issues) == 0

            return is_valid, issues

        except Exception as e:
            # Create validation issue for the error
            issue = MarketDataValidationUtils.create_validation_issue(
                field="validation_system",
                value="exception",
                expected="successful_validation",
                message=f"Signal validation system error: {e!s}",
                level=ValidationLevel.CRITICAL,
                category=ValidationCategory.INTEGRITY,
                source="DataValidator",
                metadata={"error_type": type(e).__name__},
            )
            return False, [issue]

    async def validate_cross_source_consistency(
        self, primary_data: MarketData, secondary_data: MarketData
    ) -> tuple[bool, list[ValidationIssue]]:
        """
        Validate consistency between different data sources.

        Args:
            primary_data: Primary source market data
            secondary_data: Secondary source market data

        Returns:
            Tuple of (is_consistent, validation_issues)
        """
        issues: list[ValidationIssue] = []

        try:
            if not primary_data or not secondary_data:
                return True, issues

            # Symbol mismatch check
            if primary_data.symbol != secondary_data.symbol:
                issues.append(
                    MarketDataValidationUtils.create_validation_issue(
                        field="symbol_mismatch",
                        value={"primary": primary_data.symbol, "secondary": secondary_data.symbol},
                        expected="matching_symbols",
                        message=f"Symbol mismatch: primary ({primary_data.symbol}) != secondary ({secondary_data.symbol})",
                        level=ValidationLevel.CRITICAL,
                        category=ValidationCategory.INTEGRITY,
                        source="DataValidator",
                        metadata={
                            "primary_symbol": primary_data.symbol,
                            "secondary_symbol": secondary_data.symbol,
                        },
                    )
                )

            # Price consistency check
            if primary_data.close and secondary_data.close:
                price_diff = abs(primary_data.close - secondary_data.close) / primary_data.close

                if price_diff > self.consistency_threshold:
                    issues.append(
                        MarketDataValidationUtils.create_validation_issue(
                            field="price_consistency",
                            value=price_diff,
                            expected=f"<={self.consistency_threshold}",
                            message=f"Price difference {price_diff:.2%} exceeds consistency threshold {self.consistency_threshold:.2%}",
                            level=ValidationLevel.HIGH,
                            category=ValidationCategory.INTEGRITY,
                            source="DataValidator",
                            metadata={
                                "primary_price": str(primary_data.close),
                                "secondary_price": str(secondary_data.close),
                                "difference_percentage": price_diff,
                                "threshold": self.consistency_threshold,
                            },
                        )
                    )

            is_consistent = len(issues) == 0
            return is_consistent, issues

        except Exception as e:
            issue = MarketDataValidationUtils.create_validation_issue(
                field="cross_source_validation",
                value="exception",
                expected="successful_cross_source_validation",
                message=f"Cross-source validation system error: {e!s}",
                level=ValidationLevel.CRITICAL,
                category=ValidationCategory.INTEGRITY,
                source="DataValidator",
                metadata={"error_type": type(e).__name__},
            )
            return False, [issue]

    async def _detect_outliers(self, data: MarketData) -> list[ValidationIssue]:
        """Detect statistical outliers in the data"""
        issues: list[ValidationIssue] = []

        try:
            if data.symbol not in self.price_history or len(self.price_history[data.symbol]) < 10:
                return issues  # Need sufficient history for outlier detection

            price_history = self.price_history[data.symbol]
            getcontext().prec = 28
            current_price_decimal = Decimal(str(data.close))
            current_price = float(current_price_decimal.quantize(Decimal("0.00000001")))

            # Calculate statistics
            mean_price = statistics.mean(price_history)
            std_price = statistics.stdev(price_history) if len(price_history) > 1 else 0

            if std_price > 0:
                z_score = abs(current_price - mean_price) / std_price

                if z_score > self.outlier_std_threshold:
                    issues.append(
                        MarketDataValidationUtils.create_validation_issue(
                            field="price_outlier",
                            value=current_price,
                            expected=f"within_{self.outlier_std_threshold}σ",
                            message=f"Price {current_price} is an outlier (z-score: {z_score:.2f})",
                            level=ValidationLevel.HIGH,
                            category=ValidationCategory.STATISTICAL,
                            source="DataValidator",
                            metadata={
                                "z_score": z_score,
                                "mean_price": mean_price,
                                "std_price": std_price,
                                "threshold": self.outlier_std_threshold,
                            },
                        )
                    )

            return issues

        except Exception as e:
            return [
                MarketDataValidationUtils.create_validation_issue(
                    field="outlier_detection",
                    value="exception",
                    expected="successful_outlier_detection",
                    message=f"Outlier detection failed: {e!s}",
                    level=ValidationLevel.CRITICAL,
                    category=ValidationCategory.INTEGRITY,
                    source="DataValidator",
                    metadata={"error_type": type(e).__name__},
                )
            ]

    async def _update_statistics(self, data: MarketData) -> None:
        """Update statistical tracking for outlier detection"""
        try:
            # Update price history
            if data.symbol not in self.price_history:
                self.price_history[data.symbol] = []

            getcontext().prec = 28
            price_decimal = Decimal(str(data.close))
            self.price_history[data.symbol].append(float(price_decimal.quantize(Decimal("0.00000001"))))

            # Maintain history size
            if len(self.price_history[data.symbol]) > self.max_history_size:
                self.price_history[data.symbol] = self.price_history[data.symbol][
                    -self.max_history_size :
                ]

            # Update volume history if available
            if data.volume:
                if data.symbol not in self.volume_history:
                    self.volume_history[data.symbol] = []

                volume_decimal = Decimal(str(data.volume))
                self.volume_history[data.symbol].append(float(volume_decimal.quantize(Decimal("0.00000001"))))

                # Maintain history size
                if len(self.volume_history[data.symbol]) > self.max_history_size:
                    self.volume_history[data.symbol] = self.volume_history[data.symbol][
                        -self.max_history_size :
                    ]

        except Exception as e:
            if self.error_handler:
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="DataValidator",
                    operation="update_statistics",
                    symbol=data.symbol if data and data.symbol else "unknown",
                    details={"operation": "statistics_update", "data_type": "market_data"},
                )
                await self.error_handler.handle_error(e, error_context)

            self.logger.error(f"Failed to update statistics for {data.symbol}: {e!s}")

    async def get_validation_summary(self) -> dict[str, Any]:
        """Get validation statistics and summary"""
        try:
            return {
                "price_history_size": sum(len(history) for history in self.price_history.values()),
                "volume_history_size": sum(
                    len(history) for history in self.volume_history.values()
                ),
                "validation_config": {
                    "symbols_tracked": len(self.price_history),
                    "max_history_size": self.max_history_size,
                    "outlier_std_threshold": self.outlier_std_threshold,
                    "consistency_threshold": self.consistency_threshold,
                    "price_change_threshold": self.price_change_threshold,
                    "volume_change_threshold": self.volume_change_threshold,
                    "max_data_age_seconds": self.max_data_age_seconds,
                },
                "consolidated_validator": {
                    "precision_validation": self._market_data_validator.enable_precision_validation,
                    "consistency_validation": self._market_data_validator.enable_consistency_validation,
                    "timestamp_validation": self._market_data_validator.enable_timestamp_validation,
                    "max_decimal_places": self._market_data_validator.max_decimal_places,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to generate validation summary: {e!s}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _is_valid_symbol_format(self, symbol: str) -> bool:
        """
        Check if symbol format is valid.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            True if valid format, False otherwise
        """
        try:
            return MarketDataValidationUtils.validate_symbol_format(symbol)
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup validation resources"""
        try:
            # Clear history data
            self.price_history.clear()
            self.volume_history.clear()
            self.source_data.clear()

            # Reset consolidated validator
            self._market_data_validator.reset()

            self.logger.info("DataValidator cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during DataValidator cleanup: {e!s}")
