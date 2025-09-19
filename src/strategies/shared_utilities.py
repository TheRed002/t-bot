"""
Shared Strategy Utilities - Comprehensive integration patterns for all strategies.

This module provides a comprehensive set of utilities that integrate ALL available
modules to maximize code reuse and ensure consistent patterns across all strategies.
It follows the module hierarchy and avoids circular dependencies.

CRITICAL: This module is Level 7 - only imports from Levels 1-6 (core, utils, 
error_handling, database, monitoring, state, data, exchanges, risk_management).
"""

from abc import ABC
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from src.core.caching.cache_decorators import cached
from src.core.caching.cache_manager import CacheManager, get_cache_manager

# Level 1: Core module integration
from src.core.exceptions import ExecutionError, ServiceError, StrategyError
from src.core.logging import get_logger
from src.core.types import MarketData, Signal, SignalDirection

# Level 6: State and Data integration
from src.data.interfaces import DataServiceInterface

# Level 3: Error handling integration
from src.error_handling import (
    ErrorHandler,
    ErrorSeverity,
    get_global_error_handler,
    with_circuit_breaker,
    with_error_context,
)
from src.error_handling.context import ErrorContext

# Level 5: Monitoring integration
from src.monitoring import (
    AlertManager,
    # PerformanceMonitor, TelemetryCollector  # Not available in monitoring module
    MetricsCollector,
    get_tracer,
)
from src.monitoring.alerting import Alert, AlertSeverity
from src.utils.datetime_utils import to_timestamp
from src.utils.decimal_utils import round_to_precision

# Level 2: Utils module integration
from src.utils.decorators import time_execution
from src.utils.formatters import format_currency, format_percentage

# Service registry not available yet
from src.utils.validation.market_data_validation import MarketDataValidator
from src.utils.validation.service import ValidationService


class StrategyIntegratedBase(ABC):
    """
    Enhanced base class that provides comprehensive module integration.
    
    This class provides all the shared functionality that strategies need,
    integrating ALL available modules while maintaining proper hierarchy.
    """

    def __init__(self,
                 strategy_name: str,
                 config: dict[str, Any],
                 validation_service: Optional["ValidationService"] = None):
        """Initialize with comprehensive module integration."""
        self.strategy_name = strategy_name
        self.config = config

        # Core integration
        self.logger = get_logger(f"strategy.{strategy_name}")
        self.tracer = get_tracer(f"strategy.{strategy_name}")

        # Cache management
        self.cache_manager: CacheManager = get_cache_manager()

        # Error handling
        self.error_handler: ErrorHandler = get_global_error_handler()
        self.error_context = ErrorContext(operation_name=f"strategy_{strategy_name}")

        # Validation - use dependency injection
        self.market_data_validator = MarketDataValidator()
        self.validation_service = validation_service  # Injected dependency

        # Monitoring and telemetry
        self._metrics_collector: "MetricsCollector | None" = None
        self._alert_manager: "AlertManager | None" = None
        self._performance_monitor = None  # Will be integrated later
        self._telemetry_collector = None  # Will be integrated later

        # Strategy-specific performance tracking handled by monitoring service

        # Initialize metrics storage
        self._signal_metrics: dict[str, Any] = {
            "total_generated": 0,
            "total_valid": 0,
            "total_executed": 0,
            "success_rate": 0.0,
            "average_confidence": 0.0
        }

        self.logger.info(
            "Strategy integrated base initialized",
            strategy=strategy_name,
            integrations_available=self._get_available_integrations()
        )

    async def initialize_validation_service(self) -> None:
        """Initialize validation service using dependency injection if not already set."""
        if self.validation_service is None:
            try:
                from src.core.dependency_injection import injector
                from src.utils.service_registry import register_util_services

                # Ensure util services are registered
                register_util_services()

                # Resolve validation service
                self.validation_service = injector.resolve("ValidationServiceInterface")

                # Initialize if needed
                if not self.validation_service.is_running:
                    await self.validation_service.initialize()

                self.logger.debug(
                    "ValidationService initialized via dependency injection",
                    strategy=self.strategy_name
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to initialize ValidationService via DI, using fallback",
                    strategy=self.strategy_name,
                    error=str(e)
                )
                # Leave validation service as None if DI fails - strategies should handle this
                self.validation_service = None
                self.logger.info("ValidationService not available - strategies will use basic validation")

    def set_monitoring_services(self,
                               metrics_collector: "MetricsCollector | None" = None,
                               alert_manager: "AlertManager | None" = None,
                               performance_monitor = None,
                               telemetry_collector = None):
        """Set monitoring services for comprehensive telemetry."""
        self._metrics_collector = metrics_collector
        self._alert_manager = alert_manager
        self._performance_monitor = performance_monitor
        self._telemetry_collector = telemetry_collector

        if metrics_collector:
            self.logger.info("Metrics collector configured", strategy=self.strategy_name)

    def _get_available_integrations(self) -> dict[str, bool]:
        """Get status of available integrations."""
        return {
            "cache_manager": self.cache_manager is not None,
            "error_handler": self.error_handler is not None,
            "market_data_validator": self.market_data_validator is not None,
            "validation_service": self.validation_service is not None,
            "metrics_collector": self._metrics_collector is not None,
            "alert_manager": self._alert_manager is not None,
            "performance_monitor": self._performance_monitor is not None,
            "telemetry_collector": self._telemetry_collector is not None,
        }

    @cached(ttl=60, namespace="strategy_validation")
    async def validate_market_data_comprehensive(self, data: MarketData) -> tuple[bool, list[str]]:
        """Comprehensive market data validation using all available validators."""
        errors = []

        # Ensure validation service is initialized
        await self.initialize_validation_service()

        try:
            # Core validation
            if not data:
                errors.append("Market data is None")
                return False, errors

            # Use market data validator
            validation_result = await self.market_data_validator.validate_market_data(data)
            if not validation_result.is_valid:
                errors.extend(validation_result.errors)

            # Additional validations using ValidationService
            if self.validation_service:
                try:
                    if not self.validation_service.validate_symbol(data.symbol):
                        errors.append(f"Invalid symbol format: {data.symbol}")
                except Exception as e:
                    errors.append(f"Symbol validation error: {e}")

                try:
                    if not self.validation_service.validate_price(data.price):
                        errors.append(f"Price out of range: {data.price}")
                except Exception as e:
                    errors.append(f"Price validation error: {e}")

            # Check timestamp freshness (data older than 1 hour)
            from datetime import timedelta
            if data.timestamp and (datetime.now(timezone.utc) - data.timestamp.replace(tzinfo=timezone.utc)) > timedelta(hours=1):
                errors.append(f"Data too old: {to_timestamp(data.timestamp)}")

            # Record validation metrics
            if self._metrics_collector:
                self._metrics_collector.increment_counter(
                    "strategy_data_validations",
                    labels={
                        "strategy": self.strategy_name,
                        "symbol": data.symbol,
                        "result": "valid" if not errors else "invalid"
                    }
                )

            return len(errors) == 0, errors

        except Exception as e:
            error_msg = f"Validation error: {e!s}"
            errors.append(error_msg)

            # Handle validation errors
            if self.error_handler:
                await self.error_handler.handle_error(
                    error=e,
                    context={"strategy": self.strategy_name, "operation": "validate_market_data"},
                    severity=ErrorSeverity.MEDIUM
                )

            return False, errors

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=30)
    @with_error_context(operation="calculate_technical_indicators")
    @time_execution
    async def calculate_technical_indicators(self,
                                           data: MarketData,
                                           indicators: list[str],
                                           periods: dict[str, int] = None) -> "dict[str, Decimal | None]":
        """Calculate technical indicators with comprehensive error handling and caching."""
        periods = periods or {}
        results = {}

        try:
            # Validate input
            is_valid, errors = await self.validate_market_data_comprehensive(data)
            if not is_valid:
                self.logger.warning(
                    "Invalid data for indicator calculation",
                    strategy=self.strategy_name,
                    errors=errors
                )
                return {indicator: None for indicator in indicators}

            # Cache key for this calculation
            cache_key = f"indicators_{data.symbol}_{hash(tuple(sorted(indicators)))}"

            # Try to get from cache first
            if self.cache_manager:
                try:
                    cached_result = await self.cache_manager.get(cache_key)
                    if cached_result:
                        return cached_result
                except Exception as e:
                    self.logger.warning(
                        "Failed to get cached indicators",
                        strategy=self.strategy_name,
                        error=str(e)
                    )

            price = data.price if isinstance(data.price, Decimal) else Decimal(str(data.price))

            for indicator in indicators:
                try:
                    period = periods.get(indicator, 14)  # Default period

                    if indicator.upper() == "SMA":
                        # This would need historical data - simplified for example
                        results[indicator] = price  # Placeholder

                    elif indicator.upper() == "RSI":
                        # This would need historical data - simplified for example
                        results[indicator] = Decimal("50.0")  # Placeholder

                    elif indicator.upper() == "ATR":
                        # This would need historical data - simplified for example
                        results[indicator] = price * Decimal("0.02")  # 2% of price

                    elif indicator.upper() == "VOLATILITY":
                        # Calculate simple volatility estimate
                        results[indicator] = price * Decimal("0.15")  # 15% volatility estimate

                    else:
                        results[indicator] = None
                        self.logger.warning(
                            "Unknown indicator requested",
                            strategy=self.strategy_name,
                            indicator=indicator
                        )

                except Exception as e:
                    results[indicator] = None
                    self.logger.error(
                        "Failed to calculate indicator",
                        strategy=self.strategy_name,
                        indicator=indicator,
                        error=str(e)
                    )

            # Cache the results
            if self.cache_manager:
                try:
                    await self.cache_manager.set(cache_key, results, ttl=30)
                except Exception as e:
                    self.logger.warning(
                        "Failed to cache indicator results",
                        strategy=self.strategy_name,
                        error=str(e)
                    )

            # Record metrics
            if self._metrics_collector:
                successful_indicators = sum(1 for v in results.values() if v is not None)
                self._metrics_collector.record_gauge(
                    "strategy_indicators_calculated",
                    successful_indicators,
                    labels={"strategy": self.strategy_name, "symbol": data.symbol}
                )

            return results

        except Exception as e:
            # Record error metrics
            if self._metrics_collector:
                self._metrics_collector.increment_counter(
                    "strategy_indicator_errors",
                    labels={"strategy": self.strategy_name, "error_type": type(e).__name__}
                )

            # Fire alert for critical errors
            if self._alert_manager and isinstance(e, (ServiceError, StrategyError, ExecutionError)):
                alert = Alert(
                    name=f"strategy_indicator_failure_{self.strategy_name}",
                    severity=AlertSeverity.HIGH,
                    description=f"Strategy {self.strategy_name} failed to calculate indicators: {e!s}",
                    labels={"strategy": self.strategy_name, "operation": "calculate_indicators"}
                )
                await self._alert_manager.fire_alert(alert)

            raise

    def format_signal_metadata(self,
                             signal: Signal,
                             additional_data: dict[str, Any] = None) -> dict[str, Any]:
        """Format signal metadata with comprehensive formatting."""
        metadata = signal.metadata.copy() if signal.metadata else {}

        # Add formatted versions of numeric data
        formatted_metadata = {}

        for key, value in metadata.items():
            formatted_metadata[key] = value

            # Add formatted versions for common financial values
            if isinstance(value, (int, float, Decimal)):
                if "price" in key.lower():
                    formatted_metadata[f"{key}_formatted"] = format_currency(float(value))
                elif "percentage" in key.lower() or "ratio" in key.lower():
                    formatted_metadata[f"{key}_formatted"] = format_percentage(float(value))
                elif "score" in key.lower() or "confidence" in key.lower():
                    formatted_metadata[f"{key}_formatted"] = str(round_to_precision(Decimal(str(value)), 4))

        # Add timestamp formatting
        if signal.timestamp:
            formatted_metadata["timestamp_formatted"] = to_timestamp(signal.timestamp)
            formatted_metadata["timestamp_age_seconds"] = (
                datetime.now(timezone.utc) - signal.timestamp.replace(tzinfo=timezone.utc)
            ).total_seconds()

        # Add strategy information
        formatted_metadata.update({
            "strategy_name": self.strategy_name,
            "signal_id": getattr(signal, "id", None),
            "processing_timestamp": to_timestamp(datetime.now(timezone.utc))
        })

        # Add additional data if provided
        if additional_data:
            formatted_metadata.update(additional_data)

        return formatted_metadata

    async def record_signal_metrics(self,
                                  signal: Signal,
                                  signal_type: str = "generated",
                                  additional_labels: dict[str, str] = None):
        """Record comprehensive signal metrics."""
        if not self._metrics_collector:
            return

        labels = {
            "strategy": self.strategy_name,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "signal_type": signal_type
        }

        if additional_labels:
            labels.update(additional_labels)

        # Increment signal counter
        self._metrics_collector.increment_counter("strategy_signals", labels=labels)

        # Record signal strength
        self._metrics_collector.record_histogram(
            "strategy_signal_strength",
            signal.strength if isinstance(signal.strength, Decimal) else Decimal(str(signal.strength)),
            labels=labels
        )

        # Update internal metrics
        self._signal_metrics["total_generated"] += 1
        if signal.strength >= Decimal("0.6"):  # Consider high confidence signals as valid
            self._signal_metrics["total_valid"] += 1

        # Calculate and record success rate
        if self._signal_metrics["total_generated"] > 0:
            self._signal_metrics["success_rate"] = (
                self._signal_metrics["total_valid"] / self._signal_metrics["total_generated"]
            )

            self._metrics_collector.record_gauge(
                "strategy_success_rate",
                self._signal_metrics["success_rate"],
                labels={"strategy": self.strategy_name}
            )

    # @handle_exceptions(ErrorSeverity.HIGH)  # Decorator not yet implemented
    async def safe_execute_with_monitoring(self,
                                         operation_name: str,
                                         operation_func,
                                         *args,
                                         **kwargs) -> Any:
        """Safely execute operations with comprehensive monitoring."""

        # Start tracing span
        with self.tracer.start_as_current_span(
            f"strategy.{self.strategy_name}.{operation_name}"
        ) as span:
            span.set_attribute("strategy.name", self.strategy_name)
            span.set_attribute("operation.name", operation_name)

            start_time = datetime.now(timezone.utc)

            try:
                # Execute operation with performance monitoring
                if self._performance_monitor:
                    with self._performance_monitor.monitor_operation(operation_name):
                        result = await operation_func(*args, **kwargs)
                else:
                    result = await operation_func(*args, **kwargs)

                # Record successful execution
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                if self._metrics_collector:
                    self._metrics_collector.record_histogram(
                        "strategy_operation_duration",
                        execution_time,
                        labels={"strategy": self.strategy_name, "operation": operation_name}
                    )

                    self._metrics_collector.increment_counter(
                        "strategy_operation_success",
                        labels={"strategy": self.strategy_name, "operation": operation_name}
                    )

                span.set_attribute("operation.status", "success")
                span.set_attribute("operation.duration_seconds", execution_time)

                return result

            except Exception as e:
                # Record error metrics
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

                if self._metrics_collector:
                    self._metrics_collector.increment_counter(
                        "strategy_operation_errors",
                        labels={
                            "strategy": self.strategy_name,
                            "operation": operation_name,
                            "error_type": type(e).__name__
                        }
                    )

                # Record in span
                span.record_exception(e)
                span.set_attribute("operation.status", "error")
                span.set_attribute("operation.duration_seconds", execution_time)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))

                # Send telemetry if available
                if self._telemetry_collector:
                    await self._telemetry_collector.record_error(
                        error=e,
                        context={"strategy": self.strategy_name, "operation": operation_name},
                        severity=ErrorSeverity.HIGH
                    )

                raise

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive strategy status with all integrations."""
        return {
            "strategy_name": self.strategy_name,
            "timestamp": to_timestamp(datetime.now(timezone.utc)),
            "integrations": self._get_available_integrations(),
            "signal_metrics": self._signal_metrics,
            "performance_metrics": self.performance_tracker.get_metrics() if hasattr(self.performance_tracker, "get_metrics") else {},
            "cache_stats": self.cache_manager.get_stats() if hasattr(self.cache_manager, "get_stats") else {},
            "error_stats": {
                "circuit_breaker_open": False,  # Would check actual circuit breaker state
                "total_errors": 0,  # Would get from error handler
                "last_error": None
            }
        }

    async def cleanup_resources(self):
        """Cleanup all resources and connections."""
        try:
            # Clear cache entries for this strategy
            if self.cache_manager and hasattr(self.cache_manager, "clear_namespace"):
                await self.cache_manager.clear_namespace(f"strategy_{self.strategy_name}")

            # Reset metrics
            self._signal_metrics = {
                "total_generated": 0,
                "total_valid": 0,
                "total_executed": 0,
                "success_rate": 0.0,
                "average_confidence": 0.0
            }

            # Cleanup performance tracker
            if hasattr(self.performance_tracker, "cleanup"):
                self.performance_tracker.cleanup()

            self.logger.info("Strategy resources cleaned up", strategy=self.strategy_name)

        except Exception as e:
            self.logger.error(
                "Error during resource cleanup",
                strategy=self.strategy_name,
                error=str(e)
            )


class StrategyDataAccessMixin:
    """Mixin providing data access patterns for strategies."""

    def __init__(self, data_service: "DataServiceInterface | None" = None):
        self.data_service = data_service
        self.logger = get_logger(f"{self.__class__.__name__}")

    async def get_indicator_data(self, symbol: str, indicator: str, period: int) -> "Decimal | None":
        """Generic method to get indicator data through data service."""
        if not self.data_service:
            self.logger.warning(f"No data service available for {indicator}")
            return None

        try:
            # Map indicator names to data service methods
            indicator_methods = {
                "SMA": "get_sma",
                "EMA": "get_ema",
                "RSI": "get_rsi",
                "ATR": "get_atr",
                "VOLATILITY": "get_volatility",
                "VOLUME_RATIO": "get_volume_ratio"
            }

            method_name = indicator_methods.get(indicator.upper())
            if not method_name:
                self.logger.warning(f"Unknown indicator: {indicator}")
                return None

            method = getattr(self.data_service, method_name, None)
            if not method:
                self.logger.warning(f"Data service does not support {indicator}")
                return None

            result = await method(symbol, period)
            return Decimal(str(result)) if result is not None else None

        except Exception as e:
            self.logger.error(f"Failed to get {indicator} data: {e}")
            return None


# Convenience functions for common patterns
def create_comprehensive_signal(direction: SignalDirection,
                              strength: Decimal,
                              symbol: str,
                              source: str,
                              metadata: dict[str, Any],
                              timestamp: "datetime | None" = None) -> Signal:
    """Create a signal with comprehensive metadata formatting."""

    # Ensure timezone on timestamp
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    else:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

    # Enhance metadata with formatted values
    enhanced_metadata = metadata.copy()

    for key, value in metadata.items():
        if isinstance(value, (int, float, Decimal)):
            decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
            if "price" in key.lower():
                enhanced_metadata[f"{key}_formatted"] = format_currency(decimal_value)
            elif any(term in key.lower() for term in ["pct", "percent", "ratio"]):
                enhanced_metadata[f"{key}_formatted"] = format_percentage(decimal_value)

    enhanced_metadata.update({
        "created_at": to_timestamp(timestamp),
        "strength_formatted": format_percentage(strength if isinstance(strength, Decimal) else Decimal(str(strength)))
    })

    return Signal(
        direction=direction,
        strength=strength,
        timestamp=timestamp,
        symbol=symbol,
        source=source,
        metadata=enhanced_metadata
    )


async def calculate_position_size_comprehensive(signal: Signal,
                                              account_balance: Decimal,
                                              risk_parameters: dict[str, Any],
                                              strategy_name: str) -> Decimal:
    """Calculate position size using comprehensive risk management."""

    logger = get_logger(f"position_sizing.{strategy_name}")

    try:
        # Base position size from risk parameters
        base_risk_pct = Decimal(str(risk_parameters.get("base_risk_percentage", 0.02)))  # 2% default

        # Adjust based on signal confidence
        confidence_multiplier = signal.strength

        # Apply Kelly Criterion if configured
        if risk_parameters.get("use_kelly_criterion", False):
            win_rate = Decimal(str(risk_parameters.get("historical_win_rate", 0.6)))
            avg_win = Decimal(str(risk_parameters.get("average_win", 1.5)))
            avg_loss = Decimal(str(risk_parameters.get("average_loss", 1.0)))

            # Simple Kelly criterion: f = (bp - q) / b, where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b_ratio = avg_win / avg_loss if avg_loss > 0 else Decimal("1.0")
            kelly_fraction = (b_ratio * win_rate - (Decimal("1.0") - win_rate)) / b_ratio
            base_risk_pct = min(base_risk_pct, kelly_fraction)  # Don't exceed Kelly recommendation

        # Calculate position size
        risk_amount = account_balance * base_risk_pct
        adjusted_risk = risk_amount * confidence_multiplier

        # Apply maximum position limits
        max_position_pct = Decimal(str(risk_parameters.get("max_position_percentage", 0.1)))  # 10% max
        max_position_amount = account_balance * max_position_pct

        final_position_size = min(adjusted_risk, max_position_amount)

        logger.info(
            "Position size calculated",
            strategy=strategy_name,
            symbol=signal.symbol,
            base_risk_pct=format_percentage(base_risk_pct if isinstance(base_risk_pct, Decimal) else Decimal(str(base_risk_pct))),
            confidence_multiplier=format_percentage(confidence_multiplier if isinstance(confidence_multiplier, Decimal) else Decimal(str(confidence_multiplier))),
            final_size=format_currency(final_position_size if isinstance(final_position_size, Decimal) else Decimal(str(final_position_size))),
            account_balance=format_currency(account_balance if isinstance(account_balance, Decimal) else Decimal(str(account_balance)))
        )

        return final_position_size

    except Exception as e:
        logger.error(f"Position size calculation failed: {e}")
        # Return conservative fallback
        return account_balance * Decimal("0.01")  # 1% fallback


__all__ = [
    "StrategyDataAccessMixin",
    "StrategyIntegratedBase",
    "calculate_position_size_comprehensive",
    "create_comprehensive_signal"
]
