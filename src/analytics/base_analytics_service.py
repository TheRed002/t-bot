"""
Base Analytics Service - Common patterns for all analytics services.

This module provides the base implementation for analytics services to eliminate
code duplication and ensure consistency across all analytics components.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TypeVar

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.utils.decimal_utils import to_decimal

T = TypeVar("T")


class BaseAnalyticsService(BaseService, ABC):
    """
    Base class for all analytics services providing common functionality.

    This eliminates code duplication and ensures consistent patterns across:
    - Real-time analytics
    - Portfolio analytics
    - Risk monitoring
    - Operational analytics
    - Reporting services
    """

    def __init__(
        self,
        name: str | None = None,
        config: dict | None = None,
        correlation_id: str | None = None,
        metrics_collector: Any | None = None,
    ):
        """
        Initialize base analytics service.

        Args:
            name: Service name
            config: Configuration dictionary
            correlation_id: Request correlation ID
            metrics_collector: Metrics collection instance
        """
        super().__init__(name or self.__class__.__name__, config, correlation_id)

        # Initialize metrics collector - must be injected
        self.metrics_collector = metrics_collector

        # Common analytics state
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = config.get("cache_ttl", 300) if config else 300
        self._last_update = datetime.now(timezone.utc)
        self._update_frequency = config.get("update_frequency", 60) if config else 60

        # Performance tracking
        self._calculation_times: dict[str, list[float]] = {}
        self._error_counts: dict[str, dict[str, int]] = {}

    # Common validation methods
    def validate_time_range(self, start_time: datetime, end_time: datetime) -> None:
        """
        Validate time range parameters.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Raises:
            ValidationError: If time range is invalid
        """
        if not start_time or not end_time:
            raise ValidationError(
                "Time range parameters required",
                context={"start_time": start_time, "end_time": end_time},
            )

        if start_time >= end_time:
            raise ValidationError(
                "Invalid time range: start_time must be before end_time",
                context={"start_time": start_time, "end_time": end_time},
            )

        # Check for reasonable time ranges (configurable max range)
        max_range_days = self.config.get("max_time_range_days", 365) if self.config else 365
        max_range = max_range_days * 24 * 3600  # seconds
        if (end_time - start_time).total_seconds() > max_range:
            raise ValidationError(
                f"Time range too large (max {max_range_days} days)",
                context={
                    "start_time": start_time,
                    "end_time": end_time,
                    "max_days": max_range_days,
                },
            )

    def validate_decimal_value(
        self,
        value: Any,
        field_name: str,
        min_value: Decimal | None = None,
        max_value: Decimal | None = None,
    ) -> Decimal:
        """
        Validate and convert value to Decimal for financial calculations.

        Args:
            value: Value to validate
            field_name: Name of field for error messages
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Validated Decimal value

        Raises:
            ValidationError: If value is invalid
        """
        # Use core utility for decimal conversion with proper error handling
        try:
            decimal_value = to_decimal(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                f"Invalid decimal value for {field_name}",
                field_name=field_name,
                field_value=value,
            ) from e

        # Validate range constraints
        if min_value is not None and decimal_value < min_value:
            raise ValidationError(
                f"{field_name} below minimum value",
                field_name=field_name,
                field_value=decimal_value,
                context={"min_value": min_value},
            )

        if max_value is not None and decimal_value > max_value:
            raise ValidationError(
                f"{field_name} above maximum value",
                field_name=field_name,
                field_value=decimal_value,
                context={"max_value": max_value},
            )

        return decimal_value

    # Common caching methods
    def get_from_cache(self, key: str) -> Any | None:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key in self._cache:
            entry = self._cache[key]
            if (datetime.now(timezone.utc) - entry["timestamp"]).total_seconds() < self._cache_ttl:
                return entry["value"]
            else:
                del self._cache[key]
        return None

    def set_cache(self, key: str, value: Any) -> None:
        """
        Store value in cache with timestamp.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = {"value": value, "timestamp": datetime.now(timezone.utc)}

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        self._last_update = datetime.now(timezone.utc)

    # Common metrics recording
    def record_calculation_time(self, operation: str, duration: float) -> None:
        """
        Record calculation time for performance monitoring.

        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        if operation not in self._calculation_times:
            self._calculation_times[operation] = []

        self._calculation_times[operation].append(duration)

        # Keep only last 100 entries
        if len(self._calculation_times[operation]) > 100:
            self._calculation_times[operation] = self._calculation_times[operation][-100:]

        # Record to metrics collector
        if self.metrics_collector:
            self.metrics_collector.observe_histogram(
                f"analytics_{self._name}_{operation}_duration_seconds",
                duration,
                {"service": self._name, "operation": operation},
            )

    def record_error(self, operation: str, error: Exception) -> None:
        """
        Record error occurrence for monitoring.

        Args:
            operation: Operation name
            error: Exception that occurred
        """
        error_type = type(error).__name__

        if operation not in self._error_counts:
            self._error_counts[operation] = {}

        if error_type not in self._error_counts[operation]:
            self._error_counts[operation][error_type] = 0

        self._error_counts[operation][error_type] += 1

        # Record to metrics collector
        if self.metrics_collector:
            self.metrics_collector.increment_counter(
                f"analytics_{self._name}_errors_total",
                {"service": self._name, "operation": operation, "error_type": error_type},
            )

    # Common data conversion
    def convert_for_export(self, obj: Any) -> Any:
        """
        Convert objects for export/serialization.

        Args:
            obj: Object to convert

        Returns:
            Converted object suitable for export
        """
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif hasattr(obj, "dict"):
            return obj.dict()
        elif isinstance(obj, Decimal):
            # Convert to string to preserve precision for financial data
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, list):
            return [self.convert_for_export(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_for_export(v) for k, v in obj.items()}
        else:
            return obj

    # Common monitoring wrapper
    async def execute_monitored(
        self, operation_name: str, operation_func: Callable, *args, **kwargs
    ) -> Any:
        """
        Execute operation with monitoring and error handling.

        Args:
            operation_name: Name of operation for tracking
            operation_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Operation result

        Raises:
            ServiceError: If operation fails
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Check cache first if applicable
            cache_key = f"{operation_name}_{args!s}_{kwargs!s}"
            cached_result = self.get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute operation
            result = await operation_func(*args, **kwargs)

            # Cache result
            self.set_cache(cache_key, result)

            # Record success metrics
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.record_calculation_time(operation_name, duration)

            return result

        except Exception as e:
            # Record error metrics
            self.record_error(operation_name, e)

            # Log error with context
            self.logger.error(
                f"Error in {operation_name}",
                extra={
                    "service": self._name,
                    "operation": operation_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            raise ServiceError(
                f"{operation_name} failed in {self._name}",
                context={"operation": operation_name, "error": str(e), "service": self._name},
            ) from e

    # Abstract methods for subclasses
    @abstractmethod
    async def calculate_metrics(self, *args, **kwargs) -> dict[str, Any]:
        """
        Calculate service-specific metrics.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def validate_data(self, data: Any) -> bool:
        """
        Validate service-specific data.

        Must be implemented by subclasses.
        """
        pass

    # Health check implementation
    async def _service_health_check(self) -> Any:
        """
        Service-specific health check.

        Returns:
            Health status
        """
        from src.core.base.interfaces import HealthStatus

        try:
            # Check cache health (configurable thresholds)
            max_cache_items = self.config.get("max_cache_items", 10000) if self.config else 10000
            if len(self._cache) > max_cache_items:
                return HealthStatus.DEGRADED

            # Check error rates (configurable thresholds)
            total_errors = sum(sum(counts.values()) for counts in self._error_counts.values())
            error_threshold_degraded = (
                self.config.get("error_threshold_degraded", 100) if self.config else 100
            )
            error_threshold_unhealthy = (
                self.config.get("error_threshold_unhealthy", 500) if self.config else 500
            )

            if total_errors > error_threshold_degraded:
                return HealthStatus.DEGRADED
            elif total_errors > error_threshold_unhealthy:
                return HealthStatus.UNHEALTHY

            # Check last update time
            time_since_update = (datetime.now(timezone.utc) - self._last_update).total_seconds()

            if time_since_update > self._update_frequency * 10:
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return HealthStatus.UNHEALTHY

    # Cleanup
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.clear_cache()
        self._calculation_times.clear()
        self._error_counts.clear()
        await super().stop()
