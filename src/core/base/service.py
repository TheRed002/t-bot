"""
Base service implementation for the service layer pattern.

This module provides the foundation for all business logic services in the
trading bot system, implementing comprehensive service patterns with
dependency injection, health checks, and monitoring.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, TypeVar

from src.core.base.component import BaseComponent
from src.core.base.interfaces import HealthStatus, ServiceComponent
from src.core.exceptions import DependencyError, ServiceError, ValidationError
from src.core.types.base import ConfigDict

# Type variable for service operations
T = TypeVar("T")


class BaseService(BaseComponent, ServiceComponent):
    """
    Base service class implementing the service layer pattern.

    Provides:
    - Business logic organization
    - Transaction management
    - Service-to-service communication
    - Dependency injection integration
    - Performance monitoring
    - Circuit breaker patterns
    - Retry mechanisms

    Example:
        ```python
        class TradingService(BaseService):
            def __init__(self):
                super().__init__(name="TradingService")
                self.add_dependency("OrderRepository")
                self.add_dependency("RiskManager")

            async def _do_start(self):
                self.order_repo = self.resolve_dependency("OrderRepository")
                self.risk_manager = self.resolve_dependency("RiskManager")

            async def place_order(self, order_data: dict) -> dict:
                return await self.execute_with_monitoring(
                    "place_order", self._place_order_impl, order_data
                )

            async def _place_order_impl(self, order_data: dict) -> dict:
                # Business logic implementation
                pass
        ```
    """

    def __init__(
        self,
        name: str | None = None,
        config: ConfigDict | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize base service.

        Args:
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name, config, correlation_id)

        # Service-specific metrics
        self._service_metrics: dict[str, Any] = {
            "operations_count": 0,
            "operations_success": 0,
            "operations_error": 0,
            "average_response_time": 0.0,
            "last_operation_time": None,
            "circuit_breaker_trips": 0,
        }

        # Circuit breaker configuration
        self._circuit_breaker_enabled = True
        self._circuit_breaker_threshold = 5  # failures
        self._circuit_breaker_timeout = 60  # seconds
        self._circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: datetime | None = None

        # Retry configuration
        self._retry_enabled = True
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds
        self._retry_backoff = 2.0  # multiplier

        # Operation tracking
        self._active_operations: dict[str, datetime] = {}
        self._operation_history: list[dict[str, Any]] = []
        self._max_history_size = 1000

        self._logger.debug("Service initialized", service=self._name)

    @property
    def service_metrics(self) -> dict[str, Any]:
        """Get service-specific metrics."""
        metrics = self._service_metrics.copy()
        metrics.update(
            {
                "active_operations": len(self._active_operations),
                "circuit_breaker_state": self._circuit_breaker_state,
                "circuit_breaker_failures": self._circuit_breaker_failures,
            }
        )
        return metrics

    # Service Operation Execution
    async def execute_with_monitoring(
        self, operation_name: str, operation_func: Any, *args, **kwargs
    ) -> Any:
        """
        Execute service operation with monitoring and consistent data validation.

        Args:
            operation_name: Name of the operation for tracking
            operation_func: Function to execute
            *args: Operation arguments
            **kwargs: Operation keyword arguments

        Returns:
            Operation result (with standardized format)

        Raises:
            ServiceError: If operation fails after retries
        """
        # Validate input data format consistency
        for arg in args:
            if isinstance(arg, dict) and "data_format" in arg:
                # Validate data format version compatibility
                data_format = arg.get("data_format")
                if data_format and not data_format.endswith("_v1"):
                    self._logger.warning(
                        "Data format version mismatch detected",
                        service=self._name,
                        operation=operation_name,
                        expected_format_version="v1",
                        actual_format=data_format,
                    )

        # Check circuit breaker
        if not self._check_circuit_breaker():
            raise ServiceError(
                f"Circuit breaker is OPEN for service {self._name}",
                details={"operation": operation_name},
            )

        operation_id = f"{operation_name}_{len(self._operation_history)}"
        start_time = datetime.now(timezone.utc)
        self._active_operations[operation_id] = start_time

        try:
            self._logger.debug(
                "Starting service operation",
                service=self._name,
                operation=operation_name,
                operation_id=operation_id,
            )

            # Execute with retry logic
            result = await self._execute_with_retry(operation_func, operation_name, *args, **kwargs)

            # Record successful operation
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_operation_success(operation_name, execution_time)

            self._logger.debug(
                "Service operation completed successfully",
                service=self._name,
                operation=operation_name,
                operation_id=operation_id,
                execution_time_seconds=execution_time,
            )

            return result

        except Exception as e:
            # Record failed operation (including ValidationErrors for circuit breaker)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._record_operation_failure(operation_name, execution_time, e)

            # ValidationErrors should not be wrapped - re-raise immediately after recording
            if isinstance(e, ValidationError):
                raise

            self._logger.error(
                "Service operation failed",
                service=self._name,
                operation=operation_name,
                operation_id=operation_id,
                execution_time_seconds=execution_time,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Apply consistent error propagation pattern
            self._propagate_service_error_consistently(e, operation_name, execution_time)

            raise ServiceError(
                f"Operation {operation_name} failed in service {self._name}: {e}",
                details={
                    "operation": operation_name,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__,
                    "processing_mode": "sync",
                    "data_format": "service_error_v1",
                },
            ) from e

        finally:
            self._active_operations.pop(operation_id, None)

    async def _execute_with_retry(
        self, operation_func: Any, operation_name: str, *args, **kwargs
    ) -> Any:
        """
        Execute operation with retry logic.

        Args:
            operation_func: Function to execute
            operation_name: Operation name for logging
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ServiceError: If all retry attempts fail
        """
        if not self._retry_enabled:
            return await operation_func(*args, **kwargs)

        last_exception = None
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):  # +1 for initial attempt
            try:
                return await operation_func(*args, **kwargs)

            except Exception as e:
                # ValidationErrors should not be retried - re-raise immediately
                if isinstance(e, ValidationError):
                    raise

                last_exception = e

                if attempt < self._max_retries:
                    self._logger.warning(
                        "Service operation failed, retrying",
                        service=self._name,
                        operation=operation_name,
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                        delay_seconds=delay,
                        error=str(e),
                    )

                    await asyncio.sleep(delay)
                    delay *= self._retry_backoff
                else:
                    # Final failure
                    break

        # All retries exhausted
        raise ServiceError(
            f"Operation {operation_name} failed after {self._max_retries} retries",
            details={
                "operation": operation_name,
                "attempts": self._max_retries + 1,
                "last_error": str(last_exception),
            },
        ) from last_exception

    # Circuit Breaker Implementation
    def _check_circuit_breaker(self) -> bool:
        """
        Check circuit breaker state.

        Returns:
            bool: True if operation is allowed, False if blocked
        """
        if not self._circuit_breaker_enabled:
            return True

        now = datetime.now(timezone.utc)

        # Check if circuit breaker should transition from OPEN to HALF_OPEN
        if (
            self._circuit_breaker_state == "OPEN"
            and self._circuit_breaker_last_failure
            and (now - self._circuit_breaker_last_failure).total_seconds()
            >= self._circuit_breaker_timeout
        ):
            self._circuit_breaker_state = "HALF_OPEN"
            self._logger.info(
                "Circuit breaker transitioning to HALF_OPEN",
                service=self._name,
            )

        return self._circuit_breaker_state != "OPEN"

    def _record_operation_success(self, operation_name: str, execution_time: float) -> None:
        """Record successful operation for metrics and circuit breaker."""
        self._service_metrics["operations_count"] += 1
        self._service_metrics["operations_success"] += 1
        self._service_metrics["last_operation_time"] = datetime.now(timezone.utc)

        # Update average response time
        current_avg = self._service_metrics["average_response_time"]
        total_ops = self._service_metrics["operations_count"]
        self._service_metrics["average_response_time"] = (
            current_avg * (total_ops - 1) + execution_time
        ) / total_ops

        # Reset circuit breaker on success
        if self._circuit_breaker_state == "HALF_OPEN":
            self._circuit_breaker_state = "CLOSED"
            self._circuit_breaker_failures = 0
            self._logger.info(
                "Circuit breaker reset to CLOSED",
                service=self._name,
            )

        # Add to operation history
        self._add_to_history(
            {
                "operation": operation_name,
                "status": "success",
                "execution_time": execution_time,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def _record_operation_failure(
        self, operation_name: str, execution_time: float, error: Exception
    ) -> None:
        """Record failed operation for metrics and circuit breaker."""
        self._service_metrics["operations_count"] += 1
        self._service_metrics["operations_error"] += 1
        self._service_metrics["last_operation_time"] = datetime.now(timezone.utc)

        # Update circuit breaker
        if self._circuit_breaker_enabled:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = datetime.now(timezone.utc)

            if (
                self._circuit_breaker_failures >= self._circuit_breaker_threshold
                and self._circuit_breaker_state == "CLOSED"
            ):
                self._circuit_breaker_state = "OPEN"
                self._service_metrics["circuit_breaker_trips"] += 1
                self._logger.warning(
                    "Circuit breaker tripped to OPEN",
                    service=self._name,
                    failures=self._circuit_breaker_failures,
                    threshold=self._circuit_breaker_threshold,
                )

        # Add to operation history
        self._add_to_history(
            {
                "operation": operation_name,
                "status": "failure",
                "execution_time": execution_time,
                "error": str(error),
                "error_type": type(error).__name__,
                "timestamp": datetime.now(timezone.utc),
            }
        )

    def _add_to_history(self, record: dict[str, Any]) -> None:
        """Add operation record to history."""
        self._operation_history.append(record)

        # Maintain history size limit
        if len(self._operation_history) > self._max_history_size:
            self._operation_history.pop(0)

    # Dependency Resolution
    def configure_dependencies(self, dependency_injector: Any) -> None:
        """
        Configure dependencies from the DI container.

        Args:
            dependency_injector: Dependency injector instance

        This method is called by the DI system to configure service dependencies.
        Override in subclasses to resolve specific dependencies.
        """
        self._dependency_container = dependency_injector
        self._logger.debug(
            "Dependency container configured",
            service=self._name,
        )

    def resolve_dependency(self, dependency_name: str) -> Any:
        """
        Resolve a dependency from the DI container.

        Args:
            dependency_name: Name of the dependency to resolve

        Returns:
            Resolved dependency instance

        Raises:
            DependencyError: If dependency cannot be resolved
        """
        if not self._dependency_container:
            raise DependencyError(
                f"No DI container configured for service {self._name}",
                dependency_name=dependency_name,
                error_code="DEP_008",
                suggested_action="Configure dependency container before resolving dependencies",
                context={"service": self._name},
            )

        try:
            dependency = self._dependency_container.resolve(dependency_name)
            self._logger.debug(
                "Dependency resolved",
                service=self._name,
                dependency=dependency_name,
            )
            return dependency

        except Exception as e:
            raise DependencyError(
                f"Failed to resolve dependency '{dependency_name}' for service {self._name}",
                dependency_name=dependency_name,
                error_code="DEP_009",
                suggested_action="Ensure dependency is properly registered in DI container",
                context={"service": self._name, "original_error": str(e)},
            ) from e

    # Health Checks
    async def _health_check_internal(self) -> HealthStatus:
        """Service-specific health check implementation."""
        try:
            # Check active operations
            now = datetime.now(timezone.utc)
            stuck_operations = [
                op_id
                for op_id, start_time in self._active_operations.items()
                if (now - start_time).total_seconds() > 300  # 5 minutes
            ]

            if stuck_operations:
                self._logger.warning(
                    "Detected stuck operations",
                    service=self._name,
                    stuck_operations=stuck_operations,
                )
                return HealthStatus.DEGRADED

            # Check circuit breaker state
            if self._circuit_breaker_state == "OPEN":
                return HealthStatus.DEGRADED

            # Check error rate
            if self._service_metrics["operations_count"] > 0:
                error_rate = (
                    self._service_metrics["operations_error"]
                    / self._service_metrics["operations_count"]
                )

                if error_rate > 0.5:  # More than 50% errors
                    return HealthStatus.DEGRADED
                elif error_rate > 0.8:  # More than 80% errors
                    return HealthStatus.UNHEALTHY

            # Service-specific health check with consistent error handling
            try:
                return await self._service_health_check()
            except Exception as e:
                self._logger.error(
                    "Service health check failed",
                    service=self._name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Apply consistent error propagation for health checks
                self._propagate_service_error_consistently(e, "health_check", 0.0)
                return HealthStatus.UNHEALTHY

        except Exception as e:
            self._logger.error(
                "Health check internal error",
                service=self._name,
                error=str(e),
            )
            return HealthStatus.UNHEALTHY

    async def _service_health_check(self) -> HealthStatus:
        """Override in subclasses for service-specific health checks."""
        return HealthStatus.HEALTHY

    # Configuration
    def validate_config(self, config: ConfigDict) -> bool:
        """
        Validate service configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: True if configuration is valid
        """
        if not super().validate_config(config):
            return False

        # Service-specific validation
        return self._validate_service_config(config)

    def _validate_service_config(self, config: ConfigDict) -> bool:
        """Override in subclasses for service-specific configuration validation."""
        return True

    # Metrics
    def get_metrics(self) -> dict[str, Any]:
        """Get combined component and service metrics."""
        metrics = super().get_metrics()
        metrics.update(self.service_metrics)
        return metrics

    def reset_metrics(self) -> None:
        """Reset both component and service metrics."""
        super().reset_metrics()
        self._service_metrics = {
            "operations_count": 0,
            "operations_success": 0,
            "operations_error": 0,
            "average_response_time": 0.0,
            "last_operation_time": None,
            "circuit_breaker_trips": 0,
        }
        self._circuit_breaker_failures = 0
        self._operation_history.clear()

    # Service Management
    def get_operation_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get operation history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of operation records
        """
        history = self._operation_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def configure_circuit_breaker(
        self,
        enabled: bool = True,
        threshold: int = 5,
        timeout: int = 60,
    ) -> None:
        """
        Configure circuit breaker settings.

        Args:
            enabled: Enable/disable circuit breaker
            threshold: Number of failures before opening
            timeout: Timeout in seconds before half-open
        """
        self._circuit_breaker_enabled = enabled
        self._circuit_breaker_threshold = threshold
        self._circuit_breaker_timeout = timeout

        self._logger.info(
            "Circuit breaker configured",
            service=self._name,
            enabled=enabled,
            threshold=threshold,
            timeout=timeout,
        )

    def configure_retry(
        self,
        enabled: bool = True,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
    ) -> None:
        """
        Configure retry settings.

        Args:
            enabled: Enable/disable retries
            max_retries: Maximum number of retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier for delays
        """
        self._retry_enabled = enabled
        self._max_retries = max_retries
        self._retry_delay = delay
        self._retry_backoff = backoff

        self._logger.info(
            "Retry mechanism configured",
            service=self._name,
            enabled=enabled,
            max_retries=max_retries,
            delay=delay,
            backoff=backoff,
        )

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        self._circuit_breaker_state = "CLOSED"
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None

        self._logger.info(
            "Circuit breaker manually reset",
            service=self._name,
        )

    def _propagate_service_error_consistently(
        self, error: Exception, operation: str, execution_time: float
    ) -> None:
        """Propagate service errors with consistent patterns aligned with execution module."""
        # Apply consistent error data transformation like execution module
        from src.core.exceptions import DataValidationError, RepositoryError, ServiceError, ValidationError
        
        # Use consistent data transformation for error propagation
        try:
            from src.core.data_transformer import CoreDataTransformer
            
            error_data = CoreDataTransformer.transform_for_pub_sub_pattern(
                event_type="service_error",
                data={
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "service": self._name,
                    "operation": operation,
                    "execution_time": execution_time,
                },
                metadata={
                    "severity": "high" if isinstance(error, ServiceError) else "medium",
                    "component": "BaseService",
                    "boundary_crossed": True,
                }
            )
        except Exception:
            # Fallback to basic error data if transformation fails
            error_data = {
                "error": str(error),
                "service": self._name,
                "operation": operation,
                "execution_time": execution_time,
            }

        if isinstance(error, (ValidationError, DataValidationError)):
            # Validation errors are re-raised as-is for consistency
            self._logger.debug(
                f"Validation error in {self._name}.{operation} - propagating as validation error",
                extra=error_data
            )
            # Re-raise validation errors with consistent propagation patterns
            raise error
        elif isinstance(error, RepositoryError):
            # Repository errors are propagated with service context
            self._logger.warning(
                f"Repository error in {self._name}.{operation} - adding service context",
                extra=error_data
            )
            # Wrap repository errors in ServiceError for consistency
            raise ServiceError(
                f"Repository error in {self._name}.{operation}: {error}",
                details={"original_error": str(error), "service": self._name, "operation": operation}
            ) from error
        else:
            # Generic errors get service-level error propagation consistent with execution module
            self._logger.error(
                f"Service error in {self._name}.{operation} - wrapping in ServiceError",
                extra=error_data
            )
            # Always wrap in ServiceError for consistent error handling across modules
            raise ServiceError(
                f"Service error in {self._name}.{operation}: {error}",
                details={"original_error": str(error), "service": self._name, "operation": operation}
            ) from error


class TransactionalService(BaseService):
    """
    Base service with transaction management support.

    Extends BaseService with database transaction capabilities,
    ensuring data consistency across service operations.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize transactional service."""
        super().__init__(*args, **kwargs)
        self._transaction_manager: Any | None = None

    def set_transaction_manager(self, transaction_manager: Any) -> None:
        """Set transaction manager for this service."""
        self._transaction_manager = transaction_manager
        self._logger.debug(
            "Transaction manager configured",
            service=self._name,
        )

    async def execute_in_transaction(
        self, operation_name: str, operation_func: Any, *args, **kwargs
    ) -> Any:
        """
        Execute operation within a database transaction.

        Args:
            operation_name: Name of the operation
            operation_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Operation result

        Raises:
            ServiceError: If operation or transaction fails
        """
        if not self._transaction_manager:
            # Fall back to regular execution if no transaction manager
            return await self.execute_with_monitoring(
                operation_name, operation_func, *args, **kwargs
            )

        try:
            async with self._transaction_manager.transaction():
                return await self.execute_with_monitoring(
                    operation_name, operation_func, *args, **kwargs
                )

        except Exception as e:
            self._logger.error(
                "Transaction failed",
                service=self._name,
                operation=operation_name,
                error=str(e),
            )
            raise
