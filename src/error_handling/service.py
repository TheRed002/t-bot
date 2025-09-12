"""
Error Handling Service - Service layer for error handling operations.

This service layer implements comprehensive error handling business logic, providing
a clean interface for error processing, pattern analysis, state monitoring, and recovery
operations. It orchestrates the underlying error handling components while maintaining
proper separation of concerns.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from src.error_handling.error_handler import ErrorHandler
    from src.error_handling.global_handler import GlobalErrorHandler
    from src.error_handling.pattern_analytics import ErrorPatternAnalytics

from src.core.base.interfaces import HealthCheckResult
from src.core.base.service import BaseService
from src.core.config import Config
from src.core.exceptions import DataValidationError, ServiceError, ValidationError
from src.error_handling.data_transformer import ErrorDataTransformer
from src.utils.messaging_patterns import (
    BoundaryValidator,
    ErrorPropagationMixin,
)


class StateMonitorInterface(Protocol):
    """Protocol for state monitoring service."""

    async def validate_state_consistency(self, component: str = "all") -> dict[str, Any]: ...
    async def reconcile_state(
        self, component: str, discrepancies: list[dict[str, Any]]
    ) -> bool: ...
    async def start_monitoring(self) -> None: ...
    def get_state_summary(self) -> dict[str, Any]: ...


class ErrorHandlingService(BaseService, ErrorPropagationMixin):
    """
    Service layer for error handling operations.

    This service orchestrates error handling components and provides a unified
    interface for error processing, pattern analysis, and state monitoring.
    """

    def __init__(
        self,
        config: Config,
        error_handler: Optional["ErrorHandler"] = None,
        global_handler: Optional["GlobalErrorHandler"] = None,
        pattern_analytics: Optional["ErrorPatternAnalytics"] = None,
        state_monitor: StateMonitorInterface | None = None,
    ) -> None:
        # Configuration constants
        from src.error_handling.error_handler import DEFAULT_BACKGROUND_TASK_CLEANUP_TIMEOUT

        self._background_task_timeout = DEFAULT_BACKGROUND_TASK_CLEANUP_TIMEOUT

        # Convert Config to ConfigDict properly for BaseService
        from src.core.types.base import ConfigDict

        if hasattr(config, "model_dump"):
            config_dict = ConfigDict(config.model_dump())
        elif isinstance(config, dict):
            config_dict = ConfigDict(config)
        else:
            config_dict = ConfigDict({})

        super().__init__(name="ErrorHandlingService", config=config_dict)

        # Store the original config for component initialization
        self._raw_config = config

        # Store injected dependencies - these should come from DI container
        self._error_handler = error_handler
        self._global_handler = global_handler
        self._pattern_analytics = pattern_analytics
        self._state_monitor = state_monitor

        # Service state
        self._initialized = False
        DEFAULT_MAX_CONCURRENT_TASKS = 10  # Prevent overwhelming the system
        self._max_concurrent_tasks = DEFAULT_MAX_CONCURRENT_TASKS
        self._semaphore = None  # Initialize in async context

    async def initialize(self) -> None:
        """Initialize the error handling service and its components."""
        if self._initialized:
            return

        try:
            # Validate that required dependencies are injected
            missing_deps = []
            if self._error_handler is None:
                missing_deps.append("ErrorHandler")
            if self._global_handler is None:
                missing_deps.append("GlobalErrorHandler")
            if self._pattern_analytics is None:
                missing_deps.append("ErrorPatternAnalytics")

            if missing_deps:
                raise ServiceError(
                    f"Required dependencies not injected: {', '.join(missing_deps)}. "
                    "Ensure all dependencies are registered with DI container."
                )

            # StateMonitor is optional but recommended
            if self._state_monitor is None:
                self.logger.warning(
                    "StateMonitor not injected - state monitoring will be disabled. "
                    "Register StateMonitor with DI container for full functionality."
                )

            # Initialize semaphore for backpressure control
            if self._semaphore is None:
                self._semaphore = asyncio.Semaphore(self._max_concurrent_tasks)

            self._initialized = True
            self.logger.info("Error handling service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize error handling service: {e}")
            raise ServiceError(f"Error handling service initialization failed: {e}") from e

    def configure_dependencies(self, injector) -> None:
        """Configure dependencies via dependency injector."""
        try:
            # Resolve dependencies from DI container only if not already provided
            if not self._error_handler and injector.has("ErrorHandler"):
                self._error_handler = injector.get("ErrorHandler")

            if not self._global_handler and injector.has("GlobalErrorHandler"):
                self._global_handler = injector.get("GlobalErrorHandler")

            if not self._pattern_analytics and injector.has("ErrorPatternAnalytics"):
                self._pattern_analytics = injector.get("ErrorPatternAnalytics")

            if not self._state_monitor and injector.has("StateMonitor"):
                self._state_monitor = injector.get("StateMonitor")

            self.logger.debug("Dependencies configured via DI container")
        except Exception as e:
            self.logger.error(f"Failed to configure dependencies via DI: {e}")
            # Don't suppress the error - let the caller handle it
            raise

    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
        recovery_strategy: Any | None = None,
    ) -> dict[str, Any]:
        """
        Handle an error with consistent async processing patterns.

        Args:
            error: The exception to handle
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context data
            recovery_strategy: Optional recovery strategy

        Returns:
            Dict containing error handling result with standardized format
        """
        # Apply consistent data transformation for financial context
        transformed_context = self._transform_error_context(context or {}, component)

        # Apply boundary validation using consistent patterns with risk_management
        transformed_context = ErrorDataTransformer.apply_cross_module_validation(
            transformed_context, source_module="error_handling", target_module="core"
        )

        # Use service execution pattern
        return await self.execute_with_monitoring(
            "handle_error",
            self._handle_error_impl,
            error,
            component,
            operation,
            transformed_context,
            recovery_strategy,
        )

    async def _handle_error_impl(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
        recovery_strategy: Any | None = None,
    ) -> dict[str, Any]:
        """Implementation of error handling business logic."""
        await self._ensure_initialized()

        # Validate data at monitoring/risk_management/web_interface -> error_handling boundary
        if (
            component == "RiskMonitoringService"
            or component.startswith("risk_")
            or component.startswith("monitoring_")
            or component == "web_interface"
            or component.startswith("web_")
        ):
            boundary_data = {
                "component": component,
                "error_type": type(error).__name__,
                "severity": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": "stream",  # Align with core events default
                "data_format": "event_data_v1",  # Align with state module format
                "boundary_crossed": True,
            }
            if context:
                boundary_data.update(context)

            # Use consistent boundary validation patterns
            if component == "web_interface" or component.startswith("web_"):
                # Use specific web interface boundary validation
                BoundaryValidator.validate_web_interface_to_error_boundary(boundary_data)
            else:
                # Use general monitoring boundary validation
                BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

        # Apply consistent error propagation patterns
        if isinstance(error, ValidationError | DataValidationError):
            # Delegate error propagation to service layer
            self.propagate_validation_error(error, f"{component}.{operation}")
            raise error

        # Create error context
        if self._error_handler is None:
            raise ServiceError("ErrorHandler is not available")

        error_context = self._error_handler.create_error_context(
            error=error, component=component, operation=operation, **(context or {})
        )

        # Handle error with recovery
        recovery_success = await self._error_handler.handle_error(
            error=error, context=error_context, recovery_strategy=recovery_strategy
        )

        # Add to pattern analytics with consistent processing mode
        if self._pattern_analytics is not None:
            # Apply processing paradigm alignment for pattern analytics
            error_context_dict = error_context.__dict__.copy()
            error_context_dict["processing_mode"] = transformed_context.get("processing_mode", "stream")
            error_context_dict["data_format"] = transformed_context.get("data_format", "event_data_v1")  # Align with state module format
            self._pattern_analytics.add_error_event(error_context_dict)

        # Create standardized response
        response_data = {
            "error_id": error_context.error_id,
            "handled": True,
            "recovery_success": recovery_success,
            "severity": error_context.severity.value,
            "timestamp": error_context.timestamp.isoformat(),
            "component": component,
            "operation": operation,
            "processing_mode": transformed_context.get("processing_mode", "stream"),
            "data_format": transformed_context.get("data_format", "event_data_v1"),  # Align with state module format
        }

        # Notify monitoring module of error handling completion for consistent cross-module patterns
        await self._notify_monitoring_of_error_handling(response_data)

        return response_data

    async def _notify_monitoring_of_error_handling(self, error_response: dict[str, Any]) -> None:
        """Notify monitoring module of error handling completion with consistent message patterns."""
        try:
            # Use the cross-module data consistency validation
            from src.utils.monitoring_helpers import validate_cross_module_data_consistency

            notification_data = validate_cross_module_data_consistency(
                source_module="error_handling",
                target_module="monitoring",
                data={
                    "error_id": error_response.get("error_id", "unknown"),
                    "component": error_response.get("component", "unknown"),
                    "severity": error_response.get("severity", "medium"),
                    "recovery_success": error_response.get("recovery_success", False),
                    "processing_mode": error_response.get("processing_mode", "stream"),
                    "data_format": "event_data_v1",  # Align with state module format
                    "operation": error_response.get("operation", "unknown"),
                    "timestamp": error_response.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    "notification_type": "error_handling_completed",
                }
            )

            # Log the cross-module notification for monitoring integration
            self.logger.debug(
                "Notified monitoring module of error handling completion",
                extra=notification_data
            )

        except Exception as notification_error:
            # Don't fail error handling if monitoring notification fails
            self.logger.warning(
                f"Failed to notify monitoring module: {notification_error}",
                error_id=error_response.get("error_id", "unknown")
            )

    async def handle_global_error(
        self, error: Exception, context: dict[str, Any] | None = None, severity: str = "error"
    ) -> dict[str, Any]:
        """
        Handle error through global error handler.

        Args:
            error: The exception to handle
            context: Additional context information
            severity: Error severity level

        Returns:
            Dict containing error handling result
        """
        return await self.execute_with_monitoring(
            "handle_global_error", self._handle_global_error_impl, error, context, severity
        )

    async def _handle_global_error_impl(
        self, error: Exception, context: dict[str, Any] | None = None, severity: str = "error"
    ) -> dict[str, Any]:
        """Implementation of global error handling business logic."""
        await self._ensure_initialized()

        if self._global_handler is None:
            raise ServiceError("GlobalErrorHandler is not available")

        return await self._global_handler.handle_error(
            error=error, context=context, severity=severity
        )

    async def validate_state_consistency(self, component: str = "all") -> dict[str, Any]:
        """
        Validate state consistency for specified component.

        Args:
            component: Component to validate or "all" for all components

        Returns:
            Dict containing validation result
        """
        await self._ensure_initialized()

        if self._state_monitor is None:
            raise ServiceError("State monitor not available")

        try:
            result = await self._state_monitor.validate_state_consistency(component)

            return {
                "component": component,
                "is_consistent": result.get("is_consistent", False),
                "discrepancies": result.get("discrepancies", []),
                "severity": result.get("severity", "unknown"),
                "validation_time": result.get(
                    "validation_time", datetime.now(timezone.utc).isoformat()
                ),
            }

        except Exception as e:
            self.logger.error(f"State validation failed: {e}")
            raise ServiceError(f"State validation failed: {e}") from e

    async def reconcile_state_discrepancies(
        self, component: str, discrepancies: list[dict[str, Any]]
    ) -> bool:
        """
        Attempt to reconcile state discrepancies.

        Args:
            component: Component with discrepancies
            discrepancies: List of discrepancy data

        Returns:
            True if reconciliation was successful
        """
        await self._ensure_initialized()

        if self._state_monitor is None:
            raise ServiceError("State monitor not available")

        try:
            success = await self._state_monitor.reconcile_state(component, discrepancies)

            self.logger.info(
                f"State reconciliation {'successful' if success else 'failed'}",
                component=component,
                discrepancy_count=len(discrepancies),
            )

            return success

        except Exception as e:
            self.logger.error(f"State reconciliation failed: {e}")
            return False

    async def get_error_patterns(self) -> dict[str, Any]:
        """
        Get current error patterns and analytics.

        Returns:
            Dict containing error patterns summary
        """
        await self._ensure_initialized()

        try:
            if self._pattern_analytics is None:
                raise ServiceError("ErrorPatternAnalytics is not available")

            patterns = self._pattern_analytics.get_pattern_summary()
            correlations = self._pattern_analytics.get_correlation_summary()
            trends = self._pattern_analytics.get_trend_summary()

            return {
                "patterns": patterns,
                "correlations": correlations,
                "trends": trends,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get error patterns: {e}")
            raise ServiceError(f"Error pattern retrieval failed: {e}") from e

    async def get_state_monitoring_status(self) -> dict[str, Any]:
        """
        Get current state monitoring status and statistics.

        Returns:
            Dict containing monitoring status
        """
        await self._ensure_initialized()

        if self._state_monitor is None:
            raise ServiceError("State monitor not available")

        try:
            summary = self._state_monitor.get_state_summary()

            return {
                "status": "active",
                "summary": summary,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get state monitoring status: {e}")
            raise ServiceError(f"State monitoring status retrieval failed: {e}") from e

    async def get_error_handler_metrics(self) -> dict[str, Any]:
        """
        Get error handler metrics and statistics.

        Returns:
            Dict containing error handler metrics
        """
        await self._ensure_initialized()

        try:
            if self._error_handler is None:
                raise ServiceError("ErrorHandler is not available")
            if self._global_handler is None:
                raise ServiceError("GlobalErrorHandler is not available")

            # Get error handler metrics
            error_patterns = self._error_handler.get_error_patterns()
            circuit_breaker_status = self._error_handler.get_circuit_breaker_status()
            memory_stats = self._error_handler.get_memory_usage_stats()

            # Get global handler statistics
            global_stats = self._global_handler.get_statistics()

            return {
                "error_patterns_count": len(error_patterns),
                "circuit_breakers": circuit_breaker_status,
                "memory_usage": memory_stats,
                "global_handler_stats": global_stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get error handler metrics: {e}")
            raise ServiceError(f"Error handler metrics retrieval failed: {e}") from e

    def _transform_error_context(self, context: dict[str, Any], component: str) -> dict[str, Any]:
        """Transform error context data for consistent processing matching monitoring module patterns."""
        transformed_context = context.copy()

        # Add standard processing fields with consistent module alignment
        transformed_context.update(
            {
                "processing_stage": "error_handling",
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "component": component,
                # Add consistent processing metadata to align with monitoring module
                "processing_mode": transformed_context.get("processing_mode", "stream"),
                "data_format": transformed_context.get("data_format", "bot_event_v1"),  # Consistent with risk_management module
                "message_pattern": "pub_sub",  # Consistent with risk_management module
                "boundary_validation": "applied",
                "error_propagation_pattern": "service_to_service",
            }
        )

        # Apply financial data transformations if present for consistency
        if "price" in transformed_context and transformed_context["price"] is not None:
            from src.utils.decimal_utils import to_decimal
            transformed_context["price"] = to_decimal(transformed_context["price"])

        if "quantity" in transformed_context and transformed_context["quantity"] is not None:
            from src.utils.decimal_utils import to_decimal
            transformed_context["quantity"] = to_decimal(transformed_context["quantity"])

        return transformed_context

    def _validate_data_module_boundary(self, error: Exception, component: str, operation: str, context: dict[str, Any]) -> None:
        """Validate error data at data -> error_handling module boundary."""
        if component == "data" or component.startswith("data_"):
            # Apply data-specific boundary validation
            boundary_data = {
                "component": component,
                "error_type": type(error).__name__,
                "operation": operation,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_mode": context.get("processing_mode", "stream"),
                "data_format": "event_data_v1",  # Align with state module format
                "boundary_crossed": True,
                **context
            }

            # Use consistent boundary validation patterns
            from src.utils.messaging_patterns import BoundaryValidator
            BoundaryValidator.validate_monitoring_to_error_boundary(boundary_data)

    async def start_error_monitoring(self) -> None:
        """Start error monitoring services."""
        await self._ensure_initialized()

        # Start pattern analytics if available
        if self._pattern_analytics and hasattr(self._pattern_analytics, "start_monitoring"):
            try:
                await self._pattern_analytics.start_monitoring()
                self.logger.info("Pattern analytics monitoring started")
            except Exception as e:
                self.logger.warning(f"Failed to start pattern analytics monitoring: {e}")

    async def stop_error_monitoring(self) -> None:
        """Stop error monitoring services."""
        # Stop pattern analytics if available
        if self._pattern_analytics and hasattr(self._pattern_analytics, "stop_monitoring"):
            try:
                await self._pattern_analytics.stop_monitoring()
                self.logger.info("Pattern analytics monitoring stopped")
            except Exception as e:
                self.logger.warning(f"Failed to stop pattern analytics monitoring: {e}")

    async def handle_batch_errors(
        self, errors: list[tuple[Exception, str, str, dict[str, Any]]] | None
    ) -> list[dict[str, Any]]:
        """Handle multiple errors in batch."""
        if not errors:
            return []

        return await self.execute_with_monitoring(
            "handle_batch_errors", self._handle_batch_errors_impl, errors
        )

    async def _handle_batch_errors_impl(
        self, errors: list[tuple[Exception, str, str, dict[str, Any]]] | None
    ) -> list[dict[str, Any]]:
        """Implementation of batch error handling with backpressure control."""
        await self._ensure_initialized()

        async def handle_single_error(error_data):
            error, component, operation, context = error_data
            # Use semaphore to limit concurrent processing
            async with self._semaphore:
                try:
                    return await self._handle_error_impl(error, component, operation, context)
                except Exception as batch_error:
                    # Log but don't fail entire batch
                    self.logger.error(
                        f"Failed to handle error in batch: {batch_error}", original_error=str(error)
                    )
                    import uuid

                    return {
                        "error_id": str(uuid.uuid4()),
                        "handled": False,
                        "recovery_success": False,
                        "severity": "high",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "batch_error": str(batch_error),
                    }

        # Process all errors concurrently but with backpressure control
        # Apply consistent batch processing paradigm alignment
        tasks = [handle_single_error(error_data) for error_data in errors]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Send batch results to pattern analytics for consistent processing paradigm
        if self._pattern_analytics is not None and results:
            try:
                # Convert results to error contexts for batch analytics
                batch_error_contexts = [
                    {
                        "error_id": result.get("error_id", "unknown"),
                        "component": result.get("component", "unknown"),
                        "severity": result.get("severity", "medium"),
                        "processing_mode": "batch",  # Mark as batch processed
                        "data_format": "event_data_v1",  # Align with state module format
                        "timestamp": result.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    }
                    for result in results
                    if isinstance(result, dict) and result.get("handled", False)
                ]

                if batch_error_contexts:
                    await self._pattern_analytics.add_batch_error_events(batch_error_contexts)
            except Exception as analytics_error:
                self.logger.warning(f"Failed to add batch error events to analytics: {analytics_error}")

        return results

    async def start_monitoring(self) -> None:
        """Start continuous state monitoring."""
        await self._ensure_initialized()

        if self._state_monitor is None:
            raise ServiceError("State monitor not available")

        try:
            # Start state monitoring as a background task
            monitoring_task = asyncio.create_task(self._state_monitor.start_monitoring())
            # Store reference to prevent task from being garbage collected
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = set()
            self._background_tasks.add(monitoring_task)

            # Clean up completed tasks
            monitoring_task.add_done_callback(self._background_tasks.discard)

            self.logger.info("Started continuous state monitoring")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise ServiceError(f"Monitoring startup failed: {e}") from e

    async def cleanup_resources(self) -> None:
        """Clean up error handling resources."""
        if not self._initialized:
            return

        try:
            # Cancel background tasks
            if hasattr(self, "_background_tasks"):
                for task in list(self._background_tasks):
                    if not task.done():
                        task.cancel()

                # Wait for tasks to complete with timeout
                if self._background_tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*self._background_tasks, return_exceptions=True),
                            timeout=self._background_task_timeout,
                        )
                    except asyncio.TimeoutError:
                        self.logger.warning("Background task cleanup timed out")
                    finally:
                        self._background_tasks.clear()

            # Cleanup error handler resources
            if self._error_handler:
                try:
                    await self._error_handler.cleanup_resources()
                except Exception as e:
                    self.logger.error(f"Error handler cleanup failed: {e}")

            # Cleanup pattern analytics
            if self._pattern_analytics:
                try:
                    await self._pattern_analytics.cleanup()
                except Exception as e:
                    self.logger.error(f"Pattern analytics cleanup failed: {e}")

            self.logger.info("Error handling service cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def shutdown(self) -> None:
        """Shutdown the error handling service."""
        try:
            await self.cleanup_resources()

            # Shutdown error handler
            if self._error_handler:
                try:
                    await self._error_handler.shutdown()
                except Exception as e:
                    self.logger.error(f"Error handler shutdown failed: {e}")

            self._initialized = False
            self.logger.info("Error handling service shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        finally:
            # Ensure initialized flag is reset even if shutdown fails
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            await self.initialize()

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on error handling service.

        Returns:
            Dict containing health check results
        """
        try:
            await self._ensure_initialized()

            # Check component health
            components_health = {
                "error_handler": self._error_handler is not None,
                "global_handler": self._global_handler is not None,
                "pattern_analytics": self._pattern_analytics is not None,
                "state_monitor": self._state_monitor is not None,
            }

            all_healthy = all(components_health.values())

            from src.core.base.interfaces import HealthStatus

            status = HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED
            return HealthCheckResult(
                status=status,
                details={
                    "components": components_health,
                    "initialized": self._initialized,
                },
            )

        except Exception as e:
            from src.core.base.interfaces import HealthStatus

            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)},
                message=str(e),
            )


def create_error_handling_service(
    config: Config,
    dependency_container: Any | None = None,
    error_handler: Optional["ErrorHandler"] = None,
    global_handler: Optional["GlobalErrorHandler"] = None,
    pattern_analytics: Optional["ErrorPatternAnalytics"] = None,
    state_monitor: StateMonitorInterface | None = None,
) -> "ErrorHandlingService":
    """
    Factory function for creating ErrorHandlingService with dependency injection.

    Args:
        config: Application configuration
        dependency_container: Dependency injection container
        error_handler: Injected error handler (resolved from container if not provided)
        global_handler: Injected global error handler (resolved from container if not provided)
        pattern_analytics: Injected pattern analytics (resolved from container if not provided)
        state_monitor: Injected state monitor (resolved from container if not provided)

    Returns:
        Configured ErrorHandlingService instance
    """
    # Resolve dependencies from container if not provided directly
    if dependency_container:
        if not error_handler and dependency_container.has("ErrorHandler"):
            error_handler = dependency_container.get("ErrorHandler")
        if not global_handler and dependency_container.has("GlobalErrorHandler"):
            global_handler = dependency_container.get("GlobalErrorHandler")
        if not pattern_analytics and dependency_container.has("ErrorPatternAnalytics"):
            pattern_analytics = dependency_container.get("ErrorPatternAnalytics")
        if not state_monitor and dependency_container.has("StateMonitor"):
            state_monitor = dependency_container.get("StateMonitor")

    service = ErrorHandlingService(
        config=config,
        error_handler=error_handler,
        global_handler=global_handler,
        pattern_analytics=pattern_analytics,
        state_monitor=state_monitor,
    )

    # Configure dependencies after creation
    if dependency_container:
        service.configure_dependencies(dependency_container)

    return service
