"""
Error Handling Service - Service layer for error handling operations.

This service layer implements comprehensive error handling business logic, providing
a clean interface for error processing, pattern analysis, state monitoring, and recovery
operations. It orchestrates the underlying error handling components while maintaining
proper separation of concerns.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from src.error_handling.error_handler import ErrorHandler
    from src.error_handling.global_handler import GlobalErrorHandler
    from src.error_handling.pattern_analytics import ErrorPatternAnalytics

from src.core.base.interfaces import HealthCheckResult
from src.core.base.service import BaseService
from src.core.config import Config
from src.core.exceptions import DataValidationError, ServiceError, ValidationError
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
        error_handler: "ErrorHandler | None" = None,
        global_handler: "GlobalErrorHandler | None" = None,
        pattern_analytics: "ErrorPatternAnalytics | None" = None,
        state_monitor: StateMonitorInterface | None = None,
    ) -> None:
        # Configuration constants
        self._background_task_timeout = 10.0

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

            self._initialized = True
            self.logger.info("Error handling service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize error handling service: {e}")
            raise ServiceError(f"Error handling service initialization failed: {e}") from e

    def configure_dependencies(self, injector) -> None:
        """Configure dependencies via dependency injector."""
        try:
            # Resolve dependencies from DI container only if not already provided
            if not self._error_handler and injector.has_service("ErrorHandler"):
                self._error_handler = injector.resolve("ErrorHandler")

            if not self._global_handler and injector.has_service("GlobalErrorHandler"):
                self._global_handler = injector.resolve("GlobalErrorHandler")

            if not self._pattern_analytics and injector.has_service("ErrorPatternAnalytics"):
                self._pattern_analytics = injector.resolve("ErrorPatternAnalytics")

            if not self._state_monitor and injector.has_service("StateMonitor"):
                self._state_monitor = injector.resolve("StateMonitor")

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

        # Use service execution pattern for proper monitoring and error handling
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

        # Apply consistent error propagation patterns matching database module validation
        if isinstance(error, (ValidationError, DataValidationError)):
            # Apply same error propagation pattern as database module
            from src.utils.messaging_patterns import ErrorPropagationMixin

            propagator = ErrorPropagationMixin()
            propagator.propagate_validation_error(error, f"{component}.{operation}")
            # propagate_validation_error should always raise, but add explicit raise for mypy
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

        # Add to pattern analytics
        if self._pattern_analytics is not None:
            self._pattern_analytics.add_error_event(error_context.__dict__)

        # Create response with boundary validation for monitoring module communication
        response_data = {
            "error_id": error_context.error_id,
            "handled": True,
            "recovery_success": recovery_success,
            "severity": error_context.severity.value,
            "timestamp": error_context.timestamp.isoformat(),
            "component": component,
            "operation": operation,
        }

        # Validate at error_handling -> monitoring boundary for consistent data flow
        BoundaryValidator.validate_error_to_monitoring_boundary(response_data)

        return response_data

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

    def _transform_error_context(
        self, context: dict[str, Any], component: str
    ) -> dict[str, Any]:
        """Transform error context data consistently across operations."""
        # Apply consistent data transformation patterns matching database module exactly
        transformed_context = context.copy()

        # Use same messaging patterns transformation as database module
        from src.utils.messaging_patterns import MessagingCoordinator

        coordinator = MessagingCoordinator("ErrorHandlingTransform")
        transformed_context = coordinator._apply_data_transformation(transformed_context)

        # Standardize context format for consistent processing with database module
        transformed_context.update(
            {
                "processing_mode": "async",
                "processing_stage": "error_handling",
                "data_format": "error_context_v1",
                # Add fields consistent with database module patterns
                "validation_status": "validated",
                "boundary_crossed": True,
            }
        )

        # Set audit fields consistently matching database module
        transformed_context["processed_at"] = datetime.now(timezone.utc).isoformat()

        return transformed_context

    async def handle_batch_errors(
        self, errors: list[tuple[Exception, str, str, dict[str, Any] | None]]
    ) -> list[dict[str, Any]]:
        """Handle multiple errors in batch for consistent processing paradigm."""
        if not errors:
            return []

        return await self.execute_with_monitoring(
            "handle_batch_errors", self._handle_batch_errors_impl, errors
        )

    async def _handle_batch_errors_impl(
        self, errors: list[tuple[Exception, str, str, dict[str, Any] | None]]
    ) -> list[dict[str, Any]]:
        """Implementation of batch error handling."""
        await self._ensure_initialized()

        results = []
        for error, component, operation, context in errors:
            try:
                result = await self._handle_error_impl(error, component, operation, context)
                results.append(result)
            except Exception as batch_error:
                # Log but don't fail entire batch
                self.logger.error(
                    f"Failed to handle error in batch: {batch_error}", original_error=str(error)
                )
                import uuid

                results.append(
                    {
                        "error_id": str(uuid.uuid4()),
                        "handled": False,
                        "recovery_success": False,
                        "severity": "high",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "batch_error": str(batch_error),
                    }
                )

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


# Factory function for dependency injection
def create_error_handling_service(
    config: Config,
    dependency_container: Any | None = None,
    error_handler: "ErrorHandler | None" = None,
    global_handler: "GlobalErrorHandler | None" = None,
    pattern_analytics: "ErrorPatternAnalytics | None" = None,
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
        if not error_handler and dependency_container.has_service("ErrorHandler"):
            error_handler = dependency_container.resolve("ErrorHandler")
        if not global_handler and dependency_container.has_service("GlobalErrorHandler"):
            global_handler = dependency_container.resolve("GlobalErrorHandler")
        if not pattern_analytics and dependency_container.has_service("ErrorPatternAnalytics"):
            pattern_analytics = dependency_container.resolve("ErrorPatternAnalytics")
        if not state_monitor and dependency_container.has_service("StateMonitor"):
            state_monitor = dependency_container.resolve("StateMonitor")

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
