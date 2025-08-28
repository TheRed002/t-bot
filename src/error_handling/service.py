"""
Error Handling Service - Service layer for error handling operations.

This service layer implements comprehensive error handling business logic, providing
a clean interface for error processing, pattern analysis, state monitoring, and recovery
operations. It orchestrates the underlying error handling components while maintaining
proper separation of concerns.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from src.core.base.service import BaseService
from src.core.config import Config
from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.error_handling.error_handler import ErrorHandler, get_error_handler
from src.error_handling.global_handler import GlobalErrorHandler, get_global_error_handler
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.state_monitor import StateMonitor


class ErrorHandlingService(BaseService):
    """
    Service layer for error handling operations.

    This service orchestrates error handling components and provides a unified
    interface for error processing, pattern analysis, and state monitoring.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = get_logger(self.__class__.__module__)

        # Initialize components via dependency injection
        self._error_handler: ErrorHandler | None = None
        self._global_handler: GlobalErrorHandler | None = None
        self._pattern_analytics: ErrorPatternAnalytics | None = None
        self._state_monitor: StateMonitor | None = None

        # Service state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the error handling service and its components."""
        if self._initialized:
            return

        try:
            # Initialize error handler
            self._error_handler = get_error_handler()

            # Initialize global error handler
            self._global_handler = get_global_error_handler()

            # Initialize pattern analytics
            self._pattern_analytics = ErrorPatternAnalytics(self.config)

            # Initialize state monitor
            self._state_monitor = StateMonitor(self.config)

            self._initialized = True
            self.logger.info("Error handling service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize error handling service: {e}")
            raise ServiceError(f"Error handling service initialization failed: {e}") from e

    async def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
        recovery_strategy: Any | None = None,
    ) -> dict[str, Any]:
        """
        Handle an error with proper context and recovery strategy.

        Args:
            error: The exception to handle
            component: Component where error occurred
            operation: Operation being performed
            context: Additional context data
            recovery_strategy: Optional recovery strategy

        Returns:
            Dict containing error handling result
        """
        await self._ensure_initialized()

        try:
            # Create error context
            error_context = self._error_handler.create_error_context(
                error=error, component=component, operation=operation, **(context or {})
            )

            # Handle error with recovery
            recovery_success = await self._error_handler.handle_error(
                error=error, context=error_context, recovery_strategy=recovery_strategy
            )

            # Add to pattern analytics
            self._pattern_analytics.add_error_event(error_context.__dict__)

            return {
                "error_id": error_context.error_id,
                "handled": True,
                "recovery_success": recovery_success,
                "severity": error_context.severity.value,
                "timestamp": error_context.timestamp.isoformat(),
            }

        except Exception as handling_error:
            self.logger.error(
                f"Error handling failed: {handling_error}",
                original_error=str(error),
                component=component,
            )
            raise ServiceError(f"Error handling failed: {handling_error}") from handling_error

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
        await self._ensure_initialized()

        try:
            result = await self._global_handler.handle_error(
                error=error, context=context, severity=severity
            )

            return result

        except Exception as handling_error:
            self.logger.error(f"Global error handling failed: {handling_error}")
            raise ServiceError(
                f"Global error handling failed: {handling_error}"
            ) from handling_error

    async def validate_state_consistency(self, component: str = "all") -> dict[str, Any]:
        """
        Validate state consistency for specified component.

        Args:
            component: Component to validate or "all" for all components

        Returns:
            Dict containing validation result
        """
        await self._ensure_initialized()

        try:
            result = await self._state_monitor.validate_state_consistency(component)

            return {
                "component": component,
                "is_consistent": result.is_consistent,
                "discrepancies": result.discrepancies,
                "severity": result.severity,
                "validation_time": result.validation_time.isoformat(),
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

    async def start_monitoring(self) -> None:
        """Start continuous state monitoring."""
        await self._ensure_initialized()

        try:
            # Start state monitoring as a background task
            monitoring_task = asyncio.create_task(self._state_monitor.start_monitoring())
            # Store reference to prevent task from being garbage collected
            self._monitoring_task = monitoring_task
            self.logger.info("Started continuous state monitoring")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise ServiceError(f"Monitoring startup failed: {e}") from e

    async def cleanup_resources(self) -> None:
        """Clean up error handling resources."""
        if not self._initialized:
            return

        try:
            # Cleanup error handler resources
            if self._error_handler:
                await self._error_handler.cleanup_resources()

            # Cleanup pattern analytics
            if self._pattern_analytics:
                await self._pattern_analytics.cleanup()

            self.logger.info("Error handling service cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    async def shutdown(self) -> None:
        """Shutdown the error handling service."""
        try:
            await self.cleanup_resources()

            # Shutdown error handler
            if self._error_handler:
                await self._error_handler.shutdown()

            self._initialized = False
            self.logger.info("Error handling service shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def _ensure_initialized(self) -> None:
        """Ensure service is initialized."""
        if not self._initialized:
            await self.initialize()

    async def health_check(self) -> dict[str, Any]:
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

            return {
                "status": "healthy" if all_healthy else "degraded",
                "components": components_health,
                "initialized": self._initialized,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Service factory function
def create_error_handling_service(config: Config) -> ErrorHandlingService:
    """
    Create and initialize an error handling service instance.

    Args:
        config: Application configuration

    Returns:
        Configured ErrorHandlingService instance
    """
    return ErrorHandlingService(config)


# Module-level service instance
_service_instance: ErrorHandlingService | None = None


def get_error_handling_service(config: Config | None = None) -> ErrorHandlingService:
    """
    Get the error handling service instance.

    Args:
        config: Optional configuration (creates new instance if provided)

    Returns:
        ErrorHandlingService instance
    """
    global _service_instance

    if config is not None or _service_instance is None:
        if config is None:
            # Create default config if none provided
            config = Config()
        _service_instance = create_error_handling_service(config)

    return _service_instance
