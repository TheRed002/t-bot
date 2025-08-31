"""
Service adapter for error handling module.

This adapter ensures proper service layer architecture by providing
a clean interface between the service layer and infrastructure components.
"""

from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError
from src.error_handling.interfaces import (
    ErrorHandlingServiceInterface,
    ErrorHandlingServicePort,
)


class ErrorHandlingServiceAdapter(BaseService, ErrorHandlingServicePort):
    """
    Service adapter that implements proper service layer architecture.

    This adapter ensures that:
    1. Business logic is separated from infrastructure concerns
    2. Dependencies are properly injected
    3. Service interfaces are clearly defined
    4. Cross-cutting concerns are handled consistently
    """

    def __init__(
        self,
        config: Any,
        error_handling_service: ErrorHandlingServiceInterface,
    ) -> None:
        super().__init__(name="ErrorHandlingServiceAdapter", config=config)
        self._error_handling_service = error_handling_service

        if not error_handling_service:
            raise ValueError("ErrorHandlingService must be injected via DI")

    async def process_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process an error through the service layer.

        This method provides a clean interface for error processing while
        ensuring proper service layer architecture.
        """
        try:
            # Validate inputs at service boundary
            if not error:
                raise ServiceError("Error parameter is required")
            if not component:
                raise ServiceError("Component parameter is required")
            if not operation:
                raise ServiceError("Operation parameter is required")

            # Delegate to underlying service with proper error handling
            return await self._error_handling_service.handle_error(
                error=error,
                component=component,
                operation=operation,
                context=context,
            )

        except Exception as e:
            self.logger.error(f"Failed to process error: {e}")
            raise ServiceError(f"Error processing failed: {e}") from e

    async def analyze_error_patterns(self) -> dict[str, Any]:
        """Analyze current error patterns."""
        try:
            return await self._error_handling_service.get_error_patterns()
        except Exception as e:
            self.logger.error(f"Failed to analyze error patterns: {e}")
            raise ServiceError(f"Error pattern analysis failed: {e}") from e

    async def validate_system_state(self, component: str = "all") -> dict[str, Any]:
        """Validate system state consistency."""
        try:
            return await self._error_handling_service.validate_state_consistency(component)
        except Exception as e:
            self.logger.error(f"Failed to validate system state: {e}")
            raise ServiceError(f"State validation failed: {e}") from e

    async def handle_critical_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle critical error through global handler."""
        try:
            return await self._error_handling_service.handle_global_error(
                error=error, context=context, severity="critical"
            )
        except Exception as e:
            self.logger.error(f"Failed to handle critical error: {e}")
            raise ServiceError(f"Critical error handling failed: {e}") from e

    async def get_service_health(self) -> dict[str, Any]:
        """Get error handling service health status."""
        try:
            health_result = await self._error_handling_service.health_check()
            return {
                "status": health_result.status.value,
                "details": health_result.details,
                "message": health_result.message,
            }
        except Exception as e:
            self.logger.error(f"Failed to get service health: {e}")
            return {
                "status": "unhealthy",
                "details": {"error": str(e)},
                "message": f"Health check failed: {e}",
            }


def create_error_handling_service_adapter(
    config: Any,
    error_handling_service: ErrorHandlingServiceInterface,
) -> ErrorHandlingServiceAdapter:
    """
    Factory function for creating ErrorHandlingServiceAdapter.

    Args:
        config: Application configuration
        error_handling_service: Injected error handling service

    Returns:
        Configured service adapter
    """
    return ErrorHandlingServiceAdapter(
        config=config,
        error_handling_service=error_handling_service,
    )


__all__ = [
    "ErrorHandlingServiceAdapter",
    "create_error_handling_service_adapter",
]
