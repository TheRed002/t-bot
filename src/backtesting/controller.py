"""
Backtesting Controller - HTTP/API interface layer.

This controller provides the external API for backtesting operations
while delegating all business logic to the BacktestService.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from src.core.base.interfaces import HealthCheckResult
from src.core.base.component import BaseComponent
from src.core.event_constants import BacktestEvents
from src.core.exceptions import ServiceError, ValidationError
# Robust logger import with fallback for test suite compatibility
try:
    from src.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.backtesting.interfaces import BacktestServiceInterface


class BacktestController(BaseComponent):
    """
    Controller for backtesting operations.

    This controller handles HTTP/API requests and delegates all business logic
    to the BacktestService, following proper service layer architecture.
    """

    def __init__(self, backtest_service: "BacktestServiceInterface"):
        """
        Initialize controller with service dependency.

        Args:
            backtest_service: BacktestService instance for business logic
        """
        super().__init__(name="BacktestController")
        self.backtest_service = backtest_service
        self._get_local_logger().info("BacktestController initialized")

    def _get_local_logger(self):
        """Get logger with robust fallback for test environments."""
        try:
            from src.core.logging import get_logger
            return get_logger(__name__)
        except ImportError:
            import logging
            return logging.getLogger(__name__)

    async def run_backtest(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Handle backtest request via API.

        Args:
            request_data: Raw request data from API

        Returns:
            Serialized backtest result
        """
        try:
            # Delegate request validation and processing to service layer
            result = await self.backtest_service.run_backtest_from_dict(request_data)

            # Delegate serialization to service layer
            return await self.backtest_service.serialize_result(result)

        except ValidationError as e:
            self._get_local_logger().error(f"Validation error in backtest request: {e}")
            raise
        except ServiceError as e:
            self._get_local_logger().error(f"Service error running backtest: {e}")
            raise
        except Exception as e:
            self._get_local_logger().error(f"Unexpected error in backtest controller: {e}")
            raise ServiceError(
                f"Backtest controller error: {e}", error_code="CONTROLLER_001"
            ) from e

    async def get_active_backtests(self) -> dict[str, Any]:
        """Get status of active backtests."""
        try:
            return await self.backtest_service.get_active_backtests()
        except Exception as e:
            self._get_local_logger().error(f"Error getting active backtests: {e}")
            raise ServiceError(
                f"Failed to get active backtests: {e}", error_code="CONTROLLER_002"
            ) from e

    async def cancel_backtest(self, backtest_id: str) -> dict[str, Any]:
        """Cancel a specific backtest."""
        try:
            if not backtest_id:
                raise ValidationError(
                    "Backtest ID is required",
                    field_name="backtest_id",
                    field_value=backtest_id,
                    error_code="CONTROLLER_003",
                )

            success = await self.backtest_service.cancel_backtest(backtest_id)
            return {BacktestEvents.CANCELLED.replace("backtest.", ""): success, "backtest_id": backtest_id}

        except ValidationError:
            raise
        except Exception as e:
            self._get_local_logger().error(f"Error cancelling backtest {backtest_id}: {e}")
            raise ServiceError(
                f"Failed to cancel backtest: {e}", error_code="CONTROLLER_004"
            ) from e

    async def clear_cache(self, pattern: str = "*") -> dict[str, Any]:
        """Clear backtest cache."""
        try:
            cleared_count = await self.backtest_service.clear_cache(pattern)
            return {"cleared_entries": cleared_count, "pattern": pattern}
        except Exception as e:
            self._get_local_logger().error(f"Error clearing cache: {e}")
            raise ServiceError(f"Failed to clear cache: {e}", error_code="CONTROLLER_005") from e

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            return await self.backtest_service.get_cache_stats()
        except Exception as e:
            self._get_local_logger().error(f"Error getting cache stats: {e}")
            raise ServiceError(
                f"Failed to get cache stats: {e}", error_code="CONTROLLER_006"
            ) from e

    async def health_check(self) -> HealthCheckResult:
        """Perform health check of controller and service."""
        from src.core.base.interfaces import HealthStatus

        try:
            service_health = await self.backtest_service.health_check()
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                details={
                    "component": "BacktestController",
                    "controller": "healthy",
                    "service": service_health.status.value
                    if hasattr(service_health.status, "value")
                    else str(service_health.status),
                    "service_details": service_health.details
                    if hasattr(service_health, "details")
                    else {},
                },
            )
        except Exception as e:
            self._get_local_logger().error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                details={
                    "component": "BacktestController",
                    "controller": "unhealthy",
                    "service": "unknown",
                    "error": str(e),
                },
            )

    async def get_backtest_result(self, result_id: str) -> dict[str, Any]:
        """Get a specific backtest result by ID."""
        try:
            if not result_id:
                raise ValidationError(
                    "Result ID is required",
                    field_name="result_id",
                    field_value=result_id,
                    error_code="CONTROLLER_007",
                )

            result = await self.backtest_service.get_backtest_result(result_id)
            if result is None:
                return {"error": "Backtest result not found", "result_id": result_id}

            return {"result": result}

        except ValidationError:
            raise
        except Exception as e:
            self._get_local_logger().error(f"Error getting backtest result {result_id}: {e}")
            raise ServiceError(
                f"Failed to get backtest result: {e}", error_code="CONTROLLER_008"
            ) from e

    async def list_backtest_results(
        self, limit: int = 50, offset: int = 0, strategy_type: str | None = None
    ) -> dict[str, Any]:
        """List backtest results with filtering."""
        try:
            results = await self.backtest_service.list_backtest_results(
                limit=limit, offset=offset, strategy_type=strategy_type
            )
            return {
                "results": results,
                "count": len(results),
                "limit": limit,
                "offset": offset,
                "strategy_type": strategy_type,
            }

        except Exception as e:
            self._get_local_logger().error(f"Error listing backtest results: {e}")
            raise ServiceError(
                f"Failed to list backtest results: {e}", error_code="CONTROLLER_009"
            ) from e

    async def delete_backtest_result(self, result_id: str) -> dict[str, Any]:
        """Delete a specific backtest result by ID."""
        try:
            if not result_id:
                raise ValidationError(
                    "Result ID is required",
                    field_name="result_id",
                    field_value=result_id,
                    error_code="CONTROLLER_010",
                )

            success = await self.backtest_service.delete_backtest_result(result_id)
            return {"deleted": success, "result_id": result_id}

        except ValidationError:
            raise
        except Exception as e:
            self._get_local_logger().error(f"Error deleting backtest result {result_id}: {e}")
            raise ServiceError(
                f"Failed to delete backtest result: {e}", error_code="CONTROLLER_011"
            ) from e

    async def cleanup(self) -> None:
        """Cleanup controller resources with proper async coordination."""
        try:
            if hasattr(self.backtest_service, "cleanup"):
                # Use timeout to prevent hanging cleanup
                await asyncio.wait_for(self.backtest_service.cleanup(), timeout=30.0)
            self._get_local_logger().info("BacktestController cleanup completed")
        except asyncio.TimeoutError:
            self._get_local_logger().error("BacktestController cleanup timed out")
        except Exception as e:
            self._get_local_logger().error(f"BacktestController cleanup error: {e}")
            # Don't re-raise to avoid masking original issues
