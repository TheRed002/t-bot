"""
Execution Controller.

This controller acts as the API layer for execution operations,
following proper service layer patterns by only calling services
and never accessing repositories directly.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base.component import BaseComponent
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import MarketData, OrderRequest
from src.execution.execution_orchestration_service import ExecutionOrchestrationService
from src.execution.interfaces import ExecutionServiceInterface

try:
    from src.utils import time_execution
except ImportError:
    # Fallback if time_execution is not available
    def time_execution(func):
        return func


class ExecutionController(BaseComponent):
    """
    Execution module controller.
    
    This controller provides the API interface for execution operations,
    strictly following service layer patterns by only calling services
    and never accessing repositories directly.
    """

    def __init__(
        self,
        orchestration_service: ExecutionOrchestrationService,
        execution_service: ExecutionServiceInterface,
    ):
        """
        Initialize execution controller.
        
        Args:
            orchestration_service: Main orchestration service
            execution_service: Core execution service
        """
        super().__init__(name="ExecutionController")

        self.orchestration_service = orchestration_service
        self.execution_service = execution_service
        # Note: logger is inherited from BaseComponent

        if not orchestration_service:
            raise ValueError("ExecutionOrchestrationService is required")
        if not execution_service:
            raise ValueError("ExecutionService is required")

    @time_execution
    async def execute_order(
        self,
        order_data: dict[str, Any],
        market_data: dict[str, Any],
        bot_id: str | None = None,
        strategy_name: str | None = None,
        execution_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute an order through the service layer.
        
        Args:
            order_data: Order data dictionary
            market_data: Market data dictionary
            bot_id: Bot identifier
            strategy_name: Strategy name
            execution_params: Execution parameters
            
        Returns:
            Dict containing execution result
            
        Raises:
            ValidationError: If input validation fails
            ServiceError: If execution fails
        """
        try:
            # Validate inputs
            if not order_data:
                raise ValidationError("Order data is required")
            if not market_data:
                raise ValidationError("Market data is required")

            # Convert dictionaries to proper types
            order_request = self._convert_to_order_request(order_data)
            market_data_obj = self._convert_to_market_data(market_data)

            # Execute through orchestration service
            execution_result = await self.orchestration_service.execute_order(
                order=order_request,
                market_data=market_data_obj,
                bot_id=bot_id,
                strategy_name=strategy_name,
                execution_params=execution_params,
            )

            # Convert result to API response format
            response = {
                "success": True,
                "execution_id": execution_result.execution_id,
                "status": execution_result.status.value,
                "filled_quantity": str(execution_result.total_filled_quantity),
                "average_price": (
                    str(execution_result.average_fill_price)
                    if execution_result.average_fill_price else None
                ),
                "total_fees": str(execution_result.total_fees) if execution_result.total_fees else "0",
                "execution_time": execution_result.execution_duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "algorithm": execution_result.algorithm.value if execution_result.algorithm else None,
            }

            self.logger.info(
                "Order execution completed via controller",
                execution_id=execution_result.execution_id,
                symbol=order_request.symbol,
                bot_id=bot_id,
            )

            return response

        except ValidationError as e:
            self.logger.error("Order validation failed in controller", error=str(e))
            return {
                "success": False,
                "error": "validation_error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except ServiceError as e:
            self.logger.error("Service error in order execution", error=str(e))
            return {
                "success": False,
                "error": "service_error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.logger.error("Unexpected error in order execution controller", error=str(e))
            return {
                "success": False,
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @time_execution
    async def get_execution_metrics(
        self,
        bot_id: str | None = None,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get execution metrics through service layer.
        
        Args:
            bot_id: Optional bot ID filter
            symbol: Optional symbol filter
            time_range_hours: Time range for metrics
            
        Returns:
            Dict containing metrics data
        """
        try:
            # Get metrics through orchestration service
            metrics = await self.orchestration_service.get_comprehensive_metrics(
                bot_id=bot_id,
                symbol=symbol,
                time_range_hours=time_range_hours,
            )

            return {
                "success": True,
                "data": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error("Failed to get execution metrics", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @time_execution
    async def cancel_execution(
        self,
        execution_id: str,
        reason: str = "user_request"
    ) -> dict[str, Any]:
        """
        Cancel execution through service layer.
        
        Args:
            execution_id: Execution ID to cancel
            reason: Cancellation reason
            
        Returns:
            Dict containing cancellation result
        """
        try:
            if not execution_id:
                raise ValidationError("Execution ID is required")

            # Cancel through orchestration service
            result = await self.orchestration_service.cancel_execution(
                execution_id=execution_id,
                reason=reason,
            )

            return {
                "success": result,
                "execution_id": execution_id,
                "cancelled": result,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except ValidationError as e:
            return {
                "success": False,
                "error": "validation_error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.logger.error("Failed to cancel execution", execution_id=execution_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @time_execution
    async def get_active_executions(self) -> dict[str, Any]:
        """
        Get active executions through service layer.
        
        Returns:
            Dict containing active executions
        """
        try:
            # Get active executions through orchestration service
            active_executions = await self.orchestration_service.get_active_executions()

            return {
                "success": True,
                "data": active_executions,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error("Failed to get active executions", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @time_execution
    async def validate_order(
        self,
        order_data: dict[str, Any],
        market_data: dict[str, Any],
        bot_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Validate order through service layer.
        
        Args:
            order_data: Order data to validate
            market_data: Current market data
            bot_id: Optional bot ID
            
        Returns:
            Dict containing validation results
        """
        try:
            # Validate inputs
            if not order_data:
                raise ValidationError("Order data is required")
            if not market_data:
                raise ValidationError("Market data is required")

            # Convert to proper types
            order_request = self._convert_to_order_request(order_data)
            market_data_obj = self._convert_to_market_data(market_data)

            # Validate through execution service
            validation_result = await self.execution_service.validate_order_pre_execution(
                order=order_request,
                market_data=market_data_obj,
                bot_id=bot_id,
                risk_context={"component": "ExecutionController"}
            )

            return {
                "success": True,
                "validation_results": validation_result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except ValidationError as e:
            return {
                "success": False,
                "error": "validation_error",
                "message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            self.logger.error("Order validation failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check through service layer.
        
        Returns:
            Dict containing health status
        """
        try:
            # Get health status through orchestration service
            health_status = await self.orchestration_service.health_check()

            return {
                "success": True,
                "controller": "ExecutionController",
                "is_healthy": True,
                "service_health": health_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "success": False,
                "controller": "ExecutionController",
                "is_healthy": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Private helper methods for data conversion
    def _convert_to_order_request(self, order_data: dict[str, Any]) -> OrderRequest:
        """Convert dictionary to OrderRequest."""
        from decimal import Decimal

        from src.core.types import OrderSide, OrderType

        try:
            return OrderRequest(
                symbol=order_data["symbol"],
                side=OrderSide(order_data["side"]),
                order_type=OrderType(order_data.get("order_type", "MARKET")),
                quantity=Decimal(str(order_data["quantity"])),
                price=Decimal(str(order_data["price"])) if order_data.get("price") else None,
                time_in_force=order_data.get("time_in_force"),
                exchange=order_data.get("exchange"),
                client_order_id=order_data.get("client_order_id"),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid order data: {e}") from e

    def _convert_to_market_data(self, market_data: dict[str, Any]) -> MarketData:
        """Convert dictionary to MarketData."""
        from decimal import Decimal

        try:
            return MarketData(
                symbol=market_data["symbol"],
                price=Decimal(str(market_data["price"])),
                volume=Decimal(str(market_data.get("volume", 0))) if market_data.get("volume") else None,
                bid=Decimal(str(market_data["bid"])) if market_data.get("bid") else None,
                ask=Decimal(str(market_data["ask"])) if market_data.get("ask") else None,
                timestamp=datetime.fromisoformat(market_data["timestamp"]) if market_data.get("timestamp") else datetime.now(timezone.utc),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid market data: {e}") from e
