"""
Execution Orchestration Service.

This service acts as the main orchestration layer for all execution operations,
properly implementing service layer patterns and acting as a facade for the
execution subsystem.
"""

from datetime import datetime, timezone
from typing import Any

from src.core.base.service import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import ExecutionResult, MarketData, OrderRequest
from src.execution.interfaces import (
    ExecutionEngineServiceInterface,
    ExecutionServiceInterface,
    OrderManagementServiceInterface,
    RiskValidationServiceInterface,
)
from src.execution.types import ExecutionInstruction

try:
    from src.utils import time_execution
except ImportError:
    # Fallback if time_execution is not available
    def time_execution(func):
        return func


class ExecutionOrchestrationService(BaseService):
    """
    Orchestration service for all execution operations.
    
    This service acts as the main entry point for execution operations,
    coordinating between the various execution subsystems while maintaining
    proper service layer separation.
    """

    def __init__(
        self,
        execution_service: ExecutionServiceInterface,
        order_management_service: OrderManagementServiceInterface,
        execution_engine_service: ExecutionEngineServiceInterface,
        risk_validation_service: RiskValidationServiceInterface | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """
        Initialize execution orchestration service.
        
        Args:
            execution_service: Core execution service
            order_management_service: Order management service
            execution_engine_service: Execution engine service
            risk_validation_service: Risk validation service
            correlation_id: Request correlation ID
        """
        super().__init__(
            name="ExecutionOrchestrationService",
            correlation_id=correlation_id,
        )

        self.execution_service = execution_service
        self.order_management_service = order_management_service
        self.execution_engine_service = execution_engine_service
        self.risk_validation_service = risk_validation_service

        self.logger = get_logger(__name__)

        # Dependencies are injected via constructor

    async def _do_start(self) -> None:
        """Start orchestration service."""
        # Validate all required services are available
        if not self.execution_service:
            raise ServiceError("ExecutionService is required")
        if not self.order_management_service:
            raise ServiceError("OrderManagementService is required")
        if not self.execution_engine_service:
            raise ServiceError("ExecutionEngineService is required")

        self.logger.info("Execution orchestration service started")

    @time_execution
    async def execute_order(
        self,
        order: OrderRequest,
        market_data: MarketData,
        bot_id: str | None = None,
        strategy_name: str | None = None,
        execution_params: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute an order through the execution orchestration layer.
        
        This method coordinates all aspects of order execution:
        1. Risk validation
        2. Order management setup
        3. Execution engine processing
        4. Results recording
        
        Args:
            order: Order to execute
            market_data: Current market data
            bot_id: Associated bot ID
            strategy_name: Strategy name
            execution_params: Additional execution parameters
            
        Returns:
            ExecutionResult: Execution result
            
        Raises:
            ServiceError: If execution fails
            ValidationError: If validation fails
        """
        try:
            # Step 1: Risk validation if service available
            if self.risk_validation_service:
                risk_validation = await self.risk_validation_service.validate_order_risk(
                    order=order,
                    market_data=market_data,
                    context={
                        "bot_id": bot_id,
                        "strategy_name": strategy_name,
                        "execution_params": execution_params,
                    }
                )

                if not risk_validation.get("approved", False):
                    raise ValidationError(
                        f"Risk validation failed: {risk_validation.get('reason', 'Unknown')}"
                    )

            # Step 2: Pre-execution validation through ExecutionService
            pre_validation = await self.execution_service.validate_order_pre_execution(
                order=order,
                market_data=market_data,
                bot_id=bot_id,
                risk_context={
                    "component": "ExecutionOrchestrationService",
                    "strategy_name": strategy_name,
                }
            )

            if pre_validation.get("overall_result") == "failed":
                errors = "; ".join(pre_validation.get("errors", []))
                raise ValidationError(f"Pre-execution validation failed: {errors}")

            # Step 3: Create execution instruction
            execution_instruction = ExecutionInstruction(
                order=order,
                strategy_name=strategy_name,
                **(execution_params or {})
            )

            # Step 4: Execute through execution engine service
            execution_result = await self.execution_engine_service.execute_instruction(
                instruction=execution_instruction,
                market_data=market_data,
                bot_id=bot_id,
                strategy_name=strategy_name,
            )

            # Step 5: Record execution through ExecutionService
            await self.execution_service.record_trade_execution(
                execution_result=execution_result,
                market_data=market_data,
                bot_id=bot_id,
                strategy_name=strategy_name,
                pre_trade_analysis=pre_validation,
                post_trade_analysis={
                    "orchestration_timestamp": datetime.now(timezone.utc).isoformat(),
                    "orchestration_component": "ExecutionOrchestrationService",
                }
            )

            self.logger.info(
                "Order execution completed through orchestration",
                order_symbol=order.symbol,
                execution_id=execution_result.execution_id,
                bot_id=bot_id,
                strategy_name=strategy_name,
            )

            return execution_result

        except (ValidationError, ServiceError) as e:
            self.logger.error(
                "Order execution failed in orchestration",
                order_symbol=order.symbol,
                bot_id=bot_id,
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error in order execution orchestration",
                order_symbol=order.symbol,
                bot_id=bot_id,
                error=str(e),
            )
            raise ServiceError(f"Execution orchestration failed: {e}") from e

    @time_execution
    async def get_comprehensive_metrics(
        self,
        bot_id: str | None = None,
        symbol: str | None = None,
        time_range_hours: int = 24,
    ) -> dict[str, Any]:
        """
        Get comprehensive metrics from all execution services.
        
        Args:
            bot_id: Optional bot ID filter
            symbol: Optional symbol filter
            time_range_hours: Time range for metrics
            
        Returns:
            Dict containing comprehensive metrics
        """
        try:
            # Gather metrics from all services in parallel with proper error handling
            import asyncio

            metrics_tasks = [
                self.execution_service.get_execution_metrics(
                    bot_id=bot_id,
                    symbol=symbol,
                    time_range_hours=time_range_hours,
                ),
                self.order_management_service.get_order_metrics(
                    symbol=symbol,
                    time_range_hours=time_range_hours,
                ),
                self.execution_engine_service.get_performance_metrics(),
            ]

            # Use asyncio.gather with proper timeout and exception handling
            try:
                execution_metrics, order_metrics, engine_metrics = await asyncio.wait_for(
                    asyncio.gather(*metrics_tasks, return_exceptions=True),
                    timeout=30.0  # 30 second timeout for metrics gathering
                )

                # Handle individual service exceptions
                if isinstance(execution_metrics, Exception):
                    self.logger.warning(f"Execution metrics failed: {execution_metrics}")
                    execution_metrics = {"error": str(execution_metrics)}

                if isinstance(order_metrics, Exception):
                    self.logger.warning(f"Order metrics failed: {order_metrics}")
                    order_metrics = {"error": str(order_metrics)}

                if isinstance(engine_metrics, Exception):
                    self.logger.warning(f"Engine metrics failed: {engine_metrics}")
                    engine_metrics = {"error": str(engine_metrics)}

            except asyncio.TimeoutError:
                self.logger.error("Metrics gathering timeout")
                execution_metrics = {"error": "timeout"}
                order_metrics = {"error": "timeout"}
                engine_metrics = {"error": "timeout"}

            return {
                "orchestration_service": {
                    "service_name": self.name,
                    "is_running": self.is_running,
                    "metrics_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "execution_metrics": execution_metrics,
                "order_management_metrics": order_metrics,
                "execution_engine_metrics": engine_metrics,
                "service_health": {
                    "execution_service": bool(self.execution_service),
                    "order_management_service": bool(self.order_management_service),
                    "execution_engine_service": bool(self.execution_engine_service),
                    "risk_validation_service": bool(self.risk_validation_service),
                }
            }

        except Exception as e:
            self.logger.error("Failed to get comprehensive metrics", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def cancel_execution(
        self,
        execution_id: str,
        reason: str = "user_request"
    ) -> bool:
        """
        Cancel an execution through orchestration.
        
        Args:
            execution_id: Execution ID to cancel
            reason: Cancellation reason
            
        Returns:
            bool: True if cancellation successful
        """
        try:
            # Cancel through execution engine service
            result = await self.execution_engine_service.cancel_execution(execution_id)

            if result:
                self.logger.info(
                    "Execution cancelled through orchestration",
                    execution_id=execution_id,
                    reason=reason,
                )
            else:
                self.logger.warning(
                    "Execution cancellation failed",
                    execution_id=execution_id,
                    reason=reason,
                )

            return result

        except Exception as e:
            self.logger.error(
                "Error cancelling execution",
                execution_id=execution_id,
                error=str(e),
            )
            return False

    async def get_active_executions(self) -> dict[str, Any]:
        """
        Get all active executions.
        
        Returns:
            Dict containing active executions
        """
        try:
            return await self.execution_engine_service.get_active_executions()
        except Exception as e:
            self.logger.error("Failed to get active executions", error=str(e))
            return {"error": str(e)}

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Returns:
            Dict containing health status
        """
        return {
            "service_name": self.name,
            "is_running": self.is_running,
            "dependencies": {
                "execution_service": bool(self.execution_service),
                "order_management_service": bool(self.order_management_service),
                "execution_engine_service": bool(self.execution_engine_service),
                "risk_validation_service": bool(self.risk_validation_service),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
