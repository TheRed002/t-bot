"""
Strategy Controller - API layer for strategy operations.

This controller handles HTTP requests and delegates to the service layer.
Controllers should ONLY handle request/response logic, never business logic.
"""

from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import MarketData, StrategyConfig
from src.strategies.interfaces import StrategyServiceInterface


class StrategyController(BaseComponent):
    """
    Controller for strategy operations.

    Handles HTTP requests and delegates to StrategyService.
    Contains NO business logic - only request/response handling.
    """

    def __init__(self, strategy_service: StrategyServiceInterface):
        """Initialize controller with strategy service."""
        super().__init__()
        self._strategy_service = strategy_service

    async def register_strategy(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """
        Handle strategy registration request.

        Args:
            request_data: Request payload

        Returns:
            Response data
        """
        try:
            # Validate request data
            if not request_data.get("strategy_id"):
                raise ValidationError("strategy_id is required")

            if not request_data.get("config"):
                raise ValidationError("config is required")

            # Extract data
            strategy_id = request_data["strategy_id"]
            config = StrategyConfig(**request_data["config"])
            strategy_instance = request_data.get("strategy_instance")

            # Delegate to service - NO business logic here
            await self._strategy_service.register_strategy(strategy_id, strategy_instance, config)

            return {
                "success": True,
                "message": f"Strategy {strategy_id} registered successfully",
                "strategy_id": strategy_id,
            }

        except (ValidationError, ServiceError) as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def start_strategy(self, strategy_id: str) -> dict[str, Any]:
        """
        Handle strategy start request.

        Args:
            strategy_id: Strategy to start

        Returns:
            Response data
        """
        try:
            if not strategy_id:
                raise ValidationError("strategy_id is required")

            # Delegate to service
            await self._strategy_service.start_strategy(strategy_id)

            return {
                "success": True,
                "message": f"Strategy {strategy_id} started successfully",
                "strategy_id": strategy_id,
            }

        except (ValidationError, ServiceError) as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def stop_strategy(self, strategy_id: str) -> dict[str, Any]:
        """
        Handle strategy stop request.

        Args:
            strategy_id: Strategy to stop

        Returns:
            Response data
        """
        try:
            if not strategy_id:
                raise ValidationError("strategy_id is required")

            # Delegate to service
            await self._strategy_service.stop_strategy(strategy_id)

            return {
                "success": True,
                "message": f"Strategy {strategy_id} stopped successfully",
                "strategy_id": strategy_id,
            }

        except (ValidationError, ServiceError) as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def process_market_data(self, market_data_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Handle market data processing request.

        Args:
            market_data_dict: Market data as dictionary

        Returns:
            Response with generated signals
        """
        try:
            if not market_data_dict:
                raise ValidationError("market_data is required")

            # Convert to MarketData object
            market_data = MarketData(**market_data_dict)

            # Delegate to service
            signals = await self._strategy_service.process_market_data(market_data)

            return {
                "success": True,
                "signals": signals,
                "processed_at": market_data.timestamp.isoformat(),
                "symbol": market_data.symbol,
            }

        except (ValidationError, ServiceError) as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]:
        """
        Handle strategy performance request.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Performance data response
        """
        try:
            if not strategy_id:
                raise ValidationError("strategy_id is required")

            # Delegate to service
            performance = await self._strategy_service.get_strategy_performance(strategy_id)

            return {"success": True, "performance": performance, "strategy_id": strategy_id}

        except (ValidationError, ServiceError) as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def get_all_strategies(self) -> dict[str, Any]:
        """
        Handle request for all strategies.

        Returns:
            All strategies data response
        """
        try:
            # Delegate to service
            strategies = await self._strategy_service.get_all_strategies()

            return {"success": True, "strategies": strategies, "count": len(strategies)}

        except ServiceError as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def cleanup_strategy(self, strategy_id: str) -> dict[str, Any]:
        """
        Handle strategy cleanup request.

        Args:
            strategy_id: Strategy to cleanup

        Returns:
            Response data
        """
        try:
            if not strategy_id:
                raise ValidationError("strategy_id is required")

            # Delegate to service
            await self._strategy_service.cleanup_strategy(strategy_id)

            return {
                "success": True,
                "message": f"Strategy {strategy_id} cleaned up successfully",
                "strategy_id": strategy_id,
            }

        except (ValidationError, ServiceError) as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
