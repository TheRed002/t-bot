"""
API Facade for T-Bot Trading System.

This module provides a unified interface to all trading system services.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError
from src.utils.decimal_utils import to_decimal
from src.core.types import (
    BotConfiguration,
    OrderSide,
    OrderType,
    Position,
)
from src.web_interface.interfaces import (
    WebBotServiceInterface,
    WebPortfolioServiceInterface,
    WebRiskServiceInterface,
    WebStrategyServiceInterface,
    WebTradingServiceInterface,
)


class APIFacade(BaseComponent):
    """
    Unified API facade for T-Bot Trading System.

    This class provides a single entry point for all trading system operations.
    """

    def __init__(
        self,
        service_registry=None,
        injector=None,
        trading_service: WebTradingServiceInterface = None,
        bot_service: WebBotServiceInterface = None,
        portfolio_service: WebPortfolioServiceInterface = None,
        risk_service: WebRiskServiceInterface = None,
        strategy_service: WebStrategyServiceInterface = None,
    ):
        super().__init__()
        self._service_registry = service_registry
        self._injector = injector
        self._trading_service = trading_service
        self._bot_service = bot_service
        self._portfolio_service = portfolio_service
        self._risk_service = risk_service
        self._strategy_service = strategy_service
        self._initialized = False

    def configure_dependencies(self, injector):
        """Configure dependencies using the injector."""
        self._injector = injector

        # Resolve services from injector if not provided
        if not self._trading_service and injector.has_service("WebTradingService"):
            self._trading_service = injector.resolve("WebTradingService")

        if not self._bot_service and injector.has_service("WebBotService"):
            self._bot_service = injector.resolve("WebBotService")

        if not self._portfolio_service and injector.has_service("WebPortfolioService"):
            self._portfolio_service = injector.resolve("WebPortfolioService")

        if not self._risk_service and injector.has_service("WebRiskService"):
            self._risk_service = injector.resolve("WebRiskService")

        if not self._strategy_service and injector.has_service("WebStrategyService"):
            self._strategy_service = injector.resolve("WebStrategyService")

    async def initialize(self) -> None:
        """Initialize the API facade."""
        if self._initialized:
            return

        # Initialize all services
        services = [
            self._trading_service,
            self._bot_service,
            self._portfolio_service,
            self._risk_service,
            self._strategy_service,
        ]

        for service in services:
            if service and hasattr(service, "initialize"):
                await service.initialize()

        self._initialized = True
        self.logger.info("API facade initialized")

    async def cleanup(self) -> None:
        """Cleanup the API facade."""
        if not self._initialized:
            return

        # Cleanup all services
        services = [
            self._trading_service,
            self._bot_service,
            self._portfolio_service,
            self._risk_service,
            self._strategy_service,
        ]

        for service in services:
            if service and hasattr(service, "cleanup"):
                await service.cleanup()

        self._initialized = False
        self.logger.info("API facade cleaned up")

    # Trading Operations
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Decimal,
        price: Decimal | None = None,
    ) -> str:
        """Place a trading order through the trading service."""
        if not self._trading_service:
            from src.core.exceptions import ServiceError
            raise ServiceError("Trading service not available", component="api_facade")

        result = await self._trading_service.place_order_through_service(
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            quantity=amount,
            price=price,
        )
        return result.get("order_id", "")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order through the trading service."""
        if not self._trading_service:
            from src.core.exceptions import ServiceError
            raise ServiceError("Trading service not available", component="api_facade")

        return await self._trading_service.cancel_order_through_service(order_id)

    async def get_positions(self) -> list[Position]:
        """Get current positions through the trading service."""
        if not self._trading_service:
            from src.core.exceptions import ServiceError
            raise ServiceError("Trading service not available", component="api_facade")

        # Trading service returns dict, need to convert to Position objects
        positions_data = await self._trading_service.get_positions_through_service()
        return positions_data  # Simplified - return as-is for now

    # Bot Management Operations
    async def create_bot(self, config: BotConfiguration) -> str:
        """Create a new trading bot."""
        if not self._bot_service:
            raise ServiceError("Bot service not available")
        return await self._bot_service.create_bot_through_service(config)

    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        if not self._bot_service:
            raise ServiceError("Bot service not available")
        return await self._bot_service.start_bot_through_service(bot_id)

    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        if not self._bot_service:
            raise ServiceError("Bot service not available")
        return await self._bot_service.stop_bot_through_service(bot_id)

    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot status."""
        if not self._bot_service:
            raise ServiceError("Bot service not available")
        return await self._bot_service.get_bot_status_through_service(bot_id)

    async def list_bots(self) -> list[dict[str, Any]]:
        """List all bots."""
        if not self._bot_service:
            raise ServiceError("Bot service not available")
        return await self._bot_service.list_bots_through_service()

    # Portfolio Operations
    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        if not self._portfolio_service:
            raise ServiceError("Portfolio service not available")
        balances = self._portfolio_service.generate_mock_balances()
        return {item["asset"]: to_decimal(item["balance"]) for item in balances}

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        if not self._portfolio_service:
            raise ServiceError("Portfolio service not available")
        return await self._portfolio_service.get_portfolio_summary_data()

    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Get P&L report for date range."""
        if not self._portfolio_service:
            raise ServiceError("Portfolio service not available")
        period = f"{(end_date - start_date).days}d"
        return await self._portfolio_service.calculate_pnl_metrics(period)

    # Risk Management Operations
    async def validate_order(
        self, symbol: str, side: OrderSide, amount: Decimal, price: Decimal | None = None
    ) -> bool:
        """Validate an order against risk rules."""
        if not self._risk_service:
            raise ServiceError("Risk service not available")

        return await self._risk_service.validate_order_through_service(
            symbol=symbol, side=side.value, amount=amount, price=price
        )

    async def get_risk_summary(self) -> dict[str, Any]:
        """Get comprehensive risk summary."""
        if not self._risk_service:
            raise ServiceError("Risk service not available")
        return await self._risk_service.get_risk_summary_data()

    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        return await self.get_risk_summary()

    async def calculate_position_size(
        self,
        signal,
        available_capital: Decimal,
        current_price: Decimal,
        method: str | None = None,
    ) -> Decimal:
        """Calculate optimal position size."""
        if not self._risk_service:
            raise ServiceError("Risk service not available")
        return await self._risk_service.calculate_position_size_through_service(
            signal=signal,
            available_capital=available_capital,
            current_price=current_price,
            method=method,
        )

    # Strategy Operations
    async def list_strategies(self) -> list[dict[str, Any]]:
        """List available strategies."""
        if not self._strategy_service:
            raise ServiceError("Strategy service not available")
        return await self._strategy_service.get_formatted_strategies()

    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy configuration."""
        if not self._strategy_service:
            raise ServiceError("Strategy service not available")
        return await self._strategy_service.get_strategy_config_through_service(strategy_name)

    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool:
        """Validate strategy configuration."""
        if not self._strategy_service:
            raise ServiceError("Strategy service not available")
        return await self._strategy_service.validate_strategy_config_through_service(
            strategy_name, config
        )

    # Utility Methods
    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on all services."""
        services = {
            "trading": self._trading_service is not None,
            "bot_management": self._bot_service is not None,
            "portfolio": self._portfolio_service is not None,
            "risk_management": self._risk_service is not None,
            "strategy": self._strategy_service is not None,
        }

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "services": services,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_service_status(self, service_name: str) -> dict[str, Any]:
        """Get status of a specific service."""
        service_map = {
            "trading": self._trading_service,
            "bot_management": self._bot_service,
            "portfolio": self._portfolio_service,
            "risk_management": self._risk_service,
            "strategy": self._strategy_service,
        }

        service = service_map.get(service_name)
        if service:
            return {
                "name": service_name,
                "type": type(service).__name__,
                "available": True,
                "initialized": self._initialized,
            }
        else:
            return {"name": service_name, "available": False, "error": "Service not found"}

    async def delete_bot(self, bot_id: str, force: bool = False) -> bool:
        """Delete a bot through the bot management service."""
        if not self._bot_service:
            raise ServiceError("Bot service not available")
        return await self._bot_service.delete_bot_through_service(bot_id, force)


# Global facade instance
_global_facade: APIFacade | None = None


def get_api_facade(
    trading_service: WebTradingServiceInterface = None,
    bot_service: WebBotServiceInterface = None,
    portfolio_service: WebPortfolioServiceInterface = None,
    risk_service: WebRiskServiceInterface = None,
    strategy_service: WebStrategyServiceInterface = None,
) -> APIFacade:
    """Get or create the global API facade with services."""
    global _global_facade
    if _global_facade is None:
        _global_facade = APIFacade(
            trading_service=trading_service,
            bot_service=bot_service,
            portfolio_service=portfolio_service,
            risk_service=risk_service,
            strategy_service=strategy_service,
        )
    return _global_facade
