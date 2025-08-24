"""
API Facade for T-Bot Trading System.

This module provides a unified interface to all trading system services,
abstracting away the complexity of the underlying implementations.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from src.base import BaseComponent
from src.core.types import BotConfiguration, MarketData, OrderSide, OrderType, Position

from .service_registry import ServiceInterface, get_service_registry


class TradingService(ServiceInterface):
    """Interface for trading operations."""

    async def initialize(self) -> None:
        """Initialize trading service."""
        pass

    async def cleanup(self) -> None:
        """Cleanup trading service."""
        pass

    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        amount: Decimal,
        price: Decimal | None = None,
    ) -> str:
        """Place a trading order."""
        raise NotImplementedError

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError

    async def get_positions(self) -> list[Position]:
        """Get current positions."""
        raise NotImplementedError


class BotManagementService(ServiceInterface):
    """Interface for bot management operations."""

    async def initialize(self) -> None:
        """Initialize bot management service."""
        pass

    async def cleanup(self) -> None:
        """Cleanup bot management service."""
        pass

    async def create_bot(self, config: BotConfiguration) -> str:
        """Create a new trading bot."""
        raise NotImplementedError

    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        raise NotImplementedError

    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        raise NotImplementedError

    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot status."""
        raise NotImplementedError

    async def list_bots(self) -> list[dict[str, Any]]:
        """List all bots."""
        raise NotImplementedError

    async def get_all_bots_status(self) -> dict[str, Any]:
        """Get status of all bots."""
        raise NotImplementedError

    async def delete_bot(self, bot_id: str, force: bool = False) -> bool:
        """Delete a bot."""
        raise NotImplementedError

    async def _service_health_check(self) -> dict[str, Any]:
        """Perform service health check."""
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        raise NotImplementedError


class MarketDataService(ServiceInterface):
    """Interface for market data operations."""

    async def initialize(self) -> None:
        """Initialize market data service."""
        pass

    async def cleanup(self) -> None:
        """Cleanup market data service."""
        pass

    async def get_ticker(self, symbol: str) -> MarketData:
        """Get current ticker data."""
        raise NotImplementedError

    async def subscribe_to_ticker(self, symbol: str, callback: callable) -> None:
        """Subscribe to ticker updates."""
        raise NotImplementedError

    async def unsubscribe_from_ticker(self, symbol: str) -> None:
        """Unsubscribe from ticker updates."""
        raise NotImplementedError


class PortfolioService(ServiceInterface):
    """Interface for portfolio operations."""

    async def initialize(self) -> None:
        """Initialize portfolio service."""
        pass

    async def cleanup(self) -> None:
        """Cleanup portfolio service."""
        pass

    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        raise NotImplementedError

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        raise NotImplementedError

    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Get P&L report for date range."""
        raise NotImplementedError


class RiskManagementService(ServiceInterface):
    """Interface for risk management operations."""

    async def initialize(self) -> None:
        """Initialize risk management service."""
        pass

    async def cleanup(self) -> None:
        """Cleanup risk management service."""
        pass

    async def validate_order(
        self, symbol: str, side: OrderSide, amount: Decimal, price: Decimal | None = None
    ) -> dict[str, Any]:
        """Validate an order against risk rules."""
        raise NotImplementedError

    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        raise NotImplementedError

    async def update_risk_limits(self, limits: dict[str, Any]) -> bool:
        """Update risk limits."""
        raise NotImplementedError


class StrategyService(ServiceInterface):
    """Interface for strategy operations."""

    async def initialize(self) -> None:
        """Initialize strategy service."""
        pass

    async def cleanup(self) -> None:
        """Cleanup strategy service."""
        pass

    async def list_strategies(self) -> list[dict[str, Any]]:
        """List available strategies."""
        raise NotImplementedError

    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy configuration."""
        raise NotImplementedError

    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool:
        """Validate strategy configuration."""
        raise NotImplementedError


class APIFacade(BaseComponent):
    """
    Unified API facade for T-Bot Trading System.

    This class provides a single entry point for all trading system operations,
    abstracting away the complexity of the underlying services.
    """

    def __init__(self):
        super().__init__()
        self._registry = get_service_registry()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the API facade."""
        if self._initialized:
            return

        # Initialize all registered services
        await self._registry.initialize_all()
        self._initialized = True
        self.logger.info("API facade initialized")

    async def cleanup(self) -> None:
        """Cleanup the API facade."""
        if not self._initialized:
            return

        await self._registry.cleanup_all()
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
        trading_service: TradingService = self._registry.get_service("trading")
        return await trading_service.place_order(symbol, side, order_type, amount, price)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order through the trading service."""
        trading_service: TradingService = self._registry.get_service("trading")
        return await trading_service.cancel_order(order_id)

    async def get_positions(self) -> list[Position]:
        """Get current positions through the trading service."""
        trading_service: TradingService = self._registry.get_service("trading")
        return await trading_service.get_positions()

    # Bot Management Operations
    async def create_bot(self, config: BotConfiguration) -> str:
        """Create a new trading bot."""
        bot_service: BotManagementService = self._registry.get_service("bot_management")
        return await bot_service.create_bot(config)

    async def start_bot(self, bot_id: str) -> bool:
        """Start a bot."""
        bot_service: BotManagementService = self._registry.get_service("bot_management")
        return await bot_service.start_bot(bot_id)

    async def stop_bot(self, bot_id: str) -> bool:
        """Stop a bot."""
        bot_service: BotManagementService = self._registry.get_service("bot_management")
        return await bot_service.stop_bot(bot_id)

    async def get_bot_status(self, bot_id: str) -> dict[str, Any]:
        """Get bot status."""
        bot_service: BotManagementService = self._registry.get_service("bot_management")
        return await bot_service.get_bot_status(bot_id)

    async def list_bots(self) -> list[dict[str, Any]]:
        """List all bots."""
        bot_service: BotManagementService = self._registry.get_service("bot_management")
        return await bot_service.list_bots()

    # Market Data Operations
    async def get_ticker(self, symbol: str) -> MarketData:
        """Get current ticker data."""
        market_service: MarketDataService = self._registry.get_service("market_data")
        return await market_service.get_ticker(symbol)

    async def subscribe_to_ticker(self, symbol: str, callback: callable) -> None:
        """Subscribe to ticker updates."""
        market_service: MarketDataService = self._registry.get_service("market_data")
        await market_service.subscribe_to_ticker(symbol, callback)

    async def unsubscribe_from_ticker(self, symbol: str) -> None:
        """Unsubscribe from ticker updates."""
        market_service: MarketDataService = self._registry.get_service("market_data")
        await market_service.unsubscribe_from_ticker(symbol)

    # Portfolio Operations
    async def get_balance(self) -> dict[str, Decimal]:
        """Get account balances."""
        portfolio_service: PortfolioService = self._registry.get_service("portfolio")
        return await portfolio_service.get_balance()

    async def get_portfolio_summary(self) -> dict[str, Any]:
        """Get portfolio summary."""
        portfolio_service: PortfolioService = self._registry.get_service("portfolio")
        return await portfolio_service.get_portfolio_summary()

    async def get_pnl_report(self, start_date: datetime, end_date: datetime) -> dict[str, Any]:
        """Get P&L report for date range."""
        portfolio_service: PortfolioService = self._registry.get_service("portfolio")
        return await portfolio_service.get_pnl_report(start_date, end_date)

    # Risk Management Operations
    async def validate_order(
        self, symbol: str, side: OrderSide, amount: Decimal, price: Decimal | None = None
    ) -> dict[str, Any]:
        """Validate an order against risk rules."""
        risk_service: RiskManagementService = self._registry.get_service("risk_management")
        return await risk_service.validate_order(symbol, side, amount, price)

    async def get_risk_metrics(self) -> dict[str, Any]:
        """Get current risk metrics."""
        risk_service: RiskManagementService = self._registry.get_service("risk_management")
        return await risk_service.get_risk_metrics()

    async def update_risk_limits(self, limits: dict[str, Any]) -> bool:
        """Update risk limits."""
        risk_service: RiskManagementService = self._registry.get_service("risk_management")
        return await risk_service.update_risk_limits(limits)

    # Strategy Operations
    async def list_strategies(self) -> list[dict[str, Any]]:
        """List available strategies."""
        strategy_service: StrategyService = self._registry.get_service("strategies")
        return await strategy_service.list_strategies()

    async def get_strategy_config(self, strategy_name: str) -> dict[str, Any]:
        """Get strategy configuration."""
        strategy_service: StrategyService = self._registry.get_service("strategies")
        return await strategy_service.get_strategy_config(strategy_name)

    async def validate_strategy_config(self, strategy_name: str, config: dict[str, Any]) -> bool:
        """Validate strategy configuration."""
        strategy_service: StrategyService = self._registry.get_service("strategies")
        return await strategy_service.validate_strategy_config(strategy_name, config)

    # Utility Methods
    def health_check(self) -> dict[str, Any]:
        """Perform a health check on all services."""
        services = self._registry.list_services()
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "services": services,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_service_status(self, service_name: str) -> dict[str, Any]:
        """Get status of a specific service."""
        try:
            service = self._registry.get_service(service_name)
            return {
                "name": service_name,
                "type": type(service).__name__,
                "available": True,
                "initialized": service_name in self._registry._initialized,
            }
        except KeyError:
            return {"name": service_name, "available": False, "error": "Service not found"}


# Global facade instance
_global_facade: APIFacade | None = None


def get_api_facade() -> APIFacade:
    """Get or create the global API facade."""
    global _global_facade
    if _global_facade is None:
        _global_facade = APIFacade()
    return _global_facade
