"""
Database manager for coordinating database operations through service layer.

This module provides a high-level coordinator for database operations,
delegating to appropriate business services while maintaining transaction boundaries.
Follows the controller -> service -> repository pattern strictly.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError

if TYPE_CHECKING:
    pass


class DatabaseManager(BaseComponent):
    """
    Database operations coordinator that enforces service layer pattern.

    This class coordinates database operations by delegating to business services,
    not infrastructure services. Controllers should use this manager to access
    business logic through proper service abstraction.
    """

    def __init__(self, trading_service=None, market_data_service=None) -> None:
        """
        Initialize the database manager with injected business services.

        Args:
            trading_service: Injected TradingService for trading operations
            market_data_service: Injected MarketDataService for market data operations
        """
        super().__init__()

        # Business services - these encapsulate business logic
        self.trading_service = trading_service
        self.market_data_service = market_data_service

    async def get_historical_data(
        self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str = "1m"
    ) -> list[dict[str, Any]]:
        """
        Coordinate retrieval of historical market data through business service.

        Args:
            symbol: Trading symbol
            start_time: Start of the time range
            end_time: End of the time range
            timeframe: Data timeframe

        Returns:
            List of market data records as dictionaries
        """
        try:
            # Delegate to business service - no direct database access
            if not self.market_data_service:
                raise ServiceError("MarketDataService not available")

            return await self.market_data_service.get_historical_data(
                symbol, start_time, end_time, timeframe
            )

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise ServiceError(f"Failed to retrieve historical data: {e}") from e

    async def save_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """
        Coordinate saving of trade data through business service.

        Args:
            trade_data: Trade information

        Returns:
            Created trade record as dictionary
        """
        try:
            # Delegate to business service - no direct database access
            if not self.trading_service:
                raise ServiceError("TradingService not available")

            return await self.trading_service.create_trade(trade_data)

        except Exception as e:
            self.logger.error(f"Failed to save trade: {e}")
            raise ServiceError(f"Failed to save trade: {e}") from e

    async def get_positions(
        self, strategy_id: str | None = None, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Coordinate retrieval of positions through business service.

        Args:
            strategy_id: Optional strategy filter
            symbol: Optional symbol filter

        Returns:
            List of positions as dictionaries
        """
        try:
            # Delegate to business service - no direct database access
            if not self.trading_service:
                raise ServiceError("TradingService not available")

            return await self.trading_service.get_positions(strategy_id, symbol)

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise ServiceError(f"Failed to retrieve positions: {e}") from e

    async def close(self):
        """Close database manager resources."""
        try:
            # Close business services if they have close methods
            if hasattr(self.trading_service, "stop"):
                await self.trading_service.stop()
            if hasattr(self.market_data_service, "stop"):
                await self.market_data_service.stop()
            self.logger.info("DatabaseManager closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing DatabaseManager: {e}")
            raise
