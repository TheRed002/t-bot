"""
Database manager for coordinating database service operations.

This module provides a high-level coordinator for database operations,
delegating to appropriate services while maintaining transaction boundaries.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError

if TYPE_CHECKING:
    pass


class DatabaseManager(BaseComponent):
    """
    Database operations coordinator.

    This class coordinates database operations by delegating to the
    DatabaseService while providing higher-level orchestration.
    Follows the controller -> service -> repository pattern.
    """

    def __init__(self, database_service) -> None:  # DatabaseService - runtime import
        """
        Initialize the database manager with injected dependencies.

        Args:
            database_service: Injected DatabaseService instance for data operations
        """
        super().__init__()

        # Validate required dependencies
        if database_service is None:
            raise ValueError("database_service must be injected via dependency injection")

        self.database_service = database_service

    async def get_historical_data(
        self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str = "1m"
    ) -> list[dict[str, Any]]:
        """
        Coordinate retrieval of historical market data through service layer.
        Infrastructure only - data transformation moved to service layer.

        Args:
            symbol: Trading symbol
            start_time: Start of the time range
            end_time: End of the time range
            timeframe: Data timeframe

        Returns:
            List of market data records as dictionaries
        """
        try:
            # Delegate to service layer - no business logic here
            from src.database.models.market_data import MarketData

            filters = {
                "symbol": symbol,
                "timestamp": {"gte": start_time, "lte": end_time},
                "timeframe": timeframe,
            }

            # Return raw data - let calling service handle formatting
            market_data = await self.database_service.list_entities(
                model_class=MarketData, filters=filters, order_by="timestamp", order_desc=False
            )

            # Minimal infrastructure conversion only - detailed formatting belongs in service layer
            return [
                {
                    "id": md.id,
                    "symbol": md.symbol,
                    "timestamp": md.timestamp,
                    "data": md,  # Pass raw model - let service handle field extraction
                }
                for md in market_data
            ]

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise ServiceError(f"Failed to retrieve historical data: {e}") from e

    async def save_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """
        Coordinate saving of trade data through database service.
        Infrastructure only - entity creation and validation moved to service layer.

        Args:
            trade_data: Trade information

        Returns:
            Created trade record as dictionary
        """
        try:
            # Infrastructure coordination only - entity creation should be done in service layer
            from src.database.models.trading import Trade

            # Raw entity creation without business logic
            trade = Trade(**trade_data)

            # Save through database service
            saved_trade = await self.database_service.create_entity(trade)

            # Minimal infrastructure response - detailed formatting belongs in service layer
            result = {"id": saved_trade.id, "created": True, "entity": saved_trade}

            self.logger.info(f"Trade saved: {saved_trade.id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to save trade: {e}")
            raise ServiceError(f"Failed to save trade: {e}") from e

    async def get_positions(
        self, strategy_id: str | None = None, symbol: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Coordinate retrieval of positions through database service.
        Infrastructure only - data formatting moved to service layer.

        Args:
            strategy_id: Optional strategy filter
            symbol: Optional symbol filter

        Returns:
            List of positions as dictionaries
        """
        try:
            # Infrastructure coordination only
            from src.database.models.trading import Position

            filters = {}
            if strategy_id:
                filters["strategy_id"] = strategy_id
            if symbol:
                filters["symbol"] = symbol

            # Get positions through database service
            positions = await self.database_service.list_entities(
                model_class=Position, filters=filters, order_by="created_at", order_desc=True
            )

            # Minimal infrastructure response - detailed formatting belongs in service layer
            return [
                {
                    "id": position.id,
                    "entity": position,  # Pass raw entity - let service handle field extraction
                }
                for position in positions
            ]

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise ServiceError(f"Failed to retrieve positions: {e}") from e

    async def close(self):
        """Close database manager resources."""
        try:
            if self.database_service:
                await self.database_service.stop()
                self.logger.info("DatabaseManager closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing DatabaseManager: {e}")
            raise
