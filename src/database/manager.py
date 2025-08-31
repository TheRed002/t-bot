"""
Database manager for coordinating database service operations.

This module provides a high-level coordinator for database operations,
delegating to appropriate services while maintaining transaction boundaries.
"""

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError

if TYPE_CHECKING:
    pass  # Type imports handled at runtime
from src.error_handling.decorators import with_circuit_breaker, with_fallback, with_retry
from src.utils.decorators import time_execution


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
        self.database_service = database_service

    @time_execution
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"), exceptions=(ServiceError,))
    @with_fallback(default_value=[])
    async def get_historical_data(
        self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str = "1m"
    ) -> list[dict[str, Any]]:
        """
        Coordinate retrieval of historical market data through service layer.

        Args:
            symbol: Trading symbol
            start_time: Start of the time range
            end_time: End of the time range
            timeframe: Data timeframe

        Returns:
            List of market data records as dictionaries
        """
        try:
            # Use database service for data access - no business logic here
            from src.database.models.market_data import MarketData
            
            filters = {
                "symbol": symbol,
                "timestamp": {
                    "gte": start_time,
                    "lte": end_time
                },
                "timeframe": timeframe
            }
            
            market_data = await self.database_service.list_entities(
                model_class=MarketData,
                filters=filters,
                order_by="timestamp",
                order_desc=False
            )
            
            # Convert to dictionaries for API response
            return [
                {
                    "symbol": md.symbol,
                    "timestamp": md.timestamp,
                    "open": str(md.open),
                    "high": str(md.high),
                    "low": str(md.low),
                    "close": str(md.close),
                    "volume": str(md.volume),
                    "timeframe": md.timeframe
                }
                for md in market_data
            ]

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise ServiceError(f"Failed to retrieve historical data: {e}")

    @time_execution
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=5, base_delay=Decimal("0.5"), exceptions=(ServiceError,))
    @with_fallback(default_value={"id": None, "error": "Failed to save trade"})
    async def save_trade(self, trade_data: dict[str, Any]) -> dict[str, Any]:
        """
        Coordinate saving of trade data through database service.

        Args:
            trade_data: Trade information

        Returns:
            Created trade record as dictionary
        """
        try:
            # Create trade entity from data
            from src.database.models.trading import Trade
            
            trade = Trade(
                symbol=trade_data.get("symbol"),
                side=trade_data.get("side"),
                quantity=trade_data.get("quantity"),
                entry_price=trade_data.get("entry_price"),
                exit_price=trade_data.get("exit_price"),
                pnl=trade_data.get("pnl"),
                bot_id=trade_data.get("bot_id"),
                strategy_id=trade_data.get("strategy_id"),
                exchange=trade_data.get("exchange"),
            )
            
            # Save through database service
            saved_trade = await self.database_service.create_entity(trade)
            
            # Convert to dictionary for response
            result = {
                "id": saved_trade.id,
                "symbol": saved_trade.symbol,
                "side": saved_trade.side,
                "quantity": str(saved_trade.quantity) if saved_trade.quantity else None,
                "entry_price": str(saved_trade.entry_price) if saved_trade.entry_price else None,
                "exit_price": str(saved_trade.exit_price) if saved_trade.exit_price else None,
                "pnl": str(saved_trade.pnl) if saved_trade.pnl else None,
                "timestamp": saved_trade.created_at,
            }

            self.logger.info(f"Trade saved: {saved_trade.id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to save trade: {e}")
            raise ServiceError(f"Failed to save trade: {e}")

    @time_execution
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"), exceptions=(ServiceError,))
    @with_fallback(default_value=[])
    async def get_positions(self, strategy_id: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Coordinate retrieval of positions through database service.

        Args:
            strategy_id: Optional strategy filter
            symbol: Optional symbol filter

        Returns:
            List of positions as dictionaries
        """
        try:
            # Build filters for database query
            from src.database.models.trading import Position
            
            filters = {}
            if strategy_id:
                filters["strategy_id"] = strategy_id
            if symbol:
                filters["symbol"] = symbol
            
            # Get positions through database service
            positions = await self.database_service.list_entities(
                model_class=Position,
                filters=filters,
                order_by="created_at",
                order_desc=True
            )
            
            # Convert to dictionaries for API response
            return [
                {
                    "id": position.id,
                    "symbol": position.symbol,
                    "side": position.side,
                    "quantity": str(position.quantity) if position.quantity else None,
                    "entry_price": str(position.entry_price) if position.entry_price else None,
                    "current_price": str(position.current_price) if position.current_price else None,
                    "unrealized_pnl": str(position.unrealized_pnl) if position.unrealized_pnl else None,
                    "status": position.status,
                    "created_at": position.created_at,
                }
                for position in positions
            ]

        except Exception as e:
            self.logger.error(f"Failed to get positions: {e}")
            raise ServiceError(f"Failed to retrieve positions: {e}")

    async def close(self):
        """Close database manager resources."""
        try:
            if self.database_service:
                await self.database_service.stop()
                self.logger.info("DatabaseManager closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing DatabaseManager: {e}")
            raise
