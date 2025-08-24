"""
Database manager for unified database operations.

This module provides a high-level interface for database operations,
abstracting the complexity of multiple database systems.
"""

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.base import BaseComponent
from src.database.connection import get_async_session
from src.database.models import MarketDataRecord, Position, Trade
from src.error_handling.decorators import with_circuit_breaker, with_fallback, with_retry
from src.utils.decorators import time_execution


class DatabaseManager(BaseComponent):
    """
    Unified database manager for all database operations.

    This class provides a high-level interface for interacting with
    PostgreSQL, Redis, and InfluxDB, abstracting the complexity of
    multiple database systems.
    """

    def __init__(self):
        """Initialize the database manager."""
        super().__init__()  # Initialize BaseComponent
        self.session: AsyncSession | None = None
        self._session_context = None

    @with_retry(max_attempts=3, base_delay=1.0)
    async def __aenter__(self):
        """Async context manager entry."""
        self._session_context = get_async_session()
        self.session = await self._session_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session_context:
            await self._session_context.__aexit__(exc_type, exc_val, exc_tb)
            self.session = None
            self._session_context = None

    @time_execution
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30.0)
    @with_retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,))
    @with_fallback(default_value=[])
    async def get_historical_data(
        self, symbol: str, start_time: datetime, end_time: datetime, timeframe: str = "1m"
    ) -> list[MarketDataRecord]:
        """
        Get historical market data for a symbol.

        Args:
            symbol: Trading symbol
            start_time: Start of the time range
            end_time: End of the time range
            timeframe: Data timeframe

        Returns:
            List of market data records
        """
        async with get_async_session() as session:
            # Simplified query - in production this would use proper SQLAlchemy queries
            records: list[MarketDataRecord] = []
            self.logger.info(f"Fetching historical data for {symbol} from {start_time} to {end_time}")

            # Placeholder implementation - replace with actual database query
            # In production, this would query the MarketDataRecord table
            return records

    @time_execution
    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @with_retry(max_attempts=5, base_delay=0.5, exceptions=(Exception,))
    async def save_trade(self, trade_data: dict[str, Any]) -> Trade:
        """
        Save a trade to the database.

        Args:
            trade_data: Trade information

        Returns:
            Created trade record
        """
        async with get_async_session() as session:
            trade = Trade(**trade_data)
            self.logger.info(f"Saving trade: {trade_data.get('order_id')}")

            # Placeholder implementation - replace with actual database save
            # In production, would do: session.add(trade); await session.commit()
            return trade

    @time_execution
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30.0)
    @with_retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,))
    @with_fallback(default_value=[])
    async def get_positions(
        self, strategy_id: str | None = None, symbol: str | None = None
    ) -> list[Position]:
        """
        Get positions from the database.

        Args:
            strategy_id: Optional strategy filter
            symbol: Optional symbol filter

        Returns:
            List of positions
        """
        async with get_async_session() as session:
            positions: list[Position] = []
            self.logger.info(f"Fetching positions for strategy={strategy_id}, symbol={symbol}")

            # Placeholder implementation - replace with actual database query
            return positions

    async def close(self):
        """Close the database connection."""
        if self.session:
            await self.session.close()
            self.session = None
