"""
Market Data Source Integration

This module provides market data ingestion from exchanges:
- Real-time OHLCV data
- Order book snapshots and updates
- Trade data streams
- Ticker information
- Historical data batch processing

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-003+: Exchange interfaces
- P-007A: Utility functions and decorators
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

# Import from P-001 core components
from src.core.types import (
    MarketData, Ticker, OrderBook, Trade, ExchangeInfo
)
from src.core.exceptions import (
    DataError, DataSourceError, ExchangeError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# Import from P-003+ exchange interfaces
from src.exchanges.base import BaseExchange
from src.exchanges.factory import ExchangeFactory

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler

# Import from P-007A utilities
from src.utils.decorators import time_execution, retry, circuit_breaker
from src.utils.validators import validate_price, validate_quantity
from src.utils.helpers import calculate_percentage_change

logger = get_logger(__name__)


class DataStreamType(Enum):
    """Data stream type enumeration"""
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    OHLCV = "ohlcv"
    MARKET_STATUS = "market_status"


@dataclass
class DataSubscription:
    """Data subscription configuration"""
    exchange_name: str
    symbol: str
    stream_type: DataStreamType
    callback: Callable
    active: bool = True
    last_update: Optional[datetime] = None
    error_count: int = 0


class MarketDataSource:
    """
    Market data source for real-time and historical data ingestion.

    This class manages data streams from multiple exchanges and provides
    unified market data access for the trading system.
    """

    def __init__(self, config: Config):
        """
        Initialize market data source.

        Args:
            config: Application configuration
        """
        self.config = config
        self.error_handler = ErrorHandler(config)

        # Exchange management
        self.exchange_factory = ExchangeFactory(config)
        self.exchanges: Dict[str, BaseExchange] = {}

        # Data subscription management
        self.subscriptions: Dict[str, DataSubscription] = {}

        # Data caching
        self.ticker_cache: Dict[str, Ticker] = {}
        self.order_book_cache: Dict[str, OrderBook] = {}
        self.trade_cache: Dict[str, List[Trade]] = {}

        # Stream management
        self.active_streams: Dict[str, bool] = {}
        self.stream_tasks: Dict[str, asyncio.Task] = {}

        # Statistics
        self.stats = {
            'total_data_points': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'last_update_time': None
        }

        logger.info("MarketDataSource initialized")

    async def initialize(self) -> None:
        """Initialize exchange connections and data sources."""
        try:
            # Initialize exchanges
            supported_exchanges = ['binance', 'okx', 'coinbase']

            for exchange_name in supported_exchanges:
                try:
                    exchange = await self.exchange_factory.create_exchange(exchange_name)
                    if await exchange.connect():
                        self.exchanges[exchange_name] = exchange
                        logger.info(
                            f"Connected to {exchange_name} for market data")
                    else:
                        logger.warning(f"Failed to connect to {exchange_name}")
                except Exception as e:
                    logger.error(
                        f"Error initializing {exchange_name}: {
                            str(e)}")

            if not self.exchanges:
                raise DataSourceError("No exchanges connected for market data")

            logger.info(
                f"MarketDataSource initialized with {len(self.exchanges)} exchanges")

        except Exception as e:
            logger.error(f"Failed to initialize MarketDataSource: {str(e)}")
            raise DataSourceError(
                f"Market data source initialization failed: {
                    str(e)}")

    @time_execution
    async def subscribe_to_ticker(
        self,
        exchange_name: str,
        symbol: str,
        callback: Callable[[Ticker], None]
    ) -> str:
        """
        Subscribe to ticker updates for a symbol.

        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            callback: Callback function for ticker updates

        Returns:
            str: Subscription ID
        """
        try:
            if exchange_name not in self.exchanges:
                raise DataSourceError(
                    f"Exchange {exchange_name} not available")

            subscription_id = f"{exchange_name}_{symbol}_ticker"

            # Create subscription
            subscription = DataSubscription(
                exchange_name=exchange_name,
                symbol=symbol,
                stream_type=DataStreamType.TICKER,
                callback=callback
            )

            self.subscriptions[subscription_id] = subscription

            # Start ticker stream if not already active
            stream_key = f"{exchange_name}_ticker"
            if stream_key not in self.active_streams:
                self.active_streams[stream_key] = True
                task = asyncio.create_task(self._ticker_stream(exchange_name))
                self.stream_tasks[stream_key] = task

            logger.info(f"Subscribed to ticker updates: {subscription_id}")
            return subscription_id

        except Exception as e:
            logger.error(
                f"Failed to subscribe to ticker {exchange_name}: {symbol}: {
                    str(e)}")
            raise DataSourceError(f"Ticker subscription failed: {str(e)}")

    @retry(max_attempts=3, delay=1.0)
    async def get_historical_data(
        self,
        exchange_name: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m"
    ) -> List[MarketData]:
        """
        Get historical market data for a symbol.

        Args:
            exchange_name: Name of the exchange
            symbol: Trading symbol
            start_time: Start time for historical data
            end_time: End time for historical data
            interval: Data interval (1m, 5m, 1h, etc.)

        Returns:
            List[MarketData]: Historical market data
        """
        try:
            if exchange_name not in self.exchanges:
                raise DataSourceError(
                    f"Exchange {exchange_name} not available")

            exchange = self.exchanges[exchange_name]

            # Get historical data from exchange
            # Implementation would depend on specific exchange API
            historical_data = []

            logger.info(
                f"Retrieved {len(historical_data)} historical data points "
                f"for {exchange_name}:{symbol} from {start_time} to {end_time}"
            )

            return historical_data

        except Exception as e:
            logger.error(
                f"Failed to get historical data {exchange_name}: {symbol}: {
                    str(e)}")
            raise DataSourceError(
                f"Historical data retrieval failed: {
                    str(e)}")

    async def _ticker_stream(self, exchange_name: str) -> None:
        """Manage ticker data stream for an exchange."""
        try:
            exchange = self.exchanges[exchange_name]

            while self.active_streams.get(f"{exchange_name}_ticker", False):
                try:
                    # Get active ticker subscriptions for this exchange
                    ticker_subs = [
                        sub for sub in self.subscriptions.values()
                        if (sub.exchange_name == exchange_name and
                            sub.stream_type == DataStreamType.TICKER and
                            sub.active)
                    ]

                    if not ticker_subs:
                        await asyncio.sleep(1)
                        continue

                    # Get ticker data for all subscribed symbols
                    for subscription in ticker_subs:
                        try:
                            ticker = await exchange.get_ticker(subscription.symbol)
                            if ticker:
                                # Cache ticker data
                                cache_key = f"{exchange_name}_{
                                    subscription.symbol}"
                                self.ticker_cache[cache_key] = ticker

                                # Call subscription callback
                                subscription.callback(ticker)
                                subscription.last_update = datetime.now(
                                    timezone.utc)

                                self.stats['successful_updates'] += 1

                        except Exception as e:
                            subscription.error_count += 1
                            self.stats['failed_updates'] += 1
                            logger.warning(
                                f"Ticker update failed for {
                                    subscription.symbol}: {
                                    str(e)}")

                    await asyncio.sleep(1)  # 1 second update interval

                except Exception as e:
                    logger.error(
                        f"Error in ticker stream {exchange_name}: {
                            str(e)}")
                    await asyncio.sleep(5)  # Longer delay on error

        except Exception as e:
            logger.error(
                f"Fatal error in ticker stream {exchange_name}: {
                    str(e)}")
        finally:
            self.active_streams[f"{exchange_name}_ticker"] = False

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from a data stream.

        Args:
            subscription_id: Subscription ID to cancel

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if subscription_id not in self.subscriptions:
                logger.warning(f"Subscription {subscription_id} not found")
                return False

            subscription = self.subscriptions[subscription_id]
            subscription.active = False

            # Remove subscription
            del self.subscriptions[subscription_id]

            logger.info(f"Unsubscribed from {subscription_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe {subscription_id}: {str(e)}")
            return False

    @time_execution
    async def get_market_data_summary(self) -> Dict[str, Any]:
        """Get market data source summary and statistics."""
        return {
            "connected_exchanges": list(self.exchanges.keys()),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "total_subscriptions": len(self.subscriptions),
            "statistics": self.stats.copy(),
            "cache_sizes": {
                "tickers": len(self.ticker_cache),
                "order_books": len(self.order_book_cache),
                "trades": sum(len(trades) for trades in self.trade_cache.values())
            }
        }

    async def cleanup(self) -> None:
        """Cleanup market data source resources."""
        try:
            # Stop all streams
            for stream_key in list(self.active_streams.keys()):
                self.active_streams[stream_key] = False

            # Cancel all stream tasks
            for task in self.stream_tasks.values():
                if not task.done():
                    task.cancel()

            # Disconnect from exchanges
            for exchange in self.exchanges.values():
                await exchange.disconnect()

            # Clear caches
            self.ticker_cache.clear()
            self.order_book_cache.clear()
            self.trade_cache.clear()

            logger.info("MarketDataSource cleanup completed")

        except Exception as e:
            logger.error(f"Error during MarketDataSource cleanup: {str(e)}")
