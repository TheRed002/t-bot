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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.config import Config
from src.core.exceptions import DataSourceError
from src.core.logging import get_logger

# Import from P-001 core components
from src.core.types import MarketData, OrderBook, Ticker, Trade

# Import from P-002A error handling
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery, APIRateLimitRecovery
from src.error_handling.connection_manager import ConnectionManager
from src.error_handling.pattern_analytics import ErrorPatternAnalytics

# Import from P-003+ exchange interfaces
from src.exchanges.base import BaseExchange
from src.exchanges.factory import ExchangeFactory

# Import from P-007A utilities
from src.utils.decorators import retry, time_execution

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
    last_update: datetime | None = None
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

        # Initialize error handling components
        self.error_handler = ErrorHandler(config)
        self.network_recovery = NetworkDisconnectionRecovery(config)
        self.api_rate_limit_recovery = APIRateLimitRecovery(config)
        self.connection_manager = ConnectionManager(config)
        self.pattern_analytics = ErrorPatternAnalytics(config)

        # Exchange management
        self.exchange_factory = ExchangeFactory(config)
        self.exchanges: dict[str, BaseExchange] = {}

        # Data subscription management
        self.subscriptions: dict[str, DataSubscription] = {}

        # Data caching
        self.ticker_cache: dict[str, Ticker] = {}
        self.order_book_cache: dict[str, OrderBook] = {}
        self.trade_cache: dict[str, list[Trade]] = {}

        # Stream management
        self.active_streams: dict[str, bool] = {}
        self.stream_tasks: dict[str, asyncio.Task] = {}

        # Statistics tracking
        self.stats = {
            "successful_updates": 0,
            "failed_updates": 0,
            "total_subscriptions": 0,
            "active_subscriptions": 0,
        }

        logger.info("MarketDataSource initialized")

    async def initialize(self) -> None:
        """Initialize exchange connections and data sources."""
        try:
            # Initialize exchanges
            supported_exchanges = ["binance", "okx", "coinbase"]

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
                    logger.error(f"Error initializing {exchange_name}: {e!s}")

            if not self.exchanges:
                raise DataSourceError("No exchanges connected for market data")

            logger.info(
                f"MarketDataSource initialized with {len(self.exchanges)} exchanges")

        except Exception as e:
            logger.error(f"Failed to initialize MarketDataSource: {e!s}")
            raise DataSourceError(
                f"Market data source initialization failed: {e!s}")

    @time_execution
    async def subscribe_to_ticker(
        self, exchange_name: str, symbol: str, callback: Callable[[Ticker], None]
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
                callback=callback,
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
                f"Failed to subscribe to ticker {exchange_name}: {symbol}: {e!s}")
            raise DataSourceError(f"Ticker subscription failed: {e!s}")

    @retry(max_attempts=3, base_delay=1.0)
    async def get_historical_data(
        self,
        exchange_name: str,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        interval: str = "1m",
    ) -> list[MarketData]:
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
            # Use ErrorHandler for comprehensive error management
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="get_historical_data",
                symbol=symbol,
                details={
                    "exchange_name": exchange_name,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "interval": interval
                }
            )

            # Handle the error through the error handling framework
            self.error_handler.handle_error(error_context)

            # Add error event to pattern analytics
            self.pattern_analytics.add_error_event(error_context.__dict__)

            logger.error(
                f"Failed to get historical data {exchange_name}: {symbol}: {e!s}")
            raise DataSourceError(f"Historical data retrieval failed: {e!s}")

    async def _ticker_stream(self, exchange_name: str) -> None:
        """Manage ticker data stream for an exchange."""
        try:
            exchange = self.exchanges[exchange_name]

            while self.active_streams.get(f"{exchange_name}_ticker", False):
                try:
                    # Get active ticker subscriptions for this exchange
                    ticker_subs = [
                        sub
                        for sub in self.subscriptions.values()
                        if (
                            sub.exchange_name == exchange_name
                            and sub.stream_type == DataStreamType.TICKER
                            and sub.active
                        )
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
                                cache_key = f"{exchange_name}_{subscription.symbol}"
                                self.ticker_cache[cache_key] = ticker

                                # Call subscription callback
                                subscription.callback(ticker)
                                subscription.last_update = datetime.now(
                                    timezone.utc)

                                self.stats["successful_updates"] += 1

                        except Exception as e:
                            # Use ErrorHandler for ticker update errors
                            error_context = self.error_handler.create_error_context(
                                error=e,
                                component="MarketDataSource",
                                operation="ticker_update",
                                symbol=subscription.symbol,
                                details={
                                    "exchange_name": exchange_name,
                                    "subscription_id": subscription.exchange_name + "_" + subscription.symbol + "_ticker",
                                    "error_count": subscription.error_count
                                }
                            )

                            self.error_handler.handle_error(error_context)
                            self.pattern_analytics.add_error_event(error_context.__dict__)

                            subscription.error_count += 1
                            self.stats["failed_updates"] += 1
                            logger.warning(
                                f"Ticker update failed for {subscription.symbol}: {e!s}")

                except Exception as e:
                    # Use ErrorHandler for stream errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="MarketDataSource",
                        operation="ticker_stream",
                        details={
                            "exchange_name": exchange_name,
                            "stage": "stream_processing"
                        }
                    )

                    self.error_handler.handle_error(error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    logger.error(
                        f"Error in ticker stream {exchange_name}: {e!s}")
                    await asyncio.sleep(5)  # Longer delay on error

        except Exception as e:
            # Use ErrorHandler for fatal stream errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="ticker_stream",
                details={
                    "exchange_name": exchange_name,
                    "stage": "stream_management",
                    "severity": "fatal"
                }
            )

            self.error_handler.handle_error(error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            logger.error(
                f"Fatal error in ticker stream {exchange_name}: {e!s}")
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
            # Use ErrorHandler for unsubscribe errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="unsubscribe",
                details={
                    "subscription_id": subscription_id,
                    "stage": "unsubscription"
                }
            )

            self.error_handler.handle_error(error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            logger.error(f"Failed to unsubscribe {subscription_id}: {e!s}")
            return False

    @time_execution
    async def get_market_data_summary(self) -> dict[str, Any]:
        """Get market data source summary and statistics."""
        return {
            "connected_exchanges": list(self.exchanges.keys()),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "total_subscriptions": len(self.subscriptions),
            "statistics": self.stats.copy(),
            "cache_sizes": {
                "tickers": len(self.ticker_cache),
                "order_books": len(self.order_book_cache),
                "trades": sum(len(trades) for trades in self.trade_cache.values()),
            },
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
            # Use ErrorHandler for cleanup errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="cleanup",
                details={
                    "operation": "cleanup",
                    "active_streams": list(self.active_streams.keys()),
                    "active_tasks": list(self.stream_tasks.keys())
                }
            )

            self.error_handler.handle_error(error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            logger.error(f"Error during MarketDataSource cleanup: {e!s}")

    async def get_error_analytics(self) -> dict[str, Any]:
        """Get error pattern analytics, connection status, and circuit breaker status."""
        try:
            # Get error pattern summary from analytics
            pattern_summary = self.pattern_analytics.get_pattern_summary()
            correlation_summary = self.pattern_analytics.get_correlation_summary()
            trend_summary = self.pattern_analytics.get_trend_summary()

            # Get connection status
            connection_status = self.connection_manager.get_all_connection_status()

            # Get circuit breaker status
            circuit_breaker_status = self.error_handler.get_circuit_breaker_status()

            return {
                "error_patterns": pattern_summary,
                "error_correlations": correlation_summary,
                "error_trends": trend_summary,
                "connection_status": connection_status,
                "circuit_breaker_status": circuit_breaker_status,
                "stats": self.stats.copy(),
                "active_streams": list(self.active_streams.keys()),
                "active_tasks": list(self.stream_tasks.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            # Use ErrorHandler for analytics retrieval errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="get_error_analytics",
                details={
                    "operation": "analytics_retrieval"
                }
            )

            self.error_handler.handle_error(error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            logger.error(f"Failed to get error analytics: {e!s}")
            return {
                "error": str(e),
                "error_id": error_context.error_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
