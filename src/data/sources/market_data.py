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

from src.core import BaseComponent
from src.core.config import Config
from src.core.exceptions import DataSourceError

# Import from P-001 core components
from src.core.types import MarketData, OrderBook, Ticker, Trade
from src.data.constants import (
    DEFAULT_RETRY_BASE_DELAY,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    EXCHANGE_CONNECTION_TIMEOUT_SECONDS,
    STREAM_ERROR_DELAY_SECONDS,
    STREAM_TASK_CLEANUP_TIMEOUT_SECONDS,
)

# Import from P-003+ exchange interfaces - abstracted through data interfaces
from src.data.interfaces import DataSourceInterface

# Import from P-002A error handling
from src.error_handling import (
    APIRateLimitRecovery,
    ErrorHandler,
    ErrorPatternAnalytics,
    NetworkDisconnectionRecovery,
    with_retry,
)
from src.error_handling.connection_manager import ConnectionManager

# Import from P-007A utilities
from src.utils.decorators import time_execution


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


class MarketDataSource(BaseComponent):
    """
    Market data source for real-time and historical data ingestion.

    This class manages data streams from multiple exchanges and provides
    unified market data access for the trading system.
    """

    def __init__(self, config: Config, exchange_factory=None):
        """
        Initialize market data source.

        Args:
            config: Application configuration
            exchange_factory: Injected exchange factory for dependency inversion
        """
        super().__init__()  # Initialize BaseComponent
        self.config = config
        self.error_handler = ErrorHandler(config)
        self.network_recovery = NetworkDisconnectionRecovery(config)
        self.api_rate_limit_recovery = APIRateLimitRecovery(config)
        self.connection_manager = ConnectionManager(config)
        self.pattern_analytics = ErrorPatternAnalytics(config)

        # Exchange factory for creating data sources
        self.exchange_factory = exchange_factory
        self.data_sources: dict[str, DataSourceInterface] = {}
        
        # Alias for backward compatibility with tests
        self.exchanges = self.data_sources

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

        self.logger.info("MarketDataSource initialized")

    async def initialize(self) -> None:
        """Initialize data source connections."""
        connected_sources = []
        try:
            # Initialize data sources through exchange factory if available
            if self.exchange_factory:
                # Get available exchanges from factory
                available_exchanges = getattr(self.exchange_factory, "get_available_exchanges", lambda: [])()
                for exchange_name in available_exchanges:
                    try:
                        # Create exchange instance through factory
                        exchange = self.exchange_factory.create_exchange(exchange_name)
                        if exchange:
                            self.data_sources[exchange_name] = exchange
                    except Exception as e:
                        self.logger.warning(f"Failed to create exchange {exchange_name}: {e}")

            # Initialize all data sources
            failed_sources = []
            for source_name, data_source in list(self.data_sources.items()):
                try:
                    # Add timeout to data source connection
                    await asyncio.wait_for(
                        data_source.connect(), timeout=EXCHANGE_CONNECTION_TIMEOUT_SECONDS
                    )
                    if data_source.is_connected():
                        connected_sources.append(source_name)
                        self.logger.info(f"Connected to {source_name} for market data")
                    else:
                        self.logger.warning(f"Failed to connect to {source_name}")
                        failed_sources.append(source_name)
                        # Clean up failed connection
                        try:
                            await data_source.disconnect()
                        except Exception as e:
                            self.logger.debug(
                                f"Error disconnecting {source_name} during cleanup: {e}"
                            )
                except asyncio.TimeoutError:
                    self.logger.error(f"Connection timeout for {source_name}")
                    failed_sources.append(source_name)
                    # Clean up timed out connection
                    try:
                        await data_source.disconnect()
                    except Exception as e:
                        self.logger.debug(f"Error disconnecting {source_name} after timeout: {e}")
                except Exception as e:
                    self.logger.error(f"Error initializing {source_name}: {e!s}")
                    failed_sources.append(source_name)
                    # Clean up failed connection
                    try:
                        await data_source.disconnect()
                    except Exception as cleanup_e:
                        self.logger.debug(
                            f"Error disconnecting {source_name} after failed init: {cleanup_e}"
                        )
            
            # Remove failed sources from data_sources
            for source_name in failed_sources:
                self.data_sources.pop(source_name, None)

            if not connected_sources:
                raise DataSourceError("No data sources connected for market data")

            self.logger.info(
                f"MarketDataSource initialized with {len(connected_sources)} data sources"
            )

        except Exception as e:
            # Clean up any partial connections on initialization failure
            for source_name in connected_sources:
                try:
                    if source_name in self.data_sources:
                        await self.data_sources[source_name].disconnect()
                except Exception as cleanup_e:
                    self.logger.debug(
                        f"Error disconnecting {source_name} during error cleanup: {cleanup_e}"
                    )
            self.logger.error(f"Failed to initialize MarketDataSource: {e!s}")
            raise DataSourceError(f"Market data source initialization failed: {e!s}") from e

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
            if exchange_name not in self.data_sources:
                raise DataSourceError(f"Data source {exchange_name} not available")

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

            self.logger.info(f"Subscribed to ticker updates: {subscription_id}")
            return subscription_id

        except Exception as e:
            self.logger.error(f"Failed to subscribe to ticker {exchange_name}: {symbol}: {e!s}")
            raise DataSourceError(f"Ticker subscription failed: {e!s}") from e

    @with_retry(max_attempts=DEFAULT_RETRY_MAX_ATTEMPTS, base_delay=DEFAULT_RETRY_BASE_DELAY, exponential=True)
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
            if exchange_name not in self.data_sources:
                raise DataSourceError(f"Data source {exchange_name} not available")

            self.data_sources[exchange_name]

            # Get historical data from exchange
            # Implementation would depend on specific exchange API
            historical_data = []

            self.logger.info(
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
                    "interval": interval,
                },
            )

            # Handle the error through the error handling framework
            await self.error_handler.handle_error(e, error_context)

            # Add error event to pattern analytics
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to get historical data {exchange_name}: {symbol}: {e!s}")
            raise DataSourceError(f"Historical data retrieval failed: {e!s}") from e

    async def _ticker_stream(self, exchange_name: str) -> None:
        """Manage ticker data stream for a data source."""
        try:
            data_source = self.data_sources[exchange_name]

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
                            # Use data source interface for ticker data
                            ticker_data = await data_source.fetch(
                                subscription.symbol, "ticker", limit=1
                            )
                            if ticker_data:
                                # Convert to Ticker object (implementation would depend on format)
                                ticker = ticker_data[0]  # For now, use first item
                            if ticker:
                                # Cache ticker data
                                cache_key = f"{exchange_name}_{subscription.symbol}"
                                self.ticker_cache[cache_key] = ticker

                                # Call subscription callback
                                subscription.callback(ticker)
                                subscription.last_update = datetime.now(timezone.utc)

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
                                    "subscription_id": subscription.exchange_name
                                    + "_"
                                    + subscription.symbol
                                    + "_ticker",
                                    "error_count": subscription.error_count,
                                },
                            )

                            await self.error_handler.handle_error(e, error_context)
                            self.pattern_analytics.add_error_event(error_context.__dict__)

                            subscription.error_count += 1
                            self.stats["failed_updates"] += 1
                            self.logger.warning(
                                f"Ticker update failed for {subscription.symbol}: {e!s}"
                            )

                except Exception as e:
                    # Use ErrorHandler for stream errors
                    error_context = self.error_handler.create_error_context(
                        error=e,
                        component="MarketDataSource",
                        operation="ticker_stream",
                        details={"exchange_name": exchange_name, "stage": "stream_processing"},
                    )

                    await self.error_handler.handle_error(e, error_context)
                    self.pattern_analytics.add_error_event(error_context.__dict__)

                    self.logger.error(f"Error in ticker stream {exchange_name}: {e!s}")
                    await asyncio.sleep(STREAM_ERROR_DELAY_SECONDS)  # Longer delay on error

        except Exception as e:
            # Use ErrorHandler for fatal stream errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="ticker_stream",
                details={
                    "exchange_name": exchange_name,
                    "stage": "stream_management",
                    "severity": "fatal",
                },
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Fatal error in ticker stream {exchange_name}: {e!s}")
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
                self.logger.warning(f"Subscription {subscription_id} not found")
                return False

            subscription = self.subscriptions[subscription_id]
            subscription.active = False

            # Remove subscription
            del self.subscriptions[subscription_id]

            self.logger.info(f"Unsubscribed from {subscription_id}")
            return True

        except Exception as e:
            # Use ErrorHandler for unsubscribe errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="unsubscribe",
                details={"subscription_id": subscription_id, "stage": "unsubscription"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to unsubscribe {subscription_id}: {e!s}")
            return False

    @time_execution
    async def get_market_data_summary(self) -> dict[str, Any]:
        """Get market data source summary and statistics."""
        return {
            "connected_data_sources": list(self.data_sources.keys()),
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
        stream_tasks = []
        try:
            # Stop all streams
            for stream_key in list(self.active_streams.keys()):
                self.active_streams[stream_key] = False

            # Cancel all stream tasks with proper cleanup
            stream_tasks = list(self.stream_tasks.values())
            for task in stream_tasks:
                if not task.done():
                    task.cancel()

            # Wait for task cancellation with timeout
            if stream_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*stream_tasks, return_exceptions=True),
                        timeout=STREAM_TASK_CLEANUP_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for stream tasks to complete")

            # Disconnect from data sources with proper error handling
            data_sources = list(self.data_sources.values())
            for data_source in data_sources:
                try:
                    await data_source.disconnect()
                except Exception as e:
                    self.logger.warning(f"Error disconnecting data source: {e}")

            # Clear caches
            self.ticker_cache.clear()
            self.order_book_cache.clear()
            self.trade_cache.clear()
            self.stream_tasks.clear()
            self.data_sources.clear()

            self.logger.info("MarketDataSource cleanup completed")

        except Exception as e:
            # Use ErrorHandler for cleanup errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="cleanup",
                details={
                    "operation": "cleanup",
                    "active_streams": list(self.active_streams.keys()),
                    "active_tasks": list(self.stream_tasks.keys()),
                },
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Error during MarketDataSource cleanup: {e!s}")
        finally:
            # Force cleanup any remaining resources
            try:
                # Force cancel any remaining tasks
                for task in stream_tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                        except Exception as e:
                            self.logger.debug(f"Error during task cleanup: {e}")

                # Force disconnect any remaining data sources
                for data_source in data_sources:
                    try:
                        await data_source.disconnect()
                    except Exception as e:
                        self.logger.debug(f"Error disconnecting data source during cleanup: {e}")

                # Clear all data structures
                self.ticker_cache.clear()
                self.order_book_cache.clear()
                self.trade_cache.clear()
                self.stream_tasks.clear()
                self.data_sources.clear()
                self.active_streams.clear()
            except Exception as e:
                self.logger.warning(f"Error in final cleanup: {e}")

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # Use ErrorHandler for analytics retrieval errors
            error_context = self.error_handler.create_error_context(
                error=e,
                component="MarketDataSource",
                operation="get_error_analytics",
                details={"operation": "analytics_retrieval"},
            )

            await self.error_handler.handle_error(e, error_context)
            self.pattern_analytics.add_error_event(error_context.__dict__)

            self.logger.error(f"Failed to get error analytics: {e!s}")
            return {
                "error": str(e),
                "error_id": error_context.error_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
