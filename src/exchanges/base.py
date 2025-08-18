"""
Base exchange interface for unified trading operations.

This module defines the abstract base class that all exchange implementations
must inherit from, providing a unified interface across different exchanges.

CRITICAL: This integrates with P-001 (core types, exceptions, config) and
P-002A (error handling) components.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal

from src.core.config import Config
from src.core.exceptions import (
    ExchangeConnectionError,
    ExchangeRateLimitError,
    OrderRejectionError,
    ValidationError,
)
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Ticker,
    Trade,
)

# MANDATORY: Import from P-001 (database)
from src.database.connection import get_async_session
from src.database.models import BalanceSnapshot, PerformanceMetrics
from src.database.queries import DatabaseQueries
from src.database.redis_client import RedisClient
from src.error_handling.connection_manager import ConnectionManager as ErrorConnectionManager

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import (
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
)

# MANDATORY: Import from P-007 (advanced rate limiting)
from src.exchanges.advanced_rate_limiter import get_global_rate_limiter
from src.exchanges.connection_manager import ConnectionManager

# MANDATORY: Import from P-007A (utils)
from src.utils.decorators import log_calls, log_errors, memory_usage, retry, time_execution

logger = get_logger(__name__)


class BaseExchange(ABC):
    """
    Abstract base class for all exchange implementations.

    This class defines the unified interface that all exchange implementations
    must follow, ensuring consistent behavior across different exchanges.

    CRITICAL: All exchange implementations (P-004, P-005, P-006) must inherit
    from this exact interface.
    """

    def __init__(self, config: Config, exchange_name: str):
        """
        Initialize the base exchange.

        Args:
            config: Application configuration
            exchange_name: Name of the exchange (e.g., 'binance', 'okx')
        """
        self.config = config
        self.exchange_name = exchange_name
        self.status = "initializing"
        self.connected = False
        self.last_heartbeat = None

        # Initialize error handling
        self.error_handler = ErrorHandler(config.error_handling)
        self.error_connection_manager = ErrorConnectionManager(config.error_handling)

        # P-007: Advanced rate limiting and connection management integration
        self.advanced_rate_limiter = get_global_rate_limiter(config)
        self.connection_manager = ConnectionManager(config, exchange_name)

        # Initialize rate limiter and connection manager
        self.rate_limiter = None  # Will be set by subclasses
        self.ws_manager = None  # Will be set by subclasses

        # Initialize database connection
        self.db_session = None
        self.db_queries = None

        # Initialize Redis client for real-time data
        self.redis_client = None

        # Note: Data module components removed to avoid circular dependency

        # TODO: Remove in production
        logger.debug(f"BaseExchange initialized with P-007 components for {exchange_name}")
        logger.info(f"Initialized {exchange_name} exchange interface")

    async def _initialize_database(self) -> None:
        """
        Initialize database connection and queries.
        """
        try:
            # Get database session with connection pooling
            self.db_session = get_async_session()
            self.db_queries = DatabaseQueries(self.db_session)

            # Test database connection
            if not await self.db_queries.health_check():
                raise Exception("Database health check failed")

            logger.debug(f"Database initialized for {self.exchange_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize database for {self.exchange_name}: {e!s}")
            self.db_session = None
            self.db_queries = None

    async def _initialize_redis(self) -> None:
        """
        Initialize Redis client for real-time data caching.
        """
        try:
            # Get Redis URL from config (simplified for now)
            redis_url = self.config.get_redis_url()
            self.redis_client = RedisClient(redis_url)
            await self.redis_client.connect()

            # Test Redis connection
            if not await self.redis_client.health_check():
                raise Exception("Redis health check failed")

            logger.debug(f"Redis initialized for {self.exchange_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for {self.exchange_name}: {e!s}")
            self.redis_client = None

    # Note: _initialize_data_module method removed to avoid circular dependency

    async def _handle_exchange_error(
        self, error: Exception, operation: str, context: dict | None = None
    ) -> None:
        """
        Handle exchange errors using the error handler with proper context.

        Args:
            error: The exception that occurred
            operation: The operation being performed
            context: Additional context information
        """
        try:
            # Create error context with exchange-specific information
            error_context = self.error_handler.create_error_context(
                error=error,
                component="exchange",
                operation=operation,
                symbol=context.get("symbol") if context else None,
                order_id=context.get("order_id") if context else None,
                details={
                    "exchange_name": self.exchange_name,
                    "operation": operation,
                    **(context if context else {}),
                },
            )

            # Handle the error with appropriate recovery scenario
            if isinstance(error, ExchangeConnectionError):
                recovery_scenario = NetworkDisconnectionRecovery(self.config)
            elif isinstance(error, OrderRejectionError | ValidationError):
                recovery_scenario = OrderRejectionRecovery(self.config)
            else:
                recovery_scenario = None

            # Handle the error
            await self.error_handler.handle_error(error_context, recovery_scenario)

        except Exception as e:
            # Fallback to basic logging if error handling fails
            logger.error(f"Error handling failed for {operation}: {e!s}")

    @abstractmethod
    @time_execution
    @memory_usage
    @retry(max_attempts=3, base_delay=1.0)
    @log_calls
    @log_errors
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    async def disconnect(self) -> None:
        """Disconnect from the exchange and cleanup resources."""
        try:
            # Call exchange-specific disconnect
            await self._disconnect_from_exchange()

        except Exception as e:
            logger.error(f"Error during disconnect: {e!s}")

    @abstractmethod
    async def _disconnect_from_exchange(self) -> None:
        """Disconnect from the exchange (to be implemented by subclasses)."""
        pass

    @abstractmethod
    @time_execution
    @memory_usage
    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from the exchange.

        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        pass

    async def _store_balance_snapshot(self, balances: dict[str, Decimal]) -> None:
        """
        Store balance snapshot in the database.

        Args:
            balances: Dictionary mapping asset symbols to balances
        """
        try:
            if not self.db_queries:
                return

            # Store balance snapshots for each currency
            for currency, total_balance in balances.items():
                if total_balance > 0:
                    balance_snapshot = BalanceSnapshot(
                        user_id=f"{self.exchange_name}_user",  # Placeholder user ID
                        exchange=self.exchange_name,
                        currency=currency,
                        total_balance=total_balance,
                        available_balance=total_balance,  # Simplified for now
                        locked_balance=Decimal("0"),  # Simplified for now
                        timestamp=datetime.now(),
                    )

                    await self.db_queries.create(balance_snapshot)

            logger.debug(f"Balance snapshot stored for {self.exchange_name}")

        except Exception as e:
            logger.error(f"Failed to store balance snapshot: {e!s}")

    async def _cache_market_data(self, symbol: str, data: dict, ttl: int = 300) -> None:
        """
        Cache market data in Redis for fast access.

        Args:
            symbol: Trading symbol
            data: Market data to cache
            ttl: Time to live in seconds
        """
        try:
            if not self.redis_client:
                return

            key = f"market_data:{self.exchange_name}:{symbol}"
            await self.redis_client.set(key, data, ttl=ttl)
            logger.debug(f"Market data cached for {symbol}")

        except Exception as e:
            logger.error(f"Failed to cache market data: {e!s}")

    async def _store_performance_metrics(self, metrics: dict) -> None:
        """
        Store performance metrics in the database.

        Args:
            metrics: Dictionary containing performance metrics
        """
        try:
            if not self.db_queries:
                return

            # Create performance metrics record
            performance_metrics = PerformanceMetrics(
                bot_id=f"{self.exchange_name}_bot",  # Placeholder bot ID
                metric_date=datetime.now().date(),
                total_trades=metrics.get("total_trades", 0),
                winning_trades=metrics.get("winning_trades", 0),
                losing_trades=metrics.get("losing_trades", 0),
                total_pnl=metrics.get("total_pnl", Decimal("0")),
                win_rate=metrics.get("win_rate", 0.0),
                sharpe_ratio=metrics.get("sharpe_ratio"),
                max_drawdown=metrics.get("max_drawdown"),
            )

            # Store in database
            await self.db_queries.create(performance_metrics)
            logger.debug(f"Performance metrics stored for {self.exchange_name}")

        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e!s}")

    @abstractmethod
    @time_execution
    @memory_usage
    @log_errors
    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Execute a trade order on the exchange.

        Args:
            order: Order request with all necessary details

        Returns:
            OrderResponse: Order response with execution details

        Raises:
            ExchangeError: If order placement fails
            ValidationError: If order request is invalid
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Check the status of an order.

        Args:
            order_id: ID of the order to check

        Returns:
            OrderStatus: Current status of the order
        """
        pass

    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """
        Get OHLCV market data for a symbol using centralized data source.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')

        Returns:
            MarketData: Market data with price and volume information
        """
        try:
            # Use centralized market data source if available
            if self.market_data_source:
                # Subscribe to ticker updates for real-time data
                market_data = await self.market_data_source.get_historical_data(
                    self.exchange_name, symbol, start_time=None, end_time=None, interval=timeframe
                )

                if market_data:
                    # Return the most recent data point
                    return market_data[-1] if isinstance(market_data, list) else market_data

            # Fallback to exchange-specific implementation
            return await self._get_market_data_from_exchange(symbol, timeframe)

        except Exception as e:
            logger.error(f"Failed to get market data: {e!s}")
            # Fallback to exchange-specific implementation
            return await self._get_market_data_from_exchange(symbol, timeframe)

    async def get_processed_market_data(
        self, symbol: str, timeframe: str = "1m", processing_steps: list | None = None
    ) -> dict:
        """
        Get processed market data with advanced features and validation.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            processing_steps: List of processing steps to apply

        Returns:
            dict: Processed market data with features and validation results
        """
        try:
            # Get raw market data
            raw_data = await self.get_market_data(symbol, timeframe)

            if not raw_data:
                return {}

            result = {
                "raw_data": raw_data,
                "processed_data": None,
                "technical_indicators": {},
                "statistical_features": {},
                "validation_results": {},
                "quality_score": 0.0,
            }

            # Process data if processor is available
            if self.data_processor:
                processed_result = await self.data_processor.process_market_data(
                    raw_data, processing_steps
                )
                result["processed_data"] = processed_result

            # Calculate technical indicators if available
            if self.technical_indicators:
                self.technical_indicators.add_market_data(raw_data)

                # Calculate common indicators
                result["technical_indicators"] = {
                    "sma_20": await self.technical_indicators.calculate_sma(symbol, 20, "price"),
                    "rsi_14": await self.technical_indicators.calculate_rsi(symbol, 14),
                    "macd": await self.technical_indicators.calculate_macd(symbol),
                    "bollinger": await self.technical_indicators.calculate_bollinger_bands(
                        symbol, 20, 2.0
                    ),
                }

            # Calculate statistical features if available
            if self.statistical_features:
                self.statistical_features.add_market_data(raw_data)

                # Calculate common statistical features
                result["statistical_features"] = {
                    "rolling_stats": await self.statistical_features.calculate_rolling_stats(
                        symbol, 20, "price"
                    ),
                    "autocorrelation": await self.statistical_features.calculate_autocorrelation(
                        symbol, 10, "price"
                    ),
                    "regime": await self.statistical_features.detect_regime(symbol, 50, "price"),
                }

            # Validate data if validator is available
            if self.data_validator:
                is_valid, validation_issues = await self.data_validator.validate_market_data(
                    raw_data
                )
                result["validation_results"] = {"is_valid": is_valid, "issues": validation_issues}

            # Monitor data quality if monitor is available
            if self.quality_monitor:
                quality_score, drift_alerts = await self.quality_monitor.monitor_data_quality(
                    raw_data
                )
                result["quality_score"] = quality_score

            return result

        except Exception as e:
            logger.error(f"Failed to get processed market data: {e!s}")
            return {}

    @abstractmethod
    async def _get_market_data_from_exchange(
        self, symbol: str, timeframe: str = "1m"
    ) -> MarketData:
        """
        Get market data from exchange API (to be implemented by subclasses).

        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data

        Returns:
            MarketData: Market data from exchange
        """
        pass

    @abstractmethod
    async def subscribe_to_stream(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to real-time data stream for a symbol.

        Args:
            symbol: Trading symbol to subscribe to
            callback: Callback function to handle incoming data
        """
        pass

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Get order book depth for a symbol.

        Args:
            symbol: Trading symbol
            depth: Number of levels to retrieve

        Returns:
            OrderBook: Order book with bid and ask levels
        """
        pass

    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get historical trades for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to retrieve

        Returns:
            List[Trade]: List of historical trades
        """
        try:
            # Try to get trades from database first
            if self.db_queries:
                trades = await self.db_queries.get_trades_by_symbol(symbol, limit=limit)
                if trades:
                    logger.debug(f"Retrieved {len(trades)} trades from database for {symbol}")
                    return trades

            # Fallback to exchange API if no database data
            return await self._get_trade_history_from_exchange(symbol, limit)

        except Exception as e:
            logger.error(f"Failed to get trade history: {e!s}")
            return []

    @abstractmethod
    async def _get_trade_history_from_exchange(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get trade history from exchange API (to be implemented by subclasses).

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to retrieve

        Returns:
            List[Trade]: List of trades from exchange
        """
        pass

    # Optional lifecycle/portfolio helpers for higher layers (risk/emergency)
    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """
        Optional: Return open orders. Default implementation returns empty list.

        Args:
            symbol: Optional symbol filter

        Returns:
            List[OrderResponse]: Open orders
        """
        return []

    async def get_positions(self) -> list[Position]:  # type: ignore[name-defined]
        """
        Optional: Return current open positions. Default implementation returns empty list.

        Returns:
            List[Position]: Open positions
        """
        return []

    @abstractmethod
    async def get_exchange_info(self) -> ExchangeInfo:
        """
        Get exchange information including supported symbols and features.

        Returns:
            ExchangeInfo: Exchange information
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """
        Get real-time ticker information for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker: Real-time ticker data
        """
        pass

    # Standard methods that can be overridden
    async def pre_trade_validation(self, order: OrderRequest) -> bool:
        """
        Pre-trade validation hook.

        Args:
            order: Order request to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Basic validation
            if not order.symbol or not order.quantity:
                logger.warning("Invalid order: missing symbol or quantity")
                return False

            if order.quantity <= 0:
                logger.warning("Invalid order: quantity must be positive")
                return False

            # TODO: Remove in production - Additional validation can be added
            # here
            logger.debug(f"Order validation passed for {order.symbol}")
            return True

        except Exception as e:
            logger.error(f"Order validation failed: {e!s}")
            return False

    async def post_trade_processing(self, order_response: OrderResponse) -> None:
        """
        Post-trade processing hook.

        Args:
            order_response: Response from the exchange after order execution
        """
        try:
            # Log the trade
            logger.info(
                "Order executed",
                order_id=order_response.id,
                symbol=order_response.symbol,
                side=order_response.side.value,
                filled_quantity=str(order_response.filled_quantity),
            )

            # Store trade in database if available
            if self.db_queries and order_response.filled_quantity > 0:
                await self._store_trade_in_database(order_response)

            # Update performance metrics
            await self._update_performance_metrics(order_response)

            # TODO: Remove in production - Additional processing can be added here
            # - Update position tracking
            # - Calculate fees

        except Exception as e:
            logger.error(f"Post-trade processing failed: {e!s}")

    async def _store_trade_in_database(self, order_response: OrderResponse) -> None:
        """
        Store executed trade in the database.

        Args:
            order_response: Order response with execution details
        """
        try:
            if not self.db_queries:
                return

            # Validate trade data before storage
            if not self._validate_trade_data(order_response):
                logger.warning(f"Invalid trade data for {order_response.id}, skipping storage")
                return

            # Create trade record
            trade = Trade(
                id=order_response.id,
                bot_id=f"{self.exchange_name}_bot",  # Placeholder bot ID
                symbol=order_response.symbol,
                side=order_response.side.value,
                order_type=order_response.order_type.value,
                quantity=order_response.filled_quantity,
                price=order_response.price or Decimal("0"),
                executed_price=order_response.price or Decimal("0"),
                fee=Decimal("0"),  # Will be calculated separately
                pnl=Decimal("0"),  # Will be calculated separately
                status=order_response.status,
                timestamp=order_response.timestamp,
            )

            # Store in database
            await self.db_queries.create(trade)
            logger.debug(f"Trade {order_response.id} stored in database")

        except Exception as e:
            logger.error(f"Failed to store trade in database: {e!s}")

    def _validate_trade_data(self, order_response: OrderResponse) -> bool:
        """
        Validate trade data before database storage.

        Args:
            order_response: Order response to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic validation
            if not order_response.id or not order_response.symbol:
                return False

            if order_response.filled_quantity <= 0:
                return False

            if not order_response.timestamp:
                return False

            return True

        except Exception as e:
            logger.error(f"Trade data validation failed: {e!s}")
            return False

    async def _update_performance_metrics(self, order_response: OrderResponse) -> None:
        """
        Update performance metrics after trade execution.

        Args:
            order_response: Order response with execution details
        """
        try:
            if not self.db_queries:
                return

            # Get current performance metrics for today
            today = datetime.now().date()
            existing_metrics = await self.db_queries.get_performance_metrics_by_bot(
                f"{self.exchange_name}_bot", start_date=today, end_date=today
            )

            if existing_metrics:
                # Update existing metrics
                metrics = existing_metrics[0]
                metrics.total_trades += 1

                # Calculate P&L (simplified - would need position tracking for accurate calculation)
                if order_response.side.value == "buy":
                    # This is a buy, P&L calculation would depend on position management
                    pass
                else:
                    # This is a sell, P&L calculation would depend on position management
                    pass

                # Update win rate (simplified)
                # TODO: Implement proper win/loss calculation based on position P&L

                await self.db_queries.update(metrics)
            else:
                # Create new metrics for today
                metrics_data = {
                    "total_trades": 1,
                    "winning_trades": 0,  # TODO: Calculate based on P&L
                    "losing_trades": 0,  # TODO: Calculate based on P&L
                    "total_pnl": Decimal("0"),  # TODO: Calculate based on position P&L
                    "win_rate": 0.0,  # TODO: Calculate based on win/loss ratio
                    "sharpe_ratio": None,  # TODO: Calculate based on returns
                    "max_drawdown": None,  # TODO: Calculate based on equity curve
                }

                await self._store_performance_metrics(metrics_data)

            logger.debug(f"Performance metrics updated for {self.exchange_name}")

        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e!s}")

    async def _execute_database_transaction(self, operations: list[callable]) -> bool:
        """
        Execute multiple database operations in a transaction.

        Args:
            operations: List of async functions to execute

        Returns:
            bool: True if all operations succeeded, False otherwise
        """
        try:
            if not self.db_session:
                return False

            async with self.db_session.begin():
                for operation in operations:
                    await operation()

                return True

        except Exception as e:
            logger.error(f"Database transaction failed: {e!s}")
            return False

    # Note: _cleanup_data_module method removed to avoid circular dependency

    def is_connected(self) -> bool:
        """
        Check if the exchange is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected and self.status == "connected"

    def get_status(self) -> str:
        """
        Get the current status of the exchange.

        Returns:
            str: Current status
        """
        return self.status

    def get_exchange_name(self) -> str:
        """
        Get the name of the exchange.

        Returns:
            str: Exchange name
        """
        return self.exchange_name

    async def health_check(self) -> bool:
        """
        Perform a health check on the exchange connection.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Basic health check - try to get account balance
            await self.get_account_balance()
            self.last_heartbeat = datetime.now()

            # Check database health if available
            if self.db_queries:
                db_healthy = await self.db_queries.health_check()
                if not db_healthy:
                    logger.warning("Database health check failed", exchange=self.exchange_name)
                    return False

            # Check Redis health if available
            if self.redis_client:
                redis_healthy = await self.redis_client.health_check()
                if not redis_healthy:
                    logger.warning("Redis health check failed", exchange=self.exchange_name)
                    return False

            # Note: Data module health check removed to avoid circular dependency

            return True
        except Exception as e:
            logger.warning("Health check failed", exchange=self.exchange_name, error=str(e))
            return False

    async def _check_rate_limit(self, endpoint: str, weight: int = 1) -> bool:
        """
        Check if rate limit allows the request.

        Args:
            endpoint: API endpoint
            weight: Request weight

        Returns:
            bool: True if request is allowed

        Raises:
            ExchangeRateLimitError: If rate limit is exceeded
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")

            if weight <= 0:
                raise ValidationError("Weight must be positive")

            # Check rate limit using advanced rate limiter
            return await self.advanced_rate_limiter.check_rate_limit(
                self.exchange_name, endpoint, weight
            )

        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            logger.error(
                "Rate limit check failed",
                exchange=self.exchange_name,
                endpoint=endpoint,
                error=str(e),
            )
            raise ExchangeRateLimitError(f"Rate limit check failed: {e!s}")

    def get_rate_limits(self) -> dict[str, int]:
        """
        Get the rate limits for this exchange.

        Returns:
            Dict[str, int]: Rate limits configuration
        """
        return self.config.exchanges.rate_limits.get(self.exchange_name, {})

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
