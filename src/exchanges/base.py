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
    ExchangeRateLimitError,
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
    Ticker,
    Trade,
)
from src.error_handling.connection_manager import ConnectionManager as ErrorConnectionManager

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

# MANDATORY: Import from P-007 (advanced rate limiting)
from src.exchanges.advanced_rate_limiter import AdvancedRateLimiter
from src.exchanges.connection_manager import ConnectionManager

# MANDATORY: Import from P-007A (utils)

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
        self.connection_manager = ErrorConnectionManager(config.error_handling)

        # P-007: Advanced rate limiting and connection management integration
        self.advanced_rate_limiter = AdvancedRateLimiter(config)
        self.connection_manager = ConnectionManager(config, exchange_name)

        # Initialize rate limiter and connection manager
        self.rate_limiter = None  # Will be set by subclasses
        self.ws_manager = None  # Will be set by subclasses

        # TODO: Remove in production
        logger.debug(f"BaseExchange initialized with P-007 components for {exchange_name}")
        logger.info(f"Initialized {exchange_name} exchange interface")

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            bool: True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass

    @abstractmethod
    async def get_account_balance(self) -> dict[str, Decimal]:
        """
        Get all asset balances from the exchange.

        Returns:
            Dict[str, Decimal]: Dictionary mapping asset symbols to balances
        """
        pass

    @abstractmethod
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

    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """
        Get OHLCV market data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')

        Returns:
            MarketData: Market data with price and volume information
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

    @abstractmethod
    async def get_trade_history(self, symbol: str, limit: int = 100) -> list[Trade]:
        """
        Get historical trades for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to retrieve

        Returns:
            List[Trade]: List of historical trades
        """
        pass

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

            # TODO: Remove in production - Additional processing can be added here
            # - Update position tracking
            # - Calculate fees
            # - Update performance metrics

        except Exception as e:
            logger.error(f"Post-trade processing failed: {e!s}")

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
