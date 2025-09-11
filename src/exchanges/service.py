"""
Exchange Service Layer Implementation

This service layer provides a clean abstraction over exchange implementations,
implementing proper service layer patterns:
- Controllers should call this service, not exchange implementations directly
- Business logic centralized in service layer
- Infrastructure dependencies injected via interfaces
- Proper error handling and validation
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseService
from src.core.base.interfaces import HealthStatus
from src.core.config import Config
from src.core.exceptions import (
    ServiceError,
    ValidationError,
)
from src.core.logging import get_logger
from src.core.types import (
    ExchangeInfo,
    MarketData,
    OrderBook,
    OrderRequest,
    OrderResponse,
    OrderStatus,
    Position,
    Ticker,
)

# Import error handling decorators
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Import exchange interfaces to avoid circular dependencies
from src.exchanges.interfaces import IExchange, IExchangeFactory
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class ExchangeService(BaseService):
    """
    Service layer for exchange operations.

    This service provides a clean abstraction over exchange implementations,
    handling business logic and coordination between exchanges.

    Controllers should use this service instead of calling exchanges directly.
    """

    def __init__(
        self,
        exchange_factory: IExchangeFactory,
        config: Config,
        error_handling_service=None,  # Will be type hinted after proper import
        correlation_id: str | None = None,
    ):
        """
        Initialize exchange service with dependency injection.

        Args:
            exchange_factory: Factory for creating exchange instances (injected)
            config: System configuration (injected)
            error_handling_service: Error handling service (injected)
            correlation_id: Request correlation ID
        """
        # Dependencies must be injected by caller (constructor injection pattern)

        super().__init__(config)

        self.exchange_factory = exchange_factory
        self.config = config
        self.error_handling_service = error_handling_service
        self._active_exchanges: dict[str, IExchange] = {}

        # Service configuration
        service_config = getattr(config, "exchange_service", {})
        self._default_timeout = getattr(service_config, "default_timeout_seconds", 30)
        self._max_retries = getattr(service_config, "max_retries", 3)
        self._health_check_interval = getattr(service_config, "health_check_interval_seconds", 60)

        logger.info("Exchange service initialized with dependency injection")

    async def _do_start(self) -> None:
        """Start the exchange service."""
        try:
            # Start exchange factory if it's a service
            if hasattr(self.exchange_factory, "start"):
                await self.exchange_factory.start()

            logger.info("Exchange service started")

        except Exception as e:
            logger.error(f"Failed to start exchange service: {e}")
            raise ServiceError(f"Exchange service startup failed: {e}")

    async def _do_stop(self) -> None:
        """Stop the exchange service."""
        try:
            # Disconnect all active exchanges
            await self.disconnect_all_exchanges()

            # Stop exchange factory if it's a service
            if hasattr(self.exchange_factory, "stop"):
                await self.exchange_factory.stop()

            logger.info("Exchange service stopped")

        except Exception as e:
            logger.error(f"Error stopping exchange service: {e}")

    # Exchange Management Methods

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def get_exchange(self, exchange_name: str) -> IExchange:
        """
        Get or create an exchange instance.

        Args:
            exchange_name: Name of the exchange

        Returns:
            Exchange instance

        Raises:
            ServiceError: If exchange creation fails
            ValidationError: If exchange name is invalid
        """
        try:
            # Validate exchange name
            if not exchange_name:
                raise ValidationError("Exchange name is required")

            # Check if exchange is already active and healthy
            if exchange_name in self._active_exchanges:
                exchange = self._active_exchanges[exchange_name]

                # Quick health check
                if await self._is_exchange_healthy(exchange):
                    return exchange
                else:
                    # Remove unhealthy exchange
                    await self._remove_exchange(exchange_name)

            # Create new exchange instance
            exchange: IExchange | None = await self.exchange_factory.get_exchange(
                exchange_name=exchange_name, create_if_missing=True
            )

            if not exchange:
                raise ServiceError(f"Failed to create exchange: {exchange_name}")

            # Type assertion after null check
            assert exchange is not None

            # Store active exchange
            self._active_exchanges[exchange_name] = exchange

            logger.info(f"Exchange {exchange_name} ready for use")
            return exchange

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Failed to get exchange {exchange_name}: {e}")
            raise ServiceError(f"Exchange retrieval failed: {e}")

    async def _is_exchange_healthy(self, exchange: IExchange) -> bool:
        """Check if exchange is healthy."""
        try:
            health_result = await exchange.health_check()
            # Handle both bool and HealthCheckResult return types
            if hasattr(health_result, "status"):
                # HealthCheckResult type
                return health_result.status == HealthStatus.HEALTHY
            else:
                # bool type
                return bool(health_result)
        except Exception as e:
            self.logger.warning(f"Health check failed for exchange {exchange.exchange_name}: {e}")
            return False

    async def _remove_exchange(self, exchange_name: str) -> None:
        """Remove an exchange from active pool with proper resource cleanup."""
        exchange_to_disconnect = None

        try:
            # Get exchange reference first
            exchange_to_disconnect = self._active_exchanges.get(exchange_name)
        except Exception as e:
            logger.warning(f"Error getting exchange {exchange_name}: {e}")

        # Disconnect exchange if found
        if exchange_to_disconnect:
            try:
                await exchange_to_disconnect.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting exchange {exchange_name}: {e}")

        # Always remove from active exchanges, even if disconnect failed
        try:
            if exchange_name in self._active_exchanges:
                del self._active_exchanges[exchange_name]
                logger.info(f"Removed exchange from active pool: {exchange_name}")
        except Exception as e:
            logger.warning(f"Error removing exchange {exchange_name} from active pool: {e}")

    # Trading Operations (Business Logic Layer)

    @with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def place_order(self, exchange_name: str, order: OrderRequest) -> OrderResponse:
        """
        Place a trading order through the service layer.

        Args:
            exchange_name: Target exchange name
            order: Order request details

        Returns:
            Order response with execution details

        Raises:
            ServiceError: If order placement fails
            ValidationError: If order data is invalid
        """
        try:
            # Business logic validation
            await self._validate_order_request(order)

            # Get exchange instance
            exchange = await self.get_exchange(exchange_name)

            # Execute order through exchange
            order_response = await exchange.place_order(order)

            # Post-trade business logic
            await self._process_order_response(exchange_name, order_response)

            logger.info(f"Order placed successfully: {order_response.id}")
            return order_response

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise ServiceError(f"Failed to place order: {e}")

    async def _validate_order_request(self, order: OrderRequest) -> None:
        """Validate order request business rules."""
        if not order.symbol:
            raise ValidationError("Order symbol is required")

        if not order.quantity or order.quantity <= 0:
            raise ValidationError("Order quantity must be positive")

        if not order.side:
            raise ValidationError("Order side is required")

        # Additional business logic validation can be added here
        logger.debug(f"Order validation passed for {order.symbol}")

    async def _process_order_response(
        self, exchange_name: str, order_response: OrderResponse
    ) -> None:
        """Process order response with business logic."""
        try:
            # Log order execution
            logger.info(
                f"Order executed on {exchange_name}",
                order_id=order_response.id,
                symbol=order_response.symbol,
                status=order_response.status,
            )

            # Additional business logic processing can be added here
            # e.g., risk management, notifications, metrics

        except Exception as e:
            logger.warning(f"Error processing order response: {e}")

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def cancel_order(
        self, exchange_name: str, order_id: str, symbol: str | None = None
    ) -> bool:
        """
        Cancel an existing order.

        Args:
            exchange_name: Exchange name where order was placed
            order_id: ID of order to cancel
            symbol: Trading symbol (optional, improves performance)

        Returns:
            True if cancellation successful

        Raises:
            ServiceError: If cancellation fails
        """
        try:
            if not order_id:
                raise ValidationError("Order ID is required")

            exchange = await self.get_exchange(exchange_name)

            # Use symbol if provided for better exchange compatibility
            # BaseExchange expects (symbol, order_id) and returns OrderResponse, not bool
            if symbol:
                # Use BaseExchange signature
                order_response = await exchange.cancel_order(symbol, order_id)
                # Check if it's OrderResponse or bool
                if hasattr(order_response, "status"):
                    # OrderResponse type - consider successful if not rejected/failed
                    from src.core.types import OrderStatus
                    result = order_response.status not in [OrderStatus.REJECTED]
                else:
                    # Bool type or other
                    result = bool(order_response)
            else:
                # Some exchanges might support cancelling without symbol (less common)
                try:
                    order_response = await exchange.cancel_order("", order_id)  # Empty symbol
                    if hasattr(order_response, "status"):
                        from src.core.types import OrderStatus
                        result = order_response.status not in [OrderStatus.REJECTED]
                    else:
                        result = bool(order_response)
                except Exception:
                    raise ValidationError("Symbol is required for order cancellation on this exchange")

            if result:
                logger.info(f"Order cancelled: {order_id}")
            else:
                logger.warning(f"Order cancellation failed: {order_id}")

            return result

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")
            raise ServiceError(f"Failed to cancel order: {e}")

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def get_order_status(
        self, exchange_name: str, order_id: str, symbol: str | None = None
    ) -> OrderStatus:
        """
        Get order status.

        Args:
            exchange_name: Exchange name
            order_id: Order ID
            symbol: Trading symbol (optional, improves performance)

        Returns:
            Current order status
        """
        try:
            if not order_id:
                raise ValidationError("Order ID is required")

            exchange = await self.get_exchange(exchange_name)

            # BaseExchange expects (symbol, order_id) and returns OrderResponse
            if symbol:
                # Use BaseExchange signature
                order_response = await exchange.get_order_status(symbol, order_id)
                # Extract status from OrderResponse
                if hasattr(order_response, "status"):
                    status = order_response.status
                else:
                    # Fallback - assume response is the status itself
                    status = order_response
            else:
                # Some exchanges might support lookup without symbol (less common)
                try:
                    order_response = await exchange.get_order_status("", order_id)  # Empty symbol
                    status = order_response.status if hasattr(order_response, "status") else order_response
                except Exception:
                    raise ValidationError("Symbol is required for order status lookup on this exchange")

            logger.debug(f"Order status retrieved: {order_id} -> {status}")
            return status

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get order status failed: {e}")
            raise ServiceError(f"Failed to get order status: {e}")

    # Market Data Operations

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=2, base_delay=Decimal("0.5"))
    @time_execution
    async def get_market_data(
        self, exchange_name: str, symbol: str, timeframe: str = "1m"
    ) -> MarketData:
        """
        Get market data for a symbol.

        Args:
            exchange_name: Exchange name
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Market data
        """
        try:
            if not symbol:
                raise ValidationError("Symbol is required")

            exchange = await self.get_exchange(exchange_name)
            market_data = await exchange.get_market_data(symbol, timeframe)

            logger.debug(f"Market data retrieved: {symbol} from {exchange_name}")
            return market_data

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get market data failed: {e}")
            raise ServiceError(f"Failed to get market data: {e}")

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=2, base_delay=Decimal("0.5"))
    @time_execution
    async def get_order_book(self, exchange_name: str, symbol: str, depth: int = 10) -> OrderBook:
        """Get order book data."""
        try:
            if not symbol:
                raise ValidationError("Symbol is required")

            if depth <= 0 or depth > 1000:
                raise ValidationError("Depth must be between 1 and 1000")

            exchange = await self.get_exchange(exchange_name)
            order_book = await exchange.get_order_book(symbol, depth)

            logger.debug(f"Order book retrieved: {symbol} from {exchange_name}")
            return order_book

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get order book failed: {e}")
            raise ServiceError(f"Failed to get order book: {e}")

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=2, base_delay=Decimal("0.5"))
    @time_execution
    async def get_ticker(self, exchange_name: str, symbol: str) -> Ticker:
        """Get ticker data."""
        try:
            if not symbol:
                raise ValidationError("Symbol is required")

            exchange = await self.get_exchange(exchange_name)
            ticker = await exchange.get_ticker(symbol)

            logger.debug(f"Ticker retrieved: {symbol} from {exchange_name}")
            return ticker

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get ticker failed: {e}")
            raise ServiceError(f"Failed to get ticker: {e}")

    # Account Operations

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=3, base_delay=Decimal("1.0"))
    @time_execution
    async def get_account_balance(self, exchange_name: str) -> dict[str, Decimal]:
        """Get account balances."""
        try:
            exchange = await self.get_exchange(exchange_name)
            balances = await exchange.get_account_balance()

            logger.debug(f"Account balance retrieved from {exchange_name}: {len(balances)} assets")
            return balances

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get account balance failed: {e}")
            raise ServiceError(f"Failed to get account balance: {e}")

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=2, base_delay=Decimal("1.0"))
    @time_execution
    async def get_positions(
        self, exchange_name: str, symbol: str | None = None
    ) -> list[Position]:
        """Get open positions."""
        try:
            exchange = await self.get_exchange(exchange_name)
            positions = await exchange.get_positions()

            # Filter by symbol if specified
            if symbol:
                positions = [p for p in positions if p.symbol == symbol]

            logger.debug(f"Positions retrieved from {exchange_name}: {len(positions)}")
            return positions

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get positions failed: {e}")
            raise ServiceError(f"Failed to get positions: {e}")

    # Exchange Information

    @with_circuit_breaker(failure_threshold=3, recovery_timeout=60)
    @with_retry(max_attempts=2, base_delay=Decimal("1.0"))
    @time_execution
    async def get_exchange_info(self, exchange_name: str) -> ExchangeInfo:
        """Get exchange information."""
        try:
            exchange = await self.get_exchange(exchange_name)
            info = await exchange.get_exchange_info()

            logger.debug(f"Exchange info retrieved from {exchange_name}")
            return info

        except (ValidationError, ServiceError):
            raise
        except Exception as e:
            logger.error(f"Get exchange info failed: {e}")
            raise ServiceError(f"Failed to get exchange info: {e}")

    def get_supported_exchanges(self) -> list[str]:
        """Get list of supported exchanges."""
        return self.exchange_factory.get_supported_exchanges()

    def get_available_exchanges(self) -> list[str]:
        """Get list of configured exchanges."""
        return self.exchange_factory.get_available_exchanges()

    async def get_exchange_status(self, exchange_name: str) -> dict[str, Any]:
        """
        Get exchange connection status.

        Args:
            exchange_name: Name of the exchange

        Returns:
            Exchange status information
        """
        try:
            # Check if exchange is in active pool
            if exchange_name in self._active_exchanges:
                exchange = self._active_exchanges[exchange_name]

                # Perform health check
                is_healthy = await self._is_exchange_healthy(exchange)

                return {
                    "connected": exchange.is_connected() if hasattr(exchange, "is_connected") else True,
                    "healthy": is_healthy,
                    "last_ping": datetime.now(timezone.utc).isoformat(),
                    "uptime_seconds": 0,  # Could be enhanced with actual uptime tracking
                }
            else:
                # Exchange not active
                return {
                    "connected": False,
                    "healthy": False,
                    "last_ping": None,
                    "uptime_seconds": 0,
                }

        except Exception as e:
            logger.error(f"Get exchange status failed for {exchange_name}: {e}")
            return {
                "connected": False,
                "healthy": False,
                "error": str(e),
                "last_ping": None,
                "uptime_seconds": 0,
            }

    # Health and Management

    async def get_service_health(self) -> dict[str, Any]:
        """Get comprehensive service health status."""
        health_status = {
            "service": "ExchangeService",
            "status": await self._service_health_check(),
            "active_exchanges": len(self._active_exchanges),
            "exchanges": {},
        }

        # Check health of all active exchanges
        for exchange_name, exchange in self._active_exchanges.items():
            try:
                is_healthy = await self._is_exchange_healthy(exchange)
                health_status["exchanges"][exchange_name] = {
                    "healthy": is_healthy,
                    "last_check": datetime.now(timezone.utc).isoformat(),
                }
            except Exception as e:
                health_status["exchanges"][exchange_name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.now(timezone.utc).isoformat(),
                }

        return health_status

    async def disconnect_all_exchanges(self) -> None:
        """Disconnect all active exchanges with proper resource cleanup."""
        disconnect_tasks = []
        exchanges_to_disconnect = []

        try:
            # Store references to avoid modification during iteration
            exchanges_to_disconnect = list(self._active_exchanges.keys())
        except Exception as e:
            logger.error(f"Error getting exchanges list: {e}")

        # Create disconnect tasks
        for exchange_name in exchanges_to_disconnect:
            try:
                task = asyncio.create_task(self._remove_exchange(exchange_name))
                disconnect_tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating disconnect task for {exchange_name}: {e}")

        # Execute disconnect tasks
        try:
            if disconnect_tasks:
                await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error executing disconnect tasks: {e}")
        finally:
            # Always clear exchanges dict regardless of disconnect success
            try:
                self._active_exchanges.clear()
            except Exception as e:
                logger.error(f"Error clearing exchanges dict: {e}")

            logger.info("All exchanges disconnected")

    # Multi-Exchange Operations (Advanced Business Logic)

    async def get_best_price(
        self, symbol: str, side: str, exchanges: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Find best price across multiple exchanges.

        This is an example of service-layer business logic that coordinates
        multiple exchange operations.
        """
        try:
            if not exchanges:
                exchanges = self.get_available_exchanges()

            best_price = None
            best_exchange = None
            prices = {}

            # Get ticker from all exchanges concurrently
            tasks = []
            for exchange_name in exchanges:
                task = asyncio.create_task(self._get_ticker_safe(exchange_name, symbol))
                tasks.append((exchange_name, task))

            # Collect results
            for exchange_name, task in tasks:
                try:
                    ticker = await task
                    if ticker:
                        price = ticker.bid_price if side.upper() == "SELL" else ticker.ask_price
                        prices[exchange_name] = price

                        # Track best price
                        if (
                            best_price is None
                            or (side.upper() == "SELL" and price > best_price)
                            or (side.upper() == "BUY" and price < best_price)
                        ):
                            best_price = price
                            best_exchange = exchange_name

                except Exception as e:
                    logger.warning(f"Failed to get price from {exchange_name}: {e}")
                    prices[exchange_name] = None

            return {
                "symbol": symbol,
                "side": side,
                "best_price": best_price,
                "best_exchange": best_exchange,
                "all_prices": prices,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Get best price failed: {e}")
            raise ServiceError(f"Failed to get best price: {e}")

    async def _get_ticker_safe(self, exchange_name: str, symbol: str) -> Ticker | None:
        """Get ticker with exception handling."""
        try:
            return await self.get_ticker(exchange_name, symbol)
        except Exception as e:
            self.logger.warning(f"Failed to get ticker for {symbol} on {exchange_name}: {e}")
            return None

    # WebSocket Operations

    async def subscribe_to_stream(self, exchange_name: str, symbol: str, callback: Any) -> None:
        """Subscribe to real-time data stream."""
        try:
            exchange = await self.get_exchange(exchange_name)
            await exchange.subscribe_to_stream(symbol, callback)

            logger.info(f"Subscribed to stream: {symbol} on {exchange_name}")

        except Exception as e:
            logger.error(f"Stream subscription failed: {e}")
            raise ServiceError(f"Failed to subscribe to stream: {e}")

    async def _service_health_check(self) -> HealthStatus:
        """Service-specific health check."""
        try:
            # Check if exchange factory is accessible
            supported = self.exchange_factory.get_supported_exchanges()
            if not supported:
                return HealthStatus.UNHEALTHY

            # Check if at least one exchange is healthy if any are active
            if self._active_exchanges:
                healthy_count = 0
                for exchange in self._active_exchanges.values():
                    if await self._is_exchange_healthy(exchange):
                        healthy_count += 1

                # At least one exchange should be healthy
                return HealthStatus.HEALTHY if healthy_count > 0 else HealthStatus.UNHEALTHY

            return HealthStatus.HEALTHY  # No active exchanges is OK

        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            return HealthStatus.UNHEALTHY
