"""
Coinbase Order Manager (P-006)

This module implements the Coinbase-specific order management functionality,
including order placement, cancellation, and status tracking.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

# Coinbase-specific imports
from coinbase.rest import RESTClient

# LoggerMixin not needed - BaseComponent already provides logging
from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError, ExecutionError, ValidationError

# Logger setup
from src.core.logging import get_logger

# Logger is provided by BaseExchange (via BaseComponent)
# MANDATORY: Import from P-001
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.utils import ValidationFramework
from src.utils.data_utils import normalize_price
from src.utils.decimal_utils import round_to_precision
from src.utils.exchange_conversion_utils import ExchangeConversionUtils
from src.utils.exchange_order_utils import (
    AssetPrecisionUtils,
    FeeCalculationUtils,
    OrderStatusUtils,
)

# Note: Using generic Exception handling for REST API as no specific
# exceptions are documented


class CoinbaseOrderManager:
    """
    Coinbase order manager for handling order operations.

    Provides comprehensive order management functionality including:
    - Order placement with validation
    - Order cancellation and modification
    - Order status tracking and monitoring
    - Fee calculation and reporting
    - Order history and analytics
    """

    def __init__(self, config: Config, exchange_name: str = "coinbase", error_handler: ErrorHandler | None = None):
        """
        Initialize Coinbase order manager.

        Args:
            config: Application configuration
            exchange_name: Exchange name (default: "coinbase")
            error_handler: Error handler service (injected)
        """
        self.config = config
        self.exchange_name = exchange_name
        # Use injected error handler if available
        self.error_handler = error_handler or ErrorHandler(config)

        # Initialize logger
        self.logger = get_logger(self.__class__.__module__)

        # Coinbase-specific configuration
        self.api_key = config.exchange.coinbase_api_key
        self.api_secret = config.exchange.coinbase_api_secret
        self.sandbox = config.exchange.coinbase_sandbox

        # REST client
        self.client: RESTClient | None = None

        # Order tracking
        self.pending_orders: dict[str, OrderRequest] = {}
        self.order_status_cache: dict[str, OrderStatus] = {}
        self.order_history: list[OrderResponse] = []

        # Fee tracking
        self.total_fees: dict[str, Decimal] = {}
        self.fee_currency = "USD"

        # Initialize simple rate limiter using available infrastructure

        # Create a simple rate limiter instance for this exchange
        self.rate_limiter = self._create_rate_limiter(config)

        self.logger.info(f"Initialized {exchange_name} order manager")

    def _get_asset_precision(self, symbol: str, precision_type: str = "quantity") -> int:
        """Get asset-specific precision - delegated to shared utility."""
        return AssetPrecisionUtils.get_asset_precision(symbol, precision_type)

    def _create_rate_limiter(self, config: Config) -> Any:
        """Create a simple rate limiter for this exchange."""
        # Simple rate limiter using decorators
        class SimpleRateLimiter:
            def __init__(self):
                self.last_request_time = 0.0
                self.min_interval = 1.0 / 10  # 10 requests per second max

            async def acquire(self, resource_type: str, amount: int) -> None:
                """Simple rate limiting."""
                import asyncio
                import time
                current_time = time.time()
                time_since_last = current_time - self.last_request_time

                if time_since_last < self.min_interval:
                    await asyncio.sleep(self.min_interval - time_since_last)

                self.last_request_time = time.time()

        return SimpleRateLimiter()

    async def initialize(self) -> bool:
        """
        Initialize the order manager.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize REST client with sandbox support
            base_url = (
                "api-public.sandbox.exchange.coinbase.com" if self.sandbox else "api.coinbase.com"
            )
            self.client = RESTClient(
                api_key=self.api_key, api_secret=self.api_secret, base_url=base_url
            )

            # Test connection
            await self._test_connection()

            self.logger.info(f"Successfully initialized {self.exchange_name} order manager")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize {self.exchange_name} order manager: {e!s}")
            return False

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order on Coinbase exchange.

        Args:
            order: Order request with all necessary details

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Validate order using utils validators
            if not ValidationFramework.validate_order(order.__dict__):
                raise ValidationError("Order validation failed using utils validators")

            # Additional Coinbase-specific validation
            if not await self._validate_order(order):
                raise ValidationError("Coinbase-specific validation failed")

            # Apply rate limiting
            await self.rate_limiter.acquire("orders_per_second", 1)

            # Convert order to Coinbase format
            coinbase_order = self._convert_order_to_coinbase(order)

            # Place order
            result = await self.client.create_order(**coinbase_order)

            # Convert response to unified format using shared utilities
            order_response = ExchangeConversionUtils.convert_coinbase_order_to_response(result)

            # Track order
            self.pending_orders[order_response.id] = order
            self.order_status_cache[order_response.id] = OrderStatus.PENDING
            self.order_history.append(order_response)

            self.logger.info(f"Placed order {order_response.id} for {order.symbol}")
            return order_response

        except ExchangeConnectionError:
            # Re-raise connection errors as-is
            raise
        except Exception as e:
            self.logger.error(f"Failed to place order: {e!s}")
            raise ExecutionError(f"Failed to place order: {e!s}")

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an existing order on Coinbase exchange.

        Args:
            symbol: Trading symbol  
            order_id: ID of the order to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Cancel order
            _ = await self.client.cancel_orders([order_id])

            # Update tracking
            if order_id in self.order_status_cache:
                self.order_status_cache[order_id] = OrderStatus.CANCELLED

            if order_id in self.pending_orders:
                del self.pending_orders[order_id]

            self.logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e!s}")
            return False

    async def get_order_status(self, symbol: str, order_id: str) -> OrderStatus:
        """
        Get the status of an order on Coinbase exchange.

        Args:
            symbol: Trading symbol
            order_id: ID of the order to check

        Returns:
            OrderStatus: Current status of the order
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Get order details
            order = await self.client.get_order(order_id)

            # Convert status
            status = self._convert_coinbase_status_to_order_status(order["status"])

            # Update cache
            self.order_status_cache[order_id] = status

            return status

        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e!s}")
            return OrderStatus.REJECTED

    async def get_order_details(self, order_id: str) -> OrderResponse | None:
        """
        Get detailed information about an order.

        Args:
            order_id: ID of the order to retrieve

        Returns:
            Optional[OrderResponse]: Order details if found, None otherwise
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Get order details
            order = await self.client.get_order(order_id)

            # Convert to unified format using shared utilities
            order_response = ExchangeConversionUtils.convert_coinbase_order_to_response(order)

            return order_response

        except Exception as e:
            self.logger.error(f"Failed to get order details for {order_id}: {e!s}")
            return None

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List[OrderResponse]: List of open orders
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Get open orders
            orders = await self.client.list_orders(product_id=symbol, order_status="OPEN")

            # Convert to unified format
            order_responses = []
            for order in orders:
                order_response = ExchangeConversionUtils.convert_coinbase_order_to_response(order)
                order_responses.append(order_response)

            return order_responses

        except Exception as e:
            self.logger.error(f"Failed to get open orders: {e!s}")
            return []

    async def get_order_history(
        self, symbol: str | None = None, limit: int = 100
    ) -> list[OrderResponse]:
        """
        Get order history.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of orders to retrieve

        Returns:
            List[OrderResponse]: List of historical orders
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Get order history
            orders = await self.client.list_orders(product_id=symbol, limit=limit)

            # Convert to unified format
            order_responses = []
            for order in orders:
                order_response = ExchangeConversionUtils.convert_coinbase_order_to_response(order)
                order_responses.append(order_response)

            return order_responses

        except Exception as e:
            self.logger.error(f"Failed to get order history: {e!s}")
            return []

    async def get_fills(self, order_id: str | None = None, symbol: str | None = None) -> list[dict]:
        """
        Get fill information for orders.

        Args:
            order_id: Optional order ID filter
            symbol: Optional symbol filter

        Returns:
            List[Dict]: List of fill information
        """
        try:
            if not self.client:
                raise ExchangeConnectionError("Not connected to Coinbase")

            # Get fills
            fills = await self.client.list_fills(order_id=order_id, product_id=symbol)

            return fills

        except Exception as e:
            self.logger.error(f"Failed to get fills: {e!s}")
            return []

    async def calculate_fees(self, order: OrderRequest) -> dict[str, Decimal]:
        """
        Calculate estimated fees for an order.

        Args:
            order: Order request to calculate fees for

        Returns:
            Dict[str, Decimal]: Fee breakdown
        """
        try:
            # Get product information for fee calculation
            _ = await self.client.get_product(order.symbol)

            # Calculate fee amount using shared utility
            if order.price:
                order_value = order.quantity * order.price
                is_maker = order.order_type == OrderType.LIMIT
                fee_amount = FeeCalculationUtils.calculate_fee(order_value, "coinbase", order.symbol, is_maker)
                fee_rates = FeeCalculationUtils.get_fee_rates("coinbase")
                fee_rate = fee_rates["maker"] if is_maker else fee_rates["taker"]
            else:
                # For market orders without price, use quantity-based approximation
                fee_rates = FeeCalculationUtils.get_fee_rates("coinbase")
                fee_rate = fee_rates["taker"]  # Market orders are always taker
                fee_amount = order.quantity * fee_rate

            return {
                "fee_rate": fee_rate,
                "fee_amount": fee_amount,
                "fee_currency": self.fee_currency,
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate fees: {e!s}")
            return {
                "fee_rate": Decimal("0"),
                "fee_amount": Decimal("0"),
                "fee_currency": self.fee_currency,
            }

    def get_total_fees(self) -> dict[str, Decimal]:
        """
        Get total fees paid.

        Returns:
            Dict[str, Decimal]: Total fees by currency
        """
        return self.total_fees.copy()

    def get_order_statistics(self) -> dict[str, Any]:
        """
        Get order statistics.

        Returns:
            Dict[str, Any]: Order statistics
        """
        total_orders = len(self.order_history)
        filled_orders = len([o for o in self.order_history if o.status == "FILLED"])
        cancelled_orders = len([o for o in self.order_history if o.status == "CANCELLED"])

        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "cancelled_orders": cancelled_orders,
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0,
            "total_fees": self.total_fees,
        }

    # Helper methods

    async def _test_connection(self) -> None:
        """Test connection to Coinbase API."""
        try:
            # Test connection by getting products (this should always work)
            await self.client.get_products()
        except Exception as e:
            raise ExchangeConnectionError(f"Failed to connect to Coinbase: {e!s}")

    async def _validate_order(self, order: OrderRequest) -> bool:
        """
        Validate order before placement using utils validators.

        Args:
            order: Order to validate

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Use utils validators for comprehensive validation
            if not ValidationFramework.validate_order(order.__dict__):
                self.logger.error("Order validation failed using utils validators")
                return False

            # Additional Coinbase-specific validation
            try:
                ValidationFramework.validate_symbol(order.symbol)
            except (ValueError, ValidationError) as e:
                self.logger.error(f"Invalid symbol format for Coinbase: {e}")
                return False

            # Check symbol format (Coinbase uses format like "BTC-USD")
            if "-" not in order.symbol:
                self.logger.error("Invalid symbol format for Coinbase")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Order validation failed: {e!s}")
            return False

    def _convert_order_to_coinbase(self, order: OrderRequest) -> dict[str, Any]:
        """
        Convert unified order to Coinbase format.

        Args:
            order: Unified order request

        Returns:
            Dict[str, Any]: Coinbase order format
        """
        coinbase_order = {
            "product_id": order.symbol,
            "side": order.side.value.upper(),
            "order_configuration": {},
        }

        # Get precision for the symbol
        quantity_precision = self._get_asset_precision(order.symbol, "quantity")

        # Configure order based on type
        if order.order_type == OrderType.MARKET:
            coinbase_order["order_configuration"] = {
                "market_market_ioc": {"quote_size": str(round_to_precision(order.quantity, quantity_precision))}
            }
        elif order.order_type == OrderType.LIMIT:
            coinbase_order["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": str(round_to_precision(order.quantity, quantity_precision)),
                    "limit_price": str(normalize_price(order.price, order.symbol)),
                }
            }
        elif order.order_type == OrderType.STOP_LOSS:
            coinbase_order["order_configuration"] = {
                "stop_limit_stop_limit_gtc": {
                    "base_size": str(round_to_precision(order.quantity, quantity_precision)),
                    "limit_price": str(normalize_price(order.price, order.symbol)),
                    "stop_price": str(normalize_price(order.stop_price, order.symbol)),
                }
            }

        # Add client order ID if provided
        if order.client_order_id:
            coinbase_order["client_order_id"] = order.client_order_id

        return coinbase_order

    def _convert_coinbase_order_to_response(self, result: dict) -> OrderResponse:
        """
        Convert Coinbase order response to unified format.

        Args:
            result: Coinbase order response

        Returns:
            OrderResponse: Unified order response
        """
        # Extract order configuration
        order_config = result.get("order_configuration", {})

        # Determine order type
        order_type = OrderType.LIMIT  # Default
        if "market_market_ioc" in order_config:
            order_type = OrderType.MARKET
        elif "stop_limit_stop_limit_gtc" in order_config:
            order_type = OrderType.STOP_LOSS

        # Extract quantity and price
        quantity = Decimal("0")
        price = None

        if order_type == OrderType.MARKET:
            market_config = order_config.get("market_market_ioc", {})
            quantity = Decimal(str(market_config.get("quote_size", "0")))
        else:
            limit_config = order_config.get("limit_limit_gtc", {})
            quantity = Decimal(str(limit_config.get("base_size", "0")))
            price = Decimal(str(limit_config.get("limit_price", "0")))

        return OrderResponse(
            id=result["order_id"],
            client_order_id=result.get("client_order_id"),
            symbol=result["product_id"],
            side=OrderSide.BUY if result["side"] == "BUY" else OrderSide.SELL,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=Decimal(str(result.get("filled_size", "0"))),
            status=self._convert_coinbase_status_to_order_status(result.get("status", "REJECTED")),
            created_at=datetime.fromisoformat(result["created_time"].replace("Z", "+00:00")),
            exchange=self.exchange_name,
        )

    def _convert_coinbase_status_to_order_status(self, status: str) -> OrderStatus:
        """Convert Coinbase order status to unified OrderStatus."""
        return OrderStatusUtils.convert_status(status, "coinbase")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass
