"""
Binance Order Management (P-004)

This module implements specialized order handling for Binance exchange,
including order type conversion, time-in-force handling, and order status tracking.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# Binance-specific imports
from binance.exceptions import BinanceAPIException, BinanceOrderException

from src.core.config import Config
from src.core.exceptions import ExchangeError, ExecutionError, OrderRejectionError, ValidationError
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

logger = get_logger(__name__)


class BinanceOrderManager:
    """
    Binance order management for specialized order handling.

    Provides specialized order handling for Binance:
    - Order type conversion (market, limit, stop-loss, OCO)
    - Time-in-force parameter handling (GTC, IOC, FOK)
    - Order status tracking and fill monitoring
    - Partial fill handling and notification
    - Order cancellation with confirmation
    - Fee calculation and reporting

    CRITICAL: This class handles all Binance-specific order operations.
    """

    def __init__(self, config: Config, client, exchange_name: str = "binance"):
        """
        Initialize Binance order manager.

        Args:
            config: Application configuration
            client: Binance client instance
            exchange_name: Exchange name (default: "binance")
        """
        self.config = config
        self.client = client
        self.exchange_name = exchange_name

        # Order tracking
        self.pending_orders: dict[str, dict] = {}
        self.filled_orders: dict[str, dict] = {}
        self.cancelled_orders: dict[str, dict] = {}

        # Error handling
        self.error_handler = ErrorHandler(config.error_handling)

        # Order monitoring
        self.order_monitor_task: asyncio.Task | None = None
        self.monitoring_active = False

        logger.info("Initialized Binance order manager")

    async def place_market_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place a market order on Binance.

        Args:
            order: Order request with market order details

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            # Validate order
            self._validate_market_order(order)

            # Convert to Binance format
            binance_order = self._convert_market_order_to_binance(order)

            # Place order
            result = await self.client.order_market(
                symbol=order.symbol,
                side=order.side.value.upper(),
                quantity=str(order.quantity),
                newClientOrderId=order.client_order_id,
            )

            # Convert response
            response = self._convert_binance_order_to_response(result)

            # Track order
            self._track_order(response)

            logger.info(f"Market order placed: {response.id}")
            return response

        except BinanceOrderException as e:
            logger.error(f"Binance market order error: {e}")
            raise OrderRejectionError(f"Market order rejected: {e}")
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing market order: {e}")
            raise ExchangeError(f"Failed to place market order: {e}")
        except Exception as e:
            logger.error(f"Error placing market order on Binance: {e!s}")
            raise ExecutionError(f"Failed to place market order: {e!s}")

    async def place_limit_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place a limit order on Binance.

        Args:
            order: Order request with limit order details

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            # Validate order
            self._validate_limit_order(order)

            # Convert to Binance format
            binance_order = self._convert_limit_order_to_binance(order)

            # Place order
            result = await self.client.order_limit(
                symbol=order.symbol,
                side=order.side.value.upper(),
                quantity=str(order.quantity),
                price=str(order.price),
                timeInForce=order.time_in_force,
                newClientOrderId=order.client_order_id,
            )

            # Convert response
            response = self._convert_binance_order_to_response(result)

            # Track order
            self._track_order(response)

            logger.info(f"Limit order placed: {response.id}")
            return response

        except BinanceOrderException as e:
            logger.error(f"Binance limit order error: {e}")
            raise OrderRejectionError(f"Limit order rejected: {e}")
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing limit order: {e}")
            raise ExchangeError(f"Failed to place limit order: {e}")
        except Exception as e:
            logger.error(f"Error placing limit order on Binance: {e!s}")
            raise ExecutionError(f"Failed to place limit order: {e!s}")

    async def place_stop_loss_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place a stop-loss order on Binance.

        Args:
            order: Order request with stop-loss details

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            # Validate order
            self._validate_stop_loss_order(order)

            # Convert to Binance format
            binance_order = self._convert_stop_loss_order_to_binance(order)

            # Place order
            result = await self.client.order_stop_loss(
                symbol=order.symbol,
                side=order.side.value.upper(),
                quantity=str(order.quantity),
                stopPrice=str(order.stop_price),
                newClientOrderId=order.client_order_id,
            )

            # Convert response
            response = self._convert_binance_order_to_response(result)

            # Track order
            self._track_order(response)

            logger.info(f"Stop-loss order placed: {response.id}")
            return response

        except BinanceOrderException as e:
            logger.error(f"Binance stop-loss order error: {e}")
            raise OrderRejectionError(f"Stop-loss order rejected: {e}")
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing stop-loss order: {e}")
            raise ExchangeError(f"Failed to place stop-loss order: {e}")
        except Exception as e:
            logger.error(f"Error placing stop-loss order on Binance: {e!s}")
            raise ExecutionError(f"Failed to place stop-loss order: {e!s}")

    async def place_oco_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an OCO (One-Cancels-Other) order on Binance.

        Args:
            order: Order request with OCO details

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            # Validate order
            self._validate_oco_order(order)

            # Convert to Binance format
            binance_order = self._convert_oco_order_to_binance(order)

            # Place order
            result = await self.client.order_oco(
                symbol=order.symbol,
                side=order.side.value.upper(),
                quantity=str(order.quantity),
                price=str(order.price),
                stopPrice=str(order.stop_price),
                # Use stop price as limit price
                stopLimitPrice=str(order.stop_price),
                stopLimitTimeInForce=order.time_in_force,
                newClientOrderId=order.client_order_id,
            )

            # Convert response
            response = self._convert_binance_oco_order_to_response(result)

            # Track order
            self._track_order(response)

            logger.info(f"OCO order placed: {response.id}")
            return response

        except BinanceOrderException as e:
            logger.error(f"Binance OCO order error: {e}")
            raise OrderRejectionError(f"OCO order rejected: {e}")
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing OCO order: {e}")
            raise ExchangeError(f"Failed to place OCO order: {e}")
        except Exception as e:
            logger.error(f"Error placing OCO order on Binance: {e!s}")
            raise ExecutionError(f"Failed to place OCO order: {e!s}")

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order on Binance.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            # Cancel order
            result = await self.client.cancel_order(symbol=symbol, orderId=order_id)

            # Update tracking
            if order_id in self.pending_orders:
                order_info = self.pending_orders[order_id]
                order_info["status"] = "CANCELED"
                self.cancelled_orders[order_id] = order_info
                del self.pending_orders[order_id]

            logger.info(f"Order cancelled successfully: {order_id}")
            return True

        except BinanceAPIException as e:
            logger.error(f"Binance API error cancelling order: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cancelling order on Binance: {e!s}")
            return False

    async def get_order_status(self, order_id: str, symbol: str) -> OrderStatus:
        """
        Get order status from Binance.

        Args:
            order_id: Order ID to check
            symbol: Trading symbol

        Returns:
            OrderStatus: Current order status
        """
        try:
            # Get order status
            result = await self.client.get_order(symbol=symbol, orderId=order_id)

            # Convert to OrderStatus
            status = self._convert_binance_status_to_order_status(result["status"])

            # Update tracking
            if order_id in self.pending_orders:
                self.pending_orders[order_id]["status"] = result["status"]
                self.pending_orders[order_id]["executed_qty"] = result["executedQty"]

            return status

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting order status: {e}")
            raise ExchangeError(f"Failed to get order status: {e}")
        except Exception as e:
            logger.error(f"Error getting order status from Binance: {e!s}")
            raise ExchangeError(f"Failed to get order status: {e!s}")

    async def get_open_orders(self, symbol: str | None = None) -> list[OrderResponse]:
        """
        Get open orders from Binance.

        Args:
            symbol: Optional trading symbol filter

        Returns:
            List[OrderResponse]: List of open orders
        """
        try:
            # Get open orders
            if symbol:
                result = await self.client.get_open_orders(symbol=symbol)
            else:
                result = await self.client.get_open_orders()

            # Convert to OrderResponse list
            orders = []
            for order_data in result:
                order = self._convert_binance_order_to_response(order_data)
                orders.append(order)

            return orders

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting open orders: {e}")
            raise ExchangeError(f"Failed to get open orders: {e}")
        except Exception as e:
            logger.error(f"Error getting open orders from Binance: {e!s}")
            raise ExchangeError(f"Failed to get open orders: {e!s}")

    async def get_order_history(self, symbol: str, limit: int = 500) -> list[OrderResponse]:
        """
        Get order history from Binance.

        Args:
            symbol: Trading symbol
            limit: Number of orders to retrieve (max 1000)

        Returns:
            List[OrderResponse]: List of historical orders
        """
        try:
            # Get order history
            result = await self.client.get_all_orders(symbol=symbol, limit=limit)

            # Convert to OrderResponse list
            orders = []
            for order_data in result:
                order = self._convert_binance_order_to_response(order_data)
                orders.append(order)

            return orders

        except BinanceAPIException as e:
            logger.error(f"Binance API error getting order history: {e}")
            raise ExchangeError(f"Failed to get order history: {e}")
        except Exception as e:
            logger.error(f"Error getting order history from Binance: {e!s}")
            raise ExchangeError(f"Failed to get order history: {e!s}")

    def calculate_fees(self, order: OrderRequest, fill_price: Decimal) -> Decimal:
        """
        Calculate trading fees for an order.

        Args:
            order: Order request
            fill_price: Fill price for the order

        Returns:
            Decimal: Calculated fee amount
        """
        try:
            # Binance fee structure (simplified)
            # Maker: 0.1%, Taker: 0.1% (can be reduced with BNB)

            # Calculate fee based on order value
            order_value = order.quantity * fill_price
            fee_rate = Decimal("0.001")  # 0.1%
            fee = order_value * fee_rate

            return fee

        except Exception as e:
            logger.error(f"Error calculating fees: {e!s}")
            return Decimal("0")

    # Validation methods

    def _validate_market_order(self, order: OrderRequest) -> None:
        """Validate market order parameters."""
        if order.order_type != OrderType.MARKET:
            raise ValidationError("Order type must be MARKET for market orders")

        if not order.quantity or order.quantity <= 0:
            raise ValidationError("Market order must have positive quantity")

        if order.price:
            raise ValidationError("Market orders should not have a price")

    def _validate_limit_order(self, order: OrderRequest) -> None:
        """Validate limit order parameters."""
        if order.order_type != OrderType.LIMIT:
            raise ValidationError("Order type must be LIMIT for limit orders")

        if not order.quantity or order.quantity <= 0:
            raise ValidationError("Limit order must have positive quantity")

        if not order.price or order.price <= 0:
            raise ValidationError("Limit order must have positive price")

    def _validate_stop_loss_order(self, order: OrderRequest) -> None:
        """Validate stop-loss order parameters."""
        if order.order_type != OrderType.STOP_LOSS:
            raise ValidationError("Order type must be STOP_LOSS for stop-loss orders")

        if not order.quantity or order.quantity <= 0:
            raise ValidationError("Stop-loss order must have positive quantity")

        if not order.stop_price or order.stop_price <= 0:
            raise ValidationError("Stop-loss order must have positive stop price")

    def _validate_oco_order(self, order: OrderRequest) -> None:
        """Validate OCO order parameters."""
        if order.order_type != OrderType.LIMIT:
            raise ValidationError("OCO orders must be LIMIT type")

        if not order.quantity or order.quantity <= 0:
            raise ValidationError("OCO order must have positive quantity")

        if not order.price or order.price <= 0:
            raise ValidationError("OCO order must have positive limit price")

        if not order.stop_price or order.stop_price <= 0:
            raise ValidationError("OCO order must have positive stop price")

    # Conversion methods

    def _convert_market_order_to_binance(self, order: OrderRequest) -> dict[str, Any]:
        """Convert market order to Binance format."""
        return {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": "MARKET",
            "quantity": str(order.quantity),
            "newClientOrderId": order.client_order_id,
        }

    def _convert_limit_order_to_binance(self, order: OrderRequest) -> dict[str, Any]:
        """Convert limit order to Binance format."""
        return {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": "LIMIT",
            "quantity": str(order.quantity),
            "price": str(order.price),
            "timeInForce": order.time_in_force,
            "newClientOrderId": order.client_order_id,
        }

    def _convert_stop_loss_order_to_binance(self, order: OrderRequest) -> dict[str, Any]:
        """Convert stop-loss order to Binance format."""
        return {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "type": "STOP_LOSS",
            "quantity": str(order.quantity),
            "stopPrice": str(order.stop_price),
            "newClientOrderId": order.client_order_id,
        }

    def _convert_oco_order_to_binance(self, order: OrderRequest) -> dict[str, Any]:
        """Convert OCO order to Binance format."""
        return {
            "symbol": order.symbol,
            "side": order.side.value.upper(),
            "quantity": str(order.quantity),
            "price": str(order.price),
            "stopPrice": str(order.stop_price),
            "stopLimitPrice": str(order.stop_price),
            "stopLimitTimeInForce": order.time_in_force,
            "newClientOrderId": order.client_order_id,
        }

    def _convert_binance_order_to_response(self, result: dict) -> OrderResponse:
        """Convert Binance order result to OrderResponse."""
        return OrderResponse(
            id=str(result["orderId"]),
            client_order_id=result.get("clientOrderId"),
            symbol=result["symbol"],
            side=OrderSide.BUY if result["side"] == "BUY" else OrderSide.SELL,
            order_type=self._convert_binance_type_to_order_type(result["type"]),
            quantity=Decimal(str(result["origQty"])),
            price=Decimal(str(result["price"])) if result.get("price") else None,
            filled_quantity=Decimal(str(result["executedQty"])),
            status=result["status"],
            timestamp=datetime.fromtimestamp(result["time"] / 1000, tz=timezone.utc),
        )

    def _convert_binance_oco_order_to_response(self, result: dict) -> OrderResponse:
        """Convert Binance OCO order result to OrderResponse."""
        # OCO orders return a list of orders
        if isinstance(result, list) and len(result) > 0:
            # Use the first order as the primary response
            primary_order = result[0]
            return self._convert_binance_order_to_response(primary_order)
        else:
            # Fallback for single order response
            return self._convert_binance_order_to_response(result)

    def _convert_binance_type_to_order_type(self, binance_type: str) -> OrderType:
        """Convert Binance order type to OrderType enum."""
        type_mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
        }
        return type_mapping.get(binance_type, OrderType.LIMIT)

    def _convert_binance_status_to_order_status(self, status: str) -> OrderStatus:
        """Convert Binance order status to OrderStatus enum."""
        status_mapping = {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }
        return status_mapping.get(status, OrderStatus.UNKNOWN)

    # Order tracking methods

    def _track_order(self, order: OrderResponse) -> None:
        """Track an order for monitoring."""
        order_info = {
            "id": order.id,
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price else None,
            "status": order.status,
            "timestamp": order.timestamp.isoformat(),
        }

        self.pending_orders[order.id] = order_info
        logger.debug(f"Tracking order: {order.id}")

    def get_tracked_orders(self) -> dict[str, dict]:
        """Get all tracked orders."""
        return {
            "pending": self.pending_orders,
            "filled": self.filled_orders,
            "cancelled": self.cancelled_orders,
        }

    def clear_tracked_orders(self) -> None:
        """Clear all tracked orders."""
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()
        logger.info("Cleared all tracked orders")
