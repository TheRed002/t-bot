"""
OKX Orders Implementation (P-005)

This module implements specialized order handling for OKX exchange,
including order type conversion, status tracking, and fee calculation.

CRITICAL: This integrates with P-001 (core types, exceptions, config), P-002A (error handling),
and P-003 (base exchange interface) components.

OKX Order Features:
- Order type conversion (market, limit, stop-loss, OCO)
- Time-in-force parameter handling
- Order status tracking and fill monitoring
- Partial fill handling and notification
- Order cancellation with confirmation
- Fee calculation and reporting
"""

import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

# OKX-specific imports
from okx.api import Trade as OKXTrade

from src.core.config import Config
from src.core.exceptions import ExchangeError, ExchangeInsufficientFundsError, ValidationError

# Logger setup
from src.core.logging import get_logger

# Logger is provided by BaseExchange (via BaseComponent)
# MANDATORY: Import from P-001
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.utils import ValidationFramework, normalize_price, round_to_precision


class OKXOrderManager:
    """
    OKX order manager for specialized order handling.

    Handles order-specific operations for OKX exchange:
    - Order type conversion and parameter mapping
    - Order status tracking and monitoring
    - Partial fill handling and notifications
    - Fee calculation and reporting
    - Order cancellation and confirmation
    """

    def __init__(self, config: Config, trade_client: OKXTrade):
        """
        Initialize OKX order manager.

        Args:
            config: Application configuration
            trade_client: OKX trade client instance
        """
        self.config = config
        self.trade_client = trade_client

        # Initialize logger
        self.logger = get_logger(self.__class__.__module__)

        # Order tracking
        self.active_orders: dict[str, dict] = {}
        self.order_history: dict[str, list[dict]] = {}

        # Fee structure (OKX-specific)
        self.maker_fee_rate = Decimal("0.001")  # 0.1% maker fee
        self.taker_fee_rate = Decimal("0.001")  # 0.1% taker fee

        # Initialize error handling
        self.error_handler = ErrorHandler(config)

        self.logger.info("Initialized OKX order manager")

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Place an order on OKX exchange.

        Args:
            order: Order request with all necessary details

        Returns:
            OrderResponse: Order response with execution details

        Raises:
            ExchangeError: If order placement fails
            ValidationError: If order request is invalid
        """
        try:
            # Validate order request using utils validators
            if not ValidationFramework.validate_order(order.__dict__):
                raise ValidationError("Order validation failed using utils validators")

            # Additional OKX-specific validation
            self._validate_order_request(order)

            # Convert order to OKX format
            okx_order = self._convert_order_to_okx(order)

            # Place order on OKX
            result = self.trade_client.place_order(**okx_order)

            if result.get("code") != "0":
                error_msg = result.get("msg", "Unknown error")
                if "insufficient" in error_msg.lower():
                    raise ExchangeInsufficientFundsError(f"Insufficient funds: {error_msg}")
                else:
                    raise ExchangeError(f"Order placement failed: {error_msg}")

            # Convert response to unified format
            order_response = self._convert_okx_order_to_response(result.get("data", [{}])[0])

            # Track active order
            self.active_orders[order_response.id] = {
                "order": order,
                "response": order_response,
                "timestamp": datetime.now(timezone.utc),
                "fills": [],
            }

            self.logger.info(f"Successfully placed order on OKX: {order_response.id}")
            return order_response

        except Exception as e:
            self.logger.error(f"Failed to place order on OKX: {e!s}")
            if isinstance(e, ExchangeError | ValidationError):
                raise
            raise ExchangeError(f"Failed to place order on OKX: {e!s}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on OKX.

        Args:
            order_id: ID of the order to cancel

        Returns:
            bool: True if cancellation successful, False otherwise
        """
        try:
            # Cancel order on OKX
            result = self.trade_client.cancel_order(ordId=order_id)

            if result.get("code") != "0":
                self.logger.warning(
                    f"Failed to cancel order {order_id}: {result.get('msg', 'Unknown error')}"
                )
                return False

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            self.logger.info(f"Successfully cancelled order on OKX: {order_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id} on OKX: {e!s}")
            return False

    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get the status of an order on OKX.

        Args:
            order_id: ID of the order to check

        Returns:
            OrderStatus: Current status of the order
        """
        try:
            # Get order details from OKX
            result = self.trade_client.get_order_details(ordId=order_id)

            if result.get("code") != "0":
                self.logger.warning(
                    f"Failed to get order status for {order_id}: "
                    f"{result.get('msg', 'Unknown error')}"
                )
                return OrderStatus.UNKNOWN

            data = result.get("data", [{}])[0]
            status = data.get("state", "")

            return self._convert_okx_status_to_order_status(status)

        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id} on OKX: {e!s}")
            return OrderStatus.UNKNOWN

    async def get_order_fills(self, order_id: str) -> list[OrderResponse]:
        """
        Get fill information for an order on OKX.

        Args:
            order_id: ID of the order to get fills for

        Returns:
            List[Trade]: List of trade fills for the order
        """
        try:
            # Get order fills from OKX
            result = self.trade_client.get_order_details(ordId=order_id)

            if result.get("code") != "0":
                self.logger.warning(
                    f"Failed to get order fills for {order_id}: "
                    f"{result.get('msg', 'Unknown error')}"
                )
                return []

            data = result.get("data", [{}])[0]
            fills = data.get("fills", [])

            trades: list[OrderResponse] = []
            for fill in fills:
                order_fill = OrderResponse(
                    id=fill.get("tradeId", ""),
                    client_order_id=None,
                    symbol=data.get("instId", ""),
                    side=OrderSide.BUY if data.get("side") == "buy" else OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=Decimal(fill.get("sz", "0")),
                    price=Decimal(fill.get("px", "0")),
                    filled_quantity=Decimal(fill.get("sz", "0")),
                    status="filled",
                    timestamp=datetime.fromtimestamp(
                        int(fill.get("ts", 0)) / 1000, tz=timezone.utc
                    ),
                )
                trades.append(order_fill)

            self.logger.debug(f"Retrieved {len(trades)} fills for order {order_id}")
            return trades

        except Exception as e:
            self.logger.error(f"Failed to get order fills for {order_id} on OKX: {e!s}")
            return []

    async def place_stop_loss_order(
        self, order: OrderRequest, stop_price: Decimal
    ) -> OrderResponse:
        """
        Place a stop-loss order on OKX.

        Args:
            order: Base order request
            stop_price: Stop price for the order

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            # Create stop-loss order
            stop_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.STOP_LOSS,
                quantity=order.quantity,
                price=order.price,
                stop_price=stop_price,
                client_order_id=order.client_order_id,
            )

            return await self.place_order(stop_order)

        except Exception as e:
            self.logger.error(f"Failed to place stop-loss order on OKX: {e!s}")
            raise ExchangeError(f"Failed to place stop-loss order on OKX: {e!s}")

    async def place_take_profit_order(
        self, order: OrderRequest, take_profit_price: Decimal
    ) -> OrderResponse:
        """
        Place a take-profit order on OKX.

        Args:
            order: Base order request
            take_profit_price: Take profit price for the order

        Returns:
            OrderResponse: Order response with execution details
        """
        try:
            # Create take-profit order
            tp_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.TAKE_PROFIT,
                quantity=order.quantity,
                price=order.price,
                stop_price=take_profit_price,
                client_order_id=order.client_order_id,
            )

            return await self.place_order(tp_order)

        except Exception as e:
            self.logger.error(f"Failed to place take-profit order on OKX: {e!s}")
            raise ExchangeError(f"Failed to place take-profit order on OKX: {e!s}")

    async def place_oco_order(
        self, order: OrderRequest, stop_price: Decimal, take_profit_price: Decimal
    ) -> list[OrderResponse]:
        """
        Place a One-Cancels-Other (OCO) order on OKX.

        Args:
            order: Base order request
            stop_price: Stop loss price
            take_profit_price: Take profit price

        Returns:
            List[OrderResponse]: List of order responses for the OCO orders
        """
        try:
            # OKX doesn't support OCO orders directly, so we place them separately
            # and track them as a group

            stop_order = await self.place_stop_loss_order(order, stop_price)
            tp_order = await self.place_take_profit_order(order, take_profit_price)

            # Track OCO relationship
            oco_group_id = f"oco_{int(time.time())}"
            self.active_orders[stop_order.id]["oco_group"] = oco_group_id
            self.active_orders[tp_order.id]["oco_group"] = oco_group_id

            self.logger.info(f"Placed OCO orders: stop={stop_order.id}, tp={tp_order.id}")
            return [stop_order, tp_order]

        except Exception as e:
            self.logger.error(f"Failed to place OCO order on OKX: {e!s}")
            raise ExchangeError(f"Failed to place OCO order on OKX: {e!s}")

    def calculate_fee(self, order: OrderRequest, is_maker: bool = False) -> Decimal:
        """
        Calculate trading fee for an order.

        Args:
            order: Order request
            is_maker: Whether the order is a maker order

        Returns:
            Decimal: Calculated fee amount
        """
        try:
            # Get fee rate based on order type
            fee_rate = self.maker_fee_rate if is_maker else self.taker_fee_rate

            # Calculate fee based on order value
            order_value = order.quantity * (order.price or Decimal("0"))
            fee = order_value * fee_rate

            self.logger.debug(f"Calculated fee for order: {fee} (rate: {fee_rate})")
            return fee

        except Exception as e:
            self.logger.error(f"Failed to calculate fee: {e!s}")
            return Decimal("0")

    def _validate_order_request(self, order: OrderRequest) -> None:
        """
        Validate order request for OKX-specific requirements.

        Args:
            order: Order request to validate

        Raises:
            ValidationError: If order request is invalid
        """
        try:
            # Check required fields
            if not order.symbol:
                raise ValidationError("Symbol is required")

            if not order.quantity or order.quantity <= 0:
                raise ValidationError("Quantity must be positive")

            if order.order_type == OrderType.LIMIT and not order.price:
                raise ValidationError("Price is required for limit orders")

            if (
                order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]
                and not order.stop_price
            ):
                raise ValidationError("Stop price is required for stop orders")

            # Check symbol format (OKX uses format like 'BTC-USDT')
            if "-" not in order.symbol:
                raise ValidationError(f"Invalid symbol format for OKX: {order.symbol}")

            self.logger.debug(f"Order validation passed for {order.symbol}")

        except Exception as e:
            self.logger.error(f"Order validation failed: {e!s}")
            raise ValidationError(f"Order validation failed: {e!s}")

    def _convert_order_to_okx(self, order: OrderRequest) -> dict[str, Any]:
        """
        Convert unified order request to OKX format.

        Args:
            order: Unified order request

        Returns:
            Dict[str, Any]: OKX-formatted order parameters
        """
        okx_order = {
            "instId": order.symbol,
            "tdMode": "cash",  # Spot trading
            "side": order.side.value.lower(),
            "ordType": self._convert_order_type_to_okx(order.order_type),
            "sz": str(round_to_precision(order.quantity, 8)),
        }

        # Add price for limit orders
        if order.price:
            okx_order["px"] = str(normalize_price(order.price, order.symbol))

        # Add stop price for stop orders
        if order.stop_price:
            okx_order["slTriggerPx"] = str(normalize_price(order.stop_price, order.symbol))
            okx_order["slOrdPx"] = str(
                normalize_price(order.price or order.stop_price, order.symbol)
            )

        # Add client order ID if provided
        if order.client_order_id:
            okx_order["clOrdId"] = order.client_order_id

        # Add time in force
        okx_order["tgtCcy"] = "USDT"  # Target currency for OKX

        return okx_order

    def _convert_okx_order_to_response(self, result: dict) -> OrderResponse:
        """
        Convert OKX order response to unified format.

        Args:
            result: OKX order response data

        Returns:
            OrderResponse: Unified order response
        """
        return OrderResponse(
            id=result.get("ordId", ""),
            client_order_id=result.get("clOrdId"),
            symbol=result.get("instId", ""),
            side=OrderSide.BUY if result.get("side") == "buy" else OrderSide.SELL,
            order_type=self._convert_okx_order_type_to_unified(result.get("ordType", "")),
            quantity=Decimal(result.get("sz", "0")),
            price=Decimal(result.get("px", "0")) if result.get("px") else None,
            filled_quantity=Decimal(result.get("accFillSz", "0")),
            status=self._convert_okx_status_to_order_status(result.get("state", "")).value,
            timestamp=datetime.now(timezone.utc),
        )

    def _convert_okx_status_to_order_status(self, status: str) -> OrderStatus:
        """
        Convert OKX order status to unified OrderStatus.

        Args:
            status: OKX order status string

        Returns:
            OrderStatus: Unified order status
        """
        status_mapping = {
            "live": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "expired": OrderStatus.EXPIRED,
            "failed": OrderStatus.REJECTED,
        }

        return status_mapping.get(status, OrderStatus.UNKNOWN)

    def _convert_order_type_to_okx(self, order_type: OrderType) -> str:
        """
        Convert unified order type to OKX format.

        Args:
            order_type: Unified order type

        Returns:
            str: OKX order type string
        """
        type_mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "conditional",
            OrderType.TAKE_PROFIT: "conditional",
        }

        return type_mapping.get(order_type, "limit")

    def _convert_okx_order_type_to_unified(self, okx_type: str) -> OrderType:
        """
        Convert OKX order type to unified format.

        Args:
            okx_type: OKX order type string

        Returns:
            OrderType: Unified order type
        """
        type_mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "conditional": OrderType.STOP_LOSS,  # Default mapping
            "post_only": OrderType.LIMIT,
            "fok": OrderType.LIMIT,
            "ioc": OrderType.LIMIT,
        }

        return type_mapping.get(okx_type, OrderType.LIMIT)

    def get_active_orders(self) -> dict[str, dict]:
        """
        Get all active orders.

        Returns:
            Dict[str, Dict]: Dictionary of active orders
        """
        return self.active_orders.copy()

    def get_order_history(self, symbol: str | None = None) -> list[dict]:
        """
        Get order history.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            List[Dict]: List of order history entries
        """
        history = []
        for _order_id, order_data in self.order_history.items():
            if symbol is None or order_data["response"].symbol == symbol:
                history.append(order_data)

        return history

    def clear_order_history(self) -> None:
        """Clear order history."""
        self.order_history.clear()
        self.logger.info("Cleared order history")

    def get_fee_rates(self) -> dict[str, Decimal]:
        """
        Get current fee rates.

        Returns:
            Dict[str, Decimal]: Dictionary of fee rates
        """
        return {"maker": self.maker_fee_rate, "taker": self.taker_fee_rate}

    def update_fee_rates(self, maker_rate: Decimal, taker_rate: Decimal) -> None:
        """
        Update fee rates.

        Args:
            maker_rate: New maker fee rate
            taker_rate: New taker fee rate
        """
        self.maker_fee_rate = maker_rate
        self.taker_fee_rate = taker_rate
        self.logger.info(f"Updated fee rates: maker={maker_rate}, taker={taker_rate}")
