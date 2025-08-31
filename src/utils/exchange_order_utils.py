"""
Common Exchange Order Management Utilities

This module contains shared order management utilities to eliminate duplication
across exchange implementations. It provides:
- Order validation and conversion patterns
- Common order response mapping
- Status conversion utilities
- Fee calculation helpers
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import OrderRequest, OrderResponse, OrderSide, OrderStatus, OrderType
from src.utils.decorators import retry
from src.utils.interfaces import ValidationServiceInterface

logger = get_logger(__name__)


class OrderManagementUtils:
    """Shared utilities for order management across exchanges.
    
    This class provides pure utility functions without business logic.
    Business validation should be handled by the service layer.
    """

    def __init__(self, validation_service: ValidationServiceInterface | None = None):
        """Initialize with injected validation service dependency.
        
        Args:
            validation_service: Injected validation service (required for dependency injection)
        """
        self.validation_service = validation_service

    def validate_order_request(self, order: OrderRequest, exchange_name: str = "") -> None:
        """
        DEPRECATED: Use ValidationService directly from service layer.
        
        This method violates service layer architecture by containing business logic.
        Instead, the service layer should call ValidationService directly.
        
        Args:
            order: Order request to validate
            exchange_name: Exchange name for context

        Raises:
            ValidationError: If validation fails
        """
        if not self.validation_service:
            raise ValidationError(
                "ValidationService must be injected. Use service layer for business validation.",
                error_code="SERV_001"
            )
        
        # Delegate to service layer - this should be called directly by services
        logger.warning("OrderManagementUtils.validate_order_request is deprecated. Use ValidationService directly in service layer.")
        
        # Basic structural validation only (not business validation)
        if not order.symbol or not order.symbol.strip():
            raise ValidationError("Symbol is required")
        if order.quantity <= 0:
            raise ValidationError("Quantity must be positive")
        if order.side not in [OrderSide.BUY, OrderSide.SELL]:
            raise ValidationError("Side must be BUY or SELL")

    @staticmethod
    def track_order(
        order_response: OrderResponse,
        pending_orders: dict[str, dict[str, Any]],
        order_request: OrderRequest | None = None,
    ) -> None:
        """
        Track an order in the pending orders dictionary.

        Args:
            order_response: Order response to track
            pending_orders: Dictionary to store pending orders
            order_request: Optional original order request
        """
        order_info: dict[str, Any] = {
            "id": order_response.id,
            "symbol": order_response.symbol,
            "side": order_response.side.value,
            "type": order_response.order_type.value,
            "quantity": str(order_response.quantity),
            "price": str(order_response.price) if order_response.price else None,
            "status": order_response.status,
            "timestamp": order_response.created_at.isoformat(),
        }

        if order_request:
            order_info["original_request"] = order_request

        pending_orders[order_response.id] = order_info

    @staticmethod
    def update_order_status(order_id: str, status: OrderStatus, pending_orders: dict[str, dict[str, Any]]) -> None:
        """
        Update order status in tracking dictionary.

        Args:
            order_id: Order ID to update
            status: New order status
            pending_orders: Pending orders dictionary
        """
        if order_id in pending_orders:
            pending_orders[order_id]["status"] = status.value


class OrderConversionUtils:
    """Utilities for converting orders between exchange formats."""

    @staticmethod
    def create_base_order_response(
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        client_order_id: str | None = None,
        price: Decimal | None = None,
        filled_quantity: Decimal | None = None,
        status: OrderStatus = OrderStatus.PENDING,
        timestamp: datetime | None = None,
    ) -> OrderResponse:
        """
        Create a standardized OrderResponse with common fields.

        Args:
            order_id: Exchange order ID
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type
            quantity: Order quantity
            client_order_id: Optional client order ID
            price: Order price (for limit orders)
            filled_quantity: Filled quantity
            status: Order status
            timestamp: Order timestamp

        Returns:
            OrderResponse: Standardized order response
        """
        return OrderResponse(
            id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=filled_quantity or Decimal("0"),
            status=status,
            created_at=timestamp or datetime.now(timezone.utc),
            exchange="unknown",  # Required field
        )

    @staticmethod
    def standardize_symbol_format(symbol: str, target_format: str = "dash") -> str:
        """
        Convert symbol between different formats.

        Args:
            symbol: Symbol to convert
            target_format: Target format ("dash" or "concat")

        Returns:
            str: Symbol in target format
        """
        if target_format == "dash":
            # Convert BTCUSDT to BTC-USDT
            if "-" not in symbol:
                # Common mappings
                mappings = {
                    "BTCUSDT": "BTC-USDT",
                    "ETHUSDT": "ETH-USDT",
                    "BNBUSDT": "BNB-USDT",
                    "ADAUSDT": "ADA-USDT",
                    "DOTUSDT": "DOT-USDT",
                    "LINKUSDT": "LINK-USDT",
                    "LTCUSDT": "LTC-USDT",
                    "SOLUSDT": "SOL-USDT",
                    "XRPUSDT": "XRP-USDT",
                    "BTCUSD": "BTC-USD",
                    "ETHUSD": "ETH-USD",
                    "LTCUSD": "LTC-USD",
                }

                if symbol in mappings:
                    return mappings[symbol]

                # Generic conversion
                if len(symbol) >= 6:
                    for quote in ["USDT", "USDC", "USD", "BTC", "ETH"]:
                        if symbol.endswith(quote):
                            base = symbol[: -len(quote)]
                            return f"{base}-{quote}"
            return symbol

        elif target_format == "concat":
            # Convert BTC-USDT to BTCUSDT
            return symbol.replace("-", "")

        return symbol


class OrderStatusUtils:
    """Utilities for handling order status conversions."""

    @staticmethod
    def convert_status(status: str, exchange: str) -> OrderStatus:
        """
        Convert exchange-specific status to unified OrderStatus.

        Args:
            status: Exchange-specific status string
            exchange: Exchange name

        Returns:
            OrderStatus: Unified order status
        """
        # Normalize to lowercase
        status_lower = status.lower()

        # Common mappings across exchanges
        common_mappings = {
            "new": OrderStatus.PENDING,
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.PENDING,
            "active": OrderStatus.PENDING,
            "live": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "done": OrderStatus.FILLED,
            "settled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "partial": OrderStatus.PARTIALLY_FILLED,
        }

        # Try common mapping first
        if status_lower in common_mappings:
            return common_mappings[status_lower]

        # Exchange-specific mappings
        if exchange == "binance":
            binance_mappings = {"partially_filled": OrderStatus.PARTIALLY_FILLED}
            return binance_mappings.get(status_lower, OrderStatus.REJECTED)

        elif exchange == "coinbase":
            # Coinbase specific statuses already covered in common mappings
            pass

        elif exchange == "okx":
            # OKX specific statuses already covered in common mappings
            pass

        return OrderStatus.REJECTED

    @staticmethod
    def is_terminal_status(status: OrderStatus) -> bool:
        """
        Check if order status is terminal (no further updates expected).

        Args:
            status: Order status to check

        Returns:
            bool: True if status is terminal
        """
        terminal_statuses = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
        return status in terminal_statuses


class OrderTypeUtils:
    """Utilities for order type conversions."""

    @staticmethod
    def convert_to_exchange_format(order_type: OrderType, exchange: str) -> str:
        """
        Convert unified OrderType to exchange-specific format.

        Args:
            order_type: Unified order type
            exchange: Exchange name

        Returns:
            str: Exchange-specific order type string
        """
        if exchange == "binance":
            mappings = {
                OrderType.MARKET: "MARKET",
                OrderType.LIMIT: "LIMIT",
                OrderType.STOP_LOSS: "STOP_LOSS",
                OrderType.TAKE_PROFIT: "TAKE_PROFIT",
            }
        elif exchange == "coinbase":
            mappings = {
                OrderType.MARKET: "market",
                OrderType.LIMIT: "limit",
                OrderType.STOP_LOSS: "stop",
                OrderType.TAKE_PROFIT: "take_profit",
            }
        elif exchange == "okx":
            mappings = {
                OrderType.MARKET: "market",
                OrderType.LIMIT: "limit",
                OrderType.STOP_LOSS: "conditional",
                OrderType.TAKE_PROFIT: "conditional",
            }
        else:
            # Default mappings
            mappings = {
                OrderType.MARKET: "market",
                OrderType.LIMIT: "limit",
                OrderType.STOP_LOSS: "stop_loss",
                OrderType.TAKE_PROFIT: "take_profit",
            }

        return mappings.get(order_type, "limit")

    @staticmethod
    def convert_from_exchange_format(exchange_type: str, exchange: str) -> OrderType:
        """
        Convert exchange-specific order type to unified OrderType.

        Args:
            exchange_type: Exchange-specific order type
            exchange: Exchange name

        Returns:
            OrderType: Unified order type
        """
        exchange_type_lower = exchange_type.lower()

        # Common mappings
        common_mappings = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_loss": OrderType.STOP_LOSS,
            "take_profit": OrderType.TAKE_PROFIT,
            "conditional": OrderType.STOP_LOSS,  # Common in many exchanges
            "stop": OrderType.STOP_LOSS,
        }

        return common_mappings.get(exchange_type_lower, OrderType.LIMIT)


class FeeCalculationUtils:
    """Utilities for fee calculations."""

    # Default fee structures for exchanges
    DEFAULT_FEE_RATES = {
        "binance": {"maker": Decimal("0.001"), "taker": Decimal("0.001")},
        "coinbase": {"maker": Decimal("0.004"), "taker": Decimal("0.006")},
        "okx": {"maker": Decimal("0.001"), "taker": Decimal("0.001")},
        "default": {"maker": Decimal("0.001"), "taker": Decimal("0.001")},
    }

    @staticmethod
    @retry(max_attempts=2, delay=0.1)
    def calculate_fee(order_value: Decimal, exchange: str, is_maker: bool = False) -> Decimal:
        """
        Calculate trading fee for an order.

        Args:
            order_value: Total order value (quantity * price)
            exchange: Exchange name
            is_maker: Whether this is a maker order

        Returns:
            Decimal: Calculated fee amount
        """
        try:
            fee_rates = FeeCalculationUtils.DEFAULT_FEE_RATES.get(
                exchange, FeeCalculationUtils.DEFAULT_FEE_RATES["default"]
            )

            fee_rate = fee_rates["maker"] if is_maker else fee_rates["taker"]
            return order_value * fee_rate

        except Exception as e:
            # Return zero fee on calculation error
            logger.warning(f"Fee calculation failed for {exchange}: {e}")
            return Decimal("0")

    @staticmethod
    def get_fee_rates(exchange: str) -> dict[str, Decimal]:
        """
        Get fee rates for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            Dict[str, Decimal]: Fee rates dictionary
        """
        return FeeCalculationUtils.DEFAULT_FEE_RATES.get(
            exchange, FeeCalculationUtils.DEFAULT_FEE_RATES["default"]
        ).copy()


# Factory function for dependency injection - SERVICE LAYER USE ONLY
def get_order_management_utils(validation_service: ValidationServiceInterface | None = None) -> OrderManagementUtils:
    """
    Factory function to create OrderManagementUtils with proper dependency injection.
    
    This should only be called from the service layer with proper dependency injection.
    Direct DI container access violates clean architecture.

    Args:
        validation_service: Validation service to inject (required from service layer)
        
    Returns:
        OrderManagementUtils: Instance with injected ValidationService
        
    Raises:
        ValidationError: If called without proper service injection
    """
    if validation_service is None:
        raise ValidationError(
            "OrderManagementUtils requires ValidationService injection from service layer. "
            "Do not call this factory directly - use through service layer.",
            error_code="SERV_001"
        )
    
    return OrderManagementUtils(validation_service=validation_service)
