"""
Trading service for web interface business logic.

This service handles all trading-related business logic that was previously
embedded in controllers, ensuring proper separation of concerns.
"""

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError, ValidationError
from src.core.types import OrderSide, OrderType
from src.utils.formatters import format_price, format_quantity
from src.utils.validation import validate_price, validate_quantity, validate_symbol
from src.web_interface.interfaces import WebTradingServiceInterface


class WebTradingService(BaseComponent, WebTradingServiceInterface):
    """Service handling trading business logic for web interface."""

    def __init__(self, trading_facade=None):
        super().__init__()
        self.trading_facade = trading_facade

    async def initialize(self) -> None:
        """Initialize the service."""
        self.logger.info("Web trading service initialized")

    async def cleanup(self) -> None:
        """Cleanup the service."""
        self.logger.info("Web trading service cleaned up")

    async def place_order_through_service(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Place order through service layer (wraps facade call)."""
        try:
            if self.trading_facade:
                # Convert string parameters to enums
                side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                type_enum = OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT

                # Call facade through service
                order_id = await self.trading_facade.place_order(
                    symbol=symbol,
                    side=side_enum,
                    order_type=type_enum,
                    amount=quantity,
                    price=price,
                )
                return {"order_id": order_id, "success": True}
            else:
                # Mock implementation for development
                mock_order_id = f"order_{uuid.uuid4().hex[:8]}"
                self.logger.info(f"Mock order placed: {mock_order_id}")
                return {"order_id": mock_order_id, "success": True}

        except Exception as e:
            self.logger.error(f"Error placing order through service: {e}")
            raise ServiceError(f"Failed to place order: {e}")

    async def cancel_order_through_service(self, order_id: str) -> bool:
        """Cancel order through service layer (wraps facade call)."""
        try:
            if self.trading_facade:
                return await self.trading_facade.cancel_order(order_id)
            else:
                # Mock implementation for development
                self.logger.info(f"Mock order cancelled: {order_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error cancelling order through service: {e}")
            raise ServiceError(f"Failed to cancel order: {e}")

    async def get_service_health(self) -> dict[str, Any]:
        """Get service health status (wraps facade call)."""
        try:
            if self.trading_facade and hasattr(self.trading_facade, "health_check"):
                return await self.trading_facade.health_check()
            else:
                return {
                    "status": "healthy",
                    "facade_available": self.trading_facade is not None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error getting service health: {e}")
            return {
                "status": "error",
                "error": str(e),
                "facade_available": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def get_order_details(self, order_id: str, exchange: str) -> dict[str, Any]:
        """Get order details through service layer."""
        try:
            # In production, this would fetch from trading facade/database
            if self.trading_facade and hasattr(self.trading_facade, "get_order"):
                return await self.trading_facade.get_order(order_id)
            else:
                # Mock implementation for development - business logic in service
                mock_order_data = {
                    "order_id": order_id,
                    "client_order_id": f"web_{uuid.uuid4().hex[:8]}",
                    "symbol": "BTCUSDT",
                    "side": "buy",
                    "order_type": "limit",
                    "quantity": Decimal("1.0"),
                    "price": Decimal("45000.00"),
                    "stop_price": None,
                    "status": "filled",
                    "filled_quantity": Decimal("1.0"),
                    "remaining_quantity": Decimal("0.0"),
                    "average_fill_price": Decimal("44950.00"),
                    "commission": Decimal("0.1"),
                    "commission_asset": "USDT",
                    "exchange": exchange,
                    "bot_id": "bot_001",
                    "created_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    "updated_at": datetime.now(timezone.utc) - timedelta(hours=1),
                    "fills": [
                        {
                            "trade_id": "trade_001",
                            "quantity": "1.0",
                            "price": "44950.00",
                            "commission": "0.1",
                            "commission_asset": "USDT",
                            "executed_at": (
                                datetime.now(timezone.utc) - timedelta(hours=1)
                            ).isoformat(),
                        }
                    ],
                }
                return mock_order_data

        except Exception as e:
            self.logger.error(f"Error getting order details for {order_id}: {e}")
            raise ServiceError(f"Failed to get order details: {e}")

    async def validate_order_request(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> dict[str, Any]:
        """Validate order request with web-specific business logic and consistent data transformation."""
        try:
            validation_errors = []

            # Apply consistent data transformation pattern matching execution module
            from src.execution.data_transformer import ExecutionDataTransformer
            from src.utils.messaging_patterns import BoundaryValidator

            # Transform using execution module pattern for consistency
            validation_data = {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": str(quantity) if quantity else None,
                "price": str(price) if price else None,
                "processing_mode": "stream",  # Align with execution module
                "data_format": "event_data_v1",
                "boundary_crossed": True,
                "id": str(uuid.uuid4()),  # Required for database entity validation
                "component": "web_trading_service",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Apply consistent boundary fields using execution pattern
            validation_data = ExecutionDataTransformer.ensure_boundary_fields(
                validation_data, "web_interface"
            )

            # Validate financial precision consistently
            validation_data = ExecutionDataTransformer.validate_financial_precision(validation_data)

            # Validate at module boundary with enhanced validation
            BoundaryValidator.validate_database_entity(validation_data, "update")

            # Validate symbol
            try:
                validate_symbol(symbol)
            except ValidationError as e:
                validation_errors.append(f"Invalid symbol: {e}")

            # Validate quantity
            try:
                validate_quantity(quantity)
            except ValidationError as e:
                validation_errors.append(f"Invalid quantity: {e}")

            # Validate price if provided
            if price is not None:
                try:
                    validate_price(price)
                except ValidationError as e:
                    validation_errors.append(f"Invalid price: {e}")

            # Business logic: validate order type requirements
            if order_type.upper() in ["LIMIT", "STOP_LIMIT"] and price is None:
                validation_errors.append("Price is required for limit orders")

            # Business logic: validate minimum order size
            if quantity < Decimal("0.001"):
                validation_errors.append("Quantity must be at least 0.001")

            # Return with consistent data transformation pattern matching execution module
            validated_data = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "order_type": order_type.upper(),
                "quantity": quantity,
                "price": price,
                "processing_mode": "stream",
                "data_format": "event_data_v1",
                "timestamp": self._get_current_timestamp(),
                "message_pattern": "pub_sub",  # Consistent messaging pattern
                "boundary_crossed": True,
                "validation_status": "validated",
            }

            # Apply cross-module validation for consistency with execution
            validated_data = ExecutionDataTransformer.apply_cross_module_validation(
                validated_data, source_module="web_interface", target_module="execution"
            )

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "validated_data": validated_data,
            }

        except Exception as e:
            # Use consistent error propagation patterns
            from src.utils.messaging_patterns import ErrorPropagationMixin

            # Apply ErrorPropagationMixin patterns for consistency with execution module
            error_mixin = ErrorPropagationMixin()
            try:
                error_mixin.propagate_service_error(e, "order_validation")
            except ServiceError as service_error:
                self.logger.error(f"Error validating order request: {e}")
                raise service_error

    async def format_order_response(
        self, order_result: dict[str, Any], request_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Format order response with web-specific formatting."""
        try:
            # Business logic: format response with consistent structure
            formatted_quantity = format_quantity(request_data.get("quantity", Decimal("0")))
            formatted_price = (
                format_price(request_data.get("price")) if request_data.get("price") else None
            )

            return {
                "success": True,
                "message": "Order placed successfully",
                "order_id": order_result.get("order_id", f"web_{uuid.uuid4().hex[:8]}"),
                "client_order_id": request_data.get(
                    "client_order_id", f"web_{uuid.uuid4().hex[:8]}"
                ),
                "status": "submitted",
                "quantity": formatted_quantity,
                "price": formatted_price,
                "symbol": request_data.get("symbol"),
                "side": request_data.get("side"),
                "order_type": request_data.get("order_type"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            # Use consistent error propagation patterns
            from src.utils.messaging_patterns import ErrorPropagationMixin

            error_mixin = ErrorPropagationMixin()
            try:
                error_mixin.propagate_service_error(e, "order_response_formatting")
            except ServiceError as service_error:
                self.logger.error(f"Error formatting order response: {e}")
                raise service_error

    async def get_formatted_orders(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Get orders with web-specific formatting and business logic using consistent stream processing."""
        try:
            filters = filters or {}
            formatted_orders = []

            # Apply consistent processing paradigm alignment with execution module
            from src.execution.data_transformer import ExecutionDataTransformer

            # Align processing mode to stream for consistency
            processing_context = {
                "processing_mode": "stream",
                "data_format": "event_data_v1",
                "filters": filters,
                "component": "web_trading_service",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            aligned_context = ExecutionDataTransformer.align_processing_paradigm(
                processing_context, "stream"
            )

            # Mock order data generation (business logic for development)
            symbols = (
                [filters["symbol"]] if filters.get("symbol") else ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            )
            exchanges = (
                [filters["exchange"]] if filters.get("exchange") else ["binance", "coinbase"]
            )
            statuses = (
                [filters["status"]]
                if filters.get("status")
                else ["filled", "cancelled", "partially_filled"]
            )

            limit = filters.get("limit", 50)

            for i in range(min(limit, 20)):  # Business logic: limit mock data
                order_symbol = symbols[i % len(symbols)]
                order_exchange = exchanges[i % len(exchanges)]
                order_status = statuses[i % len(statuses)]

                # Business logic: calculate realistic order data
                quantity = Decimal("1.0") + Decimal(str(i * 0.1))
                filled_qty = quantity if order_status == "filled" else quantity * Decimal("0.5")

                # Apply bot_id filter
                bot_id = f"bot_{(i % 3) + 1:03d}" if i % 4 != 0 else None
                if filters.get("bot_id") and bot_id != filters.get("bot_id"):
                    continue

                formatted_order = {
                    "order_id": f"order_{i + 1:03d}",
                    "client_order_id": f"web_{uuid.uuid4().hex[:8]}",
                    "symbol": order_symbol,
                    "side": "buy" if i % 2 == 0 else "sell",
                    "order_type": "limit",
                    "quantity": quantity,
                    "price": Decimal("45000.00") + Decimal(str(i * 100)),
                    "stop_price": None,
                    "status": order_status,
                    "filled_quantity": filled_qty,
                    "remaining_quantity": quantity - filled_qty,
                    "average_fill_price": (
                        Decimal("45000.00") + Decimal(str(i * 50)) if filled_qty > 0 else None
                    ),
                    "commission": Decimal("0.1") if filled_qty > 0 else None,
                    "commission_asset": "USDT",
                    "exchange": order_exchange,
                    "bot_id": bot_id,
                    "created_at": datetime.now(timezone.utc) - timedelta(hours=i),
                    "updated_at": datetime.now(timezone.utc) - timedelta(hours=i, minutes=30),
                }

                formatted_orders.append(formatted_order)

            return formatted_orders[:limit]

        except Exception as e:
            self.logger.error(f"Error getting formatted orders: {e}")
            raise ServiceError(f"Failed to get formatted orders: {e}")

    async def get_formatted_trades(self, filters: dict[str, Any] = None) -> list[dict[str, Any]]:
        """Get trades with web-specific formatting and business logic."""
        try:
            filters = filters or {}
            formatted_trades = []

            # Mock trade data generation (business logic for development)
            symbols = (
                [filters["symbol"]] if filters.get("symbol") else ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            )
            exchanges = (
                [filters["exchange"]] if filters.get("exchange") else ["binance", "coinbase"]
            )

            limit = filters.get("limit", 50)
            start_date = filters.get("start_date")
            end_date = filters.get("end_date")

            for i in range(min(limit, 30)):  # Business logic: limit mock data
                trade_symbol = symbols[i % len(symbols)]
                trade_exchange = exchanges[i % len(exchanges)]

                executed_at = datetime.now(timezone.utc) - timedelta(hours=i, minutes=i * 5)

                # Apply date filters
                if start_date and executed_at < start_date:
                    continue
                if end_date and executed_at > end_date:
                    continue

                # Business logic: calculate realistic trade data
                quantity = Decimal("0.5") + Decimal(str(i * 0.1))
                price = Decimal("45000.00") + Decimal(str(i * 50))
                value = quantity * price

                # Apply bot_id filter
                bot_id = f"bot_{(i % 3) + 1:03d}" if i % 4 != 0 else None
                if filters.get("bot_id") and bot_id != filters.get("bot_id"):
                    continue

                formatted_trade = {
                    "trade_id": f"trade_{i + 1:03d}",
                    "order_id": f"order_{i + 1:03d}",
                    "symbol": trade_symbol,
                    "side": "buy" if i % 2 == 0 else "sell",
                    "quantity": quantity,
                    "price": price,
                    "value": value,
                    "commission": value * Decimal("0.001"),  # 0.1% commission
                    "commission_asset": "USDT",
                    "exchange": trade_exchange,
                    "bot_id": bot_id,
                    "executed_at": executed_at,
                }

                formatted_trades.append(formatted_trade)

            return formatted_trades[:limit]

        except Exception as e:
            self.logger.error(f"Error getting formatted trades: {e}")
            raise ServiceError(f"Failed to get formatted trades: {e}")

    async def get_market_data_with_context(
        self, symbol: str, exchange: str = "binance"
    ) -> dict[str, Any]:
        """Get market data with web-specific context and formatting."""
        try:
            # Business logic: determine base price based on symbol
            base_price = Decimal("45000.00") if "BTC" in symbol else Decimal("3000.00")

            # Business logic: add realistic price variation
            from random import uniform

            price_variation = Decimal(str(uniform(-0.02, 0.02)))  # ±2%
            current_price = base_price * (Decimal("1") + price_variation)

            # Business logic: calculate bid/ask spread
            bid = current_price * Decimal("0.9999")
            ask = current_price * Decimal("1.0001")

            # Business logic: calculate 24h metrics
            change_24h = current_price - base_price
            change_24h_percentage = (change_24h / base_price) * 100

            return {
                "symbol": symbol,
                "exchange": exchange,
                "price": current_price,
                "bid": bid,
                "ask": ask,
                "volume_24h": Decimal("1234567.89"),
                "change_24h": change_24h,
                "change_24h_percentage": change_24h_percentage,
                "high_24h": current_price * Decimal("1.05"),
                "low_24h": current_price * Decimal("0.95"),
                "timestamp": datetime.now(timezone.utc),
                "formatted_price": format_price(current_price),
                "formatted_volume": format_quantity(Decimal("1234567.89")),
            }

        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            raise ServiceError(f"Failed to get market data: {e}")

    async def generate_order_book_data(
        self, symbol: str, exchange: str, depth: int
    ) -> dict[str, Any]:
        """Generate order book data with web-specific business logic."""
        try:
            # Business logic: determine base price
            base_price = Decimal("45000.00") if "BTC" in symbol else Decimal("3000.00")

            # Generate bids (below current price) with business logic
            bids = []
            from random import uniform

            for i in range(depth):
                price = base_price * (Decimal("1") - Decimal(str(0.0001 * (i + 1))))
                quantity = Decimal(str(uniform(0.1, 5.0)))
                bids.append([price, quantity])

            # Generate asks (above current price) with business logic
            asks = []
            for i in range(depth):
                price = base_price * (Decimal("1") + Decimal(str(0.0001 * (i + 1))))
                quantity = Decimal(str(uniform(0.1, 5.0)))
                asks.append([price, quantity])

            return {
                "symbol": symbol,
                "exchange": exchange,
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now(timezone.utc),
                "depth": depth,
                "spread": asks[0][0] - bids[0][0] if bids and asks else Decimal("0"),
                "mid_price": (asks[0][0] + bids[0][0]) / 2 if bids and asks else base_price,
            }

        except Exception as e:
            self.logger.error(f"Error generating order book data for {symbol}: {e}")
            raise ServiceError(f"Failed to generate order book data: {e}")

    def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        return {
            "service": "WebTradingService",
            "status": "healthy",
            "trading_facade_available": self.trading_facade is not None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_service_info(self) -> dict[str, Any]:
        """Get service information and capabilities."""
        return {
            "service": "WebTradingService",
            "description": "Web trading service handling trading business logic",
            "capabilities": [
                "order_validation",
                "response_formatting",
                "order_management",
                "trade_history",
                "market_data",
                "order_book",
            ],
            "version": "1.0.0",
        }

    def _get_current_timestamp(self) -> str:
        """Get current timestamp for consistent data transformation."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
