"""
Exchange Conversion Utilities

Common conversion utilities for transforming data between unified formats
and exchange-specific formats. Eliminates duplication across exchange implementations.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError
from src.core.types import (
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    Ticker,
)
from src.utils.data_utils import normalize_price
from src.utils.decimal_utils import round_to_precision


class SymbolConversionUtils:
    """Utilities for converting symbols between different exchange formats."""

    # Common symbol mappings
    COMMON_SYMBOL_MAPPINGS = {
        # Binance to standard mappings
        "binance": {
            "BTCUSDT": "BTC/USDT",
            "ETHUSDT": "ETH/USDT",
            "BNBUSDT": "BNB/USDT",
            "ADAUSDT": "ADA/USDT",
            "DOTUSDT": "DOT/USDT",
            "LINKUSDT": "LINK/USDT",
            "LTCUSDT": "LTC/USDT",
            "SOLUSDT": "SOL/USDT",
            "XRPUSDT": "XRP/USDT",
        },
        # Coinbase format
        "coinbase": {
            "BTC-USD": "BTC/USD",
            "ETH-USD": "ETH/USD",
            "LTC-USD": "LTC/USD",
            "BTC-USDT": "BTC/USDT",
            "ETH-USDT": "ETH/USDT",
            "ADA-USD": "ADA/USD",
        },
        # OKX format
        "okx": {
            "BTC-USDT": "BTC/USDT",
            "ETH-USDT": "ETH/USDT",
            "BNB-USDT": "BNB/USDT",
            "ADA-USDT": "ADA/USDT",
            "DOT-USDT": "DOT/USDT",
            "LINK-USDT": "LINK/USDT",
        },
    }

    @staticmethod
    def normalize_symbol(symbol: str, target_exchange: str) -> str:
        """
        Convert symbol to target exchange format.

        Args:
            symbol: Symbol to convert
            target_exchange: Target exchange format

        Returns:
            str: Symbol in target exchange format
        """
        if target_exchange == "binance":
            return SymbolConversionUtils.to_binance_format(symbol)
        elif target_exchange == "coinbase":
            return SymbolConversionUtils.to_coinbase_format(symbol)
        elif target_exchange == "okx":
            return SymbolConversionUtils.to_okx_format(symbol)
        else:
            return symbol

    @staticmethod
    def to_binance_format(symbol: str) -> str:
        """Convert symbol to Binance format (BTCUSDT)."""
        # Remove separators
        if "/" in symbol:
            return symbol.replace("/", "")
        elif "-" in symbol:
            return symbol.replace("-", "")
        return symbol

    @staticmethod
    def to_coinbase_format(symbol: str) -> str:
        """Convert symbol to Coinbase format (BTC-USD)."""
        if "-" in symbol:
            return symbol
        elif "/" in symbol:
            return symbol.replace("/", "-")
        else:
            # Parse concatenated format
            return SymbolConversionUtils._parse_concatenated_symbol(symbol, "-")

    @staticmethod
    def to_okx_format(symbol: str) -> str:
        """Convert symbol to OKX format (BTC-USDT)."""
        if "-" in symbol:
            return symbol
        elif "/" in symbol:
            return symbol.replace("/", "-")
        else:
            # Parse concatenated format
            return SymbolConversionUtils._parse_concatenated_symbol(symbol, "-")

    @staticmethod
    def to_standard_format(symbol: str) -> str:
        """Convert symbol to standard format (BTC/USDT)."""
        if "/" in symbol:
            return symbol
        elif "-" in symbol:
            return symbol.replace("-", "/")
        else:
            # Parse concatenated format
            return SymbolConversionUtils._parse_concatenated_symbol(symbol, "/")

    @staticmethod
    def _parse_concatenated_symbol(symbol: str, separator: str) -> str:
        """Parse concatenated symbol format like BTCUSDT."""
        # Common quote currencies in order of preference
        quote_currencies = ["USDT", "USDC", "USD", "BTC", "ETH", "BNB"]

        for quote in quote_currencies:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[: -len(quote)]
                return f"{base}{separator}{quote}"

        # If no match, return original
        return symbol

    @staticmethod
    def get_base_quote(symbol: str) -> tuple[str, str]:
        """
        Extract base and quote currencies from symbol.

        Args:
            symbol: Trading symbol

        Returns:
            tuple[str, str]: Base and quote currencies
        """
        if "/" in symbol:
            return tuple(symbol.split("/", 1))
        elif "-" in symbol:
            return tuple(symbol.split("-", 1))
        else:
            # Parse concatenated format
            quote_currencies = ["USDT", "USDC", "USD", "BTC", "ETH", "BNB"]
            for quote in quote_currencies:
                if symbol.endswith(quote) and len(symbol) > len(quote):
                    base = symbol[: -len(quote)]
                    return (base, quote)

        # Default fallback
        return (symbol, "")


class OrderConversionUtils:
    """Utilities for converting order parameters between exchanges."""

    @staticmethod
    def convert_order_to_exchange_format(
        order: OrderRequest, exchange: str, precision_config: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """
        Convert unified order request to exchange-specific format.

        Args:
            order: Unified order request
            exchange: Target exchange
            precision_config: Optional precision configuration

        Returns:
            Dict[str, Any]: Exchange-specific order parameters
        """
        if exchange == "binance":
            return OrderConversionUtils._to_binance_order(order, precision_config)
        elif exchange == "coinbase":
            return OrderConversionUtils._to_coinbase_order(order, precision_config)
        elif exchange == "okx":
            return OrderConversionUtils._to_okx_order(order, precision_config)
        else:
            # Generic format
            return OrderConversionUtils._to_generic_order(order, precision_config)

    @staticmethod
    def _to_binance_order(
        order: OrderRequest, precision_config: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Convert to Binance order format."""
        precision = precision_config.get("quantity", 8) if precision_config else 8

        binance_order = {
            "symbol": SymbolConversionUtils.to_binance_format(order.symbol),
            "side": order.side.value.upper(),
            "type": OrderConversionUtils._get_binance_order_type(order.order_type),
            "quantity": str(round_to_precision(order.quantity, precision)),
        }

        if order.price and order.order_type == OrderType.LIMIT:
            price_precision = precision_config.get("price", 8) if precision_config else 8
            binance_order["price"] = str(
                normalize_price(order.price, order.symbol, price_precision)
            )
            binance_order["timeInForce"] = order.time_in_force or "GTC"

        if order.stop_price and order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            stop_precision = precision_config.get("price", 8) if precision_config else 8
            binance_order["stopPrice"] = str(
                normalize_price(order.stop_price, order.symbol, stop_precision)
            )

        if order.client_order_id:
            binance_order["newClientOrderId"] = order.client_order_id

        return binance_order

    @staticmethod
    def _to_coinbase_order(
        order: OrderRequest, precision_config: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Convert to Coinbase order format."""
        precision = precision_config.get("quantity", 8) if precision_config else 8

        coinbase_order = {
            "product_id": SymbolConversionUtils.to_coinbase_format(order.symbol),
            "side": order.side.value.lower(),
            "order_configuration": {},
        }

        if order.order_type == OrderType.MARKET:
            size_key = "quote_size" if order.side == OrderSide.BUY else "base_size"
            coinbase_order["order_configuration"] = {
                "market_market_ioc": {size_key: str(round_to_precision(order.quantity, precision))}
            }
        elif order.order_type == OrderType.LIMIT:
            price_precision = precision_config.get("price", 8) if precision_config else 8
            coinbase_order["order_configuration"] = {
                "limit_limit_gtc": {
                    "base_size": str(round_to_precision(order.quantity, precision)),
                    "limit_price": str(normalize_price(order.price, order.symbol, price_precision)),
                }
            }

        if order.client_order_id:
            coinbase_order["client_order_id"] = order.client_order_id

        return coinbase_order

    @staticmethod
    def _to_okx_order(
        order: OrderRequest, precision_config: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Convert to OKX order format."""
        precision = precision_config.get("quantity", 8) if precision_config else 8

        okx_order = {
            "instId": SymbolConversionUtils.to_okx_format(order.symbol),
            "tdMode": "cash",  # Spot trading
            "side": order.side.value.lower(),
            "ordType": OrderConversionUtils._get_okx_order_type(order.order_type),
            "sz": str(round_to_precision(order.quantity, precision)),
        }

        if order.price and order.order_type == OrderType.LIMIT:
            price_precision = precision_config.get("price", 8) if precision_config else 8
            okx_order["px"] = str(normalize_price(order.price, order.symbol, price_precision))

        if order.stop_price and order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            stop_precision = precision_config.get("price", 8) if precision_config else 8
            okx_order["slTriggerPx"] = str(
                normalize_price(order.stop_price, order.symbol, stop_precision)
            )

        if order.client_order_id:
            okx_order["clOrdId"] = order.client_order_id

        return okx_order

    @staticmethod
    def _to_generic_order(
        order: OrderRequest, precision_config: dict[str, int] | None = None
    ) -> dict[str, Any]:
        """Convert to generic order format."""
        precision = precision_config.get("quantity", 8) if precision_config else 8

        generic_order = {
            "symbol": order.symbol,
            "side": order.side.value.lower(),
            "type": order.order_type.value.lower(),
            "quantity": str(round_to_precision(order.quantity, precision)),
        }

        if order.price:
            price_precision = precision_config.get("price", 8) if precision_config else 8
            generic_order["price"] = str(
                normalize_price(order.price, order.symbol, price_precision)
            )

        if order.stop_price:
            stop_precision = precision_config.get("price", 8) if precision_config else 8
            generic_order["stop_price"] = str(
                normalize_price(order.stop_price, order.symbol, stop_precision)
            )

        if order.client_order_id:
            generic_order["client_order_id"] = order.client_order_id

        return generic_order

    @staticmethod
    def _get_binance_order_type(order_type: OrderType) -> str:
        """Get Binance order type string."""
        type_mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
        }
        return type_mapping.get(order_type, "LIMIT")

    @staticmethod
    def _get_okx_order_type(order_type: OrderType) -> str:
        """Get OKX order type string."""
        type_mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "conditional",
            OrderType.TAKE_PROFIT: "conditional",
        }
        return type_mapping.get(order_type, "limit")


class ResponseConversionUtils:
    """Utilities for converting exchange responses to unified formats."""

    @staticmethod
    def create_unified_order_response(
        exchange_response: dict[str, Any], exchange: str, original_symbol: str = None
    ) -> OrderResponse:
        """
        Convert exchange-specific order response to unified format.

        Args:
            exchange_response: Raw exchange response
            exchange: Exchange name
            original_symbol: Original symbol for conversion

        Returns:
            OrderResponse: Unified order response
        """
        if exchange == "binance":
            return ResponseConversionUtils._from_binance_response(exchange_response)
        elif exchange == "coinbase":
            return ResponseConversionUtils._from_coinbase_response(
                exchange_response, original_symbol
            )
        elif exchange == "okx":
            return ResponseConversionUtils._from_okx_response(exchange_response)
        else:
            return ResponseConversionUtils._from_generic_response(exchange_response)

    @staticmethod
    def _from_binance_response(response: dict[str, Any]) -> OrderResponse:
        """Convert Binance response to unified format."""
        return OrderResponse(
            id=str(response.get("orderId", "")),
            client_order_id=response.get("clientOrderId"),
            symbol=response.get("symbol", ""),
            side=OrderSide.BUY if response.get("side") == "BUY" else OrderSide.SELL,
            order_type=ResponseConversionUtils._parse_binance_order_type(response.get("type", "")),
            quantity=Decimal(str(response.get("origQty", "0"))),
            price=Decimal(str(response.get("price", "0"))) if response.get("price") else None,
            filled_quantity=Decimal(str(response.get("executedQty", "0"))),
            status=response.get("status", "pending"),
            timestamp=ResponseConversionUtils._parse_timestamp(
                response.get("transactTime") or response.get("time"), format="milliseconds"
            ),
        )

    @staticmethod
    def _from_coinbase_response(
        response: dict[str, Any], original_symbol: str = None
    ) -> OrderResponse:
        """Convert Coinbase response to unified format."""
        # Parse order configuration for quantity and price
        order_config = response.get("order_configuration", {})
        quantity = Decimal("0")
        price = None

        if "market_market_ioc" in order_config:
            market_config = order_config["market_market_ioc"]
            quantity = Decimal(str(market_config.get("quote_size", "0")))
        elif "limit_limit_gtc" in order_config:
            limit_config = order_config["limit_limit_gtc"]
            quantity = Decimal(str(limit_config.get("base_size", "0")))
            price = Decimal(str(limit_config.get("limit_price", "0")))

        return OrderResponse(
            id=response.get("order_id", ""),
            client_order_id=response.get("client_order_id"),
            symbol=original_symbol or response.get("product_id", ""),
            side=OrderSide.BUY if response.get("side") == "BUY" else OrderSide.SELL,
            order_type=ResponseConversionUtils._parse_coinbase_order_type(order_config),
            quantity=quantity,
            price=price,
            filled_quantity=Decimal(str(response.get("filled_size", "0"))),
            status=response.get("status", "pending"),
            timestamp=ResponseConversionUtils._parse_timestamp(
                response.get("created_time"), format="iso"
            ),
        )

    @staticmethod
    def _from_okx_response(response: dict[str, Any]) -> OrderResponse:
        """Convert OKX response to unified format."""
        return OrderResponse(
            id=response.get("ordId", ""),
            client_order_id=response.get("clOrdId"),
            symbol=response.get("instId", ""),
            side=OrderSide.BUY if response.get("side") == "buy" else OrderSide.SELL,
            order_type=ResponseConversionUtils._parse_okx_order_type(response.get("ordType", "")),
            quantity=Decimal(str(response.get("sz", "0"))),
            price=Decimal(str(response.get("px", "0"))) if response.get("px") else None,
            filled_quantity=Decimal(str(response.get("accFillSz", "0"))),
            status=response.get("state", "pending"),
            timestamp=ResponseConversionUtils._parse_timestamp(
                response.get("cTime") or response.get("uTime"), format="milliseconds"
            ),
        )

    @staticmethod
    def _from_generic_response(response: dict[str, Any]) -> OrderResponse:
        """Convert generic response to unified format."""
        return OrderResponse(
            id=str(response.get("id", response.get("order_id", ""))),
            client_order_id=response.get("client_order_id"),
            symbol=response.get("symbol", ""),
            side=OrderSide.BUY if response.get("side", "").upper() == "BUY" else OrderSide.SELL,
            order_type=OrderType.LIMIT,  # Default
            quantity=Decimal(str(response.get("quantity", "0"))),
            price=Decimal(str(response.get("price", "0"))) if response.get("price") else None,
            filled_quantity=Decimal(str(response.get("filled_quantity", "0"))),
            status=response.get("status", "pending"),
            timestamp=datetime.now(timezone.utc),
        )

    @staticmethod
    def _parse_binance_order_type(binance_type: str) -> OrderType:
        """Parse Binance order type."""
        type_mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
        }
        return type_mapping.get(binance_type, OrderType.LIMIT)

    @staticmethod
    def _parse_coinbase_order_type(order_config: dict[str, Any]) -> OrderType:
        """Parse Coinbase order type from order configuration."""
        if "market_market_ioc" in order_config:
            return OrderType.MARKET
        elif "limit_limit_gtc" in order_config:
            return OrderType.LIMIT
        elif "stop_limit_stop_limit_gtc" in order_config:
            return OrderType.STOP_LOSS
        return OrderType.LIMIT

    @staticmethod
    def _parse_okx_order_type(okx_type: str) -> OrderType:
        """Parse OKX order type."""
        type_mapping = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "conditional": OrderType.STOP_LOSS,
        }
        return type_mapping.get(okx_type, OrderType.LIMIT)

    @staticmethod
    def _parse_timestamp(timestamp_str: str | int | None, format: str = "milliseconds") -> datetime:
        """Parse timestamp from various formats."""
        if not timestamp_str:
            return datetime.now(timezone.utc)

        try:
            if format == "milliseconds":
                timestamp_ms = int(timestamp_str)
                return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            elif format == "seconds":
                timestamp_s = int(timestamp_str)
                return datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
            elif format == "iso":
                # Handle ISO format with various endings
                timestamp_str = str(timestamp_str).replace("Z", "+00:00")
                return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            pass

        return datetime.now(timezone.utc)


class MarketDataConversionUtils:
    """Utilities for converting market data between formats."""

    @staticmethod
    def create_unified_ticker(exchange_data: dict[str, Any], exchange: str, symbol: str) -> Ticker:
        """
        Create unified Ticker from exchange-specific data.

        Args:
            exchange_data: Raw exchange ticker data
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Ticker: Unified ticker object
        """
        # Common fields with fallbacks
        return Ticker(
            symbol=symbol,
            bid_price=Decimal(str(exchange_data.get("bidPrice", exchange_data.get("bid", "0")))),
            bid_quantity=Decimal(
                str(exchange_data.get("bidQty", exchange_data.get("bidSize", "0")))
            ),
            ask_price=Decimal(str(exchange_data.get("askPrice", exchange_data.get("ask", "0")))),
            ask_quantity=Decimal(
                str(exchange_data.get("askQty", exchange_data.get("askSize", "0")))
            ),
            last_price=Decimal(
                str(
                    exchange_data.get(
                        "lastPrice", exchange_data.get("price", exchange_data.get("last", "0"))
                    )
                )
            ),
            volume=Decimal(str(exchange_data.get("volume", exchange_data.get("vol24h", "0")))),
            timestamp=ResponseConversionUtils._parse_timestamp(
                exchange_data.get("closeTime")
                or exchange_data.get("timestamp")
                or exchange_data.get("ts")
            ),
            exchange=exchange,
        )

    @staticmethod
    def create_unified_order_book(
        exchange_data: dict[str, Any], symbol: str, exchange: str
    ) -> OrderBook:
        """
        Create unified OrderBook from exchange-specific data.

        Args:
            exchange_data: Raw exchange order book data
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            OrderBook: Unified order book object
        """
        # Convert bids and asks to OrderBookLevel objects
        bids = []
        asks = []

        raw_bids = exchange_data.get("bids", [])
        raw_asks = exchange_data.get("asks", [])

        for bid in raw_bids:
            if isinstance(bid, list) and len(bid) >= 2:
                bids.append(
                    OrderBookLevel(price=Decimal(str(bid[0])), quantity=Decimal(str(bid[1])))
                )

        for ask in raw_asks:
            if isinstance(ask, list) and len(ask) >= 2:
                asks.append(
                    OrderBookLevel(price=Decimal(str(ask[0])), quantity=Decimal(str(ask[1])))
                )

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=ResponseConversionUtils._parse_timestamp(
                exchange_data.get("timestamp") or exchange_data.get("ts")
            ),
            exchange=exchange,
        )


class ExchangeConversionUtils:
    """Utilities for converting between unified and exchange-specific formats."""

    @staticmethod
    def create_order_response(
        exchange_data: dict[str, Any],
        field_mapping: dict[str, str],
        status_mapping: dict[str, OrderStatus],
        type_mapping: dict[str, OrderType],
        exchange_name: str = "",
    ) -> OrderResponse:
        """
        Create unified OrderResponse from exchange-specific data.

        Args:
            exchange_data: Raw exchange response data
            field_mapping: Mapping of unified fields to exchange fields
            status_mapping: Mapping of exchange statuses to unified statuses
            type_mapping: Mapping of exchange types to unified types
            exchange_name: Exchange name for debugging

        Returns:
            OrderResponse: Unified order response
        """
        try:
            # Extract fields using mapping
            order_id = str(exchange_data.get(field_mapping.get("id", "orderId"), ""))
            client_order_id = exchange_data.get(field_mapping.get("client_order_id"))
            symbol = exchange_data.get(field_mapping.get("symbol", "symbol"), "")

            # Convert side
            side_value = exchange_data.get(field_mapping.get("side", "side"), "").upper()
            side = OrderSide.BUY if side_value in ["BUY", "buy"] else OrderSide.SELL

            # Convert order type
            type_value = exchange_data.get(field_mapping.get("type", "type"), "")
            order_type = type_mapping.get(type_value, OrderType.LIMIT)

            # Convert quantities and prices
            quantity = Decimal(
                str(exchange_data.get(field_mapping.get("quantity", "origQty"), "0"))
            )
            price_field = exchange_data.get(field_mapping.get("price", "price"))
            price = (
                Decimal(str(price_field)) if price_field and price_field != "0.00000000" else None
            )

            filled_quantity = Decimal(
                str(exchange_data.get(field_mapping.get("filled_quantity", "executedQty"), "0"))
            )

            # Convert status
            status_value = exchange_data.get(field_mapping.get("status", "status"), "")
            status = status_mapping.get(status_value, OrderStatus.REJECTED).value

            # Handle timestamp
            timestamp_field = exchange_data.get(field_mapping.get("timestamp"))
            if timestamp_field:
                if isinstance(timestamp_field, int):
                    timestamp = datetime.fromtimestamp(timestamp_field / 1000, tz=timezone.utc)
                else:
                    timestamp = datetime.fromisoformat(str(timestamp_field).replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)

            # Calculate average price if available
            average_price = None
            if exchange_data.get(field_mapping.get("quote_qty")) and filled_quantity > 0:
                quote_qty = Decimal(
                    str(
                        exchange_data.get(
                            field_mapping.get("quote_qty", "cummulativeQuoteQty"), "0"
                        )
                    )
                )
                average_price = quote_qty / filled_quantity

            return OrderResponse(
                id=order_id,
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                filled_quantity=filled_quantity,
                status=status,
                created_at=timestamp,
                average_price=average_price,
                exchange=exchange_name,
            )

        except Exception as e:
            raise ValidationError(f"Failed to convert {exchange_name} order response: {e}")

    @staticmethod
    def get_binance_field_mapping() -> dict[str, str]:
        """Get field mapping for Binance responses."""
        return {
            "id": "orderId",
            "client_order_id": "clientOrderId",
            "symbol": "symbol",
            "side": "side",
            "type": "type",
            "quantity": "origQty",
            "price": "price",
            "filled_quantity": "executedQty",
            "status": "status",
            "timestamp": "transactTime",
            "quote_qty": "cummulativeQuoteQty",
            "commission": "commission",
            "commission_asset": "commissionAsset",
        }

    @staticmethod
    def get_binance_status_mapping() -> dict[str, OrderStatus]:
        """Get status mapping for Binance."""
        return {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
        }

    @staticmethod
    def get_binance_type_mapping() -> dict[str, OrderType]:
        """Get type mapping for Binance."""
        return {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT,
        }

    @staticmethod
    def get_coinbase_field_mapping() -> dict[str, str]:
        """Get field mapping for Coinbase responses."""
        return {
            "id": "order_id",
            "client_order_id": "client_order_id",
            "symbol": "product_id",
            "side": "side",
            "type": "type",
            "quantity": "size",
            "price": "price",
            "filled_quantity": "filled_size",
            "status": "status",
            "timestamp": "created_time",
        }

    @staticmethod
    def get_coinbase_status_mapping() -> dict[str, OrderStatus]:
        """Get status mapping for Coinbase."""
        return {
            "OPEN": OrderStatus.PENDING,
            "PENDING": OrderStatus.PENDING,
            "ACTIVE": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "DONE": OrderStatus.FILLED,
            "SETTLED": OrderStatus.FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
        }

    @staticmethod
    def get_coinbase_type_mapping() -> dict[str, OrderType]:
        """Get type mapping for Coinbase."""
        return {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop": OrderType.STOP_LOSS,
            "stop_limit": OrderType.STOP_LOSS,
        }

    @staticmethod
    def get_okx_field_mapping() -> dict[str, str]:
        """Get field mapping for OKX responses."""
        return {
            "id": "ordId",
            "client_order_id": "clOrdId",
            "symbol": "instId",
            "side": "side",
            "type": "ordType",
            "quantity": "sz",
            "price": "px",
            "filled_quantity": "accFillSz",
            "status": "state",
            "timestamp": "cTime",
        }

    @staticmethod
    def get_okx_status_mapping() -> dict[str, OrderStatus]:
        """Get status mapping for OKX."""
        return {
            "live": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "expired": OrderStatus.EXPIRED,
            "failed": OrderStatus.REJECTED,
        }

    @staticmethod
    def get_okx_type_mapping() -> dict[str, OrderType]:
        """Get type mapping for OKX."""
        return {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "conditional": OrderType.STOP_LOSS,
            "post_only": OrderType.LIMIT,
            "fok": OrderType.LIMIT,
            "ioc": OrderType.LIMIT,
        }

    @staticmethod
    def convert_binance_order_to_response(result: dict[str, Any]) -> OrderResponse:
        """Convert Binance order result to unified OrderResponse."""
        # Handle both single order and list responses
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        return ExchangeConversionUtils.create_order_response(
            exchange_data=result,
            field_mapping=ExchangeConversionUtils.get_binance_field_mapping(),
            status_mapping=ExchangeConversionUtils.get_binance_status_mapping(),
            type_mapping=ExchangeConversionUtils.get_binance_type_mapping(),
            exchange_name="binance",
        )

    @staticmethod
    def convert_coinbase_order_to_response(result: dict[str, Any]) -> OrderResponse:
        """Convert Coinbase order result to unified OrderResponse."""
        return ExchangeConversionUtils.create_order_response(
            exchange_data=result,
            field_mapping=ExchangeConversionUtils.get_coinbase_field_mapping(),
            status_mapping=ExchangeConversionUtils.get_coinbase_status_mapping(),
            type_mapping=ExchangeConversionUtils.get_coinbase_type_mapping(),
            exchange_name="coinbase",
        )

    @staticmethod
    def convert_okx_order_to_response(result: dict[str, Any]) -> OrderResponse:
        """Convert OKX order result to unified OrderResponse."""
        return ExchangeConversionUtils.create_order_response(
            exchange_data=result,
            field_mapping=ExchangeConversionUtils.get_okx_field_mapping(),
            status_mapping=ExchangeConversionUtils.get_okx_status_mapping(),
            type_mapping=ExchangeConversionUtils.get_okx_type_mapping(),
            exchange_name="okx",
        )
