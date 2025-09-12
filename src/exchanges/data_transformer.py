"""
Unified Data Transformation Patterns for Exchanges

This module provides consistent data transformation patterns across all exchanges,
ensuring uniform data flow from exchanges to core systems.

Patterns:
- Standardized symbol conversion
- Unified order type mapping
- Consistent price/volume formatting
- Error handling alignment
- Aligned messaging patterns (pub/sub for streams, req/reply for orders)
- Consistent batch vs stream processing modes
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import (
    OrderRequest,
    OrderResponse,
    OrderStatus,
    OrderType,
    Ticker,
)

# Import error handling decorators
from src.error_handling.decorators import with_retry

logger = get_logger(__name__)


class ExchangeDataTransformer(Protocol):
    """Protocol for exchange data transformers."""

    def transform_symbol_to_exchange(self, symbol: str) -> str:
        """Transform unified symbol to exchange-specific format."""
        ...

    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str:
        """Transform exchange-specific symbol to unified format."""
        ...

    def transform_order_to_exchange(self, order: OrderRequest) -> dict[str, Any]:
        """Transform unified order to exchange-specific format."""
        ...

    def transform_order_from_exchange(self, exchange_data: dict[str, Any]) -> OrderResponse:
        """Transform exchange-specific order data to unified format."""
        ...

    def transform_ticker_from_exchange(self, exchange_data: dict[str, Any]) -> Ticker:
        """Transform exchange-specific ticker to unified format."""
        ...


class BaseExchangeTransformer(ABC):
    """Base transformer with common patterns."""

    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = get_logger(f"{__name__}.{exchange_name}")

    def validate_decimal_precision(self, value: Decimal, symbol: str) -> Decimal:
        """Validate and format decimal with appropriate precision."""
        if not isinstance(value, Decimal):
            raise ValidationError(f"Expected Decimal, got {type(value)} for {symbol}")

        # Ensure 8 decimal places for crypto precision
        return value.quantize(Decimal("0.00000001"))

    def validate_symbol_format(self, symbol: str) -> str:
        """Validate symbol format."""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError(f"Invalid symbol format: {symbol}")

        symbol = symbol.upper().strip()
        if len(symbol) < 3:
            raise ValidationError(f"Symbol too short: {symbol}")

        return symbol

    def _apply_data_transformation(self, data: dict[str, Any], operation_type: str = "stream") -> dict[str, Any]:
        """Apply consistent data transformation matching utils module patterns exactly."""
        if data is None:
            return None

        transformed_data = data.copy()

        # Import financial utilities for consistent decimal handling
        try:
            from src.utils.decimal_utils import to_decimal

            # Apply consistent decimal conversion for financial data
            if isinstance(transformed_data, dict):
                if "price" in transformed_data:
                    transformed_data["price"] = to_decimal(transformed_data["price"])
                if "quantity" in transformed_data:
                    transformed_data["quantity"] = to_decimal(transformed_data["quantity"])
                if "volume" in transformed_data:
                    transformed_data["volume"] = to_decimal(transformed_data["volume"])
        except ImportError:
            pass  # Fallback if decimal_utils not available

        # Apply messaging patterns aligned with utils module
        if operation_type in ("stream", "websocket", "ticker", "orderbook", "trades"):
            transformed_data["message_pattern"] = "pub_sub"  # Streams use pub/sub
            transformed_data["processing_mode"] = "stream"
        elif operation_type in ("order", "cancel", "status"):
            transformed_data["message_pattern"] = "req_reply"  # Orders need responses
            transformed_data["processing_mode"] = "request_reply"  # Align with execution module pattern
        elif operation_type == "batch":
            transformed_data["message_pattern"] = "pub_sub"  # Align with utils pub_sub preference
            transformed_data["processing_mode"] = "batch"
        else:
            transformed_data["message_pattern"] = "pub_sub"  # Default to pub_sub
            transformed_data["processing_mode"] = "stream"

        # Ensure consistent data format versioning with utils module
        transformed_data["data_format"] = transformed_data.get("data_format", "event_data_v1")

        # Add timestamp if missing
        if "timestamp" not in transformed_data:
            from datetime import datetime, timezone
            transformed_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        return transformed_data

    def apply_messaging_pattern(self, data: dict[str, Any], operation_type: str) -> dict[str, Any]:
        """Apply consistent messaging patterns - DEPRECATED: Use _apply_data_transformation instead."""
        # Delegate to the new method for consistency
        return self._apply_data_transformation(data, operation_type)

    @abstractmethod
    def get_symbol_mappings(self) -> dict[str, str]:
        """Get exchange-specific symbol mappings."""
        pass

    @abstractmethod
    def get_order_type_mappings(self) -> dict[OrderType, str]:
        """Get exchange-specific order type mappings."""
        pass

    @abstractmethod
    def get_order_status_mappings(self) -> dict[str, OrderStatus]:
        """Get exchange-specific order status mappings."""
        pass

    @classmethod
    def transform_for_batch_processing(
        cls,
        batch_type: str,
        data_items: list[Any],
        batch_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Transform data for batch processing pattern to align with execution module.

        Args:
            batch_type: Type of batch operation
            data_items: List of items to process in batch
            batch_id: Unique batch identifier
            metadata: Additional batch metadata

        Returns:
            Dict formatted for batch processing
        """
        from datetime import datetime, timezone
        
        # Transform each item individually
        transformed_items = []
        for item in data_items:
            if isinstance(item, dict):
                # Apply standardization to each item
                standardized_item = validate_market_data_structure(item, "exchange_batch")
                transformed_items.append(standardized_item)
            else:
                transformed_items.append({
                    "payload": str(item),
                    "type": type(item).__name__,
                    "processing_mode": "batch",
                    "data_format": "event_data_v1",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        return {
            "batch_type": batch_type,
            "batch_id": batch_id or datetime.now(timezone.utc).isoformat(),
            "batch_size": len(data_items),
            "items": transformed_items,
            "processing_mode": "batch",
            "data_format": "batch_event_data_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "exchanges",
            "metadata": metadata or {},
            "message_pattern": "pub_sub",  # Align with execution module batch pattern
            "boundary_crossed": True,
        }


class BinanceTransformer(BaseExchangeTransformer):
    """Binance data transformer."""

    def __init__(self):
        super().__init__("binance")

    def get_symbol_mappings(self) -> dict[str, str]:
        """Binance symbol mappings."""
        return {
            "BTC-USD": "BTCUSDT",
            "ETH-USD": "ETHUSDT",
            "BNB-USD": "BNBUSDT",
            "ADA-USD": "ADAUSDT",
            "DOT-USD": "DOTUSDT",
        }

    def get_order_type_mappings(self) -> dict[OrderType, str]:
        """Binance order type mappings."""
        return {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT_LIMIT",
        }

    def get_order_status_mappings(self) -> dict[str, OrderStatus]:
        """Binance order status mappings."""
        return {
            "NEW": OrderStatus.PENDING,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.CANCELLED,
        }

    def transform_symbol_to_exchange(self, symbol: str) -> str:
        """Transform to Binance format (BTCUSDT)."""
        symbol = self.validate_symbol_format(symbol)

        # Reverse mapping
        reverse_map = {v: k for k, v in self.get_symbol_mappings().items()}
        if symbol in reverse_map:
            return reverse_map[symbol]

        # Convert from dash format to Binance format
        if "-" in symbol:
            return symbol.replace("-", "")

        return symbol

    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str:
        """Transform from Binance format to unified format."""
        symbol = self.validate_symbol_format(exchange_symbol)

        mappings = self.get_symbol_mappings()
        unified_symbol = None

        # Find mapping
        for unified, binance in mappings.items():
            if binance == symbol:
                unified_symbol = unified
                break

        if unified_symbol:
            return unified_symbol

        # Generic conversion
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-USDT"
        elif symbol.endswith("USDC"):
            base = symbol[:-4]
            return f"{base}-USDC"

        return symbol


class CoinbaseTransformer(BaseExchangeTransformer):
    """Coinbase data transformer."""

    def __init__(self):
        super().__init__("coinbase")

    def get_symbol_mappings(self) -> dict[str, str]:
        """Coinbase symbol mappings."""
        return {
            "BTC-USD": "BTC-USD",
            "ETH-USD": "ETH-USD",
            "LTC-USD": "LTC-USD",
            "ADA-USD": "ADA-USD",
            "BTC-USDT": "BTC-USDT",
            "ETH-USDT": "ETH-USDT",
        }

    def get_order_type_mappings(self) -> dict[OrderType, str]:
        """Coinbase order type mappings."""
        return {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "stop",
            OrderType.TAKE_PROFIT: "limit",  # Coinbase doesn't have specific take profit
        }

    def get_order_status_mappings(self) -> dict[str, OrderStatus]:
        """Coinbase order status mappings."""
        return {
            "pending": OrderStatus.PENDING,
            "open": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
        }

    def transform_symbol_to_exchange(self, symbol: str) -> str:
        """Transform to Coinbase format (BTC-USD)."""
        symbol = self.validate_symbol_format(symbol)

        mappings = self.get_symbol_mappings()
        if symbol in mappings:
            return mappings[symbol]

        # Ensure dash format
        if "-" not in symbol:
            if symbol.endswith("USD"):
                base = symbol[:-3]
                return f"{base}-USD"
            elif symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}-USDT"

        return symbol

    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str:
        """Transform from Coinbase format (already unified)."""
        return self.validate_symbol_format(exchange_symbol)


class OKXTransformer(BaseExchangeTransformer):
    """OKX data transformer."""

    def __init__(self):
        super().__init__("okx")

    def get_symbol_mappings(self) -> dict[str, str]:
        """OKX symbol mappings."""
        return {
            "BTC-USDT": "BTC-USDT",
            "ETH-USDT": "ETH-USDT",
            "BNB-USDT": "BNB-USDT",
            "ADA-USDT": "ADA-USDT",
            "DOT-USDT": "DOT-USDT",
            "SOL-USDT": "SOL-USDT",
        }

    def get_order_type_mappings(self) -> dict[OrderType, str]:
        """OKX order type mappings."""
        return {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP_LOSS: "conditional",
            OrderType.TAKE_PROFIT: "conditional",
        }

    def get_order_status_mappings(self) -> dict[str, OrderStatus]:
        """OKX order status mappings."""
        return {
            "live": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
        }

    def transform_symbol_to_exchange(self, symbol: str) -> str:
        """Transform to OKX format (BTC-USDT)."""
        symbol = self.validate_symbol_format(symbol)

        mappings = self.get_symbol_mappings()
        if symbol in mappings:
            return mappings[symbol]

        # Ensure dash format
        if "-" not in symbol:
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}-USDT"
            elif symbol.endswith("USDC"):
                base = symbol[:-4]
                return f"{base}-USDC"

        return symbol

    def transform_symbol_from_exchange(self, exchange_symbol: str) -> str:
        """Transform from OKX format (already unified)."""
        return self.validate_symbol_format(exchange_symbol)


class TransformerFactory:
    """Factory for creating exchange transformers."""

    _transformers = {
        "binance": BinanceTransformer,
        "coinbase": CoinbaseTransformer,
        "okx": OKXTransformer,
    }

    @classmethod
    @with_retry(max_attempts=2, base_delay=0.5)
    def create_transformer(cls, exchange_name: str) -> BaseExchangeTransformer:
        """Create transformer for exchange."""
        transformer_class = cls._transformers.get(exchange_name.lower())
        if not transformer_class:
            raise ValidationError(f"No transformer available for exchange: {exchange_name}")

        return transformer_class()

    @classmethod
    def get_supported_exchanges(cls) -> list[str]:
        """Get list of supported exchanges."""
        return list(cls._transformers.keys())


# Common transformation utilities
def standardize_decimal_precision(value: Any, symbol: str = "") -> Decimal:
    """Standardize decimal precision across all exchanges."""
    try:
        if isinstance(value, str):
            decimal_value = Decimal(value)
        elif isinstance(value, (int, float)):
            decimal_value = Decimal(str(value))
        elif isinstance(value, Decimal):
            decimal_value = value
        else:
            raise ValueError(f"Cannot convert {type(value)} to Decimal")

        # Use 8 decimal places for crypto precision
        return decimal_value.quantize(Decimal("0.00000001"))

    except Exception as e:
        raise ValidationError(f"Failed to standardize decimal for {symbol}: {e}") from e


def validate_market_data_structure(data: dict[str, Any], exchange: str) -> dict[str, Any]:
    """Validate and standardize market data structure with consistent transformation."""
    required_fields = ["symbol", "price", "timestamp"]

    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field '{field}' in {exchange} market data")

    # Apply consistent data transformation using transformer pattern
    try:
        # Create a base transformer instance for standardization
        transformer = BaseExchangeTransformer.__new__(BaseExchangeTransformer)
        transformer.exchange_name = exchange
        transformer.logger = logger

        # Apply consistent transformation
        validated_data = transformer._apply_data_transformation(data, "stream")

        # Ensure required fields are still present after transformation
        for field in required_fields:
            if field not in validated_data:
                validated_data[field] = data[field]

        return validated_data

    except Exception as e:
        logger.warning(f"Failed to apply transformer validation for {exchange}: {e}")

        # Fallback to direct standardization
        if "price" in data:
            data["price"] = standardize_decimal_precision(data["price"], data.get("symbol", ""))

        if "volume" in data:
            data["volume"] = standardize_decimal_precision(data["volume"], data.get("symbol", ""))

        # Add missing metadata for consistency
        data["data_format"] = "event_data_v1"
        data["message_pattern"] = "pub_sub"
        data["processing_mode"] = "stream"

        return data


def apply_cross_module_validation(
    data: dict[str, Any],
    source_module: str = "exchanges",
    target_module: str = "execution",
) -> dict[str, Any]:
    """
    Apply comprehensive cross-module validation for consistent data flow.

    Args:
        data: Data to validate and transform
        source_module: Source module name
        target_module: Target module name

    Returns:
        Dict with validated and aligned data for cross-module communication
    """
    validated_data = data.copy()

    # Add comprehensive boundary metadata to match execution module
    from datetime import datetime, timezone

    validated_data.update(
        {
            "cross_module_validation": True,
            "source_module": source_module,
            "target_module": target_module,
            "boundary_validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_flow_aligned": True,
        }
    )

    # Apply consistent messaging patterns
    try:
        from src.utils.messaging_patterns import ProcessingParadigmAligner

        # Apply processing paradigm alignment similar to execution module
        source_mode = validated_data.get("processing_mode", "stream")
        target_mode = "stream" if target_module == "execution" else source_mode

        validated_data = ProcessingParadigmAligner.align_processing_modes(
            source_mode=source_mode, target_mode=target_mode, data=validated_data
        )

    except ImportError:
        # Fallback to basic alignment if messaging_patterns not available
        logger.warning("ProcessingParadigmAligner not available, using basic alignment")
        if "processing_mode" not in validated_data:
            validated_data["processing_mode"] = "stream"
        if "message_pattern" not in validated_data:
            validated_data["message_pattern"] = "pub_sub"

    # Apply target-module specific boundary validation
    try:
        from src.utils.messaging_patterns import BoundaryValidator

        if target_module == "execution":
            # Validate at exchanges -> execution boundary
            boundary_data = {
                "component": validated_data.get("component", source_module),
                "operation": validated_data.get("operation", "exchange_operation"),
                "timestamp": validated_data.get(
                    "timestamp", datetime.now(timezone.utc).isoformat()
                ),
                "processing_mode": validated_data.get("processing_mode", "stream"),
                "data_format": validated_data.get("data_format", "event_data_v1"),
                "boundary_crossed": True,
            }
            BoundaryValidator.validate_risk_to_state_boundary(boundary_data)

    except Exception as e:
        # Log validation issues but don't fail the data flow
        logger.debug(f"Cross-module boundary validation failed: {e}")

    return validated_data


def ensure_boundary_fields(data: dict[str, Any], source: str = "exchanges") -> dict[str, Any]:
    """
    Ensure data has required boundary fields for cross-module communication.

    Args:
        data: Data dictionary to enhance
        source: Source module name

    Returns:
        Dict with required boundary fields
    """
    from datetime import datetime, timezone

    # Ensure processing mode is set
    if "processing_mode" not in data:
        data["processing_mode"] = "stream"

    # Ensure data format is set
    if "data_format" not in data:
        data["data_format"] = "event_data_v1"

    # Ensure timestamp is set
    if "timestamp" not in data:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Add source information
    if "source" not in data:
        data["source"] = source

    # Ensure metadata exists
    if "metadata" not in data:
        data["metadata"] = {}

    # Add boundary validation status
    data["boundary_validation"] = "applied"
    data["boundary_crossed"] = True

    return data


def transform_error_to_event_data(
    error: Exception,
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Transform error to consistent event data format matching execution module.

    Args:
        error: Exception to transform
        context: Error context information
        metadata: Additional metadata

    Returns:
        Dict with consistent event data format
    """
    from datetime import datetime, timezone

    return {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_context": context or {},
        "processing_mode": "stream",
        "data_format": "event_data_v1",
        "message_pattern": "pub_sub",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "exchanges",
        "boundary_crossed": True,
        "metadata": metadata or {},
    }
