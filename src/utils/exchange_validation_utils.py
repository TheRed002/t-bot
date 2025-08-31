"""
Common Exchange Validation Utilities

This module contains shared validation utilities to eliminate duplication
across exchange implementations. It provides:
- Common order validation patterns
- Symbol format validation
- Price and quantity validation
- Exchange-specific validation rules
"""

from decimal import ROUND_HALF_UP, Decimal

from src.core.exceptions import ValidationError
from src.core.logging import get_logger
from src.core.types import OrderRequest, OrderSide, OrderType
from src.utils.interfaces import ValidationServiceInterface

logger = get_logger(__name__)


class ExchangeValidationUtils:
    """Common validation utilities for exchanges.
    
    This class provides utility functions for exchange-specific validation.
    Business validation logic should be handled by the service layer.
    """

    def __init__(self, validation_service: ValidationServiceInterface | None = None):
        """Initialize with injected validation service dependency.
        
        Args:
            validation_service: Injected validation service (required for dependency injection)
        """
        self.validation_service = validation_service

    # Common minimum order sizes by exchange
    MIN_ORDER_SIZES = {
        "binance": Decimal("0.00001"),
        "coinbase": Decimal("0.001"),
        "okx": Decimal("0.00001"),
        "default": Decimal("0.00001"),
    }

    # Maximum order sizes to prevent accidental large orders
    MAX_ORDER_SIZES = {
        "binance": Decimal("1000000"),
        "coinbase": Decimal("1000000"),
        "okx": Decimal("1000000"),
        "default": Decimal("1000000"),
    }

    def validate_order_request(self, order: OrderRequest, exchange: str = "") -> None:
        """
        DEPRECATED: Use ValidationService directly from service layer.
        
        This method contains business logic that belongs in the service layer.
        Exchange services should validate orders using ValidationService directly.

        Args:
            order: Order request to validate
            exchange: Exchange name for specific rules

        Raises:
            ValidationError: If validation fails
        """
        logger.warning("ExchangeValidationUtils.validate_order_request is deprecated. Use ValidationService in service layer.")
        
        if not self.validation_service:
            raise ValidationError(
                "ValidationService must be injected. Business validation belongs in service layer.",
                error_code="SERV_001"
            )
        
        # Only perform exchange-specific structural validation
        self._validate_exchange_specific(order, exchange)
        self._validate_quantity_and_price(order, exchange)

    # REMOVED: _validate_basic_fields and _validate_order_type_specific
    # These are now handled by ValidationFramework.validate_order to avoid duplication

    def _validate_exchange_specific(self, order: OrderRequest, exchange: str) -> None:
        """Validate exchange-specific requirements."""
        if exchange == "binance":
            ExchangeValidationUtils._validate_binance_order(order)
        elif exchange == "coinbase":
            ExchangeValidationUtils._validate_coinbase_order(order)
        elif exchange == "okx":
            ExchangeValidationUtils._validate_okx_order(order)

    @staticmethod
    def _validate_quantity_and_price(order: OrderRequest, exchange: str) -> None:
        """Validate quantity and price ranges."""
        # Check minimum order size
        min_size = ExchangeValidationUtils.MIN_ORDER_SIZES.get(
            exchange, ExchangeValidationUtils.MIN_ORDER_SIZES["default"]
        )
        if order.quantity < min_size:
            raise ValidationError(f"Order quantity {order.quantity} below minimum {min_size}")

        # Check maximum order size
        max_size = ExchangeValidationUtils.MAX_ORDER_SIZES.get(
            exchange, ExchangeValidationUtils.MAX_ORDER_SIZES["default"]
        )
        if order.quantity > max_size:
            raise ValidationError(f"Order quantity {order.quantity} above maximum {max_size}")

        # Validate price if present
        if order.price:
            if order.price <= 0:
                raise ValidationError("Order price must be positive")
            if order.price > Decimal("1000000"):  # Reasonable upper limit
                raise ValidationError("Order price too high")

        # Validate stop price if present
        if order.stop_price:
            if order.stop_price <= 0:
                raise ValidationError("Stop price must be positive")
            if order.stop_price > Decimal("1000000"):  # Reasonable upper limit
                raise ValidationError("Stop price too high")

    @staticmethod
    def _validate_binance_order(order: OrderRequest) -> None:
        """Validate Binance-specific order requirements."""
        # Binance symbol format validation
        if not SymbolValidationUtils.is_valid_binance_symbol(order.symbol):
            raise ValidationError(f"Invalid Binance symbol format: {order.symbol}")

        # Time in force validation for limit orders
        if order.order_type == OrderType.LIMIT and order.time_in_force:
            valid_tif = ["GTC", "IOC", "FOK"]
            if order.time_in_force not in valid_tif:
                raise ValidationError(f"Invalid time in force for Binance: {order.time_in_force}")

    @staticmethod
    def _validate_coinbase_order(order: OrderRequest) -> None:
        """Validate Coinbase-specific order requirements."""
        # Coinbase symbol format validation
        if not SymbolValidationUtils.is_valid_coinbase_symbol(order.symbol):
            raise ValidationError(f"Invalid Coinbase symbol format: {order.symbol}")

    @staticmethod
    def _validate_okx_order(order: OrderRequest) -> None:
        """Validate OKX-specific order requirements."""
        # OKX symbol format validation
        if not SymbolValidationUtils.is_valid_okx_symbol(order.symbol):
            raise ValidationError(f"Invalid OKX symbol format: {order.symbol}")


class SymbolValidationUtils:
    """Utilities for validating trading symbols."""

    # Valid symbols for each exchange (can be expanded)
    VALID_BINANCE_SYMBOLS = {
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "ADAUSDT",
        "DOTUSDT",
        "LINKUSDT",
        "LTCUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "MATICUSDT",
        "AVAXUSDT",
    }

    VALID_COINBASE_SYMBOLS = {
        "BTC-USD",
        "ETH-USD",
        "LTC-USD",
        "BTC-USDT",
        "ETH-USDT",
        "ADA-USD",
        "LINK-USD",
        "DOT-USD",
        "SOL-USD",
        "MATIC-USD",
        "AVAX-USD",
    }

    VALID_OKX_SYMBOLS = {
        "BTC-USDT",
        "ETH-USDT",
        "BNB-USDT",
        "ADA-USDT",
        "DOT-USDT",
        "LINK-USDT",
        "LTC-USDT",
        "SOL-USDT",
        "XRP-USDT",
        "MATIC-USDT",
        "AVAX-USDT",
    }

    @staticmethod
    def is_valid_symbol_format(symbol: str, exchange: str) -> bool:
        """
        Check if symbol format is valid for exchange.

        Args:
            symbol: Trading symbol
            exchange: Exchange name

        Returns:
            bool: True if format is valid
        """
        if exchange == "binance":
            return SymbolValidationUtils.is_valid_binance_symbol(symbol)
        elif exchange == "coinbase":
            return SymbolValidationUtils.is_valid_coinbase_symbol(symbol)
        elif exchange == "okx":
            return SymbolValidationUtils.is_valid_okx_symbol(symbol)
        else:
            return True  # Accept any format for unknown exchanges

    @staticmethod
    def is_valid_binance_symbol(symbol: str) -> bool:
        """Check if symbol is valid for Binance."""
        if not symbol:
            return False

        # Check if in known valid symbols
        if symbol in SymbolValidationUtils.VALID_BINANCE_SYMBOLS:
            return True

        # Check format (concatenated like BTCUSDT)
        if len(symbol) < 6:
            return False

        # Should not contain separators
        if "-" in symbol or "/" in symbol:
            return False

        # Should end with common quote currencies
        quote_currencies = ["USDT", "USDC", "USD", "BTC", "ETH", "BNB"]
        for quote in quote_currencies:
            if symbol.endswith(quote) and len(symbol) > len(quote):
                return True

        return False

    @staticmethod
    def is_valid_coinbase_symbol(symbol: str) -> bool:
        """Check if symbol is valid for Coinbase."""
        if not symbol:
            return False

        # Check if in known valid symbols
        if symbol in SymbolValidationUtils.VALID_COINBASE_SYMBOLS:
            return True

        # Check format (dash separated like BTC-USD)
        if "-" not in symbol:
            return False

        parts = symbol.split("-")
        if len(parts) != 2:
            return False

        base, quote = parts
        if len(base) < 2 or len(quote) < 3:
            return False

        # Check common quote currencies
        valid_quotes = ["USD", "USDT", "USDC", "EUR", "GBP", "BTC", "ETH"]
        return quote in valid_quotes

    @staticmethod
    def is_valid_okx_symbol(symbol: str) -> bool:
        """Check if symbol is valid for OKX."""
        if not symbol:
            return False

        # Check if in known valid symbols
        if symbol in SymbolValidationUtils.VALID_OKX_SYMBOLS:
            return True

        # Check format (dash separated like BTC-USDT)
        if "-" not in symbol:
            return False

        parts = symbol.split("-")
        if len(parts) != 2:
            return False

        base, quote = parts
        if len(base) < 2 or len(quote) < 3:
            return False

        # Check common quote currencies
        valid_quotes = ["USDT", "USDC", "USD", "BTC", "ETH", "OKB"]
        return quote in valid_quotes

    @staticmethod
    def get_supported_symbols(exchange: str) -> set[str]:
        """
        Get set of supported symbols for exchange.

        Args:
            exchange: Exchange name

        Returns:
            Set[str]: Set of supported symbols
        """
        if exchange == "binance":
            return SymbolValidationUtils.VALID_BINANCE_SYMBOLS.copy()
        elif exchange == "coinbase":
            return SymbolValidationUtils.VALID_COINBASE_SYMBOLS.copy()
        elif exchange == "okx":
            return SymbolValidationUtils.VALID_OKX_SYMBOLS.copy()
        else:
            return set()


class PrecisionValidationUtils:
    """Utilities for validating precision requirements."""

    # Default precision levels for exchanges
    DEFAULT_PRECISIONS = {
        "binance": {"price": 8, "quantity": 8},
        "coinbase": {"price": 8, "quantity": 8},
        "okx": {"price": 8, "quantity": 8},
        "default": {"price": 8, "quantity": 8},
    }

    @staticmethod
    def validate_precision(value: Decimal, precision: int, value_type: str = "value") -> None:
        """
        Validate that value meets precision requirements.

        Args:
            value: Value to validate
            precision: Required decimal places
            value_type: Type of value for error messages

        Raises:
            ValidationError: If precision is invalid
        """
        if not isinstance(value, Decimal):
            raise ValidationError(f"{value_type} must be Decimal type")

        # Check decimal places
        _, digits, exponent = value.as_tuple()
        if exponent < -precision:
            raise ValidationError(f"{value_type} precision exceeds maximum {precision} decimal places")

    @staticmethod
    def validate_order_precision(
        order: OrderRequest, exchange: str, custom_precision: dict[str, int] | None = None
    ) -> None:
        """
        Validate order precision requirements.

        Args:
            order: Order request to validate
            exchange: Exchange name
            custom_precision: Optional custom precision configuration

        Raises:
            ValidationError: If precision validation fails
        """
        precision_config = custom_precision or PrecisionValidationUtils.DEFAULT_PRECISIONS.get(
            exchange, PrecisionValidationUtils.DEFAULT_PRECISIONS["default"]
        )

        # Validate quantity precision
        PrecisionValidationUtils.validate_precision(order.quantity, precision_config["quantity"], "Order quantity")

        # Validate price precision
        if order.price:
            PrecisionValidationUtils.validate_precision(order.price, precision_config["price"], "Order price")

        # Validate stop price precision
        if order.stop_price:
            PrecisionValidationUtils.validate_precision(order.stop_price, precision_config["price"], "Stop price")

    @staticmethod
    def round_to_exchange_precision(value: Decimal, precision: int, rounding_mode=ROUND_HALF_UP) -> Decimal:
        """
        Round value to exchange precision requirements.

        Args:
            value: Value to round
            precision: Number of decimal places
            rounding_mode: Decimal rounding mode

        Returns:
            Decimal: Rounded value
        """
        if precision < 0:
            precision = 0

        quantize_target = Decimal("0." + "0" * precision) if precision > 0 else Decimal("1")
        return value.quantize(quantize_target, rounding=rounding_mode)


class RiskValidationUtils:
    """Utilities for risk-based validation."""

    @staticmethod
    def validate_order_size_limits(
        order: OrderRequest, portfolio_value: Decimal, max_position_percent: Decimal = Decimal("0.02")  # 2% default
    ) -> None:
        """
        Validate order size against portfolio limits.

        Args:
            order: Order request to validate
            portfolio_value: Total portfolio value
            max_position_percent: Maximum position size as percentage

        Raises:
            ValidationError: If order size exceeds limits
        """
        if portfolio_value <= 0:
            raise ValidationError("Portfolio value must be positive")

        # Calculate order value
        if order.order_type == OrderType.MARKET:
            # For market orders, we can't know exact price, so skip this check
            return

        if not order.price:
            return

        order_value = order.quantity * order.price
        max_position_value = portfolio_value * max_position_percent

        if order_value > max_position_value:
            raise ValidationError(
                f"Order value {order_value} exceeds maximum position size {max_position_value} "
                f"({max_position_percent * 100}% of portfolio)"
            )

    @staticmethod
    def validate_price_bounds(
        order: OrderRequest,
        current_market_price: Decimal,
        max_deviation_percent: Decimal = Decimal("0.10"),  # 10% default
    ) -> None:
        """
        Validate order price is within reasonable bounds of market price.

        Args:
            order: Order request to validate
            current_market_price: Current market price
            max_deviation_percent: Maximum allowed price deviation

        Raises:
            ValidationError: If price is outside bounds
        """
        if not order.price or current_market_price <= 0:
            return

        deviation = abs(order.price - current_market_price) / current_market_price

        if deviation > max_deviation_percent:
            raise ValidationError(
                f"Order price {order.price} deviates {deviation * 100:.2f}% from market price "
                f"{current_market_price}, exceeding maximum {max_deviation_percent * 100}%"
            )

    @staticmethod
    def validate_stop_price_logic(order: OrderRequest) -> None:
        """
        Validate stop price logic makes sense.

        Args:
            order: Order request to validate

        Raises:
            ValidationError: If stop price logic is invalid
        """
        if not order.stop_price or not order.price:
            return

        if order.order_type == OrderType.STOP_LOSS:
            if order.side == OrderSide.BUY:
                # Buy stop loss: stop price should be above limit price
                if order.stop_price < order.price:
                    raise ValidationError("Buy stop loss: stop price must be above limit price")
            else:
                # Sell stop loss: stop price should be below limit price
                if order.stop_price > order.price:
                    raise ValidationError("Sell stop loss: stop price must be below limit price")

        elif order.order_type == OrderType.TAKE_PROFIT:
            if order.side == OrderSide.BUY:
                # Buy take profit: stop price should be below limit price
                if order.stop_price > order.price:
                    raise ValidationError("Buy take profit: stop price must be below limit price")
            else:
                # Sell take profit: stop price should be above limit price
                if order.stop_price < order.price:
                    raise ValidationError("Sell take profit: stop price must be above limit price")


# Factory function for dependency injection - SERVICE LAYER USE ONLY
def get_exchange_validation_utils(validation_service: ValidationServiceInterface | None = None) -> ExchangeValidationUtils:
    """
    Factory function to create ExchangeValidationUtils with proper dependency injection.
    
    This should only be called from the service layer with proper dependency injection.
    Direct DI container access violates clean architecture.

    Args:
        validation_service: Validation service to inject (required from service layer)
        
    Returns:
        ExchangeValidationUtils: Instance with injected ValidationService
        
    Raises:
        ValidationError: If called without proper service injection
    """
    if validation_service is None:
        raise ValidationError(
            "ExchangeValidationUtils requires ValidationService injection from service layer. "
            "Do not call this factory directly - use through service layer.",
            error_code="SERV_001"
        )
    
    return ExchangeValidationUtils(validation_service=validation_service)
