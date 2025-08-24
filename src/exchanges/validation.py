"""Order validation mixin for exchanges."""

from decimal import Decimal
from typing import Any

from src.core.exceptions import ValidationError

# Logger is provided by BaseExchange (via BaseComponent)


class OrderValidationMixin:
    """
    Mixin for order validation - shared across all exchanges.

    This eliminates duplication of validation logic across different
    exchange implementations.
    """

    def validate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        **kwargs,
    ) -> None:
        """
        Common order validation logic.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Order type (MARKET/LIMIT/STOP)
            quantity: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional parameters

        Raises:
            ValidationError: If order is invalid
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol is required and must be a string")

        # Validate side
        valid_sides = ["BUY", "SELL"]
        if side.upper() not in valid_sides:
            raise ValidationError(f"Invalid order side: {side}. Must be one of {valid_sides}")

        # Validate order type
        valid_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT", "STOP_MARKET"]
        if order_type.upper() not in valid_types:
            raise ValidationError(f"Invalid order type: {order_type}. Must be one of {valid_types}")

        # Validate quantity
        if not isinstance(quantity, Decimal | int | float):
            raise ValidationError(f"Quantity must be numeric, got {type(quantity)}")

        if quantity <= 0:
            raise ValidationError(f"Quantity must be positive, got {quantity}")

        # Validate price for limit orders
        if order_type.upper() in ["LIMIT", "STOP_LIMIT"]:
            if price is None:
                raise ValidationError(f"Price is required for {order_type} orders")

            if not isinstance(price, Decimal | int | float):
                raise ValidationError(f"Price must be numeric, got {type(price)}")

            if price <= 0:
                raise ValidationError(f"Price must be positive, got {price}")

        # Exchange-specific validation hook
        self._validate_exchange_specific(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs,
        )

    def _validate_exchange_specific(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        **kwargs,
    ) -> None:
        """
        Override in specific exchanges for custom validation.

        This is a hook for exchange-specific validation rules.
        """
        pass

    def validate_symbol_trading_rules(
        self, symbol: str, quantity: Decimal, price: Decimal | None = None
    ) -> None:
        """
        Validate against exchange trading rules for a symbol.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price

        Raises:
            ValidationError: If rules are violated
        """
        # This would typically check against exchange info
        # For now, we'll implement basic checks

        # Get symbol info (this would come from exchange info)
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info:
            self.logger.warning(f"No trading rules found for {symbol}")
            return

        # Check minimum quantity
        min_qty = symbol_info.get("min_quantity")
        if min_qty and quantity < min_qty:
            raise ValidationError(f"Quantity {quantity} below minimum {min_qty} for {symbol}")

        # Check maximum quantity
        max_qty = symbol_info.get("max_quantity")
        if max_qty and quantity > max_qty:
            raise ValidationError(f"Quantity {quantity} above maximum {max_qty} for {symbol}")

        # Check quantity step size
        step_size = symbol_info.get("step_size")
        if step_size:
            remainder = quantity % step_size
            if remainder != 0:
                raise ValidationError(
                    f"Quantity {quantity} not a multiple of step size {step_size}"
                )

        # Check price constraints
        if price is not None:
            # Check minimum price
            min_price = symbol_info.get("min_price")
            if min_price and price < min_price:
                raise ValidationError(f"Price {price} below minimum {min_price} for {symbol}")

            # Check maximum price
            max_price = symbol_info.get("max_price")
            if max_price and price > max_price:
                raise ValidationError(f"Price {price} above maximum {max_price} for {symbol}")

            # Check price tick size
            tick_size = symbol_info.get("tick_size")
            if tick_size:
                remainder = price % tick_size
                if remainder != 0:
                    raise ValidationError(f"Price {price} not a multiple of tick size {tick_size}")

    def _get_symbol_info(self, symbol: str) -> dict[str, Any] | None:
        """
        Get trading rules for a symbol.

        This should be overridden to fetch from actual exchange info.

        Args:
            symbol: Trading symbol

        Returns:
            Symbol trading rules or None
        """
        # This is a placeholder - should be implemented by exchange
        return None

    def validate_balance_for_order(
        self,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
        available_balance: Decimal | None = None,
    ) -> None:
        """
        Validate if account has sufficient balance for order.

        Args:
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price
            available_balance: Available balance

        Raises:
            ValidationError: If insufficient balance
        """
        if available_balance is None:
            self.logger.warning("Balance validation skipped - no balance provided")
            return

        # Calculate required balance
        if side.upper() == "BUY":
            if order_type.upper() == "MARKET":
                # For market orders, we need to estimate
                # This is simplified - real implementation would use current market price
                self.logger.warning("Cannot validate market buy order without price estimate")
                return
            elif price is not None:
                required = quantity * price
                if required > available_balance:
                    raise ValidationError(
                        f"Insufficient balance: need {required}, have {available_balance}"
                    )
        else:  # SELL
            # For sell orders, we need the asset quantity
            if quantity > available_balance:
                raise ValidationError(
                    f"Insufficient asset balance: need {quantity}, have {available_balance}"
                )
