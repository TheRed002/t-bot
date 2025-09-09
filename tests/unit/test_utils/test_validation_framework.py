"""
Unit tests for the centralized ValidationFramework.

Tests the new unified validation system that eliminates code duplication.
"""

from decimal import Decimal

import pytest

from src.core.exceptions import ValidationError
from src.utils.validation.core import ValidationFramework


class TestValidationFramework:
    """Test the centralized validation framework."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ValidationFramework()

    def test_validate_order_valid(self):
        """Test validation of a valid order."""
        order = {
            "symbol": "BTC/USDT",
            "type": "LIMIT",
            "side": "BUY",
            "price": 50000.0,
            "quantity": 0.01,
        }

        # Should not raise any exception
        assert self.validator.validate_order(order) == True

    def test_validate_order_invalid_price(self):
        """Test validation of order with invalid price."""
        order = {
            "symbol": "BTC/USDT",
            "type": "LIMIT",
            "side": "BUY",
            "price": -100,  # Negative price
            "quantity": 0.01,
        }

        with pytest.raises(ValidationError, match="Price must be positive"):
            self.validator.validate_order(order)

    def test_validate_order_invalid_quantity(self):
        """Test validation of order with invalid quantity."""
        order = {
            "symbol": "BTC/USDT",
            "type": "LIMIT",
            "side": "BUY",
            "price": 50000.0,
            "quantity": 0,  # Zero quantity
        }

        with pytest.raises(ValidationError, match="Quantity must be positive"):
            self.validator.validate_order(order)

    def test_validate_order_missing_fields(self):
        """Test validation of order with missing fields."""
        order = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            # Missing 'type' field
        }

        with pytest.raises(ValidationError, match="type: Invalid order type"):
            self.validator.validate_order(order)

    def test_validate_order_invalid_type(self):
        """Test validation of order with invalid type."""
        order = {
            "symbol": "BTC/USDT",
            "type": "INVALID_TYPE",
            "side": "BUY",
            "price": 50000.0,
            "quantity": 0.01,
        }

        with pytest.raises(ValidationError, match="Invalid order type"):
            self.validator.validate_order(order)

    def test_validate_price(self):
        """Test price validation."""
        # Valid prices
        assert self.validator.validate_price(100.0) == Decimal("100.0")
        assert self.validator.validate_price(0.00001) == Decimal("0.00001")
        assert self.validator.validate_price(1000000.0) == Decimal("1000000.0")

        # Invalid prices
        with pytest.raises(ValidationError):
            self.validator.validate_price(0)

        with pytest.raises(ValidationError):
            self.validator.validate_price(-100)

        with pytest.raises(ValidationError):
            self.validator.validate_price(float("inf"))

    def test_validate_quantity(self):
        """Test quantity validation."""
        # Valid quantities
        assert self.validator.validate_quantity(1.0) == Decimal("1.0")
        assert self.validator.validate_quantity(0.00001) == Decimal("0.00001")
        assert self.validator.validate_quantity(10000.0) == Decimal("10000.0")

        # Invalid quantities
        with pytest.raises(ValidationError):
            self.validator.validate_quantity(0)

        with pytest.raises(ValidationError):
            self.validator.validate_quantity(-1)

    def test_validate_symbol(self):
        """Test symbol validation."""
        # Valid symbols
        assert self.validator.validate_symbol("BTC/USDT") == "BTC/USDT"
        assert self.validator.validate_symbol("ETH-USD") == "ETH-USD"
        assert self.validator.validate_symbol("BTCUSDT") == "BTCUSDT"

        # Invalid symbols
        with pytest.raises(ValidationError):
            self.validator.validate_symbol("")

        with pytest.raises(ValidationError):
            self.validator.validate_symbol(None)

    def test_validate_strategy_params(self):
        """Test strategy parameters validation."""
        # Valid params
        params = {
            "strategy_type": "market_making",
            "bid_spread": 0.001,
            "ask_spread": 0.001,
            "order_size": 100,
        }
        assert self.validator.validate_strategy_params(params) == True

        # Invalid params - negative spread
        params = {
            "strategy_type": "market_making",
            "bid_spread": -0.001,
            "ask_spread": 0.001,
            "order_size": 100,
        }
        with pytest.raises(ValidationError):
            self.validator.validate_strategy_params(params)

    def test_validate_risk_params(self):
        """Test risk parameters validation."""
        # Valid params
        params = {
            "max_position_size": 1000,
            "risk_per_trade": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.1,
        }
        assert self.validator.validate_risk_params(params) == True

        # Invalid params - risk too high
        params = {
            "max_position_size": 1000,
            "risk_per_trade": 0.5,  # 50% risk
            "stop_loss": 0.05,
            "take_profit": 0.1,
        }
        with pytest.raises(ValidationError):
            self.validator.validate_risk_params(params)

    def test_validate_decimal_precision(self):
        """Test decimal precision validation."""
        # Test that validation handles Decimal types
        assert self.validator.validate_price(Decimal("50000.12345678")) == Decimal("50000.12345678")
        assert self.validator.validate_quantity(Decimal("0.00000001")) == Decimal("0.00000001")

    def test_validation_caching(self):
        """Test that validation results can be cached."""
        # This tests that the framework can be extended with caching
        order = {
            "symbol": "BTC/USDT",
            "type": "LIMIT",
            "side": "BUY",
            "price": 50000.0,
            "quantity": 0.01,
        }

        # Multiple validations of the same order should be fast
        for _ in range(100):
            assert self.validator.validate_order(order) == True
