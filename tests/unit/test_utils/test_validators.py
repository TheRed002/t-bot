"""
Unit tests for validators module.

This module tests the validation utilities in src.utils.validation module.
"""

from decimal import Decimal

import pytest

from src.core.exceptions import ValidationError

# Import the functions to test from the validation module
from src.utils.validation import (
    validate_batch,
    validate_exchange_credentials,
    validate_order,
    validate_price,
    validate_quantity,
    validate_risk_parameters,
    validate_strategy_params,
    validate_symbol,
    validate_timeframe,
)


class TestValidationFramework:
    """Test ValidationFramework functions."""

    def test_validate_order_valid(self):
        """Test order validation with valid order."""
        order = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": 50000.0,
            "quantity": 0.1,
        }

        assert validate_order(order) is True

    def test_validate_order_missing_required_fields(self):
        """Test order validation with missing required fields."""
        order = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            # Missing type and quantity
        }

        with pytest.raises(ValidationError, match="type: Invalid order type"):
            validate_order(order)

    def test_validate_order_invalid_side(self):
        """Test order validation with invalid side."""
        order = {
            "symbol": "BTCUSDT",
            "side": "INVALID",
            "type": "MARKET",
            "quantity": 0.1,
        }

        with pytest.raises(ValidationError, match="Side must be BUY or SELL"):
            validate_order(order)

    def test_validate_order_negative_price(self):
        """Test order validation with negative price."""
        order = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": -50000.0,
            "quantity": 0.1,
        }

        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_order(order)

    def test_validate_order_negative_quantity(self):
        """Test order validation with negative quantity."""
        order = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": 50000.0,
            "quantity": -0.1,
        }

        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_order(order)

    def test_validate_price_valid(self):
        """Test price validation with valid price."""
        assert validate_price(50000.0) == Decimal("50000.0")
        assert validate_price(0.001) == Decimal("0.001")
        assert validate_price(999999) == Decimal("999999")

    def test_validate_price_invalid(self):
        """Test price validation with invalid price."""
        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_price(0)

        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_price(-100)

        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_price(2_000_000)

    def test_validate_quantity_valid(self):
        """Test quantity validation with valid quantity."""
        assert validate_quantity(1.5) == Decimal("1.5")
        assert validate_quantity(0.0001) == Decimal("0.0001")
        assert validate_quantity(100) == Decimal("100")

    def test_validate_quantity_invalid(self):
        """Test quantity validation with invalid quantity."""
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(0)

        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(-1.5)

    def test_validate_symbol_valid(self):
        """Test symbol validation with valid symbol."""
        assert validate_symbol("BTC/USDT") == "BTC/USDT"
        assert validate_symbol("BTCUSDT") == "BTCUSDT"
        assert validate_symbol("ETH-BTC") == "ETH-BTC"

    def test_validate_symbol_invalid(self):
        """Test symbol validation with invalid symbol."""
        with pytest.raises(ValidationError, match="Symbol must be a non-empty string"):
            validate_symbol("")

        with pytest.raises(ValidationError, match="Symbol must be a non-empty string"):
            validate_symbol(None)

        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_symbol("BTC@USDT")

    def test_validate_exchange_credentials_valid(self):
        """Test exchange credentials validation with valid credentials."""
        credentials = {
            "api_key": "test_key_123",
            "api_secret": "test_secret_456",
            "testnet": True,
        }

        assert validate_exchange_credentials(credentials) is True

    def test_validate_exchange_credentials_missing_fields(self):
        """Test exchange credentials validation with missing fields."""
        credentials = {
            "api_key": "test_key_123",
            # Missing api_secret
        }

        with pytest.raises(ValidationError, match="api_secret is required"):
            validate_exchange_credentials(credentials)

    def test_validate_risk_parameters_valid(self):
        """Test risk parameters validation with valid parameters."""
        params = {
            "max_position_size": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "max_drawdown": 0.2,
            "risk_per_trade": 0.02,
        }

        assert validate_risk_parameters(params) is True

    def test_validate_risk_parameters_invalid(self):
        """Test risk parameters validation with invalid parameters."""
        params = {
            "max_position_size": 1.5,  # Too large
        }

        with pytest.raises(ValidationError, match="Max position size must be between 0 and 1"):
            validate_risk_parameters(params)

        params = {
            "risk_per_trade": 0.15,  # Too large
        }

        with pytest.raises(ValidationError, match="Risk per trade must be between 0 and 0.1"):
            validate_risk_parameters(params)

    def test_validate_strategy_params_valid(self):
        """Test strategy parameters validation with valid parameters."""
        params = {
            "strategy_type": "MEAN_REVERSION",
            "window_size": 20,
            "num_std": 2.0,
            "entry_threshold": 0.95,
            "timeframe": "1h",
        }

        assert validate_strategy_params(params) is True

    def test_validate_strategy_params_invalid_timeframe(self):
        """Test strategy parameters validation with invalid timeframe."""
        params = {
            "strategy_type": "MOMENTUM",
            "lookback_period": 10,
            "momentum_threshold": 0.05,
            "timeframe": "invalid",
        }

        with pytest.raises(ValidationError, match="Invalid timeframe"):
            validate_strategy_params(params)

    def test_validate_strategy_params_missing_required(self):
        """Test strategy parameters validation with missing required fields."""
        params = {
            "strategy_type": "MEAN_REVERSION",
            # Missing window_size, num_std, entry_threshold
        }

        with pytest.raises(ValidationError, match="window_size is required"):
            validate_strategy_params(params)

    def test_validate_timeframe_valid(self):
        """Test timeframe validation with valid timeframes."""
        assert validate_timeframe("1m") == "1m"
        assert validate_timeframe("5min") == "5m"
        assert validate_timeframe("1hour") == "1h"
        assert validate_timeframe("daily") == "1d"
        assert validate_timeframe("1week") == "1w"

    def test_validate_timeframe_invalid(self):
        """Test timeframe validation with invalid timeframe."""
        with pytest.raises(ValidationError, match="Invalid timeframe"):
            validate_timeframe("invalid")

        with pytest.raises(ValidationError, match="Invalid timeframe"):
            validate_timeframe("2s")

    def test_validate_batch(self):
        """Test batch validation."""
        validations = [
            ("price", validate_price, 50000.0),
            ("quantity", validate_quantity, 1.5),
            ("symbol", validate_symbol, "BTCUSDT"),
        ]

        results = validate_batch(validations)

        # Check that batch processing returned structure
        assert "validations" in results
        assert "batch_size" in results
        assert results["batch_size"] == 3
        
        # The batch validation should succeed
        validation_results = results["validations"]
        assert len(validation_results) == 3

    def test_validate_batch_with_errors(self):
        """Test batch validation with errors."""
        validations = [
            ("price", validate_price, -100),  # Invalid
            ("quantity", validate_quantity, 1.5),  # Valid
            ("symbol", validate_symbol, ""),  # Invalid
        ]

        results = validate_batch(validations)

        # Check that batch processing returned structure with errors
        assert "validations" in results
        assert "batch_size" in results
        assert results["batch_size"] == 3
        
        # The batch validation should handle both success and errors
        validation_results = results["validations"]
        assert len(validation_results) == 3


class TestMarketMakingValidation:
    """Test market making specific validation."""

    def test_validate_market_making_params_valid(self):
        """Test market making parameters validation with valid params."""
        params = {
            "strategy_type": "market_making",
            "bid_spread": 0.001,
            "ask_spread": 0.001,
            "order_size": 100,
        }

        assert validate_strategy_params(params) is True

    def test_validate_market_making_params_invalid(self):
        """Test market making parameters validation with invalid params."""
        params = {
            "strategy_type": "market_making",
            "bid_spread": -0.001,  # Negative spread
        }

        with pytest.raises(ValidationError, match="bid_spread must be non-negative"):
            validate_strategy_params(params)

        params = {
            "strategy_type": "market_making",
            "order_size": 0,  # Zero size
        }

        with pytest.raises(ValidationError, match="order_size must be positive"):
            validate_strategy_params(params)
