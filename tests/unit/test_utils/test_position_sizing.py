"""
Tests for Position Sizing utilities.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from src.core.types.risk import PositionSizeMethod
from src.core.types.trading import Signal
from src.utils.decimal_utils import ZERO, ONE, to_decimal
from src.utils.position_sizing import (
    PositionSizingAlgorithm
)


class MockPositionSizingAlgorithm(PositionSizingAlgorithm):
    """Mock implementation for testing."""
    
    def calculate_size(self, signal, portfolio_value, risk_per_trade, **kwargs):
        return to_decimal("100")


class TestPositionSizingAlgorithm:
    """Test PositionSizingAlgorithm base class."""

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("0.02")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is True

    def test_validate_inputs_invalid_signal_no_symbol(self):
        """Test input validation with signal missing symbol."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        del signal.symbol  # Remove symbol attribute
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("0.02")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_invalid_signal_empty_symbol(self):
        """Test input validation with empty symbol."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = ""  # Empty symbol
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("0.02")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_none_signal(self):
        """Test input validation with None signal."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = None
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("0.02")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_zero_portfolio_value(self):
        """Test input validation with zero portfolio value."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        portfolio_value = ZERO
        risk_per_trade = to_decimal("0.02")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_negative_portfolio_value(self):
        """Test input validation with negative portfolio value."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        portfolio_value = to_decimal("-1000")
        risk_per_trade = to_decimal("0.02")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_zero_risk_per_trade(self):
        """Test input validation with zero risk per trade."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = ZERO
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_negative_risk_per_trade(self):
        """Test input validation with negative risk per trade."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("-0.01")
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_validate_inputs_excessive_risk_per_trade(self):
        """Test input validation with excessive risk per trade."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("0.30")  # > 25%
        
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        
        assert result is False

    def test_calculate_size_abstract_method(self):
        """Test that calculate_size is abstract."""
        # Should be able to instantiate mock implementation
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        
        result = algorithm.calculate_size(signal, to_decimal("10000"), to_decimal("0.02"))
        
        assert result == to_decimal("100")

    def test_abstract_base_class_cannot_instantiate(self):
        """Test that abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PositionSizingAlgorithm()

    def test_validate_inputs_boundary_risk_values(self):
        """Test input validation with boundary risk values."""
        algorithm = MockPositionSizingAlgorithm()
        
        signal = MagicMock()
        signal.symbol = "BTCUSD"
        portfolio_value = to_decimal("10000")
        
        # Test exactly at boundary
        risk_per_trade = to_decimal("0.25")  # Exactly 25%
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        assert result is True
        
        # Test just over boundary
        risk_per_trade = to_decimal("0.2500001")  # Just over 25%
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        assert result is False

    def test_validate_inputs_signal_edge_cases(self):
        """Test input validation with signal edge cases."""
        algorithm = MockPositionSizingAlgorithm()
        
        portfolio_value = to_decimal("10000")
        risk_per_trade = to_decimal("0.02")
        
        # Test signal without hasattr capability
        class SimpleSignal:
            pass
        
        signal = SimpleSignal()
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        assert result is False
        
        # Test signal with None symbol
        class SignalWithNoneSymbol:
            symbol = None
        
        signal = SignalWithNoneSymbol()
        result = algorithm.validate_inputs(signal, portfolio_value, risk_per_trade)
        assert result is False