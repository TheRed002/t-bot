"""
Comprehensive tests for Decimal precision in financial calculations.

This test suite ensures that all financial calculations throughout the
T-Bot trading system maintain proper Decimal precision and avoid
floating-point errors that could lead to financial losses.
"""

import pytest
from decimal import Decimal, getcontext
import numpy as np

from src.utils.decimal_utils import (
    to_decimal, format_decimal, safe_divide, round_price, round_quantity,
    calculate_percentage, calculate_basis_points, validate_positive,
    validate_percentage, compare_decimals, sum_decimals, avg_decimals,
    clamp_decimal, ZERO, ONE, SATOSHI, ONE_BP
)
from src.risk_management.position_sizing import PositionSizer
from src.core.config import Config
from src.core.types import Signal, OrderSide, PositionSizeMethod


class TestDecimalUtils:
    """Test Decimal utility functions."""
    
    def test_to_decimal_conversion(self):
        """Test conversion of various types to Decimal."""
        # Test integer conversion
        assert to_decimal(100) == Decimal("100")
        
        # Test float conversion (should preserve precision)
        assert to_decimal(0.1 + 0.2) == Decimal(repr(0.1 + 0.2))
        
        # Test string conversion
        assert to_decimal("123.456") == Decimal("123.456")
        
        # Test existing Decimal
        d = Decimal("999.999")
        assert to_decimal(d) == d
        
        # Test None raises error
        with pytest.raises(Exception):
            to_decimal(None)
    
    def test_safe_divide(self):
        """Test safe division with Decimal."""
        # Normal division
        assert safe_divide(Decimal("10"), Decimal("2")) == Decimal("5")
        
        # Division by zero returns default
        assert safe_divide(Decimal("10"), ZERO) == ZERO
        assert safe_divide(Decimal("10"), ZERO, Decimal("999")) == Decimal("999")
        
        # Precision is maintained
        result = safe_divide(Decimal("1"), Decimal("3"))
        # Should have 28 significant digits
        assert len(str(result).replace(".", "").lstrip("0")) >= 20
    
    def test_round_price(self):
        """Test price rounding to tick size."""
        # Round to cents (0.01)
        assert round_price(Decimal("123.456"), Decimal("0.01")) == Decimal("123.46")
        
        # Round to satoshis (0.00000001)
        assert round_price(Decimal("0.123456789"), SATOSHI) == Decimal("0.12345679")
        
        # Round to 0.25 increments
        assert round_price(Decimal("10.30"), Decimal("0.25")) == Decimal("10.25")
        assert round_price(Decimal("10.40"), Decimal("0.25")) == Decimal("10.50")
    
    def test_round_quantity(self):
        """Test quantity rounding (always rounds down)."""
        # Round down to prevent exceeding balance
        assert round_quantity(Decimal("1.999"), Decimal("0.1")) == Decimal("1.9")
        assert round_quantity(Decimal("1.999"), Decimal("1")) == Decimal("1")
        
        # Round to lot size
        assert round_quantity(Decimal("123.456"), Decimal("0.01")) == Decimal("123.45")
    
    def test_percentage_calculations(self):
        """Test percentage and basis point calculations."""
        # 5% of 1000
        assert calculate_percentage(Decimal("1000"), Decimal("0.05")) == Decimal("50")
        
        # 50 basis points of 1000 (0.5%)
        assert calculate_basis_points(Decimal("1000"), Decimal("50")) == Decimal("5")
        
        # 1 basis point precision
        assert calculate_basis_points(Decimal("10000"), Decimal("1")) == Decimal("1")
    
    def test_decimal_comparison(self):
        """Test Decimal comparison with tolerance."""
        a = Decimal("1.00000001")
        b = Decimal("1.00000002")
        
        # Within satoshi tolerance
        assert compare_decimals(a, b, SATOSHI) == 0
        
        # Outside tolerance
        assert compare_decimals(a, b, Decimal("0.000000001")) == -1
        
        # Exact comparison
        assert compare_decimals(Decimal("5"), Decimal("3")) == 1
        assert compare_decimals(Decimal("3"), Decimal("5")) == -1


class TestPositionSizingPrecision:
    """Test position sizing maintains Decimal precision."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.risk.default_position_size_pct = 0.02  # 2%
        config.risk.max_position_size_pct = 0.05  # 5%
        config.risk.kelly_lookback_days = 30
        config.risk.volatility_window = 20
        config.risk.volatility_target = 0.15
        return config
    
    @pytest.fixture
    def position_sizer(self, config):
        """Create position sizer instance."""
        return PositionSizer(config)
    
    @pytest.mark.asyncio
    async def test_fixed_percentage_sizing_precision(self, position_sizer):
        """Test fixed percentage sizing maintains precision."""
        signal = Signal(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            confidence=0.8,
            timestamp=None
        )
        
        # Use a portfolio value that would cause precision issues with float
        portfolio_value = Decimal("10000.33333333")
        
        size = await position_sizer.calculate_position_size(
            signal, portfolio_value, PositionSizeMethod.FIXED_PCT
        )
        
        # Should be exactly 2% * 0.8 confidence = 1.6%
        expected = portfolio_value * Decimal("0.02") * Decimal("0.8")
        assert size == expected
        assert isinstance(size, Decimal)
    
    @pytest.mark.asyncio
    async def test_kelly_criterion_precision(self, position_sizer):
        """Test Kelly Criterion maintains Decimal precision."""
        signal = Signal(
            symbol="ETH/USDT",
            side=OrderSide.BUY,
            confidence=0.7,
            timestamp=None
        )
        
        # Add return history for Kelly calculation
        returns = [
            Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"),
            Decimal("-0.015"), Decimal("0.025"), Decimal("-0.005")
        ] * 10  # 60 returns
        position_sizer.return_history["ETH/USDT"] = returns
        
        portfolio_value = Decimal("50000.123456")
        
        size = await position_sizer.calculate_position_size(
            signal, portfolio_value, PositionSizeMethod.KELLY_CRITERION
        )
        
        # Result should be Decimal with proper precision
        assert isinstance(size, Decimal)
        # Should be within bounds (1% - 25%)
        assert size >= portfolio_value * Decimal("0.01")
        assert size <= portfolio_value * Decimal("0.25")
    
    @pytest.mark.asyncio
    async def test_price_history_decimal_storage(self, position_sizer):
        """Test that price history stores Decimals properly."""
        # Update price history
        await position_sizer.update_price_history("BTC/USDT", 50000.123456789)
        
        # Check stored value is Decimal
        stored_price = position_sizer.price_history["BTC/USDT"][-1]
        assert isinstance(stored_price, Decimal)
        assert stored_price == to_decimal(50000.123456789)
        
        # Add more prices and check return calculation
        await position_sizer.update_price_history("BTC/USDT", 50100.987654321)
        
        # Check return is calculated with Decimal precision
        stored_return = position_sizer.return_history["BTC/USDT"][-1]
        assert isinstance(stored_return, Decimal)


class TestExchangeOrderPrecision:
    """Test exchange order handling maintains precision."""
    
    def test_price_normalization_precision(self):
        """Test price normalization for different symbols."""
        from src.utils.helpers import normalize_price
        
        # BTC precision (8 decimals)
        btc_price = normalize_price(50123.123456789, "BTC/USDT")
        assert isinstance(btc_price, Decimal)
        assert str(btc_price) == "50123.12345679"
        
        # ETH precision (6 decimals)
        eth_price = normalize_price(3456.123456789, "ETH/USDT")
        assert str(eth_price) == "3456.123457"
        
        # USDT precision (2 decimals)
        usdt_price = normalize_price(1.005, "USDT/USD")
        assert str(usdt_price) == "1.01"
    
    def test_quantity_validation_precision(self):
        """Test quantity validation with Decimal."""
        from src.utils.validators import validate_quantity
        
        # Test with Decimal input
        qty = validate_quantity(Decimal("0.12345678"), "BTC/USDT")
        assert isinstance(qty, Decimal)
        assert str(qty) == "0.12345678"
        
        # Test with float input (should convert to Decimal)
        qty = validate_quantity(1.23456789, "ETH/USDT")
        assert isinstance(qty, Decimal)
        assert str(qty) == "1.234568"
        
        # Test minimum quantity check
        with pytest.raises(Exception):
            validate_quantity(Decimal("0.00001"), "BTC/USDT", min_qty=Decimal("0.0001"))


class TestFinancialCalculationAccuracy:
    """Test financial calculations for accuracy."""
    
    def test_compound_interest_precision(self):
        """Test compound interest calculation maintains precision."""
        principal = Decimal("10000")
        daily_rate = Decimal("0.001")  # 0.1% daily
        
        # Compound for 365 days
        result = principal
        for _ in range(365):
            result = result * (ONE + daily_rate)
        
        # Should maintain full precision
        assert isinstance(result, Decimal)
        # Approximate expected value: 10000 * (1.001^365) ≈ 14407.68
        assert abs(result - Decimal("14407.68")) < Decimal("1")
    
    def test_fee_calculation_precision(self):
        """Test trading fee calculations."""
        order_value = Decimal("10000.123456")
        fee_rate = Decimal("0.001")  # 0.1% fee
        
        fee = order_value * fee_rate
        net_value = order_value - fee
        
        assert fee == Decimal("10.000123456")
        assert net_value == Decimal("9990.123332544")
    
    def test_pnl_calculation_precision(self):
        """Test profit/loss calculation precision."""
        entry_price = Decimal("50000.12345678")
        exit_price = Decimal("51234.87654321")
        quantity = Decimal("0.12345678")
        
        # Calculate PnL
        price_diff = exit_price - entry_price
        pnl = price_diff * quantity
        
        # Calculate percentage return
        pct_return = safe_divide(price_diff, entry_price) * Decimal("100")
        
        assert isinstance(pnl, Decimal)
        assert isinstance(pct_return, Decimal)
        
        # Verify calculations maintain precision
        assert len(str(pnl).split(".")[-1]) > 6  # At least 6 decimal places
    
    def test_slippage_calculation(self):
        """Test slippage calculation with high precision."""
        expected_price = Decimal("1000.00")
        actual_price = Decimal("1000.05")
        quantity = Decimal("10.12345678")
        
        slippage_amount = (actual_price - expected_price) * quantity
        slippage_bps = safe_divide(
            actual_price - expected_price,
            expected_price
        ) * Decimal("10000")  # Convert to basis points
        
        assert slippage_amount == Decimal("0.5061728390")
        assert abs(slippage_bps - Decimal("5")) < Decimal("0.01")


class TestDecimalEdgeCases:
    """Test edge cases and potential precision issues."""
    
    def test_very_small_numbers(self):
        """Test handling of very small numbers."""
        # Satoshi-level precision
        small = Decimal("0.00000001")
        result = small * Decimal("1000000")
        assert result == Decimal("0.01")
        
        # Even smaller
        tiny = Decimal("0.000000000001")
        result = tiny * Decimal("1000000000000")
        assert result == ONE
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        large = Decimal("999999999999.999999999")
        result = large * Decimal("1.000000001")
        
        # Should maintain precision
        assert isinstance(result, Decimal)
        assert result > large
    
    def test_repeating_decimals(self):
        """Test handling of repeating decimals."""
        # 1/3 = 0.333...
        one_third = safe_divide(ONE, Decimal("3"))
        
        # Multiply back by 3
        result = one_third * Decimal("3")
        
        # Should be very close to 1 (within rounding error)
        assert abs(result - ONE) < Decimal("0.0000000000000000000000000001")
    
    def test_cumulative_rounding_errors(self):
        """Test that cumulative operations don't accumulate errors."""
        value = Decimal("100")
        
        # Perform many small operations
        for _ in range(10000):
            value = value * Decimal("1.0001")
            value = value / Decimal("1.0001")
        
        # Should return to exactly 100 (or very close)
        assert abs(value - Decimal("100")) < Decimal("0.000001")


@pytest.mark.integration
class TestIntegrationDecimalFlow:
    """Test complete flow maintains Decimal precision."""
    
    @pytest.mark.asyncio
    async def test_complete_order_flow_precision(self):
        """Test order flow from signal to execution maintains precision."""
        # This would test the complete flow in a real scenario
        # For now, we verify the key components use Decimal
        
        from src.core.types import OrderRequest, OrderType
        
        # Create order with Decimal values
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.12345678"),
            price=Decimal("50000.12345678"),
            exchange="binance"
        )
        
        assert isinstance(order.quantity, Decimal)
        assert isinstance(order.price, Decimal)
        
        # Calculate order value
        order_value = order.quantity * order.price
        assert isinstance(order_value, Decimal)
        assert order_value == Decimal("6173.147877618766684")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])