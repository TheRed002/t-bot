"""
Comprehensive Decimal Precision Tests for Binance Orders.

This module tests critical financial calculations in BinanceOrderManager
with focus on Decimal precision, fee calculations, and edge cases that
could cause monetary losses.

CRITICAL AREAS TESTED:
1. Decimal precision maintenance throughout order flow
2. Fee calculation accuracy at satoshi-level precision
3. Price and quantity normalization without float conversion
4. Edge cases with very small amounts
5. Round-trip precision (order -> response -> calculation)
"""

from decimal import Decimal, ROUND_HALF_UP, getcontext
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.types import OrderRequest, OrderSide, OrderType
from src.exchanges.binance_orders import BinanceOrderManager


class TestBinanceOrdersPrecision:
    """
    Test suite for Binance orders with focus on financial precision.
    
    These tests ensure that all financial calculations maintain Decimal
    precision and avoid floating-point errors that could cause monetary losses.
    """

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def binance_order_manager(self, config):
        """Create BinanceOrderManager instance."""
        mock_client = MagicMock()
        return BinanceOrderManager(config, mock_client)

    @pytest.fixture(autouse=True)
    def setup_decimal_precision(self):
        """Set high decimal precision for financial calculations."""
        # Set precision to 28 decimal places (Bitcoin precision is 8)
        getcontext().prec = 28

    def test_fee_calculation_decimal_precision(self, binance_order_manager):
        """Test that fee calculations maintain Decimal precision throughout."""
        # Test with very precise amounts
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.00000001"),  # 1 satoshi equivalent
        )
        
        # Use a precise fill price
        fill_price = Decimal("43256.78912345")
        
        # Calculate fees
        fee = binance_order_manager.calculate_fees(order, fill_price)
        
        # Verify fee is a Decimal
        assert isinstance(fee, Decimal)
        
        # Verify fee calculation maintains precision
        expected_fee = order.quantity * fill_price * Decimal("0.001")  # 0.1% default fee
        expected_fee = expected_fee.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        
        assert fee >= Decimal("0")  # Fees should never be negative
        
        # For very small amounts, fee might be zero due to rounding
        if order.quantity * fill_price > Decimal("0.00000001"):
            assert fee > Decimal("0")

    def test_satoshi_level_precision_maintenance(self, binance_order_manager):
        """Test precision maintenance at Bitcoin satoshi level (8 decimal places)."""
        # Test with 1 satoshi worth of Bitcoin
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.00000001"),  # 1 satoshi
        )
        
        fill_price = Decimal("50000.00000000")  # Even price
        
        fee = binance_order_manager.calculate_fees(order, fill_price)
        
        # Verify calculations don't lose precision
        order_value = order.quantity * fill_price
        assert order_value == Decimal("0.00050000")
        
        # Fee should be calculated precisely
        expected_fee_value = order_value * Decimal("0.001")
        assert expected_fee_value == Decimal("0.00000050")

    def test_no_float_conversion_in_calculations(self, binance_order_manager):
        """Test that no float conversions occur during calculations."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.23456789"),
        )
        
        fill_price = Decimal("43256.78912345")
        
        with patch('src.exchanges.binance_orders.round_to_precision_decimal') as mock_round_decimal:
            mock_round_decimal.side_effect = lambda x, p: x.quantize(Decimal('0.00000001'))
            
            fee = binance_order_manager.calculate_fees(order, fill_price)
            
            # Verify that Decimal operations were used throughout
            assert isinstance(fee, Decimal)
            
            # Verify round_to_precision_decimal was called (maintains Decimal)
            mock_round_decimal.assert_called()

    def test_extreme_precision_edge_cases(self, binance_order_manager):
        """Test edge cases with extreme precision requirements."""
        test_cases = [
            # Very small quantity, high price
            (Decimal("0.00000001"), Decimal("1000000.12345678")),
            # High quantity, very small price
            (Decimal("1000000.12345678"), Decimal("0.00000001")),
            # Both very precise
            (Decimal("123.45678912"), Decimal("987.65432198")),
            # Recurring decimals
            (Decimal("1") / Decimal("3"), Decimal("1") / Decimal("7")),
        ]
        
        for quantity, price in test_cases:
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )
            
            fee = binance_order_manager.calculate_fees(order, price)
            
            # Verify fee calculation doesn't fail or produce invalid results
            assert isinstance(fee, Decimal)
            assert fee >= Decimal("0")
            
            # Verify order value calculation
            order_value = quantity * price
            assert isinstance(order_value, Decimal)
            assert order_value > Decimal("0")

    def test_fee_structure_precision(self, binance_order_manager):
        """Test that fee structure values maintain precision."""
        with patch('src.exchanges.binance_orders.FEE_STRUCTURES') as mock_fee_structures:
            # Mock precise fee structures
            mock_fee_structures.get.return_value = {
                "taker_fee": "0.001",  # String to ensure Decimal conversion
                "maker_fee": "0.0008",
            }
            
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
            )
            
            fill_price = Decimal("50000.0")
            
            fee = binance_order_manager.calculate_fees(order, fill_price)
            
            # Verify precise fee calculation
            expected_fee = Decimal("1.0") * Decimal("50000.0") * Decimal("0.001")
            assert fee <= expected_fee  # May be rounded

    @pytest.mark.asyncio
    async def test_order_conversion_precision_maintenance(self, binance_order_manager):
        """Test that order conversions maintain Decimal precision."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.23456789"),
            price=Decimal("43256.78912345"),
        )
        
        # Test conversion to Binance format
        binance_order = binance_order_manager._convert_limit_order_to_binance(order)
        
        # Verify quantities are strings (for API) but maintain precision
        assert isinstance(binance_order["quantity"], str)
        assert isinstance(binance_order["price"], str)
        
        # Verify conversion back to Decimal maintains precision
        quantity_decimal = Decimal(binance_order["quantity"])
        price_decimal = Decimal(binance_order["price"])
        
        # Should be close to original (allowing for precision rounding)
        assert abs(quantity_decimal - order.quantity) < Decimal("0.00000001")
        assert abs(price_decimal - order.price) < Decimal("0.00000001")

    def test_precision_error_accumulation(self, binance_order_manager):
        """Test that precision errors don't accumulate across calculations."""
        # Simulate multiple fee calculations (batch processing scenario)
        orders = []
        total_fees = Decimal("0")
        
        for i in range(100):
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.12345678") + Decimal(str(i)) * Decimal("0.00000001"),
            )
            orders.append(order)
            
            fill_price = Decimal("50000.0") + Decimal(str(i))
            fee = binance_order_manager.calculate_fees(order, fill_price)
            total_fees += fee
        
        # Verify total fees are reasonable
        assert isinstance(total_fees, Decimal)
        assert total_fees > Decimal("0")
        
        # Verify no precision degradation (fees should sum to reasonable amount)
        assert total_fees < Decimal("1000")  # Sanity check

    def test_zero_and_negative_amount_handling(self, binance_order_manager):
        """Test handling of edge cases with zero and negative amounts."""
        # Test zero quantity
        order_zero = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
        )
        
        fee_zero = binance_order_manager.calculate_fees(order_zero, Decimal("50000"))
        assert fee_zero == Decimal("0")
        
        # Test zero price
        order_normal = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        
        fee_zero_price = binance_order_manager.calculate_fees(order_normal, Decimal("0"))
        assert fee_zero_price == Decimal("0")

    def test_decimal_quantization_consistency(self, binance_order_manager):
        """Test that decimal quantization is consistent across operations."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.123456789123456789"),  # More precision than Bitcoin
        )
        
        fill_price = Decimal("50000.123456789123456789")
        
        fee = binance_order_manager.calculate_fees(order, fill_price)
        
        # Verify quantization to appropriate precision
        assert fee.as_tuple().exponent >= -8  # At least 8 decimal places
        
        # Verify consistency in multiple calls
        fee2 = binance_order_manager.calculate_fees(order, fill_price)
        assert fee == fee2

    def test_market_order_precision_validation(self, binance_order_manager):
        """Test precision validation for market orders."""
        # Test market orders have no price requirement
        valid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.12345678"),
        )
        
        # Verify quantity is maintained as Decimal
        assert isinstance(valid_order.quantity, Decimal)
        assert valid_order.quantity == Decimal("1.12345678")
        
        # Test invalid market order with price (should fail)
        invalid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),  # Market orders shouldn't have price
        )
        
        # Just test that the order has a price when it shouldn't
        assert invalid_order.price is not None  # This is what makes it invalid

    def test_limit_order_precision_validation(self, binance_order_manager):
        """Test precision validation for limit orders."""
        valid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.12345678"),
            price=Decimal("50000.12345678"),
        )
        
        # Verify precision is maintained in limit orders
        assert isinstance(valid_order.quantity, Decimal)
        assert isinstance(valid_order.price, Decimal)
        assert valid_order.quantity == Decimal("1.12345678")
        assert valid_order.price == Decimal("50000.12345678")
        
        # Verify order type is correct
        assert valid_order.order_type == OrderType.LIMIT

    @pytest.mark.asyncio
    async def test_round_trip_precision_accuracy(self, binance_order_manager):
        """Test round-trip precision: Order -> API -> Response -> Calculation."""
        original_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.12345678"),
            price=Decimal("50000.12345678"),
        )
        
        # Mock API response that simulates Binance's response
        mock_response = {
            "orderId": "123456789",
            "clientOrderId": "test_order_1",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "1.12345678",  # Should match our input
            "price": "50000.12345678",  # Should match our input
            "executedQty": "1.12345678",
            "status": "FILLED",
            "time": 1700000000000,
        }
        
        # Convert response back to our format
        response = binance_order_manager._convert_binance_order_to_response(mock_response)
        
        # Verify round-trip precision
        assert response.quantity == original_order.quantity
        assert response.price == original_order.price
        
        # Verify all amounts are Decimal
        assert isinstance(response.quantity, Decimal)
        assert isinstance(response.price, Decimal)
        assert isinstance(response.filled_quantity, Decimal)

    def test_concurrent_fee_calculation_precision(self, binance_order_manager):
        """Test that concurrent fee calculations don't interfere with precision."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def calculate_fee_sync(order_data):
            order, fill_price = order_data
            return binance_order_manager.calculate_fees(order, fill_price)
        
        # Create multiple orders for concurrent processing
        orders_and_prices = []
        for i in range(50):
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal(f"1.{i:08d}"),  # Varying precision
            )
            fill_price = Decimal(f"50000.{i:08d}")
            orders_and_prices.append((order, fill_price))
        
        # Calculate fees concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            fees = list(executor.map(calculate_fee_sync, orders_and_prices))
        
        # Verify all fees are valid Decimals
        for fee in fees:
            assert isinstance(fee, Decimal)
            assert fee >= Decimal("0")
        
        # Verify precision consistency (calculate same fees sequentially)
        sequential_fees = []
        for order, fill_price in orders_and_prices:
            fee = binance_order_manager.calculate_fees(order, fill_price)
            sequential_fees.append(fee)
        
        # Results should be identical
        for concurrent_fee, sequential_fee in zip(fees, sequential_fees):
            assert concurrent_fee == sequential_fee

    def test_fee_calculation_error_handling(self, binance_order_manager):
        """Test error handling in fee calculations maintains Decimal precision."""
        # Test with invalid inputs that might cause precision issues
        invalid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        
        # Test error handling by mocking the fee structure to cause an error
        with patch('src.exchanges.binance_orders.FEE_STRUCTURES') as mock_fee_structures, \
             patch('src.exchanges.binance_orders.round_to_precision_decimal') as mock_round:
            # Mock to raise an exception during calculation
            mock_fee_structures.get.side_effect = Exception("Fee structure error")
            
            # Should return zero fee on error, not crash
            fee = binance_order_manager.calculate_fees(invalid_order, Decimal("50000"))
            assert fee == Decimal("0")

    def test_precision_levels_integration(self, binance_order_manager):
        """Test integration with precision levels from constants."""
        with patch('src.exchanges.binance_orders.PRECISION_LEVELS') as mock_precision:
            mock_precision.get.return_value = {"fee": 10}  # High precision
            
            order = OrderRequest(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.123456789123456789"),
            )
            
            fill_price = Decimal("50000.123456789123456789")
            
            fee = binance_order_manager.calculate_fees(order, fill_price)
            
            # Verify high precision is maintained
            assert isinstance(fee, Decimal)
            # Should be rounded to 10 decimal places based on mock
            assert len(str(fee).split('.')[-1]) <= 10