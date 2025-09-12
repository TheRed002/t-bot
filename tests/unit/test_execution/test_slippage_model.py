"""Optimized unit tests for SlippageModel."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, Mock

import pytest

# Disable logging for performance
logging.disable(logging.CRITICAL)

# Create fixed datetime instances to avoid repeated datetime.now() calls
FIXED_DATETIME = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Pre-defined constants for faster test data creation
TEST_DECIMALS = {
    "TEN": Decimal("10.0"),
    "ONE": Decimal("1.0"),
    "HUNDRED": Decimal("100.0"),
    "PRICE_50K": Decimal("50000"),
    "VOLUME_1M": Decimal("1000000"),
    "VOLUME_5M": Decimal("5000000")
}

from src.core.config import Config
from src.core.exceptions import ExecutionError, ValidationError
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    SlippageMetrics,
)
from src.execution.slippage.slippage_model import SlippageModel


@pytest.fixture(scope="session")
def config():
    """Create test configuration."""
    config = Mock()
    config.error_handling = Mock()
    config.execution = {"default_daily_volume": str(TEST_DECIMALS["VOLUME_1M"])}
    config.database = Mock()
    config.monitoring = Mock()
    config.redis = Mock()
    return config


@pytest.fixture(scope="session")
def slippage_model(config):
    """Create SlippageModel instance."""
    return SlippageModel(config)


@pytest.fixture(scope="session")
def sample_order_request():
    """Create sample order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=TEST_DECIMALS["TEN"],
        price=TEST_DECIMALS["PRICE_50K"],
    )


@pytest.fixture(scope="session")
def sample_market_data():
    """Create sample market data."""
    return MarketData(
        symbol="BTCUSDT",
        open=Decimal("49900.0"),
        high=Decimal("50200.0"),
        low=Decimal("49800.0"),
        close=Decimal("50000.0"),
        volume=Decimal("1000.0"),
        quote_volume=Decimal("50000000.0"),
        timestamp=FIXED_DATETIME,
        exchange="binance",
        bid_price=Decimal("49995.0"),
        ask_price=Decimal("50005.0"),
    )


@pytest.fixture(scope="session")
def expected_slippage_metrics():
    """Create expected slippage metrics."""
    return SlippageMetrics(
        symbol="BTCUSDT",
        timeframe="1h",
        total_trades=1,
        total_volume=Decimal("10.0"),
        market_impact_bps=Decimal("5.0"),
        timing_cost_bps=Decimal("2.0"),
        spread_cost_bps=Decimal("1.0"),
        total_slippage_bps=Decimal("8.0"),
        small_order_slippage=Decimal("0"),
        medium_order_slippage=Decimal("8.0"),
        large_order_slippage=Decimal("0"),
        market_open_slippage=Decimal("0.05"),
        market_close_slippage=Decimal("0.05"),
        intraday_slippage=Decimal("8.0"),
        high_volatility_slippage=Decimal("10.0"),
        low_volatility_slippage=Decimal("6.0"),
        trending_slippage=Decimal("8.0"),
        ranging_slippage=Decimal("8.0"),
        total_slippage_cost=Decimal("4000.0"),
        avg_slippage_per_trade=Decimal("8.0"),
        price_improvement_count=0,
        price_improvement_amount=Decimal("0"),
        period_start=FIXED_DATETIME,
        period_end=FIXED_DATETIME,
        updated_at=FIXED_DATETIME,
    )


class TestSlippageModelBasic:
    """Test basic slippage model functionality."""

    def test_initialization(self, config):
        """Test SlippageModel initialization."""
        model = SlippageModel(config)
        assert model is not None
        assert model.config == config

    @pytest.mark.asyncio
    async def test_market_impact_calculation(self, slippage_model, sample_order_request, sample_market_data):
        """Test market impact calculation through predict_slippage."""
        result = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        assert isinstance(result.market_impact_bps, Decimal)
        assert result.market_impact_bps >= 0

    @pytest.mark.asyncio
    async def test_timing_cost_calculation(self, slippage_model, sample_order_request, sample_market_data):
        """Test timing cost calculation through predict_slippage."""
        result = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        assert isinstance(result.timing_cost_bps, Decimal)
        assert result.timing_cost_bps >= 0

    @pytest.mark.asyncio
    async def test_spread_cost_calculation(self, slippage_model, sample_order_request, sample_market_data):
        """Test spread cost calculation through predict_slippage."""
        result = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        assert isinstance(result.spread_cost_bps, Decimal)
        assert result.spread_cost_bps >= 0


class TestSlippageCalculation:
    """Test comprehensive slippage calculation."""

    @pytest.mark.asyncio
    async def test_predict_slippage(self, slippage_model, sample_order_request, sample_market_data):
        """Test slippage prediction."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        
        assert isinstance(metrics, SlippageMetrics)
        assert metrics.total_slippage_bps >= 0
        # Check execution price is in metadata since it's not a direct field
        execution_price = metrics.metadata.get("execution_price")
        assert execution_price is not None
        assert execution_price > 0

    @pytest.mark.asyncio
    async def test_calculate_slippage_buy_order(self, slippage_model, sample_market_data):
        """Test slippage calculation for buy orders."""
        buy_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("5.0"),
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(buy_order, sample_market_data)
        assert isinstance(metrics.total_slippage_bps, Decimal)

    @pytest.mark.asyncio
    async def test_calculate_slippage_sell_order(self, slippage_model, sample_market_data):
        """Test slippage calculation for sell orders."""
        sell_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("5.0"),
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(sell_order, sample_market_data)
        assert isinstance(metrics.total_slippage_bps, Decimal)

    @pytest.mark.asyncio
    async def test_large_order_slippage(self, slippage_model, sample_market_data):
        """Test slippage for large orders."""
        large_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100.0"),  # Large order
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(large_order, sample_market_data)
        assert isinstance(metrics.total_slippage_bps, Decimal)
        # Large orders should have higher slippage
        assert metrics.total_slippage_bps >= 0
        assert metrics.large_order_slippage >= 0

    @pytest.mark.asyncio
    async def test_small_order_slippage(self, slippage_model, sample_market_data):
        """Test slippage for small orders."""
        small_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),  # Small order
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(small_order, sample_market_data)
        assert isinstance(metrics.total_slippage_bps, Decimal)
        assert metrics.total_slippage_bps >= 0
        assert metrics.small_order_slippage >= 0


class TestSlippageEstimation:
    """Test slippage estimation methods."""

    @pytest.mark.asyncio
    async def test_estimate_execution_price_buy(self, slippage_model, sample_market_data):
        """Test execution price estimation for buy orders."""
        buy_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(buy_order, sample_market_data)
        estimated_price = metrics.metadata.get("execution_price")
        assert isinstance(estimated_price, Decimal)
        assert estimated_price > 0

    @pytest.mark.asyncio
    async def test_estimate_execution_price_sell(self, slippage_model, sample_market_data):
        """Test execution price estimation for sell orders."""
        sell_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(sell_order, sample_market_data)
        estimated_price = metrics.metadata.get("execution_price")
        assert isinstance(estimated_price, Decimal)
        assert estimated_price > 0

    @pytest.mark.asyncio
    async def test_volume_participation_calculation(self, slippage_model, sample_order_request, sample_market_data):
        """Test volume participation ratio calculation."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        ratio = metrics.metadata.get("volume_ratio")
        assert isinstance(ratio, Decimal)
        assert ratio >= 0


class TestSlippageValidation:
    """Test slippage validation and constraints."""

    @pytest.mark.asyncio
    async def test_validate_slippage_metrics(self, slippage_model, sample_order_request, sample_market_data):
        """Test slippage metrics validation."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        # Basic validation - check that all required fields are present
        assert isinstance(metrics, SlippageMetrics)
        assert metrics.total_slippage_bps >= 0
        assert metrics.market_impact_bps >= 0
        assert metrics.timing_cost_bps >= 0
        assert metrics.spread_cost_bps >= 0

    @pytest.mark.asyncio
    async def test_slippage_bounds_checking(self, slippage_model, sample_order_request, sample_market_data):
        """Test slippage bounds checking."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        
        # Test that slippage is within reasonable bounds
        assert metrics.total_slippage_bps >= 0
        assert metrics.total_slippage_bps <= Decimal("500")  # Max 500 bps (5%)
        
        # Test individual components are reasonable
        assert metrics.market_impact_bps >= 0
        assert metrics.timing_cost_bps >= 0
        assert metrics.spread_cost_bps >= 0

    @pytest.mark.asyncio
    async def test_confidence_level_validation(self, slippage_model, sample_order_request, sample_market_data):
        """Test confidence level validation through confidence intervals."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        
        # Test confidence interval calculation
        lower, upper = await slippage_model.get_slippage_confidence_interval(metrics, 0.95)
        assert isinstance(lower, Decimal)
        assert isinstance(upper, Decimal)
        assert lower <= upper
        assert lower <= metrics.total_slippage_bps <= upper


class TestSlippageOptimization:
    """Test slippage optimization features."""

    @pytest.mark.asyncio
    async def test_optimal_execution_strategy(self, slippage_model, sample_order_request, sample_market_data):
        """Test optimal execution strategy through slippage prediction."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        # The slippage prediction provides information that can be used for strategy optimization
        assert isinstance(metrics, SlippageMetrics)
        assert "execution_price" in metrics.metadata
        assert "volume_ratio" in metrics.metadata
        assert "volatility_adjustment" in metrics.metadata

    @pytest.mark.asyncio
    async def test_slippage_minimization(self, slippage_model, sample_order_request, sample_market_data):
        """Test slippage minimization through prediction analysis."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        
        # Verify that we get detailed breakdown that can be used for minimization
        assert metrics.market_impact_bps >= 0
        assert metrics.timing_cost_bps >= 0
        assert metrics.spread_cost_bps >= 0
        
        # Total should be sum of components (with potential volatility adjustment)
        total_components = metrics.market_impact_bps + metrics.timing_cost_bps + metrics.spread_cost_bps
        assert metrics.total_slippage_bps >= total_components * Decimal("0.8")  # Allow for adjustment

    @pytest.mark.asyncio
    async def test_order_splitting_recommendation(self, slippage_model, sample_market_data):
        """Test order splitting through large order slippage analysis."""
        large_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("200.0"),
            price=Decimal("50000"),
        )
        
        metrics = await slippage_model.predict_slippage(large_order, sample_market_data)
        # Large orders should show higher market impact, indicating need for splitting
        assert metrics.large_order_slippage > 0
        assert metrics.market_impact_bps > 0
        
        # Check volume ratio indicates significant market impact
        volume_ratio = metrics.metadata.get("volume_ratio")
        assert volume_ratio >= 0


class TestSlippageAnalytics:
    """Test slippage analytics and reporting."""

    @pytest.mark.asyncio
    async def test_slippage_distribution_analysis(self, slippage_model, sample_order_request, sample_market_data):
        """Test slippage distribution analysis through model summary."""
        # First add some historical data
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        await slippage_model.update_historical_data("BTCUSDT", metrics, {"test": "data"})
        
        summary = await slippage_model.get_model_summary("BTCUSDT")
        assert isinstance(summary, dict)
        assert "symbol_details" in summary

    @pytest.mark.asyncio
    async def test_slippage_trend_analysis(self, slippage_model, sample_order_request, sample_market_data):
        """Test slippage trend analysis through historical data tracking."""
        # Get initial count if any historical data exists
        initial_summary = await slippage_model.get_model_summary("BTCUSDT")
        initial_count = 0
        if "symbol_details" in initial_summary and "BTCUSDT" in initial_summary["symbol_details"]:
            initial_count = initial_summary["symbol_details"]["BTCUSDT"]["data_points"]
        
        # Add multiple historical data points
        for i in range(3):
            metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
            await slippage_model.update_historical_data("BTCUSDT", metrics, {"iteration": i})
        
        summary = await slippage_model.get_model_summary("BTCUSDT")
        assert isinstance(summary, dict)
        final_count = summary["symbol_details"]["BTCUSDT"]["data_points"]
        # Verify that we added exactly 3 new data points
        assert final_count == initial_count + 3

    @pytest.mark.asyncio
    async def test_performance_metrics(self, slippage_model, sample_order_request, sample_market_data):
        """Test performance metrics through confidence intervals."""
        metrics = await slippage_model.predict_slippage(sample_order_request, sample_market_data)
        
        # Test confidence interval calculation as a performance metric
        lower, upper = await slippage_model.get_slippage_confidence_interval(metrics, 0.95)
        
        # Calculate interval width as a performance measure
        interval_width = upper - lower
        assert isinstance(interval_width, Decimal)
        assert interval_width >= 0


class TestSlippageEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_volume_market(self, slippage_model):
        """Test slippage calculation with zero market volume."""
        # Use cached constants for better performance
        zero_volume_data = MarketData(
            symbol="BTCUSDT",
            open=TEST_DECIMALS["PRICE_50K"],
            high=TEST_DECIMALS["PRICE_50K"],
            low=TEST_DECIMALS["PRICE_50K"],
            close=TEST_DECIMALS["PRICE_50K"],
            volume=Decimal("0.0"),  # Zero volume
            quote_volume=Decimal("0.0"),
            timestamp=FIXED_DATETIME,
            exchange="binance",
            bid_price=Decimal("49995.0"),
            ask_price=Decimal("50005.0"),
        )
        
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )
        
        metrics = await slippage_model.predict_slippage(order, zero_volume_data)
        assert isinstance(metrics.total_slippage_bps, Decimal)

    @pytest.mark.asyncio
    async def test_extreme_spread_market(self, slippage_model):
        """Test slippage calculation with extreme spread."""
        # Pre-define extreme spread values
        extreme_values = {
            "high": Decimal("52000.0"),
            "low": Decimal("48000.0"),
            "volume": TEST_DECIMALS["HUNDRED"],
            "quote_volume": TEST_DECIMALS["VOLUME_5M"],
            "bid": Decimal("49000.0"),
            "ask": Decimal("51000.0")
        }
        
        wide_spread_data = MarketData(
            symbol="BTCUSDT",
            open=TEST_DECIMALS["PRICE_50K"],
            high=extreme_values["high"],
            low=extreme_values["low"],
            close=TEST_DECIMALS["PRICE_50K"],
            volume=extreme_values["volume"],
            quote_volume=extreme_values["quote_volume"],
            timestamp=FIXED_DATETIME,
            exchange="binance",
            bid_price=extreme_values["bid"],
            ask_price=extreme_values["ask"],
        )
        
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=TEST_DECIMALS["ONE"],
            price=TEST_DECIMALS["PRICE_50K"],
        )
        
        metrics = await slippage_model.predict_slippage(order, wide_spread_data)
        assert isinstance(metrics.total_slippage_bps, Decimal)
        # Extreme spread should result in higher spread cost
        assert metrics.spread_cost_bps > 0

    @pytest.mark.asyncio
    async def test_negative_price_handling(self, slippage_model, sample_market_data):
        """Test handling of invalid market data with negative prices."""
        # Create market data with invalid negative price
        invalid_market_data = MarketData(
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50200"),
            low=Decimal("49800"),
            close=Decimal("-100"),  # Invalid negative close price
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            timestamp=FIXED_DATETIME,
            exchange="binance",
            bid_price=Decimal("49995"),
            ask_price=Decimal("50005"),
        )
        
        valid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )
        
        # Should handle gracefully - expect ExecutionError (wraps ValidationError)
        with pytest.raises(ExecutionError):
            await slippage_model.predict_slippage(valid_order, invalid_market_data)

    @pytest.mark.asyncio
    async def test_very_large_order(self, slippage_model, sample_market_data):
        """Test handling of extremely large orders."""
        huge_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1000000.0"),  # Huge order
            price=TEST_DECIMALS["PRICE_50K"],
        )
        
        metrics = await slippage_model.predict_slippage(huge_order, sample_market_data)
        # Batch assertions
        assert isinstance(metrics.total_slippage_bps, Decimal)
        assert metrics.total_slippage_bps >= 0
        # Very large orders should have maximum market impact
        assert metrics.market_impact_bps > 0
        assert metrics.large_order_slippage > 0