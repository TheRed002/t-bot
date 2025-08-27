"""Unit tests for SlippageModel."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.core.config import Config
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    SlippageMetrics,
)
from src.execution.slippage.slippage_model import SlippageModel


@pytest.fixture
def config():
    """Create test configuration."""
    config = MagicMock(spec=Config)
    config.error_handling = MagicMock()
    config.execution = {"default_daily_volume": "1000000"}
    return config


@pytest.fixture
def slippage_model(config):
    """Create SlippageModel instance."""
    return SlippageModel(config)


@pytest.fixture
def sample_order_request():
    """Create sample order request."""
    return OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("10.0"),
        price=Decimal("50000")
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return MarketData(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        open=Decimal("49500"),
        high=Decimal("51000"),
        low=Decimal("49000"),
        close=Decimal("50000"),
        volume=Decimal("100000"),
        exchange="binance"
    )


class TestSlippageModel:
    """Test cases for SlippageModel."""

    def test_initialization(self, slippage_model):
        """Test SlippageModel initialization."""
        assert slippage_model is not None
        assert slippage_model.market_impact_coefficient == 0.5
        assert slippage_model.volatility_adjustment_factor == 1.2
        assert slippage_model.spread_cost_factor == 0.5
        assert len(slippage_model.volatility_regimes) == 4

    @pytest.mark.asyncio
    async def test_predict_slippage_basic(self, slippage_model, sample_order_request, sample_market_data):
        """Test basic slippage prediction."""
        result = await slippage_model.predict_slippage(
            sample_order_request,
            sample_market_data,
            participation_rate=0.2,
            time_horizon_minutes=60
        )
        
        assert isinstance(result, SlippageMetrics)
        assert result.symbol == "BTCUSDT"
        assert result.order_size == Decimal("10.0")
        assert result.market_price == Decimal("50000")
        assert result.price_slippage_bps >= 0
        assert result.market_impact_bps >= 0
        assert result.timing_cost_bps >= 0
        assert result.total_cost_bps >= 0

    @pytest.mark.asyncio
    async def test_predict_slippage_invalid_order(self, slippage_model, sample_market_data):
        """Test slippage prediction with invalid order."""
        with pytest.raises(Exception):
            await slippage_model.predict_slippage(None, sample_market_data)

    @pytest.mark.asyncio
    async def test_predict_slippage_invalid_market_data(self, slippage_model, sample_order_request):
        """Test slippage prediction with invalid market data."""
        with pytest.raises(Exception):
            await slippage_model.predict_slippage(sample_order_request, None)

    @pytest.mark.asyncio
    async def test_predict_slippage_zero_quantity(self, slippage_model, sample_market_data):
        """Test slippage prediction with zero quantity."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0"),
            price=Decimal("50000")
        )
        
        with pytest.raises(Exception):
            await slippage_model.predict_slippage(order, sample_market_data)

    @pytest.mark.asyncio
    async def test_predict_slippage_zero_price(self, slippage_model, sample_order_request):
        """Test slippage prediction with zero price."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("0"),
            volume=Decimal("100000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        with pytest.raises(Exception):
            await slippage_model.predict_slippage(sample_order_request, market_data)

    @pytest.mark.asyncio
    async def test_calculate_market_impact_slippage(self, slippage_model, sample_order_request, sample_market_data):
        """Test market impact slippage calculation."""
        impact = await slippage_model._calculate_market_impact_slippage(
            sample_order_request,
            sample_market_data,
            participation_rate=0.2,
            time_horizon_minutes=60
        )
        
        assert isinstance(impact, Decimal)
        assert impact >= 0
        assert impact <= Decimal("500")  # Maximum cap

    @pytest.mark.asyncio
    async def test_calculate_market_impact_high_participation(self, slippage_model, sample_order_request, sample_market_data):
        """Test market impact with high participation rate."""
        impact_low = await slippage_model._calculate_market_impact_slippage(
            sample_order_request,
            sample_market_data,
            participation_rate=0.1,
            time_horizon_minutes=60
        )
        
        impact_high = await slippage_model._calculate_market_impact_slippage(
            sample_order_request,
            sample_market_data,
            participation_rate=0.5,
            time_horizon_minutes=60
        )
        
        # Higher participation rate should result in higher impact
        assert impact_high > impact_low

    @pytest.mark.asyncio
    async def test_calculate_market_impact_time_horizon_effect(self, slippage_model, sample_order_request, sample_market_data):
        """Test market impact with different time horizons."""
        impact_short = await slippage_model._calculate_market_impact_slippage(
            sample_order_request,
            sample_market_data,
            participation_rate=0.2,
            time_horizon_minutes=15
        )
        
        impact_long = await slippage_model._calculate_market_impact_slippage(
            sample_order_request,
            sample_market_data,
            participation_rate=0.2,
            time_horizon_minutes=240
        )
        
        # Longer time horizon should result in lower impact
        assert impact_short > impact_long

    @pytest.mark.asyncio
    async def test_calculate_timing_cost_slippage(self, slippage_model, sample_order_request, sample_market_data):
        """Test timing cost slippage calculation."""
        timing_cost = await slippage_model._calculate_timing_cost_slippage(
            sample_order_request,
            sample_market_data,
            time_horizon_minutes=60
        )
        
        assert isinstance(timing_cost, Decimal)
        assert timing_cost >= 0
        assert timing_cost <= Decimal("200")  # Maximum cap

    @pytest.mark.asyncio
    async def test_calculate_timing_cost_with_range_data(self, slippage_model, sample_order_request, sample_market_data):
        """Test timing cost calculation with high/low range data."""
        timing_cost = await slippage_model._calculate_timing_cost_slippage(
            sample_order_request,
            sample_market_data,
            time_horizon_minutes=60
        )
        
        # Should calculate volatility from high-low range
        expected_volatility = float((sample_market_data.high_price - sample_market_data.low_price) / sample_market_data.price)
        assert timing_cost > 0

    @pytest.mark.asyncio
    async def test_calculate_timing_cost_no_range_data(self, slippage_model, sample_order_request):
        """Test timing cost calculation without high/low range data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        timing_cost = await slippage_model._calculate_timing_cost_slippage(
            sample_order_request,
            market_data,
            time_horizon_minutes=60
        )
        
        # Should use default volatility
        assert timing_cost > 0

    @pytest.mark.asyncio
    async def test_calculate_spread_cost(self, slippage_model, sample_order_request, sample_market_data):
        """Test spread cost calculation."""
        spread_cost = await slippage_model._calculate_spread_cost(
            sample_order_request,
            sample_market_data
        )
        
        assert isinstance(spread_cost, Decimal)
        assert spread_cost > 0
        
        # With bid=49995, ask=50005, price=50000
        # spread = 10, spread_bps = (10/50000) * 10000 = 2
        # spread_cost = 2 * 0.5 * 1.2 (for buy) = 1.2
        expected_spread_bps = (sample_market_data.ask - sample_market_data.bid) / sample_market_data.price * Decimal("10000")
        expected_cost = expected_spread_bps * Decimal("0.5") * Decimal("1.2")  # Buy adjustment
        assert abs(spread_cost - expected_cost) < Decimal("0.1")

    @pytest.mark.asyncio
    async def test_calculate_spread_cost_sell_order(self, slippage_model, sample_market_data):
        """Test spread cost calculation for sell order."""
        sell_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            price=Decimal("50000")
        )
        
        spread_cost = await slippage_model._calculate_spread_cost(sell_order, sample_market_data)
        
        assert isinstance(spread_cost, Decimal)
        assert spread_cost > 0

    @pytest.mark.asyncio
    async def test_calculate_spread_cost_no_bid_ask(self, slippage_model, sample_order_request):
        """Test spread cost calculation without bid/ask data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        spread_cost = await slippage_model._calculate_spread_cost(sample_order_request, market_data)
        
        # Should return default spread
        assert spread_cost == Decimal("20")

    @pytest.mark.asyncio
    async def test_calculate_volatility_adjustment(self, slippage_model, sample_market_data):
        """Test volatility adjustment calculation."""
        adjustment = await slippage_model._calculate_volatility_adjustment("BTCUSDT", sample_market_data)
        
        assert isinstance(adjustment, Decimal)
        assert adjustment > 0
        
        # With high=51000, low=49000, price=50000
        # volatility = (51000-49000)/50000 = 0.04 = 4%
        # This should fall in "high" volatility regime (3-5%)
        expected_volatility = 0.04
        assert abs(float(adjustment) - 1.5) < 0.1  # High volatility multiplier

    @pytest.mark.asyncio
    async def test_calculate_volatility_adjustment_low_vol(self, slippage_model):
        """Test volatility adjustment for low volatility."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(timezone.utc),
            high_price=Decimal("50200"),  # Low volatility
            low_price=Decimal("49800")
        )
        
        adjustment = await slippage_model._calculate_volatility_adjustment("BTCUSDT", market_data)
        
        # Volatility = (50200-49800)/50000 = 0.008 = 0.8% (low)
        assert adjustment == Decimal("0.8")  # Low volatility multiplier

    @pytest.mark.asyncio
    async def test_calculate_volatility_adjustment_no_range(self, slippage_model):
        """Test volatility adjustment without range data."""
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume=Decimal("100000"),
            timestamp=datetime.now(timezone.utc)
        )
        
        adjustment = await slippage_model._calculate_volatility_adjustment("BTCUSDT", market_data)
        
        # Should use default 2% volatility (normal regime)
        assert adjustment == 1.0

    @pytest.mark.asyncio
    async def test_calculate_expected_execution_price_buy(self, slippage_model, sample_market_data):
        """Test expected execution price calculation for buy order."""
        buy_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            price=Decimal("50000")
        )
        
        slippage_bps = Decimal("50")  # 0.5% slippage
        
        execution_price = await slippage_model._calculate_expected_execution_price(
            buy_order,
            sample_market_data,
            slippage_bps
        )
        
        # For buy orders, slippage increases the price
        expected_price = sample_market_data.price * (Decimal("1") + slippage_bps / Decimal("10000"))
        assert execution_price == expected_price

    @pytest.mark.asyncio
    async def test_calculate_expected_execution_price_sell(self, slippage_model, sample_market_data):
        """Test expected execution price calculation for sell order."""
        sell_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),
            price=Decimal("50000")
        )
        
        slippage_bps = Decimal("50")  # 0.5% slippage
        
        execution_price = await slippage_model._calculate_expected_execution_price(
            sell_order,
            sample_market_data,
            slippage_bps
        )
        
        # For sell orders, slippage decreases the price
        expected_price = sample_market_data.price * (Decimal("1") - slippage_bps / Decimal("10000"))
        assert execution_price == expected_price

    @pytest.mark.asyncio
    async def test_update_historical_data(self, slippage_model, sample_market_data):
        """Test updating historical data."""
        symbol = "BTCUSDT"
        
        actual_slippage = SlippageMetrics(
            symbol=symbol,
            order_size=Decimal("10.0"),
            market_price=Decimal("50000"),
            execution_price=Decimal("50025"),
            price_slippage_bps=Decimal("5"),
            market_impact_bps=Decimal("3"),
            timing_cost_bps=Decimal("2"),
            total_cost_bps=Decimal("10"),
            spread_bps=Decimal("2"),
            volume_ratio=0.1,
            volatility=0.02,
            timestamp=datetime.now(timezone.utc)
        )
        
        market_conditions = {"spread": 0.0002, "volume": 1000000}
        
        await slippage_model.update_historical_data(symbol, actual_slippage, market_conditions)
        
        # Check that data was stored
        assert symbol in slippage_model.historical_slippage
        assert len(slippage_model.historical_slippage[symbol]) == 1
        assert symbol in slippage_model.market_conditions_history
        assert len(slippage_model.market_conditions_history[symbol]) == 1

    @pytest.mark.asyncio
    async def test_update_historical_data_multiple_points(self, slippage_model):
        """Test updating historical data with multiple points."""
        symbol = "BTCUSDT"
        
        # Add multiple data points
        for i in range(15):
            actual_slippage = SlippageMetrics(
                symbol=symbol,
                order_size=Decimal("10.0"),
                market_price=Decimal("50000"),
                execution_price=Decimal("50025"),
                price_slippage_bps=Decimal("5"),
                market_impact_bps=Decimal("3"),
                timing_cost_bps=Decimal("2"),
                total_cost_bps=Decimal("10"),
                spread_bps=Decimal("2"),
                volume_ratio=0.1,
                volatility=0.02,
                timestamp=datetime.now(timezone.utc)
            )
            
            await slippage_model.update_historical_data(symbol, actual_slippage, {})
        
        # Should have enough data to trigger model update
        assert len(slippage_model.historical_slippage[symbol]) == 15
        assert symbol in slippage_model.model_parameters

    @pytest.mark.asyncio
    async def test_get_slippage_confidence_interval(self, slippage_model, sample_order_request, sample_market_data):
        """Test confidence interval calculation."""
        # First predict slippage
        predicted_slippage = await slippage_model.predict_slippage(
            sample_order_request,
            sample_market_data
        )
        
        # Get confidence interval
        lower, upper = await slippage_model.get_slippage_confidence_interval(
            predicted_slippage,
            confidence_level=0.95
        )
        
        assert isinstance(lower, Decimal)
        assert isinstance(upper, Decimal)
        assert lower >= 0
        assert upper >= lower
        assert upper > predicted_slippage.total_cost_bps * Decimal("0.5")  # Should be wider than base

    @pytest.mark.asyncio
    async def test_get_slippage_confidence_interval_with_history(self, slippage_model, sample_order_request, sample_market_data):
        """Test confidence interval with historical data."""
        symbol = "BTCUSDT"
        
        # Add some historical data
        for i in range(10):
            actual_slippage = SlippageMetrics(
                symbol=symbol,
                order_size=Decimal("10.0"),
                market_price=Decimal("50000"),
                execution_price=Decimal("50025"),
                price_slippage_bps=Decimal("5"),
                market_impact_bps=Decimal("3"),
                timing_cost_bps=Decimal("2"),
                total_cost_bps=Decimal("10"),
                spread_bps=Decimal("2"),
                volume_ratio=0.1,
                volatility=0.02,
                timestamp=datetime.now(timezone.utc)
            )
            
            await slippage_model.update_historical_data(symbol, actual_slippage, {})
        
        # Predict slippage
        predicted_slippage = await slippage_model.predict_slippage(
            sample_order_request,
            sample_market_data
        )
        
        # Get confidence interval
        lower, upper = await slippage_model.get_slippage_confidence_interval(
            predicted_slippage,
            confidence_level=0.95
        )
        
        assert lower >= 0
        assert upper > lower

    @pytest.mark.asyncio
    async def test_get_model_summary_empty(self, slippage_model):
        """Test model summary with no data."""
        summary = await slippage_model.get_model_summary()
        
        assert summary["symbols_tracked"] == 0
        assert summary["total_predictions"] == 0
        assert summary["models_fitted"] == 0
        assert summary["symbol_details"] == {}

    @pytest.mark.asyncio
    async def test_get_model_summary_with_data(self, slippage_model):
        """Test model summary with data."""
        symbol = "BTCUSDT"
        
        # Add historical data
        actual_slippage = SlippageMetrics(
            symbol=symbol,
            order_size=Decimal("10.0"),
            market_price=Decimal("50000"),
            execution_price=Decimal("50025"),
            price_slippage_bps=Decimal("5"),
            market_impact_bps=Decimal("3"),
            timing_cost_bps=Decimal("2"),
            total_cost_bps=Decimal("10"),
            spread_bps=Decimal("2"),
            volume_ratio=0.1,
            volatility=0.02,
            timestamp=datetime.now(timezone.utc)
        )
        
        await slippage_model.update_historical_data(symbol, actual_slippage, {})
        
        summary = await slippage_model.get_model_summary()
        
        assert summary["symbols_tracked"] == 1
        assert summary["total_predictions"] == 1
        assert symbol in summary["symbol_details"]
        assert summary["symbol_details"][symbol]["data_points"] == 1

    @pytest.mark.asyncio
    async def test_get_model_summary_specific_symbol(self, slippage_model):
        """Test model summary for specific symbol."""
        symbol = "BTCUSDT"
        
        # Add data for symbol
        actual_slippage = SlippageMetrics(
            symbol=symbol,
            order_size=Decimal("10.0"),
            market_price=Decimal("50000"),
            execution_price=Decimal("50025"),
            price_slippage_bps=Decimal("5"),
            market_impact_bps=Decimal("3"),
            timing_cost_bps=Decimal("2"),
            total_cost_bps=Decimal("10"),
            spread_bps=Decimal("2"),
            volume_ratio=0.1,
            volatility=0.02,
            timestamp=datetime.now(timezone.utc)
        )
        
        await slippage_model.update_historical_data(symbol, actual_slippage, {})
        
        # Add data for another symbol
        await slippage_model.update_historical_data("ETHUSDT", actual_slippage, {})
        
        # Get summary for specific symbol
        summary = await slippage_model.get_model_summary(symbol)
        
        assert summary["symbols_tracked"] == 1
        assert symbol in summary["symbol_details"]
        assert "ETHUSDT" not in summary["symbol_details"]