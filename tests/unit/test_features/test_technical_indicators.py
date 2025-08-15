"""
Unit tests for TechnicalIndicatorCalculator.

This module tests all technical indicator calculations including:
- SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- Stochastic, Williams %R, CCI
- Error handling and edge cases
- Performance and accuracy
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.config import Config
from src.core.exceptions import DataError
from src.core.types import MarketData
from src.data.features.technical_indicators import (
    TechnicalIndicatorCalculator,
    IndicatorResult,
    IndicatorType,
    IndicatorConfig,
)


class TestTechnicalIndicatorCalculator:
    """Test suite for TechnicalIndicatorCalculator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=Config)
        config.indicators = {
            "default_periods": {
                "sma": 20,
                "ema": 20,
                "rsi": 14,
                "macd": [12, 26, 9],
                "bollinger": 20,
                "atr": 14,
            },
            "cache_enabled": True,
            "max_calculation_time": 5.0,
        }
        config.max_price_history = 1000
        return config

    @pytest.fixture
    def calculator(self, config):
        """Create test technical indicator calculator."""
        return TechnicalIndicatorCalculator(config)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        data = []
        base_price = 100.0
        
        for i in range(50):
            # Create realistic price movements
            price_change = np.random.normal(0, 0.02)
            base_price *= (1 + price_change)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.01)))
            low = base_price * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(1000, 10000)
            
            market_data = MarketData(
                symbol="BTCUSDT",
                price=base_price,
                open_price=base_price * (1 + np.random.normal(0, 0.005)),
                high_price=high,
                low_price=low,
                volume=volume,
                timestamp=datetime.now(timezone.utc),
                source="test",
            )
            data.append(market_data)
        
        return data

    @pytest.mark.asyncio
    async def test_calculator_initialization(self, config):
        """Test calculator initialization."""
        calculator = TechnicalIndicatorCalculator(config)
        
        assert calculator.config == config
        assert calculator.default_periods["sma"] == 20
        assert calculator.cache_enabled is True
        assert len(calculator.price_data) == 0
        assert calculator.calculation_stats["total_calculations"] == 0

    @pytest.mark.asyncio
    async def test_add_market_data(self, calculator, sample_market_data):
        """Test adding market data to calculator."""
        # Add single data point
        await calculator.add_market_data(sample_market_data[0])
        
        assert "BTCUSDT" in calculator.price_data
        assert len(calculator.price_data["BTCUSDT"]) == 1
        
        # Add multiple data points
        for data in sample_market_data[1:10]:
            await calculator.add_market_data(data)
        
        assert len(calculator.price_data["BTCUSDT"]) == 10
        
        # Verify data structure
        df = calculator.price_data["BTCUSDT"]
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        assert all(col in df.columns for col in required_columns)

    @pytest.mark.asyncio
    async def test_sma_calculation(self, calculator, sample_market_data):
        """Test SMA calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate SMA
        result = await calculator.calculate_sma("BTCUSDT", period=20)
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "SMA"
        assert result.symbol == "BTCUSDT"
        assert result.value is not None
        assert isinstance(result.value, float)
        assert result.metadata["period"] == 20
        assert result.calculation_time >= 0

    @pytest.mark.asyncio
    async def test_sma_insufficient_data(self, calculator, sample_market_data):
        """Test SMA calculation with insufficient data."""
        # Add only 5 data points
        for data in sample_market_data[:5]:
            await calculator.add_market_data(data)
        
        # Try to calculate SMA with period 20
        result = await calculator.calculate_sma("BTCUSDT", period=20)
        
        assert result.value is None
        assert result.metadata["reason"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_ema_calculation(self, calculator, sample_market_data):
        """Test EMA calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate EMA
        result = await calculator.calculate_ema("BTCUSDT", period=20)
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "EMA"
        assert result.value is not None
        assert isinstance(result.value, float)

    @pytest.mark.asyncio
    async def test_rsi_calculation(self, calculator, sample_market_data):
        """Test RSI calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate RSI
        result = await calculator.calculate_rsi("BTCUSDT", period=14)
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "RSI"
        assert result.value is not None
        assert 0 <= result.value <= 100
        assert result.metadata["signal"] in ["overbought", "oversold", "neutral"]

    @pytest.mark.asyncio
    async def test_macd_calculation(self, calculator, sample_market_data):
        """Test MACD calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate MACD
        result = await calculator.calculate_macd("BTCUSDT")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "MACD"
        assert result.value is not None
        assert "macd_line" in result.metadata
        assert "signal_line" in result.metadata
        assert "histogram" in result.metadata
        assert result.metadata["trend_signal"] in ["bullish", "bearish", "neutral"]

    @pytest.mark.asyncio
    async def test_bollinger_bands_calculation(self, calculator, sample_market_data):
        """Test Bollinger Bands calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate Bollinger Bands
        result = await calculator.calculate_bollinger_bands("BTCUSDT", period=20, std_dev=2.0)
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "BOLLINGER_BANDS"
        assert result.value is not None  # Should return middle band
        assert "upper_band" in result.metadata
        assert "lower_band" in result.metadata
        assert "band_width" in result.metadata
        assert result.metadata["signal"] in ["overbought", "oversold", "neutral"]

    @pytest.mark.asyncio
    async def test_atr_calculation(self, calculator, sample_market_data):
        """Test ATR calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate ATR
        result = await calculator.calculate_atr("BTCUSDT", period=14)
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "ATR"
        assert result.value is not None
        assert result.value > 0  # ATR should always be positive
        assert "volatility_percentage" in result.metadata

    @pytest.mark.asyncio
    async def test_stochastic_calculation(self, calculator, sample_market_data):
        """Test Stochastic calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate Stochastic
        result = await calculator.calculate_stochastic("BTCUSDT")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "STOCHASTIC"
        assert result.value is not None
        assert 0 <= result.value <= 100
        assert "k_value" in result.metadata
        assert "d_value" in result.metadata
        assert result.metadata["signal"] in [
            "overbought", "oversold", "bullish_crossover", "bearish_crossover", "neutral"
        ]

    @pytest.mark.asyncio
    async def test_williams_r_calculation(self, calculator, sample_market_data):
        """Test Williams %R calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate Williams %R
        result = await calculator.calculate_williams_r("BTCUSDT")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "WILLIAMS_R"
        assert result.value is not None
        assert -100 <= result.value <= 0  # Williams %R range
        assert result.metadata["signal"] in ["overbought", "oversold", "neutral"]

    @pytest.mark.asyncio
    async def test_cci_calculation(self, calculator, sample_market_data):
        """Test CCI calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate CCI
        result = await calculator.calculate_cci("BTCUSDT")
        
        assert isinstance(result, IndicatorResult)
        assert result.indicator_name == "CCI"
        assert result.value is not None
        assert result.metadata["signal"] in ["overbought", "oversold", "neutral"]

    @pytest.mark.asyncio
    async def test_batch_indicators_calculation(self, calculator, sample_market_data):
        """Test batch indicators calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate multiple indicators
        indicators = ["SMA", "EMA", "RSI", "MACD", "BOLLINGER", "ATR", "STOCHASTIC"]
        results = await calculator.calculate_batch_indicators("BTCUSDT", indicators)
        
        assert isinstance(results, dict)
        assert len(results) == len(indicators)
        
        # Check that all indicators were calculated
        for indicator in indicators:
            assert indicator in results
            assert results[indicator] is not None
            assert isinstance(results[indicator], IndicatorResult)

    @pytest.mark.asyncio
    async def test_no_data_error(self, calculator):
        """Test error handling when no data is available."""
        with pytest.raises(DataError, match="No price data available"):
            await calculator.calculate_sma("NONEXISTENT")

    @pytest.mark.asyncio
    async def test_calculation_statistics(self, calculator, sample_market_data):
        """Test calculation statistics tracking."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Perform calculations
        await calculator.calculate_sma("BTCUSDT")
        await calculator.calculate_rsi("BTCUSDT")
        
        # Get statistics
        summary = await calculator.get_calculation_summary()
        
        assert "statistics" in summary
        assert summary["statistics"]["successful_calculations"] >= 2
        assert "success_rate" in summary
        assert "symbols_tracked" in summary
        assert summary["symbols_tracked"] == 1

    @pytest.mark.asyncio
    async def test_data_validation(self, calculator):
        """Test data validation in market data."""
        # Test with invalid data
        invalid_data = MarketData(
            symbol="INVALID",
            price=0,  # Invalid price
            open_price=None,
            high_price=None,
            low_price=None,
            volume=0,
            timestamp=datetime.now(timezone.utc),
            source="test",
        )
        
        # Should handle gracefully
        await calculator.add_market_data(invalid_data)
        assert "INVALID" in calculator.price_data

    @pytest.mark.asyncio
    async def test_memory_management(self, calculator, config):
        """Test memory management with max_price_history."""
        config.max_price_history = 10
        
        # Create more data than max_price_history
        for i in range(20):
            data = MarketData(
                symbol="BTCUSDT",
                price=100 + i,
                open_price=100 + i,
                high_price=101 + i,
                low_price=99 + i,
                volume=1000,
                timestamp=datetime.now(timezone.utc),
                source="test",
            )
            await calculator.add_market_data(data)
        
        # Should only keep max_price_history records
        assert len(calculator.price_data["BTCUSDT"]) == 10

    @pytest.mark.asyncio
    async def test_cleanup(self, calculator, sample_market_data):
        """Test calculator cleanup."""
        # Add data and perform calculations
        for data in sample_market_data[:10]:
            await calculator.add_market_data(data)
        
        await calculator.calculate_sma("BTCUSDT")
        
        # Verify data exists
        assert len(calculator.price_data) > 0
        assert calculator.calculation_stats["successful_calculations"] > 0
        
        # Cleanup
        await calculator.cleanup()
        
        # Verify cleanup
        assert len(calculator.price_data) == 0
        assert calculator.calculation_stats["successful_calculations"] == 0

    @pytest.mark.asyncio
    async def test_error_analytics(self, calculator):
        """Test error analytics functionality."""
        analytics = await calculator.get_error_analytics()
        
        assert isinstance(analytics, dict)
        assert "error_patterns" in analytics
        assert "circuit_breaker_status" in analytics
        assert "calculation_stats" in analytics

    @pytest.mark.asyncio
    async def test_concurrent_calculations(self, calculator, sample_market_data):
        """Test concurrent indicator calculations."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Run multiple calculations concurrently
        tasks = [
            calculator.calculate_sma("BTCUSDT"),
            calculator.calculate_ema("BTCUSDT"),
            calculator.calculate_rsi("BTCUSDT"),
            calculator.calculate_macd("BTCUSDT"),
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All calculations should succeed
        assert len(results) == 4
        assert all(isinstance(result, IndicatorResult) for result in results)
        assert all(result.value is not None for result in results)


class TestIndicatorConfig:
    """Test suite for IndicatorConfig dataclass."""
    
    def test_indicator_config_creation(self):
        """Test IndicatorConfig creation."""
        config = IndicatorConfig(
            indicator_name="SMA",
            indicator_type=IndicatorType.PRICE_BASED,
            period=20,
            enabled=True,
            parameters={"field": "close"}
        )
        
        assert config.indicator_name == "SMA"
        assert config.indicator_type == IndicatorType.PRICE_BASED
        assert config.period == 20
        assert config.enabled is True
        assert config.parameters["field"] == "close"


class TestIndicatorResult:
    """Test suite for IndicatorResult dataclass."""
    
    def test_indicator_result_creation(self):
        """Test IndicatorResult creation."""
        result = IndicatorResult(
            indicator_name="RSI",
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            value=65.5,
            metadata={"period": 14, "signal": "neutral"},
            calculation_time=0.05
        )
        
        assert result.indicator_name == "RSI"
        assert result.symbol == "BTCUSDT"
        assert result.value == 65.5
        assert result.metadata["signal"] == "neutral"
        assert result.calculation_time == 0.05


@pytest.mark.integration
class TestIndicatorAccuracy:
    """Integration tests for indicator calculation accuracy."""
    
    @pytest.mark.asyncio
    async def test_sma_accuracy(self):
        """Test SMA calculation accuracy against known values."""
        config = MagicMock(spec=Config)
        config.indicators = {"default_periods": {"sma": 5}}
        config.max_price_history = 1000
        
        calculator = TechnicalIndicatorCalculator(config)
        
        # Use known prices for verification
        prices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        for i, price in enumerate(prices):
            data = MarketData(
                symbol="TEST",
                price=price,
                open_price=price,
                high_price=price,
                low_price=price,
                volume=100,
                timestamp=datetime.now(timezone.utc),
                source="test"
            )
            await calculator.add_market_data(data)
        
        # Calculate SMA(5) for last 5 prices: [15, 16, 17, 18, 19]
        result = await calculator.calculate_sma("TEST", period=5)
        expected_sma = sum([15, 16, 17, 18, 19]) / 5  # = 17.0
        
        assert abs(result.value - expected_sma) < 0.001