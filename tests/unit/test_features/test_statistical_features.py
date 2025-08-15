"""
Unit tests for StatisticalFeatureCalculator.

This module tests statistical feature calculations including:
- Rolling statistics, autocorrelation, regime detection
- Cross-correlation analysis, seasonality detection
- Error handling and edge cases
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.core.config import Config
from src.core.exceptions import DataError
from src.core.types import MarketData
from src.data.features.statistical_features import (
    StatisticalFeatureCalculator,
    StatisticalResult,
    StatFeatureType,
    RegimeType,
    StatisticalConfig,
)


class TestStatisticalFeatureCalculator:
    """Test suite for StatisticalFeatureCalculator."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = MagicMock(spec=Config)
        config.statistical_features = {
            "default_windows": {
                "rolling_stats": 20,
                "autocorr": 50,
                "regime": 100,
                "seasonality": 252,
            },
            "regime_threshold": 0.02,
            "correlation_threshold": 0.7,
        }
        config.max_price_history = 2000
        return config

    @pytest.fixture
    def calculator(self, config):
        """Create test statistical feature calculator."""
        return StatisticalFeatureCalculator(config)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data with realistic patterns."""
        data = []
        base_price = 100.0
        trend = 0.001  # 0.1% trend per period
        
        for i in range(100):
            # Add trend and noise
            price_change = trend + np.random.normal(0, 0.02)
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
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            data.append(market_data)
        
        return data

    @pytest.fixture
    def cyclical_market_data(self):
        """Create market data with cyclical patterns for seasonality testing."""
        data = []
        base_price = 100.0
        
        for i in range(500):  # Longer series for seasonality
            # Add cyclical pattern (daily cycle)
            cycle = np.sin(2 * np.pi * i / 24) * 0.01  # 24-period cycle
            noise = np.random.normal(0, 0.005)
            price_change = cycle + noise
            
            base_price *= (1 + price_change)
            
            market_data = MarketData(
                symbol="CYCLICAL",
                price=base_price,
                open_price=base_price,
                high_price=base_price * 1.01,
                low_price=base_price * 0.99,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(hours=i),
                source="test",
            )
            data.append(market_data)
        
        return data

    @pytest.mark.asyncio
    async def test_calculator_initialization(self, config):
        """Test calculator initialization."""
        calculator = StatisticalFeatureCalculator(config)
        
        assert calculator.config == config
        assert calculator.default_windows["rolling_stats"] == 20
        assert calculator.regime_threshold == 0.02
        assert len(calculator.price_data) == 0

    @pytest.mark.asyncio
    async def test_add_market_data(self, calculator, sample_market_data):
        """Test adding market data with returns calculation."""
        # Add first data point
        await calculator.add_market_data(sample_market_data[0])
        
        assert "BTCUSDT" in calculator.price_data
        assert len(calculator.price_data["BTCUSDT"]) == 1
        
        # Returns should be 0 for first data point
        df = calculator.price_data["BTCUSDT"]
        assert df["returns"].iloc[0] == 0.0
        assert df["log_returns"].iloc[0] == 0.0
        
        # Add second data point
        await calculator.add_market_data(sample_market_data[1])
        
        # Returns should be calculated
        assert len(calculator.price_data["BTCUSDT"]) == 2
        assert df["returns"].iloc[1] != 0.0  # Should have calculated return

    @pytest.mark.asyncio
    async def test_rolling_stats_calculation(self, calculator, sample_market_data):
        """Test rolling statistics calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate rolling stats
        result = await calculator.calculate_rolling_stats("BTCUSDT", window=20)
        
        assert isinstance(result, StatisticalResult)
        assert result.feature_name == "ROLLING_STATS"
        assert isinstance(result.value, dict)
        
        # Check all required statistics are present
        required_stats = ["mean", "std", "skewness", "kurtosis", "min", "max", 
                         "median", "q25", "q75", "z_score", "latest_value"]
        assert all(stat in result.value for stat in required_stats)
        
        # Verify z_score calculation
        assert isinstance(result.value["z_score"], (int, float))

    @pytest.mark.asyncio
    async def test_rolling_stats_insufficient_data(self, calculator, sample_market_data):
        """Test rolling stats with insufficient data."""
        # Add only 5 data points
        for data in sample_market_data[:5]:
            await calculator.add_market_data(data)
        
        # Try to calculate with window=20
        result = await calculator.calculate_rolling_stats("BTCUSDT", window=20)
        
        assert result.value is None
        assert result.metadata["reason"] == "insufficient_data"

    @pytest.mark.asyncio
    async def test_autocorrelation_calculation(self, calculator, sample_market_data):
        """Test autocorrelation calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate autocorrelation
        result = await calculator.calculate_autocorrelation("BTCUSDT", max_lags=10)
        
        assert isinstance(result, StatisticalResult)
        assert result.feature_name == "AUTOCORRELATION"
        assert isinstance(result.value, dict)
        
        # Check required fields
        required_fields = ["autocorrelations", "max_autocorr", "min_autocorr", 
                          "mean_autocorr", "significant_lags"]
        assert all(field in result.value for field in required_fields)
        
        # Autocorrelations should be a list
        assert isinstance(result.value["autocorrelations"], list)
        assert len(result.value["autocorrelations"]) == 10

    @pytest.mark.asyncio
    async def test_regime_detection(self, calculator, sample_market_data):
        """Test regime detection."""
        # Add data with trend
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate regime
        result = await calculator.detect_regime("BTCUSDT", window=50)
        
        assert isinstance(result, StatisticalResult)
        assert result.feature_name == "REGIME_DETECTION"
        assert isinstance(result.value, dict)
        
        # Check regime classification
        regime_types = [regime.value for regime in RegimeType]
        assert result.value["regime"] in regime_types
        
        # Check confidence is between 0 and 1
        assert 0 <= result.value["confidence"] <= 1
        
        # Check other metrics
        assert "price_trend" in result.value
        assert "volatility_percentile" in result.value
        assert "directional_movement" in result.value

    @pytest.mark.asyncio
    async def test_cross_correlation(self, calculator, sample_market_data):
        """Test cross-correlation between two symbols."""
        # Create correlated data for second symbol
        correlated_data = []
        for i, data in enumerate(sample_market_data):
            # Create correlated price (same trend + noise)
            correlated_price = data.price * (1 + np.random.normal(0, 0.01))
            
            correlated_market_data = MarketData(
                symbol="ETHUSDT",
                price=correlated_price,
                open_price=correlated_price,
                high_price=correlated_price * 1.01,
                low_price=correlated_price * 0.99,
                volume=data.volume,
                timestamp=data.timestamp,
                source="test",
            )
            correlated_data.append(correlated_market_data)
        
        # Add data for both symbols
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        for data in correlated_data:
            await calculator.add_market_data(data)
        
        # Calculate cross-correlation
        result = await calculator.calculate_cross_correlation("BTCUSDT", "ETHUSDT", max_lags=10)
        
        assert isinstance(result, StatisticalResult)
        assert result.feature_name == "CROSS_CORRELATION"
        assert result.symbol == "BTCUSDT_ETHUSDT"
        assert isinstance(result.value, dict)
        
        # Check correlation metrics
        required_fields = ["contemporaneous_correlation", "max_correlation", 
                          "max_correlation_lag", "correlation_strength"]
        assert all(field in result.value for field in required_fields)
        
        # Correlation should be between -1 and 1
        assert -1 <= result.value["contemporaneous_correlation"] <= 1

    @pytest.mark.asyncio
    async def test_seasonality_detection(self, calculator, cyclical_market_data):
        """Test seasonality detection with cyclical data."""
        # Add cyclical data
        for data in cyclical_market_data:
            await calculator.add_market_data(data)
        
        # Calculate seasonality
        result = await calculator.detect_seasonality("CYCLICAL")
        
        assert isinstance(result, StatisticalResult)
        assert result.feature_name == "SEASONALITY"
        assert isinstance(result.value, dict)
        
        # Check seasonal patterns
        required_fields = ["hourly_pattern", "daily_pattern", "monthly_pattern",
                          "hourly_variance", "daily_variance", "monthly_variance",
                          "strongest_pattern"]
        assert all(field in result.value for field in required_fields)
        
        # Pattern should be one of the expected types
        assert result.value["strongest_pattern"] in ["hourly", "daily", "monthly"]

    @pytest.mark.asyncio
    async def test_batch_features_calculation(self, calculator, sample_market_data):
        """Test batch feature calculation."""
        # Add data
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        # Calculate multiple features
        features = ["ROLLING_STATS", "AUTOCORRELATION", "REGIME"]
        results = await calculator.calculate_batch_features("BTCUSDT", features)
        
        assert isinstance(results, dict)
        assert len(results) == len(features)
        
        # Check that all features were calculated
        for feature in features:
            assert feature in results
            assert results[feature] is not None
            assert isinstance(results[feature], StatisticalResult)

    @pytest.mark.asyncio
    async def test_no_data_error(self, calculator):
        """Test error handling when no data is available."""
        with pytest.raises(DataError, match="No price data available"):
            await calculator.calculate_rolling_stats("NONEXISTENT")

    @pytest.mark.asyncio
    async def test_cross_correlation_missing_symbol(self, calculator, sample_market_data):
        """Test cross-correlation with missing second symbol."""
        # Add data for only one symbol
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        with pytest.raises(DataError, match="No price data available for symbols"):
            await calculator.calculate_cross_correlation("BTCUSDT", "MISSING")

    @pytest.mark.asyncio
    async def test_calculation_summary(self, calculator, sample_market_data):
        """Test calculation statistics summary."""
        # Add data and perform calculations
        for data in sample_market_data:
            await calculator.add_market_data(data)
        
        await calculator.calculate_rolling_stats("BTCUSDT")
        await calculator.calculate_autocorrelation("BTCUSDT")
        
        # Get summary
        summary = await calculator.get_calculation_summary()
        
        assert isinstance(summary, dict)
        assert "statistics" in summary
        assert "success_rate" in summary
        assert "symbols_tracked" in summary
        assert summary["symbols_tracked"] == 1
        assert summary["statistics"]["successful_calculations"] >= 2

    @pytest.mark.asyncio
    async def test_regime_trending_up(self, calculator):
        """Test regime detection for strong uptrend."""
        # Create strong uptrend data
        base_price = 100.0
        trend_data = []
        
        for i in range(100):
            base_price *= 1.01  # 1% increase per period
            
            market_data = MarketData(
                symbol="UPTREND",
                price=base_price,
                open_price=base_price,
                high_price=base_price * 1.005,
                low_price=base_price * 0.995,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            trend_data.append(market_data)
        
        # Add data
        for data in trend_data:
            await calculator.add_market_data(data)
        
        # Detect regime
        result = await calculator.detect_regime("UPTREND", window=50)
        
        # Should detect trending up
        assert result.value["regime"] == RegimeType.TRENDING_UP.value
        assert result.value["confidence"] > 0.5

    @pytest.mark.asyncio
    async def test_regime_ranging(self, calculator):
        """Test regime detection for ranging market."""
        # Create ranging market data
        base_price = 100.0
        ranging_data = []
        
        for i in range(100):
            # Small random movements around base price
            price_change = np.random.normal(0, 0.005)  # Small noise
            price = base_price * (1 + price_change)
            
            market_data = MarketData(
                symbol="RANGING",
                price=price,
                open_price=price,
                high_price=price * 1.001,
                low_price=price * 0.999,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            ranging_data.append(market_data)
        
        # Add data
        for data in ranging_data:
            await calculator.add_market_data(data)
        
        # Detect regime
        result = await calculator.detect_regime("RANGING", window=50)
        
        # Should detect ranging or low volatility
        assert result.value["regime"] in [RegimeType.RANGING.value, RegimeType.LOW_VOLATILITY.value]

    @pytest.mark.asyncio
    async def test_autocorr_with_trending_data(self, calculator):
        """Test autocorrelation with trending data."""
        # Create trending data that should show positive autocorrelation
        base_price = 100.0
        trending_data = []
        
        for i in range(60):
            base_price *= 1.005  # Consistent trend
            
            market_data = MarketData(
                symbol="TRENDING",
                price=base_price,
                open_price=base_price,
                high_price=base_price * 1.001,
                low_price=base_price * 0.999,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            trending_data.append(market_data)
        
        # Add data
        for data in trending_data:
            await calculator.add_market_data(data)
        
        # Calculate autocorrelation
        result = await calculator.calculate_autocorrelation("TRENDING", max_lags=5)
        
        # Should have positive autocorrelations at short lags for trending data
        autocorrs = result.value["autocorrelations"]
        assert len(autocorrs) == 5
        # First few lags should be positive for trending data
        assert autocorrs[0] > 0

    @pytest.mark.asyncio
    async def test_memory_management(self, calculator, config):
        """Test memory management with large datasets."""
        config.max_price_history = 50
        
        # Create more data than max_price_history
        for i in range(100):
            data = MarketData(
                symbol="BTCUSDT",
                price=100 + i * 0.1,
                open_price=100 + i * 0.1,
                high_price=100 + i * 0.1 + 0.1,
                low_price=100 + i * 0.1 - 0.1,
                volume=1000,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test",
            )
            await calculator.add_market_data(data)
        
        # Should only keep max_price_history records
        assert len(calculator.price_data["BTCUSDT"]) == 50


class TestStatisticalConfig:
    """Test suite for StatisticalConfig dataclass."""
    
    def test_statistical_config_creation(self):
        """Test StatisticalConfig creation."""
        config = StatisticalConfig(
            feature_name="ROLLING_STATS",
            feature_type=StatFeatureType.ROLLING_STATS,
            window_size=20,
            enabled=True,
            parameters={"field": "returns"}
        )
        
        assert config.feature_name == "ROLLING_STATS"
        assert config.feature_type == StatFeatureType.ROLLING_STATS
        assert config.window_size == 20
        assert config.enabled is True
        assert config.parameters["field"] == "returns"


class TestRegimeType:
    """Test suite for RegimeType enum."""
    
    def test_regime_type_values(self):
        """Test RegimeType enum values."""
        assert RegimeType.TRENDING_UP.value == "trending_up"
        assert RegimeType.TRENDING_DOWN.value == "trending_down"
        assert RegimeType.RANGING.value == "ranging"
        assert RegimeType.HIGH_VOLATILITY.value == "high_volatility"
        assert RegimeType.LOW_VOLATILITY.value == "low_volatility"
        assert RegimeType.UNKNOWN.value == "unknown"


@pytest.mark.integration
class TestStatisticalAccuracy:
    """Integration tests for statistical calculation accuracy."""
    
    @pytest.mark.asyncio
    async def test_rolling_mean_accuracy(self):
        """Test rolling mean calculation accuracy."""
        config = MagicMock(spec=Config)
        config.statistical_features = {"default_windows": {"rolling_stats": 5}}
        config.max_price_history = 1000
        
        calculator = StatisticalFeatureCalculator(config)
        
        # Use known returns for verification
        returns = [0.01, 0.02, -0.01, 0.015, 0.005]  # 5 returns
        base_price = 100.0
        
        for i, ret in enumerate([0] + returns):  # First return is 0
            price = base_price * (1 + ret)
            base_price = price
            
            data = MarketData(
                symbol="TEST",
                price=price,
                open_price=price,
                high_price=price,
                low_price=price,
                volume=100,
                timestamp=datetime.now(timezone.utc) + timedelta(minutes=i),
                source="test"
            )
            await calculator.add_market_data(data)
        
        # Calculate rolling stats for window=5
        result = await calculator.calculate_rolling_stats("TEST", window=5, field="returns")
        
        # Expected mean of returns [0.01, 0.02, -0.01, 0.015, 0.005] = 0.01
        expected_mean = np.mean(returns)
        
        assert abs(result.value["mean"] - expected_mean) < 0.001