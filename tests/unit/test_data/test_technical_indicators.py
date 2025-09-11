"""Test suite for technical indicators."""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Mock talib to avoid numpy compatibility issues
mock_talib = MagicMock()
mock_talib.SMA = Mock(return_value=np.array([45100.0]))
mock_talib.EMA = Mock(return_value=np.array([45150.0]))
mock_talib.RSI = Mock(return_value=np.array([65.5]))
mock_talib.MACD = Mock(
    return_value=(
        np.array([10.5]),  # MACD line
        np.array([8.2]),  # Signal line
        np.array([2.3]),  # Histogram
    )
)
mock_talib.BBANDS = Mock(
    return_value=(
        np.array([45200.0]),  # Upper band
        np.array([45100.0]),  # Middle band (SMA)
        np.array([45000.0]),  # Lower band
    )
)
sys.modules["talib"] = mock_talib

from src.core.config import Config
from src.core.types import MarketData
from src.data.features.technical_indicators import (
    IndicatorConfig,
    IndicatorResult,
    IndicatorType,
    TechnicalIndicators,
)


class TestIndicatorConfig:
    """Test suite for IndicatorConfig."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        config = IndicatorConfig(
            indicator_name="sma_20", indicator_type=IndicatorType.PRICE_BASED, period=20
        )

        assert config.indicator_name == "sma_20"
        assert config.indicator_type == IndicatorType.PRICE_BASED
        assert config.period == 20
        assert config.enabled is True
        assert config.parameters is None

    def test_initialization_full(self):
        """Test full initialization."""
        parameters = {"multiplier": 2.0, "source": "close"}

        config = IndicatorConfig(
            indicator_name="bollinger_bands",
            indicator_type=IndicatorType.VOLATILITY,
            period=20,
            enabled=False,
            parameters=parameters,
        )

        assert config.indicator_name == "bollinger_bands"
        assert config.indicator_type == IndicatorType.VOLATILITY
        assert config.period == 20
        assert config.enabled is False
        assert config.parameters == parameters


class TestIndicatorResult:
    """Test suite for IndicatorResult."""

    def test_initialization(self):
        """Test indicator result initialization."""
        timestamp = datetime.now(timezone.utc)
        metadata = {"period": 20, "source": "close"}

        result = IndicatorResult(
            indicator_name="sma_20",
            symbol="BTCUSDT",
            timestamp=timestamp,
            value=Decimal("45000.12"),
            metadata=metadata,
            calculation_time=0.15,
        )

        assert result.indicator_name == "sma_20"
        assert result.symbol == "BTCUSDT"
        assert result.timestamp == timestamp
        assert result.value == Decimal("45000.12")
        assert result.metadata == metadata
        assert result.calculation_time == 0.15

    def test_initialization_with_none_value(self):
        """Test initialization with None value."""
        result = IndicatorResult(
            indicator_name="rsi_14",
            symbol="ETHUSD",
            timestamp=datetime.now(timezone.utc),
            value=None,
            metadata={},
            calculation_time=0.25,
        )

        assert result.value is None
        assert result.calculation_time == 0.25


class TestTechnicalIndicators:
    """Test suite for TechnicalIndicators."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = Mock(spec=Config)

        # Create mock indicators config
        indicators_config = Mock()
        indicators_config.get = Mock(
            side_effect=lambda key, default: {
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
            }.get(key, default)
        )

        config.indicators = indicators_config
        return config

    @pytest.fixture
    def mock_config_simple(self):
        """Create simple mock config without indicators config."""
        config = Mock(spec=Config)
        config.indicators = {}  # Simple dict without get method
        return config

    @pytest.fixture
    def mock_feature_store(self):
        """Create mock feature store."""
        return Mock()

    @pytest.fixture
    def mock_data_service(self):
        """Create mock data service."""
        return Mock()

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data list."""
        base_price = Decimal("45000")
        data = []

        for i in range(50):  # Enough data for most indicators
            price = base_price + Decimal(str(i * 10))  # Increasing prices
            data.append(
                MarketData(
                    symbol="BTCUSDT",
                    open=price - Decimal("10"),
                    high=price + Decimal("50"),
                    low=price - Decimal("30"),
                    close=price,
                    volume=Decimal("1000") + Decimal(str(i)),
                    timestamp=datetime.now(timezone.utc),
                    exchange="binance",
                )
            )

        return data

    @pytest.fixture
    def indicators(self, mock_config):
        """Create technical indicators instance."""
        return TechnicalIndicators(config=mock_config)

    def test_initialization_with_full_config(
        self, mock_config, mock_feature_store, mock_data_service
    ):
        """Test initialization with full configuration."""
        with patch("src.data.features.technical_indicators.ErrorHandler"):
            indicators = TechnicalIndicators(
                config=mock_config, feature_store=mock_feature_store, data_service=mock_data_service
            )

            assert indicators.config is mock_config
            assert indicators.feature_store is mock_feature_store
            assert indicators.data_service is mock_data_service
            assert indicators.cache_enabled is True
            assert indicators.max_calculation_time == 5.0
            assert "sma" in indicators.default_periods
            assert indicators.default_periods["sma"] == 20

    def test_initialization_with_simple_config(self, mock_config_simple):
        """Test initialization with simple config."""
        with patch("src.data.features.technical_indicators.ErrorHandler"):
            indicators = TechnicalIndicators(config=mock_config_simple)

            assert indicators.config is mock_config_simple
            assert indicators.cache_enabled is True
            assert indicators.max_calculation_time == 5.0
            assert indicators.default_periods["sma"] == 20

    def test_set_feature_store(self, indicators, mock_feature_store):
        """Test setting feature store."""
        indicators.set_feature_store(mock_feature_store)

        assert indicators.feature_store is mock_feature_store

    def test_set_data_service(self, indicators, mock_data_service):
        """Test setting data service."""
        indicators.set_data_service(mock_data_service)

        assert indicators.data_service is mock_data_service

    @pytest.mark.asyncio
    async def test_calculate_indicators_batch_empty_data(self, indicators):
        """Test batch calculation with empty data."""
        result = await indicators.calculate_indicators_batch("BTCUSDT", [])

        assert result == {}

    @pytest.mark.asyncio
    async def test_calculate_indicators_batch_no_prices(self, indicators):
        """Test batch calculation with data but no prices."""
        data = [
            MarketData(
                symbol="BTCUSDT",
                open=Decimal("45000"),
                high=Decimal("45100"),
                low=Decimal("44900"),
                close=Decimal("45000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc),
                exchange="binance",
            )
        ]

        result = await indicators.calculate_indicators_batch("BTCUSDT", data)

        assert result == {}

    @pytest.mark.asyncio
    async def test_calculate_indicators_batch_default_indicators(
        self, indicators, sample_market_data
    ):
        """Test batch calculation with default indicators."""
        result = await indicators.calculate_indicators_batch("BTCUSDT", sample_market_data)

        assert isinstance(result, dict)
        # Should have attempted to calculate default indicators

    @pytest.mark.asyncio
    async def test_calculate_indicators_batch_specific_indicators(
        self, indicators, sample_market_data
    ):
        """Test batch calculation with specific indicators."""
        result = await indicators.calculate_indicators_batch(
            "BTCUSDT", sample_market_data, indicators=["sma_20"]
        )

        assert isinstance(result, dict)
        # Should have attempted calculation

    @pytest.mark.asyncio
    async def test_calculate_indicators_batch_with_parameters(self, indicators, sample_market_data):
        """Test batch calculation with custom parameters."""
        parameters = {"sma_period": 30, "rsi_period": 21}

        result = await indicators.calculate_indicators_batch(
            "BTCUSDT", sample_market_data, parameters=parameters
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_calculate_indicators_batch_exception_handling(
        self, indicators, sample_market_data
    ):
        """Test exception handling in batch calculation."""
        with patch("numpy.array", side_effect=Exception("Calculation error")):
            result = await indicators.calculate_indicators_batch("BTCUSDT", sample_market_data)

        # Should handle exception gracefully
        assert isinstance(result, dict)

    def test_calculation_stats_initialization(self, indicators):
        """Test calculation statistics initialization."""
        stats = indicators.calculation_stats

        assert stats["total_calculations"] == 0
        assert stats["successful_calculations"] == 0
        assert stats["failed_calculations"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["avg_calculation_time"] == 0.0

    def test_price_data_initialization(self, indicators):
        """Test price data storage initialization."""
        assert isinstance(indicators.price_data, dict)
        assert len(indicators.price_data) == 0

    def test_feature_cache_initialization(self, indicators):
        """Test feature cache initialization."""
        assert isinstance(indicators.feature_cache, dict)
        assert len(indicators.feature_cache) == 0


class TestEnums:
    """Test suite for indicator enums."""

    def test_indicator_type_values(self):
        """Test indicator type enum values."""
        assert IndicatorType.PRICE_BASED.value == "price_based"
        assert IndicatorType.MOMENTUM.value == "momentum"
        assert IndicatorType.VOLUME.value == "volume"
        assert IndicatorType.VOLATILITY.value == "volatility"
        assert IndicatorType.MARKET_STRUCTURE.value == "market_structure"
