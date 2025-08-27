"""
Unit tests for market regime detection module (P-010).

This module tests the MarketRegimeDetector class and related functionality
for dynamic risk management.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from src.core.types.market import MarketData
from src.core.types.strategy import MarketRegime, RegimeChangeEvent

# Import the modules to test
from src.risk_management.regime_detection import MarketRegimeDetector


class TestMarketRegimeDetector:
    """Test cases for MarketRegimeDetector class."""

    @pytest.fixture
    def config(self):
        """Test configuration for regime detector."""
        return {
            "volatility_window": 20,
            "trend_window": 50,
            "correlation_window": 30,
            "regime_change_threshold": 0.7,
        }

    @pytest.fixture
    def regime_detector(self, config):
        """Create a regime detector instance for testing."""
        return MarketRegimeDetector(config)

    @pytest.fixture
    def sample_price_data(self):
        """Generate sample price data for testing."""
        # Generate 100 price points with some volatility
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 100)  # 2% daily volatility
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        return prices

    @pytest.fixture
    def trending_up_data(self):
        """Generate trending up price data."""
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        trend = 0.005  # 0.5% daily trend (increased for stronger trend)
        prices = [base_price]
        for i in range(100):
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.01)))
        return prices

    @pytest.fixture
    def trending_down_data(self):
        """Generate trending down price data."""
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        trend = -0.001  # -0.1% daily trend
        prices = [base_price]
        for i in range(100):
            prices.append(prices[-1] * (1 + trend + np.random.normal(0, 0.01)))
        return prices

    @pytest.fixture
    def high_volatility_data(self):
        """Generate high volatility price data."""
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        prices = [base_price]
        for i in range(100):
            # High volatility: 5% daily volatility
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.05)))
        return prices

    @pytest.fixture
    def low_volatility_data(self):
        """Generate low volatility price data."""
        np.random.seed(42)  # For reproducible tests
        base_price = 100.0
        prices = [base_price]
        for i in range(100):
            # Low volatility: 0.5% daily volatility
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))
        return prices

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        timestamp = datetime.now(timezone.utc)
        return [
            MarketData(
                symbol="BTCUSDT",
                timestamp=timestamp,
                open=Decimal("49000"),
                high=Decimal("50500"),
                low=Decimal("48500"),
                close=Decimal("50000"),
                volume=Decimal("1000"),
                exchange="binance",
            ),
            MarketData(
                symbol="ETHUSDT",
                timestamp=timestamp,
                open=Decimal("2900"),
                high=Decimal("3100"),
                low=Decimal("2850"),
                close=Decimal("3000"),
                volume=Decimal("500"),
                exchange="binance",
            ),
            MarketData(
                symbol="ADAUSDT",
                timestamp=timestamp,
                open=Decimal("1.45"),
                high=Decimal("1.55"),
                low=Decimal("1.42"),
                close=Decimal("1.5"),
                volume=Decimal("2000"),
                exchange="binance",
            ),
        ]

    def test_initialization(self, config):
        """Test regime detector initialization."""
        detector = MarketRegimeDetector(config)

        assert detector.volatility_window == 20
        assert detector.trend_window == 50
        assert detector.correlation_window == 30
        assert detector.regime_change_threshold == 0.7
        assert detector.current_regime == MarketRegime.MEDIUM_VOLATILITY
        assert len(detector.regime_history) == 0

    @pytest.mark.asyncio
    async def test_detect_volatility_regime_low_volatility(
        self, regime_detector, low_volatility_data
    ):
        """Test volatility regime detection for low volatility data."""
        regime = await regime_detector.detect_volatility_regime("BTCUSDT", low_volatility_data)
        assert regime == MarketRegime.LOW_VOLATILITY

    @pytest.mark.asyncio
    async def test_detect_volatility_regime_medium_volatility(
        self, regime_detector, sample_price_data
    ):
        """Test volatility regime detection for medium volatility data."""
        regime = await regime_detector.detect_volatility_regime("BTCUSDT", sample_price_data)
        assert regime == MarketRegime.MEDIUM_VOLATILITY

    @pytest.mark.asyncio
    async def test_detect_volatility_regime_high_volatility(
        self, regime_detector, high_volatility_data
    ):
        """Test volatility regime detection for high volatility data."""
        regime = await regime_detector.detect_volatility_regime("BTCUSDT", high_volatility_data)
        assert regime == MarketRegime.HIGH_VOLATILITY

    @pytest.mark.asyncio
    async def test_detect_volatility_regime_insufficient_data(self, regime_detector):
        """Test volatility regime detection with insufficient data."""
        # Test with less data than required window
        insufficient_data = [100.0, 101.0, 99.0]  # Only 3 data points
        regime = await regime_detector.detect_volatility_regime("BTCUSDT", insufficient_data)
        assert regime == MarketRegime.MEDIUM_VOLATILITY  # Default fallback

    @pytest.mark.asyncio
    async def test_detect_trend_regime_trending_up(self, regime_detector, trending_up_data):
        """Test trend regime detection for trending up data."""
        regime = await regime_detector.detect_trend_regime("BTCUSDT", trending_up_data)
        assert regime == MarketRegime.TRENDING_UP

    @pytest.mark.asyncio
    async def test_detect_trend_regime_trending_down(self, regime_detector, trending_down_data):
        """Test trend regime detection for trending down data."""
        regime = await regime_detector.detect_trend_regime("BTCUSDT", trending_down_data)
        assert regime == MarketRegime.TRENDING_DOWN

    @pytest.mark.asyncio
    async def test_detect_trend_regime_ranging(self, regime_detector):
        """Test trend regime detection for ranging data."""
        # Generate ranging data (no clear trend)
        ranging_data = [100.0 + np.random.normal(0, 2) for _ in range(100)]
        regime = await regime_detector.detect_trend_regime("BTCUSDT", ranging_data)
        assert regime == MarketRegime.RANGING

    @pytest.mark.asyncio
    async def test_detect_trend_regime_insufficient_data(self, regime_detector):
        """Test trend regime detection with insufficient data."""
        insufficient_data = [100.0, 101.0, 99.0]  # Only 3 data points
        regime = await regime_detector.detect_trend_regime("BTCUSDT", insufficient_data)
        assert regime == MarketRegime.RANGING  # Default fallback

    @pytest.mark.asyncio
    async def test_detect_correlation_regime_high_correlation(self, regime_detector):
        """Test correlation regime detection for highly correlated data."""
        # Create highly correlated price data with deterministic generation
        np.random.seed(42)  # For reproducible tests

        # Create a common factor that will drive correlation
        n_points = 50
        common_factor = np.random.normal(0, 1, n_points)

        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        price_data = {
            "BTCUSDT": [
                100 + 0.8 * common_factor[i] + 0.2 * np.random.normal(0, 1) for i in range(n_points)
            ],
            "ETHUSDT": [
                100 + 0.8 * common_factor[i] + 0.2 * np.random.normal(0, 1) for i in range(n_points)
            ],
            "ADAUSDT": [
                100 + 0.8 * common_factor[i] + 0.2 * np.random.normal(0, 1) for i in range(n_points)
            ],
        }

        regime = await regime_detector.detect_correlation_regime(symbols, price_data)
        assert regime == MarketRegime.HIGH_CORRELATION

    @pytest.mark.asyncio
    async def test_detect_correlation_regime_low_correlation(self, regime_detector):
        """Test correlation regime detection for low correlation data."""
        # Create uncorrelated price data with deterministic generation
        np.random.seed(42)  # For reproducible tests
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        price_data = {
            "BTCUSDT": [100 + np.random.normal(0, 1) for _ in range(50)],
            "ETHUSDT": [200 + np.random.normal(0, 1) for _ in range(50)],
            "ADAUSDT": [1 + np.random.normal(0, 0.1) for _ in range(50)],
        }

        regime = await regime_detector.detect_correlation_regime(symbols, price_data)
        assert regime == MarketRegime.LOW_CORRELATION

    @pytest.mark.asyncio
    async def test_detect_correlation_regime_insufficient_symbols(self, regime_detector):
        """Test correlation regime detection with insufficient symbols."""
        symbols = ["BTCUSDT"]  # Only one symbol
        price_data = {"BTCUSDT": [100, 101, 99]}

        regime = await regime_detector.detect_correlation_regime(symbols, price_data)
        assert regime == MarketRegime.LOW_CORRELATION  # Default fallback

    @pytest.mark.asyncio
    async def test_detect_comprehensive_regime(self, regime_detector, sample_market_data):
        """Test comprehensive regime detection."""
        regime = await regime_detector.detect_comprehensive_regime(sample_market_data)
        assert regime in [
            MarketRegime.MEDIUM_VOLATILITY,
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.RANGING,
        ]

    @pytest.mark.asyncio
    async def test_detect_comprehensive_regime_empty_data(self, regime_detector):
        """Test comprehensive regime detection with empty data."""
        regime = await regime_detector.detect_comprehensive_regime([])
        assert regime == MarketRegime.MEDIUM_VOLATILITY  # Default fallback

    def test_combine_regimes(self, regime_detector):
        """Test regime combination logic."""
        # Test with dominant volatility regime
        volatility_regimes = [MarketRegime.HIGH_VOLATILITY, MarketRegime.HIGH_VOLATILITY]
        trend_regimes = [MarketRegime.TRENDING_UP, MarketRegime.RANGING]
        correlation_regime = MarketRegime.HIGH_CORRELATION

        combined = regime_detector._combine_regimes(
            volatility_regimes, trend_regimes, correlation_regime
        )
        # High volatility + high correlation = crisis
        assert combined == MarketRegime.HIGH_VOLATILITY

    def test_combine_regimes_low_volatility(self, regime_detector):
        """Test regime combination with low volatility."""
        volatility_regimes = [MarketRegime.LOW_VOLATILITY, MarketRegime.LOW_VOLATILITY]
        trend_regimes = [MarketRegime.TRENDING_UP, MarketRegime.RANGING]
        correlation_regime = MarketRegime.LOW_CORRELATION

        combined = regime_detector._combine_regimes(
            volatility_regimes, trend_regimes, correlation_regime
        )
        assert combined == MarketRegime.LOW_VOLATILITY

    def test_combine_regimes_medium_volatility(self, regime_detector):
        """Test regime combination with medium volatility."""
        volatility_regimes = [MarketRegime.MEDIUM_VOLATILITY, MarketRegime.MEDIUM_VOLATILITY]
        trend_regimes = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_UP]
        correlation_regime = MarketRegime.LOW_CORRELATION

        combined = regime_detector._combine_regimes(
            volatility_regimes, trend_regimes, correlation_regime
        )
        # Use trend regime for medium volatility
        assert combined == MarketRegime.TRENDING_UP

    @pytest.mark.asyncio
    async def test_regime_change_detection(self, regime_detector):
        """Test regime change detection and event creation."""
        # Simulate regime change
        old_regime = regime_detector.current_regime
        new_regime = MarketRegime.HIGH_VOLATILITY

        await regime_detector._check_regime_change(new_regime)

        # Check that regime changed
        assert regime_detector.current_regime == new_regime
        assert len(regime_detector.regime_history) == 1

        # Check event details
        event = regime_detector.regime_history[0]
        assert event.from_regime == old_regime
        assert event.to_regime == new_regime
        assert event.confidence > 0.7  # Should be above threshold
        assert isinstance(event.timestamp, datetime)

    def test_calculate_change_confidence_first_change(self, regime_detector):
        """Test confidence calculation for first regime change."""
        confidence = regime_detector._calculate_change_confidence(MarketRegime.HIGH_VOLATILITY)
        assert confidence == 0.8  # High confidence for first change

    def test_calculate_change_confidence_new_regime(self, regime_detector):
        """Test confidence calculation for new regime type."""
        # Add some history
        event = RegimeChangeEvent(
            from_regime=MarketRegime.MEDIUM_VOLATILITY,
            to_regime=MarketRegime.LOW_VOLATILITY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            trigger_metrics={},
            description="Test event",
        )
        regime_detector.regime_history.append(event)

        # Test new regime type
        confidence = regime_detector._calculate_change_confidence(MarketRegime.HIGH_VOLATILITY)
        assert confidence == 0.9  # Higher confidence for new regime type

    def test_get_regime_history(self, regime_detector):
        """Test getting regime history."""
        # Add some test events
        for i in range(5):
            event = RegimeChangeEvent(
                from_regime=MarketRegime.MEDIUM_VOLATILITY,
                to_regime=MarketRegime.LOW_VOLATILITY,
                confidence=0.8,
                timestamp=datetime.now(timezone.utc),
                trigger_metrics={},
                description=f"Test event {i}",
            )
            regime_detector.regime_history.append(event)

        # Test with limit
        history = regime_detector.get_regime_history(limit=3)
        assert len(history) == 3
        assert history[-1].description == "Test event 4"

    def test_get_current_regime(self, regime_detector):
        """Test getting current regime."""
        current = regime_detector.get_current_regime()
        assert current == MarketRegime.MEDIUM_VOLATILITY  # Default

    def test_get_regime_statistics_no_history(self, regime_detector):
        """Test getting regime statistics with no history."""
        stats = regime_detector.get_regime_statistics()

        assert stats["total_changes"] == 0
        assert stats["current_regime"] == MarketRegime.MEDIUM_VOLATILITY.value
        assert stats["regime_duration_hours"] == 0
        assert stats["last_change"] is None

    def test_get_regime_statistics_with_history(self, regime_detector):
        """Test getting regime statistics with history."""
        # Add a test event
        event = RegimeChangeEvent(
            from_regime=MarketRegime.MEDIUM_VOLATILITY,
            to_regime=MarketRegime.HIGH_VOLATILITY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            trigger_metrics={},
            description="Test event",
        )
        regime_detector.regime_history.append(event)
        regime_detector.current_regime = MarketRegime.HIGH_VOLATILITY

        stats = regime_detector.get_regime_statistics()

        assert stats["total_changes"] == 1
        assert stats["current_regime"] == MarketRegime.HIGH_VOLATILITY.value
        assert stats["regime_duration_hours"] > 0
        assert stats["last_change"] is not None

    @pytest.mark.asyncio
    async def test_error_handling_volatility_detection(self, regime_detector):
        """Test error handling in volatility regime detection."""
        # Test with invalid data - should return default value gracefully
        regime = await regime_detector.detect_volatility_regime("BTCUSDT", [])
        assert regime == MarketRegime.MEDIUM_VOLATILITY  # Default fallback

    @pytest.mark.asyncio
    async def test_error_handling_trend_detection(self, regime_detector):
        """Test error handling in trend regime detection."""
        # Test with invalid data - should return default value gracefully
        regime = await regime_detector.detect_trend_regime("BTCUSDT", [])
        assert regime == MarketRegime.RANGING  # Default fallback

    @pytest.mark.asyncio
    async def test_error_handling_correlation_detection(self, regime_detector):
        """Test error handling in correlation regime detection."""
        # Test with invalid data - should return default value gracefully
        regime = await regime_detector.detect_correlation_regime([], {})
        assert regime == MarketRegime.LOW_CORRELATION  # Default fallback

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with missing configuration
        config = {}
        detector = MarketRegimeDetector(config)

        # Should use defaults
        assert detector.volatility_window == 20
        assert detector.trend_window == 50
        assert detector.correlation_window == 30
        assert detector.regime_change_threshold == 0.7

    def test_threshold_configuration(self):
        """Test threshold configuration."""
        config = {
            "volatility_window": 30,
            "trend_window": 60,
            "correlation_window": 40,
            "regime_change_threshold": 0.8,
        }
        detector = MarketRegimeDetector(config)

        assert detector.volatility_window == 30
        assert detector.trend_window == 60
        assert detector.correlation_window == 40
        assert detector.regime_change_threshold == 0.8
