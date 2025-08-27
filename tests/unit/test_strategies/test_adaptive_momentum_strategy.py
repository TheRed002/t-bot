"""
Unit tests for AdaptiveMomentumStrategy.

This module tests the adaptive momentum strategy implementation,
including integration with existing regime detection and adaptive risk management.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import dependencies
from src.core.types import (
    MarketData,
    MarketRegime,
    OrderSide,
    Position,
    Signal,
    SignalDirection,
    StrategyType,
)
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.regime_detection import MarketRegimeDetector

# Import the strategy under test
from src.strategies.dynamic.adaptive_momentum import AdaptiveMomentumStrategy


class TestAdaptiveMomentumStrategy:
    """Test cases for AdaptiveMomentumStrategy."""

    @pytest.fixture
    def strategy_config(self):
        """Create a test configuration for the strategy."""
        return {
            "name": "adaptive_momentum",
            "strategy_id": "adaptive_momentum_001",
            "strategy_type": StrategyType.MOMENTUM,
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "position_size_pct": 0.02,
            "min_confidence": 0.6,
            "max_positions": 5,
            "parameters": {
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "momentum_lookback": 10,
                "volume_threshold": 1.5,
            },
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create a test strategy instance."""
        return AdaptiveMomentumStrategy(strategy_config)

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock regime detector."""
        detector = Mock(spec=MarketRegimeDetector)
        detector.detect_comprehensive_regime = AsyncMock(
            return_value=MarketRegime.MEDIUM_VOLATILITY
        )
        return detector

    @pytest.fixture
    def mock_adaptive_risk_manager(self):
        """Create a mock adaptive risk manager."""
        manager = Mock(spec=AdaptiveRiskManager)
        manager.calculate_adaptive_position_size = AsyncMock(return_value=Decimal("200"))
        manager.get_adaptive_parameters = Mock(
            return_value={
                "position_size_multiplier": 1.0,
                "stop_loss_multiplier": 1.0,
                "take_profit_multiplier": 1.0,
                "max_positions_multiplier": 1.0,
                "regime": "medium_volatility",
            }
        )
        return manager

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTC/USD",
            price=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def sample_signal(self):
        """Create a sample signal."""
        return Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTC/USD",
            strategy_name="adaptive_momentum",
            metadata={
                "momentum_score": 0.5,
                "volume_score": 0.7,
                "rsi_score": 0.3,
                "regime": "medium_volatility",
                "combined_score": 0.5,
            },
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "adaptive_momentum"
        assert strategy.strategy_type == StrategyType.MOMENTUM
        assert strategy.fast_ma_period == 20
        assert strategy.slow_ma_period == 50
        assert strategy.rsi_period == 14
        assert strategy.rsi_overbought == 70
        assert strategy.rsi_oversold == 30
        assert strategy.momentum_lookback == 10
        assert strategy.volume_threshold == 1.5

    def test_set_regime_detector(self, strategy, mock_regime_detector):
        """Test setting regime detector."""
        strategy.set_regime_detector(mock_regime_detector)
        assert strategy.regime_detector == mock_regime_detector

    def test_set_adaptive_risk_manager(self, strategy, mock_adaptive_risk_manager):
        """Test setting adaptive risk manager."""
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)
        assert strategy.adaptive_risk_manager == mock_adaptive_risk_manager

    @pytest.mark.asyncio
    async def test_update_price_history(self, strategy, sample_market_data):
        """Test price history update."""
        await strategy._update_price_history(sample_market_data)

        assert "BTC/USD" in strategy.price_history
        assert "BTC/USD" in strategy.volume_history
        assert len(strategy.price_history["BTC/USD"]) == 1
        assert len(strategy.volume_history["BTC/USD"]) == 1
        assert strategy.price_history["BTC/USD"][0] == 50000.0
        assert strategy.volume_history["BTC/USD"][0] == 1000.0

    @pytest.mark.asyncio
    async def test_get_current_regime_with_detector(
        self, strategy, mock_regime_detector, sample_market_data
    ):
        """Test getting current regime using existing regime detector."""
        strategy.set_regime_detector(mock_regime_detector)
        await strategy._update_price_history(sample_market_data)

        regime = await strategy._get_current_regime("BTC/USD")

        assert regime == MarketRegime.MEDIUM_VOLATILITY
        mock_regime_detector.detect_comprehensive_regime.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_regime_without_detector(self, strategy):
        """Test getting current regime without detector."""
        regime = await strategy._get_current_regime("BTC/USD")
        assert regime == MarketRegime.MEDIUM_VOLATILITY

    @pytest.mark.asyncio
    async def test_calculate_momentum_score(self, strategy):
        """Test momentum score calculation."""
        # Add price history with enough data points (need at least
        # slow_ma_period = 50)
        strategy.price_history["BTC/USD"] = [100 + i for i in range(60)]  # 60 data points

        momentum_score = await strategy._calculate_momentum_score("BTC/USD")

        assert isinstance(momentum_score, float)
        assert -1.0 <= momentum_score <= 1.0
        assert "BTC/USD" in strategy.momentum_scores

    @pytest.mark.asyncio
    async def test_calculate_momentum_score_insufficient_data(self, strategy):
        """Test momentum score calculation with insufficient data."""
        momentum_score = await strategy._calculate_momentum_score("BTC/USD")
        assert momentum_score == 0.0

    @pytest.mark.asyncio
    async def test_calculate_volume_score(self, strategy):
        """Test volume score calculation."""
        # Add volume history
        strategy.volume_history["BTC/USD"] = [1000, 1100, 1200, 1300, 1400]

        volume_score = await strategy._calculate_volume_score("BTC/USD")

        assert isinstance(volume_score, float)
        assert 0.0 <= volume_score <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_rsi_score(self, strategy):
        """Test RSI score calculation."""
        # Add price history for RSI calculation
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        strategy.price_history["BTC/USD"] = prices

        rsi_score = await strategy._calculate_rsi_score("BTC/USD")

        assert isinstance(rsi_score, float)
        assert -1.0 <= rsi_score <= 1.0

    @pytest.mark.asyncio
    async def test_generate_momentum_signals_positive_momentum(self, strategy, sample_market_data):
        """Test signal generation with positive momentum."""
        momentum_score = 0.8
        volume_score = 0.7
        rsi_score = 0.3
        regime = MarketRegime.MEDIUM_VOLATILITY

        signals = await strategy._generate_momentum_signals(
            sample_market_data, momentum_score, volume_score, rsi_score, regime
        )

        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].symbol == "BTC/USD"
        assert signals[0].strategy_name == "adaptive_momentum"
        assert "momentum_score" in signals[0].metadata
        assert "regime" in signals[0].metadata

    @pytest.mark.asyncio
    async def test_generate_momentum_signals_negative_momentum(self, strategy, sample_market_data):
        """Test signal generation with negative momentum."""
        momentum_score = -0.8
        volume_score = -0.6
        rsi_score = -0.4
        regime = MarketRegime.MEDIUM_VOLATILITY

        signals = await strategy._generate_momentum_signals(
            sample_market_data, momentum_score, volume_score, rsi_score, regime
        )

        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.SELL
        assert signals[0].symbol == "BTC/USD"

    @pytest.mark.asyncio
    async def test_generate_momentum_signals_weak_momentum(self, strategy, sample_market_data):
        """Test signal generation with weak momentum."""
        momentum_score = 0.1
        volume_score = 0.2
        rsi_score = 0.1
        regime = MarketRegime.MEDIUM_VOLATILITY

        signals = await strategy._generate_momentum_signals(
            sample_market_data, momentum_score, volume_score, rsi_score, regime
        )

        assert len(signals) == 0  # No signals for weak momentum

    @pytest.mark.asyncio
    async def test_apply_regime_confidence_adjustments_with_manager(
        self, strategy, mock_adaptive_risk_manager
    ):
        """Test regime confidence adjustments with adaptive risk manager."""
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTC/USD",
                strategy_name="adaptive_momentum",
                metadata={"regime": "medium_volatility"},
            )
        ]

        adjusted_signals = await strategy._apply_regime_confidence_adjustments(
            signals, MarketRegime.MEDIUM_VOLATILITY
        )

        assert len(adjusted_signals) == 1
        assert "regime_confidence_multiplier" in adjusted_signals[0].metadata
        assert "adaptive_params" in adjusted_signals[0].metadata

    @pytest.mark.asyncio
    async def test_apply_regime_confidence_adjustments_without_manager(self, strategy):
        """Test regime confidence adjustments without adaptive risk manager."""
        signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTC/USD",
                strategy_name="adaptive_momentum",
                metadata={"regime": "medium_volatility"},
            )
        ]

        adjusted_signals = await strategy._apply_regime_confidence_adjustments(
            signals, MarketRegime.MEDIUM_VOLATILITY
        )

        assert len(adjusted_signals) == 1
        assert adjusted_signals[0] == signals[0]  # No changes without manager

    def test_get_regime_confidence_multiplier(self, strategy):
        """Test regime confidence multiplier calculation."""
        # Test different regimes
        assert strategy._get_regime_confidence_multiplier(MarketRegime.LOW_VOLATILITY) == 1.1
        assert strategy._get_regime_confidence_multiplier(MarketRegime.MEDIUM_VOLATILITY) == 1.0
        assert strategy._get_regime_confidence_multiplier(MarketRegime.HIGH_VOLATILITY) == 0.8
        assert strategy._get_regime_confidence_multiplier(MarketRegime.HIGH_VOLATILITY) == 0.6

    @pytest.mark.asyncio
    async def test_validate_signal_success(self, strategy, sample_signal):
        """Test successful signal validation."""
        # Add momentum data
        strategy.momentum_scores["BTC/USD"] = 0.5

        is_valid = await strategy.validate_signal(sample_signal)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_signal_momentum_inconsistency(self, strategy, sample_signal):
        """Test signal validation with momentum inconsistency."""
        # Add different momentum data
        # Different from signal's 0.5
        strategy.momentum_scores["BTC/USD"] = 0.9

        is_valid = await strategy.validate_signal(sample_signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_position_size_with_adaptive_manager(
        self, strategy, mock_adaptive_risk_manager, sample_signal
    ):
        """Test position size calculation with adaptive risk manager."""
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        position_size = await strategy.get_position_size(sample_signal)

        assert position_size == Decimal("200")
        mock_adaptive_risk_manager.calculate_adaptive_position_size.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_position_size_without_adaptive_manager(self, strategy, sample_signal):
        """Test position size calculation without adaptive risk manager."""
        position_size = await strategy.get_position_size(sample_signal)

        assert position_size == Decimal("0.02")  # Base position size

    def test_should_exit_momentum_reversal(self, strategy):
        """Test exit condition for momentum reversal."""
        # Create position
        position = Position(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            timestamp=datetime.now(),
            metadata={"entry_momentum": 0.8},
        )

        # Add current momentum data (reversed)
        strategy.momentum_scores["BTC/USD"] = -0.3

        # Create market data
        data = MarketData(
            symbol="BTC/USD",
            price=Decimal("49000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
        )

        should_exit = strategy.should_exit(position, data)
        assert should_exit is True

    def test_should_exit_no_reversal(self, strategy):
        """Test exit condition when no momentum reversal."""
        # Create position
        position = Position(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            timestamp=datetime.now(),
            metadata={"entry_momentum": 0.8},
        )

        # Add current momentum data (no reversal)
        strategy.momentum_scores["BTC/USD"] = 0.6

        # Create market data
        data = MarketData(
            symbol="BTC/USD",
            price=Decimal("51000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
        )

        should_exit = strategy.should_exit(position, data)
        assert should_exit is False

    def test_get_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()

        assert "momentum_scores" in info
        assert "price_history_count" in info
        assert "volume_history_count" in info
        assert "regime_detector_available" in info
        assert "adaptive_risk_manager_available" in info

    @pytest.mark.asyncio
    async def test_generate_signals_impl_invalid_data(self, strategy):
        """Test signal generation with invalid data."""
        signals = await strategy._generate_signals_impl(None)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_impl_success(self, strategy, sample_market_data):
        """Test successful signal generation."""
        # Add price history
        strategy.price_history["BTC/USD"] = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        strategy.volume_history["BTC/USD"] = [1000, 1100, 1200, 1300, 1400]

        signals = await strategy._generate_signals_impl(sample_market_data)

        # Should generate signals based on momentum
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_impl_exception_handling(self, strategy, sample_market_data):
        """Test signal generation exception handling."""
        # Mock _update_price_history to raise exception
        with patch.object(strategy, "_update_price_history", side_effect=Exception("Test error")):
            signals = await strategy._generate_signals_impl(sample_market_data)
            assert len(signals) == 0  # Should return empty list on error

    def test_strategy_integration_with_existing_components(
        self, strategy, mock_regime_detector, mock_adaptive_risk_manager
    ):
        """Test integration with existing regime detection and adaptive risk management."""
        # Set up integration
        strategy.set_regime_detector(mock_regime_detector)
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        # Verify integration
        assert strategy.regime_detector is not None
        assert strategy.adaptive_risk_manager is not None
        assert strategy.regime_detector == mock_regime_detector
        assert strategy.adaptive_risk_manager == mock_adaptive_risk_manager

    @pytest.mark.asyncio
    async def test_comprehensive_signal_generation_workflow(
        self, strategy, mock_regime_detector, mock_adaptive_risk_manager, sample_market_data
    ):
        """Test comprehensive signal generation workflow."""
        # Set up integration
        strategy.set_regime_detector(mock_regime_detector)
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        # Add sufficient price history
        prices = [100 + i for i in range(60)]  # 60 data points
        volumes = [1000 + i * 10 for i in range(60)]
        strategy.price_history["BTC/USD"] = prices
        strategy.volume_history["BTC/USD"] = volumes

        # Generate signals
        signals = await strategy._generate_signals_impl(sample_market_data)

        # Verify signals
        assert isinstance(signals, list)
        for signal in signals:
            assert signal.symbol == "BTC/USD"
            assert signal.strategy_name == "adaptive_momentum"
            assert "momentum_score" in signal.metadata
            assert "regime" in signal.metadata
            assert signal.confidence > 0

    def test_strategy_configuration_validation(self):
        """Test strategy configuration validation."""
        # Test with valid config
        valid_config = {
            "name": "adaptive_momentum",
            "strategy_type": StrategyType.MOMENTUM,
            "symbols": ["BTC/USD"],
            "timeframe": "1h",
            "position_size_pct": 0.02,
            "min_confidence": 0.6,
            "parameters": {"fast_ma_period": 20, "slow_ma_period": 50},
        }

        strategy = AdaptiveMomentumStrategy(valid_config)
        assert strategy is not None

        # Test with missing required config
        invalid_config = {
            "name": "adaptive_momentum",
            "strategy_type": StrategyType.MOMENTUM,
            "symbols": ["BTC/USD"],
        }

        # Should still work with defaults
        strategy = AdaptiveMomentumStrategy(invalid_config)
        assert strategy is not None
        assert strategy.fast_ma_period == 20  # Default value
