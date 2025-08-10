"""
Unit tests for VolatilityBreakoutStrategy.

This module tests the volatility breakout strategy implementation,
including integration with existing regime detection and adaptive risk management.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Import the strategy under test
from src.strategies.dynamic.volatility_breakout import VolatilityBreakoutStrategy

# Import dependencies
from src.core.types import (
    Signal, MarketData, Position, StrategyType,
    StrategyConfig, StrategyStatus, SignalDirection, MarketRegime, OrderSide
)
from src.core.exceptions import ValidationError
from src.risk_management.regime_detection import MarketRegimeDetector
from src.risk_management.adaptive_risk import AdaptiveRiskManager


class TestVolatilityBreakoutStrategy:
    """Test cases for VolatilityBreakoutStrategy."""

    @pytest.fixture
    def strategy_config(self):
        """Create a test configuration for the strategy."""
        return {
            'name': 'volatility_breakout',
            'strategy_type': StrategyType.DYNAMIC,
            'symbols': ['BTC/USD', 'ETH/USD'],
            'timeframe': '1h',
            'position_size_pct': 0.02,
            'min_confidence': 0.6,
            'max_positions': 5,
            'parameters': {
                'atr_period': 14,
                'breakout_multiplier': 2.0,
                'consolidation_period': 20,
                'volume_confirmation': True,
                'min_consolidation_ratio': 0.8,
                'max_consolidation_ratio': 1.2,
                'time_decay_factor': 0.95
            }
        }

    @pytest.fixture
    def strategy(self, strategy_config):
        """Create a test strategy instance."""
        return VolatilityBreakoutStrategy(strategy_config)

    @pytest.fixture
    def mock_regime_detector(self):
        """Create a mock regime detector."""
        detector = Mock(spec=MarketRegimeDetector)
        detector.detect_comprehensive_regime = AsyncMock(
            return_value=MarketRegime.MEDIUM_VOLATILITY)
        return detector

    @pytest.fixture
    def mock_adaptive_risk_manager(self):
        """Create a mock adaptive risk manager."""
        manager = Mock(spec=AdaptiveRiskManager)
        manager.calculate_adaptive_position_size = AsyncMock(
            return_value=Decimal("200"))
        manager.get_adaptive_parameters = Mock(return_value={
            'position_size_multiplier': 1.0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.0,
            'max_positions_multiplier': 1.0,
            'regime': 'medium_volatility'
        })
        return manager

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTC/USD",
            price=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
            high_price=Decimal("51000"),
            low_price=Decimal("49000")
        )

    @pytest.fixture
    def sample_signal(self):
        """Create a sample signal."""
        return Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTC/USD",
            strategy_name="volatility_breakout",
            metadata={
                'atr_value': 0.02,
                'consolidation_score': 0.7,
                'breakout_level': 52000.0,
                'current_price': 52500.0,
                'regime': 'medium_volatility',
                'breakout_type': 'upper'
            }
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "volatility_breakout"
        assert strategy.strategy_type == StrategyType.DYNAMIC
        assert strategy.atr_period == 14
        assert strategy.breakout_multiplier == 2.0
        assert strategy.consolidation_period == 20
        assert strategy.volume_confirmation is True
        assert strategy.min_consolidation_ratio == 0.8
        assert strategy.max_consolidation_ratio == 1.2
        assert strategy.time_decay_factor == 0.95

    def test_set_regime_detector(self, strategy, mock_regime_detector):
        """Test setting regime detector."""
        strategy.set_regime_detector(mock_regime_detector)
        assert strategy.regime_detector == mock_regime_detector

    def test_set_adaptive_risk_manager(
            self, strategy, mock_adaptive_risk_manager):
        """Test setting adaptive risk manager."""
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)
        assert strategy.adaptive_risk_manager == mock_adaptive_risk_manager

    @pytest.mark.asyncio
    async def test_update_price_history(self, strategy, sample_market_data):
        """Test price history update."""
        await strategy._update_price_history(sample_market_data)

        assert "BTC/USD" in strategy.price_history
        assert "BTC/USD" in strategy.high_low_history
        assert len(strategy.price_history["BTC/USD"]) == 1
        assert len(strategy.high_low_history["BTC/USD"]) == 1
        assert strategy.price_history["BTC/USD"][0] == 50000.0
        assert strategy.high_low_history["BTC/USD"][0] == (51000.0, 49000.0)

    @pytest.mark.asyncio
    async def test_get_current_regime_with_detector(
            self, strategy, mock_regime_detector, sample_market_data):
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
    async def test_calculate_atr(self, strategy):
        """Test ATR calculation."""
        # Add price and high-low history
        strategy.price_history["BTC/USD"] = [100,
                                             101,
                                             102,
                                             103,
                                             104,
                                             105,
                                             106,
                                             107,
                                             108,
                                             109,
                                             110,
                                             111,
                                             112,
                                             113,
                                             114]
        strategy.high_low_history["BTC/USD"] = [(101, 99), (102, 100), (103, 101), (104, 102), (105, 103),
                                                (106, 104), (107, 105), (108, 106), (109, 107), (110, 108),
                                                (111, 109), (112, 110), (113, 111), (114, 112), (115, 113)]

        atr_value = await strategy._calculate_atr("BTC/USD")

        assert isinstance(atr_value, float)
        assert atr_value > 0
        assert "BTC/USD" in strategy.atr_values

    @pytest.mark.asyncio
    async def test_calculate_atr_insufficient_data(self, strategy):
        """Test ATR calculation with insufficient data."""
        atr_value = await strategy._calculate_atr("BTC/USD")
        assert atr_value == 0.0

    @pytest.mark.asyncio
    async def test_calculate_consolidation_score(self, strategy):
        """Test consolidation score calculation."""
        # Add price history for consolidation calculation
        prices = [100 + i * 0.1 for i in range(25)]  # Small price movements
        strategy.price_history["BTC/USD"] = prices

        consolidation_score = await strategy._calculate_consolidation_score("BTC/USD")

        assert isinstance(consolidation_score, float)
        assert 0.0 <= consolidation_score <= 1.0
        assert "BTC/USD" in strategy.consolidation_scores

    @pytest.mark.asyncio
    async def test_calculate_consolidation_score_insufficient_data(
            self, strategy):
        """Test consolidation score calculation with insufficient data."""
        consolidation_score = await strategy._calculate_consolidation_score("BTC/USD")
        assert consolidation_score == 0.0

    @pytest.mark.asyncio
    async def test_calculate_breakout_levels(self, strategy):
        """Test breakout levels calculation."""
        # Add price history
        strategy.price_history["BTC/USD"] = [100,
                                             101,
                                             102,
                                             103,
                                             104,
                                             105,
                                             106,
                                             107,
                                             108,
                                             109,
                                             110,
                                             111,
                                             112,
                                             113,
                                             114]
        strategy.high_low_history["BTC/USD"] = [(101, 99), (102, 100), (103, 101), (104, 102), (105, 103),
                                                (106, 104), (107, 105), (108, 106), (109, 107), (110, 108),
                                                (111, 109), (112, 110), (113, 111), (114, 112), (115, 113)]

        atr_value = 2.0
        regime = MarketRegime.MEDIUM_VOLATILITY

        breakout_levels = await strategy._calculate_breakout_levels("BTC/USD", atr_value, regime)

        assert isinstance(breakout_levels, dict)
        assert 'upper_breakout' in breakout_levels
        assert 'lower_breakout' in breakout_levels
        assert 'atr_value' in breakout_levels
        assert 'regime_adjustment' in breakout_levels
        assert "BTC/USD" in strategy.breakout_levels

    @pytest.mark.asyncio
    async def test_generate_breakout_signals_upper_breakout(
            self, strategy, sample_market_data):
        """Test signal generation with upper breakout."""
        atr_value = 2.0
        consolidation_score = 0.8
        breakout_levels = {
            'upper_breakout': 52000.0,
            'lower_breakout': 48000.0,
            'atr_value': atr_value,
            'regime_adjustment': 1.0
        }
        regime = MarketRegime.MEDIUM_VOLATILITY

        # Set price above upper breakout
        sample_market_data.price = Decimal("52500")

        signals = await strategy._generate_breakout_signals(
            sample_market_data, atr_value, consolidation_score, breakout_levels, regime
        )

        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.BUY
        assert signals[0].symbol == "BTC/USD"
        assert signals[0].strategy_name == "volatility_breakout"
        assert "atr_value" in signals[0].metadata
        assert "consolidation_score" in signals[0].metadata
        assert "regime" in signals[0].metadata
        assert signals[0].metadata['breakout_type'] == 'upper'

    @pytest.mark.asyncio
    async def test_generate_breakout_signals_lower_breakout(
            self, strategy, sample_market_data):
        """Test signal generation with lower breakout."""
        atr_value = 2.0
        consolidation_score = 0.8
        breakout_levels = {
            'upper_breakout': 52000.0,
            'lower_breakout': 48000.0,
            'atr_value': atr_value,
            'regime_adjustment': 1.0
        }
        regime = MarketRegime.MEDIUM_VOLATILITY

        # Set price below lower breakout
        sample_market_data.price = Decimal("47500")

        signals = await strategy._generate_breakout_signals(
            sample_market_data, atr_value, consolidation_score, breakout_levels, regime
        )

        assert len(signals) == 1
        assert signals[0].direction == SignalDirection.SELL
        assert signals[0].symbol == "BTC/USD"
        assert signals[0].strategy_name == "volatility_breakout"
        assert signals[0].metadata['breakout_type'] == 'lower'

    @pytest.mark.asyncio
    async def test_generate_breakout_signals_no_breakout(
            self, strategy, sample_market_data):
        """Test signal generation with no breakout."""
        atr_value = 2.0
        consolidation_score = 0.3  # Low consolidation score
        breakout_levels = {
            'upper_breakout': 52000.0,
            'lower_breakout': 48000.0,
            'atr_value': atr_value,
            'regime_adjustment': 1.0
        }
        regime = MarketRegime.MEDIUM_VOLATILITY

        signals = await strategy._generate_breakout_signals(
            sample_market_data, atr_value, consolidation_score, breakout_levels, regime
        )

        assert len(signals) == 0  # No signals for low consolidation

    @pytest.mark.asyncio
    async def test_apply_regime_filtering_with_manager(
            self, strategy, mock_adaptive_risk_manager):
        """Test regime filtering with adaptive risk manager."""
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTC/USD",
                strategy_name="volatility_breakout",
                metadata={
                    'atr_value': 0.02,
                    'consolidation_score': 0.7,
                    'regime': 'medium_volatility'})]

        filtered_signals = await strategy._apply_regime_filtering(signals, MarketRegime.MEDIUM_VOLATILITY)

        assert len(filtered_signals) == 1
        assert "regime_confidence_multiplier" in filtered_signals[0].metadata
        assert "adaptive_params" in filtered_signals[0].metadata

    @pytest.mark.asyncio
    async def test_apply_regime_filtering_without_manager(self, strategy):
        """Test regime filtering without adaptive risk manager."""
        signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTC/USD",
                strategy_name="volatility_breakout",
                metadata={
                    'atr_value': 0.02,
                    'consolidation_score': 0.7,
                    'regime': 'medium_volatility'})]

        filtered_signals = await strategy._apply_regime_filtering(signals, MarketRegime.MEDIUM_VOLATILITY)

        assert len(filtered_signals) == 1
        assert filtered_signals[0] == signals[0]  # No changes without manager

    def test_get_regime_breakout_adjustment(self, strategy):
        """Test regime breakout adjustment calculation."""
        # Test different regimes
        assert strategy._get_regime_breakout_adjustment(
            MarketRegime.LOW_VOLATILITY) == 1.5
        assert strategy._get_regime_breakout_adjustment(
            MarketRegime.MEDIUM_VOLATILITY) == 1.0
        assert strategy._get_regime_breakout_adjustment(
            MarketRegime.HIGH_VOLATILITY) == 0.7
        assert strategy._get_regime_breakout_adjustment(
            MarketRegime.CRISIS) == 0.5

    def test_get_regime_confidence_multiplier(self, strategy):
        """Test regime confidence multiplier calculation."""
        # Test different regimes
        assert strategy._get_regime_confidence_multiplier(
            MarketRegime.LOW_VOLATILITY) == 1.1
        assert strategy._get_regime_confidence_multiplier(
            MarketRegime.MEDIUM_VOLATILITY) == 1.0
        assert strategy._get_regime_confidence_multiplier(
            MarketRegime.HIGH_VOLATILITY) == 0.8
        assert strategy._get_regime_confidence_multiplier(
            MarketRegime.CRISIS) == 0.6

    def test_is_signal_valid_for_regime(self, strategy):
        """Test regime-specific signal validation."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTC/USD",
            strategy_name="volatility_breakout",
            metadata={'atr_value': 0.02, 'consolidation_score': 0.7}
        )

        # Test different regimes
        assert strategy._is_signal_valid_for_regime(
            signal, MarketRegime.LOW_VOLATILITY) is False  # Requires 0.7
        assert strategy._is_signal_valid_for_regime(
            signal, MarketRegime.MEDIUM_VOLATILITY) is True   # Requires 0.5
        assert strategy._is_signal_valid_for_regime(
            signal, MarketRegime.HIGH_VOLATILITY) is True   # Requires 0.3

    @pytest.mark.asyncio
    async def test_apply_time_decay(self, strategy):
        """Test time decay application."""
        signals = [
            Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="BTC/USD",
                strategy_name="volatility_breakout",
                metadata={}
            )
        ]

        # Set last breakout time to 1 hour ago
        strategy.last_breakout_time["BTC/USD"] = datetime.now() - \
            timedelta(hours=1)

        decayed_signals = await strategy._apply_time_decay(signals, "BTC/USD")

        assert len(decayed_signals) == 1
        # Should be reduced
        assert decayed_signals[0].confidence < signals[0].confidence

    @pytest.mark.asyncio
    async def test_validate_signal_success(self, strategy, sample_signal):
        """Test successful signal validation."""
        # Add ATR and consolidation data
        strategy.atr_values["BTC/USD"] = [0.02]
        strategy.consolidation_scores["BTC/USD"] = 0.7

        is_valid = await strategy.validate_signal(sample_signal)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_signal_atr_inconsistency(
            self, strategy, sample_signal):
        """Test signal validation with ATR inconsistency."""
        # Add different ATR data
        strategy.atr_values["BTC/USD"] = [0.05]  # Different from signal's 0.02

        is_valid = await strategy.validate_signal(sample_signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_signal_consolidation_inconsistency(
            self, strategy, sample_signal):
        """Test signal validation with consolidation inconsistency."""
        # Add different consolidation data
        # Different from signal's 0.7
        strategy.consolidation_scores["BTC/USD"] = 0.3

        is_valid = await strategy.validate_signal(sample_signal)
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_get_position_size_with_adaptive_manager(
            self, strategy, mock_adaptive_risk_manager, sample_signal):
        """Test position size calculation with adaptive risk manager."""
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        position_size = await strategy.get_position_size(sample_signal)

        assert position_size == Decimal("200")
        mock_adaptive_risk_manager.calculate_adaptive_position_size.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_position_size_without_adaptive_manager(
            self, strategy, sample_signal):
        """Test position size calculation without adaptive risk manager."""
        position_size = await strategy.get_position_size(sample_signal)

        assert position_size == Decimal("0.02")  # Base position size

    def test_should_exit_consolidation_breakdown(self, strategy):
        """Test exit condition for consolidation breakdown."""
        # Create position
        position = Position(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            timestamp=datetime.now(),
            metadata={'entry_consolidation': 0.8}
        )

        # Add current consolidation data (breakdown)
        strategy.consolidation_scores["BTC/USD"] = 0.3

        # Create market data
        data = MarketData(
            symbol="BTC/USD",
            price=Decimal("49000"),
            volume=Decimal("1000"),
            timestamp=datetime.now()
        )

        should_exit = strategy.should_exit(position, data)
        assert should_exit is True

    def test_should_exit_atr_expansion(self, strategy):
        """Test exit condition for ATR expansion."""
        # Create position
        position = Position(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            timestamp=datetime.now(),
            metadata={'entry_atr': 0.02}
        )

        # Add current ATR data (expansion)
        strategy.atr_values["BTC/USD"] = [0.05]  # 2.5x expansion

        # Create market data
        data = MarketData(
            symbol="BTC/USD",
            price=Decimal("51000"),
            volume=Decimal("1000"),
            timestamp=datetime.now()
        )

        should_exit = strategy.should_exit(position, data)
        assert should_exit is True

    def test_should_exit_no_conditions(self, strategy):
        """Test exit condition when no exit conditions are met."""
        # Create position
        position = Position(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            timestamp=datetime.now(),
            metadata={'entry_consolidation': 0.8, 'entry_atr': 0.02}
        )

        # Add current data (no breakdown or expansion)
        strategy.consolidation_scores["BTC/USD"] = 0.7
        strategy.atr_values["BTC/USD"] = [0.02]

        # Create market data
        data = MarketData(
            symbol="BTC/USD",
            price=Decimal("51000"),
            volume=Decimal("1000"),
            timestamp=datetime.now()
        )

        should_exit = strategy.should_exit(position, data)
        assert should_exit is False

    def test_get_strategy_info(self, strategy):
        """Test strategy information retrieval."""
        info = strategy.get_strategy_info()

        assert "atr_values_count" in info
        assert "consolidation_scores" in info
        assert "breakout_levels_count" in info
        assert "price_history_count" in info
        assert "regime_detector_available" in info
        assert "adaptive_risk_manager_available" in info

    @pytest.mark.asyncio
    async def test_generate_signals_impl_invalid_data(self, strategy):
        """Test signal generation with invalid data."""
        signals = await strategy._generate_signals_impl(None)
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_impl_success(
            self, strategy, sample_market_data):
        """Test successful signal generation."""
        # Add price history
        strategy.price_history["BTC/USD"] = [100,
                                             101,
                                             102,
                                             103,
                                             104,
                                             105,
                                             106,
                                             107,
                                             108,
                                             109,
                                             110,
                                             111,
                                             112,
                                             113,
                                             114]
        strategy.high_low_history["BTC/USD"] = [(101, 99), (102, 100), (103, 101), (104, 102), (105, 103),
                                                (106, 104), (107, 105), (108, 106), (109, 107), (110, 108),
                                                (111, 109), (112, 110), (113, 111), (114, 112), (115, 113)]

        signals = await strategy._generate_signals_impl(sample_market_data)

        # Should generate signals based on volatility breakout
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_generate_signals_impl_exception_handling(
            self, strategy, sample_market_data):
        """Test signal generation exception handling."""
        # Mock _update_price_history to raise exception
        with patch.object(strategy, '_update_price_history', side_effect=Exception("Test error")):
            signals = await strategy._generate_signals_impl(sample_market_data)
            assert len(signals) == 0  # Should return empty list on error

    def test_strategy_integration_with_existing_components(
            self, strategy, mock_regime_detector, mock_adaptive_risk_manager):
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
            self,
            strategy,
            mock_regime_detector,
            mock_adaptive_risk_manager,
            sample_market_data):
        """Test comprehensive signal generation workflow."""
        # Set up integration
        strategy.set_regime_detector(mock_regime_detector)
        strategy.set_adaptive_risk_manager(mock_adaptive_risk_manager)

        # Add sufficient price history
        prices = [100 + i for i in range(60)]  # 60 data points
        high_lows = [(p + 1, p - 1) for p in prices]
        strategy.price_history["BTC/USD"] = prices
        strategy.high_low_history["BTC/USD"] = high_lows

        # Generate signals
        signals = await strategy._generate_signals_impl(sample_market_data)

        # Verify signals
        assert isinstance(signals, list)
        for signal in signals:
            assert signal.symbol == "BTC/USD"
            assert signal.strategy_name == "volatility_breakout"
            assert "atr_value" in signal.metadata
            assert "consolidation_score" in signal.metadata
            assert "regime" in signal.metadata
            assert signal.confidence > 0

    def test_strategy_configuration_validation(self):
        """Test strategy configuration validation."""
        # Test with valid config
        valid_config = {
            'name': 'volatility_breakout',
            'strategy_type': StrategyType.DYNAMIC,
            'symbols': ['BTC/USD'],
            'timeframe': '1h',
            'position_size_pct': 0.02,
            'min_confidence': 0.6,
            'parameters': {
                'atr_period': 14,
                'breakout_multiplier': 2.0
            }
        }

        strategy = VolatilityBreakoutStrategy(valid_config)
        assert strategy is not None

        # Test with missing required config
        invalid_config = {
            'name': 'volatility_breakout',
            'strategy_type': StrategyType.DYNAMIC,
            'symbols': ['BTC/USD']
        }

        # Should still work with defaults
        strategy = VolatilityBreakoutStrategy(invalid_config)
        assert strategy is not None
        assert strategy.atr_period == 14  # Default value
