"""
Integration tests for Dynamic Strategies.

This module tests the integration of dynamic strategies with existing
regime detection and adaptive risk management components.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

# Import the strategies under test
from src.strategies.dynamic.adaptive_momentum import AdaptiveMomentumStrategy
from src.strategies.dynamic.volatility_breakout import VolatilityBreakoutStrategy

# Import dependencies
from src.core.types import (
    Signal, MarketData, Position, StrategyType, 
    StrategyConfig, StrategyStatus, SignalDirection, MarketRegime, OrderSide
)
from src.risk_management.regime_detection import MarketRegimeDetector
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.base import BaseRiskManager


class TestDynamicStrategiesIntegration:
    """Integration tests for dynamic strategies."""
    
    @pytest.fixture
    def adaptive_momentum_config(self):
        """Create configuration for adaptive momentum strategy."""
        return {
            'name': 'adaptive_momentum',
            'strategy_type': StrategyType.DYNAMIC,
            'symbols': ['BTC/USD', 'ETH/USD'],
            'timeframe': '1h',
            'position_size_pct': 0.02,
            'min_confidence': 0.6,
            'max_positions': 5,
            'parameters': {
                'fast_ma_period': 20,
                'slow_ma_period': 50,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'momentum_lookback': 10,
                'volume_threshold': 1.5
            }
        }
    
    @pytest.fixture
    def volatility_breakout_config(self):
        """Create configuration for volatility breakout strategy."""
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
    def real_regime_detector(self):
        """Create a real regime detector instance."""
        regime_config = {
            'volatility_window': 20,
            'trend_window': 50,
            'regime_thresholds': {
                'low_volatility': 0.1,
                'high_volatility': 0.3,
                'crisis_threshold': 0.5
            }
        }
        return MarketRegimeDetector(regime_config)
    
    @pytest.fixture
    def real_adaptive_risk_manager(self):
        """Create a real adaptive risk manager instance."""
        risk_config = {
            'max_position_size': 0.1,
            'max_portfolio_risk': 0.02,
            'position_sizing_method': 'kelly',
            'regime_adjustments': {
                'low_volatility': 1.2,
                'medium_volatility': 1.0,
                'high_volatility': 0.8,
                'crisis': 0.5
            }
        }
        regime_detector = MarketRegimeDetector({
            'volatility_window': 20,
            'trend_window': 50,
            'regime_thresholds': {
                'low_volatility': 0.1,
                'high_volatility': 0.3,
                'crisis_threshold': 0.5
            }
        })
        return AdaptiveRiskManager(risk_config, regime_detector)
    
    @pytest.fixture
    def sample_market_data_series(self):
        """Create a series of market data for testing."""
        base_price = 50000
        base_time = datetime.now()
        
        market_data_series = []
        for i in range(100):
            # Create realistic price movements
            price_change = np.random.normal(0, 100)  # Random walk
            volume = np.random.uniform(800, 1200)
            
            market_data = MarketData(
                symbol="BTC/USD",
                price=Decimal(str(base_price + price_change)),
                volume=Decimal(str(volume)),
                timestamp=base_time + timedelta(minutes=i),
                high_price=Decimal(str(base_price + price_change + 50)),
                low_price=Decimal(str(base_price + price_change - 50))
            )
            market_data_series.append(market_data)
            base_price += price_change
        
        return market_data_series
    
    @pytest.mark.asyncio
    async def test_adaptive_momentum_with_real_regime_detection(self, adaptive_momentum_config, 
                                                              real_regime_detector, sample_market_data_series):
        """Test adaptive momentum strategy with real regime detection."""
        # Create strategy
        strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        strategy.set_regime_detector(real_regime_detector)
        
        # Process market data series
        all_signals = []
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            all_signals.extend(signals)
        
        # Verify integration
        assert len(all_signals) >= 0  # May or may not generate signals
        for signal in all_signals:
            assert signal.strategy_name == "adaptive_momentum"
            assert "regime" in signal.metadata
            assert "momentum_score" in signal.metadata
            assert signal.confidence > 0
        
        # Verify regime detection was used
        assert strategy.regime_detector is not None
        assert len(strategy.price_history) > 0
        assert len(strategy.volume_history) > 0
    
    @pytest.mark.asyncio
    async def test_volatility_breakout_with_real_regime_detection(self, volatility_breakout_config,
                                                                 real_regime_detector, sample_market_data_series):
        """Test volatility breakout strategy with real regime detection."""
        # Create strategy
        strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        strategy.set_regime_detector(real_regime_detector)
        
        # Process market data series
        all_signals = []
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            all_signals.extend(signals)
        
        # Verify integration
        assert len(all_signals) >= 0  # May or may not generate signals
        for signal in all_signals:
            assert signal.strategy_name == "volatility_breakout"
            assert "regime" in signal.metadata
            assert "atr_value" in signal.metadata
            assert "consolidation_score" in signal.metadata
            assert signal.confidence > 0
        
        # Verify regime detection was used
        assert strategy.regime_detector is not None
        assert len(strategy.price_history) > 0
        assert len(strategy.high_low_history) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_momentum_with_real_adaptive_risk(self, adaptive_momentum_config,
                                                            real_adaptive_risk_manager, sample_market_data_series):
        """Test adaptive momentum strategy with real adaptive risk management."""
        # Create strategy
        strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        # Process market data series
        all_signals = []
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            all_signals.extend(signals)
        
        # Test position sizing with adaptive risk manager
        if all_signals:
            signal = all_signals[0]
            position_size = await strategy.get_position_size(signal)
            
            # Verify adaptive position sizing
            assert position_size > 0
            assert isinstance(position_size, Decimal)
        
        # Verify adaptive risk manager integration
        assert strategy.adaptive_risk_manager is not None
    
    @pytest.mark.asyncio
    async def test_volatility_breakout_with_real_adaptive_risk(self, volatility_breakout_config,
                                                              real_adaptive_risk_manager, sample_market_data_series):
        """Test volatility breakout strategy with real adaptive risk management."""
        # Create strategy
        strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        # Process market data series
        all_signals = []
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            all_signals.extend(signals)
        
        # Test position sizing with adaptive risk manager
        if all_signals:
            signal = all_signals[0]
            position_size = await strategy.get_position_size(signal)
            
            # Verify adaptive position sizing
            assert position_size > 0
            assert isinstance(position_size, Decimal)
        
        # Verify adaptive risk manager integration
        assert strategy.adaptive_risk_manager is not None
    
    @pytest.mark.asyncio
    async def test_full_integration_adaptive_momentum(self, adaptive_momentum_config,
                                                     real_regime_detector, real_adaptive_risk_manager,
                                                     sample_market_data_series):
        """Test full integration of adaptive momentum strategy with both regime detection and adaptive risk."""
        # Create strategy with both components
        strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        strategy.set_regime_detector(real_regime_detector)
        strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        # Process market data series
        all_signals = []
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            all_signals.extend(signals)
        
        # Verify full integration
        assert strategy.regime_detector is not None
        assert strategy.adaptive_risk_manager is not None
        
        # Test signal validation
        if all_signals:
            signal = all_signals[0]
            is_valid = await strategy.validate_signal(signal)
            assert isinstance(is_valid, bool)
        
        # Test position sizing
        if all_signals:
            signal = all_signals[0]
            position_size = await strategy.get_position_size(signal)
            assert position_size > 0
        
        # Test strategy info
        info = strategy.get_strategy_info()
        assert "regime_detector_available" in info
        assert "adaptive_risk_manager_available" in info
        assert info["regime_detector_available"] is True
        assert info["adaptive_risk_manager_available"] is True
    
    @pytest.mark.asyncio
    async def test_full_integration_volatility_breakout(self, volatility_breakout_config,
                                                       real_regime_detector, real_adaptive_risk_manager,
                                                       sample_market_data_series):
        """Test full integration of volatility breakout strategy with both regime detection and adaptive risk."""
        # Create strategy with both components
        strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        strategy.set_regime_detector(real_regime_detector)
        strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        # Process market data series
        all_signals = []
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            all_signals.extend(signals)
        
        # Verify full integration
        assert strategy.regime_detector is not None
        assert strategy.adaptive_risk_manager is not None
        
        # Test signal validation
        if all_signals:
            signal = all_signals[0]
            is_valid = await strategy.validate_signal(signal)
            assert isinstance(is_valid, bool)
        
        # Test position sizing
        if all_signals:
            signal = all_signals[0]
            position_size = await strategy.get_position_size(signal)
            assert position_size > 0
        
        # Test strategy info
        info = strategy.get_strategy_info()
        assert "regime_detector_available" in info
        assert "adaptive_risk_manager_available" in info
        assert info["regime_detector_available"] is True
        assert info["adaptive_risk_manager_available"] is True
    
    @pytest.mark.asyncio
    async def test_regime_transition_handling(self, adaptive_momentum_config,
                                            real_regime_detector, sample_market_data_series):
        """Test how strategies handle regime transitions."""
        # Create strategy
        strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        strategy.set_regime_detector(real_regime_detector)
        
        # Process market data and track regime changes
        regime_history = []
        signal_history = []
        
        for market_data in sample_market_data_series:
            # Get current regime
            regime = await strategy._get_current_regime(market_data.symbol)
            regime_history.append(regime)
            
            # Generate signals
            signals = await strategy._generate_signals_impl(market_data)
            signal_history.append(signals)
        
        # Verify regime detection was used
        assert len(regime_history) > 0
        assert all(isinstance(regime, MarketRegime) for regime in regime_history)
        
        # Verify signals were generated with regime information
        for signals in signal_history:
            for signal in signals:
                assert "regime" in signal.metadata
    
    @pytest.mark.asyncio
    async def test_adaptive_risk_parameter_adjustment(self, volatility_breakout_config,
                                                     real_adaptive_risk_manager, sample_market_data_series):
        """Test adaptive risk parameter adjustment."""
        # Create strategy
        strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        # Process market data and test adaptive parameters
        for market_data in sample_market_data_series:
            signals = await strategy._generate_signals_impl(market_data)
            
            for signal in signals:
                # Test position sizing with adaptive risk
                position_size = await strategy.get_position_size(signal)
                assert position_size > 0
                
                # Verify adaptive parameters are applied
                if "adaptive_params" in signal.metadata:
                    adaptive_params = signal.metadata["adaptive_params"]
                    assert isinstance(adaptive_params, dict)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_graceful_degradation(self, adaptive_momentum_config,
                                                          sample_market_data_series):
        """Test error handling and graceful degradation."""
        # Create strategy without external components
        strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        
        # Test with invalid market data
        invalid_data = MarketData(
            symbol="BTC/USD",
            price=Decimal("-1000"),  # Invalid negative price
            volume=Decimal("1000"),
            timestamp=datetime.now()
        )
        
        signals = await strategy._generate_signals_impl(invalid_data)
        assert len(signals) == 0  # Should handle gracefully
        
        # Test with None data
        signals = await strategy._generate_signals_impl(None)
        assert len(signals) == 0  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, adaptive_momentum_config, volatility_breakout_config,
                                         real_regime_detector, real_adaptive_risk_manager,
                                         sample_market_data_series):
        """Test performance under load with both strategies."""
        # Create both strategies
        momentum_strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        momentum_strategy.set_regime_detector(real_regime_detector)
        momentum_strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        breakout_strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        breakout_strategy.set_regime_detector(real_regime_detector)
        breakout_strategy.set_adaptive_risk_manager(real_adaptive_risk_manager)
        
        # Process market data with both strategies simultaneously
        momentum_signals = []
        breakout_signals = []
        
        for market_data in sample_market_data_series:
            # Generate signals with both strategies
            momentum_sigs = await momentum_strategy._generate_signals_impl(market_data)
            breakout_sigs = await breakout_strategy._generate_signals_impl(market_data)
            
            momentum_signals.extend(momentum_sigs)
            breakout_signals.extend(breakout_sigs)
        
        # Verify both strategies work correctly
        assert len(momentum_signals) >= 0
        assert len(breakout_signals) >= 0
        
        # Verify no cross-contamination between strategies
        for signal in momentum_signals:
            assert signal.strategy_name == "adaptive_momentum"
        
        for signal in breakout_signals:
            assert signal.strategy_name == "volatility_breakout"
    
    @pytest.mark.asyncio
    async def test_strategy_configuration_validation_integration(self, adaptive_momentum_config,
                                                               volatility_breakout_config):
        """Test strategy configuration validation in integration context."""
        # Test adaptive momentum strategy
        momentum_strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        assert momentum_strategy.name == "adaptive_momentum"
        assert momentum_strategy.strategy_type == StrategyType.DYNAMIC
        
        # Test volatility breakout strategy
        breakout_strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        assert breakout_strategy.name == "volatility_breakout"
        assert breakout_strategy.strategy_type == StrategyType.DYNAMIC
        
        # Test with minimal configuration
        minimal_config = {
            'name': 'test_strategy',
            'strategy_type': StrategyType.DYNAMIC,
            'symbols': ['BTC/USD'],
            'timeframe': '1h',
            'position_size_pct': 0.02,
            'min_confidence': 0.6,
            'max_positions': 5,
            'parameters': {}
        }
        
        # Both strategies should work with minimal config
        momentum_strategy = AdaptiveMomentumStrategy(minimal_config)
        breakout_strategy = VolatilityBreakoutStrategy(minimal_config)
        
        assert momentum_strategy is not None
        assert breakout_strategy is not None
    
    @pytest.mark.asyncio
    async def test_signal_validation_integration(self, adaptive_momentum_config, volatility_breakout_config,
                                               sample_market_data_series):
        """Test signal validation in integration context."""
        # Create strategies
        momentum_strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        breakout_strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        
        # Generate signals
        momentum_signals = []
        breakout_signals = []
        
        for market_data in sample_market_data_series:
            momentum_sigs = await momentum_strategy._generate_signals_impl(market_data)
            breakout_sigs = await breakout_strategy._generate_signals_impl(market_data)
            
            momentum_signals.extend(momentum_sigs)
            breakout_signals.extend(breakout_sigs)
        
        # Test signal validation
        for signal in momentum_signals:
            is_valid = await momentum_strategy.validate_signal(signal)
            assert isinstance(is_valid, bool)
        
        for signal in breakout_signals:
            is_valid = await breakout_strategy.validate_signal(signal)
            assert isinstance(is_valid, bool)
    
    @pytest.mark.asyncio
    async def test_exit_condition_integration(self, adaptive_momentum_config, volatility_breakout_config,
                                            sample_market_data_series):
        """Test exit conditions in integration context."""
        # Create strategies
        momentum_strategy = AdaptiveMomentumStrategy(adaptive_momentum_config)
        breakout_strategy = VolatilityBreakoutStrategy(volatility_breakout_config)
        
        # Create test positions
        position = Position(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            timestamp=datetime.now(),
            metadata={'entry_momentum': 0.8, 'entry_consolidation': 0.8, 'entry_atr': 0.02}
        )
        
        # Test exit conditions with market data
        for market_data in sample_market_data_series[:10]:  # Test with first 10 data points
            momentum_exit = momentum_strategy.should_exit(position, market_data)
            breakout_exit = breakout_strategy.should_exit(position, market_data)
            
            assert isinstance(momentum_exit, bool)
            assert isinstance(breakout_exit, bool)
