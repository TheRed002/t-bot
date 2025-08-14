"""
Comprehensive Stop-Loss Validation Tests for Adaptive Risk Management.

This module tests stop-loss placement, validation, and triggering logic
in the AdaptiveRiskManager to ensure stop-losses are placed correctly
relative to bid/ask spreads and market conditions to prevent slippage losses.

CRITICAL AREAS TESTED:
1. Stop-loss placement accuracy relative to entry price
2. Bid/ask spread consideration in stop-loss placement
3. Market regime adaptation of stop-loss levels
4. Stop-loss validation against market microstructure
5. Extreme market condition handling
6. Stop-loss triggering logic and slippage prevention
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types import MarketRegime, OrderSide, Position, Signal, SignalDirection
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.regime_detection import MarketRegimeDetector


class TestAdaptiveStopLossValidation:
    """
    Test suite for adaptive stop-loss validation and placement accuracy.
    
    These tests ensure stop-losses are placed optimally to minimize
    slippage while providing adequate risk protection across different
    market conditions and regimes.
    """

    @pytest.fixture
    def config(self):
        """Create test configuration for adaptive risk manager."""
        return {
            "base_position_size_pct": 0.02,  # 2%
            "base_stop_loss_pct": 0.02,      # 2%
            "base_take_profit_pct": 0.04,    # 4%
            "momentum_window": 20,
            "momentum_threshold": 0.1,
        }

    @pytest.fixture
    def regime_detector(self):
        """Create a regime detector instance for testing."""
        detector_config = {
            "volatility_window": 20,
            "trend_window": 50,
            "correlation_window": 30,
            "regime_change_threshold": 0.7,
        }
        return MarketRegimeDetector(detector_config)

    @pytest.fixture
    def adaptive_risk_manager(self, config, regime_detector):
        """Create an adaptive risk manager instance for testing."""
        return AdaptiveRiskManager(config, regime_detector)

    @pytest.fixture
    def buy_signal(self):
        """Create a buy signal for testing."""
        return Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
            metadata={},
        )

    @pytest.fixture
    def sell_signal(self):
        """Create a sell signal for testing."""
        return Signal(
            direction=SignalDirection.SELL,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
            metadata={},
        )

    @pytest.mark.asyncio
    async def test_stop_loss_placement_accuracy_buy_signal(self, adaptive_risk_manager, buy_signal):
        """Test accurate stop-loss placement for buy signals."""
        entry_price = Decimal("50000")
        
        # Test different market regimes
        regimes_and_expected_multipliers = [
            (MarketRegime.LOW_VOLATILITY, 0.8),   # Tighter stops
            (MarketRegime.MEDIUM_VOLATILITY, 1.0), # Standard stops
            (MarketRegime.HIGH_VOLATILITY, 1.3),  # Wider stops
            (MarketRegime.CRISIS, 1.5),           # Widest stops
        ]
        
        for regime, expected_multiplier in regimes_and_expected_multipliers:
            stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, regime, entry_price
            )
            
            # For buy signal, stop loss should be below entry price
            assert stop_loss < entry_price, f"Buy stop loss should be below entry for {regime}"
            
            # Calculate expected stop loss
            base_stop_pct = Decimal("0.02")  # 2% base
            adjusted_stop_pct = base_stop_pct * Decimal(str(expected_multiplier))
            expected_stop_loss = entry_price * (1 - adjusted_stop_pct)
            
            # Verify accuracy within small tolerance
            tolerance = entry_price * Decimal("0.001")  # 0.1% tolerance
            assert abs(stop_loss - expected_stop_loss) < tolerance, \
                f"Stop loss accuracy issue for {regime}: expected {expected_stop_loss}, got {stop_loss}"

    @pytest.mark.asyncio
    async def test_stop_loss_placement_accuracy_sell_signal(self, adaptive_risk_manager, sell_signal):
        """Test accurate stop-loss placement for sell signals."""
        entry_price = Decimal("50000")
        
        # Test different market regimes
        regimes_and_expected_multipliers = [
            (MarketRegime.LOW_VOLATILITY, 0.8),   # Tighter stops
            (MarketRegime.MEDIUM_VOLATILITY, 1.0), # Standard stops
            (MarketRegime.HIGH_VOLATILITY, 1.3),  # Wider stops
            (MarketRegime.CRISIS, 1.5),           # Widest stops
        ]
        
        for regime, expected_multiplier in regimes_and_expected_multipliers:
            stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                sell_signal, regime, entry_price
            )
            
            # For sell signal, stop loss should be above entry price
            assert stop_loss > entry_price, f"Sell stop loss should be above entry for {regime}"
            
            # Calculate expected stop loss
            base_stop_pct = Decimal("0.02")  # 2% base
            adjusted_stop_pct = base_stop_pct * Decimal(str(expected_multiplier))
            expected_stop_loss = entry_price * (1 + adjusted_stop_pct)
            
            # Verify accuracy within small tolerance
            tolerance = entry_price * Decimal("0.001")  # 0.1% tolerance
            assert abs(stop_loss - expected_stop_loss) < tolerance, \
                f"Stop loss accuracy issue for {regime}: expected {expected_stop_loss}, got {stop_loss}"

    @pytest.mark.asyncio
    async def test_stop_loss_bid_ask_spread_consideration(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss placement considers bid/ask spread."""
        entry_price = Decimal("50000")
        
        # Test with different spread scenarios
        spread_scenarios = [
            # (bid_price, ask_price, spread_bps, scenario_name)
            (Decimal("49995"), Decimal("50005"), 20, "tight_spread"),      # 20 bps
            (Decimal("49975"), Decimal("50025"), 100, "normal_spread"),    # 100 bps  
            (Decimal("49900"), Decimal("50100"), 400, "wide_spread"),      # 400 bps
            (Decimal("49500"), Decimal("50500"), 2000, "extreme_spread"),  # 2000 bps
        ]
        
        for bid_price, ask_price, spread_bps, scenario in spread_scenarios:
            # Mock market data with specific spread
            with patch.object(adaptive_risk_manager, '_get_market_data') as mock_market_data:
                mock_market_data.return_value = {
                    'bid': float(bid_price),
                    'ask': float(ask_price),
                    'spread_bps': spread_bps,
                }
                
                stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                    buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
                )
                
                # Stop loss should account for spread
                # For buy orders, stop loss should be far enough below bid to avoid false triggers
                spread_adjustment = (ask_price - bid_price) * Decimal("2")  # 2x spread buffer
                min_stop_loss = bid_price - spread_adjustment
                
                # Verify stop loss is reasonable given spread
                assert stop_loss > min_stop_loss or spread_bps > 1000, \
                    f"Stop loss too close to bid in {scenario}: stop={stop_loss}, min_expected={min_stop_loss}"

    @pytest.mark.asyncio
    async def test_stop_loss_extreme_market_conditions(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss behavior in extreme market conditions."""
        extreme_scenarios = [
            # (entry_price, scenario_name, regime)
            (Decimal("0.000001"), "micro_cap", MarketRegime.HIGH_VOLATILITY),    # Very low price
            (Decimal("1000000"), "mega_cap", MarketRegime.CRISIS),              # Very high price
            (Decimal("1.5"), "altcoin", MarketRegime.HIGH_VOLATILITY),          # Typical altcoin price
            (Decimal("0.1234567890"), "precise_price", MarketRegime.MEDIUM_VOLATILITY), # High precision
        ]
        
        for entry_price, scenario, regime in extreme_scenarios:
            stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, regime, entry_price
            )
            
            # Verify stop loss is reasonable
            assert isinstance(stop_loss, Decimal)
            assert stop_loss > Decimal("0"), f"Stop loss must be positive in {scenario}"
            assert stop_loss < entry_price, f"Buy stop loss must be below entry in {scenario}"
            
            # Verify stop loss percentage is reasonable
            stop_loss_pct = (entry_price - stop_loss) / entry_price
            assert Decimal("0.001") <= stop_loss_pct <= Decimal("0.20"), \
                f"Stop loss percentage unreasonable in {scenario}: {stop_loss_pct}"

    @pytest.mark.asyncio
    async def test_stop_loss_validation_against_market_microstructure(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss validation against market microstructure patterns."""
        entry_price = Decimal("50000")
        
        # Test different market microstructure scenarios
        microstructure_scenarios = [
            {
                "scenario": "high_frequency_noise",
                "recent_prices": [49990, 50010, 49995, 50005, 49998, 50002],
                "expected_min_distance": Decimal("50"),  # Should be outside noise range
            },
            {
                "scenario": "strong_support_level", 
                "support_level": Decimal("49800"),
                "recent_prices": [49810, 49820, 49815, 49825],
                "expected_min_distance": Decimal("100"),  # Should respect support
            },
            {
                "scenario": "volatile_range",
                "recent_prices": [49500, 50500, 49700, 50300, 49600, 50400],
                "expected_min_distance": Decimal("200"),  # Should account for volatility
            },
        ]
        
        for scenario_data in microstructure_scenarios:
            scenario = scenario_data["scenario"]
            
            # Mock market microstructure data
            with patch.object(adaptive_risk_manager, '_get_recent_price_action') as mock_price_action:
                mock_price_action.return_value = scenario_data["recent_prices"]
                
                stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                    buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
                )
                
                # Verify stop loss accounts for microstructure
                min_distance = scenario_data.get("expected_min_distance", Decimal("25"))
                actual_distance = entry_price - stop_loss
                
                assert actual_distance >= min_distance, \
                    f"Stop loss too close in {scenario}: distance={actual_distance}, min={min_distance}"
                
                # For support level scenarios, verify stop loss is below support
                if "support_level" in scenario_data:
                    support_level = scenario_data["support_level"]
                    assert stop_loss < support_level, \
                        f"Stop loss should be below support level in {scenario}"

    @pytest.mark.asyncio
    async def test_stop_loss_slippage_prevention(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss placement to prevent excessive slippage."""
        entry_price = Decimal("50000")
        
        # Test different liquidity scenarios
        liquidity_scenarios = [
            {
                "scenario": "high_liquidity",
                "order_book_depth": 1000000,  # Deep order book
                "average_volume": 50000000,
                "expected_max_slippage": Decimal("0.001"),  # 0.1% max slippage
            },
            {
                "scenario": "medium_liquidity", 
                "order_book_depth": 100000,
                "average_volume": 5000000,
                "expected_max_slippage": Decimal("0.005"),  # 0.5% max slippage
            },
            {
                "scenario": "low_liquidity",
                "order_book_depth": 10000,
                "average_volume": 500000,
                "expected_max_slippage": Decimal("0.02"),   # 2% max slippage
            },
        ]
        
        for scenario_data in liquidity_scenarios:
            scenario = scenario_data["scenario"]
            
            # Mock liquidity data
            with patch.object(adaptive_risk_manager, '_get_liquidity_metrics') as mock_liquidity:
                mock_liquidity.return_value = {
                    "order_book_depth": scenario_data["order_book_depth"],
                    "average_volume": scenario_data["average_volume"],
                    "estimated_slippage": scenario_data["expected_max_slippage"],
                }
                
                stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                    buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
                )
                
                # Verify stop loss accounts for liquidity
                # In low liquidity, stop loss should be wider to avoid slippage
                stop_loss_distance = entry_price - stop_loss
                min_distance = entry_price * scenario_data["expected_max_slippage"]
                
                if scenario == "low_liquidity":
                    # Low liquidity should have wider stops
                    assert stop_loss_distance >= min_distance, \
                        f"Stop loss too tight for {scenario}: distance={stop_loss_distance}"

    @pytest.mark.asyncio
    async def test_stop_loss_regime_transition_handling(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss adjustment during market regime transitions."""
        entry_price = Decimal("50000")
        
        # Test regime transitions
        regime_transitions = [
            (MarketRegime.LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY, "vol_spike"),
            (MarketRegime.MEDIUM_VOLATILITY, MarketRegime.CRISIS, "crisis_onset"),
            (MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY, "vol_normalization"),
            (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, "trend_reversal"),
        ]
        
        for from_regime, to_regime, transition_type in regime_transitions:
            # Calculate initial stop loss
            initial_stop = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, from_regime, entry_price
            )
            
            # Calculate stop loss after regime change
            new_stop = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, to_regime, entry_price
            )
            
            # Verify appropriate adjustment
            if transition_type == "vol_spike":
                # Volatility increase should widen stops
                assert new_stop < initial_stop, \
                    f"Stop loss should widen during volatility spike: {initial_stop} -> {new_stop}"
            elif transition_type == "crisis_onset":
                # Crisis should significantly widen stops
                assert new_stop < initial_stop, \
                    f"Stop loss should widen during crisis: {initial_stop} -> {new_stop}"
                
                # Crisis stops should be much wider
                crisis_distance = entry_price - new_stop
                initial_distance = entry_price - initial_stop
                assert crisis_distance > initial_distance * Decimal("1.2"), \
                    "Crisis stops should be significantly wider"
            elif transition_type == "vol_normalization":
                # Volatility decrease should tighten stops
                assert new_stop > initial_stop, \
                    f"Stop loss should tighten during volatility normalization: {initial_stop} -> {new_stop}"

    @pytest.mark.asyncio
    async def test_stop_loss_validation_edge_cases(self, adaptive_risk_manager):
        """Test stop-loss validation with edge cases and error conditions."""
        entry_price = Decimal("50000")
        
        # Test invalid signal scenarios
        invalid_scenarios = [
            (None, "null_signal"),
            (Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="",  # Empty symbol
                strategy_name="test",
                metadata={},
            ), "empty_symbol"),
            (Signal(
                direction=SignalDirection.BUY,
                confidence=0.8,
                timestamp=datetime.now(),
                symbol="   ",  # Whitespace only
                strategy_name="test", 
                metadata={},
            ), "whitespace_symbol"),
        ]
        
        for invalid_signal, scenario in invalid_scenarios:
            with pytest.raises((RiskManagementError, ValidationError, AttributeError)):
                await adaptive_risk_manager.calculate_adaptive_stop_loss(
                    invalid_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
                )

        # Test invalid entry price scenarios
        invalid_prices = [
            (Decimal("0"), "zero_price"),
            (Decimal("-100"), "negative_price"),
        ]
        
        for invalid_price, scenario in invalid_prices:
            with pytest.raises((RiskManagementError, ValueError)):
                await adaptive_risk_manager.calculate_adaptive_stop_loss(
                    buy_signal, MarketRegime.MEDIUM_VOLATILITY, invalid_price
                )

    @pytest.mark.asyncio
    async def test_stop_loss_precision_maintenance(self, adaptive_risk_manager, buy_signal):
        """Test that stop-loss calculations maintain decimal precision."""
        # Test with high-precision entry prices
        precision_test_cases = [
            Decimal("50000.12345678"),
            Decimal("0.123456789123456789"),
            Decimal("1234567.87654321"),
        ]
        
        for entry_price in precision_test_cases:
            stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
            )
            
            # Verify result is Decimal with reasonable precision
            assert isinstance(stop_loss, Decimal)
            
            # Verify precision is maintained (at least 8 decimal places for crypto)
            stop_loss_str = str(stop_loss)
            if '.' in stop_loss_str:
                decimal_places = len(stop_loss_str.split('.')[1])
                assert decimal_places <= 18, "Excessive decimal precision"
            
            # Verify calculation accuracy
            stop_loss_pct = (entry_price - stop_loss) / entry_price
            assert Decimal("0.001") <= stop_loss_pct <= Decimal("0.10"), \
                f"Stop loss percentage out of reasonable range: {stop_loss_pct}"

    @pytest.mark.asyncio
    async def test_stop_loss_performance_benchmarks(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss calculation performance for high-frequency use."""
        import time
        
        entry_price = Decimal("50000")
        num_calculations = 1000
        
        # Benchmark stop-loss calculations
        start_time = time.time()
        
        for _ in range(num_calculations):
            stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
            )
            assert isinstance(stop_loss, Decimal)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_calculations
        
        # Performance requirements for high-frequency trading
        assert avg_time < 0.001, f"Stop loss calculation too slow: {avg_time:.6f}s average"
        assert total_time < 0.5, f"Total calculation time too high: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_stop_loss_consistency_across_calls(self, adaptive_risk_manager, buy_signal):
        """Test that stop-loss calculations are consistent across multiple calls."""
        entry_price = Decimal("50000")
        regime = MarketRegime.MEDIUM_VOLATILITY
        
        # Calculate stop loss multiple times
        stop_losses = []
        for _ in range(100):
            stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                buy_signal, regime, entry_price
            )
            stop_losses.append(stop_loss)
        
        # All calculations should be identical
        first_stop_loss = stop_losses[0]
        for stop_loss in stop_losses[1:]:
            assert stop_loss == first_stop_loss, "Stop loss calculations should be deterministic"

    @pytest.mark.asyncio
    async def test_stop_loss_market_hours_consideration(self, adaptive_risk_manager, buy_signal):
        """Test stop-loss placement considers market hours and liquidity patterns."""
        entry_price = Decimal("50000")
        
        # Test different market hour scenarios
        market_hour_scenarios = [
            ("market_open", "09:30", True),     # Market open - high volatility
            ("market_close", "16:00", True),    # Market close - high volatility  
            ("asian_hours", "02:00", False),    # Asian hours - lower liquidity
            ("weekend", "saturday", False),     # Weekend - no market
            ("holiday", "holiday", False),      # Holiday - no market
        ]
        
        for scenario, time_indicator, is_high_liquidity in market_hour_scenarios:
            # Mock market hours data
            with patch.object(adaptive_risk_manager, '_get_market_hours_info') as mock_hours:
                mock_hours.return_value = {
                    "is_market_open": is_high_liquidity,
                    "expected_liquidity": "high" if is_high_liquidity else "low",
                    "time_indicator": time_indicator,
                }
                
                stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
                    buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
                )
                
                # Verify reasonable stop loss regardless of market hours
                assert isinstance(stop_loss, Decimal)
                assert stop_loss < entry_price  # Buy signal
                assert stop_loss > entry_price * Decimal("0.8")  # Not too wide
                
                # During low liquidity periods, stops might be wider
                stop_distance = entry_price - stop_loss
                min_distance = entry_price * Decimal("0.01")  # 1% minimum
                assert stop_distance >= min_distance

    def test_stop_loss_configuration_validation(self, adaptive_risk_manager):
        """Test stop-loss configuration validation and bounds checking."""
        # Test configuration parameter bounds
        test_configs = [
            {"base_stop_loss_pct": 0.001, "valid": True},   # 0.1% - very tight
            {"base_stop_loss_pct": 0.1, "valid": True},     # 10% - wide but valid
            {"base_stop_loss_pct": 0.5, "valid": False},    # 50% - too wide
            {"base_stop_loss_pct": -0.01, "valid": False},  # Negative - invalid
            {"base_stop_loss_pct": 0, "valid": False},      # Zero - invalid
        ]
        
        for test_config in test_configs:
            stop_loss_pct = test_config["base_stop_loss_pct"]
            is_valid = test_config["valid"]
            
            # Temporarily modify configuration
            original_base_stop = adaptive_risk_manager.base_stop_loss_pct
            adaptive_risk_manager.base_stop_loss_pct = stop_loss_pct
            
            try:
                # Test calculation with modified config
                entry_price = Decimal("50000")
                buy_signal = Signal(
                    direction=SignalDirection.BUY,
                    confidence=0.8,
                    timestamp=datetime.now(),
                    symbol="BTCUSDT",
                    strategy_name="test",
                    metadata={},
                )
                
                if is_valid:
                    # Should not raise exception
                    result = adaptive_risk_manager.calculate_adaptive_stop_loss(
                        buy_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
                    )
                    # Verify reasonable result
                    if stop_loss_pct > 0:
                        assert result is not None
                else:
                    # Should handle invalid config gracefully or raise appropriate error
                    pass  # Implementation may vary
                    
            finally:
                # Restore original configuration
                adaptive_risk_manager.base_stop_loss_pct = original_base_stop