"""
Unit tests for adaptive risk management module (P-010).

This module tests the AdaptiveRiskManager class and related functionality
for dynamic risk parameter adjustment.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.exceptions import RiskManagementError, ValidationError
from src.core.types.strategy import MarketRegime
from src.core.types.trading import (
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
)

# Import the modules to test
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.regime_detection import MarketRegimeDetector


class TestAdaptiveRiskManager:
    """Test cases for AdaptiveRiskManager class."""

    @pytest.fixture
    def config(self):
        """Test configuration for adaptive risk manager."""
        return {
            "base_position_size_pct": 0.02,  # 2%
            "base_stop_loss_pct": 0.02,  # 2%
            "base_take_profit_pct": 0.04,  # 4%
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
    def sample_signal(self):
        """Create a sample trading signal."""
        return Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            source="test_strategy",
            metadata={"confidence": 0.8},
        )

    @pytest.fixture
    def sample_position(self):
        """Create a sample position."""
        return Position(
            symbol="BTC/USDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=PositionSide.LONG,
            status=PositionStatus.OPEN,
            opened_at=datetime.now(timezone.utc),
            exchange="binance",
            metadata={},
        )

    @pytest.fixture
    def sample_portfolio_positions(self):
        """Create sample portfolio positions."""
        return [
            Position(
                symbol="BTC/USDT",
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000"),
                current_price=Decimal("51000"),
                unrealized_pnl=Decimal("100"),
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
                metadata={},
            ),
            Position(
                symbol="ETH/USDT",
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000"),
                current_price=Decimal("3100"),
                unrealized_pnl=Decimal("100"),
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
                metadata={},
            ),
            Position(
                symbol="ADA/USDT",
                quantity=Decimal("1000"),
                entry_price=Decimal("1.5"),
                current_price=Decimal("1.6"),
                unrealized_pnl=Decimal("100"),
                side=PositionSide.LONG,
                status=PositionStatus.OPEN,
                opened_at=datetime.now(timezone.utc),
                exchange="binance",
                metadata={},
            ),
        ]

    def test_initialization(self, config, regime_detector):
        """Test adaptive risk manager initialization."""
        manager = AdaptiveRiskManager(config, regime_detector)

        assert manager.base_position_size_pct == Decimal("0.02")
        assert manager.base_stop_loss_pct == Decimal("0.02")
        assert manager.base_take_profit_pct == Decimal("0.04")
        assert manager.momentum_window == 20
        assert manager.momentum_threshold == Decimal("0.1")
        assert manager.regime_detector == regime_detector

        # Check regime adjustments exist
        assert MarketRegime.LOW_VOLATILITY in manager.regime_adjustments
        assert MarketRegime.HIGH_VOLATILITY in manager.regime_adjustments
        assert MarketRegime.TRENDING_UP in manager.regime_adjustments

    @pytest.mark.asyncio
    async def test_calculate_adaptive_position_size_low_volatility(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive position size calculation for low volatility regime."""
        portfolio_value = Decimal("10000")

        size = await adaptive_risk_manager.calculate_adaptive_position_size(
            sample_signal, MarketRegime.LOW_VOLATILITY, portfolio_value
        )

        # Should be larger than base size due to low volatility multiplier
        # (1.2)
        base_size = portfolio_value * Decimal("0.02")
        expected_size = base_size * Decimal("1.2")

        assert size > base_size
        assert size <= portfolio_value * Decimal("0.1")  # Max 10%
        assert size >= portfolio_value * Decimal("0.001")  # Min 0.1%

    @pytest.mark.asyncio
    async def test_calculate_adaptive_position_size_high_volatility(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive position size calculation for high volatility regime."""
        portfolio_value = Decimal("10000")

        size = await adaptive_risk_manager.calculate_adaptive_position_size(
            sample_signal, MarketRegime.HIGH_VOLATILITY, portfolio_value
        )

        # Should be smaller than base size due to high volatility multiplier
        # (0.7)
        base_size = portfolio_value * Decimal("0.02")

        assert size < base_size
        assert size <= portfolio_value * Decimal("0.1")  # Max 10%
        assert size >= portfolio_value * Decimal("0.001")  # Min 0.1%

    @pytest.mark.asyncio
    async def test_calculate_adaptive_position_size_crisis(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive position size calculation for crisis regime."""
        portfolio_value = Decimal("10000")

        size = await adaptive_risk_manager.calculate_adaptive_position_size(
            sample_signal, MarketRegime.HIGH_VOLATILITY, portfolio_value
        )

        # Should be very small due to crisis multiplier (0.5)
        base_size = portfolio_value * Decimal("0.02")
        expected_size = base_size * Decimal("0.5")

        assert size < base_size
        assert size <= portfolio_value * Decimal("0.1")  # Max 10%
        assert size >= portfolio_value * Decimal("0.001")  # Min 0.1%

    @pytest.mark.asyncio
    async def test_calculate_adaptive_stop_loss_low_volatility(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive stop loss calculation for low volatility regime."""
        entry_price = Decimal("50000")

        stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
            sample_signal, MarketRegime.LOW_VOLATILITY, entry_price
        )

        # Should be tighter due to low volatility multiplier (0.8)
        base_stop_loss = entry_price * (1 - Decimal("0.02"))

        # Tighter stop (higher price for buy)
        assert stop_loss > base_stop_loss

    @pytest.mark.asyncio
    async def test_calculate_adaptive_stop_loss_high_volatility(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive stop loss calculation for high volatility regime."""
        entry_price = Decimal("50000")

        stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
            sample_signal, MarketRegime.HIGH_VOLATILITY, entry_price
        )

        # Should be wider due to high volatility multiplier (1.3)
        base_stop_loss = entry_price * (1 - Decimal("0.02"))

        assert stop_loss < base_stop_loss  # Wider stop (lower price for buy)

    @pytest.mark.asyncio
    async def test_calculate_adaptive_stop_loss_sell_signal(self, adaptive_risk_manager):
        """Test adaptive stop loss calculation for sell signal."""
        sell_signal = Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.SELL,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            source="test_strategy",
            metadata={},
        )
        entry_price = Decimal("50000")

        stop_loss = await adaptive_risk_manager.calculate_adaptive_stop_loss(
            sell_signal, MarketRegime.MEDIUM_VOLATILITY, entry_price
        )

        # For sell signal, stop loss should be above entry price
        assert stop_loss > entry_price

    @pytest.mark.asyncio
    async def test_calculate_adaptive_take_profit_low_volatility(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive take profit calculation for low volatility regime."""
        entry_price = Decimal("50000")

        take_profit = await adaptive_risk_manager.calculate_adaptive_take_profit(
            sample_signal, MarketRegime.LOW_VOLATILITY, entry_price
        )

        # Should be slightly higher due to low volatility multiplier (1.1)
        base_take_profit = entry_price * Decimal("0.04")
        expected_take_profit = entry_price * (1 + Decimal("0.04") * Decimal("1.1"))

        assert take_profit > entry_price
        assert take_profit > expected_take_profit * Decimal("0.9")  # Approximate

    @pytest.mark.asyncio
    async def test_calculate_adaptive_take_profit_high_volatility(
        self, adaptive_risk_manager, sample_signal
    ):
        """Test adaptive take profit calculation for high volatility regime."""
        entry_price = Decimal("50000")

        take_profit = await adaptive_risk_manager.calculate_adaptive_take_profit(
            sample_signal, MarketRegime.HIGH_VOLATILITY, entry_price
        )

        # Should be higher due to high volatility multiplier (1.2)
        base_take_profit = entry_price * Decimal("0.04")
        expected_take_profit = entry_price * (1 + Decimal("0.04") * Decimal("1.2"))

        assert take_profit > entry_price
        assert take_profit > expected_take_profit * Decimal("0.9")  # Approximate

    @pytest.mark.asyncio
    async def test_calculate_adaptive_portfolio_limits_low_volatility(self, adaptive_risk_manager):
        """Test adaptive portfolio limits for low volatility regime."""
        base_limits = {
            "max_positions": 10,
            "max_portfolio_exposure": 0.95,
            "max_correlation_exposure": 0.5,
        }

        adaptive_limits = await adaptive_risk_manager.calculate_adaptive_portfolio_limits(
            MarketRegime.LOW_VOLATILITY, base_limits
        )

        # Should allow more positions due to low volatility multiplier (1.1)
        assert adaptive_limits["max_positions"] > base_limits["max_positions"]
        assert adaptive_limits["max_portfolio_exposure"] > base_limits["max_portfolio_exposure"]

    @pytest.mark.asyncio
    async def test_calculate_adaptive_portfolio_limits_high_volatility(self, adaptive_risk_manager):
        """Test adaptive portfolio limits for high volatility regime."""
        base_limits = {
            "max_positions": 10,
            "max_portfolio_exposure": 0.95,
            "max_correlation_exposure": 0.5,
        }

        adaptive_limits = await adaptive_risk_manager.calculate_adaptive_portfolio_limits(
            MarketRegime.HIGH_VOLATILITY, base_limits
        )

        # Should allow fewer positions due to high volatility multiplier (0.8)
        assert adaptive_limits["max_positions"] < base_limits["max_positions"]
        assert adaptive_limits["max_portfolio_exposure"] < base_limits["max_portfolio_exposure"]

    @pytest.mark.asyncio
    async def test_calculate_adaptive_portfolio_limits_crisis(self, adaptive_risk_manager):
        """Test adaptive portfolio limits for crisis regime."""
        base_limits = {
            "max_positions": 10,
            "max_portfolio_exposure": 0.95,
            "max_correlation_exposure": 0.5,
        }

        adaptive_limits = await adaptive_risk_manager.calculate_adaptive_portfolio_limits(
            MarketRegime.HIGH_VOLATILITY, base_limits
        )

        # Should allow very few positions due to crisis multiplier (0.5)
        assert adaptive_limits["max_positions"] < base_limits["max_positions"]
        assert adaptive_limits["max_portfolio_exposure"] < base_limits["max_portfolio_exposure"]

    @pytest.mark.asyncio
    async def test_run_stress_test_market_crash(
        self, adaptive_risk_manager, sample_portfolio_positions
    ):
        """Test stress test for market crash scenario."""
        results = await adaptive_risk_manager.run_stress_test(
            sample_portfolio_positions, "market_crash"
        )

        assert results["scenario"] == "market_crash"
        assert results["initial_value"] > 0
        # Market crash
        assert results["stressed_value"] < results["initial_value"]
        assert results["value_change"] < 0  # Negative change
        assert results["value_change_pct"] < 0  # Negative percentage
        assert results["max_drawdown"] < 0  # Negative drawdown
        assert results["positions_affected"] == 3
        assert isinstance(results["timestamp"], datetime)

    @pytest.mark.asyncio
    async def test_run_stress_test_flash_crash(
        self, adaptive_risk_manager, sample_portfolio_positions
    ):
        """Test stress test for flash crash scenario."""
        results = await adaptive_risk_manager.run_stress_test(
            sample_portfolio_positions, "flash_crash"
        )

        assert results["scenario"] == "flash_crash"
        assert results["initial_value"] > 0
        # Flash crash
        assert results["stressed_value"] < results["initial_value"]
        assert results["value_change"] < 0  # Negative change
        assert results["value_change_pct"] < 0  # Negative percentage
        assert results["max_drawdown"] < 0  # Negative drawdown

    @pytest.mark.asyncio
    async def test_run_stress_test_volatility_spike(
        self, adaptive_risk_manager, sample_portfolio_positions
    ):
        """Test stress test for volatility spike scenario."""
        results = await adaptive_risk_manager.run_stress_test(
            sample_portfolio_positions, "volatility_spike"
        )

        assert results["scenario"] == "volatility_spike"
        assert results["initial_value"] > 0
        # Volatility spike
        assert results["stressed_value"] < results["initial_value"]
        assert results["value_change"] < 0  # Negative change
        assert results["value_change_pct"] < 0  # Negative percentage
        assert results["max_drawdown"] < 0  # Negative drawdown

    @pytest.mark.asyncio
    async def test_run_stress_test_unknown_scenario(
        self, adaptive_risk_manager, sample_portfolio_positions
    ):
        """Test stress test with unknown scenario."""
        with pytest.raises(ValidationError):
            await adaptive_risk_manager.run_stress_test(
                sample_portfolio_positions, "unknown_scenario"
            )

    @pytest.mark.asyncio
    async def test_run_stress_test_empty_portfolio(self, adaptive_risk_manager):
        """Test stress test with empty portfolio."""
        results = await adaptive_risk_manager.run_stress_test([], "market_crash")

        assert results["scenario"] == "market_crash"
        assert results["initial_value"] == 0
        assert results["stressed_value"] == 0
        assert results["value_change"] == 0
        assert results["value_change_pct"] == 0
        assert results["max_drawdown"] == 0
        assert results["positions_affected"] == 0

    def test_get_adaptive_parameters_low_volatility(self, adaptive_risk_manager):
        """Test getting adaptive parameters for low volatility regime."""
        params = adaptive_risk_manager.get_adaptive_parameters(MarketRegime.LOW_VOLATILITY)

        assert params["position_size_multiplier"] == 1.2
        assert params["stop_loss_multiplier"] == 0.8
        assert params["take_profit_multiplier"] == 1.1
        assert params["max_positions_multiplier"] == 1.1
        assert params["regime"] == MarketRegime.LOW_VOLATILITY.value

    def test_get_adaptive_parameters_high_volatility(self, adaptive_risk_manager):
        """Test getting adaptive parameters for high volatility regime."""
        params = adaptive_risk_manager.get_adaptive_parameters(MarketRegime.HIGH_VOLATILITY)

        assert params["position_size_multiplier"] == 0.7
        assert params["stop_loss_multiplier"] == 1.3
        assert params["take_profit_multiplier"] == 1.2
        assert params["max_positions_multiplier"] == 0.8
        assert params["regime"] == MarketRegime.HIGH_VOLATILITY.value

    def test_get_adaptive_parameters_high_volatility_extreme(self, adaptive_risk_manager):
        """Test getting adaptive parameters for high volatility regime."""
        params = adaptive_risk_manager.get_adaptive_parameters(MarketRegime.HIGH_VOLATILITY)

        assert params["position_size_multiplier"] == 0.7
        assert params["stop_loss_multiplier"] == 1.3
        assert params["take_profit_multiplier"] == 1.2
        assert params["max_positions_multiplier"] == 0.8
        assert params["regime"] == MarketRegime.HIGH_VOLATILITY.value

    def test_get_stress_test_scenarios(self, adaptive_risk_manager):
        """Test getting available stress test scenarios."""
        scenarios = adaptive_risk_manager.get_stress_test_scenarios()

        expected_scenarios = [
            "market_crash",
            "flash_crash",
            "volatility_spike",
            "correlation_breakdown",
        ]
        assert all(scenario in scenarios for scenario in expected_scenarios)
        assert len(scenarios) == len(expected_scenarios)

    def test_update_regime_detector(self, adaptive_risk_manager, regime_detector):
        """Test updating regime detector reference."""
        new_detector = MarketRegimeDetector({"volatility_window": 30})

        adaptive_risk_manager.update_regime_detector(new_detector)

        assert adaptive_risk_manager.regime_detector == new_detector
        assert adaptive_risk_manager.regime_detector != regime_detector

    @pytest.mark.asyncio
    async def test_error_handling_position_size_calculation(self, adaptive_risk_manager):
        """Test error handling in position size calculation."""
        # Test with invalid signal (empty symbol) - should fail at Signal construction
        with pytest.raises(ValidationError):
            invalid_signal = Signal(
                symbol="",  # This will fail at Signal construction with ValidationError
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(timezone.utc),
                source="test_strategy",
                metadata={},
            )

    @pytest.mark.asyncio
    async def test_error_handling_stop_loss_calculation(self, adaptive_risk_manager):
        """Test error handling in stop loss calculation."""
        # Test with invalid signal (empty symbol) - should fail at Signal construction
        with pytest.raises(ValidationError):
            invalid_signal = Signal(
                symbol="",  # This will fail at Signal construction with ValidationError
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(timezone.utc),
                source="test_strategy",
                metadata={},
            )

    @pytest.mark.asyncio
    async def test_error_handling_take_profit_calculation(self, adaptive_risk_manager):
        """Test error handling in take profit calculation."""
        # Test with invalid signal (empty symbol) - should fail at Signal construction
        with pytest.raises(ValidationError):
            invalid_signal = Signal(
                symbol="",  # This will fail at Signal construction with ValidationError
                direction=SignalDirection.BUY,
                strength=0.8,
                timestamp=datetime.now(timezone.utc),
                source="test_strategy",
                metadata={},
            )

    @pytest.mark.asyncio
    async def test_error_handling_portfolio_limits_calculation(self, adaptive_risk_manager):
        """Test error handling in portfolio limits calculation."""
        with pytest.raises(RiskManagementError):
            await adaptive_risk_manager.calculate_adaptive_portfolio_limits(
                MarketRegime.MEDIUM_VOLATILITY,
                None,  # Invalid base limits
            )

    @pytest.mark.asyncio
    async def test_error_handling_stress_test(self, adaptive_risk_manager):
        """Test error handling in stress test."""
        with pytest.raises(RiskManagementError):
            await adaptive_risk_manager.run_stress_test(
                None,
                "market_crash",  # Invalid positions
            )

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with missing configuration
        config = {}
        regime_detector = MarketRegimeDetector({})
        manager = AdaptiveRiskManager(config, regime_detector)

        # Should use defaults
        assert manager.base_position_size_pct == Decimal("0.02")
        assert manager.base_stop_loss_pct == Decimal("0.02")
        assert manager.base_take_profit_pct == Decimal("0.04")
        assert manager.momentum_window == 20
        assert manager.momentum_threshold == Decimal("0.1")

    def test_regime_adjustments_completeness(self, adaptive_risk_manager):
        """Test that all regime adjustments are properly configured."""
        # Check that all expected regimes have adjustments
        expected_regimes = [
            MarketRegime.LOW_VOLATILITY,
            MarketRegime.MEDIUM_VOLATILITY,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.RANGING,
        ]

        for regime in expected_regimes:
            assert regime in adaptive_risk_manager.regime_adjustments
            adj = adaptive_risk_manager.regime_adjustments[regime]
            assert "position_size_multiplier" in adj
            assert "stop_loss_multiplier" in adj
            assert "take_profit_multiplier" in adj
            assert "max_positions_multiplier" in adj

    def test_correlation_adjustments_completeness(self, adaptive_risk_manager):
        """Test that correlation adjustments are properly configured."""
        # Check that correlation regimes have adjustments
        expected_correlation_regimes = ["high_correlation", "low_correlation"]

        for regime in expected_correlation_regimes:
            assert regime in adaptive_risk_manager.correlation_adjustments
            adj = adaptive_risk_manager.correlation_adjustments[regime]
            assert "position_size_multiplier" in adj
            assert "max_positions_multiplier" in adj

    def test_stress_test_scenarios_completeness(self, adaptive_risk_manager):
        """Test that stress test scenarios are properly configured."""
        expected_scenarios = [
            "market_crash",
            "flash_crash",
            "volatility_spike",
            "correlation_breakdown",
        ]

        for scenario in expected_scenarios:
            assert scenario in adaptive_risk_manager.stress_test_scenarios
            scenario_config = adaptive_risk_manager.stress_test_scenarios[scenario]
            assert "price_shock" in scenario_config
            assert "volatility_multiplier" in scenario_config
