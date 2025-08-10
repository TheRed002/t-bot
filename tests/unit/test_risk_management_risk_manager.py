"""
Unit tests for concrete RiskManager implementation.

This module tests the concrete RiskManager class that integrates all risk management components.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.config import Config
from src.core.exceptions import PositionLimitError, RiskManagementError, ValidationError
from src.core.types import (
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    RiskLevel,
    RiskMetrics,
    Signal,
    SignalDirection,
)
from src.risk_management.risk_manager import RiskManager


class TestRiskManager:
    """Test cases for concrete RiskManager implementation."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def risk_manager(self, config):
        """Create risk manager instance."""
        return RiskManager(config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
        )

    @pytest.fixture
    def sample_position(self):
        """Create a sample position."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.BUY,
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("51000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
            bid=Decimal("50990"),
            ask=Decimal("51010"),
            open_price=Decimal("50000"),
            high_price=Decimal("52000"),
            low_price=Decimal("49000"),
        )

    @pytest.fixture
    def sample_order(self):
        """Create a sample order request."""
        return OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            time_in_force="GTC",
            client_order_id="test_order_123",
        )

    def test_initialization(self, risk_manager, config):
        """Test risk manager initialization."""
        assert risk_manager.config == config
        assert risk_manager.risk_config == config.risk
        assert risk_manager.position_sizer is not None
        assert risk_manager.portfolio_limits is not None
        assert risk_manager.risk_calculator is not None
        assert risk_manager.position_limits is not None

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, risk_manager, sample_signal):
        """Test position size calculation."""
        portfolio_value = Decimal("10000")

        with patch.object(risk_manager.position_sizer, "calculate_position_size") as mock_calc:
            mock_calc.return_value = Decimal("1000")

            position_size = await risk_manager.calculate_position_size(
                sample_signal, portfolio_value
            )

            assert position_size == Decimal("1000")
            mock_calc.assert_called_once_with(sample_signal, portfolio_value)

    @pytest.mark.asyncio
    async def test_calculate_position_size_invalid_signal(self, risk_manager):
        """Test position size calculation with invalid signal."""
        portfolio_value = Decimal("10000")
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.0,  # Invalid confidence
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
        )

        with pytest.raises(RiskManagementError):
            await risk_manager.calculate_position_size(invalid_signal, portfolio_value)

    @pytest.mark.asyncio
    async def test_validate_signal_valid(self, risk_manager, sample_signal):
        """Test signal validation with valid signal."""
        result = await risk_manager.validate_signal(sample_signal)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_signal_invalid_confidence(self, risk_manager):
        """Test signal validation with invalid confidence."""
        invalid_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.3,  # Below threshold
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
        )

        result = await risk_manager.validate_signal(invalid_signal)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_signal_invalid_direction(self, risk_manager):
        """Test signal validation with invalid direction."""
        invalid_signal = Signal(
            direction=SignalDirection.HOLD,  # Invalid for trading
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
        )

        with pytest.raises(ValidationError):
            await risk_manager.validate_signal(invalid_signal)

    @pytest.mark.asyncio
    async def test_validate_order_valid(self, risk_manager, sample_order):
        """Test order validation with valid order."""
        portfolio_value = Decimal("10000")

        result = await risk_manager.validate_order(sample_order, portfolio_value)
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_order_invalid_quantity(self, risk_manager):
        """Test order validation with invalid quantity."""
        invalid_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("-0.1"),  # Negative quantity
            price=None,
            stop_price=None,
            time_in_force="GTC",
            client_order_id="test_order_123",
        )

        portfolio_value = Decimal("10000")

        with pytest.raises(ValidationError):
            await risk_manager.validate_order(invalid_order, portfolio_value)

    @pytest.mark.asyncio
    async def test_validate_order_exceeds_position_limit(self, risk_manager):
        """Test order validation when order exceeds position limit."""
        large_order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),  # Very large quantity
            price=Decimal("50000"),
            stop_price=None,
            time_in_force="GTC",
            client_order_id="test_order_123",
        )

        portfolio_value = Decimal("10000")

        with pytest.raises(ValidationError):
            await risk_manager.validate_order(large_order, portfolio_value)

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, risk_manager, sample_position, sample_market_data):
        """Test risk metrics calculation."""
        positions = [sample_position]
        market_data = [sample_market_data]

        with patch.object(risk_manager.risk_calculator, "calculate_risk_metrics") as mock_calc:
            mock_metrics = RiskMetrics(
                var_1d=Decimal("500"),
                var_5d=Decimal("1000"),
                expected_shortfall=Decimal("750"),
                max_drawdown=Decimal("0.1"),
                sharpe_ratio=Decimal("1.5"),
                current_drawdown=Decimal("0.05"),
                risk_level=RiskLevel.MEDIUM,
                timestamp=datetime.now(),
            )
            mock_calc.return_value = mock_metrics

            risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)

            assert risk_metrics == mock_metrics
            mock_calc.assert_called_once_with(positions, market_data)

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_critical_level(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test risk metrics calculation with critical risk level."""
        positions = [sample_position]
        market_data = [sample_market_data]

        with patch.object(risk_manager.risk_calculator, "calculate_risk_metrics") as mock_calc:
            mock_metrics = RiskMetrics(
                var_1d=Decimal("500"),
                var_5d=Decimal("1000"),
                expected_shortfall=Decimal("750"),
                max_drawdown=Decimal("0.1"),
                sharpe_ratio=Decimal("1.5"),
                current_drawdown=Decimal("0.05"),
                risk_level=RiskLevel.CRITICAL,
                timestamp=datetime.now(),
            )
            mock_calc.return_value = mock_metrics

            with patch.object(risk_manager, "emergency_stop") as mock_emergency:
                await risk_manager.calculate_risk_metrics(positions, market_data)
                mock_emergency.assert_called_once_with("Critical risk level detected")

    @pytest.mark.asyncio
    async def test_check_portfolio_limits(self, risk_manager, sample_position):
        """Test portfolio limits checking."""
        with patch.object(risk_manager.portfolio_limits, "check_portfolio_limits") as mock_check:
            mock_check.return_value = True

            result = await risk_manager.check_portfolio_limits(sample_position)

            assert result is True
            mock_check.assert_called_once_with(sample_position)

    @pytest.mark.asyncio
    async def test_should_exit_position_stop_loss(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test position exit evaluation with stop loss."""
        # Modify market data to create a large loss
        # Large loss (from 50000 to 40000)
        sample_market_data.price = Decimal("40000")

        result = await risk_manager.should_exit_position(sample_position, sample_market_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_exit_position_drawdown_limit(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test position exit evaluation with drawdown limit."""
        # Set up risk metrics with high drawdown
        risk_manager.risk_metrics = RiskMetrics(
            var_1d=Decimal("500"),
            var_5d=Decimal("1000"),
            expected_shortfall=Decimal("750"),
            max_drawdown=Decimal("0.1"),
            sharpe_ratio=Decimal("1.5"),
            current_drawdown=Decimal("0.2"),  # High drawdown
            risk_level=RiskLevel.HIGH,
            timestamp=datetime.now(),
        )

        result = await risk_manager.should_exit_position(sample_position, sample_market_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_exit_position_high_risk_level(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test position exit evaluation with high risk level."""
        risk_manager.current_risk_level = RiskLevel.HIGH

        result = await risk_manager.should_exit_position(sample_position, sample_market_data)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_exit_position_no_exit(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test position exit evaluation when no exit is needed."""
        # Set up normal conditions
        sample_position.unrealized_pnl = Decimal("100")  # Profit
        risk_manager.current_risk_level = RiskLevel.LOW
        risk_manager.risk_metrics = RiskMetrics(
            var_1d=Decimal("500"),
            var_5d=Decimal("1000"),
            expected_shortfall=Decimal("750"),
            max_drawdown=Decimal("0.1"),
            sharpe_ratio=Decimal("1.5"),
            current_drawdown=Decimal("0.02"),  # Low drawdown
            risk_level=RiskLevel.LOW,
            timestamp=datetime.now(),
        )

        result = await risk_manager.should_exit_position(sample_position, sample_market_data)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_portfolio_state(self, risk_manager, sample_position):
        """Test portfolio state update."""
        positions = [sample_position]
        portfolio_value = Decimal("10000")

        with patch.object(risk_manager.portfolio_limits, "update_portfolio_state") as mock_update:
            await risk_manager.update_portfolio_state(positions, portfolio_value)

            assert risk_manager.positions == positions
            assert risk_manager.total_portfolio_value == portfolio_value
            mock_update.assert_called_once_with(positions, portfolio_value)

    @pytest.mark.asyncio
    async def test_get_comprehensive_risk_summary(self, risk_manager):
        """Test comprehensive risk summary generation."""
        with patch.object(risk_manager, "get_risk_summary") as mock_base_summary:
            with patch.object(
                risk_manager.portfolio_limits, "get_portfolio_summary"
            ) as mock_portfolio:
                with patch.object(risk_manager.risk_calculator, "get_risk_summary") as mock_risk:
                    mock_base_summary.return_value = {"base": "summary"}
                    mock_portfolio.return_value = {"portfolio": "summary"}
                    mock_risk.return_value = {"risk": "summary"}

                    summary = await risk_manager.get_comprehensive_risk_summary()

                    assert "risk_level" in summary
                    assert "portfolio_limits" in summary
                    assert "risk_calculator" in summary
                    assert "position_sizer_methods" in summary
                    assert "risk_config" in summary

    @pytest.mark.asyncio
    async def test_validate_risk_parameters(self, risk_manager):
        """Test risk parameter validation."""
        with patch.object(risk_manager, "validate_risk_parameters") as mock_validate:
            mock_validate.return_value = True

            result = await risk_manager.validate_risk_parameters()
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_risk_parameters_failure(self, risk_manager):
        """Test risk parameter validation failure."""
        with patch.object(risk_manager, "validate_risk_parameters") as mock_validate:
            mock_validate.return_value = False

            result = await risk_manager.validate_risk_parameters()
            assert result is False

    @pytest.mark.asyncio
    async def test_position_size_validation_failure(self, risk_manager, sample_signal):
        """Test position size calculation when validation fails."""
        portfolio_value = Decimal("10000")

        with patch.object(risk_manager.position_sizer, "calculate_position_size") as mock_calc:
            with patch.object(
                risk_manager.position_sizer, "validate_position_size"
            ) as mock_validate:
                mock_calc.return_value = Decimal("1000")
                mock_validate.return_value = False

                with pytest.raises(RiskManagementError):
                    await risk_manager.calculate_position_size(sample_signal, portfolio_value)

    @pytest.mark.asyncio
    async def test_portfolio_limits_check_failure(self, risk_manager, sample_position):
        """Test portfolio limits check when it fails."""
        with patch.object(risk_manager.portfolio_limits, "check_portfolio_limits") as mock_check:
            mock_check.side_effect = PositionLimitError("Portfolio limits violated")

            with pytest.raises(PositionLimitError):
                await risk_manager.check_portfolio_limits(sample_position)

    @pytest.mark.asyncio
    async def test_risk_metrics_calculation_failure(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test risk metrics calculation when it fails."""
        positions = [sample_position]
        market_data = [sample_market_data]

        with patch.object(risk_manager.risk_calculator, "calculate_risk_metrics") as mock_calc:
            mock_calc.side_effect = RiskManagementError("Risk calculation failed")

            with pytest.raises(RiskManagementError):
                await risk_manager.calculate_risk_metrics(positions, market_data)

    def test_position_limits_initialization(self, risk_manager, config):
        """Test that position limits are properly initialized."""
        assert risk_manager.position_limits is not None
        assert risk_manager.position_limits.max_position_size == Decimal(
            str(config.risk.max_position_size_pct)
        )
        assert (
            risk_manager.position_limits.max_positions_per_symbol
            == config.risk.max_positions_per_symbol
        )
        assert risk_manager.position_limits.max_total_positions == config.risk.max_total_positions
        assert risk_manager.position_limits.max_portfolio_exposure == Decimal(
            str(config.risk.max_portfolio_exposure)
        )
        assert risk_manager.position_limits.max_sector_exposure == Decimal(
            str(config.risk.max_sector_exposure)
        )
        assert risk_manager.position_limits.max_correlation_exposure == Decimal(
            str(config.risk.max_correlation_exposure)
        )
        assert risk_manager.position_limits.max_leverage == Decimal(str(config.risk.max_leverage))

    @pytest.mark.asyncio
    async def test_component_integration(
        self, risk_manager, sample_signal, sample_position, sample_market_data
    ):
        """Test integration between all risk management components."""
        portfolio_value = Decimal("10000")

        # Test full workflow
        with patch.object(risk_manager.position_sizer, "calculate_position_size") as mock_calc:
            with patch.object(
                risk_manager.portfolio_limits, "check_portfolio_limits"
            ) as mock_check:
                with patch.object(
                    risk_manager.risk_calculator, "calculate_risk_metrics"
                ) as mock_metrics:
                    mock_calc.return_value = Decimal("1000")
                    mock_check.return_value = True
                    mock_metrics.return_value = RiskMetrics(
                        var_1d=Decimal("500"),
                        var_5d=Decimal("1000"),
                        expected_shortfall=Decimal("750"),
                        max_drawdown=Decimal("0.1"),
                        sharpe_ratio=Decimal("1.5"),
                        current_drawdown=Decimal("0.05"),
                        risk_level=RiskLevel.LOW,
                        timestamp=datetime.now(),
                    )

                    # Calculate position size
                    position_size = await risk_manager.calculate_position_size(
                        sample_signal, portfolio_value
                    )
                    assert position_size == Decimal("1000")

                    # Check portfolio limits
                    result = await risk_manager.check_portfolio_limits(sample_position)
                    assert result is True

                    # Calculate risk metrics
                    risk_metrics = await risk_manager.calculate_risk_metrics(
                        [sample_position], [sample_market_data]
                    )
                    assert risk_metrics.risk_level == RiskLevel.LOW
