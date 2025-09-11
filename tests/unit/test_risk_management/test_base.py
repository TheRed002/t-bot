"""
Unit tests for BaseRiskManager abstract class.

This module tests the abstract base class for risk management implementations.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.config import Config
from src.core.exceptions import ValidationError
from src.core.types.market import MarketData
from src.core.types.risk import RiskLevel, RiskMetrics
from src.core.types.trading import (
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    PositionStatus,
    Signal,
    SignalDirection,
)
from src.risk_management.base import BaseRiskManager


class TestBaseRiskManager:
    """Test cases for BaseRiskManager abstract class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def risk_manager(self, config):
        """Create a concrete implementation of BaseRiskManager for testing."""

        class TestRiskManager(BaseRiskManager):
            async def calculate_position_size(self, signal, portfolio_value):
                return Decimal("1000")

            async def validate_signal(self, signal):
                return True

            async def validate_order(self, order, portfolio_value):
                return True

            async def calculate_risk_metrics(self, positions, market_data):
                portfolio_value = sum(pos.quantity * pos.current_price for pos in positions)
                total_exposure = portfolio_value  # Simple exposure calculation
                return RiskMetrics(
                    portfolio_value=portfolio_value,
                    total_exposure=total_exposure,
                    var_1d=Decimal("500"),
                    var_5d=Decimal("1000"),
                    expected_shortfall=Decimal("750"),
                    max_drawdown=Decimal("0.1"),
                    sharpe_ratio=1.5,  # Use float instead of Decimal
                    current_drawdown=Decimal("0.05"),
                    risk_level=RiskLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                )

            async def check_portfolio_limits(self, new_position):
                return True

            async def should_exit_position(self, position, market_data):
                return False

        return TestRiskManager(config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return Signal(
            symbol="BTC/USDT",
            direction=SignalDirection.BUY,
            strength=0.8,
            timestamp=datetime.now(timezone.utc),
            source="test_strategy",
            metadata={},
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
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("52000"),
            low=Decimal("49000"),
            close=Decimal("51000"),
            volume=Decimal("1000"),
            exchange="binance",
        )

    @pytest.fixture
    def sample_order(self):
        """Create a sample order request."""
        return OrderRequest(
            symbol="BTC/USDT",
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
        assert risk_manager.current_risk_level == RiskLevel.LOW
        assert risk_manager.positions == []
        assert risk_manager.total_portfolio_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_calculate_position_size_abstract(self, risk_manager, sample_signal):
        """Test position size calculation."""
        portfolio_value = Decimal("10000")
        position_size = await risk_manager.calculate_position_size(sample_signal, portfolio_value)

        assert position_size == Decimal("1000")
        assert isinstance(position_size, Decimal)

    @pytest.mark.asyncio
    async def test_validate_signal_abstract(self, risk_manager, sample_signal):
        """Test signal validation."""
        result = await risk_manager.validate_signal(sample_signal)

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_order_abstract(self, risk_manager, sample_order):
        """Test order validation."""
        portfolio_value = Decimal("10000")
        result = await risk_manager.validate_order(sample_order, portfolio_value)

        assert result is True

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_abstract(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test risk metrics calculation."""
        positions = [sample_position]
        market_data = [sample_market_data]

        risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)

        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.var_1d == Decimal("500")
        assert risk_metrics.var_5d == Decimal("1000")
        assert risk_metrics.risk_level == RiskLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_check_portfolio_limits_abstract(self, risk_manager, sample_position):
        """Test portfolio limits checking."""
        result = await risk_manager.check_portfolio_limits(sample_position)

        assert result is True

    @pytest.mark.asyncio
    async def test_should_exit_position_abstract(
        self, risk_manager, sample_position, sample_market_data
    ):
        """Test position exit evaluation."""
        result = await risk_manager.should_exit_position(sample_position, sample_market_data)

        assert result is False

    @pytest.mark.asyncio
    async def test_update_portfolio_state(self, risk_manager, sample_position):
        """Test portfolio state update."""
        positions = [sample_position]
        portfolio_value = Decimal("10000")

        await risk_manager.update_portfolio_state(positions, portfolio_value)

        assert risk_manager.positions == positions
        assert risk_manager.total_portfolio_value == portfolio_value

    @pytest.mark.asyncio
    async def test_get_risk_summary(self, risk_manager):
        """Test risk summary generation."""
        summary = await risk_manager.get_risk_summary()

        assert isinstance(summary, dict)
        assert "risk_level" in summary
        assert "total_positions" in summary
        assert "portfolio_value" in summary
        assert "current_drawdown" in summary
        assert "max_drawdown" in summary

    @pytest.mark.asyncio
    async def test_emergency_stop(self, risk_manager):
        """Test emergency stop functionality."""
        reason = "Test emergency stop"

        await risk_manager.emergency_stop(reason)

        assert risk_manager.current_risk_level == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_validate_risk_parameters(self, risk_manager):
        """Test risk parameter validation."""
        result = await risk_manager.validate_risk_parameters()

        assert result is True

    @pytest.mark.asyncio
    async def test_validate_risk_parameters_invalid(self, config):
        """Test risk parameter validation with invalid parameters."""

        class InvalidRiskManager(BaseRiskManager):
            async def calculate_position_size(self, signal, portfolio_value):
                return Decimal("1000")

            async def validate_signal(self, signal):
                return True

            async def validate_order(self, order, portfolio_value):
                return True

            async def calculate_risk_metrics(self, positions, market_data):
                portfolio_value = (
                    sum(pos.quantity * pos.current_price for pos in positions)
                    if positions
                    else Decimal("10000")
                )
                total_exposure = portfolio_value  # Simple exposure calculation
                return RiskMetrics(
                    portfolio_value=portfolio_value,
                    total_exposure=total_exposure,
                    var_1d=Decimal("500"),
                    var_5d=Decimal("1000"),
                    expected_shortfall=Decimal("750"),
                    max_drawdown=Decimal("0.1"),
                    sharpe_ratio=1.5,  # Use float instead of Decimal
                    current_drawdown=Decimal("0.05"),
                    risk_level=RiskLevel.MEDIUM,
                    timestamp=datetime.now(timezone.utc),
                )

            async def check_portfolio_limits(self, new_position):
                return True

            async def should_exit_position(self, position, market_data):
                return False

        # Create a new config with invalid parameters
        from src.core.config import Config

        invalid_config = Config()
        # We can't directly modify the config due to validation, so we'll test
        # the validation logic directly
        risk_manager = InvalidRiskManager(config)

        # Test with invalid parameters by mocking the validation
        with patch.object(risk_manager, "risk_config") as mock_config:
            mock_config.default_position_size_pct = 1.5  # Invalid: > 1
            with pytest.raises(ValidationError):
                await risk_manager.validate_risk_parameters()

    def test_calculate_portfolio_exposure(self, risk_manager, sample_position):
        """Test portfolio exposure calculation."""
        positions = [sample_position]
        risk_manager.total_portfolio_value = Decimal("10000")

        exposure = risk_manager._calculate_portfolio_exposure(positions)

        expected_exposure = (sample_position.quantity * sample_position.current_price) / Decimal(
            "10000"
        )
        assert exposure == expected_exposure

    def test_calculate_portfolio_exposure_zero_portfolio(self, risk_manager, sample_position):
        """Test portfolio exposure calculation with zero portfolio value."""
        positions = [sample_position]
        risk_manager.total_portfolio_value = Decimal("0")

        exposure = risk_manager._calculate_portfolio_exposure(positions)

        assert exposure == Decimal("0")

    def test_check_drawdown_limit(self, risk_manager):
        """Test drawdown limit checking."""
        current_drawdown = Decimal("0.05")  # 5%
        result = risk_manager._check_drawdown_limit(current_drawdown)

        assert result is True

    def test_check_drawdown_limit_exceeded(self, risk_manager):
        """Test drawdown limit checking when exceeded."""
        # Get the actual max drawdown from config instead of assuming 15%
        max_drawdown = Decimal(str(risk_manager.risk_config.max_drawdown))
        # Test with drawdown that exceeds the limit
        current_drawdown = max_drawdown + Decimal("0.01")  # Exceed by 1%
        result = risk_manager._check_drawdown_limit(current_drawdown)

        assert result is False

    def test_check_daily_loss_limit(self, risk_manager):
        """Test daily loss limit checking."""
        daily_pnl = Decimal("100")  # Positive P&L
        result = risk_manager._check_daily_loss_limit(daily_pnl)

        assert result is True

    def test_check_daily_loss_limit_negative_within_limit(self, risk_manager):
        """Test daily loss limit checking with negative P&L within limit."""
        risk_manager.total_portfolio_value = Decimal("10000")
        daily_pnl = Decimal("-400")  # -4% (within 5% limit)
        result = risk_manager._check_daily_loss_limit(daily_pnl)

        assert result is True

    def test_check_daily_loss_limit_negative_exceeded(self, risk_manager):
        """Test daily loss limit checking with negative P&L exceeding limit."""
        risk_manager.total_portfolio_value = Decimal("10000")
        daily_pnl = Decimal("-600")  # -6% (exceeds 5% limit)
        result = risk_manager._check_daily_loss_limit(daily_pnl)

        assert result is False

    @pytest.mark.asyncio
    async def test_log_risk_violation(self, risk_manager):
        """Test risk violation logging."""
        violation_type = "test_violation"
        details = {"test_key": "test_value"}

        # Should not raise any exceptions
        await risk_manager._log_risk_violation(violation_type, details)

    def test_abstract_methods_not_implemented(self, config):
        """Test that abstract methods must be implemented."""

        class IncompleteRiskManager(BaseRiskManager):
            pass  # No implementation of abstract methods

        with pytest.raises(TypeError):
            IncompleteRiskManager(config)
