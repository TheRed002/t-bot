"""
Unit tests for RiskManager class.

This module tests the core functionality of the RiskManager class,
ensuring it properly integrates all risk management components.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.core.config import Config
from src.core.types import Signal, SignalDirection, Position, OrderSide, MarketData
from src.risk_management import RiskManager


class TestRiskManager:
    """Test cases for RiskManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=Config)
        config.risk = Mock()
        
        # Set up the risk config attributes
        config.risk.default_position_size_pct = 0.02
        config.risk.max_position_size_pct = 0.1
        config.risk.max_total_positions = 10
        config.risk.max_positions_per_symbol = 1
        config.risk.max_portfolio_exposure = 0.95
        config.risk.max_sector_exposure = 0.25
        config.risk.max_correlation_exposure = 0.5
        config.risk.max_leverage = 1.0
        config.risk.max_daily_loss_pct = 0.05
        config.risk.max_drawdown_pct = 0.15
        config.risk.var_confidence_level = 0.95
        config.risk.kelly_lookback_days = 30
        config.risk.kelly_max_fraction = 0.25
        config.risk.volatility_window = 20
        config.risk.volatility_target = 0.02
        config.risk.var_calculation_window = 252
        config.risk.drawdown_calculation_window = 252
        config.risk.correlation_calculation_window = 60
        config.risk.emergency_close_positions = True
        config.risk.emergency_recovery_timeout_hours = 1
        config.risk.emergency_manual_override_enabled = True
        
        # Mock the dict() method to return a proper dictionary
        config.risk.__dict__ = {
            'default_position_size_pct': 0.02,
            'max_position_size_pct': 0.1,
            'max_total_positions': 10,
            'max_positions_per_symbol': 1,
            'max_portfolio_exposure': 0.95,
            'max_sector_exposure': 0.25,
            'max_correlation_exposure': 0.5,
            'max_leverage': 1.0,
            'max_daily_loss_pct': 0.05,
            'max_drawdown_pct': 0.15,
            'var_confidence_level': 0.95,
            'kelly_lookback_days': 30,
            'kelly_max_fraction': 0.25,
            'volatility_window': 20,
            'volatility_target': 0.02,
            'var_calculation_window': 252,
            'drawdown_calculation_window': 252,
            'correlation_calculation_window': 60,
            'emergency_close_positions': True,
            'emergency_recovery_timeout_hours': 1,
            'emergency_manual_override_enabled': True
        }
        
        # Mock the keys() method to return the dictionary keys
        config.risk.keys.return_value = config.risk.__dict__.keys()
        
        return config

    @pytest.fixture
    def risk_manager(self, mock_config):
        """Create a RiskManager instance for testing."""
        return RiskManager(mock_config)

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal for testing."""
        return Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            strategy_name="test_strategy"
        )

    @pytest.fixture
    def sample_position(self):
        """Create a sample position for testing."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.BUY,
            timestamp=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("51000"),
            volume=Decimal("100"),
            timestamp=datetime.now(timezone.utc),
            bid=Decimal("50999"),
            ask=Decimal("51001")
        )

    @pytest.mark.asyncio
    async def test_risk_manager_initialization(self, risk_manager):
        """Test that RiskManager initializes correctly."""
        assert risk_manager is not None
        assert risk_manager.position_sizer is not None
        assert risk_manager.portfolio_limits is not None
        assert risk_manager.risk_calculator is not None
        assert risk_manager.position_limits is not None

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, risk_manager, sample_signal):
        """Test position size calculation."""
        portfolio_value = Decimal("10000")
        position_size = await risk_manager.calculate_position_size(sample_signal, portfolio_value)
        
        assert position_size > 0
        assert position_size <= portfolio_value * Decimal("0.1")  # Max 10%

    @pytest.mark.asyncio
    async def test_validate_signal(self, risk_manager, sample_signal):
        """Test signal validation."""
        is_valid = await risk_manager.validate_signal(sample_signal)
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_validate_order(self, risk_manager):
        """Test order validation."""
        from src.core.types import OrderRequest, OrderType
        
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1")
        )
        
        portfolio_value = Decimal("10000")
        is_valid = await risk_manager.validate_order(order, portfolio_value)
        assert isinstance(is_valid, bool)

    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, risk_manager, sample_position, sample_market_data):
        """Test risk metrics calculation."""
        positions = [sample_position]
        market_data = [sample_market_data]
        
        risk_metrics = await risk_manager.calculate_risk_metrics(positions, market_data)
        
        assert risk_metrics is not None
        assert hasattr(risk_metrics, 'var_1d')
        assert hasattr(risk_metrics, 'max_drawdown')
        assert hasattr(risk_metrics, 'risk_level')

    @pytest.mark.asyncio
    async def test_check_portfolio_limits(self, risk_manager, sample_position):
        """Test portfolio limit checking."""
        is_allowed = await risk_manager.check_portfolio_limits(sample_position)
        assert isinstance(is_allowed, bool)

    @pytest.mark.asyncio
    async def test_should_exit_position(self, risk_manager, sample_position, sample_market_data):
        """Test position exit decision."""
        should_exit = await risk_manager.should_exit_position(sample_position, sample_market_data)
        assert isinstance(should_exit, bool)

    @pytest.mark.asyncio
    async def test_update_portfolio_state(self, risk_manager, sample_position):
        """Test portfolio state update."""
        portfolio_value = Decimal("10000")
        await risk_manager.update_portfolio_state([sample_position], portfolio_value)
        
        assert len(risk_manager.positions) == 1
        assert risk_manager.total_portfolio_value == portfolio_value

    @pytest.mark.asyncio
    async def test_get_risk_summary(self, risk_manager):
        """Test risk summary generation."""
        summary = await risk_manager.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert 'risk_level' in summary
        assert 'total_positions' in summary
        assert 'portfolio_value' in summary

    @pytest.mark.asyncio
    async def test_emergency_stop(self, risk_manager):
        """Test emergency stop functionality."""
        reason = "Test emergency stop"
        await risk_manager.emergency_stop(reason)
        
        assert risk_manager.current_risk_level.value == "critical"

    @pytest.mark.asyncio
    async def test_validate_risk_parameters(self, risk_manager):
        """Test risk parameter validation."""
        is_valid = await risk_manager.validate_risk_parameters()
        assert isinstance(is_valid, bool)

    def test_portfolio_exposure_calculation(self, risk_manager, sample_position):
        """Test portfolio exposure calculation."""
        portfolio_value = Decimal("10000")
        risk_manager.total_portfolio_value = portfolio_value
        
        exposure = risk_manager._calculate_portfolio_exposure([sample_position])
        assert isinstance(exposure, Decimal)
        assert exposure >= 0

    def test_drawdown_limit_check(self, risk_manager):
        """Test drawdown limit checking."""
        current_drawdown = Decimal("0.05")  # 5%
        is_within_limit = risk_manager._check_drawdown_limit(current_drawdown)
        assert isinstance(is_within_limit, bool)

    def test_daily_loss_limit_check(self, risk_manager):
        """Test daily loss limit checking."""
        portfolio_value = Decimal("10000")
        risk_manager.total_portfolio_value = portfolio_value
        
        # Test with no loss
        daily_pnl = Decimal("100")
        is_within_limit = risk_manager._check_daily_loss_limit(daily_pnl)
        assert is_within_limit is True
        
        # Test with loss within limit
        daily_pnl = Decimal("-400")  # 4% loss
        is_within_limit = risk_manager._check_daily_loss_limit(daily_pnl)
        assert is_within_limit is True
        
        # Test with loss exceeding limit
        daily_pnl = Decimal("-600")  # 6% loss
        is_within_limit = risk_manager._check_daily_loss_limit(daily_pnl)
        assert is_within_limit is False
