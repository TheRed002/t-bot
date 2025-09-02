"""
Unit tests for Emergency Controls System (P-009).

This module tests the emergency controls functionality including:
- Emergency stop activation
- Position closure procedures
- Order cancellation procedures
- Manual override capabilities
- Recovery procedures
- Order validation during emergency

CRITICAL: Tests must achieve 90% coverage for P-009 implementation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import EmergencyStopError, ValidationError

# Import from P-001
from src.core.types.risk import CircuitBreakerType, RiskLevel
from src.core.types.trading import (
    OrderRequest,
    OrderSide,
    OrderType,
    PositionSide,
    PositionStatus,
)

# Import from P-003+
from src.exchanges.base import BaseExchange

# Import from P-008
from src.risk_management.base import BaseRiskManager
from src.risk_management.circuit_breakers import CircuitBreakerManager
from src.risk_management.emergency_controls import (
    EmergencyAction,
    EmergencyControls,
    EmergencyState,
)


class TestEmergencyControls:
    """Test emergency controls functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.emergency_close_positions = True
        config.risk.emergency_recovery_timeout_hours = 1
        config.risk.emergency_manual_override_enabled = True
        config.risk.max_position_size_pct = 0.1
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        risk_manager = Mock(spec=BaseRiskManager)
        risk_manager.calculate_risk_metrics = AsyncMock()
        return risk_manager

    @pytest.fixture
    def mock_circuit_breaker_manager(self):
        """Create mock circuit breaker manager."""
        manager = Mock(spec=CircuitBreakerManager)
        manager.get_triggered_breakers = Mock(return_value=[])
        return manager

    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        exchange = Mock(spec=BaseExchange)
        exchange.get_pending_orders = AsyncMock(return_value=[])
        exchange.get_positions = AsyncMock(return_value=[])
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.place_order = AsyncMock(return_value=Mock(status="filled"))
        exchange.get_account_balance = AsyncMock(return_value={"USDT": Decimal("10000")})
        return exchange

    @pytest.fixture
    def emergency_controls(self, mock_config, mock_risk_manager, mock_circuit_breaker_manager):
        """Create emergency controls instance."""
        with patch("src.risk_management.emergency_controls.ErrorHandler", return_value=Mock()):
            return EmergencyControls(mock_config, mock_risk_manager, mock_circuit_breaker_manager)

    def test_initialization(self, emergency_controls):
        """Test emergency controls initialization."""
        assert emergency_controls.state == EmergencyState.NORMAL
        assert emergency_controls.emergency_start_time is None
        assert emergency_controls.emergency_reason is None
        assert emergency_controls.manual_override_user is None
        assert len(emergency_controls.emergency_events) == 0
        assert len(emergency_controls.exchanges) == 0

    def test_register_exchange(self, emergency_controls, mock_exchange):
        """Test exchange registration."""
        emergency_controls.register_exchange("binance", mock_exchange)

        assert "binance" in emergency_controls.exchanges
        assert emergency_controls.exchanges["binance"] == mock_exchange

    @pytest.mark.asyncio
    async def test_activate_emergency_stop(self, emergency_controls):
        """Test emergency stop activation."""
        reason = "Daily loss limit exceeded"
        trigger_type = CircuitBreakerType.DAILY_LOSS_LIMIT

        await emergency_controls.activate_emergency_stop(reason, trigger_type)

        assert emergency_controls.state == EmergencyState.EMERGENCY
        assert emergency_controls.emergency_start_time is not None
        assert emergency_controls.emergency_reason == reason
        assert len(emergency_controls.emergency_events) == 1

        event = emergency_controls.emergency_events[0]
        assert event.action == EmergencyAction.SWITCH_TO_SAFE_MODE
        assert event.trigger_type == trigger_type
        assert event.description == f"Emergency stop activated: {reason}"

    @pytest.mark.asyncio
    async def test_activate_emergency_stop_failure(self, emergency_controls):
        """Test emergency stop activation failure."""
        # Mock the _execute_emergency_procedures to raise an exception
        emergency_controls._execute_emergency_procedures = AsyncMock(
            side_effect=Exception("Test error")
        )

        with pytest.raises(EmergencyStopError):
            await emergency_controls.activate_emergency_stop(
                "Test reason", CircuitBreakerType.DAILY_LOSS_LIMIT
            )

    @pytest.mark.asyncio
    async def test_cancel_all_pending_orders(self, emergency_controls, mock_exchange):
        """Test cancellation of all pending orders."""
        # Register exchange
        emergency_controls.register_exchange("binance", mock_exchange)

        # Mock pending orders
        mock_orders = [Mock(id="order1", symbol="BTC/USDT"), Mock(id="order2", symbol="ETH/USDT")]
        mock_exchange.get_pending_orders.return_value = mock_orders

        await emergency_controls._cancel_all_pending_orders()

        # Verify orders were cancelled
        assert mock_exchange.cancel_order.call_count == 2
        mock_exchange.cancel_order.assert_any_call("order1")
        mock_exchange.cancel_order.assert_any_call("order2")

        # Verify emergency event was updated
        assert len(emergency_controls.emergency_events) > 0
        assert emergency_controls.emergency_events[-1].orders_cancelled == 2

    @pytest.mark.asyncio
    async def test_close_all_positions(self, emergency_controls, mock_exchange):
        """Test closure of all positions."""
        # Register exchange
        emergency_controls.register_exchange("binance", mock_exchange)

        # Mock positions
        mock_positions = [
            Mock(symbol="BTC/USDT", quantity=Decimal("1.0"), side=OrderSide.BUY),
            Mock(symbol="ETH/USDT", quantity=Decimal("10.0"), side=OrderSide.SELL),
        ]
        mock_exchange.get_positions.return_value = mock_positions

        await emergency_controls._close_all_positions()

        # Verify orders were placed to close positions
        assert mock_exchange.place_order.call_count == 2

        # Verify emergency event was updated
        assert len(emergency_controls.emergency_events) > 0
        assert emergency_controls.emergency_events[-1].positions_affected == 2

    @pytest.mark.asyncio
    async def test_close_all_positions_disabled(self, emergency_controls):
        """Test position closure when disabled."""
        # Disable position closure
        emergency_controls.config.risk.emergency_close_positions = False

        await emergency_controls._close_all_positions()

        # Should not call exchange methods
        # (This is a negative test - we verify no exceptions are raised)

    @pytest.mark.asyncio
    async def test_validate_order_during_emergency_normal_state(self, emergency_controls):
        """Test order validation during normal state."""
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_validate_order_during_emergency_emergency_state(self, emergency_controls):
        """Test order validation during emergency state."""
        emergency_controls.state = EmergencyState.EMERGENCY

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_validate_order_during_emergency_manual_override(self, emergency_controls):
        """Test order validation during manual override."""
        emergency_controls.state = EmergencyState.MANUAL_OVERRIDE

        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is True

    @pytest.mark.asyncio
    async def test_validate_order_during_emergency_recovery_mode(
        self, emergency_controls, mock_exchange
    ):
        """Test order validation during recovery mode."""
        emergency_controls.state = EmergencyState.RECOVERY
        emergency_controls.register_exchange("binance", mock_exchange)

        # Test order within recovery limits
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
            # Small order (0.01 BTC = $500 out of $10k = 5%)
            price=Decimal("50000"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is True

        # Test order exceeding recovery limits
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0"),  # Large order
            price=Decimal("50000"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is False

        # Test non-allowed order type
        order = OrderRequest(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,  # Not allowed during recovery
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_deactivate_emergency_stop(self, emergency_controls):
        """Test emergency stop deactivation."""
        # Set emergency state
        emergency_controls.state = EmergencyState.EMERGENCY
        emergency_controls.emergency_start_time = datetime.now(timezone.utc)
        emergency_controls.emergency_reason = "Test reason"

        await emergency_controls.deactivate_emergency_stop("Manual deactivation")

        assert emergency_controls.state == EmergencyState.RECOVERY
        assert emergency_controls.emergency_start_time is not None
        assert emergency_controls.emergency_reason == "Test reason"

    @pytest.mark.asyncio
    async def test_deactivate_emergency_stop_recovery_to_normal(self, emergency_controls):
        """Test deactivation from recovery to normal state."""
        # Set recovery state
        emergency_controls.state = EmergencyState.RECOVERY

        await emergency_controls.deactivate_emergency_stop("Recovery completed")

        assert emergency_controls.state == EmergencyState.NORMAL
        assert emergency_controls.emergency_start_time is None
        assert emergency_controls.emergency_reason is None

    @pytest.mark.asyncio
    async def test_activate_manual_override(self, emergency_controls):
        """Test manual override activation."""
        user_id = "test_user"
        reason = "Manual intervention required"

        await emergency_controls.activate_manual_override(user_id, reason)

        assert emergency_controls.state == EmergencyState.MANUAL_OVERRIDE
        assert emergency_controls.manual_override_user == user_id
        assert emergency_controls.manual_override_time is not None

    @pytest.mark.asyncio
    async def test_deactivate_manual_override_success(self, emergency_controls):
        """Test successful manual override deactivation."""
        user_id = "test_user"
        emergency_controls.manual_override_user = user_id
        emergency_controls.state = EmergencyState.MANUAL_OVERRIDE

        await emergency_controls.deactivate_manual_override(user_id)

        assert emergency_controls.state == EmergencyState.NORMAL
        assert emergency_controls.manual_override_user is None
        assert emergency_controls.manual_override_time is None

    @pytest.mark.asyncio
    async def test_deactivate_manual_override_unauthorized(self, emergency_controls):
        """Test unauthorized manual override deactivation."""
        emergency_controls.manual_override_user = "user1"
        emergency_controls.state = EmergencyState.MANUAL_OVERRIDE

        with pytest.raises(ValidationError):
            await emergency_controls.deactivate_manual_override("user2")

    @pytest.mark.asyncio
    async def test_validate_recovery_completion_success(
        self, emergency_controls, mock_risk_manager
    ):
        """Test successful recovery completion validation."""
        # Mock risk metrics
        mock_risk_metrics = Mock()
        mock_risk_metrics.risk_level = RiskLevel.LOW
        mock_risk_manager.calculate_risk_metrics.return_value = mock_risk_metrics

        result = await emergency_controls._validate_recovery_completion()
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_recovery_completion_high_risk(
        self, emergency_controls, mock_risk_manager
    ):
        """Test recovery completion validation with high risk."""
        # Mock risk metrics
        mock_risk_metrics = Mock()
        mock_risk_metrics.risk_level = RiskLevel.HIGH
        mock_risk_manager.calculate_risk_metrics.return_value = mock_risk_metrics

        result = await emergency_controls._validate_recovery_completion()
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_recovery_completion_circuit_breakers_triggered(
        self, emergency_controls
    ):
        """Test recovery completion validation with triggered circuit breakers."""
        # Mock triggered circuit breakers
        emergency_controls.circuit_breaker_manager.get_triggered_breakers.return_value = [
            "daily_loss_limit"
        ]

        result = await emergency_controls._validate_recovery_completion()
        assert result is False

    def test_get_status(self, emergency_controls):
        """Test status retrieval."""
        status = emergency_controls.get_status()

        assert "state" in status
        assert "emergency_start_time" in status
        assert "emergency_reason" in status
        assert "manual_override_user" in status
        assert "manual_override_time" in status
        assert "events_count" in status
        assert "trading_allowed" in status
        assert status["state"] == "normal"
        assert status["trading_allowed"] is True

    def test_is_trading_allowed_normal(self, emergency_controls):
        """Test trading allowed check in normal state."""
        assert emergency_controls.is_trading_allowed() is True

    def test_is_trading_allowed_emergency(self, emergency_controls):
        """Test trading allowed check in emergency state."""
        emergency_controls.state = EmergencyState.EMERGENCY
        assert emergency_controls.is_trading_allowed() is False

    def test_is_trading_allowed_manual_override(self, emergency_controls):
        """Test trading allowed check in manual override state."""
        emergency_controls.state = EmergencyState.MANUAL_OVERRIDE
        assert emergency_controls.is_trading_allowed() is True

    def test_is_trading_allowed_recovery(self, emergency_controls):
        """Test trading allowed check in recovery state."""
        emergency_controls.state = EmergencyState.RECOVERY
        assert emergency_controls.is_trading_allowed() is False

    def test_get_emergency_events(self, emergency_controls):
        """Test emergency events retrieval."""
        # Add some events
        emergency_controls.emergency_events = [Mock(), Mock(), Mock()]

        events = emergency_controls.get_emergency_events(limit=2)
        assert len(events) == 2

        events = emergency_controls.get_emergency_events()
        assert len(events) == 3


class TestEmergencyControlsIntegration:
    """Integration tests for emergency controls system."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.emergency_close_positions = True
        config.risk.emergency_recovery_timeout_hours = 1
        config.risk.emergency_manual_override_enabled = True
        config.risk.max_position_size_pct = 0.1
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        risk_manager = Mock(spec=BaseRiskManager)
        risk_manager.calculate_risk_metrics = AsyncMock()
        return risk_manager

    @pytest.fixture
    def mock_circuit_breaker_manager(self):
        """Create mock circuit breaker manager."""
        manager = Mock(spec=CircuitBreakerManager)
        manager.get_triggered_breakers = Mock(return_value=[])
        return manager

    @pytest.mark.asyncio
    async def test_complete_emergency_cycle(
        self, mock_config, mock_risk_manager, mock_circuit_breaker_manager
    ):
        """Test complete emergency cycle from activation to recovery."""
        with patch("src.risk_management.emergency_controls.ErrorHandler", return_value=Mock()):
            emergency_controls = EmergencyControls(
                mock_config, mock_risk_manager, mock_circuit_breaker_manager
            )

            # Initial state
            assert emergency_controls.state == EmergencyState.NORMAL
            assert emergency_controls.is_trading_allowed() is True

            # Activate emergency stop
            await emergency_controls.activate_emergency_stop(
                "Test emergency", CircuitBreakerType.DAILY_LOSS_LIMIT
            )

            assert emergency_controls.state == EmergencyState.EMERGENCY
            assert emergency_controls.is_trading_allowed() is False

            # Deactivate emergency stop (enters recovery)
            await emergency_controls.deactivate_emergency_stop("Manual deactivation")

            assert emergency_controls.state == EmergencyState.RECOVERY
            assert emergency_controls.is_trading_allowed() is False

            # Complete recovery
            await emergency_controls.deactivate_emergency_stop("Recovery completed")

            assert emergency_controls.state == EmergencyState.NORMAL
            assert emergency_controls.is_trading_allowed() is True

    @pytest.mark.asyncio
    async def test_emergency_with_exchanges(
        self, mock_config, mock_risk_manager, mock_circuit_breaker_manager
    ):
        """Test emergency procedures with registered exchanges."""
        with patch("src.risk_management.emergency_controls.ErrorHandler", return_value=Mock()):
            emergency_controls = EmergencyControls(
                mock_config, mock_risk_manager, mock_circuit_breaker_manager
            )

            # Create mock exchanges
            exchange1 = Mock(spec=BaseExchange)
            exchange1.get_pending_orders = AsyncMock(return_value=[Mock(id="order1")])
            exchange1.get_positions = AsyncMock(
                return_value=[Mock(symbol="BTC/USDT", quantity=Decimal("1.0"), side=OrderSide.BUY)]
            )
            exchange1.cancel_order = AsyncMock(return_value=True)
            exchange1.place_order = AsyncMock(return_value=Mock(status="filled"))

            exchange2 = Mock(spec=BaseExchange)
            exchange2.get_pending_orders = AsyncMock(return_value=[Mock(id="order2")])
            exchange2.get_positions = AsyncMock(return_value=[])
            exchange2.cancel_order = AsyncMock(return_value=True)

            # Register exchanges
            emergency_controls.register_exchange("binance", exchange1)
            emergency_controls.register_exchange("okx", exchange2)

            # Activate emergency stop
            await emergency_controls.activate_emergency_stop(
                "Test emergency", CircuitBreakerType.DAILY_LOSS_LIMIT
            )

            # Verify orders were cancelled
            assert exchange1.cancel_order.call_count == 1
            assert exchange2.cancel_order.call_count == 1

            # Verify positions were closed
            assert exchange1.place_order.call_count == 1
            assert exchange2.place_order.call_count == 0  # No positions to close

    @pytest.mark.asyncio
    async def test_manual_override_cycle(
        self, mock_config, mock_risk_manager, mock_circuit_breaker_manager
    ):
        """Test manual override cycle."""
        with patch("src.risk_management.emergency_controls.ErrorHandler", return_value=Mock()):
            emergency_controls = EmergencyControls(
                mock_config, mock_risk_manager, mock_circuit_breaker_manager
            )

            # Start in emergency state
            emergency_controls.state = EmergencyState.EMERGENCY
            assert emergency_controls.is_trading_allowed() is False

            # Activate manual override
            await emergency_controls.activate_manual_override("admin", "Emergency override")

            assert emergency_controls.state == EmergencyState.MANUAL_OVERRIDE
            assert emergency_controls.is_trading_allowed() is True
            assert emergency_controls.manual_override_user == "admin"

            # Deactivate manual override
            await emergency_controls.deactivate_manual_override("admin")

            assert emergency_controls.state == EmergencyState.NORMAL
            assert emergency_controls.is_trading_allowed() is True
            assert emergency_controls.manual_override_user is None
