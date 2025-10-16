"""
Integration tests for P-009 Circuit Breakers and Emergency Controls.

This module demonstrates the complete P-009 functionality working together:
- Circuit breaker triggering
- Emergency controls activation
- Position closure and order cancellation
- Recovery procedures
- Manual override capabilities

CRITICAL: This demonstrates the complete P-009 implementation working together.
NO MOCKS for internal services - only for external APIs (exchanges).
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.config import Config, RiskConfig
from src.core.exceptions import CircuitBreakerTriggeredError

# Import from P-001
from src.core.types import (
    CircuitBreakerType,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    RiskLevel,
    RiskMetrics,
    Signal,
)

# Import from P-003+
from src.exchanges.base import BaseExchange

# Import from P-008
from src.risk_management.base import BaseRiskManager
from src.risk_management.circuit_breakers import CircuitBreakerManager
from src.risk_management.emergency_controls import EmergencyControls, EmergencyState


class MinimalRiskManager(BaseRiskManager):
    """Minimal concrete RiskManager implementation for testing.

    This is NOT a mock - it's a real implementation with minimal functionality
    sufficient for circuit breaker and emergency controls testing.

    Note: Renamed from TestRiskManager to avoid pytest collection warning.
    """

    async def calculate_position_size(self, signal: Signal, portfolio_value: Decimal) -> Decimal:
        """Calculate position size (2% of portfolio by default)."""
        return portfolio_value * Decimal("0.02")

    async def validate_order(self, order: OrderRequest, portfolio_value: Decimal) -> bool:
        """Validate order (always returns True for tests)."""
        return True

    async def validate_signal(self, signal: Signal) -> bool:
        """Validate signal (always returns True for tests)."""
        return True

    async def calculate_risk_metrics(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> RiskMetrics:
        """Calculate basic risk metrics for testing."""
        return RiskMetrics(
            timestamp=datetime.now(),
            risk_level=RiskLevel.LOW,
            var_95=Decimal("100.0"),
            max_drawdown=Decimal("0.0"),
            sharpe_ratio=Decimal("1.0"),
            position_count=len(positions) if positions else 0,
            total_exposure=Decimal("0.0"),
            leverage=Decimal("1.0"),
        )

    async def check_portfolio_limits(self, new_position: Position) -> bool:
        """Check portfolio limits (always returns True for tests)."""
        return True

    async def should_exit_position(self, position: Position, market_data: MarketData) -> bool:
        """Check if position should be exited (always False for tests)."""
        return False

    async def update_portfolio_state(
        self, positions: list[Position], market_data: list[MarketData]
    ) -> None:
        """Update portfolio state (no-op for tests)."""
        pass


class TestP009Integration:
    """Integration tests for P-009 functionality."""

    @pytest.fixture
    def real_config(self):
        """Create REAL configuration object - NO MOCKS.

        Circuit breakers need real numeric values for comparisons like <=, >=.
        Mock objects will cause: "'<=' not supported between instances of 'Mock' and 'int'"

        The Config() already has all required fields with defaults.
        Circuit breakers use getattr(config.risk, 'field_name', default_value)
        so non-existent fields will use defaults automatically.
        """
        return Config()

    @pytest.fixture
    def real_risk_manager(self, real_config):
        """Create REAL risk manager - NO MOCKS.

        Circuit breakers need a real BaseRiskManager instance, not a Mock,
        to properly evaluate risk conditions.
        """
        return MinimalRiskManager(real_config)

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

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_complete_emergency_cycle(self, real_config, real_risk_manager, mock_exchange):
        """Test complete emergency cycle from circuit breaker to recovery."""
        # Initialize components with REAL config and risk manager - NO MOCKS
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)
        emergency_controls = EmergencyControls(
            real_config, real_risk_manager, circuit_breaker_manager
        )

        # Register exchange
        emergency_controls.register_exchange("binance", mock_exchange)

        # Initial state
        assert emergency_controls.state.value == "normal"
        assert circuit_breaker_manager.is_trading_allowed() is True

        # Simulate data that would trigger circuit breakers
        data = {
            "portfolio_value": Decimal("10000"),
            # 6% loss, should trigger daily loss breaker
            "daily_pnl": Decimal("-600"),
            "current_portfolio_value": Decimal("8000"),  # 20% drawdown
            "peak_portfolio_value": Decimal("10000"),
            "price_history": [100, 90, 110, 80, 120],  # High volatility
            "model_confidence": Decimal("0.2"),  # Low confidence
            "total_requests": 100,
            "error_occurred": False,
        }

        # This should trigger circuit breakers and emergency controls
        with pytest.raises(CircuitBreakerTriggeredError):
            await circuit_breaker_manager.evaluate_all(data)

        # Verify circuit breakers are triggered
        triggered_breakers = circuit_breaker_manager.get_triggered_breakers()
        assert len(triggered_breakers) > 0

        # Verify trading is not allowed
        assert circuit_breaker_manager.is_trading_allowed() is False

        # Simulate emergency controls activation
        await emergency_controls.activate_emergency_stop(
            "Circuit breakers triggered", CircuitBreakerType.DAILY_LOSS_LIMIT
        )

        assert emergency_controls.state.value == "emergency"
        assert emergency_controls.is_trading_allowed() is False

        # Simulate recovery
        await emergency_controls.deactivate_emergency_stop("Manual deactivation")
        assert emergency_controls.state.value == "recovery"

        # Complete recovery
        await emergency_controls.deactivate_emergency_stop("Recovery completed")
        assert emergency_controls.state.value == "normal"
        assert emergency_controls.is_trading_allowed() is True

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_emergency_with_positions_and_orders(
        self, real_config, real_risk_manager, mock_exchange
    ):
        """Test emergency procedures with actual positions and orders."""
        # Initialize components with REAL config and risk manager
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)
        emergency_controls = EmergencyControls(
            real_config, real_risk_manager, circuit_breaker_manager
        )

        # Register exchange
        emergency_controls.register_exchange("binance", mock_exchange)

        # Mock pending orders
        mock_orders = [Mock(id="order1", symbol="BTCUSDT"), Mock(id="order2", symbol="ETHUSDT")]
        mock_exchange.get_pending_orders.return_value = mock_orders

        # Mock positions
        mock_positions = [
            Mock(symbol="BTCUSDT", quantity=Decimal("1.0"), side=OrderSide.BUY),
            Mock(symbol="ETHUSDT", quantity=Decimal("10.0"), side=OrderSide.SELL),
        ]
        mock_exchange.get_positions.return_value = mock_positions

        # Activate emergency stop
        await emergency_controls.activate_emergency_stop(
            "Test emergency", CircuitBreakerType.DAILY_LOSS_LIMIT
        )

        # Verify orders were cancelled
        assert mock_exchange.cancel_order.call_count == 2
        mock_exchange.cancel_order.assert_any_call("order1")
        mock_exchange.cancel_order.assert_any_call("order2")

        # Verify positions were closed
        assert mock_exchange.place_order.call_count == 2

        # Verify emergency event was created
        assert len(emergency_controls.emergency_events) == 1
        event = emergency_controls.emergency_events[0]
        assert event.orders_cancelled == 2
        assert event.positions_affected == 2

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_manual_override_functionality(self, real_config, real_risk_manager):
        """Test manual override functionality."""
        # Initialize components with REAL config and risk manager
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)
        emergency_controls = EmergencyControls(
            real_config, real_risk_manager, circuit_breaker_manager
        )

        # Start in emergency state
        emergency_controls.state = EmergencyState.EMERGENCY
        assert emergency_controls.is_trading_allowed() is False

        # Activate manual override
        await emergency_controls.activate_manual_override("admin", "Emergency override")

        assert emergency_controls.state == EmergencyState.MANUAL_OVERRIDE
        assert emergency_controls.is_trading_allowed() is True
        assert emergency_controls.manual_override_user == "admin"

        # Test order validation during manual override
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        allowed = await emergency_controls.validate_order_during_emergency(order)
        assert allowed is True

        # Deactivate manual override
        await emergency_controls.deactivate_manual_override("admin")

        assert emergency_controls.state == EmergencyState.NORMAL
        assert emergency_controls.manual_override_user is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_circuit_breaker_recovery_cycle(self, real_config, real_risk_manager):
        """Test circuit breaker recovery cycle."""
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)

        # Get daily loss breaker
        daily_loss_breaker = circuit_breaker_manager.circuit_breakers["daily_loss_limit"]

        # Initial state
        assert daily_loss_breaker.state.value == "active"

        # Trigger circuit breaker
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-600"),  # 6% loss
        }

        with pytest.raises(CircuitBreakerTriggeredError):
            await daily_loss_breaker.evaluate(data)

        assert daily_loss_breaker.state.value == "triggered"
        assert daily_loss_breaker.trigger_count == 1

        # Simulate recovery timeout (use timezone-aware datetime to match circuit breaker code)
        daily_loss_breaker.trigger_time = datetime.now(timezone.utc) - timedelta(minutes=35)

        # Test recovery with good data
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("100"),  # Positive PnL
        }

        triggered = await daily_loss_breaker.evaluate(data)
        assert triggered is False
        assert daily_loss_breaker.state.value == "active"

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_multiple_circuit_breakers_coordination(self, real_config, real_risk_manager):
        """Test coordination between multiple circuit breakers."""
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)

        # Test data that should trigger multiple breakers
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-600"),  # 6% loss
            "current_portfolio_value": Decimal("8000"),  # 20% drawdown
            "peak_portfolio_value": Decimal("10000"),
            "price_history": [100, 90, 110, 80, 120],  # High volatility
            "model_confidence": Decimal("0.2"),  # Low confidence
            "total_requests": 100,
            "error_occurred": False,
        }

        # This should trigger multiple breakers
        with pytest.raises(CircuitBreakerTriggeredError):
            await circuit_breaker_manager.evaluate_all(data)

        # Check that multiple breakers are triggered
        triggered = circuit_breaker_manager.get_triggered_breakers()
        assert len(triggered) > 1  # Multiple breakers should be triggered

        # Verify trading is not allowed
        assert circuit_breaker_manager.is_trading_allowed() is False

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_emergency_controls_status_reporting(self, real_config, real_risk_manager):
        """Test emergency controls status reporting."""
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)
        emergency_controls = EmergencyControls(
            real_config, real_risk_manager, circuit_breaker_manager
        )

        # Test normal state status
        status = emergency_controls.get_status()
        assert status["state"] == "normal"
        assert status["trading_allowed"] is True

        # Activate emergency
        await emergency_controls.activate_emergency_stop(
            "Test", CircuitBreakerType.DAILY_LOSS_LIMIT
        )

        # Test emergency state status
        status = emergency_controls.get_status()
        assert status["state"] == "emergency"
        assert status["trading_allowed"] is False
        assert status["emergency_reason"] == "Test"

        # Test circuit breaker status
        cb_status = await circuit_breaker_manager.get_status()
        assert len(cb_status) == 6  # 6 circuit breakers (including correlation_spike)
        for breaker_name, breaker_status in cb_status.items():
            assert "state" in breaker_status
            assert "trigger_count" in breaker_status
            assert "events_count" in breaker_status


class TestP009RealWorldScenario:
    """Real-world scenario tests for P-009."""

    @pytest.fixture
    def real_config(self):
        """Create REAL configuration object - NO MOCKS.

        Use default Config with all necessary fields already defined.
        """
        return Config()

    @pytest.fixture
    def real_risk_manager(self, real_config):
        """Create REAL risk manager - NO MOCKS."""
        return MinimalRiskManager(real_config)

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_market_crash_scenario(self, real_config, real_risk_manager):
        """Test P-009 response to a simulated market crash scenario."""
        # Use REAL config and risk manager - NO MOCKS
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)
        emergency_controls = EmergencyControls(real_config, real_risk_manager, circuit_breaker_manager)

        # Simulate market crash data
        crash_data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-800"),  # 8% loss
            "current_portfolio_value": Decimal("7000"),  # 30% drawdown
            "peak_portfolio_value": Decimal("10000"),
            "price_history": [100, 80, 60, 40, 30],  # Extreme volatility
            "model_confidence": Decimal("0.1"),  # Very low confidence
            "total_requests": 100,
            "error_occurred": True,  # System errors
        }

        # This should trigger multiple circuit breakers
        with pytest.raises(CircuitBreakerTriggeredError):
            await circuit_breaker_manager.evaluate_all(crash_data)

        # Verify emergency controls activate
        await emergency_controls.activate_emergency_stop(
            "Market crash detected", CircuitBreakerType.DAILY_LOSS_LIMIT
        )

        assert emergency_controls.state.value == "emergency"
        assert emergency_controls.is_trading_allowed() is False

        # Verify circuit breakers are triggered
        triggered = circuit_breaker_manager.get_triggered_breakers()
        assert len(triggered) >= 3  # Should trigger multiple breakers

    @pytest.mark.asyncio
    @pytest.mark.timeout(10)
    async def test_system_failure_scenario(self, real_config, real_risk_manager):
        """Test P-009 response to system failure scenario."""
        # Use REAL config and risk manager - NO MOCKS
        circuit_breaker_manager = CircuitBreakerManager(real_config, real_risk_manager)
        emergency_controls = EmergencyControls(real_config, real_risk_manager, circuit_breaker_manager)

        # Simulate system failure data
        failure_data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("100"),  # Normal PnL
            "current_portfolio_value": Decimal("10000"),
            "peak_portfolio_value": Decimal("10000"),
            "price_history": [100, 101, 102],  # Normal volatility
            # Low confidence due to system issues
            "model_confidence": Decimal("0.1"),
            "total_requests": 50,
            "error_occurred": True,  # High error rate
        }

        # This should trigger system error rate breaker
        with pytest.raises(CircuitBreakerTriggeredError):
            await circuit_breaker_manager.evaluate_all(failure_data)

        # Verify emergency controls activate
        await emergency_controls.activate_emergency_stop(
            "System failure detected", CircuitBreakerType.SYSTEM_ERROR_RATE
        )

        assert emergency_controls.state.value == "emergency"
        assert emergency_controls.is_trading_allowed() is False
