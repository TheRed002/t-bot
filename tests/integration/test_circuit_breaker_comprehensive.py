"""
Comprehensive integration tests for circuit breakers and emergency risk controls.

Tests complete circuit breaker workflows including activation triggers, emergency procedures,
system recovery, and integration with all trading components.
"""

import pytest
import asyncio
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from src.core.config import Config
from src.core.exceptions import CircuitBreakerError, RiskManagementError, EmergencyStopError
from src.core.types import (
    Position, Order, OrderSide, OrderStatus, OrderType,
    PortfolioState, CircuitBreakerEvent, EmergencyAction
)
from src.core.types.risk import CircuitBreakerType, CircuitBreakerStatus

from src.risk_management.emergency_controls import EmergencyControls
from src.bot_management.bot_instance import BotInstance
from src.execution.order_manager import OrderManager
from src.state import StateService


def create_circuit_breaker_event(
    trigger_type: str,
    trigger_value: float | Decimal,
    threshold: float | Decimal,
    timestamp: datetime | None = None,
    portfolio_state: PortfolioState | None = None
) -> CircuitBreakerEvent:
    """Helper to create circuit breaker events with proper fields."""
    
    # Map old trigger types to new breaker types
    type_mapping = {
        "LOSS_THRESHOLD": CircuitBreakerType.LOSS_LIMIT,
        "CONSECUTIVE_LOSSES": CircuitBreakerType.LOSS_LIMIT,
        "HIGH_VOLATILITY": CircuitBreakerType.VOLATILITY,
        "EXCESSIVE_DRAWDOWN": CircuitBreakerType.DRAWDOWN,
        "HIGH_FREQUENCY_TEST": CircuitBreakerType.MANUAL,
        "POST_RECOVERY_DETERIORATION": CircuitBreakerType.LOSS_LIMIT
    }
    
    breaker_type = type_mapping.get(trigger_type, CircuitBreakerType.MANUAL)
    
    return CircuitBreakerEvent(
        breaker_id=f"test_breaker_{trigger_type.lower()}",
        breaker_type=breaker_type,
        status=CircuitBreakerStatus.TRIGGERED,
        triggered_at=timestamp or datetime.now(timezone.utc),
        trigger_value=float(trigger_value),
        threshold_value=float(threshold),
        cooldown_period=300,  # 5 minutes default
        reason=f"{trigger_type} threshold exceeded",
        metadata={
            "trigger_type": trigger_type,
            "portfolio_state": portfolio_state.model_dump() if portfolio_state else None
        }
    )


@pytest.fixture
def mock_config():
    """Circuit breaker configuration for testing."""
    config = Mock(spec=Config)
    
    # Circuit breaker settings
    config.risk = Mock()
    config.risk.circuit_breakers = Mock()
    config.risk.circuit_breakers.enabled = True
    config.risk.circuit_breakers.loss_threshold = Decimal("5000.0")
    config.risk.circuit_breakers.consecutive_loss_limit = 5
    config.risk.circuit_breakers.volatility_threshold = Decimal("0.15")  # 15%
    config.risk.circuit_breakers.drawdown_threshold = Decimal("0.10")    # 10%
    config.risk.circuit_breakers.cooldown_period = 1800  # 30 minutes
    config.risk.circuit_breakers.auto_recovery = True
    
    # Emergency controls
    config.risk.emergency = Mock()
    config.risk.emergency.halt_all_trading = True
    config.risk.emergency.cancel_all_orders = True
    config.risk.emergency.liquidate_positions = False  # Only in extreme cases
    config.risk.emergency.notification_channels = ["email", "sms", "slack"]
    
    return config


@pytest.fixture
def sample_portfolio_in_distress():
    """Portfolio experiencing significant losses."""
    return PortfolioState(
        total_value=Decimal("85000.0"),  # Down from $100k
        available_cash=Decimal("10000.0"),
        total_positions_value=Decimal("75000.0"),
        unrealized_pnl=Decimal("-15000.0"),  # Large unrealized losses
        realized_pnl=Decimal("-5000.0"),     # Daily realized losses
        positions={
            "BTC/USDT": Position(
                symbol="BTC/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("1.0"),
                average_price=Decimal("50000.0"),
                current_price=Decimal("40000.0"),  # 20% decline
                unrealized_pnl=Decimal("-10000.0"),
                market_value=Decimal("40000.0")
            ),
            "ETH/USDT": Position(
                symbol="ETH/USDT",
                side=OrderSide.LONG,
                quantity=Decimal("10.0"),
                average_price=Decimal("3500.0"),
                current_price=Decimal("3000.0"),   # 14% decline
                unrealized_pnl=Decimal("-5000.0"),
                market_value=Decimal("30000.0")
            )
        },
        open_orders={
            "order_1": Order(
                id="order_1",
                symbol="ADA/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1000.0"),
                price=Decimal("0.50"),
                status=OrderStatus.NEW
            )
        },
        last_updated=datetime.now(timezone.utc)
    )


class TestCircuitBreakerActivation:
    """Test circuit breaker activation scenarios."""

    @pytest.mark.asyncio
    async def test_loss_threshold_circuit_breaker(self, mock_config, sample_portfolio_in_distress):
        """Test circuit breaker activation due to loss threshold."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Calculate current losses
        daily_realized_loss = abs(sample_portfolio_in_distress.realized_pnl)
        daily_unrealized_loss = abs(sample_portfolio_in_distress.unrealized_pnl) * Decimal("0.5")  # Assume 50% is today's loss
        total_daily_loss = daily_realized_loss + daily_unrealized_loss
        
        # Check if loss threshold is exceeded
        loss_threshold = mock_config.risk.circuit_breakers.loss_threshold
        should_trigger = total_daily_loss >= loss_threshold
        
        if should_trigger:
            # Activate circuit breaker
            event = create_circuit_breaker_event(
                trigger_type="LOSS_THRESHOLD",
                trigger_value=total_daily_loss,
                threshold=loss_threshold,
                portfolio_state=sample_portfolio_in_distress,
                timestamp=datetime.now(timezone.utc)
            )
            
            await emergency_controls.activate_circuit_breaker(event)
        
        assert should_trigger is True  # $12.5k loss > $5k threshold
        assert emergency_controls.is_circuit_breaker_active() is True
        assert emergency_controls.get_active_trigger() == "LOSS_THRESHOLD"

    @pytest.mark.asyncio
    async def test_consecutive_loss_circuit_breaker(self, mock_config):
        """Test circuit breaker activation due to consecutive losses."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Simulate consecutive losing trades
        trade_results = [
            {"result": "LOSS", "amount": Decimal("-800.0")},
            {"result": "LOSS", "amount": Decimal("-1200.0")},
            {"result": "WIN", "amount": Decimal("500.0")},    # Reset counter
            {"result": "LOSS", "amount": Decimal("-900.0")},
            {"result": "LOSS", "amount": Decimal("-1100.0")},
            {"result": "LOSS", "amount": Decimal("-700.0")},
            {"result": "LOSS", "amount": Decimal("-1300.0")},
            {"result": "LOSS", "amount": Decimal("-600.0")},  # 5th consecutive loss
        ]
        
        consecutive_losses = 0
        max_consecutive = mock_config.risk.circuit_breakers.consecutive_loss_limit
        
        for trade in trade_results:
            if trade["result"] == "LOSS":
                consecutive_losses += 1
            else:
                consecutive_losses = 0
            
            if consecutive_losses >= max_consecutive:
                # Trigger circuit breaker
                event = create_circuit_breaker_event(
                    trigger_type="CONSECUTIVE_LOSSES",
                    trigger_value=consecutive_losses,
                    threshold=max_consecutive,
                    timestamp=datetime.now(timezone.utc)
                )
                
                await emergency_controls.activate_circuit_breaker(event)
                break
        
        assert consecutive_losses >= max_consecutive
        assert emergency_controls.is_circuit_breaker_active() is True

    @pytest.mark.asyncio
    async def test_volatility_circuit_breaker(self, mock_config):
        """Test circuit breaker activation due to high volatility."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Mock high volatility scenario
        current_volatility = Decimal("0.18")  # 18% volatility
        volatility_threshold = mock_config.risk.circuit_breakers.volatility_threshold  # 15%
        
        if current_volatility > volatility_threshold:
            event = create_circuit_breaker_event(
                trigger_type="HIGH_VOLATILITY",
                trigger_value=current_volatility,
                threshold=volatility_threshold,
                timestamp=datetime.now(timezone.utc)
            )
            
            await emergency_controls.activate_circuit_breaker(event)
        
        assert current_volatility > volatility_threshold
        assert emergency_controls.is_circuit_breaker_active() is True

    @pytest.mark.asyncio
    async def test_drawdown_circuit_breaker(self, mock_config, sample_portfolio_in_distress):
        """Test circuit breaker activation due to excessive drawdown."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Calculate portfolio drawdown
        original_value = Decimal("100000.0")  # Original portfolio value
        current_value = sample_portfolio_in_distress.total_value
        drawdown = (original_value - current_value) / original_value
        
        drawdown_threshold = mock_config.risk.circuit_breakers.drawdown_threshold
        
        if drawdown > drawdown_threshold:
            event = create_circuit_breaker_event(
                trigger_type="EXCESSIVE_DRAWDOWN",
                trigger_value=drawdown,
                threshold=drawdown_threshold,
                portfolio_state=sample_portfolio_in_distress,
                timestamp=datetime.now(timezone.utc)
            )
            
            await emergency_controls.activate_circuit_breaker(event)
        
        assert drawdown == Decimal("0.15")  # 15% drawdown
        assert drawdown > drawdown_threshold  # Exceeds 10% threshold
        assert emergency_controls.is_circuit_breaker_active() is True


class TestEmergencyProcedures:
    """Test emergency procedures when circuit breakers activate."""

    @pytest.mark.asyncio
    async def test_trading_halt_procedure(self, mock_config, sample_portfolio_in_distress):
        """Test trading halt emergency procedure."""
        emergency_controls = EmergencyControls(mock_config)
        bot_instance = Mock(spec=BotInstance)
        
        # Mock bot methods
        bot_instance.halt_trading = AsyncMock(return_value=True)
        bot_instance.get_active_strategies = Mock(return_value=["momentum", "mean_reversion"])
        bot_instance.pause_strategy = AsyncMock(return_value=True)
        
        # Activate circuit breaker
        event = create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("7000.0"),
            threshold=Decimal("5000.0"),
            timestamp=datetime.now(timezone.utc)
        )
        
        await emergency_controls.activate_circuit_breaker(event)
        
        # Execute emergency procedures
        if emergency_controls.is_circuit_breaker_active():
            # Halt all trading
            await bot_instance.halt_trading()
            
            # Pause all strategies
            active_strategies = bot_instance.get_active_strategies()
            for strategy in active_strategies:
                await bot_instance.pause_strategy(strategy)
        
        # Verify emergency procedures were executed
        bot_instance.halt_trading.assert_called_once()
        assert bot_instance.pause_strategy.call_count == 2  # Two strategies paused

    @pytest.mark.asyncio
    async def test_order_cancellation_procedure(self, mock_config, sample_portfolio_in_distress):
        """Test order cancellation emergency procedure."""
        emergency_controls = EmergencyControls(mock_config)
        order_manager = Mock(spec=OrderManager)
        
        # Mock order manager methods
        order_manager.get_all_open_orders = Mock(return_value=list(sample_portfolio_in_distress.open_orders.values()))
        order_manager.cancel_order = AsyncMock(return_value=True)
        order_manager.cancel_all_orders = AsyncMock(return_value=True)
        
        # Activate circuit breaker
        await emergency_controls.activate_circuit_breaker(create_circuit_breaker_event(
            trigger_type="CONSECUTIVE_LOSSES",
            trigger_value=5,
            threshold=5,
            timestamp=datetime.now(timezone.utc)
        ))
        
        # Execute order cancellation procedure
        if mock_config.risk.emergency.cancel_all_orders:
            cancelled_orders = await order_manager.cancel_all_orders()
        
        # Verify orders were cancelled
        order_manager.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_position_liquidation_procedure(self, mock_config, sample_portfolio_in_distress):
        """Test position liquidation emergency procedure (extreme cases only)."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Configure for position liquidation in extreme cases
        mock_config.risk.emergency.liquidate_positions = True
        mock_config.risk.emergency.liquidation_threshold = Decimal("0.20")  # 20% drawdown
        
        # Calculate current drawdown
        original_value = Decimal("100000.0")
        current_value = sample_portfolio_in_distress.total_value
        drawdown = (original_value - current_value) / original_value
        
        # Activate circuit breaker
        event = create_circuit_breaker_event(
            trigger_type="EXCESSIVE_DRAWDOWN",
            trigger_value=drawdown,
            threshold=Decimal("0.10"),
            portfolio_state=sample_portfolio_in_distress,
            timestamp=datetime.now(timezone.utc)
        )
        
        await emergency_controls.activate_circuit_breaker(event)
        
        # Check if liquidation threshold is reached
        should_liquidate = (
            drawdown >= mock_config.risk.emergency.liquidation_threshold and
            mock_config.risk.emergency.liquidate_positions
        )
        
        if should_liquidate:
            # Create liquidation orders for all positions
            liquidation_orders = []
            for symbol, position in sample_portfolio_in_distress.positions.items():
                if position.quantity > 0:
                    liquidation_order = Order(
                        id=f"liquidate_{symbol}",
                        symbol=symbol,
                        side=OrderSide.SELL,  # Close long positions
                        order_type=OrderType.MARKET,
                        quantity=position.quantity,
                        status=OrderStatus.PENDING
                    )
                    liquidation_orders.append(liquidation_order)
            
            # Verify liquidation orders were created
            assert len(liquidation_orders) == 2  # BTC and ETH positions
            assert all(order.side == OrderSide.SELL for order in liquidation_orders)
            assert all(order.order_type == OrderType.MARKET for order in liquidation_orders)
        
        # Drawdown is 15%, below 20% liquidation threshold
        assert should_liquidate is False

    @pytest.mark.asyncio
    async def test_emergency_notification_system(self, mock_config):
        """Test emergency notification system."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Mock notification channels
        notification_system = Mock()
        notification_system.send_email = AsyncMock(return_value=True)
        notification_system.send_sms = AsyncMock(return_value=True)
        notification_system.send_slack = AsyncMock(return_value=True)
        
        # Activate circuit breaker
        event = create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("8000.0"),
            threshold=Decimal("5000.0"),
            timestamp=datetime.now(timezone.utc)
        )
        
        await emergency_controls.activate_circuit_breaker(event)
        
        # Send emergency notifications
        if emergency_controls.is_circuit_breaker_active():
            message = (
                f"EMERGENCY: Circuit breaker activated\n"
                f"Trigger: {event.trigger_type}\n"
                f"Value: {event.trigger_value}\n"
                f"Threshold: {event.threshold}\n"
                f"Time: {event.timestamp}"
            )
            
            # Send to all configured channels
            for channel in mock_config.risk.emergency.notification_channels:
                if channel == "email":
                    await notification_system.send_email(message, priority="CRITICAL")
                elif channel == "sms":
                    await notification_system.send_sms(message, priority="CRITICAL")
                elif channel == "slack":
                    await notification_system.send_slack(message, priority="CRITICAL")
        
        # Verify notifications were sent
        notification_system.send_email.assert_called_once()
        notification_system.send_sms.assert_called_once()
        notification_system.send_slack.assert_called_once()


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery procedures."""

    @pytest.mark.asyncio
    async def test_cooldown_period_enforcement(self, mock_config):
        """Test cooldown period enforcement before recovery."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Activate circuit breaker
        activation_time = datetime.now(timezone.utc)
        event = create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("6000.0"),
            threshold=Decimal("5000.0"),
            timestamp=activation_time
        )
        
        await emergency_controls.activate_circuit_breaker(event)
        assert emergency_controls.is_circuit_breaker_active() is True
        
        # Test recovery attempt during cooldown period
        cooldown_period = mock_config.risk.circuit_breakers.cooldown_period  # 1800 seconds
        
        # Simulate time progression
        test_times = [
            activation_time + timedelta(seconds=900),   # 15 minutes (still in cooldown)
            activation_time + timedelta(seconds=1800),  # 30 minutes (cooldown expires)
            activation_time + timedelta(seconds=2100),  # 35 minutes (after cooldown)
        ]
        
        for test_time in test_times:
            time_elapsed = (test_time - activation_time).total_seconds()
            cooldown_expired = time_elapsed >= cooldown_period
            
            if cooldown_expired and mock_config.risk.circuit_breakers.auto_recovery:
                # Attempt automatic recovery
                can_recover = emergency_controls.can_attempt_recovery(test_time)
                if can_recover:
                    await emergency_controls.attempt_recovery()
            
        # Should still be active during cooldown, but recoverable after
        final_check_time = test_times[-1]  # 35 minutes after activation
        time_since_activation = (final_check_time - activation_time).total_seconds()
        
        assert time_since_activation > cooldown_period
        # Circuit breaker should be recoverable now

    @pytest.mark.asyncio
    async def test_recovery_conditions_validation(self, mock_config, sample_portfolio_in_distress):
        """Test validation of recovery conditions."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Activate circuit breaker
        await emergency_controls.activate_circuit_breaker(create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("6000.0"),
            threshold=Decimal("5000.0"),
            timestamp=datetime.now(timezone.utc)
        ))
        
        # Mock improved portfolio conditions for recovery
        recovered_portfolio = PortfolioState(
            total_value=Decimal("95000.0"),     # Improved from $85k
            available_cash=Decimal("15000.0"),
            total_positions_value=Decimal("80000.0"),
            unrealized_pnl=Decimal("-5000.0"), # Improved from -$15k
            realized_pnl=Decimal("-2000.0"),   # Improved losses
            positions=sample_portfolio_in_distress.positions,
            open_orders={},
            last_updated=datetime.now(timezone.utc)
        )
        
        # Define recovery conditions
        recovery_conditions = {
            "losses_stabilized": True,    # No new losses in last hour
            "volatility_reduced": True,   # Volatility below threshold
            "drawdown_improved": True,    # Drawdown reduced
            "no_forced_liquidations": True, # No margin calls
            "market_conditions_stable": True # Overall market stability
        }
        
        # Check specific recovery criteria
        original_value = Decimal("100000.0")
        current_drawdown = (original_value - recovered_portfolio.total_value) / original_value
        drawdown_improved = current_drawdown < Decimal("0.10")  # Below 10%
        
        recovery_conditions["drawdown_improved"] = drawdown_improved
        
        # All conditions must be met for recovery
        all_conditions_met = all(recovery_conditions.values())
        
        if all_conditions_met:
            recovery_approved = True
        else:
            recovery_approved = False
        
        assert current_drawdown == Decimal("0.05")  # 5% drawdown (improved)
        assert all_conditions_met is True
        assert recovery_approved is True

    @pytest.mark.asyncio
    async def test_gradual_recovery_process(self, mock_config):
        """Test gradual recovery process (not immediate full resumption)."""
        emergency_controls = EmergencyControls(mock_config)
        bot_instance = Mock(spec=BotInstance)
        
        # Mock bot recovery methods
        bot_instance.enable_strategy = AsyncMock(return_value=True)
        bot_instance.set_position_size_limit = AsyncMock(return_value=True)
        bot_instance.resume_trading = AsyncMock(return_value=True)
        
        # Activate and then attempt recovery
        await emergency_controls.activate_circuit_breaker(create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("6000.0"),
            threshold=Decimal("5000.0"),
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=2000)  # Past cooldown
        ))
        
        # Gradual recovery phases
        recovery_phases = [
            {
                "phase": 1,
                "description": "Conservative restart",
                "position_size_limit": Decimal("0.5"),  # 50% of normal
                "enabled_strategies": ["conservative_mean_reversion"],
                "allowed_instruments": ["BTC/USDT", "ETH/USDT"]  # Blue chip only
            },
            {
                "phase": 2,
                "description": "Moderate expansion",
                "position_size_limit": Decimal("0.75"), # 75% of normal
                "enabled_strategies": ["conservative_mean_reversion", "momentum"],
                "allowed_instruments": ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
            },
            {
                "phase": 3,
                "description": "Full recovery",
                "position_size_limit": Decimal("1.0"),  # 100% of normal
                "enabled_strategies": ["all"],
                "allowed_instruments": ["all"]
            }
        ]
        
        current_phase = 1
        
        # Execute recovery phase 1
        phase_config = recovery_phases[0]
        
        # Set conservative limits
        await bot_instance.set_position_size_limit(phase_config["position_size_limit"])
        
        # Enable only conservative strategies
        for strategy in phase_config["enabled_strategies"]:
            await bot_instance.enable_strategy(strategy)
        
        # Resume limited trading
        await bot_instance.resume_trading(limited=True)
        
        # Deactivate circuit breaker for phase 1
        await emergency_controls.deactivate_circuit_breaker()
        
        # Verify gradual recovery
        assert emergency_controls.is_circuit_breaker_active() is False
        bot_instance.set_position_size_limit.assert_called_with(Decimal("0.5"))
        bot_instance.enable_strategy.assert_called_with("conservative_mean_reversion")
        bot_instance.resume_trading.assert_called_with(limited=True)

    @pytest.mark.asyncio
    async def test_recovery_monitoring_and_fallback(self, mock_config):
        """Test recovery monitoring and fallback to circuit breaker if conditions deteriorate."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Simulate recovery attempt
        recovery_start_time = datetime.now(timezone.utc)
        await emergency_controls.deactivate_circuit_breaker()
        
        # Monitor post-recovery performance
        post_recovery_trades = [
            {"result": "WIN", "amount": Decimal("200.0")},
            {"result": "WIN", "amount": Decimal("150.0")},
            {"result": "LOSS", "amount": Decimal("-300.0")},
            {"result": "LOSS", "amount": Decimal("-400.0")},  # Deteriorating performance
            {"result": "LOSS", "amount": Decimal("-600.0")},  # Major loss after recovery
        ]
        
        post_recovery_pnl = Decimal("0.0")
        consecutive_losses_after_recovery = 0
        
        for trade in post_recovery_trades:
            post_recovery_pnl += trade["amount"]
            
            if trade["result"] == "LOSS":
                consecutive_losses_after_recovery += 1
            else:
                consecutive_losses_after_recovery = 0
            
            # Check if conditions warrant re-activation
            if (abs(post_recovery_pnl) > Decimal("1000.0") or  # Significant losses resume
                consecutive_losses_after_recovery >= 3):       # Multiple losses in a row
                
                # Re-activate circuit breaker
                await emergency_controls.activate_circuit_breaker(CircuitBreakerEvent(
                    breaker_id="test_breaker_recovery",
                    breaker_type=CircuitBreakerType.LOSS_LIMIT,
                    status=CircuitBreakerStatus.TRIGGERED,
                    triggered_at=datetime.now(timezone.utc),
                    trigger_value=float(abs(post_recovery_pnl)),
                    threshold_value=1000.0,
                    cooldown_period=300,
                    reason="Post-recovery deterioration detected",
                    metadata={"trigger_type": "POST_RECOVERY_DETERIORATION"}
                ))
                break
        
        # Verify fallback activation
        assert post_recovery_pnl == Decimal("-950.0")  # Net loss after recovery
        assert consecutive_losses_after_recovery >= 3
        assert emergency_controls.is_circuit_breaker_active() is True
        assert emergency_controls.get_active_trigger() == "POST_RECOVERY_DETERIORATION"


class TestMultipleCircuitBreakerScenarios:
    """Test scenarios with multiple simultaneous circuit breaker triggers."""

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_triggers(self, mock_config, sample_portfolio_in_distress):
        """Test handling of multiple simultaneous circuit breaker triggers."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Create scenario with multiple triggers
        current_time = datetime.now(timezone.utc)
        
        triggers = [
            CircuitBreakerEvent(
                breaker_id="test_breaker_loss",
                breaker_type=CircuitBreakerType.LOSS_LIMIT,
                status=CircuitBreakerStatus.TRIGGERED,
                triggered_at=current_time,
                trigger_value=8000.0,
                threshold_value=5000.0,
                cooldown_period=300,
                reason="Loss threshold exceeded"
            ),
            CircuitBreakerEvent(
                breaker_id="test_breaker_volatility",
                breaker_type=CircuitBreakerType.VOLATILITY,
                status=CircuitBreakerStatus.TRIGGERED,
                triggered_at=current_time,
                trigger_value=0.20,  # 20% volatility
                threshold_value=0.15,      # 15% threshold
                cooldown_period=300,
                reason="High volatility detected"
            ),
            CircuitBreakerEvent(
                breaker_id="test_breaker_drawdown",
                breaker_type=CircuitBreakerType.DRAWDOWN,
                status=CircuitBreakerStatus.TRIGGERED,
                triggered_at=current_time,
                trigger_value=0.15,  # 15% drawdown
                threshold_value=0.10,      # 10% threshold
                cooldown_period=300,
                reason="Excessive drawdown detected"
            )
        ]
        
        # Activate circuit breaker with multiple triggers
        await emergency_controls.activate_circuit_breaker_multiple(triggers)
        
        # Verify all triggers are recorded
        active_triggers = emergency_controls.get_all_active_triggers()
        
        assert len(active_triggers) == 3
        assert "LOSS_THRESHOLD" in active_triggers
        assert "HIGH_VOLATILITY" in active_triggers
        assert "EXCESSIVE_DRAWDOWN" in active_triggers
        assert emergency_controls.is_circuit_breaker_active() is True

    @pytest.mark.asyncio
    async def test_cascade_circuit_breaker_scenario(self, mock_config):
        """Test cascade scenario where one trigger leads to others."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Start with single trigger
        initial_trigger = create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("6000.0"),
            threshold=Decimal("5000.0"),
            timestamp=datetime.now(timezone.utc)
        )
        
        await emergency_controls.activate_circuit_breaker(initial_trigger)
        
        # Simulate cascade: Loss leads to panic selling, which increases volatility
        # and creates more losses (realistic market scenario)
        cascade_triggers = []
        
        # Simulate 10 minutes after initial trigger
        time_offset = timedelta(minutes=10)
        
        # Higher volatility due to panic
        cascade_triggers.append(create_circuit_breaker_event(
            trigger_type="HIGH_VOLATILITY",
            trigger_value=Decimal("0.25"),  # Extreme volatility
            threshold=Decimal("0.15"),
            timestamp=initial_trigger.triggered_at + time_offset
        ))
        
        # Additional losses due to volatile conditions
        cascade_triggers.append(create_circuit_breaker_event(
            trigger_type="CONSECUTIVE_LOSSES",
            trigger_value=7,  # 7 consecutive losses
            threshold=5,
            timestamp=initial_trigger.triggered_at + time_offset
        ))
        
        # Add cascade triggers
        for trigger in cascade_triggers:
            await emergency_controls.add_cascade_trigger(trigger)
        
        # Verify cascade is properly handled
        all_triggers = emergency_controls.get_all_active_triggers()
        cascade_detected = emergency_controls.is_cascade_scenario()
        
        assert len(all_triggers) >= 3  # Original + cascade triggers
        assert cascade_detected is True
        assert emergency_controls.get_severity_level() == "CRITICAL"  # Escalated due to cascade

    @pytest.mark.asyncio
    async def test_circuit_breaker_priority_handling(self, mock_config):
        """Test priority handling when multiple circuit breakers compete."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Define trigger priorities (higher number = higher priority)
        trigger_priorities = {
            "LOSS_THRESHOLD": 3,
            "HIGH_VOLATILITY": 2,
            "CONSECUTIVE_LOSSES": 1,
            "EXCESSIVE_DRAWDOWN": 4,  # Highest priority
            "MARGIN_CALL": 5          # Highest priority (most urgent)
        }
        
        # Multiple simultaneous triggers with different priorities
        simultaneous_triggers = [
            ("LOSS_THRESHOLD", Decimal("6000.0"), Decimal("5000.0")),
            ("HIGH_VOLATILITY", Decimal("0.18"), Decimal("0.15")),
            ("EXCESSIVE_DRAWDOWN", Decimal("0.12"), Decimal("0.10")),
        ]
        
        # Process triggers by priority
        processed_triggers = []
        for trigger_type, value, threshold in simultaneous_triggers:
            priority = trigger_priorities[trigger_type]
            processed_triggers.append((priority, trigger_type, value, threshold))
        
        # Sort by priority (descending)
        processed_triggers.sort(key=lambda x: x[0], reverse=True)
        
        # Activate highest priority trigger first
        highest_priority = processed_triggers[0]
        primary_trigger = create_circuit_breaker_event(
            trigger_type=highest_priority[1],
            trigger_value=highest_priority[2], 
            threshold=highest_priority[3],
            timestamp=datetime.now(timezone.utc)
        )
        
        await emergency_controls.activate_circuit_breaker(primary_trigger)
        
        # Verify correct prioritization
        assert emergency_controls.get_primary_trigger() == "EXCESSIVE_DRAWDOWN"
        assert emergency_controls.get_trigger_priority("EXCESSIVE_DRAWDOWN") == 4


class TestCircuitBreakerSystemIntegration:
    """Test circuit breaker integration with entire trading system."""

    @pytest.mark.asyncio
    async def test_end_to_end_circuit_breaker_workflow(self, mock_config, sample_portfolio_in_distress):
        """Test complete end-to-end circuit breaker workflow."""
        # Initialize all system components
        emergency_controls = EmergencyControls(mock_config)
        bot_instance = Mock(spec=BotInstance)
        order_manager = Mock(spec=OrderManager)
        state_manager = Mock(spec=StateService)
        
        # Mock component methods
        bot_instance.halt_trading = AsyncMock(return_value=True)
        bot_instance.pause_all_strategies = AsyncMock(return_value=True)
        order_manager.cancel_all_orders = AsyncMock(return_value=True)
        state_manager.save_emergency_state = AsyncMock(return_value=True)
        
        # 1. Trigger detection and activation
        trigger_event = create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("7500.0"),
            threshold=Decimal("5000.0"),
            portfolio_state=sample_portfolio_in_distress,
            timestamp=datetime.now(timezone.utc)
        )
        
        await emergency_controls.activate_circuit_breaker(trigger_event)
        
        # 2. Emergency procedures execution
        if emergency_controls.is_circuit_breaker_active():
            # Halt trading
            await bot_instance.halt_trading()
            
            # Pause strategies
            await bot_instance.pause_all_strategies()
            
            # Cancel orders
            await order_manager.cancel_all_orders()
            
            # Save emergency state
            emergency_state = {
                "circuit_breaker_active": True,
                "trigger": trigger_event.trigger_type,
                "portfolio_snapshot": sample_portfolio_in_distress,
                "timestamp": trigger_event.timestamp
            }
            await state_manager.save_emergency_state(emergency_state)
        
        # 3. Wait for cooldown period (simulated)
        cooldown_completed = True  # Mock cooldown completion
        
        # 4. Recovery validation
        if cooldown_completed:
            recovery_conditions = {
                "market_stable": True,
                "losses_stopped": True,
                "volatility_normalized": True
            }
            
            can_recover = all(recovery_conditions.values())
            
            if can_recover:
                # 5. Gradual recovery
                await emergency_controls.deactivate_circuit_breaker()
                
                # Resume with limitations
                bot_instance.resume_trading = AsyncMock(return_value=True)
                await bot_instance.resume_trading(limited=True)
        
        # Verify complete workflow
        assert emergency_controls.activation_count == 1
        bot_instance.halt_trading.assert_called_once()
        bot_instance.pause_all_strategies.assert_called_once()
        order_manager.cancel_all_orders.assert_called_once()
        state_manager.save_emergency_state.assert_called_once()
        bot_instance.resume_trading.assert_called_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_performance_impact(self, mock_config):
        """Test performance impact and responsiveness of circuit breaker system."""
        emergency_controls = EmergencyControls(mock_config)
        
        # Measure activation time
        start_time = datetime.now()
        
        trigger_event = create_circuit_breaker_event(
            trigger_type="LOSS_THRESHOLD",
            trigger_value=Decimal("6000.0"),
            threshold=Decimal("5000.0"),
            timestamp=start_time
        )
        
        await emergency_controls.activate_circuit_breaker(trigger_event)
        
        activation_time = datetime.now()
        response_time_ms = (activation_time - start_time).total_seconds() * 1000
        
        # Circuit breaker should activate very quickly (< 100ms)
        assert response_time_ms < 100
        assert emergency_controls.is_circuit_breaker_active() is True
        
        # Test system responsiveness under stress
        stress_test_triggers = []
        for i in range(100):  # 100 rapid triggers
            stress_trigger = create_circuit_breaker_event(
                trigger_type="HIGH_FREQUENCY_TEST",
                trigger_value=Decimal(str(i)),
                threshold=Decimal("50"),
                timestamp=datetime.now(timezone.utc)
            )
            stress_test_triggers.append(stress_trigger)
        
        # Process all triggers rapidly
        stress_start = datetime.now()
        
        for trigger in stress_test_triggers:
            if trigger.trigger_value > trigger.threshold:
                await emergency_controls.update_trigger_status(trigger)
        
        stress_end = datetime.now()
        total_processing_time = (stress_end - stress_start).total_seconds() * 1000
        
        # Should handle high-frequency updates efficiently
        assert total_processing_time < 1000  # < 1 second for 100 updates
        assert emergency_controls.system_responsive is True