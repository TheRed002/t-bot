"""
Unit tests for Circuit Breakers System (P-009).

This module tests the circuit breaker functionality including:
- Daily loss limit breaker
- Drawdown limit breaker
- Volatility spike breaker
- Model confidence breaker
- System error rate breaker
- Circuit breaker manager

CRITICAL: Tests must achieve 90% coverage for P-009 implementation.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import CircuitBreakerTriggeredError

# Import from P-001
# Import from P-008
from src.risk_management.base import BaseRiskManager
from src.risk_management.circuit_breakers import (
    CircuitBreakerManager,
    CircuitBreakerState,
    CorrelationSpikeBreaker,
    DailyLossLimitBreaker,
    DrawdownLimitBreaker,
    ModelConfidenceBreaker,
    SystemErrorRateBreaker,
    VolatilitySpikeBreaker,
)


class TestBaseCircuitBreaker:
    """Test base circuit breaker functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.max_daily_loss_pct = 0.05
        config.risk.max_drawdown_pct = 0.15
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def mock_error_handler(self):
        """Create mock error handler."""
        handler = Mock()
        handler.handle_error = AsyncMock()
        return handler

    @pytest.fixture
    def circuit_breaker(self, mock_config, mock_risk_manager):
        """Create test circuit breaker instance."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return DailyLossLimitBreaker(mock_config, mock_risk_manager)

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.trigger_count == 0
        assert circuit_breaker.trigger_time is None
        assert circuit_breaker.recovery_time is None
        assert len(circuit_breaker.events) == 0

    def test_circuit_breaker_reset(self, circuit_breaker):
        """Test circuit breaker reset functionality."""
        # Set some state
        circuit_breaker.state = CircuitBreakerState.OPEN
        circuit_breaker.trigger_count = 5
        circuit_breaker.trigger_time = datetime.now()
        circuit_breaker.events = [Mock()]

        # Reset
        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert circuit_breaker.trigger_count == 0
        assert circuit_breaker.trigger_time is None
        assert circuit_breaker.recovery_time is None
        assert len(circuit_breaker.events) == 0

    def test_get_status(self, circuit_breaker):
        """Test circuit breaker status retrieval."""
        status = circuit_breaker.get_status()

        assert "state" in status
        assert "trigger_count" in status
        assert "trigger_time" in status
        assert "recovery_time" in status
        assert "events_count" in status
        assert status["state"] == "closed"
        assert status["trigger_count"] == 0
        assert status["events_count"] == 0


class TestDailyLossLimitBreaker:
    """Test daily loss limit circuit breaker."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.max_daily_loss_pct = 0.05  # 5%
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def breaker(self, mock_config, mock_risk_manager):
        """Create daily loss limit breaker."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return DailyLossLimitBreaker(mock_config, mock_risk_manager)

    @pytest.mark.asyncio
    async def test_get_threshold_value(self, breaker):
        """Test threshold value retrieval."""
        threshold = await breaker.get_threshold_value()
        assert threshold == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_get_current_value_no_loss(self, breaker):
        """Test current value calculation with no loss."""
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("100"),  # Positive PnL
        }

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_current_value_with_loss(self, breaker):
        """Test current value calculation with loss."""
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-500"),  # Negative PnL
        }

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0.05")  # 5% loss

    @pytest.mark.asyncio
    async def test_get_current_value_zero_portfolio(self, breaker):
        """Test current value calculation with zero portfolio."""
        data = {"portfolio_value": Decimal("0"), "daily_pnl": Decimal("-100")}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_check_condition_no_trigger(self, breaker):
        """Test condition check when no trigger should occur."""
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-200"),  # 2% loss, below 5% threshold
        }

        triggered = await breaker.check_condition(data)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_check_condition_trigger(self, breaker):
        """Test condition check when trigger should occur."""
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-600"),  # 6% loss, above 5% threshold
        }

        triggered = await breaker.check_condition(data)
        assert triggered is True

    @pytest.mark.asyncio
    async def test_evaluate_no_trigger(self, breaker):
        """Test evaluation when no trigger occurs."""
        data = {"portfolio_value": Decimal("10000"), "daily_pnl": Decimal("-200")}

        triggered = await breaker.evaluate(data)
        assert triggered is False
        assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_evaluate_with_trigger(self, breaker):
        """Test evaluation when trigger occurs."""
        data = {"portfolio_value": Decimal("10000"), "daily_pnl": Decimal("-600")}

        with pytest.raises(CircuitBreakerTriggeredError):
            await breaker.evaluate(data)

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.trigger_count == 1
        assert len(breaker.events) == 1


class TestDrawdownLimitBreaker:
    """Test drawdown limit circuit breaker."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.max_drawdown_pct = 0.15  # 15%
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def breaker(self, mock_config, mock_risk_manager):
        """Create drawdown limit breaker."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return DrawdownLimitBreaker(mock_config, mock_risk_manager)

    @pytest.mark.asyncio
    async def test_get_threshold_value(self, breaker):
        """Test threshold value retrieval."""
        threshold = await breaker.get_threshold_value()
        assert threshold == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_get_current_value_no_drawdown(self, breaker):
        """Test current value calculation with no drawdown."""
        data = {
            "current_portfolio_value": Decimal("11000"),  # Higher than peak
            "peak_portfolio_value": Decimal("10000"),
        }

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_current_value_with_drawdown(self, breaker):
        """Test current value calculation with drawdown."""
        data = {
            "current_portfolio_value": Decimal("8500"),  # 15% below peak
            "peak_portfolio_value": Decimal("10000"),
        }

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0.15")  # 15% drawdown

    @pytest.mark.asyncio
    async def test_check_condition_no_trigger(self, breaker):
        """Test condition check when no trigger should occur."""
        data = {
            "current_portfolio_value": Decimal("9000"),  # 10% drawdown
            "peak_portfolio_value": Decimal("10000"),
        }

        triggered = await breaker.check_condition(data)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_check_condition_trigger(self, breaker):
        """Test condition check when trigger should occur."""
        data = {
            "current_portfolio_value": Decimal("8000"),  # 20% drawdown
            "peak_portfolio_value": Decimal("10000"),
        }

        triggered = await breaker.check_condition(data)
        assert triggered is True


class TestVolatilitySpikeBreaker:
    """Test volatility spike circuit breaker."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def breaker(self, mock_config, mock_risk_manager):
        """Create volatility spike breaker."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return VolatilitySpikeBreaker(mock_config, mock_risk_manager)

    @pytest.mark.asyncio
    async def test_get_threshold_value(self, breaker):
        """Test threshold value retrieval."""
        threshold = await breaker.get_threshold_value()
        assert threshold == Decimal("0.05")  # 5% volatility threshold

    @pytest.mark.asyncio
    async def test_get_current_value_insufficient_data(self, breaker):
        """Test current value calculation with insufficient data."""
        data = {"price_history": [100]}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_current_value_stable_prices(self, breaker):
        """Test current value calculation with stable prices."""
        data = {"price_history": [100, 101, 102, 103, 104]}  # Stable prices

        current_value = await breaker.get_current_value(data)
        assert current_value < Decimal("0.05")  # Low volatility

    @pytest.mark.asyncio
    async def test_get_current_value_volatile_prices(self, breaker):
        """Test current value calculation with volatile prices."""
        data = {"price_history": [100, 90, 110, 80, 120]}  # Volatile prices

        current_value = await breaker.get_current_value(data)
        assert current_value > Decimal("0.05")  # High volatility

    @pytest.mark.asyncio
    async def test_check_condition_no_trigger(self, breaker):
        """Test condition check when no trigger should occur."""
        data = {"price_history": [100, 101, 102, 103, 104]}

        triggered = await breaker.check_condition(data)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_check_condition_trigger(self, breaker):
        """Test condition check when trigger should occur."""
        data = {"price_history": [100, 90, 110, 80, 120]}

        triggered = await breaker.check_condition(data)
        assert triggered is True


class TestModelConfidenceBreaker:
    """Test model confidence circuit breaker."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def breaker(self, mock_config, mock_risk_manager):
        """Create model confidence breaker."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return ModelConfidenceBreaker(mock_config, mock_risk_manager)

    @pytest.mark.asyncio
    async def test_get_threshold_value(self, breaker):
        """Test threshold value retrieval."""
        threshold = await breaker.get_threshold_value()
        assert threshold == Decimal("0.3")  # 30% confidence threshold

    @pytest.mark.asyncio
    async def test_get_current_value(self, breaker):
        """Test current value calculation."""
        data = {"model_confidence": Decimal("0.8")}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0.8")

    @pytest.mark.asyncio
    async def test_get_current_value_default(self, breaker):
        """Test current value calculation with default."""
        data = {}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("1.0")  # Default confidence

    @pytest.mark.asyncio
    async def test_check_condition_no_trigger(self, breaker):
        """Test condition check when no trigger should occur."""
        data = {"model_confidence": Decimal("0.8")}  # High confidence

        triggered = await breaker.check_condition(data)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_check_condition_trigger(self, breaker):
        """Test condition check when trigger should occur."""
        data = {"model_confidence": Decimal("0.2")}  # Low confidence

        triggered = await breaker.check_condition(data)
        assert triggered is True


class TestSystemErrorRateBreaker:
    """Test system error rate circuit breaker."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def breaker(self, mock_config, mock_risk_manager):
        """Create system error rate breaker."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return SystemErrorRateBreaker(mock_config, mock_risk_manager)

    @pytest.mark.asyncio
    async def test_get_threshold_value(self, breaker):
        """Test threshold value retrieval."""
        threshold = await breaker.get_threshold_value()
        assert threshold == Decimal("0.1")  # 10% error rate threshold

    @pytest.mark.asyncio
    async def test_get_current_value_no_errors(self, breaker):
        """Test current value calculation with no errors."""
        data = {"total_requests": 100, "error_occurred": False}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_current_value_with_errors(self, breaker):
        """Test current value calculation with errors."""
        # Add some errors to the breaker
        breaker.error_times = [datetime.now() for _ in range(5)]

        data = {"total_requests": 50, "error_occurred": False}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0.1")  # 5/50 = 10%

    @pytest.mark.asyncio
    async def test_get_current_value_add_error(self, breaker):
        """Test current value calculation when adding error."""
        data = {"total_requests": 10, "error_occurred": True}

        current_value = await breaker.get_current_value(data)
        assert current_value == Decimal("0.1")  # 1/10 = 10%

    @pytest.mark.asyncio
    async def test_check_condition_no_trigger(self, breaker):
        """Test condition check when no trigger should occur."""
        data = {"total_requests": 100, "error_occurred": False}

        triggered = await breaker.check_condition(data)
        assert triggered is False

    @pytest.mark.asyncio
    async def test_check_condition_trigger(self, breaker):
        """Test condition check when trigger should occur."""
        # Add many errors
        breaker.error_times = [datetime.now() for _ in range(15)]

        data = {"total_requests": 100, "error_occurred": False}

        triggered = await breaker.check_condition(data)
        assert triggered is True  # 15/100 = 15% > 10%


class TestCircuitBreakerManager:
    """Test circuit breaker manager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.max_daily_loss_pct = 0.05
        config.risk.max_drawdown_pct = 0.15
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.fixture
    def manager(self, mock_config, mock_risk_manager):
        """Create circuit breaker manager."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            return CircuitBreakerManager(mock_config, mock_risk_manager)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.circuit_breakers) == 5
        assert "daily_loss_limit" in manager.circuit_breakers
        assert "drawdown_limit" in manager.circuit_breakers
        assert "volatility_spike" in manager.circuit_breakers
        assert "model_confidence" in manager.circuit_breakers
        assert "system_error_rate" in manager.circuit_breakers

    @pytest.mark.asyncio
    async def test_evaluate_all_no_triggers(self, manager):
        """Test evaluation of all circuit breakers with no triggers."""
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("100"),  # Positive PnL
            "current_portfolio_value": Decimal("10000"),
            "peak_portfolio_value": Decimal("10000"),
            "price_history": [100, 101, 102],
            "model_confidence": Decimal("0.8"),
            "total_requests": 100,
            "error_occurred": False,
        }

        results = await manager.evaluate_all(data)

        assert len(results) == 5
        assert all(not triggered for triggered in results.values())

    @pytest.mark.asyncio
    async def test_evaluate_all_with_trigger(self, manager):
        """Test evaluation of all circuit breakers with one trigger."""
        data = {
            "portfolio_value": Decimal("10000"),
            "daily_pnl": Decimal("-600"),  # 6% loss, should trigger
            "current_portfolio_value": Decimal("10000"),
            "peak_portfolio_value": Decimal("10000"),
            "price_history": [100, 101, 102],
            "model_confidence": Decimal("0.8"),
            "total_requests": 100,
            "error_occurred": False,
        }

        with pytest.raises(CircuitBreakerTriggeredError):
            await manager.evaluate_all(data)

    @pytest.mark.asyncio
    async def test_get_status(self, manager):
        """Test status retrieval."""
        status = await manager.get_status()

        assert len(status) == 5
        for breaker_name, breaker_status in status.items():
            assert "state" in breaker_status
            assert "trigger_count" in breaker_status
            assert "events_count" in breaker_status

    def test_reset_all(self, manager):
        """Test reset of all circuit breakers."""
        # Set some breakers to open state
        manager.circuit_breakers["daily_loss_limit"].state = CircuitBreakerState.OPEN
        manager.circuit_breakers["daily_loss_limit"].trigger_count = 5

        manager.reset_all()

        for breaker in manager.circuit_breakers.values():
            assert breaker.state == CircuitBreakerState.CLOSED
            assert breaker.trigger_count == 0

    def test_get_triggered_breakers(self, manager):
        """Test getting list of triggered breakers."""
        # Set one breaker to open state
        manager.circuit_breakers["daily_loss_limit"].state = CircuitBreakerState.OPEN

        triggered = manager.get_triggered_breakers()
        assert len(triggered) == 1
        assert "daily_loss_limit" in triggered

    def test_is_trading_allowed(self, manager):
        """Test trading allowed check."""
        # All breakers closed
        assert manager.is_trading_allowed() is True

        # One breaker open
        manager.circuit_breakers["daily_loss_limit"].state = CircuitBreakerState.OPEN
        assert manager.is_trading_allowed() is False

        # Reset
        manager.reset_all()
        assert manager.is_trading_allowed() is True


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker system."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.risk = Mock()
        config.risk.max_daily_loss_pct = 0.05
        config.risk.max_drawdown_pct = 0.15
        return config

    @pytest.fixture
    def mock_risk_manager(self):
        """Create mock risk manager."""
        return Mock(spec=BaseRiskManager)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_cycle(self, mock_config, mock_risk_manager):
        """Test complete circuit breaker recovery cycle."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            breaker = DailyLossLimitBreaker(mock_config, mock_risk_manager)

            # Initial state
            assert breaker.state == CircuitBreakerState.CLOSED

            # Trigger circuit breaker
            data = {
                "portfolio_value": Decimal("10000"),
                "daily_pnl": Decimal("-600"),  # 6% loss
            }

            with pytest.raises(CircuitBreakerTriggeredError):
                await breaker.evaluate(data)

            assert breaker.state == CircuitBreakerState.OPEN
            assert breaker.trigger_count == 1

            # Simulate recovery timeout
            breaker.trigger_time = datetime.now() - timedelta(minutes=35)  # Past 30-minute timeout

            # Test recovery
            data = {
                "portfolio_value": Decimal("10000"),
                "daily_pnl": Decimal("100"),  # Positive PnL
            }

            triggered = await breaker.evaluate(data)
            assert triggered is False
            assert breaker.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_multiple_circuit_breakers(self, mock_config, mock_risk_manager):
        """Test multiple circuit breakers working together."""
        with patch("src.risk_management.circuit_breakers.ErrorHandler", return_value=Mock()):
            manager = CircuitBreakerManager(mock_config, mock_risk_manager)

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
                await manager.evaluate_all(data)

            # Check that multiple breakers are triggered
            triggered = manager.get_triggered_breakers()
            assert len(triggered) > 0
