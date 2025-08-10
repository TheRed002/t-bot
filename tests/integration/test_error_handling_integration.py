"""
Integration tests for error handling functionality.

These tests verify error handling, recovery scenarios, and resilience
with actual component interactions.
"""

import asyncio
from datetime import datetime
from decimal import Decimal

import pytest

from src.core.config import Config
from src.core.exceptions import (
    DataError,
    ExchangeError,
    RiskManagementError,
    StateConsistencyError,
    TradingBotError,
    ValidationError,
)
from src.core.logging import get_logger, setup_logging
from src.error_handling.connection_manager import (
    ConnectionManager,
    ConnectionState,
)
from src.error_handling.error_handler import (
    CircuitBreaker,
    ErrorHandler,
    ErrorSeverity,
    error_handler_decorator,
)
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.recovery_scenarios import (
    APIRateLimitRecovery,
    DataFeedInterruptionRecovery,
    ExchangeMaintenanceRecovery,
    NetworkDisconnectionRecovery,
    OrderRejectionRecovery,
    PartialFillRecovery,
)
from src.error_handling.state_monitor import StateMonitor, StateValidationResult


@pytest.fixture(scope="session")
def config():
    """Provide test configuration."""
    return Config()


@pytest.fixture(scope="session")
def setup_logging_for_tests():
    """Setup logging for tests."""
    setup_logging(environment="test")
    logger = get_logger(__name__)


@pytest.mark.asyncio
class TestErrorHandlerIntegration:
    """Test error handler integration."""

    async def test_error_handler_initialization(self, config, setup_logging_for_tests):
        """Test error handler initialization."""
        handler = ErrorHandler(config)
        assert handler is not None
        assert handler.config == config

    async def test_error_classification(self, config, setup_logging_for_tests):
        """Test error classification functionality."""
        handler = ErrorHandler(config)

        # Test error classification
        test_errors = [
            (StateConsistencyError("State corruption"), ErrorSeverity.CRITICAL),
            (RiskManagementError("Risk limit exceeded"), ErrorSeverity.HIGH),
            (DataError("Data validation failed"), ErrorSeverity.MEDIUM),
            (ValidationError("Invalid input"), ErrorSeverity.LOW),
        ]

        for error, expected_severity in test_errors:
            severity = handler.classify_error(error)
            assert severity == expected_severity, f"Expected {expected_severity}, got {severity}"

    async def test_error_context_creation(self, config, setup_logging_for_tests):
        """Test error context creation."""
        handler = ErrorHandler(config)

        context = handler.create_error_context(
            error=ExchangeError("API timeout"),
            component="exchange",
            operation="place_order",
            user_id="test_user",
            bot_id="test_bot",
            symbol="BTCUSDT",
        )

        assert context.error_id is not None
        assert context.component == "exchange"
        assert context.operation == "place_order"
        assert context.severity == ErrorSeverity.HIGH
        assert context.user_id == "test_user"
        assert context.bot_id == "test_bot"
        assert context.symbol == "BTCUSDT"

    async def test_circuit_breaker_functionality(self, config, setup_logging_for_tests):
        """Test circuit breaker functionality."""
        circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

        # Simulate successful calls
        for i in range(2):
            result = circuit_breaker.call(lambda: "success")
            assert result == "success"

        # Simulate failures
        for i in range(3):
            try:
                circuit_breaker.call(lambda: (_ for _ in ()).throw(Exception("test error")))
            except Exception:
                pass

        # Circuit should be open now
        assert circuit_breaker.state == "OPEN"

        # Test that circuit breaker blocks calls when open
        with pytest.raises(TradingBotError):
            circuit_breaker.call(lambda: "should not execute")

    async def test_error_handler_decorator(self, config, setup_logging_for_tests):
        """Test error handler decorator."""

        @error_handler_decorator("test", "test_function")
        def test_function():
            raise ValidationError("Test validation error")

        try:
            test_function()
        except ValidationError:
            pass  # Expected


@pytest.mark.asyncio
class TestRecoveryScenariosIntegration:
    """Test recovery scenario integrations."""

    async def test_partial_fill_recovery(self, config, setup_logging_for_tests):
        """Test partial fill recovery scenario."""
        recovery = PartialFillRecovery(config)

        # Create a mock order object
        class MockOrder:
            def __init__(self, order_id, quantity):
                self.id = order_id
                self.quantity = quantity

            def get(self, key, default=None):
                return getattr(self, key, default)

        result = await recovery.execute_recovery(
            {"order": MockOrder("test_order", Decimal("1.0")), "filled_quantity": Decimal("0.5")}
        )
        assert result is not None

    async def test_network_disconnection_recovery(self, config, setup_logging_for_tests):
        """Test network disconnection recovery scenario."""
        recovery = NetworkDisconnectionRecovery(config)
        result = await recovery.execute_recovery(
            {
                "connection_type": "exchange",
                "offline_duration": 60,
                "last_heartbeat": datetime.now(),
            }
        )
        assert result is not None

    async def test_exchange_maintenance_recovery(self, config, setup_logging_for_tests):
        """Test exchange maintenance recovery scenario."""
        recovery = ExchangeMaintenanceRecovery(config)
        result = await recovery.execute_recovery(
            {
                "exchange": "binance",
                "maintenance_duration": 3600,
                "affected_symbols": ["BTCUSDT", "ETHUSDT"],
            }
        )
        assert result is not None

    async def test_data_feed_interruption_recovery(self, config, setup_logging_for_tests):
        """Test data feed interruption recovery scenario."""
        recovery = DataFeedInterruptionRecovery(config)
        result = await recovery.execute_recovery(
            {
                "data_source": "market_data",
                "staleness_duration": 45,
                "affected_symbols": ["BTCUSDT"],
            }
        )
        assert result is not None

    async def test_order_rejection_recovery(self, config, setup_logging_for_tests):
        """Test order rejection recovery scenario."""
        recovery = OrderRejectionRecovery(config)
        result = await recovery.execute_recovery(
            {
                "order_id": "test_order",
                "rejection_reason": "insufficient_balance",
                "symbol": "BTCUSDT",
                "quantity": 0.001,
            }
        )
        assert result is not None

    async def test_api_rate_limit_recovery(self, config, setup_logging_for_tests):
        """Test API rate limit recovery scenario."""
        recovery = APIRateLimitRecovery(config)
        result = await recovery.execute_recovery(
            {
                "endpoint": "/api/v3/order",
                "rate_limit_type": "requests_per_minute",
                "current_usage": 1200,
                "limit": 1200,
            }
        )
        assert result is not None


@pytest.mark.asyncio
class TestConnectionManagerIntegration:
    """Test connection manager integration."""

    async def test_connection_establishment(self, config, setup_logging_for_tests):
        """Test connection establishment."""
        manager = ConnectionManager(config)

        # Mock connection function
        async def mock_connect(**kwargs):
            await asyncio.sleep(0.1)  # Simulate connection time
            return {"status": "connected", "latency": 50}

        # Test connection establishment
        connection = await manager.establish_connection(
            connection_id="test_connection",
            connection_type="exchange",
            connect_func=mock_connect,
            host="localhost",
            port=8080,
        )

        assert connection is not None
        status = manager.get_connection_status("test_connection")
        assert status is not None
        assert status.get("state") == ConnectionState.CONNECTED.value

    async def test_message_queuing(self, config, setup_logging_for_tests):
        """Test message queuing functionality."""
        manager = ConnectionManager(config)

        # Establish connection first
        async def mock_connect(**kwargs):
            return {"status": "connected", "latency": 50}

        await manager.establish_connection(
            connection_id="test_connection",
            connection_type="exchange",
            connect_func=mock_connect,
            host="localhost",
            port=8080,
        )

        # Test message queuing
        await manager.queue_message("test_connection", {"type": "order", "data": "test"})
        message_count = await manager.flush_message_queue("test_connection")
        assert message_count > 0

    async def test_connection_closure(self, config, setup_logging_for_tests):
        """Test connection closure."""
        manager = ConnectionManager(config)

        # Establish connection first
        async def mock_connect(**kwargs):
            return {"status": "connected", "latency": 50}

        await manager.establish_connection(
            connection_id="test_connection",
            connection_type="exchange",
            connect_func=mock_connect,
            host="localhost",
            port=8080,
        )

        # Test connection closure
        await manager.close_connection("test_connection")
        status = manager.get_connection_status("test_connection")
        assert status is None or status.get("state") == ConnectionState.DISCONNECTED.value


@pytest.mark.asyncio
class TestStateMonitorIntegration:
    """Test state monitor integration."""

    async def test_state_validation(self, config, setup_logging_for_tests):
        """Test state validation functionality."""
        monitor = StateMonitor(config)

        # Test state validation
        validation_result = await monitor.validate_state_consistency()
        assert validation_result is not None
        assert isinstance(validation_result, StateValidationResult)

    async def test_state_reconciliation(self, config, setup_logging_for_tests):
        """Test state reconciliation functionality."""
        monitor = StateMonitor(config)

        # Test state reconciliation
        reconciliation_result = await monitor.reconcile_state(
            "test_component", [{"type": "test_discrepancy"}]
        )
        assert reconciliation_result is not None

    async def test_monitoring_summary(self, config, setup_logging_for_tests):
        """Test monitoring summary functionality."""
        monitor = StateMonitor(config)

        # Test monitoring summary
        summary = monitor.get_state_summary()
        assert summary is not None
        assert "total_validations" in summary

    async def test_validation_history(self, config, setup_logging_for_tests):
        """Test validation history functionality."""
        monitor = StateMonitor(config)

        # Test validation history
        history = monitor.get_state_history()
        assert history is not None


@pytest.mark.asyncio
class TestPatternAnalyticsIntegration:
    """Test pattern analytics integration."""

    async def test_error_pattern_analysis(self, config, setup_logging_for_tests):
        """Test error pattern analysis functionality."""
        analytics = ErrorPatternAnalytics(config)

        # Test error event addition
        for i in range(5):
            analytics.add_error_event(
                {
                    "error_type": "ValidationError",
                    "component": "order_manager",
                    "severity": "MEDIUM",
                    "timestamp": datetime.now(),
                    "details": {"field": "quantity", "value": "invalid"},
                }
            )

        # Test pattern analysis
        patterns = analytics.get_pattern_summary()
        assert patterns is not None

    async def test_correlation_analysis(self, config, setup_logging_for_tests):
        """Test correlation analysis functionality."""
        analytics = ErrorPatternAnalytics(config)

        # Add some error events
        for i in range(3):
            analytics.add_error_event(
                {
                    "error_type": "ConnectionError",
                    "component": "exchange",
                    "severity": "HIGH",
                    "timestamp": datetime.now(),
                    "details": {"host": "localhost", "port": 8080},
                }
            )

        # Test correlation analysis
        correlations = analytics.get_correlation_summary()
        assert correlations is not None

    async def test_trend_analysis(self, config, setup_logging_for_tests):
        """Test trend analysis functionality."""
        analytics = ErrorPatternAnalytics(config)

        # Add error events over time
        for i in range(10):
            analytics.add_error_event(
                {
                    "error_type": "DataError",
                    "component": "market_data",
                    "severity": "MEDIUM",
                    "timestamp": datetime.now(),
                    "details": {"symbol": "BTCUSDT", "issue": "stale_data"},
                }
            )

        # Test trend analysis
        trends = analytics.get_trend_summary()
        assert trends is not None


@pytest.mark.asyncio
class TestErrorHandlingIntegration:
    """Test comprehensive error handling integration."""

    async def test_error_handling_with_connection_issues(self, config, setup_logging_for_tests):
        """Test error handling with connection issues."""
        error_handler = ErrorHandler(config)
        connection_manager = ConnectionManager(config)

        # Simulate connection failure
        async def mock_failing_connect(**kwargs):
            raise ConnectionError("Connection failed")

        # Test connection failure handling
        try:
            await connection_manager.establish_connection(
                connection_id="failing_connection",
                connection_type="exchange",
                connect_func=mock_failing_connect,
                host="invalid_host",
                port=9999,
            )
        except ConnectionError:
            pass  # Expected

        # Test error handling integration
        error_context = error_handler.create_error_context(
            error=ConnectionError("Connection failed"),
            component="connection_manager",
            operation="establish_connection",
            user_id="test_user",
        )

        assert error_context is not None
        assert error_context.severity == ErrorSeverity.HIGH

    async def test_error_handling_with_state_monitoring(self, config, setup_logging_for_tests):
        """Test error handling with state monitoring integration."""
        error_handler = ErrorHandler(config)
        state_monitor = StateMonitor(config)

        # Test state monitoring integration
        validation_result = await state_monitor.validate_state_consistency()
        assert validation_result is not None

        # Test error context creation for state issues
        error_context = error_handler.create_error_context(
            error=StateConsistencyError("State inconsistency detected"),
            component="state_monitor",
            operation="validate_state_consistency",
            user_id="test_user",
        )

        assert error_context is not None
        assert error_context.severity == ErrorSeverity.CRITICAL

    async def test_error_handling_with_pattern_analytics(self, config, setup_logging_for_tests):
        """Test error handling with pattern analytics integration."""
        error_handler = ErrorHandler(config)
        pattern_analytics = ErrorPatternAnalytics(config)

        # Add error events
        pattern_analytics.add_error_event(
            {
                "error_type": "ConnectionError",
                "component": "connection_manager",
                "severity": "HIGH",
                "timestamp": datetime.now(),
                "details": {"host": "invalid_host", "port": 9999},
            }
        )

        # Test pattern analytics integration
        patterns = pattern_analytics.get_pattern_summary()
        assert patterns is not None

        # Test error context creation
        error_context = error_handler.create_error_context(
            error=ConnectionError("Connection failed"),
            component="connection_manager",
            operation="establish_connection",
            user_id="test_user",
        )

        assert error_context is not None
        assert error_context.severity == ErrorSeverity.HIGH

    async def test_comprehensive_error_recovery_flow(self, config, setup_logging_for_tests):
        """Test comprehensive error recovery flow."""
        error_handler = ErrorHandler(config)
        connection_manager = ConnectionManager(config)
        state_monitor = StateMonitor(config)
        pattern_analytics = ErrorPatternAnalytics(config)

        # Simulate a complex error scenario
        try:
            # Simulate connection failure
            async def mock_failing_connect(**kwargs):
                raise ConnectionError("Connection failed")

            await connection_manager.establish_connection(
                connection_id="failing_connection",
                connection_type="exchange",
                connect_func=mock_failing_connect,
                host="invalid_host",
                port=9999,
            )
        except ConnectionError:
            # Create error context
            error_context = error_handler.create_error_context(
                error=ConnectionError("Connection failed"),
                component="connection_manager",
                operation="establish_connection",
                user_id="test_user",
            )

            # Add to pattern analytics
            pattern_analytics.add_error_event(
                {
                    "error_type": "ConnectionError",
                    "component": "connection_manager",
                    "severity": "HIGH",
                    "timestamp": datetime.now(),
                    "details": {"host": "invalid_host", "port": 9999},
                }
            )

            # Validate state
            validation_result = await state_monitor.validate_state_consistency()

            # Verify all components worked together
            assert error_context is not None
            assert validation_result is not None
            assert pattern_analytics.get_pattern_summary() is not None
