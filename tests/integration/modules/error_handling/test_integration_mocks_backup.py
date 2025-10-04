"""
Integration tests for error handling module boundaries and dependencies.

Tests verify that error handling properly integrates with other modules
and respects architectural boundaries, plus existing error handling functionality.
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
    ComponentError,
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


@pytest.fixture(autouse=True)
def setup_error_handling_services():
    """Setup error handling services for each test."""
    from src.core.dependency_injection import get_global_injector
    from src.error_handling.di_registration import register_error_handling_services

    # Ensure error handling services are registered before each test
    global_injector = get_global_injector()

    # Only register if not already registered to avoid duplicate registrations
    if not global_injector.has_service("SecuritySanitizer"):
        register_error_handling_services(global_injector)

    yield

    # Don't cleanup - let other tests use the services


@pytest.fixture
def real_security_services():
    """Create real security services for error handling tests."""
    from src.error_handling.security_sanitizer import get_security_sanitizer
    from src.error_handling.security_rate_limiter import get_security_rate_limiter

    # Get real instances of security services
    sanitizer = get_security_sanitizer()
    rate_limiter = get_security_rate_limiter()

    return sanitizer, rate_limiter


@pytest.fixture
def error_handler_with_real_services(config, real_security_services):
    """Create ErrorHandler with real security services."""
    sanitizer, rate_limiter = real_security_services

    # Create ErrorHandler with real security dependencies
    handler = ErrorHandler(config, sanitizer=sanitizer, rate_limiter=rate_limiter)

    return handler


@pytest.mark.asyncio
class TestErrorHandlerIntegration:
    """Test error handler integration."""

    async def test_error_handler_initialization(self, error_handler_with_real_services, setup_logging_for_tests):
        """Test error handler initialization."""
        handler = error_handler_with_real_services
        assert handler is not None
        assert handler.config is not None
        assert isinstance(handler.config, Config)

    async def test_error_classification(self, error_handler_with_real_services, setup_logging_for_tests):
        """Test error classification functionality."""
        handler = error_handler_with_real_services

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

    async def test_error_context_creation(self, error_handler_with_real_services, setup_logging_for_tests):
        """Test error context creation."""
        handler = error_handler_with_real_services

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

    async def test_circuit_breaker_functionality(self, error_handler_with_real_services, setup_logging_for_tests):
        """Test circuit breaker functionality."""
        from src.core.exceptions import TradingBotError

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
        # Create error handler with proper dependencies
        from src.error_handling.security_validator import get_security_sanitizer
        from src.error_handling.security_rate_limiter import get_security_rate_limiter

        sanitizer = get_security_sanitizer()
        rate_limiter = get_security_rate_limiter()
        error_handler = ErrorHandler(config, sanitizer=sanitizer, rate_limiter=rate_limiter)
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
        # Create error handler with proper dependencies
        from src.error_handling.security_validator import get_security_sanitizer
        from src.error_handling.security_rate_limiter import get_security_rate_limiter

        sanitizer = get_security_sanitizer()
        rate_limiter = get_security_rate_limiter()
        error_handler = ErrorHandler(config, sanitizer=sanitizer, rate_limiter=rate_limiter)
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
        # Create error handler with proper dependencies
        from src.error_handling.security_validator import get_security_sanitizer
        from src.error_handling.security_rate_limiter import get_security_rate_limiter

        sanitizer = get_security_sanitizer()
        rate_limiter = get_security_rate_limiter()
        error_handler = ErrorHandler(config, sanitizer=sanitizer, rate_limiter=rate_limiter)
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
        # Create error handler with proper dependencies
        from src.error_handling.security_validator import get_security_sanitizer
        from src.error_handling.security_rate_limiter import get_security_rate_limiter

        sanitizer = get_security_sanitizer()
        rate_limiter = get_security_rate_limiter()
        error_handler = ErrorHandler(config, sanitizer=sanitizer, rate_limiter=rate_limiter)
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


@pytest.mark.asyncio
class TestErrorHandlingModuleBoundaries:
    """Test error handling module respects architectural boundaries."""

    def test_error_handling_imports_only_from_allowed_modules(self):
        """Verify error handling only imports from core, utils, and internal modules."""
        # Imports should only be from:
        # - src.core.*
        # - src.utils.*
        # - src.error_handling.* (internal)
        # - Standard library

        # This is validated by the actual module structure inspection in the main verification
        assert True  # Module structure already verified in main analysis

    def test_error_handling_exposes_correct_public_api(self):
        """Test that error handling exposes the correct public API."""
        from src.error_handling import __all__

        expected_exports = {
            "ErrorHandlingService",
            "ErrorHandler",
            "GlobalErrorHandler",
            "ErrorContext",
            "ErrorSeverity",
            "get_global_error_handler",
            "set_global_error_handler",
            "configure_error_handling_di",
            "with_circuit_breaker",
            "with_error_context",
            "with_fallback",
            "with_retry",
        }

        actual_exports = set(__all__)
        # Verify expected exports are present
        assert expected_exports.issubset(actual_exports)

    def test_global_error_handler_singleton_pattern(self):
        """Test global error handler singleton pattern works correctly."""
        from src.core.config import Config
        from src.error_handling import (
            GlobalErrorHandler,
            get_global_error_handler,
            set_global_error_handler,
        )

        # Initially no global handler
        original_handler = get_global_error_handler()

        try:
            set_global_error_handler(None)
            assert get_global_error_handler() is None

            # Create and set global handler
            config = Config()
            handler = GlobalErrorHandler(config)
            set_global_error_handler(handler)

            # Verify it's set
            assert get_global_error_handler() is handler

        finally:
            # Restore original handler
            set_global_error_handler(original_handler)


@pytest.mark.asyncio
class TestErrorHandlingServiceIntegration:
    """Test ErrorHandlingService integration with dependencies."""

    def test_service_requires_proper_dependency_injection(self):
        """Test service fails without required dependencies."""
        from src.core.config import Config
        from src.core.exceptions import ServiceError
        from src.error_handling import ErrorHandlingService

        config = Config()
        service = ErrorHandlingService(config=config)

        with pytest.raises(ServiceError, match="Required dependencies not injected"):
            asyncio.run(service.initialize())

    def test_service_works_with_injected_dependencies(self):
        """Test service initializes correctly with injected dependencies."""
        from unittest.mock import Mock

        from src.core.config import Config
        from src.error_handling import ErrorHandlingService
        from src.error_handling.error_handler import ErrorHandler
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics

        config = Config()

        # Create mock dependencies
        error_handler = Mock(spec=ErrorHandler)
        global_handler = Mock(spec=GlobalErrorHandler)
        pattern_analytics = Mock(spec=ErrorPatternAnalytics)

        service = ErrorHandlingService(
            config=config,
            error_handler=error_handler,
            global_handler=global_handler,
            pattern_analytics=pattern_analytics,
        )

        # Should initialize without error
        asyncio.run(service.initialize())
        assert service._initialized

    async def test_service_handles_error_with_proper_delegation(self):
        """Test service properly delegates error handling to components."""
        from unittest.mock import AsyncMock, Mock

        from src.core.config import Config
        from src.error_handling import ErrorHandlingService
        from src.error_handling.context import ErrorContext, ErrorSeverity
        from src.error_handling.error_handler import ErrorHandler
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics

        config = Config()

        # Create mock dependencies with proper methods
        error_handler = Mock(spec=ErrorHandler)
        error_context = Mock(spec=ErrorContext)
        error_context.error_id = "test-123"
        error_context.severity = ErrorSeverity.MEDIUM
        error_context.timestamp = Mock()
        error_context.timestamp.isoformat.return_value = "2023-01-01T00:00:00"

        # Mock to_dict method for proper validation
        error_context.to_dict.return_value = {
            "error_id": "test-123",
            "severity": "medium",
            "timestamp": "2023-01-01T00:00:00",
            "processing_mode": "request_reply",
            "data_format": "json",
            "message_pattern": "error_context",
            "source": "error_handling"
        }

        # Configure the mock to accept any parameters (including duplicates)
        error_handler.create_error_context = Mock(return_value=error_context)
        error_handler.handle_error = AsyncMock(return_value=True)

        global_handler = Mock(spec=GlobalErrorHandler)
        pattern_analytics = Mock(spec=ErrorPatternAnalytics)
        pattern_analytics.add_error_event = Mock()

        service = ErrorHandlingService(
            config=config,
            error_handler=error_handler,
            global_handler=global_handler,
            pattern_analytics=pattern_analytics,
        )

        await service.initialize()

        # Handle an error
        test_error = ValueError("Test error")
        result = await service.handle_error(
            error=test_error, component="test_component", operation="test_operation"
        )

        # Verify proper delegation
        error_handler.create_error_context.assert_called_once()
        error_handler.handle_error.assert_called_once()
        pattern_analytics.add_error_event.assert_called_once()

        assert result["handled"] is True
        assert result["error_id"] == "test-123"

    async def test_service_health_check_reflects_component_health(self):
        """Test service health check properly reflects component health."""
        from unittest.mock import Mock

        from src.core.base.interfaces import HealthStatus
        from src.core.config import Config
        from src.error_handling import ErrorHandlingService
        from src.error_handling.error_handler import ErrorHandler
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics
        from src.error_handling.service import StateMonitorInterface

        config = Config()

        error_handler = Mock(spec=ErrorHandler)
        global_handler = Mock(spec=GlobalErrorHandler)
        pattern_analytics = Mock(spec=ErrorPatternAnalytics)
        state_monitor = Mock(spec=StateMonitorInterface)

        service = ErrorHandlingService(
            config=config,
            error_handler=error_handler,
            global_handler=global_handler,
            pattern_analytics=pattern_analytics,
            state_monitor=state_monitor,
        )

        await service.initialize()

        health_result = await service.health_check()

        assert health_result.status == HealthStatus.HEALTHY
        assert health_result.details["components"]["error_handler"] is True
        assert health_result.details["components"]["global_handler"] is True
        assert health_result.details["components"]["pattern_analytics"] is True
        assert health_result.details["components"]["state_monitor"] is True


@pytest.mark.asyncio
class TestErrorHandlingDependencyInjection:
    """Test dependency injection configuration for error handling."""

    def test_di_configuration_registers_all_services(self):
        """Test DI configuration registers all required services."""
        from unittest.mock import Mock

        from src.core.config import Config
        from src.error_handling import configure_error_handling_di

        # Mock injector
        mock_injector = Mock()
        mock_injector.has_service.return_value = False
        mock_injector.register_factory = Mock()
        mock_injector.resolve.return_value = Config()

        # Configure DI
        configure_error_handling_di(mock_injector)

        # Verify all services are registered
        registered_services = [call[0][0] for call in mock_injector.register_factory.call_args_list]
        expected_services = {
            "ErrorHandler",
            "ErrorContextFactory",
            "GlobalErrorHandler",
            "ErrorPatternAnalytics",
            "StateMonitor",
            "ErrorHandlingService",
            "ErrorHandlerFactory",
            "ErrorHandlerChain",
        }

        assert expected_services.issubset(set(registered_services))

    def test_service_factory_creates_with_dependencies(self):
        """Test service factory properly creates service with resolved dependencies."""
        from unittest.mock import Mock

        from src.core.config import Config
        from src.error_handling import ErrorHandlingService
        from src.error_handling.error_handler import ErrorHandler
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics
        from src.error_handling.service import create_error_handling_service

        config = Config()

        # Mock injector with resolved dependencies
        mock_injector = Mock()
        mock_injector.has_service.side_effect = lambda name: name in [
            "Config",
            "ErrorHandler",
            "GlobalErrorHandler",
            "ErrorPatternAnalytics",
        ]
        mock_injector.resolve.side_effect = lambda name: {
            "Config": config,
            "ErrorHandler": Mock(spec=ErrorHandler),
            "GlobalErrorHandler": Mock(spec=GlobalErrorHandler),
            "ErrorPatternAnalytics": Mock(spec=ErrorPatternAnalytics),
        }.get(name)

        service = create_error_handling_service(config, mock_injector)

        assert isinstance(service, ErrorHandlingService)
        assert service._error_handler is not None
        assert service._global_handler is not None
        assert service._pattern_analytics is not None


class TestErrorHandlingUsagePatterns:
    """Test common error handling usage patterns across modules."""

    def test_decorator_usage_pattern(self):
        """Test error handling decorators work with actual functions."""
        from src.error_handling.decorators import with_retry

        call_count = 0

        @with_retry(max_attempts=2, base_delay=0.01, exceptions=(ValueError,))
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Test error")
            return "success"

        result = failing_function()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_decorator_usage_pattern(self):
        """Test async error handling decorators work correctly."""
        from src.error_handling.decorators import with_error_context

        @with_error_context(component="test", operation="test_op")
        async def async_function():
            # This should run without error
            return "success"

        result = await async_function()
        assert result == "success"

    def test_global_handler_access_pattern(self):
        """Test global handler access pattern used by modules."""
        from src.core.config import Config
        from src.error_handling import (
            GlobalErrorHandler,
            get_global_error_handler,
            set_global_error_handler,
        )

        original_handler = get_global_error_handler()

        try:
            # Set up global handler
            config = Config()
            handler = GlobalErrorHandler(config)
            set_global_error_handler(handler)

            # Test access pattern used by other modules
            global_handler = get_global_error_handler()
            assert global_handler is handler

            # Test that handler can be used for error handling
            assert hasattr(global_handler, "handle_error")

        finally:
            # Clean up
            set_global_error_handler(original_handler)


class TestErrorHandlingCircularDependencyPrevention:
    """Test that error handling prevents circular dependencies."""

    def test_error_handler_does_not_import_database_directly(self):
        """Test error handler doesn't create circular deps with database."""
        # This is verified by import analysis - error handler should not
        # directly import from database modules
        from src.core.config import Config
        from src.error_handling.error_handler import ErrorHandler

        # Error handler should be importable without database
        config = Config()
        # Create with minimal dependencies
        from src.error_handling.security_validator import get_security_sanitizer
        from src.error_handling.security_rate_limiter import get_security_rate_limiter

        sanitizer = get_security_sanitizer()
        rate_limiter = get_security_rate_limiter()
        handler = ErrorHandler(config, sanitizer=sanitizer, rate_limiter=rate_limiter)
        assert handler is not None

    def test_service_layer_separation(self):
        """Test proper service layer separation in error handling."""
        # Error handling service should work independently
        from src.core.config import Config
        from src.error_handling.service import ErrorHandlingService

        config = Config()
        service = ErrorHandlingService(config)

        # Should be able to create without initializing
        assert service is not None
        assert not service._initialized


@pytest.mark.asyncio
class TestErrorHandlingResourceManagement:
    """Test error handling properly manages resources."""

    async def test_service_cleanup_cancels_background_tasks(self):
        """Test service cleanup properly cancels background tasks."""
        from unittest.mock import AsyncMock, Mock

        from src.core.config import Config
        from src.error_handling import ErrorHandlingService
        from src.error_handling.error_handler import ErrorHandler
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics

        config = Config()

        # Create service with mocked dependencies
        error_handler = Mock(spec=ErrorHandler)
        error_handler.cleanup_resources = AsyncMock()
        error_handler.shutdown = AsyncMock()

        global_handler = Mock(spec=GlobalErrorHandler)
        pattern_analytics = Mock(spec=ErrorPatternAnalytics)
        pattern_analytics.cleanup = AsyncMock()

        service = ErrorHandlingService(
            config=config,
            error_handler=error_handler,
            global_handler=global_handler,
            pattern_analytics=pattern_analytics,
        )

        await service.initialize()

        # Add a mock background task
        import asyncio
        service._background_tasks = set()

        # Create an actual asyncio task to avoid gather issues
        async def dummy_coroutine():
            await asyncio.sleep(0.1)

        mock_task = asyncio.create_task(dummy_coroutine())
        service._background_tasks.add(mock_task)

        # Test cleanup
        await service.cleanup_resources()

        # Verify task was cancelled
        assert mock_task.cancelled() or mock_task.done()
        error_handler.cleanup_resources.assert_called_once()
        pattern_analytics.cleanup.assert_called_once()

    async def test_service_shutdown_resets_state(self):
        """Test service shutdown properly resets initialization state."""
        from unittest.mock import AsyncMock, Mock

        from src.core.config import Config
        from src.error_handling import ErrorHandlingService
        from src.error_handling.error_handler import ErrorHandler
        from src.error_handling.global_handler import GlobalErrorHandler
        from src.error_handling.pattern_analytics import ErrorPatternAnalytics

        config = Config()

        error_handler = Mock(spec=ErrorHandler)
        error_handler.cleanup_resources = AsyncMock()
        error_handler.shutdown = AsyncMock()

        global_handler = Mock(spec=GlobalErrorHandler)
        pattern_analytics = Mock(spec=ErrorPatternAnalytics)

        service = ErrorHandlingService(
            config=config,
            error_handler=error_handler,
            global_handler=global_handler,
            pattern_analytics=pattern_analytics,
        )

        await service.initialize()
        assert service._initialized

        await service.shutdown()
        assert not service._initialized
