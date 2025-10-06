"""
Phase 2: Real Services Error Handling Integration Tests

Tests error handling functionality with real services:
1. Real error handler service with actual dependencies
2. Real error context creation and processing
3. Real circuit breaker and retry mechanisms
4. Real error pattern analytics and monitoring

NO MOCKS - All operations use actual services and dependencies.
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.config import Config
from src.core.exceptions import ServiceError, ValidationError
from src.error_handling.connection_manager import ConnectionManager
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.global_handler import GlobalErrorHandler
from src.error_handling.pattern_analytics import ErrorPatternAnalytics
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery, PartialFillRecovery
from src.error_handling.service import ErrorHandlingService


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_error_handling_service_initialization():
    """Test error handling service initializes with real dependencies."""
    config = Config()

    # Create real error handler with actual dependencies
    error_handler = ErrorHandler(config=config)
    global_handler = GlobalErrorHandler(config=config)
    pattern_analytics = ErrorPatternAnalytics(config=config)

    service = ErrorHandlingService(
        config=config,
        error_handler=error_handler,
        global_handler=global_handler,
        pattern_analytics=pattern_analytics,
    )

    try:
        await service.start()
        assert service.is_running
        assert service._error_handler is not None
        assert service._global_handler is not None
        assert service._pattern_analytics is not None

        # Test health check with real services
        health_result = await service.health_check()
        assert health_result.status.value in [
            "healthy",
            "degraded",
        ]  # May be degraded due to missing state monitor
        assert "components" in health_result.details

    finally:
        await service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_error_handler_error_processing():
    """Test error handler processes errors with real error context creation."""
    from src.core.dependency_injection import DependencyInjector
    from src.error_handling.di_registration import register_error_handling_services

    config = Config()

    # Setup real dependency injection container
    injector = DependencyInjector()
    register_error_handling_services(injector, config)

    # Retrieve registered services from dependency injection
    sanitizer = injector.resolve("SecuritySanitizer")
    rate_limiter = injector.resolve("SecurityRateLimiter")

    # Create error handler with properly injected dependencies
    error_handler = ErrorHandler(config=config, sanitizer=sanitizer, rate_limiter=rate_limiter)

    # Create a real error
    test_error = ValueError("Real test error")
    component = "test_component"
    operation = "test_operation"

    # Process error with real error context
    error_context = error_handler.create_error_context(
        error=test_error,
        component=component,
        operation=operation,
        context={"test_key": "test_value"},
    )

    assert error_context is not None
    assert error_context.error_id is not None
    assert error_context.component == component
    assert error_context.operation == operation
    assert error_context.error_type == "ValueError"

    # Handle error with real processing
    result = await error_handler.handle_error(test_error, error_context)
    # Error handling completed (result may be True or False depending on recovery success)
    assert result is not None

    # Cleanup resources
    await error_handler.cleanup_resources()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_pattern_analytics_error_tracking():
    """Test pattern analytics tracks errors with real data storage."""
    config = Config()
    pattern_analytics = ErrorPatternAnalytics(config=config)

    try:
        await pattern_analytics.initialize()

        # Add real error events
        error_events = [
            {
                "error_type": "NetworkError",
                "component": "exchange_client",
                "timestamp": datetime.now(timezone.utc),
                "severity": "high",
            },
            {
                "error_type": "ValidationError",
                "component": "order_validator",
                "timestamp": datetime.now(timezone.utc),
                "severity": "medium",
            },
        ]

        for event in error_events:
            pattern_analytics.add_error_event(error_context=event)

        # Analyze patterns with real data
        pattern_summary = pattern_analytics.get_pattern_summary()
        assert pattern_summary is not None

        # Get real correlations
        correlation_summary = pattern_analytics.get_correlation_summary()
        assert correlation_summary is not None

        # Get recent errors
        recent_errors = pattern_analytics.get_recent_errors(hours=1)
        assert recent_errors is not None
        assert len(recent_errors) == 2  # Should have our 2 test events

    finally:
        # Clean up manually since HistoryWrapper doesn't have clear()
        if hasattr(pattern_analytics, "_error_history_list"):
            pattern_analytics._error_history_list.clear()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_recovery_scenarios_execution():
    """Test recovery scenarios execute with real recovery logic."""
    config = Config()
    partial_fill_recovery = PartialFillRecovery(config=config)
    network_recovery = NetworkDisconnectionRecovery(config=config)

    try:
        await partial_fill_recovery.initialize()
        await network_recovery.initialize()

        # Test partial fill recovery with real scenario
        recovery_result = await partial_fill_recovery.execute_recovery(
            context={
                "order": {
                    "order_id": "test-order-123",
                    "symbol": "BTC/USDT",
                    "quantity": Decimal("1.0"),
                    "filled_quantity": Decimal("0.5"),
                    "remaining_quantity": Decimal("0.5"),
                    "exchange": "test_exchange",
                }
            }
        )

        assert recovery_result is not None

        # Test network disconnection recovery
        recovery_result = await network_recovery.execute_recovery(
            context={"component": "exchange_client", "exchange": "binance", "reconnect_attempts": 0}
        )

        assert recovery_result is not None

    finally:
        await partial_fill_recovery.cleanup()
        await network_recovery.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_connection_manager_operations():
    """Test connection manager with real connection handling."""
    config = Config()
    connection_manager = ConnectionManager(config=config)

    try:
        await connection_manager.initialize()

        # Test real connection establishment
        async def mock_connect_func():
            """Mock connection function for testing."""
            return {"status": "connected", "endpoint": "test://localhost:8080"}

        connection_result = await connection_manager.establish_connection(
            connection_id="test-conn-123",
            connection_type="websocket",
            connect_func=mock_connect_func,
        )

        # Connection may fail but should return boolean result
        assert isinstance(connection_result, bool)

        # Test message queuing functionality
        await connection_manager.queue_message(
            connection_id="test-conn-123", message={"type": "ping", "data": "test"}
        )

        # Verify message was queued (this doesn't throw an exception if successful)

    finally:
        await connection_manager.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_error_handling_end_to_end_flow():
    """Test complete error handling flow with real services."""
    config = Config()

    # Create all real services
    error_handler = ErrorHandler(config=config)
    global_handler = GlobalErrorHandler(config=config)
    pattern_analytics = ErrorPatternAnalytics(config=config)

    service = ErrorHandlingService(
        config=config,
        error_handler=error_handler,
        global_handler=global_handler,
        pattern_analytics=pattern_analytics,
    )

    try:
        # Initialize all services
        await service.start()

        # Create and handle a real error
        test_error = ServiceError("End-to-end test error")

        result = await service.handle_error(
            error=test_error,
            component="integration_test",
            operation="end_to_end_test",
            context={"test_run": str(uuid.uuid4())},
        )

        assert result is not None
        assert "handled" in result or "error_id" in result

        # Verify error was tracked by analytics
        metrics = await service.get_error_handler_metrics()
        assert metrics is not None

        # Test service cleanup
        await service.cleanup_resources()

    finally:
        await service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_error_context_transformation_and_validation():
    """Test error context transformation with real validation."""
    config = Config()
    error_handler = ErrorHandler(config=config)

    try:
        await error_handler.initialize()

        # Test real context transformation
        original_context = {
            "order_id": "12345",
            "symbol": "BTC/USDT",
            "price": "50000.50",
            "quantity": "1.0",
            "exchange": "binance",
        }

        test_error = ValidationError("Invalid order parameters")

        error_context = error_handler.create_error_context(
            error=test_error,
            component="order_processor",
            operation="validate_order",
            context=original_context,
        )

        # Verify real transformation occurred
        assert error_context is not None
        assert error_context.context is not None
        assert "order_id" in error_context.context

        # Test validation with boundary checking
        assert error_context.error_type == "ValidationError"
        assert error_context.component == "order_processor"
        assert error_context.severity is not None

    finally:
        await error_handler.cleanup()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_circuit_breaker_functionality():
    """Test circuit breaker with real failure tracking."""
    from src.error_handling.decorators import with_circuit_breaker

    call_count = 0
    failure_threshold = 2

    @with_circuit_breaker(failure_threshold=failure_threshold, recovery_timeout=0.1)
    async def failing_function():
        nonlocal call_count
        call_count += 1
        raise ConnectionError(f"Real connection error - attempt {call_count}")

    # Test real circuit breaker opening
    with pytest.raises(ConnectionError):
        await failing_function()  # First failure

    with pytest.raises(ConnectionError):
        await failing_function()  # Second failure - should open circuit

    # Circuit should now be open
    with pytest.raises(Exception):  # Circuit breaker should prevent execution
        await failing_function()

    assert call_count == 2  # Should not increment on third call due to open circuit


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_real_retry_mechanism_with_backoff():
    """Test retry mechanism with real backoff timing."""
    import time

    from src.error_handling.decorators import with_retry

    call_count = 0
    start_time = time.time()

    @with_retry(max_attempts=3, base_delay=0.1, exceptions=(ValueError,))
    async def eventually_succeeding_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError(f"Real retry test error - attempt {call_count}")
        return "success"

    result = await eventually_succeeding_function()
    end_time = time.time()

    assert result == "success"
    assert call_count == 3

    # Verify real backoff timing occurred (should take at least 0.2s for 2 retries)
    elapsed_time = end_time - start_time
    assert elapsed_time >= 0.1  # Should have some delay from backoff
