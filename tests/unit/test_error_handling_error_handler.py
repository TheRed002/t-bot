"""
Unit tests for error handler component.

These tests verify error categorization, severity classification, retry policies,
circuit breaker integration, and error context preservation.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from src.core.config import Config
from src.core.exceptions import (
    TradingBotError, ExchangeError, RiskManagementError,
    ValidationError, ExecutionError, ModelError, DataError,
    StateConsistencyError, SecurityError
)

from src.error_handling.error_handler import (
    ErrorHandler,
    ErrorSeverity,
    ErrorContext,
    CircuitBreaker,
    error_handler_decorator)
from src.error_handling.pattern_analytics import ErrorPattern


class TestErrorSeverity:
    """Test error severity enumeration."""

    def test_error_severity_values(self):
        """Test error severity enum values."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.LOW.value == "low"

    def test_error_severity_comparison(self):
        """Test error severity comparison."""
        assert ErrorSeverity.CRITICAL != ErrorSeverity.HIGH
        assert ErrorSeverity.HIGH != ErrorSeverity.MEDIUM
        assert ErrorSeverity.MEDIUM != ErrorSeverity.LOW


class TestErrorContext:
    """Test error context creation and management."""

    def test_error_context_creation(self):
        """Test error context creation with all fields."""
        error = ExchangeError("API timeout")
        context = ErrorContext(
            error_id="test_error_123",
            timestamp=datetime.now(timezone.utc),
            severity=ErrorSeverity.HIGH,
            component="exchange",
            operation="place_order",
            error=error,
            user_id="test_user",
            bot_id="test_bot",
            symbol="BTCUSDT",
            order_id="order_123",
            details={"key": "value"},
            stack_trace="traceback info"
        )

        assert context.error_id == "test_error_123"
        assert context.severity == ErrorSeverity.HIGH
        assert context.component == "exchange"
        assert context.operation == "place_order"
        assert context.user_id == "test_user"
        assert context.bot_id == "test_bot"
        assert context.symbol == "BTCUSDT"
        assert context.order_id == "order_123"
        assert context.details == {"key": "value"}
        assert context.stack_trace == "traceback info"
        assert context.recovery_attempts == 0
        assert context.max_recovery_attempts == 3

    def test_error_context_minimal(self):
        """Test error context creation with minimal fields."""
        error = DataError("Database connection failed")
        context = ErrorContext(
            error_id="test_error_123",
            timestamp=datetime.now(timezone.utc),
            severity=ErrorSeverity.MEDIUM,
            component="database",
            operation="query",
            error=error
        )

        assert context.error_id == "test_error_123"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.component == "database"
        assert context.operation == "query"
        assert context.user_id is None
        assert context.bot_id is None
        assert context.symbol is None
        assert context.order_id is None
        assert context.details == {}
        assert context.stack_trace is None


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10
        assert cb.failure_count == 0
        assert cb.state == "CLOSED"
        assert cb.last_failure_time is None

    def test_circuit_breaker_successful_call(self):
        """Test circuit breaker with successful function call."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5)

        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0

    def test_circuit_breaker_failure_but_not_open(self):
        """Test circuit breaker with failure but not enough to open."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == "CLOSED"
        assert cb.failure_count == 1
        assert cb.last_failure_time is not None

    def test_circuit_breaker_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=5)

        def failing_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == "OPEN"
        assert cb.failure_count == 2

    def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=5)

        def failing_func():
            raise ValueError("test error")

        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == "OPEN"

        # Should block calls when open
        with pytest.raises(TradingBotError, match="Circuit breaker is OPEN"):
            cb.call(lambda: "should not execute")

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open recovery."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        def failing_func():
            raise ValueError("test error")

        # Trigger failure to open circuit
        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert cb.state == "OPEN"

        # Wait for recovery timeout
        time.sleep(0.2)

        # Should be ready to transition to half-open
        assert cb.should_transition_to_half_open()

        # Next call should transition to half-open and then close
        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0


class TestErrorHandler:
    """Test error handler functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def error_handler(self, config):
        """Provide error handler instance."""
        return ErrorHandler(config)

    def test_error_handler_initialization(self, config):
        """Test error handler initialization."""
        handler = ErrorHandler(config)
        assert handler.config == config
        assert handler.error_patterns == {}
        assert "network_errors" in handler.retry_policies
        assert "api_rate_limits" in handler.retry_policies
        assert "database_errors" in handler.retry_policies

    def test_error_classification(self, error_handler):
        """Test error classification functionality."""
        # Test critical errors
        assert error_handler.classify_error(
            StateConsistencyError("State corruption")) == ErrorSeverity.CRITICAL
        assert error_handler.classify_error(
            SecurityError("Security breach")) == ErrorSeverity.CRITICAL

        # Test high severity errors
        assert error_handler.classify_error(
            RiskManagementError("Risk limit exceeded")) == ErrorSeverity.HIGH
        assert error_handler.classify_error(
            ExchangeError("API timeout")) == ErrorSeverity.HIGH
        assert error_handler.classify_error(ExecutionError(
            "Order execution failed")) == ErrorSeverity.HIGH

        # Test medium severity errors
        assert error_handler.classify_error(
            DataError("Data validation failed")) == ErrorSeverity.MEDIUM
        assert error_handler.classify_error(
            ModelError("Model inference failed")) == ErrorSeverity.MEDIUM

        # Test low severity errors
        assert error_handler.classify_error(
            ValidationError("Invalid input")) == ErrorSeverity.LOW

        # Test unknown errors
        assert error_handler.classify_error(
            ValueError("Unknown error")) == ErrorSeverity.MEDIUM

    def test_error_context_creation(self, error_handler):
        """Test error context creation."""
        error = ExchangeError("API timeout")
        context = error_handler.create_error_context(
            error=error,
            component="exchange",
            operation="place_order",
            user_id="test_user",
            bot_id="test_bot",
            symbol="BTCUSDT"
        )

        assert context.error_id is not None
        assert context.component == "exchange"
        assert context.operation == "place_order"
        assert context.severity == ErrorSeverity.HIGH
        assert context.user_id == "test_user"
        assert context.bot_id == "test_bot"
        assert context.symbol == "BTCUSDT"
        assert context.timestamp is not None
        assert "error_type" in context.details
        assert "error_message" in context.details

    def test_error_context_creation_with_details(self, error_handler):
        """Test error context creation with additional details."""
        error = RiskManagementError("Risk limit exceeded")
        context = error_handler.create_error_context(
            error=error,
            component="risk_management",
            operation="validate_position",
            details={"position_size": 1000, "limit": 500}
        )

        assert context.error_id is not None
        assert context.component == "risk_management"
        assert context.operation == "validate_position"
        assert context.severity == ErrorSeverity.HIGH
        assert context.timestamp is not None
        assert "error_type" in context.details
        assert "error_message" in context.details
        assert "kwargs" in context.details

    @pytest.mark.asyncio
    async def test_handle_error_success(self, error_handler):
        """Test successful error handling."""
        error = ValidationError("Invalid input")
        context = error_handler.create_error_context(
            error=error,
            component="validation",
            operation="validate_input"
        )

        # Mock recovery strategy
        recovery_called = False

        async def mock_recovery(ctx):
            nonlocal recovery_called
            recovery_called = True
            return True

        result = await error_handler.handle_error(error, context, mock_recovery)

        assert result is True
        assert recovery_called

    @pytest.mark.asyncio
    async def test_handle_error_without_recovery(self, error_handler):
        """Test error handling without recovery strategy."""
        error = DataError("Data validation failed")
        context = error_handler.create_error_context(
            error=error,
            component="data",
            operation="validate_data"
        )

        result = await error_handler.handle_error(error, context)

        # Should return False for non-critical errors without recovery
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_error_critical_escalation(self, error_handler):
        """Test critical error escalation."""
        error = StateConsistencyError("State corruption")
        context = error_handler.create_error_context(
            error=error,
            component="state",
            operation="validate_state"
        )

        with patch.object(error_handler, '_escalate_error') as mock_escalate:
            result = await error_handler.handle_error(error, context)

            assert result is False
            mock_escalate.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_handle_error_max_attempts_exceeded(self, error_handler):
        """Test error handling when max attempts exceeded."""
        error = ExchangeError("API timeout")
        context = error_handler.create_error_context(
            error=error,
            component="exchange",
            operation="place_order"
        )

        # Set recovery attempts to max
        context.recovery_attempts = context.max_recovery_attempts

        with patch.object(error_handler, '_escalate_error') as mock_escalate:
            result = await error_handler.handle_error(error, context)

            assert result is False
            # Escalation should be called for HIGH severity errors
            mock_escalate.assert_called_once_with(context)

    def test_get_retry_policy(self, error_handler):
        """Test retry policy retrieval."""
        network_policy = error_handler.get_retry_policy("network_errors")
        assert network_policy["max_attempts"] == 5
        assert network_policy["backoff_strategy"] == "exponential"

        api_policy = error_handler.get_retry_policy("api_rate_limits")
        assert api_policy["max_attempts"] == 3
        assert api_policy["backoff_strategy"] == "linear"

    def test_get_circuit_breaker_status(self, error_handler):
        """Test circuit breaker status retrieval."""
        status = error_handler.get_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "api_calls" in status
        assert "database_connections" in status

    def test_get_error_patterns(self, error_handler):
        """Test error patterns retrieval."""
        patterns = error_handler.get_error_patterns()
        assert isinstance(patterns, dict)
        assert patterns == error_handler.error_patterns


class TestErrorHandlerDecorator:
    """Test error handler decorator functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.mark.asyncio
    async def test_error_handler_decorator_success(self, config):
        """Test error handler decorator with successful function."""
        @error_handler_decorator("test", "successful_function")
        async def successful_function():
            return "success"

        result = await successful_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_error_handler_decorator_exception(self, config):
        """Test error handler decorator with exception."""
        @error_handler_decorator("test", "failing_function")
        async def failing_function():
            raise ValidationError("Test validation error")

        with pytest.raises(ValidationError):
            await failing_function()

    @pytest.mark.asyncio
    async def test_error_handler_decorator_with_recovery(self, config):
        """Test error handler decorator with recovery strategy."""
        recovery_called = False

        async def mock_recovery(context):
            nonlocal recovery_called
            recovery_called = True
            return True

        @error_handler_decorator("test",
                                 "function_with_recovery",
                                 recovery_strategy=mock_recovery)
        async def function_with_recovery():
            raise ExchangeError("Test exchange error")

        with pytest.raises(ExchangeError):
            await function_with_recovery()

        # Recovery should be called
        assert recovery_called

    def test_error_handler_decorator_sync_function(self, config):
        """Test error handler decorator with synchronous function."""
        @error_handler_decorator("test", "sync_function")
        def sync_function():
            return "sync success"

        result = sync_function()
        assert result == "sync success"

    def test_error_handler_decorator_sync_exception(self, config):
        """Test error handler decorator with synchronous exception."""
        @error_handler_decorator("test", "sync_failing_function")
        def sync_failing_function():
            raise ValidationError("Test validation error")

        with pytest.raises(ValidationError):
            sync_failing_function()


class TestErrorPattern:
    """Test error pattern functionality."""

    def test_error_pattern_creation(self):
        """Test error pattern creation."""
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="exchange",
            error_type="timeout",
            frequency=5.0,
            severity="high",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=5,
            confidence=0.8,
            description="Test pattern",
            suggested_action="Monitor"
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.component == "exchange"
        assert pattern.error_type == "timeout"
        assert pattern.occurrence_count == 5
        assert pattern.severity == "high"

    def test_error_pattern_to_dict(self):
        """Test error pattern to dictionary conversion."""
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="database",
            error_type="connection_failed",
            frequency=3.0,
            severity="medium",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=3,
            confidence=0.7,
            description="Test pattern",
            suggested_action="Investigate"
        )

        pattern_dict = pattern.to_dict()
        assert pattern_dict["pattern_id"] == "test_pattern"
        assert pattern_dict["component"] == "database"
        assert pattern_dict["error_type"] == "connection_failed"
        assert pattern_dict["occurrence_count"] == 3
        assert pattern_dict["severity"] == "medium"

    def test_error_pattern_increment(self):
        """Test error pattern occurrence increment."""
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            component="api",
            error_type="rate_limit",
            frequency=1.0,
            severity="low",
            first_detected=timestamp,
            last_detected=timestamp,
            occurrence_count=1,
            confidence=0.6,
            description="Test pattern",
            suggested_action="Monitor"
        )

        initial_count = pattern.occurrence_count
        pattern.occurrence_count += 1

        assert pattern.occurrence_count == initial_count + 1


class TestErrorHandlerIntegration:
    """Test error handler integration scenarios."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def error_handler(self, config):
        """Provide error handler instance."""
        return ErrorHandler(config)

    @pytest.mark.asyncio
    async def test_error_handler_with_circuit_breaker(self, error_handler):
        """Test error handler with circuit breaker integration."""
        # Create a function that fails
        def failing_function():
            raise ExchangeError("API timeout")

        # Get circuit breaker for API calls
        cb_key = "api_calls"
        circuit_breaker = error_handler.circuit_breakers.get(cb_key)

        # First few failures should not open circuit
        for _ in range(3):
            with pytest.raises(ExchangeError):
                circuit_breaker.call(failing_function)

        # Circuit should still be closed
        assert circuit_breaker.state == "CLOSED"

        # More failures should open circuit
        for _ in range(2):
            with pytest.raises(ExchangeError):
                circuit_breaker.call(failing_function)

        # Circuit should be open
        assert circuit_breaker.state == "OPEN"

    @pytest.mark.asyncio
    async def test_error_handler_pattern_detection(self, error_handler):
        """Test error handler pattern detection."""
        # Create multiple similar errors
        for i in range(3):
            error = ExchangeError(f"API timeout {i}")
            context = error_handler.create_error_context(
                error=error,
                component="exchange",
                operation="place_order"
            )

            await error_handler.handle_error(error, context)

        # Check if pattern was detected
        patterns = error_handler.get_error_patterns()
        assert len(patterns) > 0

        # Should have a pattern for exchange errors
        exchange_patterns = [
            p for p in patterns.values() if p.component == "exchange"]
        assert len(exchange_patterns) > 0

    @pytest.mark.asyncio
    async def test_error_handler_retry_policy(self, error_handler):
        """Test error handler retry policy application."""
        # Test network error retry policy
        network_policy = error_handler.get_retry_policy("network_errors")
        assert network_policy["max_attempts"] == 5
        assert network_policy["backoff_strategy"] == "exponential"
        assert network_policy["base_delay"] == 1
        assert network_policy["max_delay"] == 60
        assert network_policy["jitter"] is True

        # Test API rate limit retry policy
        api_policy = error_handler.get_retry_policy("api_rate_limits")
        assert api_policy["max_attempts"] == 3
        assert api_policy["backoff_strategy"] == "linear"
        assert api_policy["base_delay"] == 5
        assert api_policy["respect_retry_after"] is True

    @pytest.mark.asyncio
    async def test_error_handler_escalation(self, error_handler):
        """Test error handler escalation for critical errors."""
        error = StateConsistencyError("Critical state corruption")
        context = error_handler.create_error_context(
            error=error,
            component="state",
            operation="validate_state"
        )

        with patch.object(error_handler, '_escalate_error') as mock_escalate:
            await error_handler.handle_error(error, context)

            # Should escalate critical errors
            mock_escalate.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_error_handler_context_preservation(self, error_handler):
        """Test error handler context preservation."""
        original_error = ExchangeError("Original error message")
        context = error_handler.create_error_context(
            error=original_error,
            component="exchange",
            operation="place_order",
            user_id="test_user",
            bot_id="test_bot",
            symbol="BTCUSDT"
        )

        # Verify context is preserved
        assert context.error_id is not None
        assert context.component == "exchange"
        assert context.operation == "place_order"
        assert context.user_id == "test_user"
        assert context.bot_id == "test_bot"
        assert context.symbol == "BTCUSDT"
        assert context.severity == ErrorSeverity.HIGH
        assert context.timestamp is not None
        assert context.stack_trace is not None
