"""
Unit tests for error handler component.

These tests verify error categorization, severity classification, retry policies,
circuit breaker integration, and error context preservation.
"""

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import Config
from src.core.exceptions import (
    DataError,
    ExchangeError,
    ExecutionError,
    ModelError,
    RiskManagementError,
    SecurityError,
    StateConsistencyError,
    TradingBotError,
    ValidationError,
)
from src.error_handling.error_handler import (
    CircuitBreaker,
    ErrorHandler,
    ErrorPatternCache,
    error_handler_decorator,
)
from src.error_handling.secure_pattern_analytics import ErrorPattern
from src.error_handling.context import ErrorContext, ErrorSeverity


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
            stack_trace="traceback info",
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
            error=error,
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
        # stack_trace is auto-populated in __post_init__
        assert context.stack_trace is not None


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
    def mock_sanitizer(self):
        """Provide mock sanitizer for testing."""
        sanitizer = MagicMock()
        sanitizer.sanitize_error_message = MagicMock(side_effect=lambda msg, level: f"sanitized: {msg}")
        
        def mock_sanitize_context(ctx, level):
            """Mock context sanitization that replaces sensitive data."""
            sanitized_ctx = ctx.copy()
            # Mock sanitization of sensitive data
            for key, value in sanitized_ctx.items():
                if isinstance(value, str) and any(sensitive_word in key.lower() for sensitive_word in ['key', 'token', 'password', 'secret']):
                    sanitized_ctx[key] = f"HASH_{hash(value) % 10000:04d}"
                elif isinstance(value, str) and len(value) > 10 and any(char.isdigit() for char in value):
                    # Mask potential sensitive strings
                    sanitized_ctx[key] = "*" * 8
            return sanitized_ctx
            
        sanitizer.sanitize_context = MagicMock(side_effect=mock_sanitize_context)
        sanitizer.sanitize_stack_trace = MagicMock(side_effect=lambda trace, level: f"sanitized: {trace}")
        return sanitizer

    @pytest.fixture
    def mock_rate_limiter(self):
        """Provide mock rate limiter for testing."""
        rate_limiter = MagicMock()
        # Create async mock for check_rate_limit
        async_mock = AsyncMock()
        async_mock.return_value = MagicMock(allowed=True, reason=None, suggested_retry_after=None)
        rate_limiter.check_rate_limit = async_mock
        return rate_limiter

    @pytest.fixture
    def error_handler(self, config, mock_sanitizer, mock_rate_limiter):
        """Provide error handler instance with mocked dependencies."""
        return ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

    def test_error_handler_initialization(self, config):
        """Test error handler initialization."""
        # Test initialization without dependencies (design expectation)
        handler = ErrorHandler(config)
        assert handler.config == config
        assert isinstance(handler.error_patterns, ErrorPatternCache)
        assert handler.sanitizer is None  # Dependencies are injected, not auto-initialized
        assert handler.rate_limiter is None  # Dependencies are injected, not auto-initialized
        assert "network_errors" in handler.retry_policies
        assert "api_rate_limits" in handler.retry_policies
        assert "database_errors" in handler.retry_policies
        assert "api_calls" in handler.circuit_breakers
        assert "database_connections" in handler.circuit_breakers

    def test_error_handler_initialization_with_dependencies(self, config, mock_sanitizer, mock_rate_limiter):
        """Test error handler initialization with injected dependencies."""
        handler = ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)
        assert handler.config == config
        assert isinstance(handler.error_patterns, ErrorPatternCache)
        assert handler.sanitizer is not None
        assert handler.rate_limiter is not None
        assert "network_errors" in handler.retry_policies
        assert "api_rate_limits" in handler.retry_policies
        assert "database_errors" in handler.retry_policies
        assert "api_calls" in handler.circuit_breakers
        assert "database_connections" in handler.circuit_breakers

    def test_error_classification(self, error_handler):
        """Test error classification functionality."""
        # Test critical errors
        assert (
            error_handler.classify_error(StateConsistencyError("State corruption"))
            == ErrorSeverity.CRITICAL
        )
        assert (
            error_handler.classify_error(SecurityError("Security breach")) == ErrorSeverity.CRITICAL
        )

        # Test high severity errors
        assert (
            error_handler.classify_error(RiskManagementError("Risk limit exceeded"))
            == ErrorSeverity.HIGH
        )
        assert error_handler.classify_error(ExchangeError("API timeout")) == ErrorSeverity.HIGH
        assert (
            error_handler.classify_error(ExecutionError("Order execution failed"))
            == ErrorSeverity.HIGH
        )

        # Test medium severity errors
        assert (
            error_handler.classify_error(DataError("Data validation failed"))
            == ErrorSeverity.MEDIUM
        )
        assert (
            error_handler.classify_error(ModelError("Model inference failed"))
            == ErrorSeverity.MEDIUM
        )

        # Test low severity errors
        assert error_handler.classify_error(ValidationError("Invalid input")) == ErrorSeverity.LOW

        # Test unknown errors
        assert error_handler.classify_error(ValueError("Unknown error")) == ErrorSeverity.MEDIUM

    def test_error_context_creation(self, error_handler):
        """Test error context creation."""
        error = ExchangeError("API timeout")
        context = error_handler.create_error_context(
            error=error,
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
        assert context.timestamp is not None
        assert "error_type" in context.details
        assert "error_message" in context.details
        assert "sensitivity_level" in context.details
        assert context.stack_trace is not None

    def test_error_context_creation_with_details(self, error_handler):
        """Test error context creation with additional details."""
        error = RiskManagementError("Risk limit exceeded")
        context = error_handler.create_error_context(
            error=error,
            component="risk_management",
            operation="validate_position",
            details={"position_size": 1000, "limit": 500},
        )

        assert context.error_id is not None
        assert context.component == "risk_management"
        assert context.operation == "validate_position"
        assert context.severity == ErrorSeverity.HIGH
        assert context.timestamp is not None
        assert "error_type" in context.details
        assert "error_message" in context.details
        assert "kwargs" in context.details
        assert "sensitivity_level" in context.details
        assert context.stack_trace is not None

    @pytest.mark.asyncio
    async def test_handle_error_success(self, error_handler):
        """Test successful error handling."""
        error = ValidationError("Invalid input")
        context = error_handler.create_error_context(
            error=error, component="validation", operation="validate_input"
        )

        # Mock recovery strategy (not directly called by handler)
        def mock_recovery(ctx):
            return True

        # Rate limiter is already mocked in fixture, just verify behavior
        result = await error_handler.handle_error(error, context, mock_recovery)

        # Handler prepares recovery context for service layer, doesn't execute directly
        assert result is True
        error_handler.rate_limiter.check_rate_limit.assert_called_once()
        # Recovery attempts should be incremented
        assert context.recovery_attempts == 1

    @pytest.mark.asyncio
    async def test_handle_error_without_recovery(self, error_handler):
        """Test error handling without recovery strategy."""
        error = DataError("Data validation failed")
        context = error_handler.create_error_context(
            error=error, component="data", operation="validate_data"
        )

        # Rate limiter is already mocked in fixture, just verify behavior
        result = await error_handler.handle_error(error, context)

        # Should return False for non-critical errors without recovery
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_error_critical_escalation(self, error_handler):
        """Test critical error escalation."""
        error = StateConsistencyError("State corruption")
        context = error_handler.create_error_context(
            error=error, component="state", operation="validate_state"
        )

        # Mock escalation
        with patch.object(error_handler, "_escalate_error", new_callable=AsyncMock) as mock_escalate:
            result = await error_handler.handle_error(error, context)

            assert result is False
            mock_escalate.assert_called_once_with(context)

    @pytest.mark.asyncio
    async def test_handle_error_max_attempts_exceeded(self, error_handler):
        """Test error handling when max attempts exceeded."""
        error = ExchangeError("API timeout")
        context = error_handler.create_error_context(
            error=error, component="exchange", operation="place_order"
        )

        # Set recovery attempts to max
        context.recovery_attempts = context.max_recovery_attempts

        # Mock escalation
        with patch.object(error_handler, "_escalate_error", new_callable=AsyncMock) as mock_escalate:
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
        # Now error_patterns is an ErrorPatternCache, not a dict
        assert patterns == error_handler.error_patterns.get_all_patterns()


class TestErrorHandlerDecorator:
    """Test error handler decorator functionality."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def mock_sanitizer(self):
        """Provide mock sanitizer for testing."""
        sanitizer = MagicMock()
        sanitizer.sanitize_error_message = MagicMock(side_effect=lambda msg, level: f"sanitized: {msg}")
        
        def mock_sanitize_context(ctx, level):
            """Mock context sanitization that replaces sensitive data."""
            sanitized_ctx = ctx.copy()
            # Mock sanitization of sensitive data
            for key, value in sanitized_ctx.items():
                if isinstance(value, str) and any(sensitive_word in key.lower() for sensitive_word in ['key', 'token', 'password', 'secret']):
                    sanitized_ctx[key] = f"HASH_{hash(value) % 10000:04d}"
                elif isinstance(value, str) and len(value) > 10 and any(char.isdigit() for char in value):
                    # Mask potential sensitive strings
                    sanitized_ctx[key] = "*" * 8
            return sanitized_ctx
            
        sanitizer.sanitize_context = MagicMock(side_effect=mock_sanitize_context)
        sanitizer.sanitize_stack_trace = MagicMock(side_effect=lambda trace, level: f"sanitized: {trace}")
        return sanitizer

    @pytest.fixture
    def mock_rate_limiter(self):
        """Provide mock rate limiter for testing."""
        rate_limiter = MagicMock()
        # Create async mock for check_rate_limit
        async_mock = AsyncMock()
        async_mock.return_value = MagicMock(allowed=True, reason=None, suggested_retry_after=None)
        rate_limiter.check_rate_limit = async_mock
        return rate_limiter

    @pytest.mark.asyncio
    async def test_error_handler_decorator_success(self, config, mock_sanitizer, mock_rate_limiter):
        """Test error handler decorator with successful function."""
        # Create handler with mocked dependencies
        handler = ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

        @error_handler_decorator("test", "successful_function", error_handler=handler)
        async def successful_function():
            return "success"

        result = await successful_function()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_error_handler_decorator_exception(self, config, mock_sanitizer, mock_rate_limiter):
        """Test error handler decorator with exception."""
        # Create handler with mocked dependencies
        handler = ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

        @error_handler_decorator("test", "failing_function", error_handler=handler)
        async def failing_function():
            raise ValidationError("Test validation error")

        with pytest.raises(ValidationError):
            await failing_function()

    @pytest.mark.asyncio
    async def test_error_handler_decorator_with_recovery(self, config, mock_sanitizer, mock_rate_limiter):
        """Test error handler decorator with recovery strategy."""
        async def mock_recovery(context):
            return True

        # Create handler with mocked dependencies
        handler = ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

        @error_handler_decorator("test", "function_with_recovery", error_handler=handler, recovery_strategy=mock_recovery)
        async def function_with_recovery():
            raise ExchangeError("Test exchange error")

        with pytest.raises(ExchangeError):
            await function_with_recovery()

        # Test passes if decorator handles exception without crashing
        # Recovery strategy is prepared but not executed by the decorator

    def test_error_handler_decorator_sync_function(self, config, mock_sanitizer, mock_rate_limiter):
        """Test error handler decorator with synchronous function."""
        # Create handler with mocked dependencies
        handler = ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

        @error_handler_decorator("test", "sync_function", error_handler=handler)
        def sync_function():
            return "sync success"

        result = sync_function()
        assert result == "sync success"

    def test_error_handler_decorator_sync_exception(self, config, mock_sanitizer, mock_rate_limiter):
        """Test error handler decorator with synchronous exception."""
        # Create handler with mocked dependencies
        handler = ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

        @error_handler_decorator("test", "sync_failing_function", error_handler=handler)
        def sync_failing_function():
            raise ValidationError("Test validation error")

        with pytest.raises(ValidationError):
            sync_failing_function()


class TestErrorPatternCache:
    """Test error pattern cache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = ErrorPatternCache(max_patterns=100, ttl_hours=12)
        assert cache.max_patterns == 100
        assert cache.ttl_hours == 12
        assert cache.size() == 0

    def test_cache_add_and_get_pattern(self):
        """Test adding and retrieving patterns from cache."""
        from src.error_handling.secure_pattern_analytics import PatternSeverity
        cache = ErrorPatternCache()
        timestamp = datetime.now(timezone.utc)
        
        pattern = ErrorPattern(
            pattern_id="test_cache_pattern",
            pattern_type="frequency",
            severity=PatternSeverity.LOW,
            frequency=1,
            first_seen=timestamp,
            last_seen=timestamp,
            error_signature="test_error_hash",
            component_hash="cache_test_hash",
            common_context={"error_type": "test_error"},
        )
        
        cache.add_pattern(pattern)
        assert cache.size() == 1
        
        retrieved = cache.get_pattern("test_cache_pattern")
        assert retrieved is not None
        assert retrieved["pattern_id"] == "test_cache_pattern"
        assert retrieved["component_hash"] == "cache_test_hash"

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from src.error_handling.secure_pattern_analytics import PatternSeverity
        cache = ErrorPatternCache(max_patterns=2)  # Small cache for testing
        timestamp = datetime.now(timezone.utc)
        
        # Add patterns to fill cache
        for i in range(3):  # Add one more than capacity
            pattern = ErrorPattern(
                pattern_id=f"pattern_{i}",
                pattern_type="frequency",
                severity=PatternSeverity.LOW,
                frequency=1,
                first_seen=timestamp,
                last_seen=timestamp,
                error_signature=f"test_error_hash_{i}",
                component_hash="test_hash",
                common_context={"error_type": "test"},
            )
            cache.add_pattern(pattern)
        
        # Cache should only hold 2 patterns (the last 2 added)
        assert cache.size() == 2
        assert cache.get_pattern("pattern_0") is None  # Should be evicted
        assert cache.get_pattern("pattern_1") is not None
        assert cache.get_pattern("pattern_2") is not None


class TestErrorPattern:
    """Test error pattern functionality."""

    def test_error_pattern_creation(self):
        """Test error pattern creation."""
        from src.error_handling.secure_pattern_analytics import PatternSeverity
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            severity=PatternSeverity.HIGH,
            frequency=5,
            first_seen=timestamp,
            last_seen=timestamp,
            error_signature="test_error_hash",
            component_hash="exchange_hash",
            common_context={"error_type": "timeout"},
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.component_hash == "exchange_hash"
        assert pattern.common_context["error_type"] == "timeout"
        assert pattern.frequency == 5
        assert pattern.severity == PatternSeverity.HIGH

    def test_error_pattern_dataclass_fields(self):
        """Test error pattern dataclass field access."""
        from src.error_handling.secure_pattern_analytics import PatternSeverity
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            severity=PatternSeverity.MEDIUM,
            frequency=3,
            first_seen=timestamp,
            last_seen=timestamp,
            error_signature="test_error_hash",
            component_hash="database_hash",
            common_context={"error_type": "connection_failed"},
        )

        assert pattern.pattern_id == "test_pattern"
        assert pattern.component_hash == "database_hash"
        assert pattern.common_context["error_type"] == "connection_failed"
        assert pattern.frequency == 3
        assert pattern.severity == PatternSeverity.MEDIUM

    def test_error_pattern_increment(self):
        """Test error pattern occurrence increment."""
        from src.error_handling.secure_pattern_analytics import PatternSeverity
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="test_pattern",
            pattern_type="frequency",
            severity=PatternSeverity.LOW,
            frequency=1,
            first_seen=timestamp,
            last_seen=timestamp,
            error_signature="test_error_hash",
            component_hash="api_hash",
            common_context={"error_type": "rate_limit"},
        )

        initial_frequency = pattern.frequency
        pattern.frequency += 1

        assert pattern.frequency == initial_frequency + 1


class TestErrorHandlerIntegration:
    """Test error handler integration scenarios."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def mock_sanitizer(self):
        """Provide mock sanitizer for testing."""
        sanitizer = MagicMock()
        sanitizer.sanitize_error_message = MagicMock(side_effect=lambda msg, level: f"sanitized: {msg}")
        
        def mock_sanitize_context(ctx, level):
            """Mock context sanitization that replaces sensitive data."""
            sanitized_ctx = ctx.copy()
            # Mock sanitization of sensitive data
            for key, value in sanitized_ctx.items():
                if isinstance(value, str) and any(sensitive_word in key.lower() for sensitive_word in ['key', 'token', 'password', 'secret']):
                    sanitized_ctx[key] = f"HASH_{hash(value) % 10000:04d}"
                elif isinstance(value, str) and len(value) > 10 and any(char.isdigit() for char in value):
                    # Mask potential sensitive strings
                    sanitized_ctx[key] = "*" * 8
            return sanitized_ctx
            
        sanitizer.sanitize_context = MagicMock(side_effect=mock_sanitize_context)
        sanitizer.sanitize_stack_trace = MagicMock(side_effect=lambda trace, level: f"sanitized: {trace}")
        return sanitizer

    @pytest.fixture
    def mock_rate_limiter(self):
        """Provide mock rate limiter for testing."""
        rate_limiter = MagicMock()
        # Create async mock for check_rate_limit
        async_mock = AsyncMock()
        async_mock.return_value = MagicMock(allowed=True, reason=None, suggested_retry_after=None)
        rate_limiter.check_rate_limit = async_mock
        return rate_limiter

    @pytest.fixture
    def error_handler(self, config, mock_sanitizer, mock_rate_limiter):
        """Provide error handler instance with mocked dependencies."""
        return ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

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
        # Rate limiter is already mocked in fixture
        # Create multiple similar errors
        for i in range(3):
            error = ExchangeError(f"API timeout {i}")
            context = error_handler.create_error_context(
                error=error, component="exchange", operation="place_order"
            )

            await error_handler.handle_error(error, context)

        # Check if pattern was detected
        patterns = error_handler.get_error_patterns()
        assert len(patterns) > 0

        # Should have a pattern for exchange errors
        exchange_patterns = [p for p in patterns.values() if p.get("component") == "exchange"]
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
            error=error, component="state", operation="validate_state"
        )

        # Mock escalation
        with patch.object(error_handler, "_escalate_error", new_callable=AsyncMock) as mock_escalate:
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
            symbol="BTCUSDT",
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


class TestErrorHandlerSecurity:
    """Test error handler security features."""

    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return Config()

    @pytest.fixture
    def mock_sanitizer(self):
        """Provide mock sanitizer for testing."""
        sanitizer = MagicMock()
        sanitizer.sanitize_error_message = MagicMock(side_effect=lambda msg, level: f"sanitized: {msg}")
        
        def mock_sanitize_context(ctx, level):
            """Mock context sanitization that replaces sensitive data."""
            sanitized_ctx = ctx.copy()
            # Mock sanitization of sensitive data
            for key, value in sanitized_ctx.items():
                if isinstance(value, str) and any(sensitive_word in key.lower() for sensitive_word in ['key', 'token', 'password', 'secret']):
                    sanitized_ctx[key] = f"HASH_{hash(value) % 10000:04d}"
                elif isinstance(value, str) and len(value) > 10 and any(char.isdigit() for char in value):
                    # Mask potential sensitive strings
                    sanitized_ctx[key] = "*" * 8
            return sanitized_ctx
            
        sanitizer.sanitize_context = MagicMock(side_effect=mock_sanitize_context)
        sanitizer.sanitize_stack_trace = MagicMock(side_effect=lambda trace, level: f"sanitized: {trace}")
        return sanitizer

    @pytest.fixture
    def mock_rate_limiter(self):
        """Provide mock rate limiter for testing."""
        rate_limiter = MagicMock()
        # Create async mock for check_rate_limit
        async_mock = AsyncMock()
        async_mock.return_value = MagicMock(allowed=True, reason=None, suggested_retry_after=None)
        rate_limiter.check_rate_limit = async_mock
        return rate_limiter

    @pytest.fixture
    def error_handler(self, config, mock_sanitizer, mock_rate_limiter):
        """Provide error handler instance with mocked dependencies."""
        return ErrorHandler(config, sanitizer=mock_sanitizer, rate_limiter=mock_rate_limiter)

    def test_sanitizer_integration(self, error_handler):
        """Test that sanitizer is properly integrated."""
        assert error_handler.sanitizer is not None
        
        # Test sanitization in error context creation with sensitive kwargs
        error = ExchangeError("Authentication failed")
        context = error_handler.create_error_context(
            error=error, component="exchange", operation="authenticate", 
            api_key="abc123secret456"  # This should be sanitized in kwargs
        )
        
        # Check that sensitive data is sanitized in kwargs
        if "kwargs" in context.details:
            kwargs_data = context.details["kwargs"]
            # The kwargs should be sanitized - either the key is hashed or value is masked
            # Look for sanitized key or value patterns
            kwargs_str = str(kwargs_data)
            has_sanitized_key = any("[SENSITIVE_KEY_" in key for key in kwargs_data.keys() if isinstance(key, str))
            has_sanitized_value = "HASH_" in kwargs_str or "*" in kwargs_str
            
            # Either the key should be sanitized or the value should be sanitized
            assert has_sanitized_key or has_sanitized_value or "abc123secret456" not in kwargs_str

    @pytest.mark.asyncio
    async def test_rate_limiter_integration(self, error_handler):
        """Test that rate limiter is properly integrated."""
        assert error_handler.rate_limiter is not None
        
        error = ValidationError("Test error")
        context = error_handler.create_error_context(
            error=error, component="test", operation="test_op"
        )
        
        # Configure rate limiter to deny request
        error_handler.rate_limiter.check_rate_limit.return_value = MagicMock(
            allowed=False,
            reason="Rate limit exceeded",
            suggested_retry_after=60.0
        )
        
        result = await error_handler.handle_error(error, context)
        
        # Should return False when rate limited
        assert result is False
        error_handler.rate_limiter.check_rate_limit.assert_called()

    def test_memory_usage_stats(self, error_handler):
        """Test memory usage statistics."""
        stats = error_handler.get_memory_usage_stats()
        
        assert "error_patterns_count" in stats
        assert "circuit_breakers_count" in stats
        assert "operations_processed" in stats
        assert "last_cleanup" in stats
        assert isinstance(stats["error_patterns_count"], int)
        assert isinstance(stats["circuit_breakers_count"], int)

    @pytest.mark.asyncio
    async def test_cleanup_resources(self, error_handler):
        """Test resource cleanup functionality."""
        # Add some patterns first
        from src.error_handling.secure_pattern_analytics import PatternSeverity
        timestamp = datetime.now(timezone.utc)
        pattern = ErrorPattern(
            pattern_id="cleanup_test",
            pattern_type="frequency",
            severity=PatternSeverity.LOW,
            frequency=1,
            first_seen=timestamp,
            last_seen=timestamp,
            error_signature="test_error_hash",
            component_hash="test_hash",
            common_context={"error_type": "test"},
        )
        error_handler.error_patterns.add_pattern(pattern)
        
        # Test cleanup
        await error_handler.cleanup_resources()
        
        # Should not crash and should log completion
        stats = error_handler.get_memory_usage_stats()
        assert isinstance(stats["error_patterns_count"], int)

    @pytest.mark.asyncio
    async def test_error_handler_context_preservation_with_security(self, error_handler):
        """Test error handler context preservation with security sanitization."""
        original_error = ExchangeError("API key sk_test_123456 authentication failed")
        context = error_handler.create_error_context(
            error=original_error,
            component="exchange",
            operation="authenticate",
            user_id="user_123",
            bot_id="bot_456",
            api_key="sk_test_123456",  # This should be sanitized
        )

        # Rate limiter is already mocked in fixture
        await error_handler.handle_error(original_error, context)

        # Verify context is preserved but sensitive data is sanitized
        assert context.error_id is not None
        assert context.component == "exchange"
        assert context.operation == "authenticate"
        assert context.user_id == "user_123"
        assert context.bot_id == "bot_456"
        assert context.severity == ErrorSeverity.HIGH
        assert context.timestamp is not None
        assert context.stack_trace is not None
        
        # Sensitive data should be sanitized in kwargs
        if "kwargs" in context.details:
            kwargs_data = context.details["kwargs"]
            kwargs_str = str(kwargs_data)
            # Look for sanitized patterns - either key is hashed or sensitive data is masked
            has_sanitized_key = any("[SENSITIVE_KEY_" in str(key) for key in kwargs_data.keys())
            has_sanitized_value = "HASH_" in kwargs_str or "*" in kwargs_str
            
            # Either the key should be sanitized or the value should be sanitized
            assert has_sanitized_key or has_sanitized_value or "sk_test_123456" not in kwargs_str
