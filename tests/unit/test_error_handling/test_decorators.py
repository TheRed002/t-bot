"""
Tests for error handling decorators.

Testing retry, circuit breaker, fallback, and error context decorators.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.core.exceptions import DatabaseConnectionError, NetworkError, ServiceError
from src.error_handling.decorators import (
    CircuitBreakerConfig,
    FallbackConfig,
    FallbackStrategy,
    RetryConfig,
    _calculate_delay,
    _handle_fallback,
    _should_circuit_break,
    _should_retry,
    enhanced_error_handler,
    get_active_handler_count,
    shutdown_all_error_handlers,
    with_circuit_breaker,
    with_error_context,
    with_fallback,
    with_retry,
    circuit_breaker,
    retry,
    fallback,
    _active_handlers,
    _error_counts,
    _circuit_breakers,
)


class TestFallbackStrategy:
    """Test fallback strategy enum."""

    def test_fallback_strategy_values(self):
        """Test fallback strategy enum values."""
        assert FallbackStrategy.NONE.value == "none"
        assert FallbackStrategy.RETURN_NONE.value == "return_none"
        assert FallbackStrategy.RETURN_EMPTY.value == "return_empty"
        assert FallbackStrategy.RAISE_ERROR.value == "raise_error"


class TestRetryConfig:
    """Test retry configuration."""

    def test_retry_config_defaults(self):
        """Test retry configuration default values."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential is True

    def test_retry_config_custom(self):
        """Test retry configuration custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential=False
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential is False


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_circuit_breaker_config_defaults(self):
        """Test circuit breaker configuration default values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.expected_exception is None

    def test_circuit_breaker_config_custom(self):
        """Test circuit breaker configuration custom values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ValueError
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.expected_exception == ValueError


class TestFallbackConfig:
    """Test fallback configuration."""

    def test_fallback_config_defaults(self):
        """Test fallback configuration default values."""
        config = FallbackConfig()
        
        assert config.strategy == FallbackStrategy.RETURN_NONE
        assert config.fallback_value is None
        assert config.fallback_function is None
        assert config.default_value is None

    def test_fallback_config_custom(self):
        """Test fallback configuration custom values."""
        fallback_func = lambda: "fallback"
        config = FallbackConfig(
            strategy=FallbackStrategy.RETURN_EMPTY,
            fallback_value="default",
            fallback_function=fallback_func,
            default_value="default_val"
        )
        
        assert config.strategy == FallbackStrategy.RETURN_EMPTY
        assert config.fallback_value == "default"
        assert config.fallback_function == fallback_func
        assert config.default_value == "default_val"


class TestShouldCircuitBreak:
    """Test circuit breaker logic."""

    def setUp(self):
        """Clear state before each test."""
        _error_counts.clear()

    def test_should_circuit_break_below_threshold(self):
        """Test circuit breaker below failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=5)
        _error_counts["test_func"] = 3
        
        result = _should_circuit_break("test_func", config)
        assert result is False

    def test_should_circuit_break_at_threshold(self):
        """Test circuit breaker at failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=5)
        _error_counts["test_func"] = 5
        
        result = _should_circuit_break("test_func", config)
        assert result is True

    def test_should_circuit_break_above_threshold(self):
        """Test circuit breaker above failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=5)
        _error_counts["test_func"] = 10
        
        result = _should_circuit_break("test_func", config)
        assert result is True

    def test_should_circuit_break_no_errors(self):
        """Test circuit breaker with no errors."""
        config = CircuitBreakerConfig(failure_threshold=5)
        
        result = _should_circuit_break("nonexistent_func", config)
        assert result is False


class TestShouldRetry:
    """Test retry logic."""

    def test_should_retry_no_config(self):
        """Test should retry with no configuration."""
        result = _should_retry(ValueError("test"), None)
        assert result is False

    def test_should_retry_with_specific_exceptions(self):
        """Test should retry with specific exceptions."""
        config = RetryConfig()
        
        # Should retry on specified exception
        result = _should_retry(ValueError("test"), config, (ValueError, TypeError))
        assert result is True
        
        # Should not retry on unspecified exception
        result = _should_retry(RuntimeError("test"), config, (ValueError, TypeError))
        assert result is False

    def test_should_retry_default_exceptions(self):
        """Test should retry with default exceptions."""
        config = RetryConfig()
        
        # Should retry on default exceptions
        result = _should_retry(NetworkError("network failed"), config)
        assert result is True
        
        result = _should_retry(DatabaseConnectionError("db failed"), config)
        assert result is True
        
        result = _should_retry(ServiceError("service failed"), config)
        assert result is True
        
        # Should not retry on other exceptions
        result = _should_retry(ValueError("value error"), config)
        assert result is False


class TestCalculateDelay:
    """Test delay calculation."""

    def test_calculate_delay_no_config(self):
        """Test delay calculation with no configuration."""
        result = _calculate_delay(2, None)
        assert result == 0

    def test_calculate_delay_linear(self):
        """Test linear delay calculation."""
        config = RetryConfig(base_delay=2.0, exponential=False)
        
        result = _calculate_delay(0, config)
        assert result == 2.0
        
        result = _calculate_delay(3, config)
        assert result == 2.0

    def test_calculate_delay_exponential(self):
        """Test exponential delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential=True)
        
        result = _calculate_delay(0, config)
        assert result == 1.0
        
        result = _calculate_delay(1, config)
        assert result == 2.0
        
        result = _calculate_delay(2, config)
        assert result == 4.0

    def test_calculate_delay_max_cap(self):
        """Test delay calculation respects max delay."""
        config = RetryConfig(base_delay=1.0, max_delay=5.0, exponential=True)
        
        result = _calculate_delay(10, config)  # Would be 1024 without cap
        assert result == 5.0


class TestHandleFallback:
    """Test fallback handling."""

    def test_handle_fallback_no_config(self):
        """Test fallback handling with no configuration."""
        result = _handle_fallback(None)
        assert result is None

    def test_handle_fallback_return_none(self):
        """Test fallback returning None."""
        config = FallbackConfig(strategy=FallbackStrategy.RETURN_NONE)
        
        result = _handle_fallback(config)
        assert result is None

    def test_handle_fallback_return_empty(self):
        """Test fallback returning empty dict."""
        config = FallbackConfig(strategy=FallbackStrategy.RETURN_EMPTY)
        
        result = _handle_fallback(config)
        assert result == {}

    def test_handle_fallback_raise_error(self):
        """Test fallback raising error."""
        config = FallbackConfig(strategy=FallbackStrategy.RAISE_ERROR)
        
        with pytest.raises(ServiceError, match="Circuit breaker open"):
            _handle_fallback(config)

    def test_handle_fallback_function(self):
        """Test fallback with function."""
        def fallback_func():
            return "fallback_result"
        
        config = FallbackConfig(
            strategy=FallbackStrategy.NONE,  # Use NONE to avoid RETURN_NONE
            fallback_function=fallback_func
        )
        
        result = _handle_fallback(config)
        assert result == "fallback_result"

    def test_handle_fallback_value(self):
        """Test fallback with value."""
        config = FallbackConfig(
            strategy=FallbackStrategy.NONE,  # Use NONE to avoid RETURN_NONE
            fallback_value="fallback_value"
        )
        
        result = _handle_fallback(config)
        assert result == "fallback_value"


class TestEnhancedErrorHandler:
    """Test enhanced error handler decorator."""

    def setUp(self):
        """Clear state before each test."""
        _error_counts.clear()
        _active_handlers.clear()

    def test_enhanced_error_handler_sync_success(self):
        """Test enhanced error handler with sync function success."""
        @enhanced_error_handler()
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"

    def test_enhanced_error_handler_sync_with_retry(self):
        """Test enhanced error handler with sync function and retry."""
        call_count = 0
        
        @enhanced_error_handler(retry_config=RetryConfig(max_attempts=3))
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Network failed")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_enhanced_error_handler_async_success(self):
        """Test enhanced error handler with async function success."""
        @enhanced_error_handler()
        async def test_func():
            return "async_success"
        
        result = await test_func()
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_enhanced_error_handler_async_with_retry(self):
        """Test enhanced error handler with async function and retry."""
        call_count = 0
        
        @enhanced_error_handler(retry_config=RetryConfig(max_attempts=2, base_delay=0.01))
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise DatabaseConnectionError("DB failed")
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 2

    def test_enhanced_error_handler_with_circuit_breaker(self):
        """Test enhanced error handler with circuit breaker."""
        func_name = f"{TestEnhancedErrorHandler.__module__}.TestEnhancedErrorHandler.test_enhanced_error_handler_with_circuit_breaker.<locals>.test_func"
        _error_counts[func_name] = 10  # Above threshold
        
        @enhanced_error_handler(
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
            fallback_config=FallbackConfig(strategy=FallbackStrategy.RETURN_NONE)
        )
        def test_func():
            return "should_not_reach"
        
        result = test_func()
        assert result is None

    def test_enhanced_error_handler_with_fallback_on_failure(self):
        """Test enhanced error handler with fallback on all retries failed."""
        @enhanced_error_handler(
            retry_config=RetryConfig(max_attempts=2),
            fallback_config=FallbackConfig(
                strategy=FallbackStrategy.NONE,
                fallback_value="fallback_result"
            )
        )
        def test_func():
            raise ValueError("Always fails")
        
        result = test_func()
        assert result == "fallback_result"

    def test_enhanced_error_handler_reraises_without_fallback(self):
        """Test enhanced error handler reraises error without fallback."""
        @enhanced_error_handler(retry_config=RetryConfig(max_attempts=1))
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            test_func()

    def test_enhanced_error_handler_logging_disabled(self):
        """Test enhanced error handler with logging disabled."""
        with patch('src.error_handling.decorators.logger') as mock_logger:
            @enhanced_error_handler(
                retry_config=RetryConfig(max_attempts=1),
                enable_logging=False
            )
            def test_func():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                test_func()
            
            mock_logger.warning.assert_not_called()

    @patch('src.error_handling.decorators.logger')
    def test_enhanced_error_handler_logging_enabled(self, mock_logger):
        """Test enhanced error handler with logging enabled."""
        @enhanced_error_handler(
            retry_config=RetryConfig(max_attempts=2),
            enable_logging=True
        )
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_func()
        
        mock_logger.warning.assert_called()


class TestWithRetry:
    """Test retry decorator."""

    def test_with_retry_defaults(self):
        """Test retry decorator with default values."""
        @with_retry()
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"

    def test_with_retry_custom_params(self):
        """Test retry decorator with custom parameters."""
        call_count = 0
        
        @with_retry(max_attempts=2, base_delay=0.01, exponential=False)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("Network failed")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 2

    def test_with_retry_specific_exceptions(self):
        """Test retry decorator with specific exceptions."""
        @with_retry(max_attempts=2, exceptions=(ValueError,))
        def test_func():
            raise NetworkError("Should not retry")
        
        with pytest.raises(NetworkError):
            test_func()

    def test_with_retry_backoff_factor_ignored(self):
        """Test retry decorator ignores backoff_factor parameter."""
        @with_retry(max_attempts=1, backoff_factor=2.0)
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"


class TestWithCircuitBreaker:
    """Test circuit breaker decorator."""

    def test_with_circuit_breaker_defaults(self):
        """Test circuit breaker decorator with defaults."""
        @with_circuit_breaker()
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"

    def test_with_circuit_breaker_custom_params(self):
        """Test circuit breaker decorator with custom parameters."""
        @with_circuit_breaker(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ValueError
        )
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"

    def test_with_circuit_breaker_additional_kwargs(self):
        """Test circuit breaker decorator handles additional kwargs."""
        @with_circuit_breaker(failure_threshold=5, extra_param="ignored")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"


class TestWithFallback:
    """Test fallback decorator."""

    def test_with_fallback_defaults(self):
        """Test fallback decorator with defaults."""
        @with_fallback()
        def test_func():
            raise ValueError("Error")
        
        result = test_func()
        assert result is None

    def test_with_fallback_custom_strategy(self):
        """Test fallback decorator with custom strategy."""
        @with_fallback(strategy=FallbackStrategy.RETURN_EMPTY)
        def test_func():
            raise ValueError("Error")
        
        result = test_func()
        assert result == {}

    def test_with_fallback_value(self):
        """Test fallback decorator with fallback value."""
        @with_fallback(
            strategy=FallbackStrategy.NONE,
            fallback_value="fallback"
        )
        def test_func():
            raise ValueError("Error")
        
        result = test_func()
        assert result == "fallback"

    def test_with_fallback_function(self):
        """Test fallback decorator with fallback function."""
        def fallback_func():
            return "function_result"
        
        @with_fallback(
            strategy=FallbackStrategy.NONE,
            fallback_function=fallback_func
        )
        def test_func():
            raise ValueError("Error")
        
        result = test_func()
        assert result == "function_result"

    def test_with_fallback_default_value_priority(self):
        """Test fallback decorator prioritizes default_value over fallback_value."""
        @with_fallback(
            strategy=FallbackStrategy.NONE,
            fallback_value="fallback", 
            default_value="default"
        )
        def test_func():
            raise ValueError("Error")
        
        result = test_func()
        assert result == "default"


class TestWithErrorContext:
    """Test error context decorator."""

    def test_with_error_context_sync_success(self):
        """Test error context decorator with sync function success."""
        @with_error_context(component="test", operation="sync_test")
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"

    def test_with_error_context_sync_error(self):
        """Test error context decorator adds context to sync function error."""
        @with_error_context(component="test", operation="sync_error")
        def test_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        assert hasattr(exc_info.value, 'context')
        assert exc_info.value.context['component'] == 'test'
        assert exc_info.value.context['operation'] == 'sync_error'

    @pytest.mark.asyncio
    async def test_with_error_context_async_success(self):
        """Test error context decorator with async function success."""
        @with_error_context(component="test", operation="async_test")
        async def test_func():
            return "async_success"
        
        result = await test_func()
        assert result == "async_success"

    @pytest.mark.asyncio
    async def test_with_error_context_async_error(self):
        """Test error context decorator adds context to async function error."""
        @with_error_context(component="test", operation="async_error")
        async def test_func():
            raise ValueError("Async test error")
        
        with pytest.raises(ValueError) as exc_info:
            await test_func()
        
        assert hasattr(exc_info.value, 'context')
        assert exc_info.value.context['component'] == 'test'
        assert exc_info.value.context['operation'] == 'async_error'

    def test_with_error_context_existing_context(self):
        """Test error context decorator updates existing context."""
        @with_error_context(new_key="new_value")
        def test_func():
            error = ValueError("Test error")
            error.context = {"existing_key": "existing_value"}
            raise error
        
        with pytest.raises(ValueError) as exc_info:
            test_func()
        
        assert exc_info.value.context['existing_key'] == 'existing_value'
        assert exc_info.value.context['new_key'] == 'new_value'


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_active_handler_count(self):
        """Test getting active handler count."""
        _active_handlers.clear()
        assert get_active_handler_count() == 0
        
        _active_handlers.add("handler1")
        _active_handlers.add("handler2")
        assert get_active_handler_count() == 2

    def test_shutdown_all_error_handlers(self):
        """Test shutting down all error handlers."""
        _active_handlers.add("handler1")
        _error_counts["func1"] = 5
        _circuit_breakers["cb1"] = {"state": "open"}
        
        shutdown_all_error_handlers()
        
        assert len(_active_handlers) == 0
        assert len(_error_counts) == 0
        assert len(_circuit_breakers) == 0


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_circuit_breaker_alias(self):
        """Test circuit_breaker is alias for with_circuit_breaker."""
        assert circuit_breaker is with_circuit_breaker

    def test_retry_alias(self):
        """Test retry is alias for with_retry."""
        assert retry is with_retry

    def test_fallback_alias(self):
        """Test fallback is alias for with_fallback."""
        assert fallback is with_fallback