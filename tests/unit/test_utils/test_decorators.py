"""
Unit tests for decorators module.

This module tests all decorators in the utils.decorators module to ensure
they work correctly with both sync and async functions, handle errors properly,
and provide the expected functionality.
"""

import asyncio
import time

import pytest

from src.core.exceptions import TimeoutError, ValidationError
from src.utils.decorators import (
    api_throttle,
    cache_result,
    circuit_breaker,
    cpu_usage,
    log_calls,
    log_errors,
    log_performance,
    memory_usage,
    rate_limit,
    redis_cache,
    retry,
    time_execution,
    timeout,
    type_check,
    validate_input,
    validate_output,
)


class TestTimeExecution:
    """Test the time_execution decorator."""

    @pytest.mark.asyncio
    async def test_time_execution_async_success(self):
        """Test time_execution with successful async function."""

        @time_execution
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_time_execution_async_error(self):
        """Test time_execution with async function that raises error."""

        @time_execution
        async def test_func():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await test_func()

    def test_time_execution_sync_success(self):
        """Test time_execution with successful sync function."""

        @time_execution
        def test_func():
            time.sleep(0.01)
            return "success"

        result = test_func()
        assert result == "success"

    def test_time_execution_sync_error(self):
        """Test time_execution with sync function that raises error."""

        @time_execution
        def test_func():
            time.sleep(0.01)
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()


class TestMemoryUsage:
    """Test the memory_usage decorator."""

    @pytest.mark.asyncio
    async def test_memory_usage_async(self):
        """Test memory_usage with async function."""

        @memory_usage
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    def test_memory_usage_sync(self):
        """Test memory_usage with sync function."""

        @memory_usage
        def test_func():
            time.sleep(0.01)
            return "success"

        result = test_func()
        assert result == "success"


class TestCpuUsage:
    """Test the cpu_usage decorator."""

    @pytest.mark.asyncio
    async def test_cpu_usage_async(self):
        """Test cpu_usage with async function."""

        @cpu_usage
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    def test_cpu_usage_sync(self):
        """Test cpu_usage with sync function."""

        @cpu_usage
        def test_func():
            time.sleep(0.01)
            return "success"

        result = test_func()
        assert result == "success"


class TestRetry:
    """Test the retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_async_success_first_try(self):
        """Test retry with async function that succeeds on first try."""

        @retry(max_attempts=3)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_async_success_after_failures(self):
        """Test retry with async function that succeeds after failures."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_max_attempts_exceeded(self):
        """Test retry with async function that always fails."""

        @retry(max_attempts=2, base_delay=0.01)
        async def test_func():
            raise ValueError("permanent error")

        with pytest.raises(ValueError, match="permanent error"):
            await test_func()

    def test_retry_sync_success_first_try(self):
        """Test retry with sync function that succeeds on first try."""

        @retry(max_attempts=3)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_retry_sync_success_after_failures(self):
        """Test retry with sync function that succeeds after failures."""
        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3


class TestCircuitBreaker:
    """Test the circuit_breaker decorator."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_async_success(self):
        """Test circuit_breaker with async function that succeeds."""

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_async_opens_circuit(self):
        """Test circuit_breaker with async function that opens circuit."""

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        async def test_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            await test_func()

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await test_func()

        # Third call - circuit should be open
        with pytest.raises(TimeoutError):
            await test_func()

    def test_circuit_breaker_sync_success(self):
        """Test circuit_breaker with sync function that succeeds."""

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_circuit_breaker_sync_opens_circuit(self):
        """Test circuit_breaker with sync function that opens circuit."""

        @circuit_breaker(failure_threshold=2, recovery_timeout=0.1)
        def test_func():
            raise ValueError("test error")

        # First failure
        with pytest.raises(ValueError):
            test_func()

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            test_func()

        # Third call - circuit should be open
        with pytest.raises(TimeoutError):
            test_func()


class TestTimeout:
    """Test the timeout decorator."""

    @pytest.mark.asyncio
    async def test_timeout_async_success(self):
        """Test timeout with async function that completes in time."""

        @timeout(1.0)
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_timeout_async_timeout(self):
        """Test timeout with async function that times out."""

        @timeout(0.01)
        async def test_func():
            await asyncio.sleep(0.1)
            return "success"

        with pytest.raises(TimeoutError):
            await test_func()

    def test_timeout_sync_warning(self):
        """Test timeout with sync function (should log warning)."""

        @timeout(1.0)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"


class TestCacheResult:
    """Test the cache_result decorator."""

    @pytest.mark.asyncio
    async def test_cache_result_async(self):
        """Test cache_result with async function."""
        call_count = 0

        @cache_result(ttl_seconds=1.0)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # First call
        result1 = await test_func()
        assert result1 == "result_1"
        assert call_count == 1

        # Second call - should use cache
        result2 = await test_func()
        assert result2 == "result_1"
        assert call_count == 1

    def test_cache_result_sync(self):
        """Test cache_result with sync function."""
        call_count = 0

        @cache_result(ttl_seconds=1.0)
        def test_func():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # Clear any existing cache
        from src.utils.decorators import _cache, _cache_timestamps

        _cache.clear()
        _cache_timestamps.clear()

        # First call
        result1 = test_func()
        assert result1 == "result_1"
        assert call_count == 1

        # Second call - should use cache
        result2 = test_func()
        assert result2 == "result_1"
        assert call_count == 1  # Function should not be called again


class TestRedisCache:
    """Test the redis_cache decorator."""

    @pytest.mark.asyncio
    async def test_redis_cache_async(self):
        """Test redis_cache with async function."""

        @redis_cache(ttl_seconds=1.0)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    def test_redis_cache_sync(self):
        """Test redis_cache with sync function."""

        @redis_cache(ttl_seconds=1.0)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"


class TestLogCalls:
    """Test the log_calls decorator."""

    @pytest.mark.asyncio
    async def test_log_calls_async_success(self):
        """Test log_calls with async function that succeeds."""

        @log_calls
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_log_calls_async_error(self):
        """Test log_calls with async function that raises error."""

        @log_calls
        async def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await test_func()

    def test_log_calls_sync_success(self):
        """Test log_calls with sync function that succeeds."""

        @log_calls
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_log_calls_sync_error(self):
        """Test log_calls with sync function that raises error."""

        @log_calls
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()


class TestLogPerformance:
    """Test the log_performance decorator."""

    @pytest.mark.asyncio
    async def test_log_performance_async_success(self):
        """Test log_performance with async function that succeeds."""

        @log_performance
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_log_performance_async_error(self):
        """Test log_performance with async function that raises error."""

        @log_performance
        async def test_func():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await test_func()

    def test_log_performance_sync_success(self):
        """Test log_performance with sync function that succeeds."""

        @log_performance
        def test_func():
            time.sleep(0.01)
            return "success"

        result = test_func()
        assert result == "success"

    def test_log_performance_sync_error(self):
        """Test log_performance with sync function that raises error."""

        @log_performance
        def test_func():
            time.sleep(0.01)
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()


class TestLogErrors:
    """Test the log_errors decorator."""

    @pytest.mark.asyncio
    async def test_log_errors_async_success(self):
        """Test log_errors with async function that succeeds."""

        @log_errors
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_log_errors_async_error(self):
        """Test log_errors with async function that raises error."""

        @log_errors
        async def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await test_func()

    def test_log_errors_sync_success(self):
        """Test log_errors with sync function that succeeds."""

        @log_errors
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_log_errors_sync_error(self):
        """Test log_errors with sync function that raises error."""

        @log_errors
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()


class TestValidateInput:
    """Test the validate_input decorator."""

    @pytest.mark.asyncio
    async def test_validate_input_async_success(self):
        """Test validate_input with async function that passes validation."""

        def validation_func(*args, **kwargs):
            if len(args) > 0 and args[0] == "valid":
                return True
            raise ValidationError("Invalid input")

        @validate_input(validation_func)
        async def test_func(value):
            return f"processed_{value}"

        result = await test_func("valid")
        assert result == "processed_valid"

    @pytest.mark.asyncio
    async def test_validate_input_async_validation_fails(self):
        """Test validate_input with async function that fails validation."""

        def validation_func(*args, **kwargs):
            raise ValidationError("Invalid input")

        @validate_input(validation_func)
        async def test_func(value):
            return f"processed_{value}"

        with pytest.raises(ValidationError, match="Invalid input"):
            await test_func("invalid")

    def test_validate_input_sync_success(self):
        """Test validate_input with sync function that passes validation."""

        def validation_func(*args, **kwargs):
            if len(args) > 0 and args[0] == "valid":
                return True
            raise ValidationError("Invalid input")

        @validate_input(validation_func)
        def test_func(value):
            return f"processed_{value}"

        result = test_func("valid")
        assert result == "processed_valid"

    def test_validate_input_sync_validation_fails(self):
        """Test validate_input with sync function that fails validation."""

        def validation_func(*args, **kwargs):
            raise ValidationError("Invalid input")

        @validate_input(validation_func)
        def test_func(value):
            return f"processed_{value}"

        with pytest.raises(ValidationError, match="Invalid input"):
            test_func("invalid")


class TestValidateOutput:
    """Test the validate_output decorator."""

    @pytest.mark.asyncio
    async def test_validate_output_async_success(self):
        """Test validate_output with async function that passes validation."""

        def validation_func(result):
            if result == "valid_result":
                return True
            raise ValidationError("Invalid output")

        @validate_output(validation_func)
        async def test_func():
            return "valid_result"

        result = await test_func()
        assert result == "valid_result"

    @pytest.mark.asyncio
    async def test_validate_output_async_validation_fails(self):
        """Test validate_output with async function that fails validation."""

        def validation_func(result):
            raise ValidationError("Invalid output")

        @validate_output(validation_func)
        async def test_func():
            return "invalid_result"

        with pytest.raises(ValidationError, match="Invalid output"):
            await test_func()

    def test_validate_output_sync_success(self):
        """Test validate_output with sync function that passes validation."""

        def validation_func(result):
            if result == "valid_result":
                return True
            raise ValidationError("Invalid output")

        @validate_output(validation_func)
        def test_func():
            return "valid_result"

        result = test_func()
        assert result == "valid_result"

    def test_validate_output_sync_validation_fails(self):
        """Test validate_output with sync function that fails validation."""

        def validation_func(result):
            raise ValidationError("Invalid output")

        @validate_output(validation_func)
        def test_func():
            return "invalid_result"

        with pytest.raises(ValidationError, match="Invalid output"):
            test_func()


class TestTypeCheck:
    """Test the type_check decorator."""

    @pytest.mark.asyncio
    async def test_type_check_async_success(self):
        """Test type_check with async function that has correct types."""

        @type_check
        async def test_func(value: str) -> str:
            return f"processed_{value}"

        result = await test_func("test")
        assert result == "processed_test"

    @pytest.mark.asyncio
    async def test_type_check_async_type_mismatch(self):
        """Test type_check with async function that has type mismatch."""

        @type_check
        async def test_func(value: str) -> str:
            return f"processed_{value}"

        with pytest.raises(ValidationError):
            await test_func(123)  # Should be string

    def test_type_check_sync_success(self):
        """Test type_check with sync function that has correct types."""

        @type_check
        def test_func(value: str) -> str:
            return f"processed_{value}"

        result = test_func("test")
        assert result == "processed_test"

    def test_type_check_sync_type_mismatch(self):
        """Test type_check with sync function that has type mismatch."""

        @type_check
        def test_func(value: str) -> str:
            return f"processed_{value}"

        with pytest.raises(ValidationError):
            test_func(123)  # Should be string


class TestRateLimit:
    """Test the rate_limit decorator."""

    @pytest.mark.asyncio
    async def test_rate_limit_async_success(self):
        """Test rate_limit with async function that doesn't exceed limits."""

        @rate_limit(max_calls=5, time_window=1.0)
        async def test_func():
            return "success"

        # Make multiple calls within limit
        results = []
        for _ in range(3):
            result = await test_func()
            results.append(result)

        assert all(r == "success" for r in results)

    @pytest.mark.asyncio
    async def test_rate_limit_async_exceeds_limit(self):
        """Test rate_limit with async function that exceeds limits."""

        @rate_limit(max_calls=2, time_window=1.0)
        async def test_func():
            return "success"

        # First two calls should succeed
        result1 = await test_func()
        result2 = await test_func()
        assert result1 == "success"
        assert result2 == "success"

        # Third call should be delayed but eventually succeed
        result3 = await test_func()
        assert result3 == "success"

    def test_rate_limit_sync_success(self):
        """Test rate_limit with sync function that doesn't exceed limits."""

        @rate_limit(max_calls=5, time_window=1.0)
        def test_func():
            return "success"

        # Make multiple calls within limit
        results = []
        for _ in range(3):
            result = test_func()
            results.append(result)

        assert all(r == "success" for r in results)

    def test_rate_limit_sync_exceeds_limit(self):
        """Test rate_limit with sync function that exceeds limits."""

        @rate_limit(max_calls=2, time_window=1.0)
        def test_func():
            return "success"

        # First two calls should succeed
        result1 = test_func()
        result2 = test_func()
        assert result1 == "success"
        assert result2 == "success"

        # Third call should be delayed but eventually succeed
        result3 = test_func()
        assert result3 == "success"


class TestApiThrottle:
    """Test the api_throttle decorator."""

    @pytest.mark.asyncio
    async def test_api_throttle_async(self):
        """Test api_throttle with async function."""

        @api_throttle(max_calls=5, time_window=1.0)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    def test_api_throttle_sync(self):
        """Test api_throttle with sync function."""

        @api_throttle(max_calls=5, time_window=1.0)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"
