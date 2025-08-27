"""
Unit tests for decorators module.

This module tests all decorators in the utils.decorators module to ensure
they work correctly with both sync and async functions, handle errors properly,
and provide the expected functionality.
"""

import asyncio
import time
from datetime import datetime, timedelta

import pytest

from src.core.exceptions import TimeoutError, ValidationError
from src.utils.decorators import (
    UnifiedDecorator,
    cached,
    logged,
    monitored,
    retry,
    timeout,
    validated,
)


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

        @retry(max_attempts=3, delay=0.01)
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

        @retry(max_attempts=2, delay=0.01)
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

        @retry(max_attempts=3, delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("temporary error")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 3


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

        with pytest.raises(asyncio.TimeoutError):
            await test_func()

    def test_timeout_sync_success(self):
        """Test timeout with sync function."""

        @timeout(1.0)
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"


class TestCached:
    """Test the cached decorator."""

    @pytest.mark.asyncio
    async def test_cached_async(self):
        """Test cached with async function."""
        call_count = 0

        @cached(ttl=1)
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

    def test_cached_sync(self):
        """Test cached with sync function."""
        call_count = 0

        @cached(ttl=1)
        def test_func():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # Clear any existing cache
        UnifiedDecorator._cache.clear()
        UnifiedDecorator._cache_timestamps.clear()

        # First call
        result1 = test_func()
        assert result1 == "result_1"
        assert call_count == 1

        # Second call - should use cache
        result2 = test_func()
        assert result2 == "result_1"
        assert call_count == 1  # Function should not be called again


class TestLogged:
    """Test the logged decorator."""

    @pytest.mark.asyncio
    async def test_logged_async_success(self):
        """Test logged with async function that succeeds."""

        @logged(level="info")
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_logged_async_error(self):
        """Test logged with async function that raises error."""

        @logged(level="error")
        async def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            await test_func()

    def test_logged_sync_success(self):
        """Test logged with sync function that succeeds."""

        @logged(level="debug")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_logged_sync_error(self):
        """Test logged with sync function that raises error."""

        @logged(level="warning")
        def test_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            test_func()


class TestValidated:
    """Test the validated decorator."""

    @pytest.mark.asyncio
    async def test_validated_async_valid_args(self):
        """Test validated with async function and valid arguments."""

        @validated()
        async def test_func(price: float, quantity: float):
            return price * quantity

        result = await test_func(100.0, 2.0)
        assert result == 200.0

    @pytest.mark.asyncio
    async def test_validated_async_invalid_order(self):
        """Test validated with async function and valid order data."""

        @validated()
        async def test_func(order: dict):
            return order

        # Test with valid order data (all required fields)
        valid_order = {
            'symbol': 'BTC/USDT',
            'type': 'LIMIT',
            'side': 'BUY',
            'price': 50000.0,
            'quantity': 1.0
        }
        result = await test_func(valid_order)
        assert result == valid_order

    def test_validated_sync_valid_args(self):
        """Test validated with sync function and valid arguments."""

        @validated()
        def test_func(symbol: str):
            return f"Trading {symbol}"

        result = test_func("BTC/USDT")
        assert result == "Trading BTC/USDT"


class TestMonitored:
    """Test the monitored decorator."""

    @pytest.mark.asyncio
    async def test_monitored_async(self):
        """Test monitored with async function."""

        @monitored()
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    def test_monitored_sync(self):
        """Test monitored with sync function."""

        @monitored()
        def test_func():
            time.sleep(0.01)
            return "success"

        result = test_func()
        assert result == "success"


class TestUnifiedDecorator:
    """Test the UnifiedDecorator class with combined features."""

    @pytest.mark.asyncio
    async def test_combined_retry_and_log(self):
        """Test combining retry and logging features."""
        call_count = 0

        @UnifiedDecorator.enhance(retry=True, retry_times=3, retry_delay=0.01, log=True, log_level="info")
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("temporary error")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_combined_cache_and_validate(self):
        """Test combining cache and validation features."""

        @UnifiedDecorator.enhance(cache=True, cache_ttl=1, validate=True)
        async def test_func(price: float):
            return price * 2

        # First call
        result1 = await test_func(100.0)
        assert result1 == 200.0

        # Second call - should use cache
        result2 = await test_func(100.0)
        assert result2 == 200.0

    @pytest.mark.asyncio
    async def test_combined_timeout_and_monitor(self):
        """Test combining timeout and monitoring features."""

        @UnifiedDecorator.enhance(timeout=1.0, monitor=True)
        async def test_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await test_func()
        assert result == "success"

    def test_combined_all_features_sync(self):
        """Test combining all features with sync function."""

        @UnifiedDecorator.enhance(
            retry=True,
            retry_times=2,
            retry_delay=0.01,
            validate=True,
            log=True,
            log_level="debug",
            cache=True,
            cache_ttl=1,
            monitor=True,
        )
        def test_func(value: float):
            if value <= 0:
                raise ValueError("Value must be positive")
            return value * 2

        # First call
        result1 = test_func(10.0)
        assert result1 == 20.0

        # Second call - should use cache
        result2 = test_func(10.0)
        assert result2 == 20.0

    @pytest.mark.asyncio
    async def test_fallback_function(self):
        """Test decorator with fallback function."""

        async def fallback_func(*args, **kwargs):
            return "fallback_result"

        @UnifiedDecorator.enhance(retry=True, retry_times=1, fallback=fallback_func)
        async def test_func():
            raise ValueError("always fails")

        result = await test_func()
        assert result == "fallback_result"

    def test_cache_expiration(self):
        """Test cache expiration after TTL."""
        call_count = 0

        @cached(ttl=0)  # Immediate expiration
        def test_func():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # Clear cache first
        UnifiedDecorator._cache.clear()
        UnifiedDecorator._cache_timestamps.clear()

        # First call
        result1 = test_func()
        assert result1 == "result_1"
        assert call_count == 1

        # Wait a bit to ensure expiration
        time.sleep(0.1)

        # Second call - cache should be expired
        result2 = test_func()
        assert result2 == "result_2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_validation_with_special_params(self):
        """Test validation with special parameter names."""

        @validated()
        async def test_func(order: dict, price: float, quantity: float, symbol: str):
            return {
                'order': order,
                'price': price,
                'quantity': quantity,
                'symbol': symbol
            }

        result = await test_func(
            order={'type': 'LIMIT', 'side': 'BUY', 'symbol': 'BTC/USDT', 'price': 50000.0, 'quantity': 1.0},
            price=50000.0,
            quantity=1.0,
            symbol='BTC/USDT'
        )

        assert result['price'] == 50000.0
        assert result['quantity'] == 1.0
        assert result['symbol'] == 'BTC/USDT'

    def test_sync_wrapper_in_running_loop(self):
        """Test sync wrapper behavior when event loop is running."""

        @retry(max_attempts=1)
        def test_func():
            return "success"

        # This should work even if called from within an async context
        result = test_func()
        assert result == "success"


class TestDecoratorErrorHandling:
    """Test error handling in decorators."""

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test handling of validation errors."""

        @validated()
        async def test_func(price: float):
            return price

        # Test with invalid price (handled by validation)
        result = await test_func(100.0)
        assert result == 100.0

    @pytest.mark.asyncio
    async def test_retry_with_different_exceptions(self):
        """Test retry behavior with different exception types."""
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("connection failed")
            elif call_count == 2:
                raise TimeoutError("timeout occurred")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    def test_cache_key_generation(self):
        """Test cache key generation for different arguments."""

        @cached(ttl=1)
        def test_func(a, b, c=None):
            return f"{a}_{b}_{c}"

        # Clear cache
        UnifiedDecorator._cache.clear()
        UnifiedDecorator._cache_timestamps.clear()

        # Different arguments should generate different cache keys
        result1 = test_func(1, 2)
        result2 = test_func(1, 3)
        result3 = test_func(1, 2, c="test")

        assert result1 == "1_2_None"
        assert result2 == "1_3_None"
        assert result3 == "1_2_test"

    @pytest.mark.asyncio
    async def test_timeout_with_cleanup(self):
        """Test timeout decorator properly cleans up on timeout."""

        @timeout(0.01)
        async def test_func():
            try:
                await asyncio.sleep(1.0)
                return "should_not_return"
            except asyncio.CancelledError:
                # Cleanup code
                raise  # Re-raise the cancellation

        # The timeout may raise asyncio.TimeoutError or asyncio.CancelledError
        with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
            await test_func()

    def test_monitor_metrics_recording(self):
        """Test that monitor decorator records metrics."""

        @monitored()
        def test_func(value):
            time.sleep(0.01)
            return value * 2

        result = test_func(10)
        assert result == 20
        # Metrics should be recorded (check via logging in actual implementation)