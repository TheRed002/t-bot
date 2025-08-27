"""
Unit tests for the exchange rate limiter.

This module tests the RateLimiter and TokenBucket classes to ensure
proper rate limiting functionality and error handling.
"""

import asyncio
import time
from unittest.mock import Mock

import pytest

from src.core.config import Config
from src.core.exceptions import ExchangeRateLimitError

# Import the components to test
from src.exchanges.rate_limiter import RateLimitDecorator, RateLimiter, TokenBucket


class TestTokenBucket:
    """Test cases for the TokenBucket class."""

    def test_token_bucket_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=5, refill_time=1.0)

        assert bucket.capacity == 10
        assert bucket.refill_rate == 5
        assert bucket.refill_time == 1.0
        assert bucket.tokens == 10  # Should start with full capacity
        assert bucket.last_refill is not None

    def test_token_consumption(self):
        """Test token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=5, refill_time=1.0)

        # Consume tokens
        assert bucket.consume(5)
        assert bucket.tokens == 5

        # Consume more tokens
        assert bucket.consume(3)
        assert bucket.tokens == 2

        # Try to consume more than available
        assert not bucket.consume(5)
        assert bucket.tokens == 2  # Should remain unchanged

    def test_token_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=5, refill_time=1.0)

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0

        # Simulate time passing (1 second)
        bucket.last_refill = time.time() - 1.0

        # Consume should refill tokens
        assert bucket.consume(3)
        assert bucket.tokens == pytest.approx(2, abs=0.1)  # 5 refilled - 3 consumed = 2

    def test_token_refill_capacity_limit(self):
        """Test that tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=5, refill_time=1.0)

        # Simulate time passing (3 seconds)
        bucket.last_refill = time.time() - 3.0

        # Should not exceed capacity
        bucket.consume(1)  # This triggers refill
        # After 3 seconds, should have refilled 3 * 5 = 15 tokens, but capped
        # at 10
        assert bucket.tokens == pytest.approx(9, abs=0.1)  # 10 - 1 consumed = 9

    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        bucket = TokenBucket(capacity=10, refill_rate=5, refill_time=1.0)

        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0

        # Calculate wait time for 3 tokens
        wait_time = bucket.get_wait_time(3)
        assert wait_time > 0
        # 3 tokens / 5 per second = 0.6 seconds
        assert wait_time == pytest.approx(0.6, abs=0.1)

    def test_wait_time_no_wait(self):
        """Test wait time when tokens are available."""
        bucket = TokenBucket(capacity=10, refill_rate=5, refill_time=1.0)

        # Tokens are available
        wait_time = bucket.get_wait_time(3)
        assert wait_time == 0.0


class TestRateLimiter:
    """Test cases for the RateLimiter class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def rate_limiter(self, config):
        """Create a rate limiter instance."""
        return RateLimiter(config, "test_exchange")

    def test_rate_limiter_initialization(self, config):
        """Test rate limiter initialization."""
        rate_limiter = RateLimiter(config, "test_exchange")

        assert rate_limiter.config == config
        assert rate_limiter.exchange_name == "test_exchange"
        assert rate_limiter.error_handler is not None
        assert "requests_per_minute" in rate_limiter.buckets
        assert "orders_per_second" in rate_limiter.buckets
        assert "websocket_connections" in rate_limiter.buckets

    @pytest.mark.asyncio
    async def test_acquire_tokens_success(self, rate_limiter):
        """Test successful token acquisition."""
        # Should succeed immediately
        result = await rate_limiter.acquire("requests_per_minute", tokens=1, timeout=1.0)
        assert result

    @pytest.mark.asyncio
    async def test_acquire_tokens_unknown_bucket(self, rate_limiter):
        """Test acquiring tokens from unknown bucket."""
        # Should return True for unknown buckets
        result = await rate_limiter.acquire("unknown_bucket", tokens=1, timeout=1.0)
        assert result

    @pytest.mark.asyncio
    async def test_acquire_tokens_timeout(self, rate_limiter):
        """Test token acquisition timeout."""
        # Consume all tokens
        bucket = rate_limiter.buckets["requests_per_minute"]
        bucket.tokens = 0
        bucket.last_refill = time.time() - 0.1  # Recent refill

        # Try to acquire tokens with short timeout
        with pytest.raises(
            ExchangeRateLimitError, match="Rate limit exceeded for requests_per_minute"
        ):
            await rate_limiter.acquire("requests_per_minute", tokens=100, timeout=0.1)

    @pytest.mark.asyncio
    async def test_acquire_tokens_exceed_wait_time(self, rate_limiter):
        """Test token acquisition when wait time exceeds timeout."""
        # Consume all tokens
        bucket = rate_limiter.buckets["requests_per_minute"]
        bucket.tokens = 0
        bucket.last_refill = time.time() - 0.1  # Recent refill

        # Try to acquire tokens with wait time > timeout
        with pytest.raises(ExchangeRateLimitError, match="Wait time.*exceeds timeout"):
            await rate_limiter.acquire("requests_per_minute", tokens=100, timeout=0.1)

    def test_record_request(self, rate_limiter):
        """Test request recording."""
        rate_limiter._record_request("requests_per_minute", 5)

        assert len(rate_limiter.request_history["requests_per_minute"]) == 1
        assert rate_limiter.request_history["requests_per_minute"][0]["tokens"] == 5

        # Record more requests
        rate_limiter._record_request("requests_per_minute", 3)
        assert len(rate_limiter.request_history["requests_per_minute"]) == 2

    def test_record_request_history_cleanup(self, rate_limiter):
        """Test request history cleanup."""
        # Add more than 1000 requests
        for i in range(1001):
            rate_limiter._record_request("requests_per_minute", 1)

        # Should keep only last 1000
        assert len(rate_limiter.request_history["requests_per_minute"]) == 1000

    def test_get_bucket_status(self, rate_limiter):
        """Test getting bucket status."""
        status = rate_limiter.get_bucket_status("requests_per_minute")

        assert "tokens_available" in status
        assert "capacity" in status
        assert "refill_rate" in status
        assert "refill_time" in status
        assert "last_request" in status
        assert "request_count" in status

    def test_get_bucket_status_unknown(self, rate_limiter):
        """Test getting status of unknown bucket."""
        status = rate_limiter.get_bucket_status("unknown_bucket")
        assert "error" in status

    def test_get_all_bucket_status(self, rate_limiter):
        """Test getting status of all buckets."""
        all_status = rate_limiter.get_all_bucket_status()

        assert "requests_per_minute" in all_status
        assert "orders_per_second" in all_status
        assert "websocket_connections" in all_status

    @pytest.mark.asyncio
    async def test_wait_for_capacity(self, rate_limiter):
        """Test waiting for capacity."""
        # Consume all tokens
        bucket = rate_limiter.buckets["requests_per_minute"]
        bucket.tokens = 0
        bucket.last_refill = time.time() - 0.1  # Recent refill

        # Wait for capacity
        wait_time = await rate_limiter.wait_for_capacity("requests_per_minute", tokens=5)
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_wait_for_capacity_unknown_bucket(self, rate_limiter):
        """Test waiting for capacity of unknown bucket."""
        wait_time = await rate_limiter.wait_for_capacity("unknown_bucket", tokens=5)
        assert wait_time == 0.0

    def test_is_rate_limited(self, rate_limiter):
        """Test rate limit checking."""
        # Should not be rate limited initially
        assert not rate_limiter.is_rate_limited("requests_per_minute")

        # Consume all tokens
        bucket = rate_limiter.buckets["requests_per_minute"]
        bucket.tokens = 0

        # Should be rate limited
        assert rate_limiter.is_rate_limited("requests_per_minute")

    def test_is_rate_limited_unknown_bucket(self, rate_limiter):
        """Test rate limit checking for unknown bucket."""
        assert not rate_limiter.is_rate_limited("unknown_bucket")

    @pytest.mark.asyncio
    async def test_context_manager(self, rate_limiter):
        """Test async context manager functionality."""
        async with rate_limiter as rl:
            assert rl == rate_limiter

        # Should not raise any exceptions


class TestRateLimitDecorator:
    """Test cases for the RateLimitDecorator class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def rate_limiter(self, config):
        """Create a rate limiter instance."""
        return RateLimiter(config, "test_exchange")

    @pytest.fixture
    def mock_exchange(self, rate_limiter):
        """Create a mock exchange with rate limiter."""
        mock = Mock()
        mock.rate_limiter = rate_limiter
        return mock

    def test_decorator_initialization(self):
        """Test decorator initialization."""
        decorator = RateLimitDecorator("requests_per_minute", tokens=5, timeout=10.0)

        assert decorator.bucket_name == "requests_per_minute"
        assert decorator.tokens == 5
        assert decorator.timeout == 10.0

    @pytest.mark.asyncio
    async def test_decorator_success(self, mock_exchange):
        """Test successful decorator usage."""
        decorator = RateLimitDecorator("requests_per_minute", tokens=1, timeout=1.0)

        @decorator
        async def test_function(self, arg):
            return f"result: {arg}"

        result = await test_function(mock_exchange, "test")
        assert result == "result: test"

    @pytest.mark.asyncio
    async def test_decorator_no_rate_limiter(self):
        """Test decorator with no rate limiter."""
        decorator = RateLimitDecorator("requests_per_minute", tokens=1, timeout=1.0)

        @decorator
        async def test_function(self, arg):
            return f"result: {arg}"

        # Should work without rate limiter
        mock = Mock()
        result = await test_function(mock, "test")
        assert result == "result: test"

    @pytest.mark.asyncio
    async def test_decorator_rate_limit_exceeded(self, mock_exchange):
        """Test decorator when rate limit is exceeded."""
        # Consume all tokens
        bucket = mock_exchange.rate_limiter.buckets["requests_per_minute"]
        bucket.tokens = 0
        bucket.last_refill = time.time() - 0.1  # Recent refill

        decorator = RateLimitDecorator("requests_per_minute", tokens=100, timeout=0.1)

        @decorator
        async def test_function(self, arg):
            return f"result: {arg}"

        # Should raise rate limit error
        with pytest.raises(ExchangeRateLimitError):
            await test_function(mock_exchange, "test")


class TestRateLimiterIntegration:
    """Integration tests for rate limiter with different scenarios."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config()

    @pytest.fixture
    def rate_limiter(self, config):
        """Create a rate limiter instance."""
        return RateLimiter(config, "test_exchange")

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, rate_limiter):
        """Test concurrent requests with rate limiting."""

        async def make_request(request_id):
            await rate_limiter.acquire("requests_per_minute", tokens=1, timeout=1.0)
            return f"request_{request_id}"

        # Make multiple concurrent requests
        tasks = [make_request(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result == f"request_{i}"

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self, rate_limiter):
        """Test rate limit recovery after waiting."""
        # Use orders_per_second bucket which has a 1-second refill time
        bucket = rate_limiter.buckets["orders_per_second"]
        bucket.tokens = 0
        bucket.last_refill = time.time() - 0.1  # Recent refill

        # Should fail initially
        with pytest.raises(ExchangeRateLimitError):
            await rate_limiter.acquire("orders_per_second", tokens=5, timeout=0.1)

        # Wait for recovery (1 second for orders_per_second bucket)
        await asyncio.sleep(1.1)

        # Should succeed after recovery
        result = await rate_limiter.acquire("orders_per_second", tokens=1, timeout=1.0)
        assert result

    @pytest.mark.asyncio
    async def test_multiple_buckets(self, rate_limiter):
        """Test multiple rate limit buckets."""
        # Test requests per minute bucket
        result1 = await rate_limiter.acquire("requests_per_minute", tokens=1, timeout=1.0)
        assert result1

        # Test orders per second bucket
        result2 = await rate_limiter.acquire("orders_per_second", tokens=1, timeout=1.0)
        assert result2

        # Test websocket connections bucket
        result3 = await rate_limiter.acquire("websocket_connections", tokens=1, timeout=1.0)
        assert result3

    def test_request_history_tracking(self, rate_limiter):
        """Test request history tracking."""
        # Make some requests
        rate_limiter._record_request("requests_per_minute", 1)
        rate_limiter._record_request("requests_per_minute", 2)
        rate_limiter._record_request("orders_per_second", 1)

        # Check history
        assert len(rate_limiter.request_history["requests_per_minute"]) == 2
        assert len(rate_limiter.request_history["orders_per_second"]) == 1

        # Check last request times
        assert rate_limiter.last_request_time["requests_per_minute"] > 0
        assert rate_limiter.last_request_time["orders_per_second"] > 0
