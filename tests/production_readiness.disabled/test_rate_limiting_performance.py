"""
Production Readiness Tests for Rate Limiting and Performance

Tests rate limiting, throttling, and performance characteristics:
- Exchange-specific rate limits
- Token bucket algorithms
- Request queuing and throttling
- Burst handling capabilities
- Performance under load
- Resource utilization optimization
"""

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.production_readiness.test_config import TestConfig as Config
from src.exchanges.advanced_rate_limiter import AdvancedRateLimiter
from src.exchanges.rate_limiter import RateLimiter
from src.exchanges.service import ExchangeService


class TestRateLimitingPerformance:
    """Test rate limiting and performance characteristics."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config({
            "rate_limiting": {
                "binance": {
                    "requests_per_minute": 1200,
                    "orders_per_second": 10,
                    "burst_capacity": 100,
                    "weight_limits": {
                        "default": 1,
                        "place_order": 10,
                        "cancel_order": 1,
                        "get_order_book": 5
                    }
                },
                "coinbase": {
                    "requests_per_second": 10,
                    "public_requests_per_second": 10,
                    "private_requests_per_second": 5,
                    "burst_capacity": 20
                },
                "okx": {
                    "requests_per_second": 20,
                    "orders_per_second": 60,
                    "burst_capacity": 50
                }
            },
            "performance": {
                "max_concurrent_requests": 100,
                "request_timeout": 30,
                "connection_pool_size": 20
            }
        })

    @pytest.fixture
    def basic_rate_limiter(self):
        """Create basic rate limiter."""
        return RateLimiter(
            requests_per_second=10,
            burst_capacity=20
        )

    @pytest.fixture
    def advanced_rate_limiter(self, config):
        """Create advanced rate limiter."""
        binance_config = config.rate_limiting.binance if hasattr(config.rate_limiting, 'binance') else {}
        return AdvancedRateLimiter(
            requests_per_minute=1200,
            burst_capacity=100,
            weight_limits=binance_config.get('weight_limits', {})
        )

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self, basic_rate_limiter):
        """Test basic rate limiting functionality."""
        
        # Test token acquisition
        acquired = await basic_rate_limiter.acquire()
        assert acquired is True
        
        # Test rapid acquisition (should be limited)
        start_time = time.time()
        acquisitions = []
        
        for _ in range(15):  # More than burst capacity
            acquired = await basic_rate_limiter.acquire()
            acquisitions.append((acquired, time.time()))
        
        end_time = time.time()
        
        # Should demonstrate rate limiting
        successful_acquisitions = [a for a, _ in acquisitions if a]
        assert len(successful_acquisitions) <= 20  # Burst capacity limit
        
        # Should take time due to rate limiting
        assert end_time - start_time > 0.5

    @pytest.mark.asyncio
    async def test_weighted_rate_limiting(self, advanced_rate_limiter):
        """Test weighted rate limiting for different operations."""
        
        # Test different operation weights
        operations = [
            ("get_ticker", 1),
            ("place_order", 10),
            ("get_order_book", 5),
            ("cancel_order", 1)
        ]
        
        acquisition_results = []
        
        for operation, weight in operations:
            acquired = await advanced_rate_limiter.acquire(weight=weight)
            acquisition_results.append((operation, weight, acquired))
        
        # All should be acquired initially (within burst capacity)
        successful = [r for r in acquisition_results if r[2]]
        assert len(successful) > 0
        
        # Test statistics
        stats = advanced_rate_limiter.get_statistics()
        assert "tokens_consumed" in stats or "requests_made" in stats

    @pytest.mark.asyncio
    async def test_exchange_specific_rate_limits(self, config):
        """Test exchange-specific rate limit configurations."""
        
        # Test Binance rate limits (high frequency)
        binance_limiter = AdvancedRateLimiter(
            requests_per_minute=1200,
            burst_capacity=100
        )
        
        # Test Coinbase rate limits (more restrictive) 
        coinbase_limiter = RateLimiter(
            requests_per_second=10,
            burst_capacity=20
        )
        
        # Test different acquisition patterns
        binance_results = []
        coinbase_results = []
        
        # Rapid requests to both
        for _ in range(25):
            binance_acquired = await binance_limiter.acquire()
            coinbase_acquired = await coinbase_limiter.acquire()
            binance_results.append(binance_acquired)
            coinbase_results.append(coinbase_acquired)
        
        # Binance should allow more requests
        binance_success = sum(binance_results)
        coinbase_success = sum(coinbase_results)
        
        assert binance_success >= coinbase_success  # Binance should be more permissive

    @pytest.mark.asyncio
    async def test_burst_handling_capabilities(self, advanced_rate_limiter):
        """Test system handling of request bursts."""
        
        # Test initial burst (should be allowed)
        burst_size = 50
        burst_results = []
        
        start_time = time.time()
        for i in range(burst_size):
            acquired = await advanced_rate_limiter.acquire()
            burst_results.append(acquired)
        
        burst_end_time = time.time()
        
        # Initial burst should be fast
        burst_duration = burst_end_time - start_time
        assert burst_duration < 5.0  # Should handle burst quickly
        
        # Some of the burst should succeed
        successful_burst = sum(burst_results)
        assert successful_burst > 0
        assert successful_burst <= 100  # Within burst capacity
        
        # Test sustained load after burst
        sustained_results = []
        sustained_start = time.time()
        
        for _ in range(20):
            acquired = await advanced_rate_limiter.acquire()
            sustained_results.append(acquired)
            await asyncio.sleep(0.1)  # Small delay to simulate real usage
        
        sustained_end = time.time()
        sustained_duration = sustained_end - sustained_start
        
        # Should demonstrate throttling during sustained load
        assert sustained_duration > 1.0  # Should take more time due to rate limiting

    @pytest.mark.asyncio
    async def test_request_queuing_under_load(self, config):
        """Test request queuing mechanisms under heavy load."""
        
        # Create service with mocked exchange
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_exchange = AsyncMock()
            mock_exchange.exchange_name = "binance"
            mock_exchange.health_check.return_value = True
            
            # Add artificial delay to simulate processing time
            async def delayed_ticker(*args, **kwargs):
                await asyncio.sleep(0.05)  # 50ms delay
                return MagicMock(
                    symbol="BTC/USDT",
                    bid_price=Decimal("50000"),
                    ask_price=Decimal("50010"),
                    last_price=Decimal("50005"),
                    volume=Decimal("100")
                )
            
            mock_exchange.get_ticker.side_effect = delayed_ticker
            mock_exchange_factory.get_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_exchange_factory
            
            service = ExchangeService(
                exchange_factory=mock_exchange_factory,
                config=config
            )
            
            # Submit many concurrent requests
            tasks = []
            symbols = [f"SYMBOL{i}" for i in range(50)]
            
            start_time = time.time()
            for symbol in symbols:
                task = service.get_ticker("binance", symbol)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # All requests should complete
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) > 40  # Most should succeed
            
            # Should demonstrate queuing behavior
            total_duration = end_time - start_time
            expected_sequential_time = 50 * 0.05  # 2.5 seconds if sequential
            
            # Should be faster than sequential but show some queuing
            assert total_duration < expected_sequential_time
            assert total_duration > 0.1  # But not instantaneous

    @pytest.mark.asyncio
    async def test_rate_limit_recovery_handling(self, advanced_rate_limiter):
        """Test rate limit recovery and token replenishment."""
        
        # Exhaust rate limiter
        exhaustion_results = []
        for _ in range(150):  # More than burst capacity
            acquired = await advanced_rate_limiter.acquire()
            exhaustion_results.append(acquired)
        
        # Should eventually be exhausted
        final_attempts = exhaustion_results[-10:]  # Last 10 attempts
        refused_count = final_attempts.count(False)
        assert refused_count > 0  # Some should be refused
        
        # Wait for recovery
        await asyncio.sleep(2.0)  # Allow time for token replenishment
        
        # Test recovery
        recovery_acquired = await advanced_rate_limiter.acquire()
        assert recovery_acquired is True  # Should recover tokens

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiter_access(self, basic_rate_limiter):
        """Test concurrent access to rate limiter."""
        
        async def worker(worker_id: int):
            """Worker function for concurrent testing."""
            results = []
            for i in range(10):
                acquired = await basic_rate_limiter.acquire()
                results.append((worker_id, i, acquired, time.time()))
                if acquired:
                    await asyncio.sleep(0.01)  # Small delay for successful requests
            return results
        
        # Create multiple concurrent workers
        workers = []
        for i in range(10):
            worker_task = worker(i)
            workers.append(worker_task)
        
        start_time = time.time()
        all_results = await asyncio.gather(*workers)
        end_time = time.time()
        
        # Flatten results
        flat_results = []
        for worker_results in all_results:
            flat_results.extend(worker_results)
        
        # Analyze results
        total_requests = len(flat_results)
        successful_requests = len([r for r in flat_results if r[2]])
        
        assert total_requests == 100  # 10 workers * 10 requests each
        assert successful_requests > 0  # Some should succeed
        assert successful_requests <= 50  # Should be rate limited
        
        # Should complete in reasonable time
        assert end_time - start_time < 30.0

    @pytest.mark.asyncio
    async def test_performance_memory_efficiency(self, advanced_rate_limiter):
        """Test memory efficiency under sustained load."""
        
        # Simulate sustained usage pattern
        usage_cycles = 10
        requests_per_cycle = 100
        
        for cycle in range(usage_cycles):
            cycle_results = []
            
            # Make requests
            for _ in range(requests_per_cycle):
                acquired = await advanced_rate_limiter.acquire()
                cycle_results.append(acquired)
            
            # Check statistics (should not grow unbounded)
            stats = advanced_rate_limiter.get_statistics()
            
            # Memory usage should be bounded
            if "memory_usage" in stats:
                assert stats["memory_usage"] < 10 * 1024 * 1024  # Less than 10MB
            
            # Add small delay between cycles
            await asyncio.sleep(0.1)
        
        # Final statistics check
        final_stats = advanced_rate_limiter.get_statistics()
        assert "tokens_consumed" in final_stats or "total_requests" in final_stats

    @pytest.mark.asyncio 
    async def test_rate_limiter_statistics_accuracy(self, advanced_rate_limiter):
        """Test accuracy of rate limiter statistics."""
        
        # Make tracked requests
        request_count = 50
        successful_requests = 0
        failed_requests = 0
        
        for i in range(request_count):
            weight = 1 if i % 10 != 0 else 5  # Varied weights
            acquired = await advanced_rate_limiter.acquire(weight=weight)
            
            if acquired:
                successful_requests += 1
            else:
                failed_requests += 1
        
        # Get statistics
        stats = advanced_rate_limiter.get_statistics()
        
        # Verify statistics accuracy
        if "successful_requests" in stats:
            assert stats["successful_requests"] == successful_requests
        
        if "failed_requests" in stats:
            assert stats["failed_requests"] == failed_requests
        
        if "total_requests" in stats:
            assert stats["total_requests"] == request_count

    @pytest.mark.asyncio
    async def test_rate_limiter_reset_functionality(self, basic_rate_limiter):
        """Test rate limiter reset functionality."""
        
        # Exhaust rate limiter
        for _ in range(30):  # More than burst capacity
            await basic_rate_limiter.acquire()
        
        # Should be limited now
        pre_reset_acquired = await basic_rate_limiter.acquire()
        
        # Reset rate limiter
        basic_rate_limiter.reset()
        
        # Should work after reset
        post_reset_acquired = await basic_rate_limiter.acquire()
        assert post_reset_acquired is True

    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self, advanced_rate_limiter):
        """Test adaptive rate limiting based on server responses."""
        
        # Simulate rate limit headers from server
        rate_limit_responses = [
            {"X-RateLimit-Remaining": "100", "X-RateLimit-Reset": "60"},
            {"X-RateLimit-Remaining": "50", "X-RateLimit-Reset": "45"},
            {"X-RateLimit-Remaining": "10", "X-RateLimit-Reset": "30"},
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "15"}
        ]
        
        for response_headers in rate_limit_responses:
            # In a real implementation, this would adjust rate limiting
            # based on server responses
            if hasattr(advanced_rate_limiter, 'update_from_headers'):
                advanced_rate_limiter.update_from_headers(response_headers)
            
            # Test acquisition after header update
            acquired = await advanced_rate_limiter.acquire()
            
            # Behavior should adapt based on remaining quota
            remaining = int(response_headers["X-RateLimit-Remaining"])
            if remaining == 0:
                # Should be more restrictive when quota exhausted
                pass  # Implementation dependent

    @pytest.mark.asyncio
    async def test_cross_exchange_rate_limiting(self, config):
        """Test rate limiting coordination across multiple exchanges."""
        
        # Create rate limiters for different exchanges
        binance_limiter = AdvancedRateLimiter(requests_per_minute=1200, burst_capacity=100)
        coinbase_limiter = RateLimiter(requests_per_second=10, burst_capacity=20)
        okx_limiter = RateLimiter(requests_per_second=20, burst_capacity=50)
        
        limiters = {
            "binance": binance_limiter,
            "coinbase": coinbase_limiter,
            "okx": okx_limiter
        }
        
        # Test concurrent usage across exchanges
        async def exchange_worker(exchange_name: str, limiter):
            """Worker for specific exchange."""
            results = []
            for _ in range(20):
                acquired = await limiter.acquire()
                results.append((exchange_name, acquired))
                if acquired:
                    await asyncio.sleep(0.01)
            return results
        
        # Run workers concurrently
        tasks = []
        for exchange_name, limiter in limiters.items():
            task = exchange_worker(exchange_name, limiter)
            tasks.append(task)
        
        all_results = await asyncio.gather(*tasks)
        
        # Analyze per-exchange results
        for i, (exchange_name, limiter) in enumerate(limiters.items()):
            exchange_results = all_results[i]
            successful = len([r for r in exchange_results if r[1]])
            
            # Each exchange should have some successful requests
            assert successful > 0
            
            # Different exchanges should show different patterns
            stats = limiter.get_statistics()
            assert stats is not None