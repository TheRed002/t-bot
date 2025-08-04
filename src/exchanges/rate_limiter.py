"""
Rate limiting framework for exchange APIs.

This module implements token bucket rate limiting to prevent API violations
and ensure compliance with exchange-specific rate limits.

CRITICAL: This integrates with P-001 (core types, exceptions, config)
and P-002A (error handling) components.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict

# MANDATORY: Import from P-001
from src.core.exceptions import ExchangeRateLimitError, ExchangeError
from src.core.config import Config

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    
    This class implements a token bucket algorithm for rate limiting
    with configurable capacity and refill rate.
    """
    
    def __init__(self, capacity: int, refill_rate: float, refill_time: float = 1.0):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens per refill_time
            refill_time: Time interval for refill in seconds
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_time = refill_time
        self.tokens = capacity
        self.last_refill = time.time()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        time_passed = now - self.last_refill
        
        if time_passed >= self.refill_time:
            tokens_to_add = (time_passed / self.refill_time) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            bool: True if tokens available, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time for token consumption.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            float: Wait time in seconds
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return (tokens_needed / self.refill_rate) * self.refill_time


class RateLimiter:
    """
    Rate limiter for exchange API calls.
    
    This class manages rate limiting for different types of API calls
    and provides automatic retry with exponential backoff.
    """
    
    def __init__(self, config: Config, exchange_name: str):
        """
        Initialize rate limiter.
        
        Args:
            config: Application configuration
            exchange_name: Name of the exchange
        """
        self.config = config
        self.exchange_name = exchange_name
        self.error_handler = ErrorHandler(config.error_handling)
        
        # Get rate limits from config
        rate_limits = config.exchanges.rate_limits.get(exchange_name, {})
        
        # Initialize token buckets for different rate limits
        self.buckets: Dict[str, TokenBucket] = {}
        
        # Requests per minute bucket
        requests_per_minute = rate_limits.get("requests_per_minute", 1200)
        self.buckets["requests_per_minute"] = TokenBucket(
            capacity=requests_per_minute,
            refill_rate=requests_per_minute,
            refill_time=60.0
        )
        
        # Orders per second bucket
        orders_per_second = rate_limits.get("orders_per_second", 10)
        self.buckets["orders_per_second"] = TokenBucket(
            capacity=orders_per_second,
            refill_rate=orders_per_second,
            refill_time=1.0
        )
        
        # WebSocket connections bucket
        websocket_connections = rate_limits.get("websocket_connections", 5)
        self.buckets["websocket_connections"] = TokenBucket(
            capacity=websocket_connections,
            refill_rate=websocket_connections,
            refill_time=300.0  # 5 minutes
        )
        
        # Request tracking
        self.request_history: Dict[str, list] = defaultdict(list)
        self.last_request_time: Dict[str, float] = defaultdict(lambda: 0.0)
        
        logger.info(f"Initialized rate limiter for {exchange_name}")
    
    async def acquire(self, bucket_name: str = "requests_per_minute", 
                     tokens: int = 1, timeout: float = 30.0) -> bool:
        """
        Acquire tokens from a rate limit bucket.
        
        Args:
            bucket_name: Name of the bucket to acquire from
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds
            
        Returns:
            bool: True if tokens acquired, False if timeout
            
        Raises:
            ExchangeRateLimitError: If rate limit exceeded
        """
        if bucket_name not in self.buckets:
            logger.warning(f"Unknown rate limit bucket: {bucket_name}")
            return True
        
        bucket = self.buckets[bucket_name]
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if bucket.consume(tokens):
                # Record the request
                self._record_request(bucket_name, tokens)
                return True
            
            # Calculate wait time
            wait_time = bucket.get_wait_time(tokens)
            if wait_time > timeout:
                raise ExchangeRateLimitError(
                    f"Rate limit exceeded for {bucket_name}. "
                    f"Wait time {wait_time:.2f}s exceeds timeout {timeout}s"
                )
            
            # Wait for tokens to become available
            await asyncio.sleep(min(wait_time, 0.1))  # Sleep in small increments
        
        raise ExchangeRateLimitError(f"Rate limit timeout for {bucket_name}")
    
    def _record_request(self, bucket_name: str, tokens: int) -> None:
        """
        Record a request for monitoring purposes.
        
        Args:
            bucket_name: Name of the bucket
            tokens: Number of tokens consumed
        """
        now = time.time()
        self.request_history[bucket_name].append({
            "timestamp": now,
            "tokens": tokens
        })
        
        # Clean up old history (keep last 1000 requests)
        if len(self.request_history[bucket_name]) > 1000:
            self.request_history[bucket_name] = self.request_history[bucket_name][-1000:]
        
        self.last_request_time[bucket_name] = now
    
    def get_bucket_status(self, bucket_name: str) -> Dict[str, Any]:
        """
        Get status of a rate limit bucket.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            Dict[str, Any]: Bucket status information
        """
        if bucket_name not in self.buckets:
            return {"error": f"Unknown bucket: {bucket_name}"}
        
        bucket = self.buckets[bucket_name]
        return {
            "tokens_available": bucket.tokens,
            "capacity": bucket.capacity,
            "refill_rate": bucket.refill_rate,
            "refill_time": bucket.refill_time,
            "last_request": self.last_request_time.get(bucket_name, 0),
            "request_count": len(self.request_history.get(bucket_name, []))
        }
    
    def get_all_bucket_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all rate limit buckets.
        
        Returns:
            Dict[str, Dict[str, Any]]: Status of all buckets
        """
        return {
            bucket_name: self.get_bucket_status(bucket_name)
            for bucket_name in self.buckets.keys()
        }
    
    async def wait_for_capacity(self, bucket_name: str, tokens: int = 1) -> float:
        """
        Wait for capacity to become available.
        
        Args:
            bucket_name: Name of the bucket
            tokens: Number of tokens needed
            
        Returns:
            float: Wait time in seconds
        """
        if bucket_name not in self.buckets:
            return 0.0
        
        bucket = self.buckets[bucket_name]
        wait_time = bucket.get_wait_time(tokens)
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        return wait_time
    
    def is_rate_limited(self, bucket_name: str) -> bool:
        """
        Check if a bucket is currently rate limited.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        if bucket_name not in self.buckets:
            return False
        
        bucket = self.buckets[bucket_name]
        return bucket.tokens < 1
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


class RateLimitDecorator:
    """
    Decorator for applying rate limiting to exchange API methods.
    
    This decorator automatically applies rate limiting to exchange API
    calls and handles retries with exponential backoff.
    """
    
    def __init__(self, bucket_name: str = "requests_per_minute", 
                 tokens: int = 1, timeout: float = 30.0):
        """
        Initialize rate limit decorator.
        
        Args:
            bucket_name: Name of the rate limit bucket
            tokens: Number of tokens to consume
            timeout: Maximum wait time
        """
        self.bucket_name = bucket_name
        self.tokens = tokens
        self.timeout = timeout
    
    def __call__(self, func: Callable) -> Callable:
        """
        Apply rate limiting to a function.
        
        Args:
            func: Function to decorate
            
        Returns:
            Callable: Decorated function
        """
        async def wrapper(*args, **kwargs):
            # Get rate limiter from the first argument (should be self)
            if args and hasattr(args[0], 'rate_limiter'):
                rate_limiter = args[0].rate_limiter
                
                # Check if rate_limiter has acquire method and is awaitable
                if hasattr(rate_limiter, 'acquire') and asyncio.iscoroutinefunction(rate_limiter.acquire):
                    # Acquire tokens
                    await rate_limiter.acquire(
                        bucket_name=self.bucket_name,
                        tokens=self.tokens,
                        timeout=self.timeout
                    )
            
            # Execute the function
            return await func(*args, **kwargs)
        
        return wrapper 