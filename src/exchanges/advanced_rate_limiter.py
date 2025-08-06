"""
Advanced rate limiting framework for multi-exchange coordination.

This module implements sophisticated rate limiting with exchange-specific
implementations and global coordination across all supported exchanges.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003+ (exchange interfaces).
"""

import asyncio
import time
from typing import Dict, Optional, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

# MANDATORY: Import from P-001
from src.core.types import (
    MarketData, OrderRequest, OrderResponse, Position,
    ExchangeType, RequestType
)
from src.core.exceptions import (
    ExchangeRateLimitError, ExchangeConnectionError, ExchangeError, ValidationError
)
from src.core.config import Config
from src.core.logging import get_logger

# MANDATORY: Import from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import RecoveryScenario

# MANDATORY: Import from P-007A (placeholder until P-007A is implemented)
# from src.utils.decorators import time_execution

logger = get_logger(__name__)


class AdvancedRateLimiter:
    """
    Advanced rate limiter coordinating across all exchanges.
    
    This class coordinates rate limiting across all exchanges and provides
    exchange-specific rate limiting implementations.
    """
    
    def __init__(self, config: Config):
        """
        Initialize advanced rate limiter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.exchange_limiters: Dict[str, Any] = {}
        self.global_limits: Dict[str, Any] = {}
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        
        # Initialize exchange-specific limiters
        self._initialize_exchange_limiters()
        
        # TODO: Remove in production
        logger.debug(f"AdvancedRateLimiter initialized with {len(self.exchange_limiters)} exchanges")
    
    def _initialize_exchange_limiters(self) -> None:
        """Initialize exchange-specific rate limiters."""
        try:
            self.exchange_limiters[ExchangeType.BINANCE.value] = BinanceRateLimiter(self.config)
            self.exchange_limiters[ExchangeType.OKX.value] = OKXRateLimiter(self.config)
            self.exchange_limiters[ExchangeType.COINBASE.value] = CoinbaseRateLimiter(self.config)
            
            logger.info(f"Exchange rate limiters initialized: {list(self.exchange_limiters.keys())}")
        except Exception as e:
            logger.error(f"Failed to initialize exchange rate limiters: {str(e)}")
            raise ExchangeError(f"Rate limiter initialization failed: {str(e)}")
    
    async def check_rate_limit(self, exchange: str, endpoint: str, weight: int = 1) -> bool:
        """
        Check if rate limit allows the request.
        
        Args:
            exchange: Exchange name
            endpoint: API endpoint
            weight: Request weight
            
        Returns:
            bool: True if request is allowed
            
        Raises:
            ExchangeRateLimitError: If rate limit is exceeded
            ValidationError: If parameters are invalid
        """
        try:
            # Validate input parameters
            if not exchange or not endpoint:
                raise ValidationError("Exchange and endpoint are required")
            
            if weight <= 0:
                raise ValidationError("Weight must be positive")
            
            # Get exchange-specific limiter
            if exchange not in self.exchange_limiters:
                raise ExchangeRateLimitError(f"Unknown exchange: {exchange}")
            
            limiter = self.exchange_limiters[exchange]
            
            # Check exchange-specific limits
            if not await limiter.check_limit(endpoint, weight):
                logger.warning(f"Rate limit exceeded: exchange {exchange}, endpoint {endpoint}, weight {weight}")
                return False
            
            # Check global limits
            if not await self._check_global_limits(exchange, endpoint, weight):
                logger.warning(f"Global rate limit exceeded: exchange {exchange}, endpoint {endpoint}, weight {weight}")
                return False
            
            # Record request
            self._record_request(exchange, endpoint, weight)
            
            logger.debug(f"Rate limit check passed: exchange {exchange}, endpoint {endpoint}, weight {weight}")
            return True
            
        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Rate limit check failed: exchange {exchange}, endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"Rate limit check failed: {str(e)}")
    
    async def wait_if_needed(self, exchange: str, endpoint: str) -> float:
        """
        Wait if rate limit is exceeded.
        
        Args:
            exchange: Exchange name
            endpoint: API endpoint
            
        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not exchange or not endpoint:
                raise ValidationError("Exchange and endpoint are required")
            
            if exchange not in self.exchange_limiters:
                raise ExchangeRateLimitError(f"Unknown exchange: {exchange}")
            
            limiter = self.exchange_limiters[exchange]
            
            # Wait for exchange-specific limits
            wait_time = await limiter.wait_for_reset(endpoint)
            
            logger.debug(f"Rate limit wait completed: exchange {exchange}, endpoint {endpoint}, wait_time {wait_time}")
            return wait_time
            
        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            logger.error(f"Rate limit wait failed: exchange {exchange}, endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"Rate limit wait failed: {str(e)}")
    
    async def _check_global_limits(self, exchange: str, endpoint: str, weight: int) -> bool:
        """Check global rate limits."""
        # TODO: Implement global limit checking
        # For now, always return True
        return True
    
    def _record_request(self, exchange: str, endpoint: str, weight: int) -> None:
        """Record request for tracking."""
        now = datetime.now()
        self.request_history[f"{exchange}:{endpoint}"].append(now)
        
        # Clean old history (keep last 1000 requests)
        if len(self.request_history[f"{exchange}:{endpoint}"]) > 1000:
            self.request_history[f"{exchange}:{endpoint}"] = \
                self.request_history[f"{exchange}:{endpoint}"][-1000:]


class BinanceRateLimiter:
    """
    Binance-specific rate limiter implementing weight-based limiting.
    
    Binance uses a weight-based system with 1200 requests/minute limit.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Binance rate limiter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.weight_limit = 1200  # requests per minute
        self.order_limit_10s = 50  # orders per 10 seconds
        self.order_limit_24h = 160000  # orders per 24 hours
        
        # Track usage
        self.weight_usage: Dict[str, List[datetime]] = defaultdict(list)
        self.order_usage: List[datetime] = []
        
        # TODO: Remove in production
        logger.debug(f"BinanceRateLimiter initialized with weight_limit {self.weight_limit}")
    
    
    async def check_limit(self, endpoint: str, weight: int = 1) -> bool:
        """
        Check if request is within Binance rate limits.
        
        Args:
            endpoint: API endpoint
            weight: Request weight
            
        Returns:
            bool: True if request is allowed
            
        Raises:
            ValidationError: If parameters are invalid
            ExchangeRateLimitError: If rate limit is exceeded
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")
            
            if weight <= 0:
                raise ValidationError("Weight must be positive")
            
            if weight > self.weight_limit:
                raise ValidationError(f"Weight {weight} exceeds limit {self.weight_limit}")
            
            # Check weight-based limits
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.weight_usage[endpoint] = [
                t for t in self.weight_usage[endpoint] 
                if t > minute_ago
            ]
            
            # Calculate current weight usage
            current_weight = sum(1 for _ in self.weight_usage[endpoint]) * weight
            
            if current_weight + weight > self.weight_limit:
                logger.warning("Binance weight limit exceeded", 
                             endpoint=endpoint, weight=weight, 
                             current_weight=current_weight, limit=self.weight_limit)
                return False
            
            # Check order limits if applicable
            if "order" in endpoint.lower():
                if not await self._check_order_limits():
                    return False
            
            logger.debug("Binance rate limit check passed", 
                        endpoint=endpoint, weight=weight)
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Binance rate limit check failed: endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"Binance rate limit check failed: {str(e)}")
    
    
    async def wait_for_reset(self, endpoint: str) -> float:
        """
        Wait for rate limit reset.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")
            
            # Calculate wait time based on current usage
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.weight_usage[endpoint] = [
                t for t in self.weight_usage[endpoint] 
                if t > minute_ago
            ]
            
            # Calculate time until reset
            if self.weight_usage[endpoint]:
                oldest_request = min(self.weight_usage[endpoint])
                reset_time = oldest_request + timedelta(minutes=1)
                wait_time = max(0, (reset_time - now).total_seconds())
            else:
                wait_time = 0
            
            if wait_time > 0:
                logger.info("Waiting for Binance rate limit reset", 
                           endpoint=endpoint, wait_time=wait_time)
                await asyncio.sleep(wait_time)
            
            return wait_time
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Binance rate limit wait failed: endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"Binance rate limit wait failed: {str(e)}")
    
    async def _check_order_limits(self) -> bool:
        """Check order-specific rate limits."""
        now = datetime.now()
        ten_seconds_ago = now - timedelta(seconds=10)
        day_ago = now - timedelta(days=1)
        
        # Clean old entries
        self.order_usage = [t for t in self.order_usage if t > day_ago]
        
        # Check 10-second limit
        recent_orders = [t for t in self.order_usage if t > ten_seconds_ago]
        if len(recent_orders) >= self.order_limit_10s:
            logger.warning(f"Binance order limit exceeded (10s): {len(recent_orders)} orders, limit {self.order_limit_10s}")
            return False
        
        # Check 24-hour limit
        if len(self.order_usage) >= self.order_limit_24h:
            logger.warning(f"Binance order limit exceeded (24h): {len(self.order_usage)} orders, limit {self.order_limit_24h}")
            return False
        
        return True


class OKXRateLimiter:
    """
    OKX-specific rate limiter implementing endpoint-based limiting.
    
    OKX uses endpoint-based limits: REST (60/2s), Orders (600/2s), Historical (20/2s).
    """
    
    def __init__(self, config: Config):
        """
        Initialize OKX rate limiter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Endpoint-specific limits
        self.limits = {
            "rest": {"requests": 60, "window": 2},  # 60 requests per 2 seconds
            "orders": {"requests": 600, "window": 2},  # 600 orders per 2 seconds
            "historical": {"requests": 20, "window": 2},  # 20 requests per 2 seconds
        }
        
        # Track usage per endpoint type
        self.usage: Dict[str, List[datetime]] = defaultdict(list)
        
        # TODO: Remove in production
        logger.debug(f"OKXRateLimiter initialized with limits: {self.limits}")
    
    
    async def check_limit(self, endpoint: str, weight: int = 1) -> bool:
        """
        Check if request is within OKX rate limits.
        
        Args:
            endpoint: API endpoint
            weight: Request weight (not used for OKX)
            
        Returns:
            bool: True if request is allowed
            
        Raises:
            ValidationError: If parameters are invalid
            ExchangeRateLimitError: If rate limit is exceeded
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")
            
            # Determine endpoint type
            endpoint_type = self._get_endpoint_type(endpoint)
            
            # Check limits
            now = datetime.now()
            window_seconds = self.limits[endpoint_type]["window"]
            max_requests = self.limits[endpoint_type]["requests"]
            
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old entries
            self.usage[endpoint_type] = [
                t for t in self.usage[endpoint_type] 
                if t > window_start
            ]
            
            # Check if limit exceeded
            if len(self.usage[endpoint_type]) >= max_requests:
                logger.warning(f"OKX rate limit exceeded: {endpoint_type} endpoint {endpoint}, {len(self.usage[endpoint_type])} requests, limit {max_requests}")
                return False
            
            logger.debug(f"OKX rate limit check passed: {endpoint_type} endpoint {endpoint}")
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"OKX rate limit check failed: endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"OKX rate limit check failed: {str(e)}")
    
    
    async def wait_for_reset(self, endpoint: str) -> float:
        """
        Wait for rate limit reset.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")
            
            endpoint_type = self._get_endpoint_type(endpoint)
            window_seconds = self.limits[endpoint_type]["window"]
            
            now = datetime.now()
            window_start = now - timedelta(seconds=window_seconds)
            
            # Clean old entries
            self.usage[endpoint_type] = [
                t for t in self.usage[endpoint_type] 
                if t > window_start
            ]
            
            # Calculate wait time
            if self.usage[endpoint_type]:
                oldest_request = min(self.usage[endpoint_type])
                reset_time = oldest_request + timedelta(seconds=window_seconds)
                wait_time = max(0, (reset_time - now).total_seconds())
            else:
                wait_time = 0
            
            if wait_time > 0:
                logger.info("Waiting for OKX rate limit reset", 
                           endpoint_type=endpoint_type, endpoint=endpoint, 
                           wait_time=wait_time)
                await asyncio.sleep(wait_time)
            
            return wait_time
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"OKX rate limit wait failed: endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"OKX rate limit wait failed: {str(e)}")
    
    def _get_endpoint_type(self, endpoint: str) -> str:
        """Determine endpoint type for rate limiting."""
        endpoint_lower = endpoint.lower()
        
        if "order" in endpoint_lower:
            return "orders"
        elif "history" in endpoint_lower or "historical" in endpoint_lower:
            return "historical"
        else:
            return "rest"


class CoinbaseRateLimiter:
    """
    Coinbase-specific rate limiter implementing point-based limiting.
    
    Coinbase uses a point-based system with 8000 points/minute limit.
    """
    
    def __init__(self, config: Config):
        """
        Initialize Coinbase rate limiter.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.points_limit = 8000  # points per minute
        self.private_limit = 10  # requests per second for private endpoints
        self.public_limit = 15  # requests per second for public endpoints
        
        # Track usage
        self.points_usage: List[datetime] = []
        self.private_usage: List[datetime] = []
        self.public_usage: List[datetime] = []
        
        # TODO: Remove in production
        logger.debug("CoinbaseRateLimiter initialized", 
                    points_limit=self.points_limit)
    
    
    async def check_limit(self, endpoint: str, is_private: bool = False) -> bool:
        """
        Check if request is within Coinbase rate limits.
        
        Args:
            endpoint: API endpoint
            is_private: Whether this is a private endpoint
            
        Returns:
            bool: True if request is allowed
            
        Raises:
            ValidationError: If parameters are invalid
            ExchangeRateLimitError: If rate limit is exceeded
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")
            
            # Calculate points for this request
            points = self._calculate_points(endpoint)
            
            # Check points limit
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.points_usage = [t for t in self.points_usage if t > minute_ago]
            
            # Calculate current points usage
            current_points = sum(self._calculate_points("") for _ in self.points_usage)
            
            if current_points + points > self.points_limit:
                logger.warning("Coinbase points limit exceeded", 
                             endpoint=endpoint, points=points,
                             current_points=current_points, limit=self.points_limit)
                return False
            
            # Check per-second limits
            if is_private:
                if not await self._check_private_limit():
                    return False
            else:
                if not await self._check_public_limit():
                    return False
            
            logger.debug("Coinbase rate limit check passed", 
                        endpoint=endpoint, points=points, is_private=is_private)
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Coinbase rate limit check failed: endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"Coinbase rate limit check failed: {str(e)}")
    
    
    async def wait_for_reset(self, endpoint: str) -> float:
        """
        Wait for rate limit reset.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not endpoint:
                raise ValidationError("Endpoint is required")
            
            # Calculate wait time based on points usage
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.points_usage = [t for t in self.points_usage if t > minute_ago]
            
            # Calculate time until reset
            if self.points_usage:
                oldest_request = min(self.points_usage)
                reset_time = oldest_request + timedelta(minutes=1)
                wait_time = max(0, (reset_time - now).total_seconds())
            else:
                wait_time = 0
            
            if wait_time > 0:
                logger.info(f"Waiting for Coinbase rate limit reset: endpoint {endpoint}, wait time {wait_time}")
                await asyncio.sleep(wait_time)
            
            return wait_time
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Coinbase rate limit wait failed: endpoint {endpoint}, error {str(e)}")
            raise ExchangeRateLimitError(f"Coinbase rate limit wait failed: {str(e)}")
    
    def _calculate_points(self, endpoint: str) -> int:
        """
        Calculate points for an endpoint.
        
        Args:
            endpoint: API endpoint
            
        Returns:
            int: Points for this endpoint
        """
        endpoint_lower = endpoint.lower()
        
        # Point calculation based on endpoint type
        if "order" in endpoint_lower:
            return 10  # Order placement is expensive
        elif "balance" in endpoint_lower or "account" in endpoint_lower:
            return 5   # Account queries are moderate
        elif "market" in endpoint_lower or "ticker" in endpoint_lower:
            return 1   # Market data is cheap
        else:
            return 2   # Default cost
    
    async def _check_private_limit(self) -> bool:
        """Check private endpoint rate limits."""
        now = datetime.now()
        second_ago = now - timedelta(seconds=1)
        
        # Clean old entries
        self.private_usage = [t for t in self.private_usage if t > second_ago]
        
        if len(self.private_usage) >= self.private_limit:
            logger.warning(f"Coinbase private rate limit exceeded: {len(self.private_usage)} requests, limit {self.private_limit}")
            return False
        
        return True
    
    async def _check_public_limit(self) -> bool:
        """Check public endpoint rate limits."""
        now = datetime.now()
        second_ago = now - timedelta(seconds=1)
        
        # Clean old entries
        self.public_usage = [t for t in self.public_usage if t > second_ago]
        
        if len(self.public_usage) >= self.public_limit:
            logger.warning(f"Coinbase public rate limit exceeded: {len(self.public_usage)} requests, limit {self.public_limit}")
            return False
        
        return True 