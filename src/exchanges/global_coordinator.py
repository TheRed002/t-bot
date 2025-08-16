"""
Global rate coordinator for cross-exchange rate limit management.

This module implements global coordination of rate limits across all
supported exchanges to prevent system-wide rate limit violations.

CRITICAL: This integrates with P-001 (core types, exceptions, config),
P-002A (error handling), and P-003+ (exchange interfaces).
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from src.core.config import Config
from src.core.exceptions import (
    ExchangeRateLimitError,
    ValidationError,
)
from src.core.logging import get_logger

# MANDATORY: Import from P-001
from src.core.types import RequestType

# MANDATORY: Import from P-002A
# MANDATORY: Import from P-007A (placeholder until P-007A is implemented)
from src.utils.decorators import log_calls, time_execution

logger = get_logger(__name__)


class GlobalRateCoordinator:
    """
    Global rate coordinator for cross-exchange rate limit management.

    This class coordinates rate limits across all exchanges to prevent
    system-wide rate limit violations and ensure optimal resource usage.
    """

    def __init__(self, config: Config):
        """
        Initialize global rate coordinator.

        Args:
            config: Application configuration
        """
        self.config = config

        # Global limits
        self.global_limits = {
            "total_requests_per_minute": 5000,
            "orders_per_minute": 1000,
            "concurrent_connections": 50,
            "websocket_messages_per_second": 1000,
        }

        # Track global usage
        self.request_history: dict[str, list[datetime]] = defaultdict(list)
        self.order_history: list[datetime] = []
        self.connection_history: list[datetime] = []
        self.websocket_message_history: list[datetime] = []

        # Exchange-specific tracking
        self.exchange_usage: dict[str, dict[str, list[datetime]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # TODO: Remove in production
        logger.debug("GlobalRateCoordinator initialized", global_limits=self.global_limits)

    @time_execution
    @log_calls
    async def check_global_limits(self, request_type: str, count: int = 1) -> bool:
        """
        Check if request is within global rate limits.

        Args:
            request_type: Type of request (from RequestType enum)
            count: Number of requests

        Returns:
            bool: True if request is allowed

        Raises:
            ExchangeRateLimitError: If global rate limit is exceeded
            ValidationError: If parameters are invalid
        """
        try:
            # Validate input parameters
            if not request_type:
                raise ValidationError("Request type is required")

            if count <= 0:
                raise ValidationError("Count must be positive")

            # Validate request type
            try:
                RequestType(request_type)
            except ValueError:
                raise ValidationError(f"Invalid request type: {request_type}")

            # Check specific limits based on request type
            if request_type == RequestType.ORDER_PLACEMENT.value:
                if not await self._check_order_limits(count):
                    return False
            elif request_type == RequestType.WEBSOCKET_CONNECTION.value:
                if not await self._check_connection_limits(count):
                    return False
            else:
                if not await self._check_general_request_limits(count):
                    return False

            # Record request
            self._record_request(request_type, count)

            logger.debug("Global rate limit check passed", request_type=request_type, count=count)
            return True

        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            logger.error(
                "Global rate limit check failed",
                request_type=request_type,
                count=count,
                error=str(e),
            )
            raise ExchangeRateLimitError(f"Global rate limit check failed: {e!s}")

    @time_execution
    @log_calls
    async def coordinate_request(self, exchange: str, endpoint: str, request_type: str) -> bool:
        """
        Coordinate request across global and exchange-specific limits.

        Args:
            exchange: Exchange name
            endpoint: API endpoint
            request_type: Type of request

        Returns:
            bool: True if request is allowed

        Raises:
            ExchangeRateLimitError: If rate limit is exceeded
            ValidationError: If parameters are invalid
        """
        try:
            # Validate input parameters
            if not exchange or not endpoint or not request_type:
                raise ValidationError("Exchange, endpoint, and request_type are required")

            # Check global limits first
            if not await self.check_global_limits(request_type, 1):
                logger.warning(
                    "Global rate limit exceeded",
                    exchange=exchange,
                    endpoint=endpoint,
                    request_type=request_type,
                )
                return False

            # Check exchange-specific limits
            if not await self._check_exchange_specific_limits(exchange, endpoint, request_type):
                logger.warning(
                    "Exchange-specific rate limit exceeded",
                    exchange=exchange,
                    endpoint=endpoint,
                    request_type=request_type,
                )
                return False

            # Record exchange usage
            self._record_exchange_usage(exchange, endpoint, request_type)

            logger.debug(
                "Request coordination successful",
                exchange=exchange,
                endpoint=endpoint,
                request_type=request_type,
            )
            return True

        except (ValidationError, ExchangeRateLimitError):
            raise
        except Exception as e:
            logger.error(
                "Request coordination failed",
                exchange=exchange,
                endpoint=endpoint,
                request_type=request_type,
                error=str(e),
            )
            raise ExchangeRateLimitError(f"Request coordination failed: {e!s}")

    async def _check_order_limits(self, count: int) -> bool:
        """Check order-specific global limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.order_history = [t for t in self.order_history if t > minute_ago]

        # Check if adding these orders would exceed limit
        if len(self.order_history) + count > self.global_limits["orders_per_minute"]:
            logger.warning(
                "Global order limit exceeded",
                current_orders=len(self.order_history),
                new_orders=count,
                limit=self.global_limits["orders_per_minute"],
            )
            return False

        return True

    async def _check_connection_limits(self, count: int) -> bool:
        """Check connection-specific global limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.connection_history = [t for t in self.connection_history if t > minute_ago]

        # Check if adding these connections would exceed limit
        if len(self.connection_history) + count > self.global_limits["concurrent_connections"]:
            logger.warning(
                "Global connection limit exceeded",
                current_connections=len(self.connection_history),
                new_connections=count,
                limit=self.global_limits["concurrent_connections"],
            )
            return False

        return True

    async def _check_general_request_limits(self, count: int) -> bool:
        """Check general request limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.request_history["general"] = [
            t for t in self.request_history["general"] if t > minute_ago
        ]

        # Check if adding these requests would exceed limit
        if (
            len(self.request_history["general"]) + count
            > self.global_limits["total_requests_per_minute"]
        ):
            logger.warning(
                "Global request limit exceeded",
                current_requests=len(self.request_history["general"]),
                new_requests=count,
                limit=self.global_limits["total_requests_per_minute"],
            )
            return False

        return True

    async def _check_exchange_specific_limits(
        self, exchange: str, endpoint: str, request_type: str
    ) -> bool:
        """Check exchange-specific limits."""
        # This would integrate with the advanced rate limiter
        # For now, return True as a placeholder
        return True

    def _record_request(self, request_type: str, count: int) -> None:
        """Record request for tracking."""
        now = datetime.now()

        # Record based on request type
        if request_type == RequestType.ORDER_PLACEMENT.value:
            for _ in range(count):
                self.order_history.append(now)
        elif request_type == RequestType.WEBSOCKET_CONNECTION.value:
            for _ in range(count):
                self.connection_history.append(now)
        else:
            for _ in range(count):
                self.request_history["general"].append(now)

        # Clean old history (keep last 10000 entries)
        if len(self.request_history["general"]) > 10000:
            self.request_history["general"] = self.request_history["general"][-10000:]

        if len(self.order_history) > 10000:
            self.order_history = self.order_history[-10000:]

        if len(self.connection_history) > 10000:
            self.connection_history = self.connection_history[-10000:]

    def _record_exchange_usage(self, exchange: str, endpoint: str, request_type: str) -> None:
        """Record exchange-specific usage."""
        now = datetime.now()
        self.exchange_usage[exchange][endpoint].append(now)

        # Clean old history (keep last 1000 entries per endpoint)
        if len(self.exchange_usage[exchange][endpoint]) > 1000:
            self.exchange_usage[exchange][endpoint] = self.exchange_usage[exchange][endpoint][
                -1000:
            ]

    def get_global_usage_stats(self) -> dict[str, Any]:
        """
        Get global usage statistics.

        Returns:
            Dict containing usage statistics
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Calculate current usage
        recent_orders = [t for t in self.order_history if t > minute_ago]
        recent_connections = [t for t in self.connection_history if t > minute_ago]
        recent_requests = [t for t in self.request_history["general"] if t > minute_ago]

        return {
            "orders_per_minute": len(recent_orders),
            "concurrent_connections": len(recent_connections),
            "total_requests_per_minute": len(recent_requests),
            "limits": self.global_limits,
            "timestamp": now.isoformat(),
        }

    def get_exchange_usage_stats(self, exchange: str) -> dict[str, Any]:
        """
        Get exchange-specific usage statistics.

        Args:
            exchange: Exchange name

        Returns:
            Dict containing exchange usage statistics
        """
        if exchange not in self.exchange_usage:
            return {}

        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        stats = {}
        for endpoint, timestamps in self.exchange_usage[exchange].items():
            recent_requests = [t for t in timestamps if t > minute_ago]
            stats[endpoint] = {
                "requests_per_minute": len(recent_requests),
                "total_requests": len(timestamps),
            }

        return stats

    async def wait_for_global_capacity(self, request_type: str, count: int = 1) -> float:
        """
        Wait for global capacity to become available.

        Args:
            request_type: Type of request
            count: Number of requests needed

        Returns:
            float: Wait time in seconds
        """
        try:
            # Validate parameters
            if not request_type:
                raise ValidationError("Request type is required")

            if count <= 0:
                raise ValidationError("Count must be positive")

            # Calculate wait time based on request type
            if request_type == RequestType.ORDER_PLACEMENT.value:
                wait_time = await self._calculate_order_wait_time(count)
            elif request_type == RequestType.WEBSOCKET_CONNECTION.value:
                wait_time = await self._calculate_connection_wait_time(count)
            else:
                wait_time = await self._calculate_general_wait_time(count)

            if wait_time > 0:
                logger.info(
                    "Waiting for global capacity",
                    request_type=request_type,
                    count=count,
                    wait_time=wait_time,
                )
                await asyncio.sleep(wait_time)

            return wait_time

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Global capacity wait failed", request_type=request_type, count=count, error=str(e)
            )
            raise ExchangeRateLimitError(f"Global capacity wait failed: {e!s}")

    async def _calculate_order_wait_time(self, count: int) -> float:
        """Calculate wait time for order capacity."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.order_history = [t for t in self.order_history if t > minute_ago]

        # Calculate time until capacity is available
        if len(self.order_history) + count > self.global_limits["orders_per_minute"]:
            if self.order_history:
                oldest_order = min(self.order_history)
                reset_time = oldest_order + timedelta(minutes=1)
                return max(0, (reset_time - now).total_seconds())

        return 0.0

    async def _calculate_connection_wait_time(self, count: int) -> float:
        """Calculate wait time for connection capacity."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.connection_history = [t for t in self.connection_history if t > minute_ago]

        # Calculate time until capacity is available
        if len(self.connection_history) + count > self.global_limits["concurrent_connections"]:
            if self.connection_history:
                oldest_connection = min(self.connection_history)
                reset_time = oldest_connection + timedelta(minutes=1)
                return max(0, (reset_time - now).total_seconds())

        return 0.0

    async def _calculate_general_wait_time(self, count: int) -> float:
        """Calculate wait time for general request capacity."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self.request_history["general"] = [
            t for t in self.request_history["general"] if t > minute_ago
        ]

        # Calculate time until capacity is available
        if (
            len(self.request_history["general"]) + count
            > self.global_limits["total_requests_per_minute"]
        ):
            if self.request_history["general"]:
                oldest_request = min(self.request_history["general"])
                reset_time = oldest_request + timedelta(minutes=1)
                return max(0, (reset_time - now).total_seconds())

        return 0.0

    def reset_global_limits(self) -> None:
        """Reset all global limits (for testing/debugging)."""
        self.request_history.clear()
        self.order_history.clear()
        self.connection_history.clear()
        self.websocket_message_history.clear()
        self.exchange_usage.clear()

        logger.info("Global rate limits reset")

    def update_global_limits(self, new_limits: dict[str, int]) -> None:
        """
        Update global limits.

        Args:
            new_limits: New limit values
        """
        for key, value in new_limits.items():
            if key in self.global_limits:
                self.global_limits[key] = value
                logger.info(f"Updated global limit: {key} = {value}")
            else:
                logger.warning(f"Unknown global limit key: {key}")

    def get_health_status(self) -> dict[str, Any]:
        """
        Get health status of the global coordinator.

        Returns:
            Dict containing health information
        """
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Calculate current usage percentages
        recent_orders = [t for t in self.order_history if t > minute_ago]
        recent_connections = [t for t in self.connection_history if t > minute_ago]
        recent_requests = [t for t in self.request_history["general"] if t > minute_ago]

        order_usage_pct = (len(recent_orders) / self.global_limits["orders_per_minute"]) * 100
        connection_usage_pct = (
            len(recent_connections) / self.global_limits["concurrent_connections"]
        ) * 100
        request_usage_pct = (
            len(recent_requests) / self.global_limits["total_requests_per_minute"]
        ) * 100

        return {
            "status": (
                "healthy"
                if max(order_usage_pct, connection_usage_pct, request_usage_pct) < 90
                else "warning"
            ),
            "order_usage_percent": round(order_usage_pct, 2),
            "connection_usage_percent": round(connection_usage_pct, 2),
            "request_usage_percent": round(request_usage_pct, 2),
            "timestamp": now.isoformat(),
        }
