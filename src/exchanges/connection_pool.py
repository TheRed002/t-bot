"""
Enhanced Connection Pooling for Exchange APIs

This module implements a sophisticated connection pooling system optimized for
high-frequency trading with multiple exchanges. Features include:

- Per-exchange connection pools with different optimization profiles
- Adaptive pool sizing based on load and latency
- Connection health monitoring and automatic recovery
- Rate limiting integration with connection management
- Circuit breaker patterns for resilience
- Connection affinity for WebSocket streams
- Comprehensive metrics and monitoring

Performance targets:
- Connection acquisition < 5ms
- Pool utilization > 80%
- Connection failure rate < 0.1%
- WebSocket reconnection < 2s
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import aiohttp
import asyncio_throttle
from aiohttp import ClientSession, ClientTimeout, TCPConnector

from src.core.base import BaseComponent
from src.core.config import Config
from src.core.exceptions import ExchangeConnectionError
from src.core.logging import get_logger
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class ConnectionType(Enum):
    """Types of connections for different use cases."""

    REST_API = "rest_api"
    WEBSOCKET = "websocket"
    MARKET_DATA = "market_data"
    TRADING = "trading"
    ANALYTICS = "analytics"


class ConnectionStatus(Enum):
    """Connection status enumeration."""

    IDLE = "idle"
    ACTIVE = "active"
    UNHEALTHY = "unhealthy"
    DISCONNECTED = "disconnected"
    RECOVERING = "recovering"


@dataclass
class ConnectionMetrics:
    """Metrics for a single connection."""

    connection_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    status: ConnectionStatus = ConnectionStatus.IDLE
    error_count: int = 0
    last_error: str | None = None


@dataclass
class PoolMetrics:
    """Metrics for connection pool."""

    pool_id: str
    exchange: str
    connection_type: ConnectionType
    active_connections: int = 0
    idle_connections: int = 0
    total_connections: int = 0
    max_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    queue_depth: int = 0
    queue_wait_time: float = 0.0
    connection_failures: int = 0
    rate_limit_hits: int = 0
    circuit_breaker_trips: int = 0


class ConnectionWrapper:
    """Wrapper for HTTP connections with health monitoring."""

    def __init__(
        self,
        connection_id: str,
        session: ClientSession,
        connection_type: ConnectionType,
        base_url: str,
    ):
        self.connection_id = connection_id
        self.session = session
        self.connection_type = connection_type
        self.base_url = base_url
        self.metrics = ConnectionMetrics(connection_id=connection_id)
        self.last_health_check = datetime.now(timezone.utc)
        self.health_check_interval = 60  # seconds
        self._lock = asyncio.Lock()

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request with metrics tracking."""
        start_time = time.time()

        async with self._lock:
            try:
                self.metrics.total_requests += 1
                self.metrics.last_used = datetime.now(timezone.utc)
                self.metrics.status = ConnectionStatus.ACTIVE

                response = await self.session.request(method, url, **kwargs)

                # Update metrics
                response_time = time.time() - start_time
                self.metrics.last_response_time = response_time
                self._update_avg_response_time(response_time)
                self.metrics.status = ConnectionStatus.IDLE

                return response

            except Exception as e:
                self.metrics.failed_requests += 1
                self.metrics.error_count += 1
                self.metrics.last_error = str(e)
                self.metrics.status = ConnectionStatus.UNHEALTHY
                raise

    def _update_avg_response_time(self, new_time: float) -> None:
        """Update average response time with exponential moving average."""
        alpha = 0.1  # Weight for new measurement
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = new_time
        else:
            self.metrics.avg_response_time = (
                alpha * new_time + (1 - alpha) * self.metrics.avg_response_time
            )

    async def health_check(self) -> bool:
        """Perform health check on connection."""
        try:
            # Skip if recently checked
            if (
                datetime.now(timezone.utc) - self.last_health_check
            ).total_seconds() < self.health_check_interval:
                return self.metrics.status != ConnectionStatus.UNHEALTHY

            # Simple ping to base URL
            start_time = time.time()
            async with self.session.get(
                f"{self.base_url}/ping", timeout=ClientTimeout(total=5)
            ) as response:
                time.time() - start_time

                if response.status == 200:
                    self.metrics.status = ConnectionStatus.IDLE
                    self.last_health_check = datetime.now(timezone.utc)
                    return True
                else:
                    self.metrics.status = ConnectionStatus.UNHEALTHY
                    return False

        except Exception as e:
            self.metrics.status = ConnectionStatus.UNHEALTHY
            self.metrics.last_error = str(e)
            logger.warning(f"Health check failed for connection {self.connection_id}: {e}")
            return False

    async def close(self) -> None:
        """Close the connection."""
        try:
            await self.session.close()
            self.metrics.status = ConnectionStatus.DISCONNECTED
        except Exception as e:
            logger.warning(f"Error closing connection {self.connection_id}: {e}")
            # Ensure connection is marked as disconnected even if close fails
            self.metrics.status = ConnectionStatus.DISCONNECTED
            # Invalidate the session to prevent reuse
            self.session = None


class EnhancedConnectionPool(BaseComponent):
    """
    Enhanced connection pool for exchange APIs with adaptive sizing and health monitoring.

    Features:
    - Per-connection-type optimization profiles
    - Adaptive pool sizing based on load patterns
    - Health monitoring and automatic recovery
    - Rate limiting integration
    - Circuit breaker protection
    - Connection affinity for consistent routing
    """

    def __init__(
        self,
        config: Config,
        exchange: str,
        connection_type: ConnectionType,
        base_url: str,
        max_connections: int = 20,
        min_connections: int = 5,
    ):
        """Initialize enhanced connection pool."""
        super().__init__()
        self.config = config
        self.exchange = exchange
        self.connection_type = connection_type
        self.base_url = base_url
        self.max_connections = max_connections
        self.min_connections = min_connections

        # Connection management
        self.connections: list[ConnectionWrapper] = []
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.active_connections: set[str] = set()

        # Pool metrics
        self.metrics = PoolMetrics(
            pool_id=f"{exchange}_{connection_type.value}",
            exchange=exchange,
            connection_type=connection_type,
            max_connections=max_connections,
        )

        # Adaptive sizing
        self.load_history: list[tuple[datetime, int]] = []  # (timestamp, active_count)
        self.last_resize = datetime.now(timezone.utc)
        self.resize_interval = 300  # 5 minutes

        # Health monitoring
        self._health_monitor_task: asyncio.Task | None = None
        self._health_check_interval = 30  # seconds

        # Rate limiting integration
        self.rate_limiter: asyncio_throttle.Throttler | None = None
        self._setup_rate_limiter()

        # Circuit breaker
        self.circuit_breaker_threshold = 5  # failures
        self.circuit_breaker_timeout = 60  # seconds
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.circuit_breaker_open = False

        # Connection affinity for sticky sessions
        self.affinity_map: dict[str, str] = {}  # user_id -> connection_id

        # Background tasks
        self._background_tasks: list[asyncio.Task] = []

    def _setup_rate_limiter(self) -> None:
        """Setup rate limiter based on exchange and connection type."""
        # Configure rate limits based on exchange and connection type
        rate_limits = {
            "binance": {
                ConnectionType.REST_API: (1200, 60),  # 1200 requests per minute
                ConnectionType.MARKET_DATA: (6000, 60),  # Higher limit for market data
                ConnectionType.TRADING: (100, 10),  # More conservative for trading
            },
            "okx": {
                ConnectionType.REST_API: (600, 60),
                ConnectionType.MARKET_DATA: (2000, 60),
                ConnectionType.TRADING: (60, 10),
            },
            "coinbase": {
                ConnectionType.REST_API: (300, 60),
                ConnectionType.MARKET_DATA: (1000, 60),
                ConnectionType.TRADING: (30, 10),
            },
        }

        if self.exchange in rate_limits and self.connection_type in rate_limits[self.exchange]:
            rate, period = rate_limits[self.exchange][self.connection_type]
            self.rate_limiter = asyncio_throttle.Throttler(rate_limit=rate, period=period)
        else:
            # Default conservative rate limit
            self.rate_limiter = asyncio_throttle.Throttler(rate_limit=100, period=60)

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        try:
            self.logger.info(
                f"Initializing connection pool for {self.exchange} {self.connection_type.value}"
            )

            # Create initial connections
            await self._create_initial_connections()

            # Start background tasks
            await self._start_background_tasks()

            self.logger.info(
                f"Connection pool initialized with {len(self.connections)} connections",
                extra={
                    "exchange": self.exchange,
                    "connection_type": self.connection_type.value,
                    "initial_connections": len(self.connections),
                },
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize connection pool: {e}")
            raise ExchangeConnectionError(f"Pool initialization failed: {e}")

    async def _create_initial_connections(self) -> None:
        """Create initial pool of connections."""
        for i in range(self.min_connections):
            connection = await self._create_connection(
                f"{self.exchange}_{self.connection_type.value}_{i}"
            )
            self.connections.append(connection)
            await self.available_connections.put(connection)

        self._update_pool_metrics()

    async def _create_connection(self, connection_id: str) -> ConnectionWrapper:
        """Create a new connection with optimized settings."""
        # Configure connector based on connection type
        connector_settings = self._get_connector_settings()
        connector = TCPConnector(**connector_settings)

        # Configure timeout based on connection type
        timeout_settings = self._get_timeout_settings()
        timeout = ClientTimeout(**timeout_settings)

        # Create session with optimizations
        session = ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_default_headers(),
            raise_for_status=False,  # Handle status codes manually
        )

        connection = ConnectionWrapper(
            connection_id=connection_id,
            session=session,
            connection_type=self.connection_type,
            base_url=self.base_url,
        )

        # Test the connection
        if await connection.health_check():
            self.logger.debug(f"Created healthy connection: {connection_id}")
        else:
            self.logger.warning(f"Created unhealthy connection: {connection_id}")

        return connection

    def _get_connector_settings(self) -> dict[str, Any]:
        """Get TCP connector settings optimized for connection type."""
        base_settings = {
            "limit": 100,  # Connection pool limit per host
            "limit_per_host": 50,
            "keepalive_timeout": 300,  # 5 minutes
            "enable_cleanup_closed": True,
            "force_close": False,
            "use_dns_cache": True,
            "ttl_dns_cache": 300,
        }

        # Optimize based on connection type
        if self.connection_type == ConnectionType.MARKET_DATA:
            base_settings.update(
                {
                    "limit": 200,  # Higher limit for market data
                    "limit_per_host": 100,
                    "keepalive_timeout": 600,  # Longer keepalive
                }
            )
        elif self.connection_type == ConnectionType.TRADING:
            base_settings.update(
                {
                    "keepalive_timeout": 60,  # Shorter keepalive for fresh connections
                    "force_close": True,  # Force close for trading reliability
                }
            )

        return base_settings

    def _get_timeout_settings(self) -> dict[str, Any]:
        """Get timeout settings optimized for connection type."""
        base_settings = {
            "total": 30,
            "connect": 10,
            "sock_read": 30,
            "sock_connect": 10,
        }

        # Optimize based on connection type
        if self.connection_type == ConnectionType.MARKET_DATA:
            base_settings.update(
                {
                    "total": 10,  # Faster timeout for market data
                    "sock_read": 5,
                }
            )
        elif self.connection_type == ConnectionType.TRADING:
            base_settings.update(
                {
                    "total": 60,  # Longer timeout for trading operations
                    "connect": 15,
                }
            )

        return base_settings

    def _get_default_headers(self) -> dict[str, str]:
        """Get default headers for requests."""
        return {
            "User-Agent": f"T-Bot/{self.exchange}/{self.connection_type.value}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.append(health_task)

        # Adaptive resizing task
        resize_task = asyncio.create_task(self._adaptive_resize_loop())
        self._background_tasks.append(resize_task)

        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self._background_tasks.append(metrics_task)

    @time_execution
    async def acquire_connection(self, user_id: str | None = None) -> ConnectionWrapper:
        """
        Acquire a connection from the pool with affinity support.

        Args:
            user_id: Optional user ID for connection affinity

        Returns:
            Available connection wrapper
        """
        start_time = time.time()

        try:
            # Check circuit breaker
            if self._is_circuit_breaker_open():
                raise ExchangeConnectionError("Circuit breaker is open")

            # Apply rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # Try connection affinity first
            if user_id and user_id in self.affinity_map:
                connection = self._get_affinity_connection(user_id)
                if connection and connection.metrics.status == ConnectionStatus.IDLE:
                    self.active_connections.add(connection.connection_id)
                    self.metrics.total_requests += 1
                    return connection

            # Get connection from pool
            try:
                connection = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=10.0,  # 10 second timeout
                )
            except asyncio.TimeoutError:
                # Pool exhausted, try to create new connection if under limit
                if len(self.connections) < self.max_connections:
                    connection = await self._create_connection(
                        f"{self.exchange}_{self.connection_type.value}_{len(self.connections)}"
                    )
                    self.connections.append(connection)
                else:
                    raise ExchangeConnectionError("Connection pool exhausted")

            # Track connection
            self.active_connections.add(connection.connection_id)
            self.metrics.total_requests += 1

            # Set up affinity if user provided
            if user_id:
                self.affinity_map[user_id] = connection.connection_id

            # Record queue wait time
            wait_time = time.time() - start_time
            self.metrics.queue_wait_time = (
                0.1 * wait_time + 0.9 * self.metrics.queue_wait_time
            )  # Exponential moving average

            return connection

        except Exception as e:
            self.metrics.failed_requests += 1
            self._record_circuit_breaker_failure()
            self.logger.error(f"Failed to acquire connection: {e}")
            raise

    def _get_affinity_connection(self, user_id: str) -> ConnectionWrapper | None:
        """Get connection based on user affinity."""
        connection_id = self.affinity_map.get(user_id)
        if connection_id:
            for connection in self.connections:
                if connection.connection_id == connection_id:
                    return connection
        return None

    async def release_connection(self, connection: ConnectionWrapper) -> None:
        """Release connection back to the pool."""
        try:
            # Remove from active set
            self.active_connections.discard(connection.connection_id)

            # Check connection health before returning to pool
            if await connection.health_check():
                await self.available_connections.put(connection)
            else:
                # Replace unhealthy connection
                await self._replace_unhealthy_connection(connection)

            self._update_pool_metrics()

        except Exception as e:
            self.logger.error(f"Error releasing connection: {e}")

    async def _replace_unhealthy_connection(self, unhealthy_connection: ConnectionWrapper) -> None:
        """Replace an unhealthy connection with a new one."""
        try:
            # Close the unhealthy connection
            await unhealthy_connection.close()

            # Remove from connections list
            self.connections.remove(unhealthy_connection)

            # Create replacement if pool size allows
            if len(self.connections) < self.max_connections:
                new_connection = await self._create_connection(
                    f"{self.exchange}_{self.connection_type.value}_replacement_{int(time.time())}"
                )
                self.connections.append(new_connection)
                await self.available_connections.put(new_connection)

                self.metrics.connection_failures += 1
                self.logger.info(
                    f"Replaced unhealthy connection with {new_connection.connection_id}"
                )

        except Exception as e:
            self.logger.error(f"Failed to replace unhealthy connection: {e}")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_breaker_open:
            return False

        # Check if timeout has passed
        if self.circuit_breaker_last_failure:
            time_since_failure = (
                datetime.now(timezone.utc) - self.circuit_breaker_last_failure
            ).total_seconds()
            if time_since_failure > self.circuit_breaker_timeout:
                self.circuit_breaker_open = False
                self.circuit_breaker_failures = 0
                self.logger.info("Circuit breaker reset")
                return False

        return True

    def _record_circuit_breaker_failure(self) -> None:
        """Record a failure for circuit breaker tracking."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = datetime.now(timezone.utc)

        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            self.metrics.circuit_breaker_trips += 1
            self.logger.warning(
                f"Circuit breaker opened after {self.circuit_breaker_failures} failures",
                extra={
                    "exchange": self.exchange,
                    "connection_type": self.connection_type.value,
                    "failures": self.circuit_breaker_failures,
                },
            )

    async def _health_monitor_loop(self) -> None:
        """Background loop for monitoring connection health."""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                # Check all connections
                unhealthy_connections = []
                for connection in self.connections:
                    if not await connection.health_check():
                        unhealthy_connections.append(connection)

                # Replace unhealthy connections
                for connection in unhealthy_connections:
                    if connection.connection_id not in self.active_connections:
                        await self._replace_unhealthy_connection(connection)

                # Log health status
                healthy_count = len(self.connections) - len(unhealthy_connections)
                self.logger.debug(
                    f"Health check complete: {healthy_count}/{len(self.connections)} healthy",
                    extra={
                        "exchange": self.exchange,
                        "connection_type": self.connection_type.value,
                        "healthy_connections": healthy_count,
                        "total_connections": len(self.connections),
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")

    async def _adaptive_resize_loop(self) -> None:
        """Background loop for adaptive pool resizing."""
        while True:
            try:
                await asyncio.sleep(self.resize_interval)

                # Analyze load patterns
                current_time = datetime.now(timezone.utc)
                active_count = len(self.active_connections)

                # Record load sample
                self.load_history.append((current_time, active_count))

                # Keep only recent history (last hour)
                cutoff_time = current_time - timedelta(hours=1)
                self.load_history = [
                    (t, count) for t, count in self.load_history if t > cutoff_time
                ]

                # Analyze and resize if needed
                await self._analyze_and_resize()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Adaptive resize error: {e}")

    async def _analyze_and_resize(self) -> None:
        """Analyze load patterns and resize pool if needed."""
        if len(self.load_history) < 5:  # Need sufficient data
            return

        # Calculate load statistics
        recent_loads = [count for _, count in self.load_history[-12:]]  # Last hour samples
        avg_load = sum(recent_loads) / len(recent_loads)
        max_load = max(recent_loads)

        current_pool_size = len(self.connections)

        # Resize logic
        if avg_load > current_pool_size * 0.8:  # High utilization
            # Increase pool size
            target_size = min(self.max_connections, int(max_load * 1.2))
            if target_size > current_pool_size:
                await self._resize_pool(target_size)
                self.logger.info(f"Increased pool size to {target_size} due to high load")

        elif avg_load < current_pool_size * 0.3 and current_pool_size > self.min_connections:
            # Decrease pool size
            target_size = max(self.min_connections, int(avg_load * 1.5))
            if target_size < current_pool_size:
                await self._resize_pool(target_size)
                self.logger.info(f"Decreased pool size to {target_size} due to low load")

    async def _resize_pool(self, target_size: int) -> None:
        """Resize the connection pool to target size."""
        current_size = len(self.connections)

        if target_size > current_size:
            # Add connections
            for i in range(target_size - current_size):
                connection = await self._create_connection(
                    f"{self.exchange}_{self.connection_type.value}_resize_{int(time.time())}_{i}"
                )
                self.connections.append(connection)
                await self.available_connections.put(connection)

        elif target_size < current_size:
            # Remove connections (only idle ones)
            connections_to_remove = []
            for connection in self.connections:
                if (
                    len(connections_to_remove) < current_size - target_size
                    and connection.connection_id not in self.active_connections
                ):
                    connections_to_remove.append(connection)

            # Remove connections in O(n) by creating a new list
            self.connections = [
                conn for conn in self.connections if conn not in connections_to_remove
            ]

            # Close removed connections
            for connection in connections_to_remove:
                await connection.close()

        self._update_pool_metrics()

    async def _metrics_collection_loop(self) -> None:
        """Background loop for metrics collection."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute

                # Update pool metrics
                self._update_pool_metrics()

                # Calculate derived metrics
                self._calculate_derived_metrics()

                # Log metrics
                self.logger.info(
                    "Pool metrics",
                    extra={
                        "exchange": self.exchange,
                        "connection_type": self.connection_type.value,
                        "active_connections": self.metrics.active_connections,
                        "total_connections": self.metrics.total_connections,
                        "avg_response_time": self.metrics.avg_response_time,
                        "queue_wait_time": self.metrics.queue_wait_time,
                        "failure_rate": self.metrics.failed_requests
                        / max(self.metrics.total_requests, 1),
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")

    def _update_pool_metrics(self) -> None:
        """Update pool-level metrics."""
        self.metrics.active_connections = len(self.active_connections)
        self.metrics.idle_connections = len(self.connections) - self.metrics.active_connections
        self.metrics.total_connections = len(self.connections)
        self.metrics.queue_depth = self.available_connections.qsize()

    def _calculate_derived_metrics(self) -> None:
        """Calculate derived metrics from connection data."""
        if not self.connections:
            return

        # Calculate average response time across all connections
        total_response_time = 0
        connection_count = 0

        for connection in self.connections:
            if connection.metrics.total_requests > 0:
                total_response_time += connection.metrics.avg_response_time
                connection_count += 1

        if connection_count > 0:
            self.metrics.avg_response_time = total_response_time / connection_count

    async def get_pool_status(self) -> dict[str, Any]:
        """Get comprehensive pool status."""
        return {
            "pool_id": self.metrics.pool_id,
            "exchange": self.exchange,
            "connection_type": self.connection_type.value,
            "metrics": {
                "active_connections": self.metrics.active_connections,
                "idle_connections": self.metrics.idle_connections,
                "total_connections": self.metrics.total_connections,
                "max_connections": self.metrics.max_connections,
                "queue_depth": self.metrics.queue_depth,
                "avg_response_time_ms": self.metrics.avg_response_time * 1000,
                "queue_wait_time_ms": self.metrics.queue_wait_time * 1000,
                "total_requests": self.metrics.total_requests,
                "failed_requests": self.metrics.failed_requests,
                "failure_rate": self.metrics.failed_requests / max(self.metrics.total_requests, 1),
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
            },
            "circuit_breaker": {
                "open": self.circuit_breaker_open,
                "failures": self.circuit_breaker_failures,
                "threshold": self.circuit_breaker_threshold,
                "last_failure": (
                    self.circuit_breaker_last_failure.isoformat()
                    if self.circuit_breaker_last_failure
                    else None
                ),
            },
            "connections": [
                {
                    "connection_id": conn.connection_id,
                    "status": conn.metrics.status.value,
                    "total_requests": conn.metrics.total_requests,
                    "failed_requests": conn.metrics.failed_requests,
                    "avg_response_time_ms": conn.metrics.avg_response_time * 1000,
                    "last_used": conn.metrics.last_used.isoformat(),
                    "error_count": conn.metrics.error_count,
                }
                for conn in self.connections
            ],
            "affinity_mappings": len(self.affinity_map),
        }

    async def cleanup(self) -> None:
        """Cleanup pool resources."""
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Close all connections
            for connection in self.connections:
                await connection.close()

            # Clear state
            self.connections.clear()
            self.active_connections.clear()
            self.affinity_map.clear()

            # Clear queue
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self.logger.info(
                f"Connection pool cleaned up for {self.exchange} {self.connection_type.value}"
            )

        except Exception as e:
            self.logger.error(f"Error during pool cleanup: {e}")


class ConnectionPoolManager(BaseComponent):
    """
    Manager for multiple connection pools across exchanges and connection types.
    """

    def __init__(self, config: Config):
        """Initialize connection pool manager."""
        super().__init__()
        self.config = config
        self.pools: dict[str, EnhancedConnectionPool] = {}

    async def initialize(self) -> None:
        """Initialize all connection pools."""
        # Define pool configurations
        pool_configs = [
            # Binance pools
            ("binance", ConnectionType.REST_API, "https://api.binance.com", 20, 5),
            ("binance", ConnectionType.MARKET_DATA, "https://api.binance.com", 30, 10),
            ("binance", ConnectionType.TRADING, "https://api.binance.com", 10, 3),
            # OKX pools
            ("okx", ConnectionType.REST_API, "https://www.okx.com", 15, 5),
            ("okx", ConnectionType.MARKET_DATA, "https://www.okx.com", 25, 8),
            ("okx", ConnectionType.TRADING, "https://www.okx.com", 8, 3),
            # Coinbase pools
            ("coinbase", ConnectionType.REST_API, "https://api.exchange.coinbase.com", 10, 3),
            ("coinbase", ConnectionType.MARKET_DATA, "https://api.exchange.coinbase.com", 15, 5),
            ("coinbase", ConnectionType.TRADING, "https://api.exchange.coinbase.com", 6, 2),
        ]

        # Initialize pools
        for exchange, conn_type, base_url, max_conn, min_conn in pool_configs:
            pool_key = f"{exchange}_{conn_type.value}"
            pool = EnhancedConnectionPool(
                config=self.config,
                exchange=exchange,
                connection_type=conn_type,
                base_url=base_url,
                max_connections=max_conn,
                min_connections=min_conn,
            )

            await pool.initialize()
            self.pools[pool_key] = pool

        self.logger.info(f"Initialized {len(self.pools)} connection pools")

    def get_pool(self, exchange: str, connection_type: ConnectionType) -> EnhancedConnectionPool:
        """Get connection pool for exchange and type."""
        pool_key = f"{exchange}_{connection_type.value}"
        if pool_key not in self.pools:
            raise ExchangeConnectionError(f"No pool found for {exchange} {connection_type.value}")
        return self.pools[pool_key]

    async def get_global_status(self) -> dict[str, Any]:
        """Get status of all connection pools."""
        pool_statuses = {}

        for pool_key, pool in self.pools.items():
            pool_statuses[pool_key] = await pool.get_pool_status()

        # Calculate global metrics
        total_connections = sum(
            status["metrics"]["total_connections"] for status in pool_statuses.values()
        )
        total_active = sum(
            status["metrics"]["active_connections"] for status in pool_statuses.values()
        )
        total_requests = sum(
            status["metrics"]["total_requests"] for status in pool_statuses.values()
        )
        total_failures = sum(
            status["metrics"]["failed_requests"] for status in pool_statuses.values()
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_metrics": {
                "total_pools": len(self.pools),
                "total_connections": total_connections,
                "active_connections": total_active,
                "idle_connections": total_connections - total_active,
                "utilization_rate": (
                    total_active / total_connections if total_connections > 0 else 0
                ),
                "total_requests": total_requests,
                "total_failures": total_failures,
                "global_failure_rate": total_failures / total_requests if total_requests > 0 else 0,
            },
            "pools": pool_statuses,
        }

    async def cleanup(self) -> None:
        """Cleanup all connection pools."""
        for pool in self.pools.values():
            await pool.cleanup()
        self.pools.clear()
        self.logger.info("All connection pools cleaned up")
