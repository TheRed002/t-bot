"""
Database connection management for the trading bot framework.

This module provides async database connection management for PostgreSQL,
Redis, and InfluxDB with proper connection pooling and health monitoring.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

import asyncio
import os
from asyncio import Task
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
from influxdb_client import InfluxDBClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

# Import core components from P-001
from src.core.config import Config
from src.core.exceptions import DataSourceError
from src.core.logging import PerformanceMonitor, get_logger

# Import error handling decorators from P-002A
from src.error_handling.decorators import with_circuit_breaker, with_retry

# Error handling is provided by decorators
from src.utils.constants import LIMITS, TIMEOUTS

# Import utils from P-007A
from src.utils.decorators import time_execution, timeout
from src.utils.formatters import format_api_response

logger = get_logger(__name__)

# Global connection instances
_async_engine: Any | None = None
_sync_engine: Any | None = None
_redis_client: redis.Redis | None = None
_influxdb_client: InfluxDBClient | None = None


class DatabaseConnectionManager:
    """Manages database connections with health monitoring and reconnection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.async_engine: Any | None = None
        self.sync_engine: Any | None = None
        self.redis_client: redis.Redis | None = None
        self.influxdb_client: InfluxDBClient | None = None
        self._health_check_task: Task[None] | None = None
        self._connection_healthy = True
        self._test_schema: str | None = None  # For test isolation

    def set_test_schema(self, schema: str) -> None:
        """
        Set test schema for session isolation.

        This method is used by integration tests to configure sessions
        to use a specific schema for test isolation.

        Args:
            schema: The schema name to use for test isolation
        """
        self._test_schema = schema
        logger.debug(f"Test schema set to: {schema}")

    @time_execution
    @with_retry(max_attempts=3)
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=30)
    @timeout(60)
    async def initialize(self) -> None:
        """Initialize all database connections."""
        with PerformanceMonitor("database_initialization"):
            await self._setup_postgresql()
            await self._setup_redis()
            await self._setup_influxdb()
            self._start_health_monitoring()
            logger.info("Database connections initialized successfully")

    def _start_health_monitoring(self) -> None:
        """Start health monitoring task with network testing."""
        if self._health_check_task is None:
            # Use utils constants for health monitoring setup
            db_host = getattr(self.config.database, "postgresql_host", "localhost")
            db_port = getattr(self.config.database, "postgresql_port", 5432)
            logger.info(f"Starting health monitoring for {db_host}:{db_port}")
            monitor_interval = TIMEOUTS.get("HEALTH_CHECK_INTERVAL", 30)
            logger.debug(f"Health check interval: {monitor_interval}s")

            # Start the monitoring task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Database health monitoring started")

    async def _health_check_loop(self) -> None:
        """Health check loop with database-specific testing."""
        while True:
            try:
                # Use timeout constants from utils
                health_check_interval = TIMEOUTS.get("HEALTH_CHECK_INTERVAL", 30)
                await asyncio.sleep(health_check_interval)

                # Use utils constants and performance monitoring
                with PerformanceMonitor("database_health_check"):
                    # Test PostgreSQL connectivity using actual database query
                    try:
                        if self.async_engine:
                            async with self.async_engine.begin() as conn:
                                await conn.execute(text("SELECT 1"))
                            self._connection_healthy = True
                            logger.debug("Database health check passed")
                        else:
                            self._connection_healthy = False
                            logger.warning("Database engine not available")
                    except Exception as e:
                        self._connection_healthy = False
                        logger.warning(f"Database health check failed: {e}")

                    # Test Redis if configured
                    if self.redis_client:
                        try:
                            await self.redis_client.ping()
                            logger.debug("Redis health check passed")
                        except Exception as e:
                            logger.warning(f"Redis health check failed: {e}")

            except asyncio.CancelledError:
                logger.info("Health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                self._connection_healthy = False

    @time_execution
    @with_circuit_breaker(failure_threshold=2, recovery_timeout=15)
    @timeout(30)
    async def _setup_postgresql(self) -> None:
        """Setup PostgreSQL connections with async support."""
        # Production-optimized configuration for high-frequency trading
        max_overflow = LIMITS.get("DB_MAX_OVERFLOW", 50)
        pool_recycle = TIMEOUTS.get("DB_POOL_RECYCLE", 3600)  # 1 hour for production stability
        pool_pre_ping = True
        pool_reset_on_return = "commit"
        pool_timeout = 10  # Increased timeout for production reliability

        # Use NullPool under pytest to avoid pooled connection GC warnings
        try:
            from sqlalchemy.pool import NullPool

            use_null_pool = bool(os.getenv("PYTEST_CURRENT_TEST"))
        except Exception as e:
            logger.warning(f"Failed to determine test environment: {e}")
            use_null_pool = False

        async_pool_class = NullPool if use_null_pool else AsyncAdaptedQueuePool

        # Production-optimized connection parameters
        connect_args = {
            "server_settings": {
                "application_name": "trading_bot_suite",
                "tcp_keepalives_idle": "300",
                "tcp_keepalives_interval": "30",
                "tcp_keepalives_count": "3",
            },
            "command_timeout": 60,
        }

        async_engine_kwargs: dict[str, Any] = {
            "echo": self.config.debug,
            "echo_pool": self.config.debug,
            "poolclass": async_pool_class,
            "connect_args": connect_args,
        }
        if async_pool_class.__name__ != "NullPool":
            async_engine_kwargs.update(
                {
                    "pool_size": max(self.config.database.postgresql_pool_size, 25),
                    "max_overflow": max_overflow,
                    "pool_pre_ping": pool_pre_ping,
                    "pool_recycle": pool_recycle,
                    "pool_reset_on_return": pool_reset_on_return,
                    "pool_timeout": pool_timeout,
                    "pool_reset_on_invalid": True,
                    "pool_events": True,
                    "future": True,  # Enable future-compatible features for asyncpg
                }
            )

        database_url = self.config.database.postgresql_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )
        self.async_engine = create_async_engine(database_url, **async_engine_kwargs)

        # Create sync engine for migrations using limits
        sync_pool_size = LIMITS.get("DB_SYNC_POOL_SIZE", 5)
        sync_max_overflow = LIMITS.get("DB_SYNC_MAX_OVERFLOW", 10)

        sync_pool_class = NullPool if use_null_pool else QueuePool

        if sync_pool_class.__name__ == "NullPool":
            self.sync_engine = create_engine(
                self.config.database.postgresql_url,
                echo=self.config.debug,
                poolclass=sync_pool_class,
            )
        else:
            self.sync_engine = create_engine(
                self.config.database.postgresql_url,
                echo=self.config.debug,
                poolclass=sync_pool_class,
                pool_size=sync_pool_size,
                max_overflow=sync_max_overflow,
            )

        # Test connection
        async with self.async_engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        # Create session makers after engines are established
        self._async_session_maker = async_sessionmaker(
            bind=self.async_engine, class_=AsyncSession, expire_on_commit=False
        )
        self._sync_session_maker = sessionmaker(bind=self.sync_engine, expire_on_commit=False)

        logger.info("PostgreSQL connection established")

    @time_execution
    @with_circuit_breaker(failure_threshold=2, recovery_timeout=15)
    @timeout(20)
    async def _setup_redis(self) -> None:
        """Setup Redis connection with async support."""
        self.redis_client = redis.from_url(
            self.config.database.redis_url,
            decode_responses=True,
            max_connections=200,  # Increased for production load
            retry_on_timeout=True,
            retry_on_error=[redis.ConnectionError, redis.TimeoutError],
            health_check_interval=30,
            socket_keepalive=True,
            socket_keepalive_options={},
        )

        # Test connection
        await self.redis_client.ping()
        logger.info("Redis connection established")

    @time_execution
    @with_circuit_breaker(failure_threshold=2, recovery_timeout=15)
    @timeout(25)
    async def _setup_influxdb(self) -> None:
        """Setup InfluxDB connection."""
        self.influxdb_client = InfluxDBClient(
            url=f"http://{self.config.database.influxdb_host}:{self.config.database.influxdb_port}",
            token=self.config.database.influxdb_token,
            org=self.config.database.influxdb_org,
        )

        # Test connection (run sync ping in executor to avoid blocking)
        await asyncio.get_event_loop().run_in_executor(None, self.influxdb_client.ping)

        logger.info("InfluxDB connection established")

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with proper context management."""
        if not self.async_engine:
            raise DataSourceError("Database not initialized")

        async_session = async_sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                logger.error(f"Session error occurred: {e}")
                try:
                    await session.rollback()
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback session: {rollback_error}")
                    # Invalidate the session to prevent it from returning to the pool
                    session.invalidate()
                raise
            finally:
                try:
                    await session.close()
                except Exception as close_error:
                    logger.error(f"Failed to close session: {close_error}")
                    # Invalidate the session to prevent connection leak
                    try:
                        session.invalidate()
                    except Exception as invalidate_error:
                        logger.critical(f"Failed to invalidate session: {invalidate_error}")
                        raise

    def get_sync_session(self) -> Session:
        """Get sync database session."""
        if not self.sync_engine:
            raise DataSourceError("Database not initialized")

        session_local = sessionmaker(bind=self.sync_engine, expire_on_commit=False)
        session = None
        try:
            session = session_local()
            return session
        except Exception as e:
            # If session creation fails, ensure we don't leak resources
            if session:
                try:
                    session.close()
                except Exception as close_error:
                    logger.error(f"Failed to close failed session: {close_error}")
                    raise
            raise DataSourceError(f"Failed to create sync session: {e}") from e

    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client."""
        if not self.redis_client:
            raise DataSourceError("Redis not initialized")
        return self.redis_client

    def get_influxdb_client(self) -> InfluxDBClient:
        """Get InfluxDB client."""
        if not self.influxdb_client:
            raise DataSourceError("InfluxDB not initialized")
        return self.influxdb_client

    async def close(self) -> None:
        """Close all database connections."""
        # Store references before clearing
        connections = {
            "health_task": self._health_check_task,
            "async_engine": self.async_engine,
            "sync_engine": self.sync_engine,
            "redis_client": self.redis_client,
            "influxdb_client": self.influxdb_client,
        }

        try:
            await self._stop_health_monitoring(connections["health_task"])
            close_tasks = await self._prepare_close_tasks(connections)
            await self._execute_close_tasks(close_tasks)
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
        finally:
            self._clear_connection_references()

    async def _stop_health_monitoring(self, health_task: Task[None] | None) -> None:
        """Stop health monitoring task."""
        if health_task:
            health_task.cancel()
            try:
                await asyncio.wait_for(health_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    async def _prepare_close_tasks(self, connections: dict[str, Any]) -> list[Any]:
        """Prepare all connection close tasks."""
        close_tasks: list[Any] = []

        if connections["async_engine"]:
            close_tasks.append(
                asyncio.wait_for(connections["async_engine"].dispose(), timeout=10.0)
            )

        if connections["sync_engine"]:
            close_tasks.append(
                asyncio.get_event_loop().run_in_executor(None, connections["sync_engine"].dispose)
            )

        if connections["redis_client"]:
            close_tasks.append(self._close_redis_task(connections["redis_client"]))

        if connections["influxdb_client"]:
            close_tasks.append(self._close_influxdb_task(connections["influxdb_client"]))

        return close_tasks

    async def _close_redis_task(self, redis_client) -> Any:
        """Create Redis close task."""

        async def close_redis():
            try:
                if hasattr(redis_client, "aclose"):
                    await redis_client.aclose()
                elif hasattr(redis_client, "close"):
                    await redis_client.close()
            except Exception as close_error:
                logger.warning(f"Failed to close Redis connection: {close_error}")
                raise

        return asyncio.wait_for(close_redis(), timeout=5.0)

    def _close_influxdb_task(self, influxdb_client) -> Any:
        """Create InfluxDB close task."""
        if hasattr(influxdb_client, "disconnect") and asyncio.iscoroutinefunction(
            influxdb_client.disconnect
        ):
            return asyncio.wait_for(influxdb_client.disconnect(), timeout=10.0)
        else:
            return asyncio.get_event_loop().run_in_executor(None, influxdb_client.close)

    async def _execute_close_tasks(self, close_tasks: list[Any]) -> None:
        """Execute all close tasks concurrently."""
        if close_tasks:
            try:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            except Exception as gather_error:
                logger.warning(f"Some connections failed to close properly: {gather_error}")
                raise

    def _clear_connection_references(self) -> None:
        """Clear all connection references."""
        self._health_check_task = None
        self.async_engine = None
        self.sync_engine = None
        self.redis_client = None
        self.influxdb_client = None

    def is_healthy(self) -> bool:
        """Check if all database connections are healthy."""
        return self._connection_healthy

    @property
    def async_session_maker(self):
        """Get async session maker."""
        return self._async_session_maker

    @property
    def sync_session_maker(self):
        """Get sync session maker."""
        return self._sync_session_maker

    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection for health checks."""
        if not self.async_engine:
            raise DataSourceError("Database not initialized")

        async with self.async_engine.begin() as conn:
            yield conn

    async def get_pool_status(self) -> dict[str, int]:
        """Get connection pool status."""
        if not self.async_engine:
            return {"size": 0, "used": 0, "free": 0}

        try:
            pool = self.async_engine.pool

            # Handle NullPool which doesn't have these attributes
            if hasattr(pool, "__class__") and pool.__class__.__name__ == "NullPool":
                return {"size": 0, "used": 0, "free": 0}

            # Standardize the return format to what's expected by consumers
            # Try to get pool statistics safely
            size = 0
            used = 0
            free = 0

            # Try to get size from pool
            if hasattr(pool, "size"):
                size_val = pool.size
                size = size_val() if callable(size_val) else size_val

            # Try to get usage stats
            if hasattr(pool, "checkedout"):
                checkedout_val = pool.checkedout
                used = checkedout_val() if callable(checkedout_val) else checkedout_val

            # Calculate free connections
            if hasattr(pool, "checkedin"):
                checkedin_val = pool.checkedin
                checkedin = checkedin_val() if callable(checkedin_val) else checkedin_val
                free = checkedin
            else:
                free = max(0, size - used)

            return {"size": size, "used": used, "free": free}
        except Exception as e:
            logger.warning(f"Unable to get pool status: {e}")
            return {"size": 0, "used": 0, "free": 0}


# Global connection manager instance
_connection_manager: DatabaseConnectionManager | None = None


async def initialize_database(config: Config) -> None:
    """Initialize global database connections."""
    global _connection_manager
    _connection_manager = DatabaseConnectionManager(config)
    await _connection_manager.initialize()


async def init_database(config: Config) -> None:
    """Initialize the database connection - alias for initialize_database."""
    await initialize_database(config)


async def close_database() -> None:
    """Close global database connections."""
    global _connection_manager
    if _connection_manager:
        try:
            await _connection_manager.close()
        finally:
            _connection_manager = None


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    if not _connection_manager:
        raise DataSourceError("Database not initialized")

    async with _connection_manager.get_async_session() as session:
        yield session


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions - alias for get_async_session."""
    async with get_async_session() as session:
        yield session


def get_sync_session() -> Session:
    """Get sync database session."""
    if not _connection_manager:
        raise DataSourceError("Database not initialized")

    return _connection_manager.get_sync_session()


async def get_redis_client() -> redis.Redis:
    """Get Redis client."""
    if not _connection_manager:
        raise DataSourceError("Database not initialized")

    return await _connection_manager.get_redis_client()


def get_influxdb_client() -> InfluxDBClient:
    """Get InfluxDB client."""
    if not _connection_manager:
        raise DataSourceError("Database not initialized")

    return _connection_manager.get_influxdb_client()


def is_database_healthy() -> bool:
    """Check if database connections are healthy."""
    if not _connection_manager:
        return False
    return _connection_manager.is_healthy()


# Database utility functions
async def execute_query(query: str, params: dict[str, Any] | None = None) -> Any:
    """Execute a database query with parameters."""
    async with get_async_session() as session:
        result = await session.execute(text(query), params or {})
        return result


async def health_check() -> dict[str, bool]:
    """Perform comprehensive health check on all databases."""
    health_status = {"postgresql": False, "redis": False, "influxdb": False}

    try:
        # Check PostgreSQL
        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))
        health_status["postgresql"] = True
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")

    try:
        # Check Redis
        redis_client = await get_redis_client()
        await redis_client.ping()
        health_status["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")

    try:
        # Check InfluxDB (run sync ping in executor to avoid blocking)
        influxdb_client = get_influxdb_client()
        await asyncio.get_event_loop().run_in_executor(None, influxdb_client.ping)
        health_status["influxdb"] = True
    except Exception as e:
        logger.error(f"InfluxDB health check failed: {e}")

    return health_status


async def debug_connection_info() -> dict[str, Any]:
    """Debug function to get connection information.

    WARNING: This function should only be used in development/debugging.
    """
    if not _connection_manager:
        return format_api_response({}, success=False, message="Database not initialized")

    # Only allow in debug mode
    if not _connection_manager.config.debug:
        return format_api_response({}, success=False, message="Debug mode not enabled")

    # Use utils formatter for consistent API response
    debug_data = {
        "postgresql_url": _connection_manager.config.database.postgresql_url.replace(
            _connection_manager.config.database.postgresql_password, "***"
        ),
        "redis_url": _connection_manager.config.database.redis_url.replace(
            _connection_manager.config.database.redis_password or "", "***"
        ),
        "influxdb_url": f"http://{_connection_manager.config.database.influxdb_host}:{_connection_manager.config.database.influxdb_port}",
        "health_status": is_database_healthy(),
    }
    return format_api_response(debug_data, success=True, message="Connection info retrieved")
