"""
Database connection management for the trading bot framework.

This module provides async database connection management for PostgreSQL,
Redis, and InfluxDB with proper connection pooling and health monitoring.

CRITICAL: This module integrates with P-001 core framework and will be
used by all subsequent prompts for data persistence.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, AsyncAdaptedQueuePool
import redis.asyncio as redis
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

# Import core components from P-001
from src.core.config import Config
from src.core.exceptions import DataError, DataSourceError
from src.core.logging import get_logger, PerformanceMonitor

# Import utils from P-007A
from src.utils.decorators import time_execution, retry, circuit_breaker, timeout, log_performance
from src.utils.constants import TIMEOUTS, LIMITS
from src.utils.formatters import format_api_response

# Import error handling from P-002A
from src.error_handling.error_handler import ErrorHandler
from src.error_handling.recovery_scenarios import NetworkDisconnectionRecovery

logger = get_logger(__name__)

# Global connection instances
_async_engine: Optional[Any] = None
_sync_engine: Optional[Any] = None
_redis_client: Optional[redis.Redis] = None
_influxdb_client: Optional[InfluxDBClient] = None


class DatabaseConnectionManager:
    """Manages database connections with health monitoring and reconnection."""

    def __init__(self, config: Config):
        self.config = config
        self.async_engine = None
        self.sync_engine = None
        self.redis_client = None
        self.influxdb_client = None
        self._health_check_task = None
        self._connection_healthy = True
        self.error_handler = ErrorHandler(config)

    @time_execution
    @retry(max_attempts=3)
    @circuit_breaker(failure_threshold=3, recovery_timeout=30)
    @timeout(seconds=60)
    async def initialize(self) -> None:
        """Initialize all database connections."""
        with PerformanceMonitor("database_initialization"):
            try:
                await self._setup_postgresql()
                await self._setup_redis()
                await self._setup_influxdb()
                self._start_health_monitoring()
                logger.info("Database connections initialized successfully")
            except Exception as e:
                # Create error context for comprehensive error handling
                error_context = self.error_handler.create_error_context(
                    error=e,
                    component="database_connection",
                    operation="initialize_all_connections",
                    details={"failed_during": "initialization"}
                )

                # Use ErrorHandler for sophisticated error management
                recovery_scenario = NetworkDisconnectionRecovery(self.config)
                handled = await self.error_handler.handle_error(error_context, recovery_scenario)

                if not handled:
                    logger.error(
                        "Failed to initialize database connections",
                        error=str(e))
                    raise DataSourceError(
                        f"Database initialization failed: {
                            str(e)}")
                else:
                    logger.info(
                        "Database connections recovered after error handling")

    def _start_health_monitoring(self) -> None:
        """Start health monitoring task with network testing."""
        if self._health_check_task is None:
            # Use utils constants for health monitoring setup
            try:
                db_host = self.config.database.postgresql_host
                db_port = self.config.database.postgresql_port
                logger.info(
                    f"Starting health monitoring for {db_host}:{db_port}")
                # Health monitoring interval from utils constants
                monitor_interval = getattr(
                    TIMEOUTS, 'HEALTH_CHECK_INTERVAL', 30)
                logger.debug(f"Health check interval: {monitor_interval}s")
            except Exception as e:
                logger.warning(f"Health monitoring setup warning: {e}")

            # Start the monitoring task
            self._health_check_task = asyncio.create_task(
                self._health_check_loop())
            logger.info("Database health monitoring started")

    async def _health_check_loop(self) -> None:
        """Health check loop with database-specific testing."""
        while True:
            try:
                # Use timeout constants from utils
                health_check_interval = getattr(
                    TIMEOUTS, 'HEALTH_CHECK_INTERVAL', 30)
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
    @circuit_breaker(failure_threshold=2, recovery_timeout=15)
    @timeout(seconds=30)
    async def _setup_postgresql(self) -> None:
        """Setup PostgreSQL connections with async support."""
        try:
            # Create async engine using limits from utils
            max_overflow = getattr(LIMITS, 'DB_MAX_OVERFLOW', 20)
            pool_recycle = getattr(TIMEOUTS, 'DB_POOL_RECYCLE', 3600)

            self.async_engine = create_async_engine(
                self.config.get_async_database_url(),
                echo=self.config.debug,
                poolclass=AsyncAdaptedQueuePool,
                pool_size=self.config.database.postgresql_pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                pool_recycle=pool_recycle,  # Recycle connections every hour
                connect_args={
                    "server_settings": {
                        "application_name": "trading_bot_suite"
                    }
                }
            )

            # Create sync engine for migrations using limits
            sync_pool_size = getattr(LIMITS, 'DB_SYNC_POOL_SIZE', 5)
            sync_max_overflow = getattr(LIMITS, 'DB_SYNC_MAX_OVERFLOW', 10)

            self.sync_engine = create_engine(
                self.config.get_database_url(),
                echo=self.config.debug,
                poolclass=QueuePool,
                pool_size=sync_pool_size,
                max_overflow=sync_max_overflow
            )

            # Test connection
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            logger.info("PostgreSQL connection established")

        except Exception as e:
            # Create error context for comprehensive error handling
            error_context = self.error_handler.create_error_context(
                error=e,
                component="database_connection",
                operation="setup_postgresql",
                details={"database_url": self.config.get_database_url()}
            )

            # Use ErrorHandler for sophisticated error management
            recovery_scenario = NetworkDisconnectionRecovery(self.config)
            handled = await self.error_handler.handle_error(error_context, recovery_scenario)

            if not handled:
                logger.error("PostgreSQL connection failed", error=str(e))
                raise DataSourceError(
                    f"PostgreSQL connection failed: {
                        str(e)}")
            else:
                logger.info(
                    "PostgreSQL connection recovered after error handling")

    @time_execution
    @circuit_breaker(failure_threshold=2, recovery_timeout=15)
    @timeout(seconds=20)
    async def _setup_redis(self) -> None:
        """Setup Redis connection with async support."""
        try:
            self.redis_client = redis.from_url(
                self.config.get_redis_url(),
                decode_responses=True,
                max_connections=100,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            # Create error context for comprehensive error handling
            error_context = self.error_handler.create_error_context(
                error=e,
                component="database_connection",
                operation="setup_redis",
                details={"redis_url": self.config.get_redis_url()}
            )

            # Use ErrorHandler for sophisticated error management
            recovery_scenario = NetworkDisconnectionRecovery(self.config)
            handled = await self.error_handler.handle_error(error_context, recovery_scenario)

            if not handled:
                logger.error("Redis connection failed", error=str(e))
                raise DataSourceError(f"Redis connection failed: {str(e)}")
            else:
                logger.info("Redis connection recovered after error handling")

    @time_execution
    @circuit_breaker(failure_threshold=2, recovery_timeout=15)
    @timeout(seconds=25)
    async def _setup_influxdb(self) -> None:
        """Setup InfluxDB connection."""
        try:
            self.influxdb_client = InfluxDBClient(
                url=f"http://{self.config.database.influxdb_host}:{self.config.database.influxdb_port}",
                token=self.config.database.influxdb_token,
                org=self.config.database.influxdb_org)

            # Test connection
            try:
                self.influxdb_client.ping()
            except Exception as e:
                raise DataSourceError(
                    f"InfluxDB health check failed: {
                        str(e)}")

            logger.info("InfluxDB connection established")

        except Exception as e:
            # Create error context for comprehensive error handling
            error_context = self.error_handler.create_error_context(
                error=e, component="database_connection", operation="setup_influxdb", details={
                    "influxdb_url": f"http: // {
                        self.config.database.influxdb_host}: {
                        self.config.database.influxdb_port}"})

            # Use ErrorHandler for sophisticated error management
            recovery_scenario = NetworkDisconnectionRecovery(self.config)
            handled = await self.error_handler.handle_error(error_context, recovery_scenario)

            if not handled:
                logger.error("InfluxDB connection failed", error=str(e))
                raise DataSourceError(f"InfluxDB connection failed: {str(e)}")
            else:
                logger.info(
                    "InfluxDB connection recovered after error handling")

    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring."""
        self._health_check_task = asyncio.create_task(self._health_monitor())
        logger.info("Database health monitoring started")

    async def _health_monitor(self) -> None:
        """Background health monitoring for all database connections."""
        while True:
            try:
                # Check PostgreSQL
                async with self.async_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))

                # Check Redis
                await self.redis_client.ping()

                # Check InfluxDB
                try:
                    self.influxdb_client.ping()
                except Exception as e:
                    # TODO: Remove in production
                    logger.warning(f"InfluxDB health check failed: {str(e)}")
                    # Don't raise generic exception, just log the warning

                if not self._connection_healthy:
                    logger.info("Database connections recovered")
                    self._connection_healthy = True

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                if self._connection_healthy:
                    logger.warning(
                        "Database health check failed", error=str(e))
                    self._connection_healthy = False

                await asyncio.sleep(10)  # Retry sooner on failure

    @time_execution
    async def get_async_session(self) -> AsyncSession:
        """Get async database session."""
        if not self.async_engine:
            raise DataSourceError("Database not initialized")

        async_session = async_sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        return async_session()

    def get_sync_session(self) -> Session:
        """Get sync database session."""
        if not self.sync_engine:
            raise DataSourceError("Database not initialized")

        SessionLocal = sessionmaker(
            bind=self.sync_engine,
            expire_on_commit=False
        )
        return SessionLocal()

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
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            if self.async_engine:
                # Close all connections in the pool gracefully
                await self.async_engine.dispose()

            if self.sync_engine:
                self.sync_engine.dispose()

            if self.redis_client:
                await self.redis_client.disconnect()

            if self.influxdb_client:
                self.influxdb_client.close()

            logger.info("Database connections closed")

        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
        finally:
            # Ensure we clear the global reference
            global _connection_manager
            _connection_manager = None

    def is_healthy(self) -> bool:
        """Check if all database connections are healthy."""
        return self._connection_healthy


# Global connection manager instance
_connection_manager: Optional[DatabaseConnectionManager] = None


async def initialize_database(config: Config) -> None:
    """Initialize global database connections."""
    global _connection_manager
    _connection_manager = DatabaseConnectionManager(config)
    await _connection_manager.initialize()


async def close_database() -> None:
    """Close global database connections."""
    global _connection_manager
    if _connection_manager:
        await _connection_manager.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    if not _connection_manager:
        raise DataSourceError("Database not initialized")

    session = await _connection_manager.get_async_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


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
async def execute_query(
        query: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Execute a database query with parameters."""
    async with get_async_session() as session:
        result = await session.execute(text(query), params or {})
        return result


async def health_check() -> Dict[str, bool]:
    """Perform comprehensive health check on all databases."""
    health_status = {
        "postgresql": False,
        "redis": False,
        "influxdb": False
    }

    try:
        # Check PostgreSQL
        async with get_async_session() as session:
            await session.execute(text("SELECT 1"))
        health_status["postgresql"] = True
    except Exception as e:
        logger.error("PostgreSQL health check failed", error=str(e))

    try:
        # Check Redis
        redis_client = await get_redis_client()
        await redis_client.ping()
        health_status["redis"] = True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))

    try:
        # Check InfluxDB
        influxdb_client = get_influxdb_client()
        influxdb_client.ping()
        health_status["influxdb"] = True
    except Exception as e:
        logger.error("InfluxDB health check failed", error=str(e))

    return health_status


# TODO: Remove in production - Debug functions
async def debug_connection_info() -> Dict[str, Any]:
    """Debug function to get connection information."""
    if not _connection_manager:
        return format_api_response(
            {}, success=False, message="Database not initialized")

    # Use utils formatter for consistent API response
    debug_data = {
        "postgresql_url": _connection_manager.config.get_database_url().replace(
            _connection_manager.config.database.postgresql_password,
            "***"),
        "redis_url": _connection_manager.config.get_redis_url().replace(
            _connection_manager.config.database.redis_password or "",
            "***"),
        "influxdb_url": f"http: // {
            _connection_manager.config.database.influxdb_host}: {
            _connection_manager.config.database.influxdb_port}",
        "health_status": is_database_healthy()}
    return format_api_response(
        debug_data,
        success=True,
        message="Connection info retrieved")
