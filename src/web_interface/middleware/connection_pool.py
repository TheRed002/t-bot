"""
Connection Pool Middleware for T-Bot web interface.

This middleware manages database and Redis connection pools to optimize
performance and resource utilization for high-frequency trading operations.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import Config
from src.core.logging import get_logger
from src.database import RedisClient

try:
    SQLALCHEMY_AVAILABLE = True
except ImportError as e:
    SQLALCHEMY_AVAILABLE = False
    get_logger(__name__).error(f"SQLAlchemy not available: {e}")

logger = get_logger(__name__)


# Simplified AsyncUnitOfWork for connection pool usage
class PoolAsyncUnitOfWork:
    """
    Simplified async Unit of Work for connection pool usage.

    This is a lightweight version that doesn't initialize all repositories
    but provides the basic async context manager interface.
    """

    def __init__(self, async_session_factory):
        """Initialize with async session factory."""
        self.session_factory = async_session_factory
        self.session: AsyncSession | None = None

    async def __aenter__(self):
        """Async enter context manager."""
        self.session = self.session_factory()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit context manager."""
        if exc_type:
            await self.rollback()
        else:
            try:
                await self.commit()
            except Exception:
                await self.rollback()
                raise

        await self.close()

    async def commit(self):
        """Commit transaction."""
        if self.session:
            await self.session.commit()

    async def rollback(self):
        """Rollback transaction."""
        if self.session:
            await self.session.rollback()

    async def close(self):
        """Close session."""
        if self.session:
            await self.session.close()
            self.session = None


class ConnectionPoolManager:
    """
    Manages connection pools for database and Redis connections.

    This class provides efficient connection management for high-performance
    trading applications where connection overhead can impact latency.
    """

    def __init__(self, config: Config):
        """
        Initialize connection pool manager.

        Args:
            config: Application configuration
        """
        self.config = config
        self._async_engine = None
        self._async_session_factory = None
        self._uow_factory = None
        self.redis_client: RedisClient | None = None
        self._initialized = False

        # Connection pool configuration
        self.db_pool_config = {
            "min_connections": getattr(config, "db_pool_min", 5),
            "max_connections": getattr(config, "db_pool_max", 20),
            "pool_timeout": getattr(config, "db_pool_timeout", 30.0),
            "idle_timeout": getattr(config, "db_idle_timeout", 300.0),  # 5 minutes
            "connection_lifetime": getattr(config, "db_connection_lifetime", 3600.0),  # 1 hour
        }

        self.redis_pool_config = {
            "min_connections": getattr(config, "redis_pool_min", 5),
            "max_connections": getattr(config, "redis_pool_max", 50),
            "pool_timeout": getattr(config, "redis_pool_timeout", 10.0),
            "retry_on_timeout": True,
            "health_check_interval": 30.0,
        }

    async def initialize(self):
        """Initialize connection pools."""
        if self._initialized:
            return

        try:
            logger.info("Initializing connection pools...")

            # Initialize database pool
            await self._initialize_database_pool()

            # Initialize Redis pool
            await self._initialize_redis_pool()

            self._initialized = True
            logger.info("Connection pools initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise

    async def _initialize_database_pool(self):
        """Initialize async database connection pool."""
        try:
            # Get async database URL from config with proper validation
            database_url = None
            if hasattr(self.config, "get_async_database_url"):
                database_url = self.config.get_async_database_url()
            elif hasattr(self.config, "database") and hasattr(
                self.config.database, "postgresql_url"
            ):
                # Convert sync URL to async URL
                sync_url = self.config.database.postgresql_url
                if sync_url.startswith("postgresql://"):
                    database_url = sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
                else:
                    database_url = sync_url

            if database_url and SQLALCHEMY_AVAILABLE:
                # Use NullPool under pytest to avoid pooled connection GC warnings
                use_null_pool = bool(os.getenv("PYTEST_CURRENT_TEST"))
                pool_class = NullPool if use_null_pool else AsyncAdaptedQueuePool

                # Create async engine with pool configuration
                engine_kwargs = {
                    "echo": getattr(self.config, "debug", False),
                    "poolclass": pool_class,
                    "connect_args": {
                        "server_settings": {"application_name": "t_bot_connection_pool"}
                    },
                }

                if not use_null_pool:
                    engine_kwargs.update(
                        {
                            "pool_size": self.db_pool_config["min_connections"],
                            "max_overflow": (
                                self.db_pool_config["max_connections"]
                                - self.db_pool_config["min_connections"]
                            ),
                            "pool_timeout": self.db_pool_config["pool_timeout"],
                            "pool_recycle": self.db_pool_config["connection_lifetime"],
                            "pool_pre_ping": True,
                            "pool_reset_on_return": "commit",
                        }
                    )

                self._async_engine = create_async_engine(database_url, **engine_kwargs)

                # Create async session factory
                self._async_session_factory = async_sessionmaker(
                    self._async_engine, class_=AsyncSession, expire_on_commit=False
                )

                # Create async UnitOfWork factory
                def create_async_uow():
                    """Create a new async Unit of Work instance."""
                    # Use simplified UoW for connection pool to avoid circular dependencies
                    return PoolAsyncUnitOfWork(self._async_session_factory)

                self._uow_factory = create_async_uow

                # Test the connection
                async with self._async_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))

                logger.info(
                    f"Async database pool initialized: "
                    f"min={self.db_pool_config['min_connections']}, "
                    f"max={self.db_pool_config['max_connections']}"
                )
            else:
                logger.warning("No async database URL configured")

        except Exception as e:
            logger.error(f"Failed to initialize async database pool: {e}")
            # Don't raise, let the app continue without connection pool
            logger.warning("Continuing without database connection pool")

    async def _initialize_redis_pool(self):
        """Initialize Redis connection pool."""
        try:
            # Use the database module's RedisClient
            self.redis_client = RedisClient(self.config, auto_close=False)
            await self.redis_client.connect()

            # Test connection
            await self.redis_client.ping()

            logger.info("Redis pool initialized using database module's RedisClient")

        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            # Don't raise for Redis, it's not critical
            logger.warning("Continuing without Redis connection pool")

    @asynccontextmanager
    async def get_db_connection(self):
        """
        Get async database connection from pool.

        Yields:
            AsyncSession instance
        """
        if not self._initialized:
            await self.initialize()

        if not self._async_session_factory:
            raise RuntimeError("Async database pool not initialized")

        async with self._async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def get_uow(self):
        """
        Get async Unit of Work instance.

        Yields:
            AsyncUnitOfWork instance
        """
        if not self._initialized:
            await self.initialize()

        if not self._uow_factory:
            raise RuntimeError("UnitOfWork factory not initialized")

        uow = self._uow_factory()
        async with uow:
            yield uow

    @asynccontextmanager
    async def get_redis_connection(self):
        """
        Get Redis connection from pool.

        Yields:
            Redis client
        """
        if not self._initialized:
            await self.initialize()

        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        # RedisClient manages its own connection pooling
        yield self.redis_client

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on connection pools.

        Returns:
            Health check results
        """
        results = {
            "database": {"status": "unknown", "details": {}},
            "redis": {"status": "unknown", "details": {}},
        }

        # Check database pool
        try:
            async with self.get_db_connection() as session:
                result = await session.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    results["database"]["status"] = "healthy"

                    # Get pool info if available from the engine
                    if self._async_engine and hasattr(self._async_engine, "pool"):
                        pool = self._async_engine.pool
                        if hasattr(pool, "size") and hasattr(pool, "checked_out"):
                            try:
                                size_val = pool.size() if callable(pool.size) else pool.size
                                checked_out_val = (
                                    pool.checked_out()
                                    if callable(pool.checked_out)
                                    else pool.checked_out
                                )
                                overflow_val = (
                                    pool.overflow()
                                    if callable(getattr(pool, "overflow", lambda: 0))
                                    else getattr(pool, "overflow", 0)
                                )
                                results["database"]["details"] = {
                                    "pool_size": size_val,
                                    "checked_out": checked_out_val,
                                    "overflow": overflow_val,
                                }
                            except Exception as pool_err:
                                logger.debug(f"Could not get pool stats: {pool_err}")
                                results["database"]["details"] = {"pool_stats": "unavailable"}
        except Exception as e:
            results["database"]["status"] = "unhealthy"
            results["database"]["details"] = {"error": str(e)}

        # Check Redis pool
        try:
            if self.redis_client:
                is_healthy = await self.redis_client.ping()
                if is_healthy:
                    results["redis"]["status"] = "healthy"
                    info = await self.redis_client.info()
                    results["redis"]["details"] = {
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                    }
                else:
                    results["redis"]["status"] = "unhealthy"
                    results["redis"]["details"] = {"error": "Ping failed"}
            else:
                results["redis"]["status"] = "not_initialized"
        except Exception as e:
            results["redis"]["status"] = "unhealthy"
            results["redis"]["details"] = {"error": str(e)}

        return results

    async def get_pool_stats(self) -> dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Pool statistics
        """
        stats = {
            "database": {
                "config": self.db_pool_config,
                "status": "initialized" if self._async_session_factory else "not_initialized",
                "pool_info": {},
            },
            "redis": {
                "config": self.redis_pool_config,
                "status": "initialized" if self.redis_client else "not_initialized",
                "pool_info": {},
            },
        }

        # Get database pool stats from async engine
        if self._async_engine:
            try:
                pool = getattr(self._async_engine, "pool", None)
                if pool and hasattr(pool, "size"):
                    try:
                        # Handle both callable and property access patterns
                        size_val = pool.size() if callable(pool.size) else pool.size
                        checked_out_val = (
                            pool.checked_out()
                            if callable(getattr(pool, "checked_out", lambda: 0))
                            else getattr(pool, "checked_out", 0)
                        )
                        overflow_val = (
                            pool.overflow()
                            if callable(getattr(pool, "overflow", lambda: 0))
                            else getattr(pool, "overflow", 0)
                        )

                        stats["database"]["pool_info"] = {
                            "size": size_val,
                            "checked_out": checked_out_val,
                            "overflow": overflow_val,
                            "total": size_val + overflow_val,
                        }
                    except Exception as pool_err:
                        stats["database"]["pool_info"] = {"error": f"Pool stats error: {pool_err}"}
                else:
                    stats["database"]["pool_info"] = {"type": "NullPool or no stats available"}
            except Exception as e:
                stats["database"]["pool_info"] = {"error": str(e)}

        # Get Redis stats
        if self.redis_client:
            try:
                info = await self.redis_client.info()
                stats["redis"]["pool_info"] = {
                    "connected_clients": info.get("connected_clients", 0),
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "uptime_in_days": info.get("uptime_in_days", 0),
                }
            except Exception as e:
                stats["redis"]["pool_info"] = {"error": str(e)}

        return stats

    async def close(self):
        """Close all connection pools."""
        if not self._initialized:
            return

        logger.info("Closing connection pools...")

        # Close async database pool
        if self._async_engine:
            try:
                # SQLAlchemy async pools are closed when engine is disposed
                await self._async_engine.dispose()
                logger.info("Async database pool closed")
            except Exception as e:
                logger.error(f"Error closing async database pool: {e}")

        # Close Redis pool
        if self.redis_client:
            try:
                await self.redis_client.disconnect()
                logger.info("Redis pool closed")
            except Exception as e:
                logger.error(f"Error closing Redis pool: {e}")

        self._initialized = False
        logger.info("All connection pools closed")


class ConnectionPoolMiddleware(BaseHTTPMiddleware):
    """
    Middleware to provide connection pool access to requests.

    This middleware ensures that database and Redis connections are available
    to request handlers through the request state.
    """

    def __init__(self, app, config: Config):
        """
        Initialize connection pool middleware.

        Args:
            app: FastAPI application
            config: Application configuration
        """
        super().__init__(app)
        self.config = config
        self.pool_manager = ConnectionPoolManager(config)

    async def dispatch(self, request: Request, call_next):
        """
        Process request with connection pool access.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Initialize pools if not already done
        if not self.pool_manager._initialized:
            await self.pool_manager.initialize()

        # Add pool manager to request state
        request.state.pool_manager = self.pool_manager

        # Add helper methods to request state
        request.state.get_db_connection = self.pool_manager.get_db_connection
        request.state.get_redis_connection = self.pool_manager.get_redis_connection
        request.state.get_uow = self.pool_manager.get_uow

        # Process request
        response = await call_next(request)

        return response

    async def startup(self):
        """Startup event handler."""
        await self.pool_manager.initialize()

    async def shutdown(self):
        """Shutdown event handler."""
        await self.pool_manager.close()


# Global pool manager instance
_global_pool_manager: ConnectionPoolManager | None = None


def get_global_pool_manager() -> ConnectionPoolManager | None:
    """Get global pool manager instance."""
    return _global_pool_manager


def set_global_pool_manager(pool_manager: ConnectionPoolManager):
    """Set global pool manager instance."""
    global _global_pool_manager
    _global_pool_manager = pool_manager


async def get_db_connection():
    """
    Dependency for getting database connection.

    Yields:
        Database connection
    """
    pool_manager = get_global_pool_manager()
    if not pool_manager:
        raise RuntimeError("Connection pool not initialized")

    async with pool_manager.get_db_connection() as conn:
        yield conn


async def get_redis_connection():
    """
    Dependency for getting Redis connection.

    Yields:
        Redis connection
    """
    pool_manager = get_global_pool_manager()
    if not pool_manager:
        raise RuntimeError("Connection pool not initialized")

    async with pool_manager.get_redis_connection() as conn:
        yield conn


async def get_uow():
    """
    Dependency for getting Unit of Work.

    Yields:
        UnitOfWork instance
    """
    pool_manager = get_global_pool_manager()
    if not pool_manager:
        raise RuntimeError("Connection pool not initialized")

    async with pool_manager.get_uow() as uow:
        yield uow


class ConnectionHealthMonitor:
    """
    Monitor connection pool health and performance.

    This class provides monitoring and alerting for connection pool issues
    that could impact trading system performance.
    """

    def __init__(self, pool_manager: ConnectionPoolManager):
        """
        Initialize connection health monitor.

        Args:
            pool_manager: Connection pool manager to monitor
        """
        self.pool_manager = pool_manager
        self.monitoring_enabled = True
        self.check_interval = 60.0  # 1 minute
        self._monitor_task = None

        # Alert thresholds
        self.alert_thresholds = {
            "db_connection_usage": 0.8,  # 80% pool utilization
            "redis_connection_usage": 0.8,  # 80% pool utilization
            "connection_timeout_rate": 0.1,  # 10% timeout rate
            "health_check_failure_rate": 0.05,  # 5% failure rate
        }

    async def start_monitoring(self):
        """Start connection health monitoring."""
        if self._monitor_task:
            return

        logger.info("Starting connection health monitoring")
        self.monitoring_enabled = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """Stop connection health monitoring."""
        if not self._monitor_task:
            return

        logger.info("Stopping connection health monitoring")
        self.monitoring_enabled = False
        self._monitor_task.cancel()

        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass

        self._monitor_task = None

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _perform_health_checks(self):
        """Perform connection pool health checks."""
        try:
            health_results = await self.pool_manager.health_check()
            pool_stats = await self.pool_manager.get_pool_stats()

            # Check database health
            if health_results["database"]["status"] != "healthy":
                await self._alert_connection_issue("database", health_results["database"])

            # Check Redis health
            if health_results["redis"]["status"] != "healthy":
                await self._alert_connection_issue("redis", health_results["redis"])

            # Check pool utilization
            await self._check_pool_utilization(pool_stats)

            # Log periodic health status
            logger.debug(
                "Connection pool health check completed",
                db_status=health_results["database"]["status"],
                redis_status=health_results["redis"]["status"],
            )

        except Exception as e:
            logger.error(f"Error performing connection health checks: {e}")

    async def _check_pool_utilization(self, pool_stats: dict[str, Any]):
        """Check connection pool utilization and alert if high."""
        try:
            # Check database pool utilization
            db_stats = pool_stats.get("database", {}).get("pool_info", {})
            if isinstance(db_stats, dict) and "checked_out" in db_stats and "size" in db_stats:
                size = db_stats["size"]
                checked_out = db_stats["checked_out"]
                if size > 0:
                    usage = checked_out / size
                    if usage > self.alert_thresholds["db_connection_usage"]:
                        await self._alert_high_utilization("database", usage)

            # Redis utilization is harder to measure with the abstracted client
            # Log stats for monitoring
            redis_stats = pool_stats.get("redis", {}).get("pool_info", {})
            if isinstance(redis_stats, dict):
                logger.debug("Redis pool stats", **redis_stats)

        except Exception as e:
            logger.error(f"Error checking pool utilization: {e}")

    async def _alert_connection_issue(self, pool_type: str, details: dict[str, Any]):
        """Alert about connection pool issues."""
        logger.error(
            f"Connection pool {pool_type} health check failed", pool_type=pool_type, details=details
        )

        # In a production system, this would trigger alerts through
        # monitoring systems like Prometheus, Grafana, or PagerDuty

    async def _alert_high_utilization(self, pool_type: str, utilization: float):
        """Alert about high connection pool utilization."""
        logger.warning(
            "High connection pool utilization detected",
            pool_type=pool_type,
            utilization=f"{utilization:.2%}",
            threshold=f"{self.alert_thresholds[f'{pool_type}_connection_usage']:.2%}",
        )

        # In a production system, this would trigger scaling alerts
