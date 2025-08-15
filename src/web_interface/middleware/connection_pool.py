"""
Connection Pool Middleware for T-Bot web interface.

This middleware manages database and Redis connection pools to optimize
performance and resource utilization for high-frequency trading operations.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import aioredis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import Config
from src.core.logging import get_logger
from src.database.manager import DatabaseManager
from src.database.redis_client import RedisClient

logger = get_logger(__name__)


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
        self.db_pool = None
        self.redis_pool = None
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
        """Initialize database connection pool."""
        try:
            from src.database.manager import DatabaseManager
            
            self.db_pool = DatabaseManager()
            # Note: DatabaseManager uses context manager pattern, no initialize method needed
            
            logger.info(
                f"Database pool initialized: "
                f"min={self.db_pool_config['min_connections']}, "
                f"max={self.db_pool_config['max_connections']}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            # Don't raise, let the app continue without connection pool
            logger.warning("Continuing without database connection pool")

    async def _initialize_redis_pool(self):
        """Initialize Redis connection pool."""
        try:
            # Configure Redis connection
            redis_config = {
                "host": getattr(self.config, "redis_host", "localhost"),
                "port": getattr(self.config, "redis_port", 6379),
                "db": getattr(self.config, "redis_db", 0),
                "password": getattr(self.config, "redis_password", None),
                "ssl": getattr(self.config, "redis_ssl", False),
                "encoding": "utf-8",
                "decode_responses": True,
                "socket_keepalive": True,
                "socket_keepalive_options": {},
                "health_check_interval": self.redis_pool_config["health_check_interval"],
                "retry_on_timeout": self.redis_pool_config["retry_on_timeout"],
                "max_connections": self.redis_pool_config["max_connections"],
            }
            
            # Create Redis connection pool
            self.redis_pool = aioredis.ConnectionPool.from_url(
                f"redis://{redis_config['host']}:{redis_config['port']}/{redis_config['db']}",
                password=redis_config["password"],
                ssl=redis_config["ssl"],
                encoding=redis_config["encoding"],
                decode_responses=redis_config["decode_responses"],
                max_connections=redis_config["max_connections"],
                retry_on_timeout=redis_config["retry_on_timeout"],
                health_check_interval=redis_config["health_check_interval"],
            )
            
            # Test Redis connection
            redis_client = aioredis.Redis(connection_pool=self.redis_pool)
            await redis_client.ping()
            await redis_client.close()
            
            logger.info(
                f"Redis pool initialized: "
                f"max_connections={self.redis_pool_config['max_connections']}, "
                f"host={redis_config['host']}:{redis_config['port']}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}")
            raise

    @asynccontextmanager
    async def get_db_connection(self):
        """
        Get database connection from pool.
        
        Yields:
            Database connection
        """
        if not self._initialized:
            await self.initialize()
        
        connection = None
        try:
            connection = await self.db_pool.get_connection()
            yield connection
        finally:
            if connection:
                await self.db_pool.return_connection(connection)

    @asynccontextmanager
    async def get_redis_connection(self):
        """
        Get Redis connection from pool.
        
        Yields:
            Redis connection
        """
        if not self._initialized:
            await self.initialize()
        
        redis_client = None
        try:
            redis_client = aioredis.Redis(connection_pool=self.redis_pool)
            yield redis_client
        finally:
            if redis_client:
                await redis_client.close()

    async def health_check(self) -> Dict[str, Any]:
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
            async with self.get_db_connection() as conn:
                await conn.execute("SELECT 1")
                results["database"]["status"] = "healthy"
                results["database"]["details"] = {
                    "pool_size": self.db_pool.pool_size if hasattr(self.db_pool, 'pool_size') else "unknown",
                    "available_connections": getattr(self.db_pool, 'available_connections', "unknown"),
                }
        except Exception as e:
            results["database"]["status"] = "unhealthy"
            results["database"]["details"] = {"error": str(e)}
        
        # Check Redis pool
        try:
            async with self.get_redis_connection() as redis:
                await redis.ping()
                results["redis"]["status"] = "healthy"
                results["redis"]["details"] = {
                    "pool_size": self.redis_pool.max_connections if self.redis_pool else "unknown",
                    "created_connections": getattr(self.redis_pool, 'created_connections', "unknown"),
                }
        except Exception as e:
            results["redis"]["status"] = "unhealthy"
            results["redis"]["details"] = {"error": str(e)}
        
        return results

    async def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        stats = {
            "database": {
                "config": self.db_pool_config,
                "status": "initialized" if self.db_pool else "not_initialized",
                "pool_info": {},
            },
            "redis": {
                "config": self.redis_pool_config,
                "status": "initialized" if self.redis_pool else "not_initialized",
                "pool_info": {},
            },
        }
        
        # Get database pool stats
        if self.db_pool and hasattr(self.db_pool, 'get_stats'):
            try:
                stats["database"]["pool_info"] = self.db_pool.get_stats()
            except Exception as e:
                stats["database"]["pool_info"] = {"error": str(e)}
        
        # Get Redis pool stats
        if self.redis_pool:
            try:
                stats["redis"]["pool_info"] = {
                    "max_connections": self.redis_pool.max_connections,
                    "created_connections": getattr(self.redis_pool, 'created_connections', 0),
                    "available_connections": getattr(self.redis_pool, 'available_connections', 0),
                    "in_use_connections": getattr(self.redis_pool, 'in_use_connections', 0),
                }
            except Exception as e:
                stats["redis"]["pool_info"] = {"error": str(e)}
        
        return stats

    async def close(self):
        """Close all connection pools."""
        if not self._initialized:
            return
        
        logger.info("Closing connection pools...")
        
        # Close database pool
        if self.db_pool:
            try:
                await self.db_pool.close()
                logger.info("Database pool closed")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")
        
        # Close Redis pool
        if self.redis_pool:
            try:
                await self.redis_pool.disconnect()
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
        request.state.db_pool = self.pool_manager
        request.state.redis_pool = self.pool_manager
        
        # Add helper methods to request state
        request.state.get_db_connection = self.pool_manager.get_db_connection
        request.state.get_redis_connection = self.pool_manager.get_redis_connection
        
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
_global_pool_manager: Optional[ConnectionPoolManager] = None


def get_global_pool_manager() -> Optional[ConnectionPoolManager]:
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
                redis_status=health_results["redis"]["status"]
            )
            
        except Exception as e:
            logger.error(f"Error performing connection health checks: {e}")

    async def _check_pool_utilization(self, pool_stats: Dict[str, Any]):
        """Check connection pool utilization and alert if high."""
        try:
            # Check database pool utilization
            db_stats = pool_stats.get("database", {}).get("pool_info", {})
            if isinstance(db_stats, dict) and "pool_usage" in db_stats:
                usage = db_stats["pool_usage"]
                if usage > self.alert_thresholds["db_connection_usage"]:
                    await self._alert_high_utilization("database", usage)
            
            # Check Redis pool utilization
            redis_stats = pool_stats.get("redis", {}).get("pool_info", {})
            if isinstance(redis_stats, dict):
                max_conn = redis_stats.get("max_connections", 0)
                in_use = redis_stats.get("in_use_connections", 0)
                if max_conn > 0:
                    usage = in_use / max_conn
                    if usage > self.alert_thresholds["redis_connection_usage"]:
                        await self._alert_high_utilization("redis", usage)
            
        except Exception as e:
            logger.error(f"Error checking pool utilization: {e}")

    async def _alert_connection_issue(self, pool_type: str, details: Dict[str, Any]):
        """Alert about connection pool issues."""
        logger.error(
            f"Connection pool {pool_type} health check failed",
            pool_type=pool_type,
            details=details
        )
        
        # In a production system, this would trigger alerts through
        # monitoring systems like Prometheus, Grafana, or PagerDuty

    async def _alert_high_utilization(self, pool_type: str, utilization: float):
        """Alert about high connection pool utilization."""
        logger.warning(
            f"High connection pool utilization detected",
            pool_type=pool_type,
            utilization=f"{utilization:.2%}",
            threshold=f"{self.alert_thresholds[f'{pool_type}_connection_usage']:.2%}"
        )
        
        # In a production system, this would trigger scaling alerts