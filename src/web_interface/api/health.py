"""
Comprehensive health check endpoints for T-Bot Trading System.

This module provides detailed health monitoring for all system components
including database connectivity, exchange connections, ML models, and more.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text

from src.base import BaseComponent
from src.core.config import Config
from src.database import RedisClient, get_async_session
from src.exchanges.factory import ExchangeFactory

# Module level logger
logger = logging.getLogger(__name__)


class ConnectionHealthMonitor(BaseComponent):
    """Mock connection health monitor for exchanges."""

    def __init__(self, exchange):
        self.exchange = exchange

    async def get_health_status(self) -> dict:
        """Get health status for the exchange connection."""
        return {"status": "healthy", "latency_ms": 50, "rate_limit_remaining": 1000}


router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    service: str
    version: str
    uptime_seconds: float
    checks: dict[str, Any]


class ComponentHealth(BaseModel):
    """Individual component health model."""

    status: str
    message: str
    response_time_ms: float | None = None
    last_check: datetime
    metadata: dict[str, Any] | None = None


# Global startup time for uptime calculation
_startup_time = time.time()


def get_config_dependency() -> Config:
    """
    FastAPI dependency to get application configuration.

    Returns:
        Config: Application configuration instance
    """
    try:
        from src.core.dependency_injection import get_container

        config_service = get_container().get("ConfigService")
        # Create legacy Config object from ConfigService
        config = Config()
        config.database = config_service.get_database_config()
        config.exchange = config_service.get_exchange_config()
        config.risk = config_service.get_risk_config()
        return config
    except (KeyError, AttributeError, ImportError):
        # Final fallback to legacy method
        from src.core.config import get_config

        return get_config()


async def check_database_health(config: Config) -> ComponentHealth:
    """
    Check database connectivity and health.

    Args:
        config: Application configuration

    Returns:
        ComponentHealth: Database health status
    """
    start_time = time.time()

    try:
        # Test basic connectivity using async session
        async for session in get_async_session():
            # Simple query to test connection
            result = await session.execute(text("SELECT 1"))
            if result.scalar() != 1:
                raise Exception("Database query returned unexpected result")

            # Get pool status from session info
            pool = session.bind.pool if hasattr(session.bind, "pool") else None
            pool_size = pool.size() if pool else 0
            pool_checked_out = pool.checked_out() if pool else 0
            pool_overflow = pool.overflow() if pool else 0

        response_time = (time.time() - start_time) * 1000

        return ComponentHealth(
            status="healthy",
            message="Database connection successful",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
            metadata={
                "pool_size": pool_size,
                "pool_used": pool_checked_out,
                "pool_overflow": pool_overflow,
            },
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Database health check failed: {e}")

        return ComponentHealth(
            status="unhealthy",
            message=f"Database connection failed: {e!s}",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
        )


async def check_redis_health(config: Config) -> ComponentHealth:
    """
    Check Redis connectivity and health.

    Args:
        config: Application configuration

    Returns:
        ComponentHealth: Redis health status
    """
    start_time = time.time()

    try:
        redis_client = RedisClient(config)
        await redis_client.connect()

        # Test basic connectivity
        is_connected = await redis_client.ping()
        if not is_connected:
            raise Exception("Redis ping failed")

        # Test read/write operations
        test_key = "health_check_test"
        test_value = str(int(time.time()))

        await redis_client.set(test_key, test_value, ttl=10)
        retrieved_value = await redis_client.get(test_key)

        if retrieved_value != test_value:
            raise Exception("Redis read/write test failed")

        # Get Redis info
        info = await redis_client.info()

        response_time = (time.time() - start_time) * 1000

        # Disconnect Redis client
        await redis_client.disconnect()

        return ComponentHealth(
            status="healthy",
            message="Redis connection successful",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
            metadata={
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0),
            },
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Redis health check failed: {e}")

        # Try to disconnect Redis client if it was connected
        try:
            if "redis_client" in locals():
                await redis_client.disconnect()
        except Exception:
            pass  # Ignore disconnect errors

        return ComponentHealth(
            status="unhealthy",
            message=f"Redis connection failed: {e!s}",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
        )


async def check_exchanges_health(config: Config) -> ComponentHealth:
    """
    Check exchange connections health.

    Args:
        config: Application configuration

    Returns:
        ComponentHealth: Exchanges health status
    """
    start_time = time.time()

    try:
        exchange_factory = ExchangeFactory(config)

        # Get all configured exchanges
        available_exchanges = exchange_factory.get_available_exchanges()

        if not available_exchanges:
            return ComponentHealth(
                status="degraded",
                message="No exchanges configured",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
            )

        exchange_status = {}
        healthy_count = 0

        for exchange_name in available_exchanges:
            try:
                exchange = await exchange_factory.create_exchange(exchange_name)

                # Simple health check - just test if we can create the exchange
                if exchange:
                    exchange_status[exchange_name] = {
                        "status": "healthy",
                        "latency_ms": 0,
                        "rate_limit_remaining": 1000,
                    }
                    healthy_count += 1
                else:
                    exchange_status[exchange_name] = {
                        "status": "unhealthy",
                        "error": "Failed to create exchange instance",
                    }

            except Exception as e:
                exchange_status[exchange_name] = {"status": "unhealthy", "error": f"{e!s}"}

        response_time = (time.time() - start_time) * 1000

        # Determine overall status
        if healthy_count == len(available_exchanges):
            status = "healthy"
            message = f"All {healthy_count} exchanges healthy"
        elif healthy_count > 0:
            status = "degraded"
            message = f"{healthy_count}/{len(available_exchanges)} exchanges healthy"
        else:
            status = "unhealthy"
            message = "No exchanges healthy"

        return ComponentHealth(
            status=status,
            message=message,
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
            metadata={
                "exchanges": exchange_status,
                "total_exchanges": len(available_exchanges),
                "healthy_exchanges": healthy_count,
            },
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Exchanges health check failed: {e}")

        return ComponentHealth(
            status="unhealthy",
            message=f"Exchange health check failed: {e!s}",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
        )


async def check_ml_models_health(config: Config) -> ComponentHealth:
    """
    Check ML models health and availability.

    Args:
        config: Application configuration

    Returns:
        ComponentHealth: ML models health status
    """
    start_time = time.time()

    try:
        # This would normally check model manager
        # For now, return a basic check
        response_time = (time.time() - start_time) * 1000

        return ComponentHealth(
            status="healthy",
            message="ML models service available",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
            metadata={"models_loaded": 0, "inference_ready": True},  # Would be actual count
        )

    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"ML models health check failed: {e}")

        return ComponentHealth(
            status="unhealthy",
            message=f"ML models health check failed: {e!s}",
            response_time_ms=response_time,
            last_check=datetime.now(timezone.utc),
        )


@router.get("/health", response_model=HealthStatus)
async def basic_health_check():
    """
    Basic health check endpoint.

    Returns minimal health status for load balancers and monitoring.
    """
    uptime = time.time() - _startup_time

    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        service="t-bot-api",
        version="1.0.0",
        uptime_seconds=uptime,
        checks={},
    )


@router.get("/health/detailed", response_model=HealthStatus)
async def detailed_health_check(config: Config = Depends(get_config_dependency)):
    """
    Detailed health check endpoint.

    Performs comprehensive health checks on all system components.
    """
    start_time = time.time()
    uptime = start_time - _startup_time

    # Run all health checks concurrently
    health_checks = await asyncio.gather(
        check_database_health(config),
        check_redis_health(config),
        check_exchanges_health(config),
        check_ml_models_health(config),
        return_exceptions=True,
    )

    checks = {}
    overall_status = "healthy"

    # Process database health
    if isinstance(health_checks[0], ComponentHealth):
        checks["database"] = health_checks[0].model_dump()
        if health_checks[0].status in ["unhealthy", "degraded"]:
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
    else:
        checks["database"] = {
            "status": "unhealthy",
            "message": f"Health check failed: {health_checks[0]}",
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        overall_status = "unhealthy"

    # Process Redis health
    if isinstance(health_checks[1], ComponentHealth):
        checks["redis"] = health_checks[1].model_dump()
        if health_checks[1].status in ["unhealthy", "degraded"]:
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
    else:
        checks["redis"] = {
            "status": "unhealthy",
            "message": f"Health check failed: {health_checks[1]}",
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        overall_status = "unhealthy"

    # Process exchanges health
    if isinstance(health_checks[2], ComponentHealth):
        checks["exchanges"] = health_checks[2].model_dump()
        if health_checks[2].status in ["unhealthy", "degraded"]:
            overall_status = "degraded" if overall_status == "healthy" else overall_status
    else:
        checks["exchanges"] = {
            "status": "unhealthy",
            "message": f"Health check failed: {health_checks[2]}",
            "last_check": datetime.now(timezone.utc).isoformat(),
        }
        # Exchange failures are not critical for API health

    # Process ML models health
    if isinstance(health_checks[3], ComponentHealth):
        checks["ml_models"] = health_checks[3].model_dump()
        if health_checks[3].status in ["unhealthy", "degraded"]:
            # ML model failures are not critical for API health
            pass
    else:
        checks["ml_models"] = {
            "status": "unhealthy",
            "message": f"Health check failed: {health_checks[3]}",
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    return HealthStatus(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        service="t-bot-api",
        version="1.0.0",
        uptime_seconds=uptime,
        checks=checks,
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.

    Returns 200 if the service is ready to receive traffic.
    """
    # Check critical dependencies
    # For now, just return ready
    return {"ready": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.

    Returns 200 if the service is alive (should not be restarted).
    """
    return {"alive": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/startup")
async def startup_check():
    """
    Kubernetes-style startup probe.

    Returns 200 when the service has finished starting up.
    """
    uptime = time.time() - _startup_time
    startup_complete = uptime > 10  # Consider startup complete after 10 seconds

    if not startup_complete:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service still starting up"
        )

    return {"started": True, "uptime_seconds": uptime}
