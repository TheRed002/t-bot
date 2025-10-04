"""
Dual-Mode Test Configuration for Integration Tests

Supports both real Docker services and mock services for environments
where Docker is not available.
"""

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncGenerator

import pytest

from src.core.config import get_config
from tests.integration.infrastructure.mock_services import (
    MockDatabaseService,
    MockInfluxDBClient,
    MockRedisClient,
    setup_mock_infrastructure,
    teardown_mock_infrastructure,
)

logger = logging.getLogger(__name__)

# Check if we should use mock services
USE_MOCK_SERVICES = os.getenv("USE_MOCK_SERVICES", "true").lower() == "true"


@pytest.fixture
async def test_services() -> AsyncGenerator[dict, None]:
    """
    Provides test services (real or mock based on environment).

    Returns a dictionary with database, redis, and influxdb services.
    """
    if USE_MOCK_SERVICES:
        logger.info("Using mock services for integration testing")
        services = await setup_mock_infrastructure()

        yield services

        await teardown_mock_infrastructure(services)
    else:
        logger.info("Using real Docker services for integration testing")
        # Import real service setup
        from tests.integration.infrastructure.conftest import clean_database

        # This would use the real Docker services
        # For now, we'll raise an error if Docker is required but not available
        raise RuntimeError(
            "Real Docker services requested but not available. "
            "Set USE_MOCK_SERVICES=true to use mock services."
        )


@pytest.fixture
async def database_service(test_services):
    """Provides database service (real or mock)."""
    return test_services["database"]


@pytest.fixture
async def redis_client(test_services):
    """Provides Redis client (real or mock)."""
    return test_services["redis"]


@pytest.fixture
async def influxdb_client(test_services):
    """Provides InfluxDB client (real or mock)."""
    return test_services["influxdb"]


@pytest.fixture
async def db_session(database_service):
    """Provides a database session."""
    async with database_service.get_session() as session:
        yield session
        await session.rollback()


@pytest.fixture(autouse=True)
async def cleanup_between_tests(test_services):
    """Clean up data between tests."""
    yield

    if USE_MOCK_SERVICES:
        # Clean mock data
        redis = test_services["redis"]
        await redis.flushdb()

        # Clear InfluxDB data
        influx = test_services["influxdb"]
        influx.data.clear()

        # Clear database (recreate tables)
        db = test_services["database"]
        if db._started:
            from src.database.models.base import Base
            async with db.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)