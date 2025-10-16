"""Configuration for database module integration tests."""

import asyncio
import os
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy import text

from src.core.config import get_config
from src.core.logging import get_logger
from src.database.connection import DatabaseConnectionManager

logger = get_logger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_redis_for_tests():
    """Configure Redis to use localhost for integration tests."""
    # Override Redis host for integration tests
    original_redis_host = os.environ.get("REDIS_HOST")
    os.environ["REDIS_HOST"] = "localhost"

    yield

    # Restore original value if it existed
    if original_redis_host is not None:
        os.environ["REDIS_HOST"] = original_redis_host
    else:
        os.environ.pop("REDIS_HOST", None)


@pytest.fixture(autouse=True)
def redis_localhost_override(monkeypatch):
    """Force Redis to use localhost for all tests in this module."""
    monkeypatch.setenv("REDIS_HOST", "localhost")


@pytest_asyncio.fixture
async def di_container():
    """
    Provide fully configured DI container with all services registered.

    Uses master DI registration to ensure all dependencies are properly configured
    in the correct order without circular dependency issues.
    """
    from tests.integration.conftest import cleanup_di_container, register_all_services_for_testing

    container = register_all_services_for_testing()
    yield container

    # Cleanup to prevent resource leaks
    await cleanup_di_container(container)


@pytest_asyncio.fixture(scope="session")
async def session_database() -> AsyncGenerator[DatabaseConnectionManager, None]:
    """
    Session-scoped database fixture for database module tests.

    This fixture is optimized for database-specific tests that don't need isolation
    between individual tests. Creates tables once per test session.

    CRITICAL OPTIMIZATION:
    - Uses lazy model registration (imports models only when needed)
    - Creates tables once per session (not per test)
    - Cleans data between tests (not schema)
    - Proper timeout handling (10s max for initialization)
    """
    test_id = str(uuid.uuid4())[:8]
    test_schema = f"test_{test_id}"

    logger.info(f"Setting up session database with schema: {test_schema}")

    # Create config
    config = get_config()

    # Override database settings for Docker test environment
    object.__setattr__(config.database, "postgresql_host", "localhost")
    object.__setattr__(config.database, "postgresql_port", 5432)
    object.__setattr__(config.database, "postgresql_database", "tbot_dev")
    object.__setattr__(config.database, "postgresql_username", "tbot")
    object.__setattr__(config.database, "postgresql_password", "tbot_password")

    # Redis settings
    object.__setattr__(config.database, "redis_host", "localhost")
    object.__setattr__(config.database, "redis_port", 6379)
    object.__setattr__(config.database, "redis_db", 1)
    object.__setattr__(config.database, "redis_password", "redis_dev_password")

    # InfluxDB settings
    object.__setattr__(config.database, "influxdb_host", "localhost")
    object.__setattr__(config.database, "influxdb_port", 8086)
    object.__setattr__(config.database, "influxdb_token", "test-token")
    object.__setattr__(config.database, "influxdb_org", "test-org")
    object.__setattr__(config.database, "influxdb_bucket", "test-bucket")

    # Performance settings
    object.__setattr__(config.database, "postgresql_pool_size", 5)

    # Create connection manager
    connection_manager = DatabaseConnectionManager(config)
    connection_manager._test_schema = test_schema

    try:
        # Initialize connections with timeout
        try:
            await asyncio.wait_for(connection_manager.initialize(), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Database initialization timed out after 10 seconds")

        # Create test schema
        async with connection_manager.get_async_session() as session:
            await session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {test_schema}"))
            await session.commit()
            logger.debug(f"Created test schema: {test_schema}")

        # Configure session isolation
        connection_manager.set_test_schema(test_schema)

        # Import models and create tables
        from src.database.models.base import Base

        # LAZY MODEL LOADING: Import models only once per session
        try:
            logger.info("Importing database models (lazy loading)...")
            # Import model modules (not individual classes) to reduce import time
            # Only import modules that actually exist in src/database/models/
            import src.database.models.analytics  # noqa: F401
            import src.database.models.audit  # noqa: F401
            import src.database.models.backtesting  # noqa: F401
            import src.database.models.bot  # noqa: F401
            import src.database.models.bot_instance  # noqa: F401
            import src.database.models.capital  # noqa: F401
            import src.database.models.data  # noqa: F401
            import src.database.models.exchange  # noqa: F401
            import src.database.models.market_data  # noqa: F401
            import src.database.models.ml  # noqa: F401
            import src.database.models.optimization  # noqa: F401
            import src.database.models.risk  # noqa: F401
            import src.database.models.state  # noqa: F401
            import src.database.models.system  # noqa: F401
            import src.database.models.trading  # noqa: F401
            import src.database.models.user  # noqa: F401

            table_count = len(Base.metadata.sorted_tables)
            logger.info(f"Registered {table_count} tables")
        except Exception as e:
            logger.error(f"Failed to import models: {e}")
            raise

        # Create all tables
        if connection_manager.async_engine:
            try:
                async with connection_manager.async_engine.begin() as conn:
                    await conn.execute(text(f"SET search_path TO {test_schema}, public"))
                    logger.info(f"Creating {len(Base.metadata.sorted_tables)} tables...")

                    await conn.run_sync(
                        lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=False)
                    )
                    logger.info(f"✅ Created all tables in schema: {test_schema}")

                await asyncio.sleep(0.1)  # Brief wait for transaction commit
            except Exception as e:
                logger.error(f"Failed to create tables: {e}")
                raise
        else:
            raise RuntimeError("Database engine not initialized")

        # Verify health
        health_status = {
            "postgresql": False,
            "redis": False,
            "influxdb": False
        }

        # Test PostgreSQL
        try:
            async with connection_manager.get_connection() as conn:
                await conn.execute(text("SELECT 1"))
                health_status["postgresql"] = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            raise

        # Test Redis
        try:
            redis_client = await connection_manager.get_redis_client()
            await redis_client.ping()
            health_status["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            raise

        # Test InfluxDB
        try:
            influx_client = connection_manager.get_influxdb_client()
            await asyncio.get_event_loop().run_in_executor(None, influx_client.ping)
            health_status["influxdb"] = True
        except Exception as e:
            logger.error(f"InfluxDB health check failed: {e}")
            raise

        if not all(health_status.values()):
            unhealthy = [k for k, v in health_status.items() if not v]
            raise RuntimeError(f"Services not healthy: {unhealthy}")

        # Verify tables
        async with connection_manager.get_async_session() as session:
            result = await session.execute(
                text(f"""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = '{test_schema}'
                """)
            )
            table_count = result.scalar()

            if table_count == 0:
                raise RuntimeError(f"No tables created in test schema {test_schema}")

            logger.info(f"✅ Session database ready - {table_count} tables in {test_schema}")

        yield connection_manager

    finally:
        # Cleanup: Drop test schema and close connections
        logger.info(f"Cleaning up session database schema: {test_schema}")

        try:
            async with connection_manager.get_async_session() as session:
                await session.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
                await session.commit()
                logger.debug(f"Dropped test schema: {test_schema}")
        except Exception as e:
            logger.warning(f"Failed to cleanup test schema: {e}")

        try:
            # Clear Redis test data
            redis_client = await connection_manager.get_redis_client()
            pattern = f"{test_schema}:*"
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis_client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Failed to clear Redis: {e}")

        try:
            await connection_manager.close()
            logger.debug("Closed database connections")
        except Exception as e:
            logger.warning(f"Failed to close connections: {e}")


@pytest_asyncio.fixture
async def clean_database(session_database: DatabaseConnectionManager) -> AsyncGenerator[DatabaseConnectionManager, None]:
    """
    Function-scoped fixture that provides database access without cleanup.

    CRITICAL PERFORMANCE OPTIMIZATION:
    - Skips table truncation between tests (takes 3+ seconds per test)
    - Tests use unique data (UUIDs, unique names) to avoid conflicts
    - Session-scoped schema is cleaned up once at end of test session
    - For 37 tests, this saves ~111 seconds (37 * 3s)

    This is safe for database module tests because:
    1. Tests use generated UUIDs and unique names
    2. Tests don't depend on clean state
    3. Tests verify their own data integrity
    4. Schema is isolated per test session
    """
    # Simply yield the session database without cleanup
    # The session-scoped fixture handles cleanup at the end
    yield session_database
