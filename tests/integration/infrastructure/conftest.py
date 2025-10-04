"""
Real Service Infrastructure Test Configuration

Provides fixtures for real PostgreSQL, Redis, and InfluxDB connections
for infrastructure integration testing.

CRITICAL: These fixtures use actual Docker services, not mocks.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator

import pytest

import pytest_asyncio
from src.core.caching.cache_manager import CacheManager
from src.core.config import get_config
from src.database.connection import DatabaseConnectionManager
from src.database.service import DatabaseService
from tests.integration.infrastructure.service_factory import (
    ProductionReadyTestBase,
    RealServiceFactory,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def event_loop():
    """Create an event loop for the test function."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    yield loop

    # Clean up
    try:
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # Wait for tasks to be cancelled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception as e:
        logger.warning(f"Error during event loop cleanup: {e}")
    finally:
        try:
            loop.close()
        except Exception as e:
            logger.warning(f"Error closing event loop: {e}")


@pytest_asyncio.fixture
async def real_test_config():
    """Real configuration for infrastructure tests using Docker services."""
    config = get_config()

    # Override database settings for Docker test environment
    # Use actual Docker container credentials
    object.__setattr__(config.database, "postgresql_host", "localhost")
    object.__setattr__(config.database, "postgresql_port", 5432)
    object.__setattr__(config.database, "postgresql_database", "tbot_dev")
    object.__setattr__(config.database, "postgresql_username", "tbot")
    object.__setattr__(config.database, "postgresql_password", "tbot_password")

    # Redis settings for Docker test environment
    object.__setattr__(config.database, "redis_host", "localhost")
    object.__setattr__(config.database, "redis_port", 6379)
    object.__setattr__(config.database, "redis_db", 1)  # Use DB 1 for tests
    object.__setattr__(config.database, "redis_password", "redis_dev_password")  # Development Redis password

    # InfluxDB settings for Docker test environment
    object.__setattr__(config.database, "influxdb_host", "localhost")
    object.__setattr__(config.database, "influxdb_port", 8086)
    object.__setattr__(config.database, "influxdb_token", "test-token")
    object.__setattr__(config.database, "influxdb_org", "test-org")
    object.__setattr__(config.database, "influxdb_bucket", "test-bucket")

    # Performance settings for testing
    object.__setattr__(config.database, "postgresql_pool_size", 5)

    logger.info("Real test configuration created with Docker service endpoints")
    return config


@pytest_asyncio.fixture
async def clean_database() -> AsyncGenerator[DatabaseConnectionManager, None]:
    """
    Provides a clean database connection manager for each test.

    This fixture:
    1. Creates a new database connection manager
    2. Initializes connections to real PostgreSQL, Redis, InfluxDB
    3. Creates isolated test environment
    4. Cleans up after test completion
    """
    # Create unique database schema/namespace for this test
    test_id = str(uuid.uuid4())[:8]
    test_schema = f"test_{test_id}"

    logger.info(f"Setting up clean database environment with schema: {test_schema}")

    # Create config for this test function
    config = get_config()

    # Override database settings for Docker test environment
    # Use actual Docker container credentials
    object.__setattr__(config.database, "postgresql_host", "localhost")
    object.__setattr__(config.database, "postgresql_port", 5432)
    object.__setattr__(config.database, "postgresql_database", "tbot_dev")
    object.__setattr__(config.database, "postgresql_username", "tbot")
    object.__setattr__(config.database, "postgresql_password", "tbot_password")

    # Redis settings for Docker test environment
    object.__setattr__(config.database, "redis_host", "localhost")
    object.__setattr__(config.database, "redis_port", 6379)
    object.__setattr__(config.database, "redis_db", 1)  # Use DB 1 for tests
    object.__setattr__(config.database, "redis_password", "redis_dev_password")  # Development Redis password

    # InfluxDB settings for Docker test environment
    object.__setattr__(config.database, "influxdb_host", "localhost")
    object.__setattr__(config.database, "influxdb_port", 8086)
    object.__setattr__(config.database, "influxdb_token", "test-token")
    object.__setattr__(config.database, "influxdb_org", "test-org")
    object.__setattr__(config.database, "influxdb_bucket", "test-bucket")

    # Performance settings for testing
    object.__setattr__(config.database, "postgresql_pool_size", 5)

    # Create database connection manager
    connection_manager = DatabaseConnectionManager(config)

    # Store test schema for cleanup and configure session isolation
    connection_manager._test_schema = test_schema
    connection_manager.set_test_schema(test_schema)

    try:
        # Initialize all database connections
        await connection_manager.initialize()

        # Create test schema for isolation
        from sqlalchemy import text
        async with connection_manager.get_async_session() as session:
            await session.execute(text(f"CREATE SCHEMA IF NOT EXISTS {test_schema}"))
            await session.commit()
            logger.debug(f"Created test schema: {test_schema}")

        # Set search path to use test schema
        async with connection_manager.get_async_session() as session:
            await session.execute(text(f"SET search_path TO {test_schema}, public"))
            await session.commit()

        # Create all tables from models in the test schema
        from src.database.models.base import Base

        # Create all tables using the connection with proper error handling
        if connection_manager.async_engine:
            try:
                # First attempt: try creating tables with checkfirst
                async with connection_manager.async_engine.begin() as conn:
                    await conn.execute(text(f"SET search_path TO {test_schema}, public"))
                    await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True))
                    logger.info(f"Created all database tables in schema: {test_schema}")
            except Exception as e:
                # If we get conflicts, rollback and retry with fresh schema
                if "already exists" in str(e):
                    logger.warning(f"Objects already exist in {test_schema}, recreating schema")
                    try:
                        # Use a separate connection for schema recreation
                        async with connection_manager.async_engine.connect() as conn:
                            # Drop and recreate schema outside transaction
                            await conn.execute(text(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE"))
                            await conn.execute(text(f"CREATE SCHEMA {test_schema}"))
                            await conn.commit()  # Commit schema changes

                        # Now create tables in a fresh transaction
                        async with connection_manager.async_engine.begin() as conn:
                            await conn.execute(text(f"SET search_path TO {test_schema}, public"))
                            await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn))
                            logger.info(f"Recreated all database tables in schema: {test_schema}")
                    except Exception as recreate_error:
                        logger.error(f"Failed to recreate schema {test_schema}: {recreate_error}")
                        raise
                else:
                    logger.error(f"Database table creation failed: {e}")
                    raise
        else:
            raise RuntimeError("Database engine not initialized")

        # Verify all services are healthy
        health_status = {
            "postgresql": False,
            "redis": False,
            "influxdb": False
        }

        # Test PostgreSQL
        try:
            async with connection_manager.get_connection() as conn:
                result = await conn.execute(text("SELECT 1"))
                health_status["postgresql"] = True
                logger.debug("PostgreSQL connection verified")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise

        # Test Redis
        try:
            redis_client = await connection_manager.get_redis_client()
            await redis_client.ping()
            health_status["redis"] = True
            logger.debug("Redis connection verified")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

        # Test InfluxDB
        try:
            influx_client = connection_manager.get_influxdb_client()
            # Run ping in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(None, influx_client.ping)
            health_status["influxdb"] = True
            logger.debug("InfluxDB connection verified")
        except Exception as e:
            logger.error(f"InfluxDB connection failed: {e}")
            raise

        # Verify all services are healthy
        if not all(health_status.values()):
            unhealthy = [k for k, v in health_status.items() if not v]
            raise RuntimeError(f"Services not healthy: {unhealthy}")

        # Verify tables were created
        async with connection_manager.get_async_session() as session:
            result = await session.execute(
                text(f"""
                    SELECT COUNT(*)
                    FROM information_schema.tables
                    WHERE table_schema = '{test_schema}'
                """)
            )
            table_count = result.scalar()
            logger.info(f"Created {table_count} tables in test schema")

            if table_count == 0:
                raise RuntimeError(f"No tables created in test schema {test_schema}")

        logger.info(f"✅ Clean database environment ready - schema: {test_schema} with {table_count} tables")

        yield connection_manager

    finally:
        # Cleanup: Drop test schema and close connections
        logger.info(f"Cleaning up test schema: {test_schema}")

        try:
            # Clean up test schema - try both the stored attribute and the local variable
            schema_to_clean = getattr(connection_manager, "_test_schema", test_schema)
            from sqlalchemy import text
            async with connection_manager.get_async_session() as session:
                await session.execute(text(f"DROP SCHEMA IF EXISTS {schema_to_clean} CASCADE"))
                await session.commit()
                logger.debug(f"Dropped test schema: {schema_to_clean}")
        except Exception as e:
            logger.warning(f"Failed to cleanup test schema {test_schema}: {e}")

        try:
            # Clear Redis test data using key pattern to avoid affecting concurrent tests
            redis_client = await connection_manager.get_redis_client()
            # Use test_schema as prefix for Redis keys to isolate them
            schema_to_clean = getattr(connection_manager, "_test_schema", test_schema)
            pattern = f"{schema_to_clean}:*"
            cursor = 0
            while True:
                cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis_client.delete(*keys)
                if cursor == 0:
                    break
                logger.debug(f"Cleared Redis test data for pattern: {pattern}")
            else:
                # Fallback to flush if no test schema
                await redis_client.flushdb()
                logger.debug("Cleared Redis test data")
        except Exception as e:
            logger.warning(f"Failed to clear Redis test data: {e}")

        try:
            # Close all connections
            await connection_manager.close()
            logger.debug("Database connections closed")
        except Exception as e:
            logger.warning(f"Error closing database connections: {e}")

        # Give a moment for tasks to finish naturally
        await asyncio.sleep(0.1)

        # Cancel remaining tasks
        try:
            current_task = asyncio.current_task()
            pending = [task for task in asyncio.all_tasks() if not task.done() and task is not current_task]
            if pending:
                logger.debug(f"Cancelling {len(pending)} remaining tasks")
                for task in pending:
                    task.cancel()
                # Wait for cancellations with timeout
                await asyncio.wait(pending, timeout=2.0)
        except Exception as e:
            logger.warning(f"Error cancelling tasks: {e}")


@pytest_asyncio.fixture
async def real_cache_manager(clean_database) -> AsyncGenerator[CacheManager, None]:
    """
    Provides a real CacheManager connected to Redis for testing.

    This fixture creates a CacheManager instance that uses the real Redis
    connection from the clean_database fixture.
    """
    logger.info("Setting up real cache manager")

    cache_manager = CacheManager()

    try:
        # Verify cache connection
        await cache_manager.health_check()
        logger.info("✅ Real cache manager ready")

        yield cache_manager

    finally:
        try:
            # Cleanup
            if hasattr(cache_manager, "shutdown"):
                await cache_manager.shutdown()
            logger.debug("Cache manager cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up cache manager: {e}")


@pytest_asyncio.fixture
async def real_database_service(clean_database) -> AsyncGenerator[DatabaseService, None]:
    """
    Provides a real DatabaseService connected to PostgreSQL for testing.

    This fixture creates a DatabaseService instance that uses the real
    PostgreSQL connection from the clean_database fixture.
    """
    logger.info("Setting up real database service")

    database_service = DatabaseService(
        connection_manager=clean_database,
        config_service=None,  # Optional for testing
        validation_service=None,  # Optional for testing
    )

    try:
        # Start the database service
        await database_service.start()

        # Verify service health
        await database_service.health_check()
        logger.info("✅ Real database service ready")

        yield database_service

    finally:
        try:
            # Cleanup
            if hasattr(database_service, "shutdown"):
                await database_service.shutdown()
            logger.debug("Database service cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up database service: {e}")


@pytest_asyncio.fixture
async def real_service_factory(clean_database) -> AsyncGenerator[RealServiceFactory, None]:
    """
    Provides a pre-configured RealServiceFactory for integration tests.

    This fixture creates a service factory that can instantiate all
    real services with proper dependency chains.
    """
    logger.info("Setting up real service factory")

    factory = RealServiceFactory()

    try:
        # Initialize with the clean database
        await factory.initialize_core_services(clean_database)
        logger.info("✅ Real service factory ready")

        yield factory

    finally:
        try:
            # Cleanup all services
            await factory.cleanup()
            logger.debug("Service factory cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up service factory: {e}")


@pytest_asyncio.fixture
async def production_ready_test_base(clean_database) -> AsyncGenerator[ProductionReadyTestBase, None]:
    """
    Provides a ProductionReadyTestBase instance for infrastructure tests.

    This fixture creates a test base class instance with all real services
    properly initialized and ready for testing.
    """
    logger.info("Setting up production-ready test base")

    test_base = ProductionReadyTestBase()

    try:
        # Setup real services using the clean database
        container = await test_base.setup_real_services(clean_database)
        logger.info("✅ Production-ready test base ready")

        yield test_base

    finally:
        try:
            # Cleanup all services
            await test_base.cleanup_real_services()
            logger.debug("Test base cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up test base: {e}")


@pytest_asyncio.fixture
async def real_container(real_service_factory):
    """
    Provides a fully configured DependencyContainer with real services.

    This fixture creates a container with all real services registered
    and ready for dependency injection testing.
    """
    logger.info("Setting up real dependency container")

    container = await real_service_factory.create_dependency_container()
    await real_service_factory.create_advanced_services(container)

    logger.info("✅ Real dependency container ready")

    yield container

    # Cleanup handled by service factory fixture
    logger.debug("Container fixture completed")


# Health check fixtures for verifying service states
@pytest_asyncio.fixture
async def verify_services_healthy(clean_database, real_cache_manager, real_database_service):
    """
    Verification fixture that ensures all services are healthy before tests run.

    This fixture runs health checks on all services and fails fast if any
    service is not responding properly.
    """
    logger.info("Verifying all services are healthy")

    health_checks = {}

    # Check database service
    try:
        await real_database_service.health_check()
        health_checks["database_service"] = True
        logger.debug("Database service health check passed")
    except Exception as e:
        health_checks["database_service"] = False
        logger.error(f"Database service health check failed: {e}")

    # Check cache manager
    try:
        await real_cache_manager.health_check()
        health_checks["cache_manager"] = True
        logger.debug("Cache manager health check passed")
    except Exception as e:
        health_checks["cache_manager"] = False
        logger.error(f"Cache manager health check failed: {e}")

    # Check raw database connections
    try:
        async with clean_database.get_connection() as conn:
            await conn.execute("SELECT 1")
        health_checks["postgresql"] = True
        logger.debug("PostgreSQL health check passed")
    except Exception as e:
        health_checks["postgresql"] = False
        logger.error(f"PostgreSQL health check failed: {e}")

    try:
        redis_client = await clean_database.get_redis_client()
        await redis_client.ping()
        health_checks["redis"] = True
        logger.debug("Redis health check passed")
    except Exception as e:
        health_checks["redis"] = False
        logger.error(f"Redis health check failed: {e}")

    try:
        influx_client = clean_database.get_influxdb_client()
        await asyncio.get_event_loop().run_in_executor(None, influx_client.ping)
        health_checks["influxdb"] = True
        logger.debug("InfluxDB health check passed")
    except Exception as e:
        health_checks["influxdb"] = False
        logger.error(f"InfluxDB health check failed: {e}")

    # Fail fast if any service is unhealthy
    failed_services = [service for service, healthy in health_checks.items() if not healthy]
    if failed_services:
        raise RuntimeError(f"Service health verification failed for: {failed_services}")

    logger.info("✅ All services verified healthy")

    yield health_checks


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """
    Simple performance monitoring for infrastructure tests.
    """
    import time

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = {}

        def start(self):
            self.start_time = time.time()
            self.metrics = {
                "operations": 0,
                "errors": 0,
                "latencies": []
            }

        def record_operation(self, latency=None):
            self.metrics["operations"] += 1
            if latency:
                self.metrics["latencies"].append(latency)

        def record_error(self):
            self.metrics["errors"] += 1

        def get_summary(self):
            duration = time.time() - (self.start_time or time.time())
            avg_latency = sum(self.metrics["latencies"]) / len(self.metrics["latencies"]) if self.metrics["latencies"] else 0

            return {
                "duration": duration,
                "operations": self.metrics["operations"],
                "errors": self.metrics["errors"],
                "error_rate": self.metrics["errors"] / max(1, self.metrics["operations"]),
                "avg_latency": avg_latency,
                "ops_per_second": self.metrics["operations"] / max(0.001, duration)
            }

    monitor = PerformanceMonitor()
    monitor.start()

    yield monitor


# Pytest configuration for infrastructure tests
@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for infrastructure tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Reduce noise from external libraries
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logger.info("Infrastructure integration test logging configured")
