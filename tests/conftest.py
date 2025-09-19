"""
Pytest configuration for the trading bot test suite.

This module provides fixtures and configuration for all test types:
- Unit tests: No external dependencies
- Integration tests: Require temporary databases
- Performance tests: Measure system performance
"""

import os
import subprocess

# Set testing environment variables before importing anything else
os.environ["TESTING"] = "1"
os.environ["DISABLE_ERROR_HANDLER_LOGGING"] = "true"

import asyncio
import gc
import sys
import uuid
from typing import Any, Dict
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.core.config import Config
from src.core.exceptions import DataSourceError
from src.database.connection import close_database, initialize_database


class TestConfig:
    """Test configuration helper class."""

    def __init__(self, **kwargs):
        """Initialize test config with optional overrides."""
        self.config = Config()

        # Apply any override values
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.database, key):
                setattr(self.config.database, key, value)
            elif hasattr(self.config.exchange, key):
                setattr(self.config.exchange, key, value)

    def get_config(self) -> Config:
        """Get the underlying config object."""
        return self.config


# Enhanced Test Isolation Framework
def clear_prometheus_metrics():
    """Clear Prometheus metrics registry."""
    try:
        import prometheus_client
        prometheus_client.REGISTRY._collector_to_names.clear()
        prometheus_client.REGISTRY._names_to_collectors.clear()
    except (ImportError, AttributeError):
        pass


def clear_singleton_registry():
    """Enhanced singleton cleanup covering all identified classes."""
    singleton_classes = [
        # Core Infrastructure
        'src.core.dependency_injection.DependencyInjector',
        'src.core.dependency_injection.DependencyContainer',
        'src.database.connection.DatabaseManager',
        'src.database.connection.DatabaseConnection',
        'src.core.caching.cache_manager.CacheManager',
        # Monitoring & Metrics
        'src.monitoring.services.MetricsCollector',
        'src.monitoring.alerting.AlertManager',
        'src.monitoring.performance_monitor.PerformanceMonitor',
        'src.analytics.services.dashboard_service.DashboardService',
        # Data Services
        'src.data.streaming.streaming_service.StreamingService',
        'src.data.services.data_service.DataService',
        'src.data.monitoring.data_monitoring_service.DataMonitoringService',
        # Trading Components
        'src.execution.order_manager.OrderManager',
        'src.risk_management.core.calculator.RiskCalculator',
        'src.strategies.performance_monitor.PerformanceMonitor',
        # State Management
        'src.state.state_manager.StateManager',
        'src.state.state_persistence.StatePersistence',
        'src.state.state_service.StateService',
        # Bot Management
        'src.bot_management.bot_coordinator.BotCoordinator',
        'src.bot_management.resource_manager.ResourceManager',
        'src.bot_management.factory.BotFactory',
        # Error Handling
        'src.error_handling.decorators.HandlerPool',
        'src.error_handling.base.ErrorHandler',
        # Web Interface
        'src.web_interface.socketio_manager.SocketIOManager',
        'src.web_interface.facade.service_registry.ServiceRegistry',
        # Validation & Security
        'src.utils.validation.core.ValidationFramework',
        'src.web_interface.security.auth.SecuritySanitizer',
        # Capital & Optimization
        'src.capital_management.service.CapitalManagementService',
        'src.optimization.service.OptimizationService',
        'src.backtesting.service.BacktestService',
    ]

    for class_path in singleton_classes:
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            if module_path in sys.modules:
                module = sys.modules[module_path]
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    # Clear all possible singleton patterns
                    for attr in ['_instance', '_instances', '_registry', '_cache',
                               '_handlers', '_observers', '_services', '_connections']:
                        if hasattr(cls, attr):
                            attr_value = getattr(cls, attr)
                            if isinstance(attr_value, dict):
                                attr_value.clear()
                            elif isinstance(attr_value, list):
                                attr_value.clear()
                            elif isinstance(attr_value, set):
                                attr_value.clear()
                            else:
                                setattr(cls, attr, None)
        except Exception:
            pass


def cleanup_event_loops():
    """Clean up event loops and async tasks."""
    try:
        loop = asyncio.get_running_loop()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            try:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass
    except RuntimeError:
        pass


def nuclear_module_reset():
    """Nuclear option: completely remove and reimport contaminated modules."""
    critical_modules = [
        'src.core.types', 'src.core.types.bot', 'src.core.types.risk',
        'src.core.types.strategy', 'src.core.types.position', 'src.core.types.order',
        'src.core.types.signal', 'src.core.types.base', 'src.backtesting.engine',
        'src.monitoring.alerting', 'src.monitoring.telemetry', 'src.strategies.base',
    ]

    contaminated_modules = set()
    for module_name in critical_modules:
        if module_name not in sys.modules:
            continue
        try:
            module = sys.modules[module_name]
            is_contaminated = False
            # Check for Mock contamination
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(module, attr_name, None)
                if attr is None:
                    continue
                # Check if it's a Mock object
                if hasattr(attr, '_mock_name') or 'Mock' in str(type(attr)):
                    is_contaminated = True
                    break
                # Check enum contamination
                if hasattr(attr, '__members__'):  # It's an enum
                    for enum_value in attr:
                        if hasattr(enum_value, '_mock_name') or 'Mock' in str(type(enum_value)):
                            is_contaminated = True
                            break
                    if is_contaminated:
                        break
            if is_contaminated:
                contaminated_modules.add(module_name)
        except Exception:
            contaminated_modules.add(module_name)

    # Nuclear removal of contaminated modules
    for module_name in contaminated_modules:
        modules_to_remove = [key for key in sys.modules.keys()
                           if key == module_name or key.startswith(module_name + '.')]
        for mod_name in modules_to_remove:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

    # Force reimport critical modules
    for module_name in ['src.core.types', 'src.backtesting.engine', 'src.monitoring.telemetry']:
        if module_name not in sys.modules:
            try:
                __import__(module_name)
            except ImportError:
                pass


def comprehensive_mock_cleanup():
    """Enhanced mock cleanup targeting all mock persistence issues."""
    # Stop all active patches
    mock.patch.stopall()

    # Clear unittest.mock internal state
    try:
        from unittest.mock import _patch_object
        if hasattr(_patch_object, '_active_patches'):
            active_patches = list(_patch_object._active_patches)
            for patch in active_patches:
                try:
                    patch.stop()
                except Exception:
                    pass
            _patch_object._active_patches.clear()
    except Exception:
        pass

    # Scan loaded modules for Mock contamination
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith('src.') or module is None:
            continue
        try:
            # Check module-level attributes
            for attr_name in dir(module):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(module, attr_name, None)
                if attr and (hasattr(attr, '_mock_name') or 'Mock' in str(type(attr))):
                    try:
                        delattr(module, attr_name)
                    except Exception:
                        pass
            # Check class attributes for Mock contamination
            for attr_name in dir(module):
                attr = getattr(module, attr_name, None)
                if attr and hasattr(attr, '__dict__') and hasattr(attr, '__module__'):
                    try:
                        for class_attr_name in list(attr.__dict__.keys()):
                            class_attr = getattr(attr, class_attr_name, None)
                            if class_attr and (hasattr(class_attr, '_mock_name') or
                                             'Mock' in str(type(class_attr))):
                                try:
                                    delattr(attr, class_attr_name)
                                except Exception:
                                    pass
                    except Exception:
                        pass
        except Exception:
            pass


@pytest.fixture(autouse=True, scope="function")
def test_isolation():
    """Enhanced test isolation with surgical cleanup."""
    # PRE-TEST CLEANUP
    clear_singleton_registry()
    comprehensive_mock_cleanup()
    clear_prometheus_metrics()

    yield

    # POST-TEST CLEANUP
    try:
        # 1. Comprehensive mock cleanup
        comprehensive_mock_cleanup()
        # 2. Event loop cleanup
        cleanup_event_loops()
        # 3. Singleton cleanup
        clear_singleton_registry()
        # 4. Nuclear module reset for persistent contamination
        nuclear_module_reset()
        # 5. Clear metrics
        clear_prometheus_metrics()
        # 6. Force garbage collection
        gc.collect()
    except Exception as e:
        # Non-fatal cleanup errors
        print(f"Test cleanup warning: {e}")


@pytest.fixture(scope="function")
def isolated_event_loop():
    """Provide completely isolated event loop."""
    # Create fresh loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    # Cleanup
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


class CleanMockFactory:
    """Factory for creating proper, isolated mocks."""

    @staticmethod
    def create_async_mock(name: str, return_value: Any = None, **kwargs) -> AsyncMock:
        """Create AsyncMock with proper attributes."""
        mock = AsyncMock(name=name, **kwargs)
        mock.__name__ = name
        mock.__qualname__ = name

        if return_value is not None:
            mock.return_value = return_value

        return mock

    @staticmethod
    def create_bot_instance_mocks():
        """Create clean bot instance mocks."""

        async def mock_get_bot_summary():
            return {
                "bot_id": "test_bot",
                "status": "running",
                "uptime": 100,
                "last_heartbeat": "2023-01-01T00:00:00Z"
            }

        async def mock_get_heartbeat():
            return {
                "status": "healthy",
                "timestamp": "2023-01-01T00:00:00Z",
                "memory_usage": 50.0
            }

        return {
            'get_bot_summary': mock_get_bot_summary,
            'get_heartbeat': mock_get_heartbeat,
        }


@pytest.fixture
def clean_mock_factory():
    """Provide clean mock factory for tests."""
    return CleanMockFactory()


@pytest.fixture(scope="session")
def config():
    """Provide application configuration for tests."""
    from src.core.config import DatabaseConfig

    # Create DatabaseConfig separately
    db_config = DatabaseConfig(
        postgresql_host="localhost",
        postgresql_port=5432,
        postgresql_database="trading_bot_test",
        postgresql_username="trading_bot_test",
        postgresql_password="test_password",
        redis_host="localhost",
        redis_port=6379,
        redis_db=1,  # Use different DB to avoid conflicts
        redis_password="redis_password",  # Password from docker-compose
        influxdb_host="localhost",
        influxdb_port=8086,
        influxdb_bucket="trading_data_test",
        influxdb_org="trading_bot_dev",
        influxdb_token="trading_bot_token",
    )

    # Create Config instance
    config = Config()
    config.environment = "development"
    config.debug = True
    config.database = db_config

    return config


@pytest.fixture(scope="session")
def test_db_config():
    """Provide test-specific database configuration."""
    return {
        "postgresql": {
            "host": "localhost",
            "port": 5432,
            "database": "trading_bot_test",
            "username": "trading_bot_test",
            "password": "test_password",
        },
        "redis": {"host": "localhost", "port": 6379, "db": 1},
        "influxdb": {
            "host": "localhost",
            "port": 8086,
            "bucket": "trading_data_test",
            "org": "test-org",
            "token": "test-token",
        },
    }


def check_database_availability(config: Config) -> dict[str, bool]:
    """
    Check if required databases are available.
    Returns a dict with database status.
    """
    import psycopg2
    import redis
    from influxdb_client import InfluxDBClient

    status = {"postgresql": False, "redis": False, "influxdb": False}

    # Check PostgreSQL - try to connect as trading_bot user first
    try:
        conn = psycopg2.connect(
            host=config.database.postgresql_host,
            port=config.database.postgresql_port,
            database="trading_bot",
            user="trading_bot",  # Use trading_bot user for availability check
            password="trading_bot_password",  # Default password from docker-compose
        )
        conn.close()
        status["postgresql"] = True
    except Exception:
        pass

    # Check Redis
    try:
        r = redis.Redis(
            host=config.database.redis_host,
            port=config.database.redis_port,
            db=config.database.redis_db,
            password=config.database.redis_password,
            socket_connect_timeout=5,
        )
        r.ping()
        r.close()
        status["redis"] = True
    except Exception:
        pass

    # Check InfluxDB
    try:
        client = InfluxDBClient(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org,
        )
        client.ping()
        client.close()
        status["influxdb"] = True
    except Exception:
        pass

    return status


def setup_test_databases(config: Config) -> None:
    """
    Set up test databases if they don't exist.
    This creates the test database and user for PostgreSQL, clears Redis, and creates InfluxDB bucket.
    """
    import psycopg2
    import redis
    from influxdb_client import InfluxDBClient

    # Setup PostgreSQL
    try:
        conn = psycopg2.connect(
            host=config.database.postgresql_host,
            port=config.database.postgresql_port,
            database="trading_bot",
            user="trading_bot",  # Use trading_bot user
            password="trading_bot_password",  # Default password from docker-compose
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create test user if it doesn't exist
        cursor.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s", (config.database.postgresql_username,)
        )
        if not cursor.fetchone():
            cursor.execute(
                f"CREATE USER {config.database.postgresql_username} WITH PASSWORD '{config.database.postgresql_password}' CREATEDB"
            )
        else:
            # Update password if user exists
            cursor.execute(
                f"ALTER USER {config.database.postgresql_username} WITH PASSWORD '{config.database.postgresql_password}'"
            )

        # Create test database if it doesn't exist
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (config.database.postgresql_database,)
        )
        if not cursor.fetchone():
            cursor.execute(
                f"CREATE DATABASE {config.database.postgresql_database} OWNER {config.database.postgresql_username}"
            )

        cursor.close()
        conn.close()

        # Run migrations for test database
        env = os.environ.copy()
        env["DATABASE_URL"] = config.get_database_url()
        env["ALEMBIC_CONFIG"] = "alembic.ini"
        env["TESTING"] = "true"

        # Run alembic upgrade using Python module
        result = subprocess.run(
            ["python3", "-m", "alembic", "upgrade", "head"],
            env=env,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise DataSourceError(f"Migration failed: {result.stderr}")

    except Exception as e:
        raise DataSourceError(f"Failed to setup PostgreSQL test database: {e!s}")

    # Setup Redis
    try:
        r = redis.Redis(
            host=config.database.redis_host,
            port=config.database.redis_port,
            db=config.database.redis_db,
            password=config.database.redis_password,
        )
        r.ping()
        # Clear the test database
        r.flushdb()
        r.close()

    except Exception as e:
        raise DataSourceError(f"Failed to setup Redis test database: {e!s}")

    # Setup InfluxDB
    try:
        client = InfluxDBClient(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org,
        )

        # Check if bucket exists
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets

        bucket_exists = any(bucket.name == config.database.influxdb_bucket for bucket in buckets)

        if not bucket_exists:
            # Create test bucket
            buckets_api.create_bucket(
                bucket_name=config.database.influxdb_bucket, org=config.database.influxdb_org
            )

        client.close()

    except Exception as e:
        raise DataSourceError(f"Failed to setup InfluxDB test bucket: {e!s}")


def cleanup_test_databases(config: Config) -> None:
    """
    Clean up test databases after tests.
    This drops the test database and user for PostgreSQL, clears Redis, and removes InfluxDB bucket.
    """
    import psycopg2
    import redis
    from influxdb_client import InfluxDBClient

    # Clean up PostgreSQL
    try:
        conn = psycopg2.connect(
            host=config.database.postgresql_host,
            port=config.database.postgresql_port,
            database="trading_bot",
            user="trading_bot",
            password="trading_bot_password",
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Drop test database
        cursor.execute(f"DROP DATABASE IF EXISTS {config.database.postgresql_database}")

        # Drop test user
        cursor.execute(f"DROP USER IF EXISTS {config.database.postgresql_username}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Warning: Failed to cleanup PostgreSQL: {e!s}")

    # Clean up Redis
    try:
        r = redis.Redis(
            host=config.database.redis_host,
            port=config.database.redis_port,
            db=config.database.redis_db,
            password=config.database.redis_password,
        )
        r.flushdb()
        r.close()

    except Exception as e:
        print(f"Warning: Failed to cleanup Redis: {e!s}")

    # Clean up InfluxDB
    try:
        client = InfluxDBClient(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org,
        )
        buckets_api = client.buckets_api()
        bucket = buckets_api.find_bucket_by_name(config.database.influxdb_bucket)
        if bucket:
            buckets_api.delete_bucket(bucket)
        client.close()

    except Exception as e:
        print(f"Warning: Failed to cleanup InfluxDB: {e!s}")


@pytest.fixture(scope="session")
def database_setup(config, test_db_config):
    """
    Set up test databases once for the entire test session.
    This fixture will fail the test session if databases are not available.
    """
    # Check database availability first
    status = check_database_availability(config)

    if not any(status.values()):
        raise DataSourceError(
            "No databases are available for testing. "
            "Please ensure PostgreSQL, Redis, and/or InfluxDB are running. "
            f"Status: {status}"
        )

    # Set up test databases
    try:
        setup_test_databases(config)
    except Exception as e:
        raise DataSourceError(f"Failed to setup test databases: {e!s}")

    # Return the config - the test will handle async initialization
    return config


@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_after_all_tests():
    """
    Clean up all database connections and error handlers after all tests complete.
    This ensures no lingering connections or async tasks cause warnings.
    """
    yield

    # Clean up any remaining error handlers
    try:
        from src.error_handling.decorators import shutdown_all_error_handlers

        await shutdown_all_error_handlers()
    except Exception:
        pass  # Ignore cleanup errors

    # Clean up any remaining connections
    try:
        await close_database()
    except Exception:
        pass  # Ignore cleanup errors

    # Force garbage collection to clean up any remaining connections
    import gc

    gc.collect()


@pytest.fixture(scope="function")
async def clean_database(database_setup):
    """
    Clean database tables between tests to ensure isolation.
    This runs for each test function but doesn't recreate the database.
    """
    from sqlalchemy import text

    from src.database.connection import get_async_session

    # Initialize database for this test
    await initialize_database(database_setup)

    try:
        # Get a session and clean all tables
        async with get_async_session() as session:
            try:
                # Disable foreign key checks temporarily
                await session.execute(text("SET session_replication_role = replica;"))

                # Get all table names
                result = await session.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                )
                tables = [row[0] for row in result.fetchall()]

                # Truncate all tables
                for table in tables:
                    await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

                # Re-enable foreign key checks
                await session.execute(text("SET session_replication_role = DEFAULT;"))
                await session.commit()

            except Exception as e:
                await session.rollback()
                raise DataSourceError(f"Failed to clean database: {e!s}")

        # Yield the config - database stays initialized during test
        yield database_setup

    finally:
        # Clean up after test completes
        await close_database()


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on their file location
        if "tests/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "tests/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "tests/performance/" in str(item.fspath):
            item.add_marker(pytest.mark.performance)

        # Mark tests that take longer than 1 second as slow
        if "performance" in str(item.fspath) or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
