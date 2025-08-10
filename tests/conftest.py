"""
Pytest configuration for the trading bot test suite.

This module provides fixtures and configuration for all test types:
- Unit tests: No external dependencies
- Integration tests: Require temporary databases
- Performance tests: Measure system performance
"""

import pytest
import pytest_asyncio
import asyncio
import os
import tempfile
import subprocess
import time
from typing import Dict, Any, Optional
from pathlib import Path

from src.core.config import Config
from src.core.exceptions import DataSourceError
from src.database.connection import initialize_database, close_database, health_check


@pytest.fixture(scope="session")
def config():
    """Provide application configuration for tests."""
    from src.core.config import DatabaseConfig

    return Config(
        environment="development",
        debug=True,
        database=DatabaseConfig(
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
            influxdb_token="trading_bot_token"
        )
    )


@pytest.fixture(scope="session")
def test_db_config():
    """Provide test-specific database configuration."""
    return {
        "postgresql": {
            "host": "localhost",
            "port": 5432,
            "database": "trading_bot_test",
            "username": "trading_bot_test",
            "password": "test_password"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 1
        },
        "influxdb": {
            "host": "localhost",
            "port": 8086,
            "bucket": "trading_data_test",
            "org": "test-org",
            "token": "test-token"
        }
    }


def check_database_availability(config: Config) -> Dict[str, bool]:
    """
    Check if required databases are available.
    Returns a dict with database status.
    """
    import psycopg2
    import redis
    from influxdb_client import InfluxDBClient

    status = {
        "postgresql": False,
        "redis": False,
        "influxdb": False
    }

    # Check PostgreSQL - try to connect as trading_bot user first
    try:
        conn = psycopg2.connect(
            host=config.database.postgresql_host,
            port=config.database.postgresql_port,
            database="trading_bot",
            user="trading_bot",  # Use trading_bot user for availability check
            password="trading_bot_password"  # Default password from docker-compose
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
            socket_connect_timeout=5
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
            org=config.database.influxdb_org
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
    import subprocess
    import os
    from influxdb_client import InfluxDBClient

    # Setup PostgreSQL
    try:
        conn = psycopg2.connect(
            host=config.database.postgresql_host,
            port=config.database.postgresql_port,
            database="trading_bot",
            user="trading_bot",  # Use trading_bot user
            password="trading_bot_password"  # Default password from docker-compose
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Create test user if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_roles WHERE rolname = %s",
                       (config.database.postgresql_username,))
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
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                       (config.database.postgresql_database,))
        if not cursor.fetchone():
            cursor.execute(
                f"CREATE DATABASE {config.database.postgresql_database} OWNER {config.database.postgresql_username}"
            )

        cursor.close()
        conn.close()

        # Run migrations for test database
        env = os.environ.copy()
        env['DATABASE_URL'] = config.get_database_url()
        env['ALEMBIC_CONFIG'] = 'alembic.ini'
        env['TESTING'] = 'true'

        # Run alembic upgrade using Python module
        result = subprocess.run(
            ['python3', '-m', 'alembic', 'upgrade', 'head'],
            env=env,
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise DataSourceError(f"Migration failed: {result.stderr}")

    except Exception as e:
        raise DataSourceError(
            f"Failed to setup PostgreSQL test database: {str(e)}"
        )

    # Setup Redis
    try:
        r = redis.Redis(
            host=config.database.redis_host,
            port=config.database.redis_port,
            db=config.database.redis_db,
            password=config.database.redis_password
        )
        r.ping()
        # Clear the test database
        r.flushdb()
        r.close()

    except Exception as e:
        raise DataSourceError(f"Failed to setup Redis test database: {str(e)}")

    # Setup InfluxDB
    try:
        client = InfluxDBClient(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org
        )

        # Check if bucket exists
        buckets_api = client.buckets_api()
        buckets = buckets_api.find_buckets().buckets

        bucket_exists = any(
            bucket.name == config.database.influxdb_bucket for bucket in buckets)

        if not bucket_exists:
            # Create test bucket
            buckets_api.create_bucket(
                bucket_name=config.database.influxdb_bucket,
                org=config.database.influxdb_org
            )

        client.close()

    except Exception as e:
        raise DataSourceError(
            f"Failed to setup InfluxDB test bucket: {str(e)}"
        )


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
            password="trading_bot_password"
        )
        conn.autocommit = True
        cursor = conn.cursor()

        # Drop test database
        cursor.execute(
            f"DROP DATABASE IF EXISTS {config.database.postgresql_database}"
        )

        # Drop test user
        cursor.execute(
            f"DROP USER IF EXISTS {config.database.postgresql_username}"
        )

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Warning: Failed to cleanup PostgreSQL: {str(e)}")

    # Clean up Redis
    try:
        r = redis.Redis(
            host=config.database.redis_host,
            port=config.database.redis_port,
            db=config.database.redis_db,
            password=config.database.redis_password
        )
        r.flushdb()
        r.close()

    except Exception as e:
        print(f"Warning: Failed to cleanup Redis: {str(e)}")

    # Clean up InfluxDB
    try:
        client = InfluxDBClient(
            url=f"http://{config.database.influxdb_host}:{config.database.influxdb_port}",
            token=config.database.influxdb_token,
            org=config.database.influxdb_org
        )
        buckets_api = client.buckets_api()
        bucket = buckets_api.find_bucket_by_name(
            config.database.influxdb_bucket)
        if bucket:
            buckets_api.delete_bucket(bucket)
        client.close()

    except Exception as e:
        print(f"Warning: Failed to cleanup InfluxDB: {str(e)}")


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
        raise DataSourceError(f"Failed to setup test databases: {str(e)}")

    # Return the config - the test will handle async initialization
    return config


@pytest_asyncio.fixture(scope="function", autouse=True)
async def cleanup_after_all_tests():
    """
    Clean up all database connections after all tests complete.
    This ensures no lingering connections cause warnings.
    """
    yield

    # Clean up any remaining connections
    from src.database.connection import close_database
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
    from src.database.connection import get_async_session, initialize_database, close_database
    from sqlalchemy import text

    # Initialize database for this test
    await initialize_database(database_setup)

    try:
        # Get a session and clean all tables
        async with get_async_session() as session:
            try:
                # Disable foreign key checks temporarily
                await session.execute(text("SET session_replication_role = replica;"))

                # Get all table names
                result = await session.execute(text(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                ))
                tables = [row[0] for row in result.fetchall()]

                # Truncate all tables
                for table in tables:
                    await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

                # Re-enable foreign key checks
                await session.execute(text("SET session_replication_role = DEFAULT;"))
                await session.commit()

            except Exception as e:
                await session.rollback()
                raise DataSourceError(f"Failed to clean database: {str(e)}")

        # Yield the config - database stays initialized during test
        yield database_setup

    finally:
        # Clean up after test completes
        await close_database()


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


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
        if "performance" in str(
                item.fspath) or "integration" in str(
                item.fspath):
            item.add_marker(pytest.mark.slow)
