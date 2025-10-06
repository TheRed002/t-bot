"""Configuration for database module integration tests."""

import os

import pytest
import pytest_asyncio

# Import the real clean_database fixture from infrastructure tests
# This ensures database integration tests use the properly configured fixture
from tests.integration.infrastructure.conftest import clean_database  # noqa: F401


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
