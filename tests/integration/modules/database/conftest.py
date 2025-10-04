"""Configuration for database module integration tests."""

import os
import pytest


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