"""
Analytics Module Integration Test Configuration.

Provides pytest fixtures for analytics integration tests with full DI support.
"""

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def di_container():
    """
    Provide fully configured DI container with all services registered.

    Uses master DI registration to ensure all dependencies are properly configured
    in the correct order without circular dependency issues.
    """
    from tests.integration.conftest import register_all_services_for_testing, cleanup_di_container

    container = register_all_services_for_testing()
    yield container

    # Cleanup to prevent resource leaks
    await cleanup_di_container(container)
