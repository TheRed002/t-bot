"""
State Module Integration Test Configuration.

Provides pytest fixtures for state integration tests with full DI support.
"""

import logging

import pytest
import pytest_asyncio

from src.core.config import get_config
from src.core.dependency_injection import DependencyContainer
from src.state import register_state_services
from src.state.di_registration import create_state_service_with_dependencies

logger = logging.getLogger(__name__)

# Note: Infrastructure fixtures are available via tests/integration/conftest.py


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


@pytest_asyncio.fixture
async def test_state_service(real_database_service):
    """
    Create a real StateService for testing with actual database connections.

    This fixture provides a fully initialized state service ready for testing.
    """
    config = get_config()

    logger.info("Creating test state service")
    state_service = await create_state_service_with_dependencies(
        config=config,
        database_service=real_database_service
    )

    # Initialize the service
    await state_service.initialize()
    logger.info("Test state service initialized")

    yield state_service

    # Cleanup
    try:
        await state_service.cleanup()
        logger.info("Test state service cleaned up")
    except Exception as e:
        logger.warning(f"Error cleaning up state service: {e}")
