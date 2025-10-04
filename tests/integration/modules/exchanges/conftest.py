"""
Exchange integration test fixtures.

Provides fixtures for testing real exchange services with mock exchange implementations.
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from src.core.config import Config


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
async def container(config):
    """Create DI injector with all real services registered."""
    from src.core.dependency_injection import DependencyInjector
    from src.core.di_master_registration import register_all_services

    # Register all services using the master registration
    # This returns a DependencyInjector, not a DependencyContainer
    injector = register_all_services(injector=None, config=config)

    yield injector

    # Cleanup - stop any services that were started
    # Try to get and stop exchange service if it exists
    try:
        exchange_service = injector.resolve("exchange_service")
        if exchange_service and hasattr(exchange_service, 'stop'):
            await exchange_service.stop()
    except:
        pass


@pytest.fixture
async def async_session(container):
    """Create mock async database session for testing."""
    # Create a mock session for tests that need it
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.add = Mock()
    mock_session.flush = AsyncMock()
    mock_session.refresh = AsyncMock()

    yield mock_session

    # Cleanup
    await mock_session.rollback()
    await mock_session.close()


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
