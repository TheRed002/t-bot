"""
Exchange integration test fixtures.

Provides fixtures for testing real exchange services with mock exchange implementations.
"""

import os
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from src.core.config import Config

# Load environment variables
load_dotenv()


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def binance_config() -> dict[str, Any]:
    """Create proper Binance config dictionary using real credentials from .env file."""
    # Support both naming conventions, prefer *_SECRET_KEY as used in .env
    api_secret = os.getenv("BINANCE_SECRET_KEY") or os.getenv("BINANCE_API_SECRET", "")
    return {
        "api_key": os.getenv("BINANCE_API_KEY", ""),
        "api_secret": api_secret,
        "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
    }


@pytest.fixture
def okx_config() -> dict[str, Any]:
    """Create proper OKX config dictionary using real credentials from .env file."""
    # Support both naming conventions, prefer *_SECRET_KEY as used in .env
    api_secret = os.getenv("OKX_SECRET_KEY") or os.getenv("OKX_API_SECRET", "")
    return {
        "api_key": os.getenv("OKX_API_KEY", ""),
        "api_secret": api_secret,
        "passphrase": os.getenv("OKX_PASSPHRASE", ""),
        "sandbox": os.getenv("OKX_SANDBOX", "true").lower() == "true",
    }


@pytest.fixture
def coinbase_config() -> dict[str, Any]:
    """Create proper Coinbase config dictionary using real credentials from .env file."""
    # Support both naming conventions, prefer *_SECRET_KEY as used in .env
    api_secret = os.getenv("COINBASE_SECRET_KEY") or os.getenv("COINBASE_API_SECRET", "")
    return {
        "api_key": os.getenv("COINBASE_API_KEY", ""),
        "api_secret": api_secret,
        "sandbox": os.getenv("COINBASE_SANDBOX", "true").lower() == "true",
    }


@pytest_asyncio.fixture
async def container(config):
    """Create DI injector with all real services registered and initialized."""
    from src.core.di_master_registration import register_all_services
    from src.database.models import Base

    # Register all services using the master registration
    # This returns a DependencyInjector, not a DependencyContainer
    injector = register_all_services(injector=None, config=config)

    # Initialize the DI-created connection manager
    connection_manager = injector.resolve("DatabaseConnectionManager")
    await connection_manager.initialize()

    # Create all database tables
    async with connection_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield injector

    # Cleanup - stop any services that were started
    # Try to get and stop exchange service if it exists
    try:
        exchange_service = injector.resolve("exchange_service")
        if exchange_service and hasattr(exchange_service, "stop"):
            await exchange_service.stop()
    except:
        pass

    # Drop all tables for clean test isolation
    try:
        async with connection_manager.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    except Exception as e:
        # Ignore errors from dropping ENUMs that are still referenced by other test schemas
        # PostgreSQL creates ENUMs globally (not per schema) with create_type=False
        # Multiple test schemas may reference the same ENUM types
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Ignoring database cleanup error (likely ENUM dependency): {e}")

    # Close connection manager
    await connection_manager.close()


@pytest_asyncio.fixture
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
    from tests.integration.conftest import cleanup_di_container, register_all_services_for_testing

    container = register_all_services_for_testing()
    yield container

    # Cleanup to prevent resource leaks
    await cleanup_di_container(container)
