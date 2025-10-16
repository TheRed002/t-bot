"""Pytest configuration for risk management integration tests."""

import pytest
import pytest_asyncio
from sqlalchemy import text

# Import and expose all fixtures from the fixtures module
from .fixtures.real_service_fixtures import (
    generate_bear_market_scenario,
    generate_bull_market_scenario,
    generate_high_volatility_scenario,
    generate_realistic_market_data_sequence,
    minimal_state_service,
    real_risk_factory,
    real_risk_service,
    sample_positions,
    sample_signal,
)

__all__ = [
    "generate_bear_market_scenario",
    "generate_bull_market_scenario",
    "generate_high_volatility_scenario",
    "generate_realistic_market_data_sequence",
    "minimal_state_service",
    "real_risk_factory",
    "real_risk_service",
    "sample_positions",
    "sample_signal",
]


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


@pytest_asyncio.fixture(scope="module")
async def container():
    """Create DI injector with all real services registered and initialized - module scoped for performance."""
    from src.core.config import Config
    from src.core.di_master_registration import register_all_services
    from src.database.models import Base
    from tests.integration.conftest import cleanup_di_container

    config = Config()
    config.environment = "test"

    # Register all services using the master registration
    # This returns a DependencyInjector, not a DependencyContainer
    injector = register_all_services(injector=None, config=config)

    # Initialize the DI-created connection manager
    connection_manager = injector.resolve("DatabaseConnectionManager")
    await connection_manager.initialize()

    # Drop all tables first for clean state
    try:
        async with connection_manager.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            # NOTE: SQLAlchemy will try to drop enum types even with create_type=False
            # This may fail if other test schemas still reference the enum types
    except Exception as e:
        # Ignore enum drop errors - they're shared across test schemas
        if "cannot drop type" not in str(e):
            raise

    # Clear metadata cache
    Base.metadata.clear()

    # Create all database tables with fresh metadata
    async with connection_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield injector

    # Cleanup - use proper cleanup function from integration conftest
    # This stops ALL services properly and prevents resource leaks
    try:
        # Drop all tables for clean test isolation
        async with connection_manager.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            # NOTE: SQLAlchemy will try to drop enum types even with create_type=False
            # This may fail if other test schemas still reference the enum types
    except Exception as e:
        # Ignore enum drop errors - they're shared across test schemas
        if "cannot drop type" not in str(e):
            raise

    # Clear metadata
    Base.metadata.clear()

    # Close connection manager
    try:
        await connection_manager.close()
    except:
        pass

    # Cleanup all services and clear container
    await cleanup_di_container(injector)


@pytest.fixture
def config():
    """Create test configuration."""
    from src.core.config import Config

    return Config()
